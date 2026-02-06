from model import SimpleRNN, VRNN, ModuleNetwork
from objectives import Objective
import os
from pathlib import Path
import torch
import sys
import gc
import pandas as pd
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for plotting
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('error')
from multiprocessing import cpu_count

# Try to import pathos for better multiprocessing (handles lambdas/closures)
# Falls back to standard multiprocessing if not available
try:
    from pathos.multiprocessing import ProcessingPool
    HAS_PATHOS = True
except ImportError:
    HAS_PATHOS = False


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from PreProParadigm.audit_gm import NonHierachicalAuditGM, HierarchicalAuditGM


# Check which folder exists and append the correct path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
if os.path.exists(os.path.join(base_path, 'Kalman')):
    from Kalman.kalman import kalman_fit_predict_batch, kalman_fit_predict_multicontext_batch, MIN_OBS_FOR_EM, kalman_fit_predict, kalman_fit_predict_multicontext
elif os.path.exists(os.path.join(base_path, 'KalmanFilterViz1D')):
    from KalmanFilterViz1D.kalman import kalman_fit_predict_batch, kalman_fit_predict_multicontext_batch, MIN_OBS_FOR_EM, kalman_fit_predict, kalman_fit_predict_multicontext
else:
    raise ImportError("Neither 'Kalman' nor 'KalmanFilterViz1D' folder found.")


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FREQ_MIN = 1400
FREQ_MAX = 1650


def contexts_to_responsibilities(contexts, n_ctx):
    """
    Convert hard context labels to one-hot responsibilities (soft assignments).
    
    This function bridges the gap between generative models that produce hard
    context labels and Kalman filter functions that expect soft responsibilities.
    
    Parameters
    ----------
    contexts : np.array
        1D integer array of shape (T,) with values in [0, n_ctx-1]
    n_ctx : int
        Number of contexts
    
    Returns
    -------
    responsibilities : np.array
        One-hot encoded array of shape (T, n_ctx) where each row sums to 1.0
    """
    T = len(contexts)
    responsibilities = np.zeros((T, n_ctx))
    responsibilities[np.arange(T), contexts.astype(int)] = 1.0
    return responsibilities


# TODO CHECK-LIST
# - a loss tolerance?
# - an early stopping in case of overfitting (ABSOLUTELY according to Seyma) --> include IF i observe overfitting
# - dropout?
# - assess need to change activation function or not (hand in hand with changing the models' inner layers' architecture)
# - set requires_grad / autograd


# TODO:
# - load params


class MinMaxScaler():
    def __init__(self):
        self.freq_min = FREQ_MIN
        self.freq_max = FREQ_MAX

    def fit_normalize(self, x, freq_min=None, freq_max=None, margin=None):
        if freq_min is None:
            self.freq_min = x.min()
        if freq_max is None:
            self.freq_max = x.max()
        if margin is not None:
            self.freq_min -= margin
            self.freq_max += margin
        return self.normalize(x)

    def normalize(self, x):
        return (x - self.freq_min) / (self.freq_max - self.freq_min)
    
    def denormalize_mean(self, mu_norm):
        return mu_norm * (self.freq_max - self.freq_min) + self.freq_min
    
    def denormalize_var(self, var_norm):
        return var_norm * (self.freq_max - self.freq_min)**2


# =============================================================================
# Data Mode and Learning Objective Helpers
# =============================================================================

# Supported data modes (determines data generation and context handling):
#   'single_ctx': Single context scenario (N_ctx=1), works with SimpleRNN/VRNN
#   'multi_ctx': Multi-context scenario (N_ctx>1), contexts are generated and available
VALID_DATA_MODES = ['single_ctx', 'multi_ctx']

# Supported learning objectives for ModuleNetwork (only applies when data_mode='multi_ctx'):
#   'obs': Train observation module only (hidden process prediction)
#   'ctx': Train context module only (context inference)  
#   'obs_ctx': Train both modules with combined loss (weighted by kappa)
VALID_LEARNING_OBJECTIVES = ['obs', 'ctx', 'obs_ctx']

# Kappa values to explore for 'obs_ctx' learning objective
# kappa=1.0 means full weight on observation loss, kappa=0.0 on context loss
DEFAULT_KAPPA_VALUES = [0.3, 0.5, 0.7]


def prepare_batch_data(gm, gm_name, data_mode, device, return_pars=False):
    """
    Generate a batch of data and prepare tensors based on data mode.
    
    Parameters
    ----------
    gm : NonHierachicalAuditGM or HierarchicalAuditGM
        The generative model instance
    gm_name : str
        Name of the generative model ('NonHierarchicalGM' or 'HierarchicalGM')
    data_mode : str
        Data mode: 'single_ctx' (N_ctx=1) or 'multi_ctx' (N_ctx>1)
    device : str or torch.device
        Device to place tensors on
    return_pars : bool
        Whether to return the parameters used for generation
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'y': Observation tensor of shape (batch, seq_len, 1)
        - 'contexts': Context tensor of shape (batch, seq_len) if data_mode='multi_ctx', else None
        - 'pars': Parameters dict if return_pars=True, else None
    """
    if gm_name == 'NonHierarchicalGM':
        if return_pars:
            contexts, _, y, pars = gm.generate_batch(return_pars=True)
        else:
            contexts, _, y = gm.generate_batch(return_pars=False)
            pars = None
    elif gm_name == 'HierarchicalGM':
        if return_pars:
            _, _, _, _, _, contexts, _, y, pars = gm.generate_batch(return_pars=True)
        else:
            _, _, _, _, _, contexts, _, y = gm.generate_batch(return_pars=False)
            pars = None
    else:
        raise ValueError(f"Invalid GM name: {gm_name}")
    
    # Convert observations to tensor
    y_tensor = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(device)
    
    # Prepare context tensor if in multi-context mode
    if data_mode == 'multi_ctx':
        # contexts is a numpy array of shape (batch, seq_len) with integer context labels
        contexts_tensor = torch.tensor(contexts, dtype=torch.long, requires_grad=False).to(device)
    else:
        contexts_tensor = None
    
    return {
        'y': y_tensor,
        'contexts': contexts_tensor,
        'pars': pars
    }


def prepare_benchmark_data(benchmarks, data_mode, device):
    """
    Prepare benchmark data for validation based on data mode.
    
    Parameters
    ----------
    benchmarks : dict
        Benchmark dictionary containing 'y', 'contexts', 'mu_kal_pred', etc.
    data_mode : str
        Data mode: 'single_ctx' (N_ctx=1) or 'multi_ctx' (N_ctx>1)
    device : str or torch.device
        Device to place tensors on
    
    Returns
    -------
    dict
        Dictionary containing prepared tensors and benchmark data
    """
    y = benchmarks['y']
    y_tensor = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(-1).to(device)
    
    result = {
        'y': y_tensor,
        'mu_kal_pred': benchmarks['mu_kal_pred'],
        'sigma_kal_pred': benchmarks['sigma_kal_pred'],
        'mse_kal': benchmarks['perf'],
        'pars': benchmarks['pars'],
        'min_obs_for_em': benchmarks.get('min_obs_for_em', MIN_OBS_FOR_EM),
    }
    
    if data_mode == 'multi_ctx':
        contexts = benchmarks.get('contexts')
        if contexts is not None:
            result['contexts'] = torch.tensor(contexts, dtype=torch.long, requires_grad=False).to(device)
        else:
            result['contexts'] = None
    else:
        result['contexts'] = None
    
    return result


def compute_model_loss(model, objective, y_tensor, model_output, data_mode, learning_objective='obs_ctx', contexts_tensor=None, kappa=0.5):
    """
    Compute loss based on model type, data mode, and learning objective.
    
    Parameters
    ----------
    model : nn.Module
        The model (SimpleRNN, VRNN, or ModuleNetwork)
    objective : Objective
        The objective function wrapper
    y_tensor : torch.Tensor
        Target observations of shape (batch, seq_len, 1)
    model_output : tuple or torch.Tensor
        Output from model forward pass
    data_mode : str
        Data mode: 'single_ctx' (N_ctx=1) or 'multi_ctx' (N_ctx>1)
    learning_objective : str
        Learning objective for ModuleNetwork: 'obs', 'ctx', or 'obs_ctx' (default)
        Only used when data_mode='multi_ctx' and model is ModuleNetwork
    contexts_tensor : torch.Tensor, optional
        Target contexts for ModuleNetwork training
    kappa : float
        Weighting factor between observation and context losses (default: 0.5)
        Only used when learning_objective='obs_ctx'
    
    Returns
    -------
    torch.Tensor
        The computed loss
    """
    if data_mode == 'multi_ctx' and model.name == 'module_network':
        # ModuleNetwork returns (obs_output, context_output)
        # Pass learning_objective to control which loss components are used
        return objective.loss(model, y_tensor[:, 1:, :], model_output, 
                             target_ctx=contexts_tensor[:, 1:],
                             kappa=kappa, learning_objective=learning_objective)
    else:
        # Standard RNN/VRNN loss
        return objective.loss(model, y_tensor[:, 1:, :], model_output)


def extract_model_predictions(model, model_output):
    """
    Extract mu and variance estimates from model output based on model type.
    
    Parameters
    ----------
    model : nn.Module
        The model (SimpleRNN, VRNN, or ModuleNetwork)
    model_output : tuple or torch.Tensor
        Output from model forward pass
    
    Returns
    -------
    tuple
        (mu_estim, var_estim, context_output) where context_output is None for non-ModuleNetwork models
    """
    if model.name == 'module_network':
        obs_output, context_output = model_output
        model_output_dist = obs_output
    elif model.name == 'vrnn':
        model_output_dist = model_output[0]
        context_output = None
    else:
        model_output_dist = model_output
        context_output = None
    
    mu_estim = model_output_dist[..., 0].detach().cpu().numpy()
    var_estim = (F.softplus(model_output_dist[..., 1]) + 1e-6).detach().cpu().numpy()
    
    return mu_estim, var_estim, context_output


def bin_params(data_config):
    tau_bins = np.logspace(np.log10(data_config["mu_tau_bounds"]["low"]), np.log10(data_config["mu_tau_bounds"]["high"]), 10)
    si_stat_bins = np.linspace(data_config["si_stat_bounds"]["low"], data_config["si_stat_bounds"]["high"], 10)
    si_r_bins = np.linspace(data_config["si_r_bounds"]["low"], data_config["si_r_bounds"]["high"], 10)

    # Create a grid of all parameter combinations
    param_bins = {'tau': tau_bins,
                    'si_stat': si_stat_bins,
                    'si_r': si_r_bins}

    return param_bins

def map_binned_params_2_metrics(param_bins, y, mu_estim, pars, mu_kal=None):
    """
    Map parameters to bins and compute metrics for each parameter combination.
    
    Args:
        param_bins: Dictionary with 'tau', 'si_stat', 'si_r' bin edges
        y: Observations, shape (N_samples, seq_len)
        mu_estim: Model estimates, shape (N_samples, seq_len)
        pars: Dictionary with parameter arrays. For multi-context scenarios (N_ctx > 1),
              'tau' and 'lim' may be 2D arrays with shape (N_samples, N_ctx). In this case,
              we use the first context's values for binning (typically std).
        mu_kal: Optional Kalman filter estimates, shape (N_samples, seq_len). If provided,
                computes KF MSE and model-vs-KF MSE per bin.
    
    Returns:
        DataFrame with binned metrics including:
        - mse: Model MSE wrt target
        - mse_kal: KF MSE wrt target (if mu_kal provided)
        - mse_model2kal: Model MSE wrt KF predictions (if mu_kal provided)
        - count: Number of samples in this bin
    """
    # Initialize storage array: each row stores tau, lim, si_stat, si_q, and the corresponding mse
    param_combinations = np.array(np.meshgrid(param_bins['tau'], param_bins['si_stat'], param_bins['si_r'])).T.reshape(-1, 3)
    
    # Initialize metrics dict with optional KF fields
    if mu_kal is not None:
        binned_metrics = {tuple(param_combination): {'mse': [], 'mse_kal': [], 'mse_model2kal': [], 'count': 0} for param_combination in param_combinations}
    else:
        binned_metrics = {tuple(param_combination): {'mse': [], 'count': 0} for param_combination in param_combinations}

    # Extract parameters from dictionary, handling multi-context case
    # For tau: if 2D (multi-context), take first context (std); otherwise use as-is
    tau_vals = pars['tau']
    if tau_vals.ndim > 1:
        tau_vals = tau_vals[:, 0]  # Use first context (std) for binning
    
    si_stat_vals = pars['si_stat']
    si_r_vals = pars['si_r']
    
    # Digitize each of these parameters to find the corresponding bin
    tau_bin_id = np.digitize(tau_vals, param_bins['tau']) - 1
    si_stat_bin_id = np.digitize(si_stat_vals, param_bins['si_stat']) - 1
    si_r_bin_id = np.digitize(si_r_vals, param_bins['si_r']) - 1

    # Use the bins found to get the corresponding combination of parameters
    param_combination = (param_bins['tau'][tau_bin_id], param_bins['si_stat'][si_stat_bin_id], param_bins['si_r'][si_r_bin_id])
    
    # Get MSE per sample in batch (model vs target)
    mse_per_sample = ((mu_estim-y)**2).mean(axis=1)
    
    # Compute KF metrics if KF predictions provided
    if mu_kal is not None:
        mse_kal_per_sample = ((mu_kal-y)**2).mean(axis=1)  # KF vs target
        mse_model2kal_per_sample = ((mu_estim-mu_kal)**2).mean(axis=1)  # Model vs KF

    # Then, zip batch's param_combination array and MSE arrays and store
    if mu_kal is not None:
        for *pc, m, m_kal, m_m2k in zip(*param_combination, mse_per_sample, mse_kal_per_sample, mse_model2kal_per_sample):
            binned_metrics[tuple(pc)]['mse'].append(m)
            binned_metrics[tuple(pc)]['mse_kal'].append(m_kal)
            binned_metrics[tuple(pc)]['mse_model2kal'].append(m_m2k)
    else:
        for *pc, m in zip(*param_combination, mse_per_sample):
            binned_metrics[tuple(pc)]['mse'].append(m)


    # If tracking performance along data parameters, average stored MSEs for each parameter combination
    for param_combination in binned_metrics.keys():
        binned_metrics[param_combination]['count'] = len(binned_metrics[param_combination]['mse'])
        if binned_metrics[param_combination]['count'] > 0:
            binned_metrics[param_combination]['mse'] = np.mean(binned_metrics[param_combination]['mse'])
            if mu_kal is not None:
                binned_metrics[param_combination]['mse_kal'] = np.mean(binned_metrics[param_combination]['mse_kal'])
                binned_metrics[param_combination]['mse_model2kal'] = np.mean(binned_metrics[param_combination]['mse_model2kal'])
        else:
            binned_metrics[param_combination]['mse'] = np.nan
            if mu_kal is not None:
                binned_metrics[param_combination]['mse_kal'] = np.nan
                binned_metrics[param_combination]['mse_model2kal'] = np.nan


    # Save binned metrics
    if mu_kal is not None:
        binned_metrics_list = [
            {
            'tau': tau,
            'si_stat': si_stat,
            'si_r': si_r,
            'mse': metrics['mse'],
            'mse_kal': metrics['mse_kal'],
            'mse_model2kal': metrics['mse_model2kal'],
            'count': metrics['count']
            }
            for (tau, si_stat, si_r), metrics in binned_metrics.items()
        ]
    else:
        binned_metrics_list = [
            {
            'tau': tau,
            'si_stat': si_stat,
            'si_r': si_r,
            'mse': metrics['mse'],
            'count': metrics['count']
            }
            for (tau, si_stat, si_r), metrics in binned_metrics.items()
        ]
    binned_metrics_df = pd.DataFrame(binned_metrics_list)
    binned_metrics_df['si_q'] = binned_metrics_df['si_stat'] * ((2 * binned_metrics_df['tau'] - 1) ** 0.5) / binned_metrics_df['tau']

    return binned_metrics_df


def get_ctx_gm_subpath(N_ctx, gm_name):
    """
    Build the subpath for N_ctx and gm_name.
    Only includes gm_name folder when N_ctx > 1 (since N_ctx=1 can only use NonHierarchicalGM).
    
    Returns:
        Path: e.g., 'N_ctx_1' or 'N_ctx_2/HierarchicalGM'
    """
    if N_ctx == 1:
        return Path(f"N_ctx_{N_ctx}")
    else:
        return Path(f"N_ctx_{N_ctx}") / gm_name


def benchmark_filename(benchmarkpath, N_ctx, gm_name, N_samples, suffix=''):
    if suffix != '':
        suffix = '_' + suffix
    return benchmarkpath / get_ctx_gm_subpath(N_ctx, gm_name) / f"benchmarks_{N_samples}{suffix}.pkl"


def benchmark_individual_dir(benchmarkpath, N_ctx, gm_name, N_samples, suffix=''):
    """Get directory for individual benchmark files."""
    if suffix != '':
        suffix = '_' + suffix
    return benchmarkpath / get_ctx_gm_subpath(N_ctx, gm_name) / f"benchmarks_{N_samples}{suffix}_individual"


def aggregate_individual_benchmarks(benchmarkpath, N_ctx, gm_name, N_samples, suffix=''):
    """
    Aggregate individual benchmark files into a single benchmark file.
    
    This function reads all sample-wise .pkl files from the individual directory,
    stacks them into arrays, and saves the aggregated benchmark file.
    
    Returns:
        dict: Aggregated benchmark dictionary, or None if no individual files found
    """
    incr_dir = benchmark_individual_dir(benchmarkpath, N_ctx, gm_name, N_samples, suffix)
    final_file = benchmark_filename(benchmarkpath, N_ctx, gm_name, N_samples, suffix)
    
    if not incr_dir.exists():
        print(f"  No individual directory found at {incr_dir}")
        return None
    
    # Find all sample files
    sample_files = sorted(incr_dir.glob("sample_*.pkl"), key=lambda f: int(f.stem.split('_')[1]))
    
    if len(sample_files) == 0:
        print(f"  No sample files found in {incr_dir}")
        return None
    
    print(f"  Found {len(sample_files)} individual sample files, aggregating...")
    
    # Load all samples
    samples = []
    for sf in sample_files:
        with open(sf, 'rb') as f:
            samples.append(pickle.load(f))
    
    # Stack arrays from all samples
    benchmark_kit = {
        'y': np.stack([s['y'] for s in samples], axis=0),
        'contexts': np.stack([s['contexts'] for s in samples], axis=0) if samples[0]['contexts'] is not None else None,
        'mu_kal_pred': np.stack([s['mu_kal_pred'] for s in samples], axis=0),
        'sigma_kal_pred': np.stack([s['sigma_kal_pred'] for s in samples], axis=0),
        'pars': {key: np.stack([s['pars'][key] for s in samples], axis=0) for key in samples[0]['pars'].keys()},
        'perf': np.array([s['perf'] for s in samples]),
        'n_ctx': samples[0]['n_ctx'],
        'min_obs_for_em': samples[0]['min_obs_for_em'],
    }
    
    # Save aggregated file
    with open(final_file, 'wb') as f:
        pickle.dump(benchmark_kit, f)
    print(f"  Benchmark saved to {final_file}")
    
    return benchmark_kit


def _process_single_kf_sample(args):
    """
    Process a single sample for KF benchmarking. Helper for parallel processing.
    
    Args:
        args: Tuple of (i, y_i, ctx_i, pars_i, n_ctx, n_iter, incr_dir)
            - i: Sample index
            - y_i: Observation array of shape (T,)
            - ctx_i: Context array of shape (T,) or None
            - pars_i: Parameters dict for this sample
            - n_ctx: Number of contexts
            - n_iter: Number of EM iterations
            - incr_dir: Path to save individual sample files (or None to skip saving)
    
    Returns:
        dict: Sample data dictionary with 'y', 'contexts', 'mu_kal_pred', 'sigma_kal_pred', 
              'pars', 'perf', 'n_ctx', 'min_obs_for_em'
    """
    i, y_i, ctx_i, pars_i, n_ctx, n_iter, incr_dir = args
    
    # Fit Kalman filter for this single sample
    if n_ctx == 1:
        mu_pred_i, sigma_pred_i, _ = kalman_fit_predict(y_i, n_iter=n_iter)
    else:
        # Convert context labels to responsibilities for multicontext KF
        responsibilities_i = contexts_to_responsibilities(ctx_i, n_ctx)
        mu_pred_i, sigma_pred_i, _ = kalman_fit_predict_multicontext(
            y_i, responsibilities_i, n_iter=n_iter
        )
    
    # Compute MSE for this sample
    mse_i = ((mu_pred_i - y_i[MIN_OBS_FOR_EM:]) ** 2).mean()
    
    # Build sample data
    sample_data = {
        'y': y_i,
        'contexts': ctx_i,
        'mu_kal_pred': mu_pred_i,
        'sigma_kal_pred': sigma_pred_i,
        'pars': pars_i,
        'perf': float(mse_i),
        'n_ctx': n_ctx,
        'min_obs_for_em': MIN_OBS_FOR_EM,
    }
    
    # Save this sample if directory provided
    if incr_dir is not None:
        sample_file = incr_dir / f"sample_{i:06d}.pkl"
        with open(sample_file, 'wb') as f:
            pickle.dump(sample_data, f)
    
    return sample_data


def compute_benchmarks(data_config, N_ctx, gm_name, N_samples=None, n_iter=5, benchmarkpath=None, save=False, suffix='', individual=True, max_workers=None):
    """
    Benchmarks the Kalman filter on a batch of data, tracking MSE along parameter configurations.
    Uses standard Kalman filter for single-context (N_ctx=1) and context-aware Kalman filter
    for multi-context (N_ctx>1) scenarios.
    
    Supports individual saving: each sample is saved individually so that progress is not lost
    if the computation is interrupted. Use individual=True (default) for long-running jobs.
    
    Supports parallel processing: set max_workers > 1 to process multiple samples simultaneously.

    Args:
        data_config (dict): Configuration parameters for the data generative model.
        N_samples (int, optional): Number of samples to generate. If None, uses data_config["N_samples"].
        n_iter (int): Number of iterations for kalman_fit_batch.
        benchmarkpath (Path or None): Path to save the benchmark file. If None, benchmarks are not saved.
        save (bool): Whether to save the benchmark data.
        suffix (str): Suffix for the benchmark filename.
        individual (bool): If True, save each sample individually to avoid losing progress on crash.
                           The samples are aggregated at the end. Default: True.
        max_workers (int, optional): Number of parallel workers for KF fitting.
                                     None or 1 = sequential processing (default).
                                     >1 = parallel processing with that many workers.
                                     -1 = use all available CPU cores.

    Returns:
        dict: A dictionary containing observations, contexts (if N_ctx>1), Kalman estimates, 
              parameters, and performance metrics.
    """
    
    # Define data generative model
    if gm_name == 'NonHierarchicalGM': gm = NonHierachicalAuditGM(data_config)
    elif gm_name == 'HierarchicalGM': gm = HierarchicalAuditGM(data_config)
    else: raise ValueError("Invalid GM name")
    
    # Determine number of samples
    if N_samples is None:
        N_samples = data_config["N_samples"]
    
    # Determine parallelization settings
    if max_workers == -1:
        max_workers = cpu_count()
    use_parallel = HAS_PATHOS and max_workers is not None and max_workers > 1 and N_samples > 1
    if use_parallel:
        print(f"  Parallel processing enabled with {max_workers} workers (pathos available: {HAS_PATHOS})")
    
    # Set up individual directory if needed
    if save and individual and benchmarkpath is not None:
        incr_dir = benchmark_individual_dir(benchmarkpath, N_ctx, gm_name, N_samples, suffix)
        os.makedirs(incr_dir, exist_ok=True)
        
        # Check if we have saved input data from a previous run (for reproducibility on resume)
        input_data_file = incr_dir / "input_data.pkl"
        
        # Check which samples already exist (for resuming)
        existing_samples = set(int(f.stem.split('_')[1]) for f in incr_dir.glob("sample_*.pkl"))
        start_sample = len(existing_samples)
        
        if input_data_file.exists() and start_sample > 0:
            # Resume: load the original input data to ensure consistency
            print(f"  Resuming from sample {start_sample}/{N_samples} (found {start_sample} existing samples)")
            print(f"  Loading original input data for consistency...")
            with open(input_data_file, 'rb') as f:
                input_data = pickle.load(f)
            y_batch = input_data['y']
            contexts_batch = input_data['contexts']
            pars_batch = input_data['pars']
        else:
            # Fresh start: generate new samples and save input data
            print(f"  Generating {N_samples} samples...")
            if gm_name == 'NonHierarchicalGM':
                contexts_batch, _, y_batch, pars_batch = gm.generate_batch(N_samples, return_pars=True)
            elif gm_name == 'HierarchicalGM':
                _, _, _, _, _, contexts_batch, _, y_batch, pars_batch = gm.generate_batch(N_samples, return_pars=True)
            
            # Save input data for potential resume
            input_data = {'y': y_batch, 'contexts': contexts_batch, 'pars': pars_batch}
            with open(input_data_file, 'wb') as f:
                pickle.dump(input_data, f)
            print(f"  Input data saved")
    else:
        existing_samples = set()
        start_sample = 0
        incr_dir = None  # No saving in non-individual mode
        # Generate samples (non-individual mode)
        print(f"  Generating {N_samples} samples...")
        if gm_name == 'NonHierarchicalGM':
            contexts_batch, _, y_batch, pars_batch = gm.generate_batch(N_samples, return_pars=True)
        elif gm_name == 'HierarchicalGM':
            _, _, _, _, _, contexts_batch, _, y_batch, pars_batch = gm.generate_batch(N_samples, return_pars=True)
    
    # Process sample-by-sample with individual saving (supports parallel processing)
    if individual and save and benchmarkpath is not None:
        # Build list of samples to process (skip already-processed ones)
        samples_to_process = [i for i in range(N_samples) if i not in existing_samples]
        
        if len(samples_to_process) == 0:
            print(f"  All {N_samples} samples already processed.")
        else:
            print(f"  Processing {len(samples_to_process)} samples with individual saving...")
            
            # Prepare arguments for each sample
            args_list = [
                (i, 
                 y_batch[i], 
                 contexts_batch[i] if contexts_batch is not None else None,
                 {key: val[i] for key, val in pars_batch.items()},
                 N_ctx,
                 n_iter,
                 incr_dir)
                for i in samples_to_process
            ]
            
            if use_parallel:
                # Parallel processing with pathos
                print(f"  Using {max_workers} parallel workers...")
                with ProcessingPool(nodes=max_workers) as pool:
                    # Use imap for progress tracking
                    results = list(tqdm(
                        pool.imap(_process_single_kf_sample, args_list),
                        total=N_samples,
                        desc="KF fitting (parallel)",
                        initial=start_sample
                    ))
            else:
                # Sequential processing with progress bar
                for args in tqdm(args_list, desc="KF fitting", total=N_samples, initial=start_sample):
                    _process_single_kf_sample(args)
        
        # Aggregate all individual files into final benchmark file
        print(f"  Aggregating individual files...")
        benchmark_kit = aggregate_individual_benchmarks(benchmarkpath, N_ctx, gm_name, N_samples, suffix)
        
    else:
        # Original batch processing (faster but no crash recovery)
        print(f"  Fitting Kalman filters (batch mode)...")
        if N_ctx == 1:
            kalman_mu_pred, kalman_sigma_pred = kalman_fit_predict_batch(y_batch, n_iter=n_iter)
        else:
            # Convert context labels to responsibilities for multicontext KF
            responsibilities_batch = np.array([
                contexts_to_responsibilities(contexts_batch[i], N_ctx) 
                for i in range(len(contexts_batch))
            ])  # Shape: (N_samples, T, N_ctx)
            kalman_mu_pred, kalman_sigma_pred = kalman_fit_predict_multicontext_batch(
                y_batch, responsibilities_batch, n_iter=n_iter
            )
        
        # Compute MSE
        mses = ((kalman_mu_pred - y_batch[:, MIN_OBS_FOR_EM:]) ** 2).mean(axis=1)

        benchmark_kit = {
            'y': y_batch,
            'contexts': contexts_batch,
            'mu_kal_pred': kalman_mu_pred,
            'sigma_kal_pred': kalman_sigma_pred,
            'pars': pars_batch, 
            'perf': mses,
            'n_ctx': N_ctx,
            'min_obs_for_em': MIN_OBS_FOR_EM
        }

        if save:
            benchmark_file = benchmark_filename(benchmarkpath, N_ctx, gm_name, N_samples, suffix=suffix)
            if benchmarkpath is not None and not benchmark_file.parent.exists():
                os.makedirs(benchmark_file.parent)
            with open(benchmark_file, 'wb') as f:
                pickle.dump(benchmark_kit, f)

    return benchmark_kit


def benchmarks_pars_viz(benchmarks, data_config, N_ctx, gm_name, save_path=None, suffix=''):
    """
    Visualize parameter distributions from benchmark data.
    
    Args:
        benchmarks: Dictionary with 'y', 'mu_kal_pred', 'sigma_kal_pred', 'perf', 'pars'
                   where 'pars' is a dictionary with keys: 'tau', 'lim', 'si_stat', 'si_q', 'si_r'
                   Note: 'mu_kal_pred' has shape (N_samples, T-3) - predictions for y[:, 3:]
        data_config: Data configuration dictionary
        save_path: Optional path to save visualizations (defaults to benchmarks/ folder)
        suffix: Optional suffix to add to filenames (e.g., 'train', 'test')
    """
    param_bins = bin_params(data_config)
    y = benchmarks['y']
    # Use prediction estimates for MSE computation (matches model evaluation)
    # mu_kal has shape (N_samples, T-MIN_OBS_FOR_EM) and predicts y[:, MIN_OBS_FOR_EM:]
    min_obs = benchmarks.get('min_obs_for_em', MIN_OBS_FOR_EM)
    mu_kal = benchmarks['mu_kal_pred']
    sigma_kal = benchmarks['sigma_kal_pred']
    mse_kal = benchmarks['perf']
    pars_kal = benchmarks['pars']
    
    # Compute binned metrics - use y[:, min_obs:] to match mu_kal dimensions
    binned_metrics_df = map_binned_params_2_metrics(param_bins, y[:, min_obs:], mu_kal, pars_kal)
    
    # Set up save path - include gm_name only when N_ctx > 1 to distinguish different GMs
    if save_path is None:
        save_path = Path(os.path.abspath(os.path.dirname(__file__))) / 'benchmarks' / get_ctx_gm_subpath(N_ctx, gm_name) / f'visualizations{data_config["N_samples"]}'
    os.makedirs(save_path, exist_ok=True)
    
    # Add suffix to filename if provided
    suffix_str = f'_{suffix}' if suffix else ''
    label_str = f' ({suffix})' if suffix else ''
    
    # Extract parameters from dictionary, handling multi-context case
    # For tau: if 2D (multi-context), take first context (std); otherwise use as-is
    tau_vals = pars_kal['tau']
    if tau_vals.ndim > 1:
        tau_vals = tau_vals[:, 0]  # Use first context (std) for visualization
    
    si_stat_vals = pars_kal['si_stat']
    si_r_vals = pars_kal['si_r']
    
    # Compute si_q: si_q = si_stat * sqrt(2*tau - 1) / tau
    si_q_vals = si_stat_vals * np.sqrt(2 * tau_vals - 1) / tau_vals
    
    # Visualize parameter distributions
    print(f"  Saving parameter distribution plots to {save_path}")
    
    # Map param_bins keys to pars_kal dictionary keys
    param_mapping = {'tau': tau_vals, 'si_stat': si_stat_vals, 'si_r': si_r_vals}
    
    # Plot tau, si_stat, si_r from param_bins
    for param_name in param_bins.keys():
        plt.figure(figsize=(10, 5))
        plt.hist(param_mapping[param_name], bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Distribution of {param_name}{label_str}')
        plt.xlabel(param_name)
        plt.ylabel('Frequency')
        plt.savefig(save_path / f'param_distribution_{param_name}{suffix_str}.png')
        plt.close()
    
    # Plot si_q (computed parameter)
    plt.figure(figsize=(10, 5))
    plt.hist(si_q_vals, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.title(f'Distribution of si_q{label_str}')
    plt.xlabel('si_q')
    plt.ylabel('Frequency')
    plt.savefig(save_path / f'param_distribution_si_q{suffix_str}.png')
    plt.close()
    
    # Save binned metrics
    binned_metrics_df.to_csv(save_path / f'binned_metrics_kalman{suffix_str}.csv', index=False)
    print(f"  Saved binned metrics to {save_path / f'binned_metrics_kalman{suffix_str}.csv'}")


def train(model, model_config, lr, lr_id, h_dim, model_name, gm_name, data_config, save_path, benchmarks=None, device=DEVICE, seq_len_viz=None, data_mode='single_ctx', learning_objective='obs_ctx', kappa=0.5):
    """
    Train the model to predict the next observation (and optionally infer contexts).
    
    Args:
        model: The RNN model to train (SimpleRNN, VRNN, or ModuleNetwork)
        model_config: Model configuration dictionary
        lr: Learning rate
        lr_id: Learning rate index for logging
        h_dim: Hidden dimension size
        model_name: Name of the model
        gm_name: Name of the generative model
        data_config: Data configuration dictionary
        save_path: Path to save results
        benchmarks: Optional benchmark dictionary containing Kalman filter results for comparison
        device: Device to run on (cuda/cpu)
        seq_len_viz: Sequence length for visualization (optional)
        data_mode: Data mode - 'single_ctx' (N_ctx=1) or 'multi_ctx' (N_ctx>1)
        learning_objective: Learning objective for ModuleNetwork - 'obs', 'ctx', or 'obs_ctx' (default)
                           Only used when data_mode='multi_ctx' and model is ModuleNetwork
        kappa: Weight for observation loss in combined loss (only used when learning_objective='obs_ctx')
               total_loss = kappa * obs_loss + (1 - kappa) * ctx_loss
    
    """
    # Validate data mode
    if data_mode not in VALID_DATA_MODES:
        raise ValueError(f"Invalid data_mode: {data_mode}. Must be one of {VALID_DATA_MODES}")
    
    # Check model compatibility with data mode
    if data_mode == 'multi_ctx' and model.name != 'module_network':
        raise ValueError(f"data_mode='multi_ctx' requires ModuleNetwork, got {model.name}")
    
    # Validate learning objective for ModuleNetwork
    if model.name == 'module_network' and learning_objective not in VALID_LEARNING_OBJECTIVES:
        raise ValueError(f"Invalid learning_objective: {learning_objective}. Must be one of {VALID_LEARNING_OBJECTIVES}")
    
    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=model_config["weight_decay"])
    
    # Pass model to GPU or CPU
    model.to(device)

    # Determine if we have benchmarks for comparison
    use_benchmarks = benchmarks is not None
    
    # Define data generative model
    if gm_name == 'NonHierarchicalGM': gm = NonHierachicalAuditGM(data_config)
    elif gm_name == 'HierarchicalGM': gm = HierarchicalAuditGM(data_config)
    else: raise ValueError("Invalid GM name")
    

    # Prepare to save the results
    lr_title = f"Model: {model_name} | Learning rate: {lr:>6.0e} | #units: {h_dim}"
    os.makedirs(save_path / 'samples/', exist_ok=True)

    epoch_steps         = []
    train_losses_report = []
    train_steps         = []
    valid_losses_report = []
    valid_steps         = []
    valid_mse_report    = []
    valid_sigma_report  = []
    if use_benchmarks:
        model_kf_mse_report = []

    # Define loss instance using Objective class for model-agnostic loss computation
    loss_function = torch.nn.GaussianNLLLoss(reduction='mean')
    # For ModuleNetwork with context learning, add CrossEntropyLoss for context classification
    if data_mode == 'multi_ctx' and learning_objective in ['ctx', 'obs_ctx']:
        loss_function_ctx = torch.nn.CrossEntropyLoss(reduction='mean')
        objective = Objective(loss_function, loss_func_ctx=loss_function_ctx) # TODO: make Objective more modular 
    else:
        objective = Objective(loss_function)

    # Track learning of model parameters 
    weights_updates   = []
    param_names     = model.state_dict().keys()

     ### TRAINING 
    print("TRAINING \n")

    # Epochs
    for epoch in tqdm(range(model_config["num_epochs"]), desc="Epochs", leave=False):
        train_losses = []
        tt = time.time()
        model.train()

        # Generate N_samples=batch_size per batch in N_batch=n_batches
        for batch in range(model_config["n_batches"]):

            optimizer.zero_grad()
            
            # Generate data (N_samples samples) using helper function
            batch_data = prepare_batch_data(gm, gm_name, data_mode, device, return_pars=False)
            y = batch_data['y']
            contexts = batch_data['contexts']  # None if data_mode='single_ctx'

            # Always train to predict the next observation
            # Input: y[0:T-1], Target: y[1:T]
            model_output = model(y[:, :-1, :])
            loss = compute_model_loss(model, objective, y, model_output, data_mode, 
                                       learning_objective=learning_objective,
                                       contexts_tensor=contexts, kappa=kappa)

            # Learning model weigths
            # Only compute weight changes when you're going to log them
            if batch % model_config["batch_res"] == model_config["batch_res"]-1:
                weights_before = [w.detach().clone() for w in model.parameters()]
            
            loss.backward()     
            optimizer.step()

            if batch % model_config["batch_res"] == model_config["batch_res"]-1:
                weights_after = [w.detach().clone() for w in model.parameters()]

            train_losses.append(float(loss.detach().cpu().numpy().item()))
            

            # Logging and reporting
            if batch % model_config["batch_res"] == model_config["batch_res"]-1:
                train_steps.append(epoch * model_config["n_batches"] + batch)
                
                # Store loss
                train_losses_report.append(float(loss.detach().cpu().numpy().item()))
                loss_report = np.mean(train_losses[-model_config["batch_res"]:])
                sprint=f'LR: {lr:>6.0e}; epoch: {epoch:0>3}; batch: {batch:>3}; loss: {float(loss.detach().cpu().numpy().item()):>7.4f}; batch loss: {loss_report:7.4f}; time: {time.time()-tt:>.2f}; training step: {epoch * model_config["n_batches"] + batch}'
                logfilename = f'{save_path}/training_loss_lr{lr_id}.txt'
                with open(logfilename, 'a') as f:
                    f.write(f'{sprint}\n')
                tt = time.time()

                # Store evolution of parameters
                weights_update = torch.stack([torch.mean((w_after-w_before)**2) for w_before, w_after in zip(weights_before, weights_after)])
                weights_updates.append(weights_update)
            
        avg_train_loss = np.mean(train_losses) # average loss for current epoch


        ### VALIDATION
        model.eval()

        with torch.no_grad():    
            # If benchmarks available, use benchmark batch, else, generate new batch
            if use_benchmarks:
                bench_data = prepare_benchmark_data(benchmarks, data_mode, device)
                y = bench_data['y']
                contexts_valid = bench_data['contexts']  # None if data_mode='single_ctx'
                mu_kal_pred = bench_data['mu_kal_pred']
                sigma_kal_pred = bench_data['sigma_kal_pred']
                mse_kal = bench_data['mse_kal']
                pars_bench = bench_data['pars']
                min_obs = bench_data['min_obs_for_em']
            else:
                batch_data = prepare_batch_data(gm, gm_name, data_mode, device, return_pars=True)
                y = batch_data['y']
                contexts_valid = batch_data['contexts']
                pars_bench = batch_data['pars']
            
            
            # Always predict next observation
            # Input: y[0:T-1], Target: y[1:T]
            model_output = model(y[:, :-1, :]) # model is given y[0:T-1] to predict y[1:T]
            valid_loss = compute_model_loss(model, objective, y, model_output, data_mode,
                                            learning_objective=learning_objective,
                                            contexts_tensor=contexts_valid, kappa=kappa).detach().cpu().numpy()

            # Extract model's predictions using helper function
            mu_estim, var_estim, context_output = extract_model_predictions(model, model_output)
            

            # Compute standard deviation                    
            sigma_estim = np.sqrt(var_estim)

            # Reshape variables
            y = y.detach().cpu().numpy().squeeze()

            # Compute MSE: compare model's prediction at t with observation at t+1
            mse_model = ((mu_estim - y[:, 1:])**2).mean()

            # If benchmarks available, also compare model output with Kalman filter PREDICTIONS
            if use_benchmarks:
                # Compare with Kalman filter predictions
                # mu_estim has shape (N_samples, T-1) predicting y[:, 1:]
                # mu_kal_pred has shape (N_samples, T-MIN_OBS_FOR_EM) predicting y[:, MIN_OBS_FOR_EM:]
                # Align by slicing mu_estim to match KF predictions (skip first MIN_OBS_FOR_EM-1 predictions)
                min_obs = benchmarks.get('min_obs_for_em', MIN_OBS_FOR_EM)
                mu_estim_aligned = mu_estim[:, min_obs - 1:]  # shape: (N_samples, T-MIN_OBS_FOR_EM)
                mse_model2kal = ((mu_estim_aligned - mu_kal_pred)**2).mean()
            
            # Save valid samples for this epoch
            # Use PREDICTION estimates for visualization (both model and KF predict y[t+1])
            if epoch % model_config["epoch_res"] == model_config["epoch_res"]-1:
                if use_benchmarks:
                    # Pass KF predictions for visualization
                    # mu_estim has shape (N_samples, T-1) predicting y[:, 1:]
                    # mu_kal_pred has shape (N_samples, T-MIN_OBS_FOR_EM) predicting y[:, MIN_OBS_FOR_EM:]
                    plot_samples(y, mu_estim, sigma_estim, params=pars_bench, save_path=f'{save_path}/samples/lr{lr_id}-epoch-{epoch:0>3}_samples', title=lr_title, kalman_mu=mu_kal_pred, kalman_sigma=sigma_kal_pred, data_config=data_config, seq_len=seq_len_viz, min_obs_for_em=min_obs) 
                else:
                    plot_samples(y, mu_estim, sigma_estim, params=pars_bench, save_path=f'{save_path}/samples/lr{lr_id}-epoch-{epoch:0>3}_samples', title=lr_title, data_config=data_config, seq_len=seq_len_viz)

            # Store valid metrics for this epoch
            valid_mse_report.append(mse_model)
            valid_losses_report.append(valid_loss) # avg loss for epoch (over batches)
            valid_steps.append(epoch*model_config["n_batches"]+model_config["n_batches"])
            epoch_steps.append(epoch)
            sigma_estim_avg = sigma_estim.mean()
            valid_sigma_report.append(sigma_estim_avg)
            if use_benchmarks:
                model_kf_mse_report.append(mse_model2kal)
            
            # Save epoch log
            sprint=f'LR: {lr:>6.0e}; epoch: {epoch:>3}; mean var: {sigma_estim_avg:>7.2f}; mean MSE: {mse_model:>7.2f}; time: {time.time()-tt:>.2f}; training step: {epoch * model_config["n_batches"] + model_config["n_batches"]}'
            if use_benchmarks:
                sprint += f'; Model-KF MSE: {mse_model2kal:>7.2f}'
            logfilename = save_path / f'training_log_lr{lr_id}.txt'
            with open(logfilename, 'a') as f:
                f.write(f'{sprint}\n')

    # Print last epoch report
    print(f"Model: {model.name:>4}, LR: {lr:>6.0e}, Epoch: {epoch:>3}, Training Loss: {avg_train_loss:>7.2f}, Valid Loss: {valid_loss:>7.2f}")
            

    # Save training logs at the end of epochs loop
    plot_losses(train_steps, valid_steps, train_losses_report, valid_losses_report, x_label='Training steps', y_label='Loss', title=lr_title, save_path=save_path/f'loss_trainvalid_lr{lr_id}.png')
    plot_variance(epoch_steps, valid_sigma_report, title=lr_title, save_path=save_path/f'variance_valid_lr{lr_id}.png')
    if use_benchmarks:
        plot_mse(epoch_steps, valid_mse_report, title=lr_title, save_path=save_path/f'mse_valid_lr{lr_id}.png', mse_kal=mse_kal, model_mse_kal=model_kf_mse_report)
    else:
        plot_mse(epoch_steps, valid_mse_report, title=lr_title, save_path=save_path/f'mse_valid_lr{lr_id}.png')
    

    # Save model's weights
    torch.save(model.state_dict(), f'{save_path}/lr{lr_id}_weights.pth')

    # Plot training weights updates
    if len(weights_updates) > 0:
        weights_updates = torch.stack(weights_updates, dim=1)
        plot_weights(train_steps, weights_updates, list(param_names), lr_title, save_path=save_path/f'weights_updates_lr{lr_id}.png')




def test(model, model_config, lr, lr_id, gm_name, data_config, save_path, device=DEVICE, benchmarks=None, data_mode='single_ctx', learning_objective='obs_ctx', kappa=0.5):
    """
    Test the model on held-out data.
    
    Args:
        model: The RNN model to test (SimpleRNN, VRNN, or ModuleNetwork)
        model_config: Model configuration dictionary
        lr: Learning rate (for logging)
        lr_id: Learning rate index for logging
        gm_name: Name of the generative model
        data_config: Data configuration dictionary
        save_path: Path to save results
        device: Device to run on (cuda/cpu)
        benchmarks: Optional benchmark dictionary containing Kalman filter results for comparison
        data_mode: 'single_ctx' (N_ctx=1) or 'multi_ctx' (N_ctx>1)
        learning_objective: Learning objective for ModuleNetwork - 'obs', 'ctx', or 'obs_ctx' (default)
        kappa: Weight for observation loss in combined loss (only used when learning_objective='obs_ctx')
    
    Returns:
        tuple: 
            - Without benchmarks: (test_sigma, test_mse)
            - With benchmarks: (test_sigma, test_mse, test_mse_kal, test_mse_model2kal)
              where test_mse_kal is the KF MSE wrt target (insight on task difficulty)
              and test_mse_model2kal is the model MSE wrt KF predictions (model performance wrt upper bound)
        
        When params_testing is enabled, saves binned metrics CSV with:
            - mse: Model MSE wrt target per parameter bin
            - mse_kal: KF MSE wrt target per parameter bin (if benchmarks provided)
            - mse_model2kal: Model MSE wrt KF predictions per parameter bin (if benchmarks provided)
    """

    ### TESTING
    print("TESTING \n")

    # Paths 
    if not os.path.exists(save_path):
        raise ValueError("Model has not been trained yet - no folder found")

    # Load the saved weights
    model.load_state_dict(torch.load(f'{save_path}/lr{lr_id}_weights.pth'))
    model.to(device)
    model.eval()  # Switch to evaluation mode

    # Determine if we have benchmarks for comparison
    use_benchmarks = benchmarks is not None
    
    # Define data generative model
    if gm_name == 'NonHierarchicalGM': gm = NonHierachicalAuditGM(data_config)
    elif gm_name == 'HierarchicalGM': gm = HierarchicalAuditGM(data_config)
    else: raise ValueError("Invalid GM name")

    # If testing data parameters and performance
    if data_config["params_testing"]:
        # NOTE: same spacing as sampled in audit_gm.py "if param_testing" but regularly spaced
        # NOTE: here assuming that all 3 params are being tested
        param_bins = bin_params(data_config)

    test_sigma = []
    test_mse = []
    test_mse_kal = []  # KF MSE wrt target
    if use_benchmarks:
        test_mse_model2kal = []

    # If benchmarks provided, use benchmark data; else generate new test data
    if use_benchmarks:
        bench_data = prepare_benchmark_data(benchmarks, data_mode, device)
        y = bench_data['y']
        contexts_test = bench_data['contexts']  # None if data_mode='single_ctx'
        mu_kal_pred = bench_data['mu_kal_pred']
        sigma_kal_pred = bench_data['sigma_kal_pred']
        pars = bench_data['pars']
        mse_kal = bench_data['mse_kal']
        test_mse_kal = mse_kal  # Store KF MSE for return
    else:
        # Generate data using helper function
        batch_data = prepare_batch_data(gm, gm_name, data_mode, device, return_pars=True)
        # Override batch size for testing
        data_config_test = data_config.copy()
        data_config_test['N_samples'] = model_config["batch_size_test"]
        gm_test = NonHierachicalAuditGM(data_config_test) if gm_name == 'NonHierarchicalGM' else HierarchicalAuditGM(data_config_test)
        batch_data = prepare_batch_data(gm_test, gm_name, data_mode, device, return_pars=True)
        y = batch_data['y']
        contexts_test = batch_data['contexts']
        pars = batch_data['pars']
   

    with torch.no_grad():
        # Always predict next observation
        # Input: y[0:T-1], Target: y[1:T]
        model_output = model(y[:, :-1, :])

        # Extract model's predictions using helper function
        mu_estim, var_estim, context_output = extract_model_predictions(model, model_output)

        # Compute standard deviation                    
        sigma_estim = np.sqrt(var_estim)
        test_sigma.append(sigma_estim)

        # Reshape variables
        y = y.detach().cpu().numpy().squeeze()

        # Compute MSE: compare model's prediction at t with observation at t+1
        mse_model = ((mu_estim - y[:, 1:])**2).mean()
        test_mse.append(mse_model)

        # If benchmarks available, also compare model output with Kalman filter PREDICTIONS
        if use_benchmarks:
            # Compare with Kalman filter predictions
            # mu_estim has shape (N_samples, T-1) predicting y[:, 1:]
            # mu_kal_pred has shape (N_samples, T-MIN_OBS_FOR_EM) predicting y[:, MIN_OBS_FOR_EM:]
            # Align by slicing mu_estim to match KF predictions (skip first MIN_OBS_FOR_EM-1 predictions)
            min_obs = benchmarks.get('min_obs_for_em', MIN_OBS_FOR_EM)
            mu_estim_aligned = mu_estim[:, min_obs - 1:]  # shape: (N_samples, T-MIN_OBS_FOR_EM)
            mse_model2kal = ((mu_estim_aligned - mu_kal_pred)**2).mean()
            test_mse_model2kal.append(mse_model2kal)
        
    # Track performance along data parameters
    if data_config["params_testing"]:
        # Pass KF predictions if available for comparison metrics per bin
        # Note: KF predictions are shorter, need alignment for binned metrics
        if use_benchmarks:
            min_obs = benchmarks.get('min_obs_for_em', MIN_OBS_FOR_EM)
            # Align model predictions and observations with KF predictions
            mu_estim_aligned = mu_estim[:, min_obs - 1:]
            y_aligned = y[:, min_obs:]
            binned_metrics_df = map_binned_params_2_metrics(param_bins, y_aligned, mu_estim_aligned, pars, mu_kal=mu_kal_pred)
        else:
            binned_metrics_df = map_binned_params_2_metrics(param_bins, y[:, 1:], mu_estim, pars)
        binned_metrics_df.to_csv((save_path/f'test_binned_metrics_lr{lr_id}.csv'), index=False)

    if use_benchmarks:
        return test_sigma, test_mse, test_mse_kal, test_mse_model2kal
    else:
        return test_sigma, test_mse




def plot_weights(train_steps, weights_updates, names, title, save_path):
    plt.figure(figsize=(10, 5))
    for param in range(weights_updates.shape[0]):
        plt.plot(train_steps, weights_updates[param], label=names[param], alpha=0.8)
    plt.yscale('log')
    plt.xlabel("Training steps")
    plt.ylabel("Weights updates (MSE between steps)")
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_mse(valid_steps, valid_diff, title, save_path, mse_kal=None, model_mse_kal=None):
    # Plot MSE depending on what's given
    plt.figure(figsize=(10, 5))
    plt.plot(valid_steps, valid_diff, label='model')
    if mse_kal is not None:
        plt.axhline(y=np.mean(mse_kal), color='tab:orange', linestyle='--', label='kalman (mean)')
    if model_mse_kal is not None:
        plt.plot(valid_steps, model_mse_kal, label='model-kalman')
    plt.xlabel("Epoch steps")
    plt.yscale('log')
    plt.ylabel(f"Validation MSE")
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def plot_variance(valid_steps, valid_sigma, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(valid_steps, valid_sigma, label="valid std")
    plt.xlabel("Epoch steps")
    plt.ylabel("Valid variance (std)")
    plt.yscale('log')
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_losses(train_steps, valid_steps, train_losses_report, valid_losses_report, x_label, y_label, title, save_path):
# def plot_losses(train_steps, valid_steps, epoch_steps, train_losses_report, valid_losses_report, x_label, y_label, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_losses_report, label="train loss", color='tab:blue')
    plt.plot(valid_steps, valid_losses_report, label="valid loss", color='tab:orange')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    # sec = plt.gca().secondary_xaxis(location=0)
    # sec.set_xticks(epoch_steps)
    # sec.set_xlabel(f'epoch=', loc='left', labelpad=-9)
    plt.savefig(save_path)
    plt.close()


def plot_samples(obs, mu_estim, sigma_estim, save_path, title=None, params=None, kalman_mu=None, kalman_sigma=None, seq_len=None, data_config=None, hidden_states=None, contexts=None, min_obs_for_em=None):
    """
    Plot observation sequences with model and optional Kalman filter predictions.
    
    Note on alignment:
    - obs has length T (full sequence)
    - mu_estim has length T-1 (predictions for y[1:T])
    - kalman_mu has length T-MIN_OBS_FOR_EM (predictions for y[MIN_OBS_FOR_EM:T])
      KF needs MIN_OBS_FOR_EM observations for EM, so starts predicting later
    - Model predictions plotted at x=[1, T-1], KF predictions at x=[MIN_OBS_FOR_EM, T-1]
    
    Args:
        min_obs_for_em: Minimum observations needed for KF EM (default: MIN_OBS_FOR_EM constant)
    """
    # Set default for min_obs_for_em
    if min_obs_for_em is None:
        min_obs_for_em = MIN_OBS_FOR_EM
    
    # Convert to numpy if it's a tensor
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    obs = obs.squeeze() # for safety
    
    # Get N_ctx from data_config
    n_ctx = data_config.get("N_ctx", 1) if data_config is not None else 1

    # For HierarchicalGM, plot only the last 8 blocks if sequence is long enough
    if data_config is not None and data_config.get("gm_name") == "HierarchicalGM":
        N_tones = data_config.get("N_tones", 8)
        last_8_blocks_len = 8 * N_tones
        if obs.shape[1] > last_8_blocks_len:
            seq_len = last_8_blocks_len
    
    # Truncate sequences if seq_len is provided
    if seq_len is not None:
        # Take last seq_len observations: obs indices [T-seq_len, ..., T-1]
        obs = obs[:, -seq_len:]
        # Take last seq_len-1 model predictions that predict obs[1:seq_len] (the last seq_len-1 obs)
        mu_estim = mu_estim[:, -(seq_len-1):]
        sigma_estim = sigma_estim[:, -(seq_len-1):]
        # Take last seq_len-min_obs_for_em KF predictions that predict obs[min_obs_for_em:seq_len]
        if kalman_mu is not None:
            kalman_mu = kalman_mu[:, -(seq_len-min_obs_for_em):]
        if kalman_sigma is not None:
            kalman_sigma = kalman_sigma[:, -(seq_len-min_obs_for_em):]
        if hidden_states is not None:
            hidden_states = hidden_states[:, -seq_len:, :]
        if contexts is not None:
            contexts = contexts[:, -seq_len:]
            
    # for some 8 randomly sampled sequences out of the whole batch of length obs.shape[0]
    N = min(8, obs.shape[0])
    for id, i in enumerate(np.random.choice(range(N), size=(N,), replace=False)):
            
        plt.figure(figsize=(20, 6))
        
        # Plot observation (full length of truncated sequence)
        plt.plot(range(len(obs[i])), obs[i], color='tab:blue', label='y_obs', alpha=0.8)

        # Plot model prediction as a distribution (with uncertainty)
        # After truncation: mu_estim has seq_len-1 elements predicting obs[1:seq_len]
        # So mu_estim[0] predicts obs[1], mu_estim[k] predicts obs[k+1]
        # Plot at positions [1, 2, ..., seq_len-1] to align with the observations they predict
        estim_x = range(1, len(obs[i]))  # [1, ..., len(obs)-1]
        plt.plot(estim_x, mu_estim[i], color='k', label='y_hat (model pred)')
        plt.fill_between(estim_x, mu_estim[i]-sigma_estim[i], mu_estim[i]+sigma_estim[i], color='k', alpha=0.2)
        
        # Plot Kalman PREDICTION if provided
        # KF predictions start at min_obs_for_em (needs 3 obs for EM before first prediction)
        # kalman_mu has length T-min_obs_for_em, predicting obs[min_obs_for_em:]
        # kalman_mu[k] predicts obs[k+min_obs_for_em]
        if kalman_mu is not None and kalman_sigma is not None:
            kal_x = range(min_obs_for_em, len(obs[i]))  # [min_obs, ..., len(obs)-1]
            plt.plot(kal_x, kalman_mu[i], label='y_kal_pred', color='green', alpha=0.8)
            plt.fill_between(kal_x, kalman_mu[i]-kalman_sigma[i], kalman_mu[i]+kalman_sigma[i], color='green', alpha=0.2)

        # Plot hidden states if provided
        if hidden_states is not None:
            if n_ctx > 1 and contexts is not None:
                # Plot only hidden state of current active context
                # --> Stack hidden states for each unique context along a new axis
                unique_contexts = np.unique(contexts)
                hidden_states_active = np.stack(
                    [np.where((contexts == ctx)[..., None], hidden_states, 0) for ctx in unique_contexts],
                    axis=-1
                )
                hidden_states_active = np.sum(hidden_states_active, axis=-1)  # Sum to get active context hidden states
            else:
                hidden_states_active = hidden_states  # Single context case  
            plt.plot(range(len(hidden_states)), hidden_states[i, :], label=f'hidden state', color='orange', alpha=0.8)

        # For different contexts, add vertical lines to indicate context switches
        if n_ctx > 1 and contexts is not None:
            context_changes = np.where(np.diff(contexts[i]) != 0)[0] + 1  # +1 to get the index of the new context start
            for cc in context_changes:
                plt.axvline(x=cc, color='red', linestyle='--', alpha=0.5)

        plt.legend()
        plot_title = title
        if params is not None:
            # Extract parameters for sample i, handling both dict and legacy array formats
            # Dictionary format: access by key
            tau_i = params['tau'][i] if params['tau'].ndim > 0 else params['tau']
            lim_i = params['lim'][i] if params['lim'].ndim > 0 else params['lim']
            si_stat_i = params['si_stat'][i] if np.asarray(params['si_stat']).ndim > 0 else params['si_stat']
            si_q_i = params['si_q'][i] if params['si_q'].ndim > 0 else params['si_q']
            si_r_i = params['si_r'][i] if np.asarray(params['si_r']).ndim > 0 else params['si_r']
            
            # Compute si_r/si_stat ratio
            si_ratio = si_r_i / si_stat_i if si_stat_i != 0 else np.nan
            
            if n_ctx == 2:
                # Two-context case: format like audit_gm with std/dvt labels
                tau_str = f"std: {tau_i[0]:.2f}, dvt: {tau_i[1]:.2f}"
                lim_str = f"std: {lim_i[0]:.2f}, dvt: {lim_i[1]:.2f}"
                si_q_str = f"std: {si_q_i[0]:.2f}, dvt: {si_q_i[1]:.2f}"
                
                # Compute d and d_eff for two-context case
                # d_eff = lim[1] - lim[0] (actual distance)
                # d = d_eff / si_eff where si_eff = sqrt(si_stat**2 + si_r**2)
                d_eff = lim_i[1] - lim_i[0]
                si_eff = np.sqrt(si_stat_i**2 + si_r_i**2)
                d = d_eff / si_eff if si_eff != 0 else np.nan
                
                title_line1 = f"tau: {tau_str}  |  lim: {lim_str}  | d: {d:.2f},  d_eff: {d_eff:.2f}"
                title_line2 = f"si_stat: {si_stat_i:.2f}  |  si_q: {si_q_str}  |  si_r: {si_r_i:.2f}  |  si_r/si_stat: {si_ratio:.2f}"
                plot_title = f"{title}\n{title_line1}\n{title_line2}"
            else:
                # Single-context case
                title_line1 = f"tau: {tau_i:.2f}, lim: {lim_i:.2f}, si_stat: {si_stat_i:.2f}, si_q: {si_q_i:.2f}, si_r: {si_r_i:.2f}, si_r/si_stat: {si_ratio:.2f}"
                plot_title = f"{title}\n{title_line1}"

        plt.title(plot_title)
        plt.savefig(f'{save_path}_s{id}.png')
        plt.close()



def load_or_compute_benchmarks(data_config, model_config, N_ctx, gm_name, visualize=True, max_workers=None):
    """
    Load precomputed benchmarks if available, otherwise compute and save them.
    
    Supports resuming from partial individual computation if the job was interrupted.
    
    Args:
        data_config: Data configuration dictionary
        model_config: Model configuration dictionary (needed for batch_size_test)
        N_ctx: Number of contexts
        gm_name: Name of the generative model
        visualize: Whether to visualize parameter distributions for newly computed benchmarks
        max_workers: Number of parallel workers for KF fitting. None or 1 = sequential, 
                     >1 = parallel with that many workers, -1 = use all CPU cores.
    
    Returns:
        tuple: (benchmarks_train, benchmarks_test)
    """
    benchmarkpath = Path(os.path.abspath(os.path.dirname(__file__))) / 'benchmarks'
    benchmarkfile_train = benchmark_filename(benchmarkpath, N_ctx, gm_name, data_config["N_samples"], suffix='train')
    benchmarkfile_test = benchmark_filename(benchmarkpath, N_ctx, gm_name, data_config["N_samples"], suffix='test')
    
    # Check for individual directories (partial computation from previous interrupted run)
    incr_dir_train = benchmark_individual_dir(benchmarkpath, N_ctx, gm_name, data_config["N_samples"], suffix='train')
    incr_dir_test = benchmark_individual_dir(benchmarkpath, N_ctx, gm_name, model_config['batch_size_test'], suffix='test')

    benchmarks_exist = benchmarkfile_train.exists() and benchmarkfile_test.exists()
    
    if not benchmarks_exist:
        # Check if we have partial individual results to resume from
        has_partial_train = incr_dir_train.exists() and len(list(incr_dir_train.glob("sample_*.pkl"))) > 0
        has_partial_test = incr_dir_test.exists() and len(list(incr_dir_test.glob("sample_*.pkl"))) > 0
        
        if has_partial_train or has_partial_test:
            print("Found partial individual benchmarks from previous run, resuming...")
        else:
            print("Computing benchmarks...")
        
        print(f"  Using {'context-aware' if N_ctx > 1 else 'standard'} Kalman filter (N_ctx={N_ctx})")
        
        # Compute/resume benchmarks (individual=True enables sample-by-sample saving and resume)
        benchmarks_train = compute_benchmarks(data_config, N_ctx, gm_name, n_iter=5, benchmarkpath=benchmarkpath, save=True, suffix='train', individual=True, max_workers=max_workers)
        benchmarks_test = compute_benchmarks(data_config, N_ctx, gm_name, N_samples=model_config['batch_size_test'], n_iter=5, benchmarkpath=benchmarkpath, save=True, suffix='test', individual=True, max_workers=max_workers)
        
        # Visualize parameter distributions
        if visualize:
            print("Visualizing parameter distributions...")
            benchmarks_pars_viz(benchmarks_train, data_config, N_ctx, gm_name, suffix='train')
            benchmarks_pars_viz(benchmarks_test, data_config, N_ctx, gm_name, suffix='test')
    else:
        # Load precomputed benchmarks
        print("Loading precomputed benchmarks...")
        with open(benchmarkfile_train, 'rb') as f:
            benchmarks_train = pickle.load(f)
        with open(benchmarkfile_test, 'rb') as f:
            benchmarks_test = pickle.load(f)
        print("Benchmarks loaded successfully.")
    
    return benchmarks_train, benchmarks_test


def pipeline_single_config(N_ctx, gm_name, h_dim, lr_id, learning_rate, model_name, model_config, data_config, benchmarks_train=None, benchmarks_test=None, device=DEVICE, train_only=False, test_only=False, data_mode='single_ctx', learning_objective='obs_ctx', kappa=0.5):
    """
    Run training and testing for a single model configuration.
    
    Args:
        N_ctx: Number of contexts
        gm_name: Name of the generative model
        h_dim: Hidden dimension size
        lr_id: Learning rate index
        learning_rate: Learning rate value
        model_name: Name of the model ('rnn', 'vrnn', or 'module_network')
        model_config: Model configuration dictionary
        data_config: Data configuration dictionary
        benchmarks_train: Training benchmarks
        benchmarks_test: Test benchmarks
        device: Device to run on
        train_only: If True, only train
        test_only: If True, only test
        data_mode: 'single_ctx' (N_ctx=1) or 'multi_ctx' (N_ctx>1)
        learning_objective: Learning objective for ModuleNetwork - 'obs', 'ctx', or 'obs_ctx' (default)
        kappa: Weighting factor for obs_ctx objective (default: 0.5)
    """
    # Define save path - include gm_name only when N_ctx > 1
    # For ModuleNetwork, also include learning_objective in path to separate different variants
    # For obs_ctx, also include kappa to distinguish different balance settings
    if model_name == 'module_network':
        if learning_objective == 'obs_ctx':
            save_path = Path(os.path.abspath(os.path.dirname(__file__))) / 'training_results' / get_ctx_gm_subpath(N_ctx, gm_name) / f'{model_name}_{learning_objective}_kappa{kappa}/'
        else:
            save_path = Path(os.path.abspath(os.path.dirname(__file__))) / 'training_results' / get_ctx_gm_subpath(N_ctx, gm_name) / f'{model_name}_{learning_objective}/'
    else:
        save_path = Path(os.path.abspath(os.path.dirname(__file__))) / 'training_results' / get_ctx_gm_subpath(N_ctx, gm_name) / f'{model_name}_h{h_dim}/'
    
    # Define models with current hidden dimension
    if model_name == 'rnn':
        rnn_config = {
            'input_dim': model_config['input_dim'],
            'output_dim': model_config['output_dim'],
            'hidden_dim': h_dim,
            'n_layers': model_config['rnn_n_layers'],
            'device': device
        }
        model = SimpleRNN(rnn_config)
    elif model_name == 'vrnn':
        vrnn_config = {
            'input_dim': model_config['input_dim'],
            'output_dim': model_config['output_dim'],
            'latent_dim': h_dim,
            'phi_x_dim': h_dim,
            'phi_z_dim': h_dim,
            'phi_prior_dim': h_dim,
            'rnn_hidden_states_dim': h_dim,
            'rnn_n_layers': model_config['rnn_n_layers'],
            'device': device
        }
        model = VRNN(vrnn_config)
    elif model_name == 'module_network':
        # ModuleNetwork uses fixed hidden dims from module_config (not overridden by h_dim)
        model = ModuleNetwork(model_config)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    if not test_only:
        # Train
        kappa_str = f", kappa={kappa}" if learning_objective == 'obs_ctx' else ""
        print(f"\nTraining {model_name} with h_dim={h_dim}, lr={learning_rate}, learning_objective={learning_objective}{kappa_str}...")
        train(model, model_config=model_config, lr=learning_rate, lr_id=lr_id, h_dim=h_dim, gm_name=gm_name, model_name=model_name, data_config=data_config, save_path=save_path, device=device, benchmarks=benchmarks_train, seq_len_viz=model_config["seq_len_viz"], data_mode=data_mode, learning_objective=learning_objective, kappa=kappa)

    if not train_only:
        # Test
        kappa_str = f", kappa={kappa}" if learning_objective == 'obs_ctx' else ""
        print(f"\nTesting {model_name} with h_dim={h_dim}, lr={learning_rate}, learning_objective={learning_objective}{kappa_str}...")
        test(model, model_config, lr=learning_rate, lr_id=lr_id, gm_name=gm_name, data_config=data_config, save_path=save_path, device=device, benchmarks=benchmarks_test, data_mode=data_mode, learning_objective=learning_objective)


def pipeline_multi_config(model_config, data_config, benchmark_only=False, device=DEVICE, test_only=False, train_only=False, data_mode='single_ctx', learning_objective='obs_ctx', max_workers=None, skip_benchmarks=False):
    """
    Run the full training and testing pipeline with multiple hyperparameter configurations.
    
    Always trains to predict the next observation and uses Kalman filter (or context-aware KF 
    for N_ctx > 1) as benchmark for comparison.
    
    Args:
        model_config: Model configuration dictionary
        data_config: Data configuration dictionary  
        benchmark_only: If True, only compute benchmarks and exit
        device: Device to run on (cuda/cpu)
        test_only: If True, only test (skip training)
        train_only: If True, only train (skip testing)
        data_mode: 'single_ctx' (N_ctx=1) or 'multi_ctx' (N_ctx>1)
        learning_objective: Learning objective for ModuleNetwork - 'obs', 'ctx', or 'obs_ctx' (default)
                           Can also be a list to train all variants: ['obs', 'ctx', 'obs_ctx']
        max_workers: Number of parallel workers for KF benchmark fitting.
                     None or 1 = sequential, >1 = parallel, -1 = use all CPU cores.
        skip_benchmarks: If True, skip benchmark computation and train without KF comparison.
                        Useful for faster iteration when benchmarks are slow to compute.
    """
    
    N_ctx = data_config["N_ctx"]
    gm_name = data_config["gm_name"]

    # Load or compute benchmarks (unless skipped)
    if skip_benchmarks:
        print("Skipping benchmark computation (skip_benchmarks=True)")
        benchmarks_train, benchmarks_test = None, None
    else:
        benchmarks_train, benchmarks_test = load_or_compute_benchmarks(
            data_config, model_config, N_ctx, gm_name, visualize=True, max_workers=max_workers
        )
    
    # Exit early if only benchmarking
    if benchmark_only:
        print("Benchmark step complete. Exiting (benchmark_only=True).")
        return
    
    # Support both list and scalar for hidden dims
    hidden_dims = model_config.get("rnn_hidden_dims", model_config.get("rnn_hidden_dim", 64))
    if not isinstance(hidden_dims, list):
        hidden_dims = [hidden_dims]
    
    # Support list of learning objectives for training multiple variants
    learning_objectives = learning_objective if isinstance(learning_objective, list) else [learning_objective]
           
    for model_name in model_config["model"]:
        for lr_id, learning_rate in enumerate(model_config["learning_rates"]): 
            for h_dim in hidden_dims:
                # For ModuleNetwork, iterate over learning objectives to train different variants
                if model_name == 'module_network':
                    for obj in learning_objectives:
                        # For 'obs_ctx' objective, iterate over kappa values
                        if obj == 'obs_ctx':
                            for kappa in DEFAULT_KAPPA_VALUES:
                                pipeline_single_config(N_ctx, gm_name, h_dim, lr_id, learning_rate, model_name, model_config, data_config, benchmarks_train=benchmarks_train, benchmarks_test=benchmarks_test, device=device, train_only=train_only, test_only=test_only, data_mode=data_mode, learning_objective=obj, kappa=kappa)
                                
                                # Explicit garbage collection to free memory between configurations
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                        else:
                            # For 'obs' or 'ctx' objectives, kappa is not used (default 0.5)
                            pipeline_single_config(N_ctx, gm_name, h_dim, lr_id, learning_rate, model_name, model_config, data_config, benchmarks_train=benchmarks_train, benchmarks_test=benchmarks_test, device=device, train_only=train_only, test_only=test_only, data_mode=data_mode, learning_objective=obj)
                            
                            # Explicit garbage collection to free memory between configurations
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    # ModuleNetwork uses fixed hidden dims, so only run once per learning rate
                    break
                else:
                    pipeline_single_config(N_ctx, gm_name, h_dim, lr_id, learning_rate, model_name, model_config, data_config, benchmarks_train=benchmarks_train, benchmarks_test=benchmarks_test, device=device, train_only=train_only, test_only=test_only, data_mode=data_mode, learning_objective=learning_objective)
                
                    # Explicit garbage collection to free memory between configurations
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()



def pipeline_train_valid(model_config, data_config, test_only=False, train_only=False, data_mode='single_ctx', learning_objective='obs_ctx', max_workers=None, skip_benchmarks=False):
    """
    Run the full model training pipeline.
    
    Args:
        model_config: Model configuration dictionary
        data_config: Data configuration dictionary
        test_only: If True, only test
        train_only: If True, only train
        data_mode: 'single_ctx' (N_ctx=1) or 'multi_ctx' (N_ctx>1)
        learning_objective: Learning objective for ModuleNetwork - 'obs', 'ctx', or 'obs_ctx' (default)
                           Can also be a list to train all variants: ['obs', 'ctx', 'obs_ctx']
        max_workers: Number of parallel workers for KF benchmark fitting.
                     None or 1 = sequential, >1 = parallel, -1 = use all CPU cores.
        skip_benchmarks: If True, skip benchmark computation and train without KF comparison.
                        Useful for faster iteration when benchmarks are slow to compute.
    """
    # STEP 1: Pre-compute benchmarks and visualize parameter distributions
    # This computes Kalman filter estimates on training and test data, and saves:
    #   - benchmarks/benchmarks_<N>_<GM>_train.pkl
    #   - benchmarks/benchmarks_<N>_<GM>_test.pkl
    #   - benchmarks/visualizations_<N>/param_distribution_*.png
    #   - benchmarks/visualizations_<N>/binned_metrics_kalman.csv
    
    if not skip_benchmarks:
        print("\n" + "="*60)
        print("STEP 1: Computing benchmarks (Kalman filter baseline)")
        print("="*60)
        pipeline_multi_config(model_config, data_config, benchmark_only=True, test_only=test_only, train_only=train_only, data_mode=data_mode, learning_objective=learning_objective, max_workers=max_workers, skip_benchmarks=False)
    else:
        print("\n" + "="*60)
        print("STEP 1: Skipping benchmarks (skip_benchmarks=True)")
        print("="*60)
    
    
    # STEP 2: Train the RNN model with different learning rates and number of hidden units
    # This will train the model using the pre-computed benchmarks for validation
    # Uncomment the following line to run training:
    
    print("\n" + "="*60)
    print("STEP 2: Training RNN models")
    print("="*60)
    pipeline_multi_config(model_config, data_config, benchmark_only=False, test_only=test_only, train_only=train_only, data_mode=data_mode, learning_objective=learning_objective, max_workers=max_workers, skip_benchmarks=skip_benchmarks)


def pipeline_test(model_config, data_config, data_mode='single_ctx', learning_objective='obs_ctx'):
    """
    Test trained models on held-out data.
    
    Args:
        model_config: Model configuration dictionary
        data_config: Data configuration dictionary
        data_mode: 'single_ctx' (N_ctx=1) or 'multi_ctx' (N_ctx>1)
        learning_objective: Learning objective for ModuleNetwork - 'obs', 'ctx', or 'obs_ctx' (default)
    """
    print("\n" + "="*60)
    print("Testing trained RNN models")
    print("="*60)
    pipeline_multi_config(model_config, data_config, test_only=True, data_mode=data_mode, learning_objective=learning_objective)


if __name__=='__main__':
    # DO NOTHING: run relevant pipeline_single.py or pipeline_multi.py
    pass
