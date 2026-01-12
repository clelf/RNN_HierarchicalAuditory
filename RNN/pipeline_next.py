from model import SimpleRNN, VRNN
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


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from PreProParadigm.audit_gm import NonHierachicalAuditGM, HierarchicalAuditGM


# Check which folder exists and append the correct path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
kalman_folder = None
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
if os.path.exists(os.path.join(base_path, 'Kalman')):
    from Kalman.kalman import kalman_tau, plot_estim, kalman_batch, kalman_fit_batch, kalman_fit_predict_batch, kalman_fit_context_aware_batch, kalman_fit_context_aware_predict_batch, MIN_OBS_FOR_EM
elif os.path.exists(os.path.join(base_path, 'KalmanFilterViz1D')):
    from KalmanFilterViz1D.kalman import kalman_tau, plot_estim, kalman_batch, kalman_fit_batch, kalman_fit_predict_batch, kalman_fit_context_aware_batch, kalman_fit_context_aware_predict_batch, MIN_OBS_FOR_EM
else:
    raise ImportError("Neither 'Kalman' nor 'KalmanFilterViz1D' folder found.")




DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FREQ_MIN = 1400
FREQ_MAX = 1650



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


def compute_benchmarks(data_config, N_ctx, gm_name, N_samples=None, n_iter=5, benchmarkpath=None, save=False, suffix=''):
    """
    Benchmarks the Kalman filter on a batch of data, tracking MSE along parameter configurations.
    Uses standard Kalman filter for single-context (N_ctx=1) and context-aware Kalman filter
    for multi-context (N_ctx>1) scenarios.

    Args:
        data_config (dict): Configuration parameters for the data generative model.
        N_samples (int, optional): Number of samples to generate. If None, uses data_config["N_samples"].
        n_iter (int): Number of iterations for kalman_fit_batch.
        benchmarkpath (Path or None): Path to save the benchmark file. If None, benchmarks are not saved.
        save (bool): Whether to save the benchmark data.
        suffix (str): Suffix for the benchmark filename.

    Returns:
        dict: A dictionary containing observations, contexts (if N_ctx>1), Kalman estimates, 
              parameters, and performance metrics.
    """
    
    # Define data generative model
    if gm_name == 'NonHierarchicalGM': gm = NonHierachicalAuditGM(data_config)
    elif gm_name == 'HierarchicalGM': gm = HierarchicalAuditGM(data_config)
    else: raise ValueError("Invalid GM name")
    
    # Generate a batch of data
    if N_samples is None:
        N_samples = data_config["N_samples"]
    else:
        N_samples = N_samples
    
    # Generate batch - includes contexts and states
    if gm_name == 'NonHierarchicalGM':
        contexts_batch, _, y_batch, pars_batch = gm.generate_batch(N_samples, return_pars=True)
    elif gm_name == 'HierarchicalGM':
        # rules, rules_long, dpos, timbres, timbres_long, contexts, states, obs, pars
        _, _, _, _, _, contexts_batch, _, y_batch, pars_batch = gm.generate_batch(N_samples, return_pars=True)

    # Fit Kalman filters - get predictions (for MSE and visualization)
    # Note: kalman_fit_predict functions return T-MIN_OBS_FOR_EM predictions (EM needs min 3 observations)
    #   y_hat[t] predicts y[t+MIN_OBS_FOR_EM] based on parameters fit on y[0:t+MIN_OBS_FOR_EM]
    if data_config["N_ctx"] == 1:
        # Standard Kalman filter for single context
        kalman_mu_pred, kalman_sigma_pred = kalman_fit_predict_batch(y_batch, n_iter=n_iter)
    else:
        # Context-aware Kalman filter for multiple contexts
        kalman_mu_pred, kalman_sigma_pred = kalman_fit_context_aware_predict_batch(
            y_batch, contexts_batch, n_iter=n_iter
        )
    
    # Compute MSE using PREDICTIONS: compare KF prediction at t with observation at t+MIN_OBS_FOR_EM
    # kalman_mu_pred has shape (N_samples, T-MIN_OBS_FOR_EM) where kalman_mu_pred[:, t] predicts y[:, t+MIN_OBS_FOR_EM]
    mses = ((kalman_mu_pred - y_batch[:, MIN_OBS_FOR_EM:]) ** 2).mean(axis=1)  # shape: (N_samples,)

    # Return observations, Kalman estimates, parameters, and performance
    benchmark_kit = {
        'y': y_batch,                        # shape: (N_samples, T)
        'contexts': contexts_batch,
        'mu_kal_pred': kalman_mu_pred,       # shape: (N_samples, T-MIN_OBS_FOR_EM), predicts y[:, MIN_OBS_FOR_EM:]
        'sigma_kal_pred': kalman_sigma_pred, # shape: (N_samples, T-MIN_OBS_FOR_EM)
        'pars': pars_batch, 
        'perf': mses,
        'n_ctx': N_ctx,
        'min_obs_for_em': MIN_OBS_FOR_EM     # Store for downstream alignment
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


def train(model, model_config, lr, lr_id, h_dim, model_name, gm_name, data_config, save_path, benchmarks=None, device=DEVICE, seq_len_viz=None):
    """
    Train the model to predict the next observation.
    
    Args:
        model: The RNN model to train
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
    
    """
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

    # Define loss instance
    loss_function   = torch.nn.GaussianNLLLoss(reduction='mean') # GaussianNLL

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
            
            # Generate data (N_samples samples)
            if gm_name == 'NonHierarchicalGM':
                _, _, y = gm.generate_batch(return_pars=False)
            elif gm_name == 'HierarchicalGM':
                _, _, _, _, _, _, _, y = gm.generate_batch(return_pars=False)

            y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(device)

            # Always train to predict the next observation
            # Input: y[0:T-1], Target: y[1:T]
            model_output = model(y[:, :-1, :])
            loss = model.loss(y[:, 1:, :], model_output, loss_func=loss_function)

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
                y = benchmarks['y']
                y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(-1).to(device)
                # Use predictions for MSE comparison with model and visualization
                # mu_kal_filt = benchmarks['mu_kal_filt']  # Filtered estimates - not used
                # sigma_kal_filt = benchmarks['sigma_kal_filt']
                mu_kal_pred = benchmarks['mu_kal_pred']
                sigma_kal_pred = benchmarks['sigma_kal_pred']
                mse_kal = benchmarks['perf']
                pars_bench = benchmarks['pars']
            else:
                _, _, y, pars_bench = gm.generate_batch(return_pars=True)
                y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(device)
            
            
            # Always predict next observation
            # Input: y[0:T-1], Target: y[1:T]
            model_output = model(y[:, :-1, :]) # model is given y[0:T-1] to predict y[1:T]
            valid_loss = model.loss(y[:, 1:, :], model_output, loss_func=loss_function).detach().cpu().numpy()

            # Get estimated distribution
            if model.name=='vrnn': model_output_dist = model_output[0]
            else: model_output_dist = model_output

            # Extract model's output
            mu_estim = model_output_dist[...,0].detach().cpu().numpy()
            var_estim = (F.softplus(model_output_dist[..., 1]) + 1e-6).detach().cpu().numpy()
            

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

    if model_config["use_minmax"]:
        return minmax_train
    else:
        return None



def test(model, model_config, lr, lr_id, gm_name, data_config, save_path, minmax_train, device=DEVICE, benchmarks=None):
    """
    Test the model on held-out data.
    
    Args:
        model: The RNN model to test
        model_config: Model configuration dictionary
        lr: Learning rate (for logging)
        lr_id: Learning rate index for logging
        data_config: Data configuration dictionary
        save_path: Path to save results
        minmax_train: MinMaxScaler used during training (or None)
        device: Device to run on (cuda/cpu)
        benchmarks: Optional benchmark dictionary containing Kalman filter results for comparison
    
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
        return
        # raise ValueError("Model has not been trained yet - no folder found")

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
        y = benchmarks['y']
        y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(-1).to(device)
        mu_kal_pred = benchmarks['mu_kal_pred']
        sigma_kal_pred = benchmarks['sigma_kal_pred']
        pars = benchmarks['pars']
        mse_kal = benchmarks['perf']
        test_mse_kal = mse_kal  # Store KF MSE for return
    else:
        # Generate data
        _, _, y, pars = gm.generate_batch(N_samples=model_config["batch_size_test"], return_pars=True)
        y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(device)
   

    with torch.no_grad():
        # Always predict next observation
        # Input: y[0:T-1], Target: y[1:T]
        model_output = model(y[:, :-1, :])

        # Get estimated distribution
        if model.name=='vrnn': model_output_dist = model_output[0]
        else: model_output_dist = model_output
        
        # Extract model's output
        mu_estim = model_output_dist[...,0].detach().cpu().numpy()
        var_estim = (F.softplus(model_output_dist[..., 1]) + 1e-6).detach().cpu().numpy()

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



def load_or_compute_benchmarks(data_config, model_config, N_ctx, gm_name, visualize=True):
    """
    Load precomputed benchmarks if available, otherwise compute and save them.
    
    Args:
        data_config: Data configuration dictionary
        model_config: Model configuration dictionary (needed for batch_size_test)
        N_ctx: Number of contexts
        gm_name: Name of the generative model
        visualize: Whether to visualize parameter distributions for newly computed benchmarks
    
    Returns:
        tuple: (benchmarks_train, benchmarks_test)
    """
    benchmarkpath = Path(os.path.abspath(os.path.dirname(__file__))) / 'benchmarks'
    benchmarkfile_train = benchmark_filename(benchmarkpath, N_ctx, gm_name, data_config["N_samples"], suffix='train')
    benchmarkfile_test = benchmark_filename(benchmarkpath, N_ctx, gm_name, data_config["N_samples"], suffix='test')

    benchmarks_exist = benchmarkfile_train.exists() and benchmarkfile_test.exists()
    
    if not benchmarks_exist:
        # Compute benchmarks if not existing
        # Uses standard KF for N_ctx=1, context-aware KF for N_ctx>1
        print("Computing benchmarks...")
        print(f"  Using {'context-aware' if N_ctx > 1 else 'standard'} Kalman filter (N_ctx={N_ctx})")
        benchmarks_train = compute_benchmarks(data_config, N_ctx, gm_name, n_iter=5, benchmarkpath=benchmarkpath, save=True, suffix='train')
        benchmarks_test = compute_benchmarks(data_config, N_ctx, gm_name, N_samples=model_config['batch_size_test'], n_iter=5, benchmarkpath=benchmarkpath, save=True, suffix='test')
        
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


def pipeline_single_config(N_ctx, gm_name, h_dim, lr_id, learning_rate, model_name, model_config, data_config, benchmarks_train=None, benchmarks_test=None, device=DEVICE, train_only=False, test_only=False):
    
    # Define save path - include gm_name only when N_ctx > 1
    save_path = Path(os.path.abspath(os.path.dirname(__file__))) / 'training_results' / get_ctx_gm_subpath(N_ctx, gm_name) / f'{model_name}_h{h_dim}/'
    
    # Define models with current hidden dimension
    if model_name == 'rnn':
        model = SimpleRNN(x_dim=model_config['input_dim'], output_dim=model_config['output_dim'], hidden_dim=h_dim, n_layers=model_config['rnn_n_layers'], device=device)
    else:
        model = VRNN(x_dim=model_config['input_dim'], output_dim=model_config['output_dim'], latent_dim=h_dim, phi_x_dim=h_dim, phi_z_dim=h_dim, phi_prior_dim=h_dim, rnn_hidden_states_dim=h_dim, rnn_n_layers=model_config['rnn_n_layers'], device=device)
    
    minmax=None
    if not test_only:
        # Train
        print(f"\nTraining {model_name} with h_dim={h_dim}, lr={learning_rate}...")
        minmax = train(model, model_config=model_config, lr=learning_rate, lr_id=lr_id, h_dim=h_dim, gm_name=gm_name, model_name=model_name, data_config=data_config, save_path=save_path, device=device, benchmarks=benchmarks_train, seq_len_viz=model_config["seq_len_viz"])

    if not train_only:
        # Test
        print(f"\nTesting {model_name} with h_dim={h_dim}, lr={learning_rate}...")
        test(model, model_config, lr=learning_rate, lr_id=lr_id, gm_name=gm_name, data_config=data_config, save_path=save_path, minmax_train=minmax, device=device, benchmarks=benchmarks_test)


def pipeline_multi_config(model_config, data_config, benchmark_only=False, device=DEVICE, test_only=False, train_only=False):
    """
    Run the full training and testing pipeline with multiple hyperparameter configurations.
    
    Always trains to predict the next observation and uses Kalman filter (or context-aware KF 
    for N_ctx > 1) as benchmark for comparison.
    
    Args:
        model_config: Model configuration dictionary
        data_config: Data configuration dictionary  
        benchmark_only: If True, only compute benchmarks and exit
        device: Device to run on (cuda/cpu)
    """
    
    N_ctx = data_config["N_ctx"]
    gm_name = data_config["gm_name"]

    # Load or compute benchmarks
    benchmarks_train, benchmarks_test = load_or_compute_benchmarks(
        data_config, model_config, N_ctx, gm_name, visualize=True
    )
    
    # Exit early if only benchmarking
    if benchmark_only:
        print("Benchmark step complete. Exiting (benchmark_only=True).")
        return
           
    for h_dim in model_config["rnn_hidden_dim"]:
        for lr_id, learning_rate in enumerate(model_config["learning_rates"]): 
            for model_name in model_config["model"]:
                pipeline_single_config(N_ctx, gm_name, h_dim, lr_id, learning_rate, model_name, model_config, data_config, benchmarks_train=benchmarks_train, benchmarks_test=benchmarks_test, device=device, train_only=train_only, test_only=test_only)
                
                # Explicit garbage collection to free memory between configurations
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()



def pipeline_model(model_config, data_config, test_only=False, train_only=False):
    # STEP 1: Pre-compute benchmarks and visualize parameter distributions
    # This computes Kalman filter estimates on training and test data, and saves:
    #   - benchmarks/benchmarks_<N>_<GM>_train.pkl
    #   - benchmarks/benchmarks_<N>_<GM>_test.pkl
    #   - benchmarks/visualizations_<N>/param_distribution_*.png
    #   - benchmarks/visualizations_<N>/binned_metrics_kalman.csv
    
    print("\n" + "="*60)
    print("STEP 1: Computing benchmarks (Kalman filter baseline)")
    print("="*60)
    pipeline_multi_config(model_config, data_config, benchmark_only=True, test_only=test_only, train_only=train_only)
    
    
    # STEP 2: Train the RNN model with different learning rates and number of hidden units
    # This will train the model using the pre-computed benchmarks for validation
    # Uncomment the following line to run training:
    
    print("\n" + "="*60)
    print("STEP 2: Training RNN models")
    print("="*60)
    pipeline_multi_config(model_config, data_config, benchmark_only=False, test_only=test_only, train_only=train_only)


def pipeline_test(model_config, data_config):
    # Test trained models on held-out data
    print("\n" + "="*60)
    print("Testing trained RNN models")
    print("="*60)
    pipeline_multi_config(model_config, data_config, test_only=True)


if __name__=='__main__':
    # DO NOTHING: run relevant pipeline_single.py or pipeline_multi.py
    pass
