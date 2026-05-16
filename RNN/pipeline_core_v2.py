"""
Refactored core pipeline functions with clear, linear flow.

This module provides the essential building blocks:
- Model creation
- Training loop (single config)
- Testing loop (single config)
- Data provision

Key design principles:
1. Each function has a SINGLE responsibility
2. No nested loops for hyperparameter iteration (that's in the runner)
3. Clear input/output contracts
4. Reuse existing utility functions from pipeline_next.py
"""

import os
import sys
import time
import gc
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
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

# Local imports - reuse existing code (files are now in the same directory)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from model import SimpleRNN, VRNN, ObsCtxModuleNetwork, PopulationNetwork
from objectives import Objective


# Generative models (PreProParadigm is one level up from RNN/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from PreProParadigm.audit_gm import NonHierarchicalAuditGM, HierarchicalAuditGM

# Local config (config_v2 is now in the same directory)
from config_v2 import RunConfig, run_config_to_model_dict, run_config_to_training_dict

# Check which folder exists and append the correct path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
if os.path.exists(os.path.join(base_path, 'Kalman')):
    from Kalman.kalman import MIN_OBS_FOR_EM, kalman_online_fit_predict, kalman_online_fit_predict_multicontext, contexts_to_probabilities
elif os.path.exists(os.path.join(base_path, 'KalmanFilterViz1D')):
    from KalmanFilterViz1D.kalman import MIN_OBS_FOR_EM, kalman_online_fit_predict, kalman_online_fit_predict_multicontext, contexts_to_probabilities
else:
    raise ImportError("Neither 'Kalman' nor 'KalmanFilterViz1D' folder found.")


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


# =============================================================================
# Data utilities
# =============================================================================

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
    batch = gm.generate_batch(return_pars=return_pars)
    
    # Convert observations to tensor
    y_tensor = torch.tensor(batch['obs'], dtype=torch.float, requires_grad=False).unsqueeze(2).to(device)
    
    batch_data = {
        'y': y_tensor,
        'pars': batch['pars'] if return_pars else None
    }

    # Prepare context tensor if in multi-context mode
    if data_mode == 'multi_ctx':
        # contexts is a numpy array of shape (batch, seq_len) with integer context labels
        batch_data.update({
        'contexts': torch.tensor(batch['contexts'], dtype=torch.long, requires_grad=False).to(device)
        })
        if gm_name == 'HierarchicalGM':
            # Assuming here HGM would always have cues
            batch_data.update({
                'rules': torch.tensor(batch['rules_long'], dtype=torch.long, requires_grad=False).unsqueeze(2).to(device),
                'dpos': torch.tensor(batch['dpos_long'], dtype=torch.long, requires_grad=False).unsqueeze(2).to(device),
            })
            # Transform cues to one-hot encoding for rule module input (shape: batch, seq_len, N_cues)
            q_onehot = torch.nn.functional.one_hot(
                    torch.tensor(batch['cues_long'], dtype=torch.long, requires_grad=False), 
                    num_classes=gm.N_cues
                ).float().to(device)
            batch_data.update({
                'q': q_onehot # Now (T, N, N_cat)
            })
    
    return batch_data

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
        contexts = benchmarks['contexts']
        if contexts is not None:
            result['contexts'] = torch.tensor(contexts, dtype=torch.long, requires_grad=False).to(device)
        else:
            result['contexts'] = None
    else:
        result['contexts'] = None
    
    return result


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


# =============================================================================
# Loss and prediction extraction
# =============================================================================

def compute_model_loss(model, objective, y_tensor, model_output, data_mode, 
                       learning_objective='obs_ctx', contexts_tensor=None, 
                       dpos_tensor=None, rules_tensor=None, kappa=0.5):
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
    elif model.name == 'population_network':
        return objective.loss(model, y_tensor[:, 1:, :], model_output,
                              target_ctx=contexts_tensor[:, 1:],
                              target_dpos=dpos_tensor[:, 1:], target_rule=rules_tensor[:, 1:],
                              learning_objective=learning_objective)
    else:
        # Standard RNN/VRNN loss
        return objective.loss(model, y_tensor[:, 1:, :], model_output)

def get_model_predictions(model, model_output):
    """
    Extract and process all model predictions from output based on model type.
    
    Handles extraction of observation estimates (mu, var, sigma) and optional
    categorical outputs (contexts, dpos, rules) with softmax probabilities and
    argmax predictions already computed.
    
    Parameters
    ----------
    model : nn.Module
        The model (SimpleRNN, VRNN, ModuleNetwork, or PopulationNetwork)
    model_output : tuple or torch.Tensor
        Output from model forward pass
    
    Returns
    -------
    dict
        Dictionary containing:
        - mu_estim: Mean observation estimates (numpy)
        - var_estim: Variance estimates (numpy)
        - sigma_estim: Standard deviation estimates (numpy)
        - ctx_prob: Context probabilities (N, T, N_ctx) or None
        - ctx_pred: Context predictions (N, T) or None
        - dpos_prob: Dpos probabilities (N, T-1, N_dpos) or None
        - dpos_pred: Dpos predictions (N, T-1) or None
        - rule_prob: Rule probabilities (N, T-1, N_rules) or None
        - rule_pred: Rule predictions (N, T-1) or None
        - output: Raw output dict for backward compatibility
    """
    # Parse model output based on model type
    if model.name == 'module_network':
        obs_output, context_output = model_output
        output = {'obs_dist': obs_output, 'ctx': context_output}
    elif model.name == 'population_network':
        obs_output, context_output, dpos_output, rule_output = model_output
        output = {'obs_dist': obs_output, 'ctx': context_output, 'dpos': dpos_output, 'rule': rule_output}
    elif model.name == 'vrnn':
        output = {'obs_dist': model_output[0]}
    else:
        output = {'obs_dist': model_output}
    
    # Extract and process observation estimates
    mu_estim = output['obs_dist'][..., 0].detach().cpu().numpy()
    var_estim = (F.softplus(output['obs_dist'][..., 1]) + 1e-6).detach().cpu().numpy()
    sigma_estim = np.sqrt(var_estim)
    
    # Initialize optional output fields
    ctx_prob = None
    ctx_pred = None
    dpos_prob = None
    dpos_pred = None
    rule_prob = None
    rule_pred = None
    
    # Process context outputs (if present)
    if 'ctx' in output and output['ctx'] is not None:
        ctx_prob = torch.softmax(output['ctx'], dim=-1).detach().cpu().numpy()
        ctx_pred = np.argmax(ctx_prob, axis=-1).squeeze()
    
    # Process dpos outputs (if present)
    if 'dpos' in output and output['dpos'] is not None:
        dpos_prob = torch.softmax(output['dpos'], dim=-1).detach().cpu().numpy()
        dpos_pred = np.argmax(dpos_prob, axis=-1).squeeze()
    
    # Process rule outputs (if present)
    if 'rule' in output and output['rule'] is not None:
        rule_prob = torch.softmax(output['rule'], dim=-1).detach().cpu().numpy()
        rule_pred = np.argmax(rule_prob, axis=-1).squeeze()
    
    return {
        'mu_estim': mu_estim,
        'var_estim': var_estim,
        'sigma_estim': sigma_estim,
        'ctx_prob': ctx_prob,
        'ctx_pred': ctx_pred,
        'dpos_prob': dpos_prob,
        'dpos_pred': dpos_pred,
        'rule_prob': rule_prob,
        'rule_pred': rule_pred,
        'output': output,  # Keep raw output for backward compatibility
    }

# =============================================================================
# Kalman filter usage
# =============================================================================

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
        mu_pred_i, sigma_pred_i, _ = kalman_online_fit_predict(y_i, n_iter=n_iter)
    else:
        # Convert context labels to responsibilities for multicontext KF
        responsibilities_i = contexts_to_probabilities(ctx_i, n_ctx)
        mu_pred_i, sigma_pred_i, _ = kalman_online_fit_predict_multicontext(
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


# =============================================================================
# Benchmark computation
# =============================================================================

def benchmark_individual_dir(benchmarkpath, N_ctx, gm_name, N_samples, suffix=''):
    """Get directory for individual benchmark files."""
    if suffix != '':
        suffix = '_' + suffix
    return benchmarkpath / get_ctx_gm_subpath(N_ctx, gm_name) / f"benchmarks_{N_samples}{suffix}_individual"


def benchmark_filename(benchmarkpath, N_ctx, gm_name, N_samples, suffix=''):
    if suffix != '':
        suffix = '_' + suffix
    return benchmarkpath / get_ctx_gm_subpath(N_ctx, gm_name) / f"benchmarks_{N_samples}{suffix}.pkl"



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



def compute_benchmarks(data_config, N_ctx, gm_name, N_samples=None, n_iter=5, benchmarkpath=None, save=False, suffix='', individual=True, max_cores=None):
    """
    Benchmarks the Kalman filter on a batch of data, tracking MSE along parameter configurations.
    Uses standard Kalman filter for single-context (N_ctx=1) and context-aware Kalman filter
    for multi-context (N_ctx>1) scenarios.
    
    Supports individual saving: each sample is saved individually so that progress is not lost
    if the computation is interrupted. Use individual=True (default) for long-running jobs.
    
    Supports parallel processing: set max_cores > 1 to process multiple samples simultaneously.

    Args:
        data_config (dict): Configuration parameters for the data generative model.
        N_samples (int, optional): Number of samples to generate. If None, uses data_config["N_samples"].
        n_iter (int): Number of iterations for kalman_fit_predict.
        benchmarkpath (Path or None): Path to save the benchmark file. If None, benchmarks are not saved.
        save (bool): Whether to save the benchmark data.
        suffix (str): Suffix for the benchmark filename.
        individual (bool): If True, save each sample individually to allow resuming on crash.
                           The samples are aggregated at the end. Default: True.
        max_cores (int, optional): Number of parallel cores for KF fitting.
                                     None or 1 = sequential processing (default).
                                     >1 = parallel processing with that many cores.
                                     -1 = use all available CPU cores.

    Returns:
        dict: A dictionary containing observations, contexts (if N_ctx>1), Kalman estimates, 
              parameters, and performance metrics.
    """

    # Define data generative model
    if gm_name == 'NonHierarchicalGM': gm = NonHierarchicalAuditGM(data_config)
    elif gm_name == 'HierarchicalGM': gm = HierarchicalAuditGM(data_config)
    else: raise ValueError("Invalid GM name")

    # Determine number of samples
    if N_samples is None:
        N_samples = data_config["N_samples"]

    # Determine parallelization settings
    if max_cores == -1:
        max_cores = cpu_count()
    use_parallel = HAS_PATHOS and max_cores is not None and max_cores > 1 and N_samples > 1
    if use_parallel:
        print(f"  Parallel processing enabled with {max_cores} cores (pathos available: {HAS_PATHOS})")

    use_individual = individual and save and benchmarkpath is not None

    # --- Data generation (with optional resume from a previous run) ---
    existing_samples = set()
    incr_dir = None

    if use_individual:
        incr_dir = benchmark_individual_dir(benchmarkpath, N_ctx, gm_name, N_samples, suffix)
        os.makedirs(incr_dir, exist_ok=True)
        existing_samples = set(int(f.stem.split('_')[1]) for f in incr_dir.glob("sample_*.pkl"))
        start_sample = len(existing_samples)
        input_data_file = incr_dir / "input_data.pkl"

        if input_data_file.exists() and start_sample > 0:
            # Resume: reload original input data to guarantee consistency
            print(f"  Resuming from sample {start_sample}/{N_samples} (found {start_sample} existing samples)")
            with open(input_data_file, 'rb') as f:
                input_data = pickle.load(f)
            y_batch, contexts_batch, pars_batch = input_data['y'], input_data['contexts'], input_data['pars']
        else:
            # Fresh start: generate and persist input data for potential future resume
            print(f"  Generating {N_samples} samples...")
            batch = gm.generate_batch(N_samples, return_pars=True)
            y_batch = batch['obs']
            contexts_batch = batch['contexts']
            pars_batch = batch['pars']
            with open(input_data_file, 'wb') as f:
                pickle.dump({'y': y_batch, 'contexts': contexts_batch, 'pars': pars_batch}, f)
            print(f"  Input data saved to {input_data_file}")
    else:
        start_sample = 0
        print(f"  Generating {N_samples} samples...")
        batch = gm.generate_batch(N_samples, return_pars=True)
        y_batch = batch['obs']
        contexts_batch = batch['contexts']
        pars_batch = batch['pars']

    # --- KF fitting (sample-by-sample, sequential or parallel) ---
    samples_to_process = [i for i in range(N_samples) if i not in existing_samples]

    if len(samples_to_process) == 0:
        print(f"  All {N_samples} samples already processed.")
        results = []
    else:
        print(f"  Processing {len(samples_to_process)} samples...")
        args_list = [
            (i,
             y_batch[i],
             contexts_batch[i] if contexts_batch is not None else None,
             {key: val[i] for key, val in pars_batch.items()},
             N_ctx, n_iter, incr_dir)
            for i in samples_to_process
        ]

        if use_parallel:
            print(f"  Using {max_cores} parallel cores...")
            with ProcessingPool(nodes=max_cores) as pool:
                results = list(tqdm(
                    pool.imap(_process_single_kf_sample, args_list),
                    total=N_samples, desc="KF fitting (parallel)", initial=start_sample
                ))
        else:
            results = [
                _process_single_kf_sample(args)
                for args in tqdm(args_list, desc="KF fitting", total=N_samples, initial=start_sample)
            ]

    # --- Assemble benchmark_kit ---
    if use_individual:
        # Aggregate from individually saved files (handles both fresh and resumed runs)
        print(f"  Aggregating individual files...")
        benchmark_kit = aggregate_individual_benchmarks(benchmarkpath, N_ctx, gm_name, N_samples, suffix)
    else:
        # Build from in-memory results
        benchmark_kit = {
            'y': np.stack([s['y'] for s in results]),
            'contexts': np.stack([s['contexts'] for s in results]) if results[0]['contexts'] is not None else None,
            'mu_kal_pred': np.stack([s['mu_kal_pred'] for s in results]),
            'sigma_kal_pred': np.stack([s['sigma_kal_pred'] for s in results]),
            'pars': {key: np.stack([s['pars'][key] for s in results]) for key in results[0]['pars'].keys()},
            'perf': np.array([s['perf'] for s in results]),
            'n_ctx': N_ctx,
            'min_obs_for_em': MIN_OBS_FOR_EM,
        }
        if save and benchmarkpath is not None:
            benchmark_file = benchmark_filename(benchmarkpath, N_ctx, gm_name, N_samples, suffix=suffix)
            os.makedirs(benchmark_file.parent, exist_ok=True)
            with open(benchmark_file, 'wb') as f:
                pickle.dump(benchmark_kit, f)

    return benchmark_kit

def load_or_compute_benchmarks(data_config, model_config, N_ctx, gm_name, visualize=True, max_cores=None, benchmark_mode='both', suffix_tag=''):
    """
    Load precomputed benchmarks if available, otherwise compute and save them.
    
    Supports resuming from partial individual computation if the job was interrupted.
    
    Args:
        data_config: Data configuration dictionary
        model_config: Model configuration dictionary (needed for batch_size_test)
        N_ctx: Number of contexts
        gm_name: Name of the generative model
        visualize: Whether to visualize parameter distributions for newly computed benchmarks
        max_cores: Number of parallel cores for KF fitting. None or 1 = sequential, 
                     >1 = parallel with that many cores, -1 = use all CPU cores.
        benchmark_mode: Which benchmarks to compute. Options:
                       'both' (default): Compute/load both train and test benchmarks
                       'test_only': Compute/load only test benchmarks (train=None)
                       'train_only': Compute/load only train benchmarks (test=None)
        suffix_tag: Optional string appended to file/dir suffixes (e.g. 'unit_test')
                    to produce distinct benchmark files without overwriting existing ones.
    
    Returns:
        tuple: (benchmarks_train, benchmarks_test) - either may be None based on benchmark_mode
    """
    benchmarkpath = Path(os.path.abspath(os.path.dirname(__file__))) / 'benchmarks'
    tag = f'_{suffix_tag}' if suffix_tag else ''
    suffix_train = f'train{tag}'
    suffix_test = f'test{tag}'
    benchmarkfile_train = benchmark_filename(benchmarkpath, N_ctx, gm_name, data_config["N_samples"], suffix=suffix_train)
    benchmarkfile_test = benchmark_filename(benchmarkpath, N_ctx, gm_name, data_config["N_samples"], suffix=suffix_test)
    
    # Check for individual directories (partial computation from previous interrupted run)
    incr_dir_train = benchmark_individual_dir(benchmarkpath, N_ctx, gm_name, data_config["N_samples"], suffix=suffix_train)
    incr_dir_test = benchmark_individual_dir(benchmarkpath, N_ctx, gm_name, model_config['batch_size_test'], suffix=suffix_test)

    # Determine which benchmarks to compute based on mode
    compute_train = benchmark_mode in ['both', 'train_only']
    compute_test = benchmark_mode in ['both', 'test_only']
    
    benchmarks_train = None
    benchmarks_test = None
    
    # Handle training benchmarks
    if compute_train:
        if benchmarkfile_train.exists():
            print(f"Loading precomputed training benchmarks from {benchmarkfile_train}")
            with open(benchmarkfile_train, 'rb') as f:
                benchmarks_train = pickle.load(f)
        else:
            has_partial_train = incr_dir_train.exists() and len(list(incr_dir_train.glob("sample_*.pkl"))) > 0
            if has_partial_train:
                print(f"Found partial training benchmarks in {incr_dir_train}, resuming...")
            else:
                print(f"Computing training benchmarks (will save to {benchmarkfile_train})...")
            print(f"  Using {'context-aware' if N_ctx > 1 else 'standard'} Kalman filter (N_ctx={N_ctx})")
            benchmarks_train = compute_benchmarks(data_config, N_ctx, gm_name, n_iter=5, benchmarkpath=benchmarkpath, save=True, suffix=suffix_train, individual=True, max_cores=max_cores)
            if visualize:
                benchmarks_pars_viz(benchmarks_train, data_config, N_ctx, gm_name, suffix=suffix_train)
    
    # Handle test benchmarks
    if compute_test:
        if benchmarkfile_test.exists():
            print(f"Loading precomputed test benchmarks from {benchmarkfile_test}")
            with open(benchmarkfile_test, 'rb') as f:
                benchmarks_test = pickle.load(f)
        else:
            has_partial_test = incr_dir_test.exists() and len(list(incr_dir_test.glob("sample_*.pkl"))) > 0
            if has_partial_test:
                print(f"Found partial test benchmarks in {incr_dir_test}, resuming...")
            else:
                print(f"Computing test benchmarks (will save to {benchmarkfile_test})...")
            print(f"  Using {'context-aware' if N_ctx > 1 else 'standard'} Kalman filter (N_ctx={N_ctx})")
            benchmarks_test = compute_benchmarks(data_config, N_ctx, gm_name, N_samples=model_config['batch_size_test'], n_iter=5, benchmarkpath=benchmarkpath, save=True, suffix=suffix_test, individual=True, max_cores=max_cores)
            if visualize:
                benchmarks_pars_viz(benchmarks_test, data_config, N_ctx, gm_name, suffix=suffix_test)
    
    return benchmarks_train, benchmarks_test


# =============================================================================
# Metrics and binning
# =============================================================================

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


# =============================================================================
# Plotting
# =============================================================================

def plot_weights(train_steps, weights_updates, names, title, save_path):
    plt.figure(figsize=(10, 5))
    for param in range(weights_updates.shape[0]):
        plt.plot(train_steps, weights_updates[param].numpy(), alpha=0.8) # , label=names[param]
    plt.yscale('log')
    plt.xlabel("Training steps")
    plt.ylabel("Weights updates (MSE between steps)")
    # plt.legend()
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

def _get_dpos_color_map(n_dpos_classes=5):
    """
    Create a color mapping for deviant positions using jet colormap.
    Maps min value -> light, max value -> dark.
    Creates mappings for all possible dpos classes (default 5).
    
    Args:
        dpos_true: True deviant positions array of shape (N, T) or None
        dpos_prob: Predicted probabilities array of shape (N, T, N_classes) or None
        n_dpos_classes: Expected number of dpos classes (default 5). Used to create
                       complete color map even if not all classes appear in data.
    
    Returns:
        dict: Mapping from dpos class index to color (as hex string) for all classes
    """

    # Create mapping for all classes (0 to n_dpos_classes-1)
    dpos_cmap = plt.get_cmap('rainbow')
    if n_dpos_classes == 1:
        rgba = dpos_cmap(0.5)
        return {0: mcolors.rgb2hex(rgba[:3])}
    return {val: mcolors.rgb2hex(dpos_cmap(val / (n_dpos_classes - 1))[:3]) for val in range(n_dpos_classes)}


def _get_rule_color_map(n_rule_classes=2):
    """
    Create a color mapping for rules using jet colormap.
    Maps min value -> light, max value -> dark.
    Creates mappings for all possible rule classes (default 2).
    
    Args:
        rule_true: True rules array of shape (N, T) or None
        rule_prob: Predicted probabilities array of shape (N, T, N_classes) or None
        n_rule_classes: Expected number of rule classes (default 2). Used to create
                       complete color map even if not all classes appear in data.
    
    Returns:
        dict: Mapping from rule class index to color (as hex string) for all classes
    """
    
    # Create mapping for all classes (0 to n_rule_classes-1)
    rule_cmap = plt.get_cmap('plasma')
    if n_rule_classes == 1:
        rgba = rule_cmap(0.5)
        return {0: mcolors.rgb2hex(rgba[:3])}
    return {val: mcolors.rgb2hex(rule_cmap((val + 1) / (n_rule_classes + 1))[:3]) for val in range(n_rule_classes)}


def plot_sample(id, i, obs, mu_estim, sigma_estim, save_path, n_ctx, title=None, 
                params=None, kalman_mu=None, kalman_sigma=None, hidden_states=None,
                contexts=None, contexts_prob=None, contexts_pred=None,
                preds_aligned=False, shared_ylim=False, ylim=None, kal_x_start=None,
                dpos_true=None, dpos_pred=None, dpos_prob=None, rule_true=None, rule_pred=None, rule_prob=None, cues=None):
    """
    Plot a single observation sequence with model and optional Kalman filter predictions.
    
    Args:
        id: Figure index for naming
        i: Sample index (index into all arrays)
        obs: Observation sequences array with shape (N, T)
        mu_estim: Model estimates with shape (N, T-1)
        sigma_estim: Model uncertainties with shape (N, T-1)
        save_path: Path/prefix for saving the figure
        n_ctx: Number of contexts
        title: Base title for the plot
        params: Parameter dictionary for display in title
        kalman_mu: Kalman filter predictions
        kalman_sigma: Kalman filter uncertainties
        hidden_states: Hidden state values
        contexts: True context assignments
        contexts_prob: Inferred context probabilities, shape (N, T, n_ctx)
        contexts_pred: Predicted contexts, shape (N, T)
        preds_aligned: Whether model predictions align with obs x-axis
        shared_ylim: Whether to use pre-computed shared y-limits
        ylim: Pre-computed shared y-axis limits (required if shared_ylim=True)
        kal_x_start: Starting x-position for Kalman filter predictions
        data_config: Data configuration dictionary
        dpos_true: True deviant positions, shape (N, T)
        dpos_prob: Predicted deviant position probabilities, shape (N, T, N_dpos)
        rule_true: True rules, shape (N, T)
        rule_prob: Predicted rule probabilities, shape (N, T, N_rules)
    """
    # Decide whether to use context panel (with optional dpos and rule tracks)
    use_ctx_panel = n_ctx > 1 and (contexts is not None or contexts_prob is not None or contexts_pred is not None or dpos_true is not None or dpos_prob is not None or rule_true is not None or rule_prob is not None or cues is not None)
    
    # Context colors: use reversed Spectral colormap (0 -> min, 1 -> max)
    ctx_cmap = plt.get_cmap('tab10')
    ctx_colors = {i: ctx_cmap(i) for i in range(n_ctx)}
    
    # Get color mappings for dpos and rule
    dpos_colors = _get_dpos_color_map()
    rule_colors = _get_rule_color_map()
    
    # Get color mapping for cues (Accent colormap, starting from index 1)
    cue_colors = {}
    if cues is not None:
        # cues has shape (N, T, N_cues) - need to find which cue is active at each timestep
        n_cues = 2 # cues.shape[2]
        cue_cmap = plt.get_cmap('tab10')
        cue_colors = {i: cue_cmap((i+1)*5-1) for i in range(n_cues)}

    if use_ctx_panel:
        # Calculate number of rows needed in context panel
        n_context_rows = 0
        if contexts is not None or contexts_pred is not None:
            n_context_rows = 2  # ROW_TRUE, ROW_PRED only (no probabilities)
        n_cues_rows = 1 if cues is not None else 0
        n_dpos_rows = 2 if (dpos_true is not None or dpos_prob is not None) else 0
        n_rule_rows = 2 if (rule_true is not None or rule_prob is not None) else 0
        total_ctx_rows = n_cues_rows + n_context_rows + n_dpos_rows + n_rule_rows
        
        # Adjust height ratios based on content
        height_ratio_ctx = max(1, total_ctx_rows / 4) if total_ctx_rows > 0 else 1
        
        fig, (ax_obs, ax_ctx) = plt.subplots(
            2, 1, figsize=(20, 8),
            sharex=True,
            gridspec_kw={'height_ratios': [4, height_ratio_ctx], 'hspace': 0.05}
        )
    else:
        fig, ax_obs = plt.subplots(1, 1, figsize=(20, 6))
        ax_ctx = None

    # ── top subplot: observations + predictions ──────────────────────────
    ax_obs.plot(range(len(obs[i])), obs[i], color='tab:blue', label='y_obs', alpha=0.8)

    # When preds_aligned, predictions cover the same x-range as obs;
    # otherwise they start one step later (no prediction for the first obs).
    if preds_aligned:
        estim_x = range(len(obs[i]))
    else:
        estim_x = range(1, len(obs[i]))
    ax_obs.plot(estim_x, mu_estim[i], color='k', label='y_hat (model pred)')
    ax_obs.fill_between(estim_x, mu_estim[i]-sigma_estim[i], mu_estim[i]+sigma_estim[i], color='k', alpha=0.2, label='std (model)')

    if kalman_mu is not None and kalman_sigma is not None:
        kal_x = range(kal_x_start, kal_x_start + len(kalman_mu[i]))
        ax_obs.plot(kal_x, kalman_mu[i], label='y_kal (KF pred)', color='green', alpha=0.8)
        ax_obs.fill_between(kal_x, kalman_mu[i]-kalman_sigma[i], kalman_mu[i]+kalman_sigma[i], color='green', alpha=0.2, label='std (KF)')
    
    if hidden_states is not None:
        ax_obs.plot(range(len(hidden_states[i])), hidden_states[i, :], label='hidden state', color='orange', alpha=0.8)

    # context switch vertical lines on the obs panel
    if n_ctx > 1 and contexts is not None:
        context_changes = np.where(np.diff(contexts[i]) != 0)[0] + 1
        for cc in context_changes:
            ax_obs.axvline(x=cc, color='red', linestyle='--', alpha=0.5)

    if shared_ylim:
        ax_obs.set_ylim(ylim)
    if use_ctx_panel:
        ax_obs.tick_params(labelbottom=False)  # x labels only on bottom panel
    else:
        ax_obs.set_xlabel('time step')

    # ── bottom subplot: context lines (and optionally dpos, rule) ──────────────────
    if use_ctx_panel:
        # Row layout: cues (if available), rule (true/pred), dpos (true/pred), then context (true/pred)
        # Calculate base row numbers
        ROW_CUES = 0 if n_cues_rows > 0 else None
        
        # Then rule rows after cues
        row_offset = n_cues_rows
        ROW_RULE_TRUE = row_offset if n_rule_rows > 0 else None
        ROW_RULE_PRED = row_offset + 1 if n_rule_rows > 0 else None
        
        # Then dpos rows after rule
        row_offset += n_rule_rows
        ROW_DPOS_TRUE = row_offset if n_dpos_rows > 0 else None
        ROW_DPOS_PRED = row_offset + 1 if n_dpos_rows > 0 else None
        
        # Finally context rows after dpos
        row_offset += n_dpos_rows
        ROW_TRUE = row_offset if n_context_rows > 0 else None
        ROW_PRED = row_offset + 1 if n_context_rows > 0 else None

        # x-offset for prediction-aligned arrays:
        # When preds_aligned, preds have the same length as obs → offset 0
        # Otherwise they are 1 shorter → offset 1 (skip first obs timestep)
        pred_x_off = 0 if preds_aligned else 1
        
        # Plot cues track
        if cues is not None and ROW_CUES is not None:
            if cues.ndim > 2:
                # cues has shape (N, T, N_cues) - argmax to get active cue per timestep
                cue_indices = np.argmax(cues[i], axis=-1)
            else:
                # cues has shape (N, T) with integer cue indices
                cue_indices = cues[i]
            
            # Plot cues as continuous segments with changes
            boundaries = np.concatenate(([0], np.where(np.diff(cue_indices) != 0)[0] + 1, [len(cue_indices)]))
            cue_labels_placed = set()
            for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
                cue_val = int(cue_indices[seg_start])
                label = f'cue {cue_val}' if cue_val not in cue_labels_placed else None
                cue_labels_placed.add(cue_val)
                ax_ctx.hlines(ROW_CUES, seg_start, seg_end, colors=cue_colors[cue_val], linewidth=6, label=label)

        if contexts_pred is not None:
            pred_labels_placed = set()
            for t in range(len(contexts_pred[i])):
                ctx_val = int(contexts_pred[i][t])
                # Determine alpha from probability if available, otherwise use 0.8
                alpha = float(contexts_prob[i][t, ctx_val])
                label = f'ctx {ctx_val} pred' if ctx_val not in pred_labels_placed else None
                pred_labels_placed.add(ctx_val)
                ax_ctx.hlines(ROW_PRED, t + pred_x_off, t + pred_x_off + 1,
                              colors=ctx_colors[ctx_val],
                              linewidth=6, alpha=alpha, label=label)

        if contexts is not None:
            ctx_counts = {0: int((contexts[i] == 0).sum()), 1: int((contexts[i] == 1).sum())}
            boundaries = np.concatenate(([0], np.where(np.diff(contexts[i]) != 0)[0] + 1, [len(contexts[i])]))
            ctx_labels_placed = set()
            for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
                ctx_val = int(contexts[i][seg_start])
                label = f'ctx {ctx_val} (N={ctx_counts[ctx_val]})' if ctx_val not in ctx_labels_placed else None
                ctx_labels_placed.add(ctx_val)
                ax_ctx.hlines(ROW_TRUE, seg_start, seg_end, colors=ctx_colors[ctx_val], linewidth=6, label=label) # alpha=0.8, 

        # Plot dpos tracks
        if dpos_true is not None and dpos_pred is not None and dpos_prob is not None:
            dpos_pred_classes = dpos_pred[i]
            dpos_pred_probs = np.max(dpos_prob[i], axis=1)  # max probability for each timestep
            dpos_labels_placed = set()
            for t in range(len(dpos_pred_classes)):
                dpos_val = int(dpos_pred_classes[t])
                alpha = float(dpos_pred_probs[t])
                label = f'dpos {dpos_val} pred' if dpos_val not in dpos_labels_placed else None
                dpos_labels_placed.add(dpos_val)
                ax_ctx.hlines(ROW_DPOS_PRED, t + pred_x_off, t + pred_x_off + 1,
                              colors=dpos_colors[dpos_val], linewidth=6, alpha=alpha, label=label)
            
            true_dpos = dpos_true[i]
            boundaries = np.concatenate(([0], np.where(np.diff(true_dpos) != 0)[0] + 1, [len(true_dpos)]))
            dpos_labels_placed = set()
            for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
                dpos_val = int(true_dpos[seg_start])
                label = f'dpos {dpos_val}' if dpos_val not in dpos_labels_placed else None
                dpos_labels_placed.add(dpos_val)
                ax_ctx.hlines(ROW_DPOS_TRUE, seg_start, seg_end, colors=dpos_colors[dpos_val], 
                              linewidth=6, label=label) # alpha=0.8, 

        # Plot rule tracks
        if rule_true is not None and rule_pred is not None and rule_prob is not None:
            rule_pred_classes = rule_pred[i]
            rule_pred_probs = np.max(rule_prob[i], axis=1)  # max probability for each timestep
            rule_labels_placed = set()
            for t in range(len(rule_pred_classes)):
                rule_val = int(rule_pred_classes[t])
                alpha = float(rule_pred_probs[t])
                label = f'rule {rule_val} pred' if rule_val not in rule_labels_placed else None
                rule_labels_placed.add(rule_val)
                ax_ctx.hlines(ROW_RULE_PRED, t + pred_x_off, t + pred_x_off + 1,
                              colors=rule_colors[rule_val], linewidth=6, alpha=alpha, label=label)
            
            true_rule = rule_true[i]
            boundaries = np.concatenate(([0], np.where(np.diff(true_rule) != 0)[0] + 1, [len(true_rule)]))
            rule_labels_placed = set()
            for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
                rule_val = int(true_rule[seg_start])
                label = f'rule {rule_val}' if rule_val not in rule_labels_placed else None
                rule_labels_placed.add(rule_val)
                ax_ctx.hlines(ROW_RULE_TRUE, seg_start, seg_end, colors=rule_colors[rule_val], 
                              linewidth=6, label=label) # alpha=0.8, 

        # Adjust y-axis to fit all rows
        ytick_positions = [r for r in ([ROW_CUES] if ROW_CUES is not None else []) +
                                ([ROW_RULE_TRUE, ROW_RULE_PRED] if n_rule_rows > 0 else []) +
                                ([ROW_DPOS_TRUE, ROW_DPOS_PRED] if n_dpos_rows > 0 else []) +
                                ([ROW_TRUE, ROW_PRED] if n_context_rows > 0 else [])
                          if r is not None]
        yticklabels = (['cues'] if ROW_CUES is not None else []) +\
                      (['rule true', 'rule pred'] if n_rule_rows > 0 else []) +\
                      (['dpos true', 'dpos pred'] if n_dpos_rows > 0 else []) +\
                      (['ctx_true', 'ctx_pred'] if n_context_rows > 0 else [])
        
        ax_ctx.set_ylim(-0.5, total_ctx_rows - 0.5)
        ax_ctx.set_yticks(ytick_positions)
        ax_ctx.set_yticklabels(yticklabels, fontsize=8)
        ax_ctx.invert_yaxis()  # Invert to show cues at the top
        ax_ctx.set_xlabel('time step')

    # ── shared legend on the right (only selected labels) ────────────────
    # Collect labels to show: y_obs, y_hat, sigma, y_kal, and "ctx N (N=...)" from the ctx panel
    _LEGEND_KEEP = {'y_obs', 'y_hat (model pred)', 'std (model)', 'y_kal (KF pred)', 'std (KF)'}
    handles, labels = [], []
    for ax in ([ax_obs, ax_ctx] if use_ctx_panel else [ax_obs]):
        if ax is None:
            continue
        h, l = ax.get_legend_handles_labels()
        for handle, lbl in zip(h, l):
            if lbl not in labels and (lbl in _LEGEND_KEEP or (lbl is not None and lbl.startswith('ctx') and '(N=' in lbl)):
                handles.append(handle)
                labels.append(lbl)
    # Add legend patches for all possible dpos and rule classes
    for dpos_val in sorted(dpos_colors.keys()):
        dpos_patch = mpatches.Patch(color=dpos_colors[dpos_val], label=f'dpos {dpos_val}')
        if dpos_patch.get_label() not in labels:
            handles.append(dpos_patch)
            labels.append(dpos_patch.get_label())
    
    for rule_val in sorted(rule_colors.keys()):
        rule_patch = mpatches.Patch(color=rule_colors[rule_val], label=f'rule {rule_val}')
        if rule_patch.get_label() not in labels:
            handles.append(rule_patch)
            labels.append(rule_patch.get_label())
    
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.92, 0.5), borderaxespad=0, frameon=True)
    plot_title = title
    if params is not None:
        # Extract parameters for sample i, handling both dict and legacy array formats
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

    ax_obs.set_title(plot_title)
    fig.savefig(f'{save_path}_s{id}.png', bbox_inches='tight')
    plt.close(fig)


# def plot_samples(obs, mu_estim, sigma_estim, save_path, title=None, params=None, kalman_mu=None, kalman_sigma=None, seq_start=None, seq_end=None, data_config=None, hidden_states=None, contexts=None, min_obs_for_em=None, N_plots=8, shared_ylim=False, contexts_prob=None, contexts_pred=None, dpos_true=None, dpos_pred=None, rule_true=None, rule_pred=None):
def plot_samples(sample_metrics, save_path, title=None, params=None, seq_start=None, seq_end=None, data_config=None, min_obs_for_em=None, N_plots=8, shared_ylim=False):
    """
    Plot observation sequences with model and optional Kalman filter predictions.
    
    Alignment:
    - obs has shape (N, T)
    - mu_estim has shape (N, T-1): mu_estim[:, k] predicts obs[:, k+1], so their ends are aligned
    - kalman_mu has shape (N, T-MIN_OBS_FOR_EM): predicts obs[:, MIN_OBS_FOR_EM:], ends also aligned
    - contexts_pred / contexts_prob have shape (N, T-1): aligned with mu_estim
    - contexts has shape (N, T): aligned with obs

    Windowing via seq_start / seq_end (both refer to obs indices, negative = from end):
    - obs is sliced as obs[:, seq_start:seq_end]
    - all other arrays are sliced from their END by the same number of obs steps (since ends are aligned)
    
    Args:
        seq_start: Start obs index for the window (default None = beginning). Use negative values
                   (e.g. -100) to take the last N observations.
        seq_end:   End obs index for the window (default None = end of sequence).
        min_obs_for_em: Minimum observations needed for KF EM (default: MIN_OBS_FOR_EM constant)
        shared_ylim: If True, use the same y-axis limits across all plotted samples (default: False)
    """
    # Extract variables from sample_metrics
    obs = sample_metrics['y']
    mu_estim = sample_metrics['mu_estim']
    sigma_estim = sample_metrics['sigma_estim']
    kalman_mu = sample_metrics['kalman_mu']         if 'kalman_mu' in sample_metrics else None
    kalman_sigma = sample_metrics['kalman_sigma']   if 'kalman_sigma' in sample_metrics else None
    hidden_states = sample_metrics['hidden_states'] if 'hidden_states' in sample_metrics else None
    contexts = sample_metrics['contexts']           if 'contexts' in sample_metrics else None
    contexts_prob = sample_metrics['ctx_prob'] if 'ctx_prob' in sample_metrics else None
    contexts_pred = sample_metrics['ctx_pred'] if 'ctx_pred' in sample_metrics else None
    dpos_true = sample_metrics['dpos_true'] if 'dpos_true' in sample_metrics else None
    dpos_prob = sample_metrics['dpos_prob'] if 'dpos_prob' in sample_metrics else None
    dpos_pred = sample_metrics['dpos_pred'] if 'dpos_pred' in sample_metrics else None
    rule_true = sample_metrics['rule_true'] if 'rule_true' in sample_metrics else None
    rule_prob = sample_metrics['rule_prob'] if 'rule_prob' in sample_metrics else None
    rule_pred = sample_metrics['rule_pred'] if 'rule_pred' in sample_metrics else None
    cues = sample_metrics['cues'] if 'cues' in sample_metrics else None

    # Set default for min_obs_for_em
    if min_obs_for_em is None:
        min_obs_for_em = MIN_OBS_FOR_EM
    
    # Convert to numpy if it's a tensor
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    obs = obs.squeeze() # for safety
    
    # Get N_ctx from data_config
    n_ctx = data_config["N_ctx"]

    # Define plotting passes: for HierarchicalGM with long sequences, plot both truncated and full
    passes = []
    if data_config is not None and data_config["gm_name"] == "HierarchicalGM":
        N_tones = data_config["N_tones"]
        last_8_blocks_len = 8 * N_tones
        if obs.shape[1] > last_8_blocks_len:
            passes = [
                {'seq_start': -last_8_blocks_len, 'seq_end': None, 'suffix': '_last8blocks'},
                {'seq_start': None, 'seq_end': None, 'suffix': '_full'}
            ]
        else:
            passes = [{'seq_start': seq_start, 'seq_end': seq_end, 'suffix': ''}]
    else:
        passes = [{'seq_start': seq_start, 'seq_end': seq_end, 'suffix': ''}]
    
    # Save original arrays (will be restored for each pass)
    obs_orig = obs.copy()
    mu_estim_orig = mu_estim.copy()
    sigma_estim_orig = sigma_estim.copy()
    kalman_mu_orig = kalman_mu.copy() if kalman_mu is not None else None
    kalman_sigma_orig = kalman_sigma.copy() if kalman_sigma is not None else None
    hidden_states_orig = hidden_states.copy() if hidden_states is not None else None
    contexts_orig = contexts.copy() if contexts is not None else None
    contexts_prob_orig = contexts_prob.copy() if contexts_prob is not None else None
    contexts_pred_orig = contexts_pred.copy() if contexts_pred is not None else None
    dpos_true_orig = dpos_true.copy() if dpos_true is not None else None
    dpos_prob_orig = dpos_prob.copy() if dpos_prob is not None else None
    dpos_pred_orig = dpos_pred.copy() if dpos_pred is not None else None
    rule_true_orig = rule_true.copy() if rule_true is not None else None
    rule_prob_orig = rule_prob.copy() if rule_prob is not None else None
    rule_pred_orig = rule_pred.copy() if rule_pred is not None else None
    cues_orig = cues.copy() if cues is not None else None

    # For each pass (normally 1, or 2 for HierarchicalGM with long sequences)
    for pass_info in passes:
        # Restore arrays from originals for this pass
        obs = obs_orig.copy()
        mu_estim = mu_estim_orig.copy()
        sigma_estim = sigma_estim_orig.copy()
        kalman_mu = kalman_mu_orig.copy() if kalman_mu_orig is not None else None
        kalman_sigma = kalman_sigma_orig.copy() if kalman_sigma_orig is not None else None
        hidden_states = hidden_states_orig.copy() if hidden_states_orig is not None else None
        contexts = contexts_orig.copy() if contexts_orig is not None else None
        contexts_prob = contexts_prob_orig.copy() if contexts_prob_orig is not None else None
        contexts_pred = contexts_pred_orig.copy() if contexts_pred_orig is not None else None
        dpos_true = dpos_true_orig.copy() if dpos_true_orig is not None else None
        dpos_prob = dpos_prob_orig.copy() if dpos_prob_orig is not None else None
        dpos_pred = dpos_pred_orig.copy() if dpos_pred_orig is not None else None
        rule_true = rule_true_orig.copy() if rule_true_orig is not None else None
        rule_prob = rule_prob_orig.copy() if rule_prob_orig is not None else None
        rule_pred = rule_pred_orig.copy() if rule_pred_orig is not None else None
        cues = cues_orig.copy() if cues_orig is not None else None
        
        # Get parameters for this pass
        pass_seq_start = pass_info['seq_start']
        pass_seq_end = pass_info['seq_end']
        pass_suffix = pass_info['suffix']
        pass_save_path = str(save_path) + pass_suffix

        # ── Windowing ──────────────────────────────────────────────────────────────
        # Resolve start/end to concrete obs indices so we can compute how many obs steps
        # the window covers and slice all other arrays consistently from their ends.
        #
        # Alignment note:
        #   mu_estim[k] predicts obs[k+1] (shape T-1). When the window starts past the
        #   very first obs (start_idx > 0), we can include mu_estim[start_idx-1] which
        #   predicts obs[start_idx], giving predictions the SAME length as the obs window.
        #   In that case preds_aligned=True and predictions are plotted at x = 0..W-1
        #   just like obs. When start_idx == 0 the first obs has no prediction, so
        #   preds_aligned=False and predictions are plotted at x = 1..W-1 (one fewer).
        preds_aligned = False  # default: predictions are 1 shorter than obs

        if pass_seq_start is not None or pass_seq_end is not None:
            T_obs = obs.shape[1]
            # Resolve to non-negative indices
            start_idx = (pass_seq_start % T_obs) if pass_seq_start is not None else 0
            end_idx   = (pass_seq_end   % T_obs) if pass_seq_end   is not None else T_obs

            # === Slice observation-aligned arrays (shape T) ===
            obs = obs[:, start_idx:end_idx]
            if contexts is not None:
                contexts = contexts[:, start_idx:end_idx]
            if dpos_true is not None:
                dpos_true = dpos_true[:, start_idx:end_idx]
            if rule_true is not None:
                rule_true = rule_true[:, start_idx:end_idx]
            if hidden_states is not None:
                hidden_states = hidden_states[:, start_idx:end_idx, :]
            if cues is not None:
                cues = cues[:, start_idx:end_idx]

            # === Slice prediction-aligned arrays (shape T-1) ===
            # When start_idx > 0, we can include mu_estim[start_idx-1] which predicts obs[start_idx],
            # so predictions have the same length as the windowed obs. Otherwise they're 1 shorter.
            preds_aligned = start_idx > 0
            
            if preds_aligned:
                # prediction arrays: slice [start_idx-1:end_idx-1]
                pred_slice_start = start_idx - 1
                pred_slice_end = end_idx - 1
            else:
                # prediction arrays: slice [:end_idx-1]
                pred_slice_start = 0
                pred_slice_end = end_idx - 1
            
            # Apply the same slice to all prediction-aligned arrays
            mu_estim = mu_estim[:, pred_slice_start:pred_slice_end]
            sigma_estim = sigma_estim[:, pred_slice_start:pred_slice_end]
            if dpos_prob is not None:
                dpos_prob = dpos_prob[:, pred_slice_start:pred_slice_end]
            if dpos_pred is not None:
                dpos_pred = dpos_pred[:, pred_slice_start:pred_slice_end]
            if rule_prob is not None:
                rule_prob = rule_prob[:, pred_slice_start:pred_slice_end]
            if rule_pred is not None:
                rule_pred = rule_pred[:, pred_slice_start:pred_slice_end]
            if contexts_pred is not None:
                contexts_pred = contexts_pred[:, pred_slice_start:pred_slice_end]
            if contexts_prob is not None:
                contexts_prob = contexts_prob[:, pred_slice_start:pred_slice_end]

            # === Slice Kalman filter predictions ===
            # KF predicts obs[min_obs_for_em:], so within window obs[start_idx:end_idx],
            # KF covers obs[max(start_idx, min_obs_for_em):end_idx]
            if kalman_mu is not None or kalman_sigma is not None:
                kal_obs_start = max(start_idx, min_obs_for_em)
                kal_slice_start = kal_obs_start - min_obs_for_em
                kal_slice_end = end_idx - min_obs_for_em
                if kalman_mu is not None:
                    kalman_mu = kalman_mu[:, kal_slice_start:kal_slice_end]
                if kalman_sigma is not None:
                    kalman_sigma = kalman_sigma[:, kal_slice_start:kal_slice_end]
                kal_x_start = kal_obs_start - start_idx
            else:
                kal_x_start = min_obs_for_em
        else:
            preds_aligned = False
            kal_x_start = min_obs_for_em

        # for some N randomly sampled sequences out of the whole batch
        N = min(N_plots, obs.shape[0])
        selected_indices = np.random.choice(obs.shape[0], size=N, replace=False)
        # Debug prints removed to avoid index errors when plotting single samples

        # Compute shared y-axis limits across all selected samples if requested
        if shared_ylim:
            all_values = [obs[i] for i in selected_indices]
            all_values += [mu_estim[i] for i in selected_indices]
            all_values += [mu_estim[i] + sigma_estim[i] for i in selected_indices]
            all_values += [mu_estim[i] - sigma_estim[i] for i in selected_indices]
            if kalman_mu is not None:
                all_values += [kalman_mu[i] for i in selected_indices]
            if kalman_sigma is not None and kalman_mu is not None:
                all_values += [kalman_mu[i] + kalman_sigma[i] for i in selected_indices]
                all_values += [kalman_mu[i] - kalman_sigma[i] for i in selected_indices]
            global_ymin = min(np.min(v) for v in all_values)
            global_ymax = max(np.max(v) for v in all_values)
            y_margin = 0.05 * (global_ymax - global_ymin)
            ylim = (global_ymin - y_margin, global_ymax + y_margin)

        # Plot each selected sample using plot_sample
        for id, i in enumerate(selected_indices):
            plot_sample(
                id=id,
                i=i,
                obs=obs,
                mu_estim=mu_estim,
                sigma_estim=sigma_estim,
                save_path=pass_save_path,
                n_ctx=n_ctx,
                title=title,
                params=params,
                kalman_mu=kalman_mu,
                kalman_sigma=kalman_sigma,
                hidden_states=hidden_states,
                contexts=contexts,
                contexts_prob=contexts_prob,
                contexts_pred=contexts_pred,
                preds_aligned=preds_aligned,
                shared_ylim=shared_ylim,
                ylim=ylim if shared_ylim else None,
                kal_x_start=kal_x_start,
                dpos_true=dpos_true,
                dpos_prob=dpos_prob,
                dpos_pred=dpos_pred,
                rule_true=rule_true,
                rule_prob=rule_prob,
                rule_pred=rule_pred,
                cues=cues
            )



# =============================================================================
# xRNNx Model Factory
# =============================================================================

def create_model(config: RunConfig) -> nn.Module:
    """
    Create a model instance from a RunConfig.
    """
    model_dict = run_config_to_model_dict(config)
    
    if config.model_type == 'rnn':
        model = SimpleRNN(model_dict)
        
    elif config.model_type == 'vrnn':
        model = VRNN(model_dict)
        
    elif config.model_type == 'module_network':
        # ModuleNetwork needs the full config with module sub-configs
        module_config = _create_module_network_config(config)
        model = ObsCtxModuleNetwork(module_config)
    
    elif config.model_type == 'population_network':
        # PopulationNetwork needs the full config with module sub-configs
        module_config = _create_population_network_config(config)
        model = PopulationNetwork(module_config)
        
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    return model.to(config.device)


def _create_module_network_config(config: RunConfig) -> dict:
    """
    Build the specialized config dict for ModuleNetwork (obs + ctx modules).
    
    Reads per-module hidden dimensions from config.module_hidden_dims,
    which is a dict like {'obs': 64, 'ctx': 32}.
    Falls back to defaults if not provided.
    
    ModuleNetwork typically uses just observation and context modules.
    """
    # Default module dimensions for ModuleNetwork
    default_dims = {'obs': 64, 'ctx': 32}
    module_dims = config.module_hidden_dims or {}
    # Ensure we have at least the defaults for obs and ctx
    module_dims = {**default_dims, **module_dims}
    
    module_config = {
        'kappa': config.kappa,
    }
    
    # Build observation module config (always present for ModuleNetwork)
    if 'obs' in module_dims:
        module_config['observation_module'] = {
            'input_dim': config.model_arch.input_dim,
            'output_dim': config.model_arch.output_dim,
            'rnn_hidden_dim': module_dims['obs'],
            'rnn_n_layers': config.model_arch.rnn_n_layers,
            'bottleneck_dim': config.bottleneck_dim or 16,
        }
    
    # Build context module config (always present for ModuleNetwork)
    if 'ctx' in module_dims:
        module_config['context_module'] = {
            'input_dim': 2, # NOTE: one for each context? (TODO: verify)
            'output_dim': config.data.N_ctx,
            'rnn_hidden_dim': module_dims['ctx'],
            'rnn_n_layers': config.model_arch.rnn_n_layers,
            'bottleneck_dim': config.bottleneck_dim or 16,
        }
    
    return module_config


def _create_population_network_config(config: RunConfig) -> dict:
    """
    Build the specialized config dict for PopulationNetwork (obs + ctx + dpos + rule modules).
    
    Reads per-module hidden dimensions from config.module_hidden_dims,
    which is a dict like {'obs': 64, 'ctx': 32, 'dpos': 16, 'rule': 8}.
    Falls back to defaults if not provided.
    
    PopulationNetwork uses observation, context, dpos (for HierarchicalGM), 
    and rule modules for richer multi-module architecture.
    """
    # Default module dimensions for PopulationNetwork
    default_dims = {'obs': 64, 'ctx': 32, 'dpos': 16, 'rule': 8} # TOOD: modify this!
    module_dims = config.module_hidden_dims or {}
    # Merge with defaults, preserving user-provided values
    module_dims = {**default_dims, **module_dims}
    
    module_config = {
        'kappa': config.kappa,
    }
    
    # Build rule module config (processes visual cues q)
    # rule receives q (1D) in forward pass and enc_dpos2rule (2D) in feedback pass
    # So input_dim = 1, output_dim = N_rules (asymmetric)
    module_config['rule_module'] = {
        'input_dim': config.data.N_cues,  # Matches q dimension
        'output_dim': config.data.N_rules,
        'output_ext_dim': config.data.N_rules,
        'rnn_hidden_dim': module_dims['rule'],
        'rnn_n_layers': config.model_arch.rnn_n_layers,
        'bottleneck_dim': config.bottleneck_dim,
    }
    
    # Build dpos module config (deviant position identifier)
    # dpos receives enc_rule2dpos (N_dpos) in forward pass and enc_ctx2dpos (N_dpos) in feedback pass
    # So input_dim = output_dim = N_dpos (symmetric)
    module_config['dpos_module'] = {
        'input_dim': config.data.N_dpos,
        'output_dim': config.data.N_dpos,
        'output_ext_dim': config.data.N_dpos,
        'rnn_hidden_dim': module_dims['dpos'],
        'rnn_n_layers': config.model_arch.rnn_n_layers,
        'bottleneck_dim': config.bottleneck_dim,
    }
    
    # Build context module config (context identifier)
    # ctx receives enc_dpos2ctx (N_ctx) in forward pass and enc_obs2ctx (N_ctx) in feedback pass
    # So input_dim = output_dim = N_ctx (symmetric)
    module_config['context_module'] = {
        'input_dim': config.data.N_ctx,
        'output_dim': config.data.N_ctx,
        'output_ext_dim': config.data.N_ctx,
        'rnn_hidden_dim': module_dims['ctx'],
        'rnn_n_layers': config.model_arch.rnn_n_layers,
        'bottleneck_dim': config.bottleneck_dim,
    }
    
    # Build observation module config (pitch/observation estimator)
    # obs receives x (input_dim) in second forward pass and enc_ctx2obs (input_dim) in forward step 4
    # So input_dim = model_arch.input_dim, output_dim = 2 (asymmetric)
    module_config['observation_module'] = {
        'input_dim': config.model_arch.input_dim,  # 1D
        'output_dim': config.model_arch.output_dim,  # 2D (mu, sigma)
        'output_ext_dim': config.model_arch.output_dim,
        'rnn_hidden_dim': module_dims['obs'],
        'rnn_n_layers': config.model_arch.rnn_n_layers,
        'bottleneck_dim': config.bottleneck_dim,
    }
    
    return module_config


# =============================================================================
# Generative Model Factory
# =============================================================================

def create_generative_model(config: RunConfig, batch_size: Optional[int] = None):
    """Create the appropriate generative model for data generation."""
    gm_dict = config.data.to_gm_dict(batch_size or config.training.batch_size)
    
    if config.data.gm_name == 'NonHierarchicalGM':
        return NonHierarchicalAuditGM(gm_dict)
    elif config.data.gm_name == 'HierarchicalGM':
        return HierarchicalAuditGM(gm_dict)
    else:
        raise ValueError(f"Unknown GM name: {config.data.gm_name}")

# =============================================================================
# Training Loop
# =============================================================================

def _compute_forward_and_loss(
    model: nn.Module,
    batch_data: Dict,
    config: RunConfig,
    objective: Objective,
    data_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shared forward pass and loss computation for both training and validation.
    Ensures consistent logic across both training (grad on) and validation (grad off) phases.
    
    Args:
        model: The model to run forward pass on
        batch_data: Dictionary with 'y', 'contexts', and optionally 'q', 'dpos', 'rules'
        config: Run configuration
        objective: Loss objective with loss functions
        data_mode: Data mode string ('single_ctx' or 'multi_ctx')
    
    Returns:
        Tuple of (model_output, loss_tensor)
    """
    y = batch_data['y']
    contexts = batch_data['contexts']
    
    # Forward pass and loss computation (single source of truth)
    if config.data.gm_name == 'HierarchicalGM':
        q = batch_data['q']
        dpos = batch_data['dpos']
        rules = batch_data['rules']
        model_output = model(y[:, :-1, :], q[:, :-1, :])
        loss = compute_model_loss(
            model, objective, y, model_output, data_mode,
            learning_objective=config.learning_objective,
            contexts_tensor=contexts,
            dpos_tensor=dpos,
            rules_tensor=rules
        )
    else:
        model_output = model(y[:, :-1, :])
        loss = compute_model_loss(
            model, objective, y, model_output, data_mode,
            learning_objective=config.learning_objective,
            contexts_tensor=contexts,
            kappa=config.kappa
        )
    
    return model_output, loss


def train_model(
    model: nn.Module,
    config: RunConfig,
    benchmarks: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Train a model for one configuration.
    
    This is the clean, linear training loop:
    1. Setup (optimizer, loss, data generator)
    2. Epoch loop
       - Batch loop (train)
       - Validation step
       - Logging
    3. Save model and plots
    
    Args:
        model: The model to train (already on correct device)
        config: Complete run configuration
        benchmarks: Optional benchmark data for validation comparison
    
    Returns:
        Dict with training history (losses, metrics, etc.)
    """
    # ========== SETUP ==========
    device = config.device
    save_path = config.save_dir
    os.makedirs(save_path / 'samples', exist_ok=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Loss function
    loss_fn_obs = torch.nn.GaussianNLLLoss(reduction='mean')
    if config.data.data_mode == 'multi_ctx' and config.learning_objective in ['ctx', 'obs_ctx', 'all']:
        loss_fn_ctx = torch.nn.CrossEntropyLoss(reduction='mean')
    else:
        loss_fn_ctx = None
    objective = Objective(loss_fn_obs, loss_func_ctx=loss_fn_ctx)
    
    # Data generator
    gm = create_generative_model(config)
    data_mode = config.data.data_mode
    
    # Training config shortcuts
    num_epochs = config.training.num_epochs
    n_batches = config.training.n_batches
    batch_res = config.training.batch_res
    epoch_res = config.training.epoch_res
    
    # History tracking
    history = {
        'train_losses': [],
        'valid_losses': [],
        'train_steps': [],
        'valid_steps': [],
        'valid_mse': [],
        'valid_sigma': [],
        'model_kf_mse': [] if benchmarks else None,
        'weights_updates': [],
    }
    param_names = list(model.state_dict().keys())
    
    # Logging setup
    if config.hidden_dim is not None:
        lr_title = f"Model: {config.model_type} | LR: {config.learning_rate:>6.0e} | hidden_dim: {config.hidden_dim}"
    else:
        dims_str = ', '.join(f"{k}={v}" for k, v in sorted(config.module_hidden_dims.items())) if config.module_hidden_dims else 'default'
        lr_title = f"Model: {config.model_type} | LR: {config.learning_rate:>6.0e} | modules: {dims_str}"
    
    print(f"\n{'='*60}")
    print(f"TRAINING: {config.name}")
    print(f"{'='*60}")
    
    # ========== TRAINING LOOP ==========
    for epoch in tqdm(range(num_epochs), desc="Epochs", leave=False):
        epoch_losses = []
        t_start = time.time()
        model.train()
        
        # ----- BATCH LOOP -----
        for batch_idx in range(n_batches):
            optimizer.zero_grad()
            
            # Generate batch
            batch_data = prepare_batch_data(gm, config.data.gm_name, data_mode, device)
            
            # Track weights before update (for logging)
            if batch_idx % batch_res == batch_res - 1:
                weights_before = [w.detach().clone() for w in model.parameters()]
            
            # Forward pass and compute loss (shared logic)
            model_output, loss = _compute_forward_and_loss(
                model, batch_data, config, objective, data_mode
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track weights after update
            if batch_idx % batch_res == batch_res - 1:
                weights_after = [w.detach().clone() for w in model.parameters()]
                weights_update = torch.stack([
                    torch.mean((wa - wb)**2) 
                    for wb, wa in zip(weights_before, weights_after)
                ])
                history['weights_updates'].append(weights_update)
            
            epoch_losses.append(loss.item())
            
            # Batch logging
            if batch_idx % batch_res == batch_res - 1:
                history['train_steps'].append(epoch * n_batches + batch_idx)
                history['train_losses'].append(loss.item())
                
                _log_batch(
                    save_path, config.lr_id, config.learning_rate,
                    epoch, batch_idx, loss.item(),
                    np.mean(epoch_losses[-batch_res:]),
                    time.time() - t_start,
                    epoch * n_batches + batch_idx
                )
                t_start = time.time()
        
        # ----- VALIDATION -----
        model.eval()
        with torch.no_grad():
            valid_metrics = _validate_epoch(
                model, config, benchmarks, gm, objective
            )
        
        # Store validation metrics
        history['valid_losses'].append(valid_metrics['loss'])
        history['valid_mse'].append(valid_metrics['mse'])
        history['valid_sigma'].append(valid_metrics['sigma_mean'])
        history['valid_steps'].append((epoch + 1) * n_batches)
        
        if benchmarks and 'mse_model2kal' in valid_metrics:
            history['model_kf_mse'].append(valid_metrics['mse_model2kal'])
        
        # Save sample plots periodically
        if epoch % epoch_res == epoch_res - 1:
            _save_validation_samples(
                valid_metrics, config, epoch, lr_title, benchmarks
            )
        
        # Epoch logging
        _log_epoch(
            save_path, config.lr_id, config.learning_rate,
            epoch, valid_metrics, time.time() - t_start,
            (epoch + 1) * n_batches, benchmarks is not None
        )
    
    # ========== SAVE RESULTS ==========
    print(f"Final - Train Loss: {np.mean(epoch_losses):.4f}, Valid Loss: {valid_metrics['loss']:.4f}")
    
    # Save model weights
    torch.save(model.state_dict(), save_path / f'lr{config.lr_id}_weights.pth')
    
    # Save config for reproducibility and easy loading
    config_path = config.save()
    print(f"Config saved to: {config_path}")
    
    # Save plots
    _save_training_plots(history, config, lr_title, benchmarks)
    
    return history


def _validate_epoch(
    model: nn.Module,
    config: RunConfig,
    benchmarks: Optional[Dict],
    gm,
    objective: Objective,
) -> Dict[str, Any]:
    """Run validation for one epoch and return metrics."""
    device = config.device
    data_mode = config.data.data_mode
    
    # Get validation data
    if benchmarks:
        bench_data = prepare_benchmark_data(benchmarks, data_mode, device)
        y = bench_data['y']
        contexts = bench_data['contexts']
        mu_kal = bench_data['mu_kal_pred']
        pars = bench_data['pars']
        min_obs = bench_data['min_obs_for_em']
    else:
        batch_data = prepare_batch_data(gm, config.data.gm_name, data_mode, device, return_pars=True)
        y = batch_data['y']
        contexts = batch_data['contexts']
        pars = batch_data['pars']
        mu_kal = None
        min_obs = None
    
    # Forward pass and compute loss (shared logic)
    model_output, loss = _compute_forward_and_loss(
        model, batch_data, config, objective, data_mode
    )
    loss = loss.item()
    
    # Extract all predictions from model (already processed: softmax, argmax, sigma computed)
    predictions = get_model_predictions(model, model_output)
    mu_estim = predictions['mu_estim']
    sigma_estim = predictions['sigma_estim']
    
    # Compute MSE
    y_np = y.detach().cpu().numpy().squeeze()
    mse = ((mu_estim - y_np[:, 1:])**2).mean()

    # Initialize metrics with core measurements
    metrics = {
        'loss': loss,
        'mse': mse,
        'sigma_mean': sigma_estim.mean(),
        'y': y_np,
        'pars': pars
    }
    
    # Add all model predictions (obs mu_estim and sigma_estim, contexts, dpos, rule probabilities and predictions)
    metrics.update(predictions)
    
    # Process true labels from batch data
    contexts_np = None
    if contexts is not None and predictions['ctx_prob'] is not None:
        contexts_np = contexts.detach().cpu().numpy().squeeze()

    # Extract dpos and rule true labels (for HierarchicalGM)
    dpos_true = None
    rule_true = None
    if 'dpos' in batch_data and batch_data['dpos'] is not None:
        dpos_true = batch_data['dpos'].detach().cpu().numpy().squeeze()  # shape (N, T)
    if 'rules' in batch_data and batch_data['rules'] is not None:
        rule_true = batch_data['rules'].detach().cpu().numpy().squeeze()  # shape (N, T)

    # Extract cues (for HierarchicalGM)
    cues = None
    if 'q' in batch_data and batch_data['q'] is not None:
        cues = batch_data['q'].detach().cpu().numpy().squeeze()  # shape (N, T, N_cues)

    # Add true labels to metrics
    metrics.update({
        'contexts': contexts_np,
        'dpos_true': dpos_true,
        'rule_true': rule_true,
        'cues': cues,
    })
    
    # Compare with Kalman filter if available
    if mu_kal is not None:
        mu_estim_aligned = mu_estim[:, min_obs - 1:]
        mse_model2kal = ((mu_estim_aligned - mu_kal)**2).mean()
        metrics['mse_model2kal'] = mse_model2kal
        metrics['mu_kal'] = mu_kal
        metrics['min_obs'] = min_obs
    
    return metrics


# =============================================================================
# Testing
# =============================================================================

def test_model(
    model: nn.Module,
    config: RunConfig,
    benchmarks: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Test a trained model on held-out data.
    
    Args:
        model: The model to test
        config: Complete run configuration  
        benchmarks: Optional benchmark data for comparison
    
    Returns:
        Dict with test metrics
    """
    print(f"\n{'='*60}")
    print(f"TESTING: {config.name}")
    print(f"{'='*60}")
    
    save_path = config.save_dir
    device = config.device
    
    # Check model exists
    weights_path = save_path / f'lr{config.lr_id}_weights.pth'
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    # Load model
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()
    
    # Get test data
    data_mode = config.data.data_mode
    if benchmarks:
        bench_data = prepare_benchmark_data(benchmarks, data_mode, device)
        y = bench_data['y']
        contexts = bench_data['contexts']
        mu_kal = bench_data['mu_kal_pred']
        pars = bench_data['pars']
        mse_kal = bench_data['mse_kal']
        min_obs = bench_data['min_obs_for_em']
    else:
        gm = create_generative_model(config, config.training.batch_size_test)
        batch_data = prepare_batch_data(gm, config.data.gm_name, data_mode, device, return_pars=True)
        y = batch_data['y']
        contexts = batch_data['contexts']
        pars = batch_data['pars']
        mu_kal = None
        min_obs = None
    
    # Forward pass
    with torch.no_grad():
        model_output = model(y[:, :-1, :])
        predictions = get_model_predictions(model, model_output)
        mu_estim = predictions['mu_estim']
        sigma_estim = predictions['sigma_estim']
    
    # Compute metrics
    y_np = y.detach().cpu().numpy().squeeze()
    mse = ((mu_estim - y_np[:, 1:])**2).mean()
    
    results = {
        'mse': mse,
        'sigma': sigma_estim.mean(),
    }
    
    if mu_kal is not None:
        mu_estim_aligned = mu_estim[:, min_obs - 1:]
        results['mse_model2kal'] = ((mu_estim_aligned - mu_kal)**2).mean()
        results['mse_kal'] = mse_kal.mean() if hasattr(mse_kal, 'mean') else mse_kal
    
    # Save binned metrics if parameter testing enabled
    if config.data.params_testing:
        param_bins = bin_params(config.data.to_gm_dict(config.training.batch_size))
        if mu_kal is not None:
            mu_estim_aligned = mu_estim[:, min_obs - 1:]
            y_aligned = y_np[:, min_obs:]
            binned_df = map_binned_params_2_metrics(param_bins, y_aligned, mu_estim_aligned, pars, mu_kal=mu_kal)
        else:
            binned_df = map_binned_params_2_metrics(param_bins, y_np[:, 1:], mu_estim, pars)
        binned_df.to_csv(save_path / f'test_binned_metrics_lr{config.lr_id}.csv', index=False)
    
    print(f"Test MSE: {results['mse']:.4f}")
    if 'mse_model2kal' in results:
        print(f"Model-KF MSE: {results['mse_model2kal']:.4f}")
    
    return results


# =============================================================================
# Logging Helpers
# =============================================================================

def _log_batch(save_path, lr_id, lr, epoch, batch, loss, batch_loss, elapsed, step):
    """Log batch-level training info."""
    msg = f'LR: {lr:>6.0e}; epoch: {epoch:0>3}; batch: {batch:>3}; loss: {loss:>7.4f}; batch loss: {batch_loss:7.4f}; time: {elapsed:>.2f}; step: {step}'
    with open(save_path / f'training_loss_lr{lr_id}.txt', 'a') as f:
        f.write(f'{msg}\n')


def _log_epoch(save_path, lr_id, lr, epoch, metrics, elapsed, step, has_benchmarks):
    """Log epoch-level validation info."""
    msg = f"LR: {lr:>6.0e}; epoch: {epoch:>3}; mean var: {metrics['sigma_mean']:>7.2f}; mean MSE: {metrics['mse']:>7.2f}; time: {elapsed:>.2f}; step: {step}"
    if has_benchmarks and 'mse_model2kal' in metrics:
        msg += f"; Model-KF MSE: {metrics['mse_model2kal']:>7.2f}"
    with open(save_path / f'training_log_lr{lr_id}.txt', 'a') as f:
        f.write(f'{msg}\n')


def _save_validation_samples(metrics, config, epoch, title, benchmarks):
    """Save validation sample plots."""
    save_path = config.save_dir / 'samples' / f'lr{config.lr_id}-epoch-{epoch:0>3}_samples'
    
    kwargs = {
        'params': metrics['pars'],
        'title': title,
        'seq_start': -config.seq_len_viz if config.seq_len_viz is not None else None,
        'N_plots': 4,
        'data_config': config.data.to_gm_dict(config.training.batch_size),
    }
    
    
    plot_samples(metrics, save_path, **kwargs)


def _save_training_plots(history, config, title, benchmarks):
    """Save end-of-training plots."""
    save_path = config.save_dir
    lr_id = config.lr_id
    
    # Loss curves
    plot_losses(
        history['train_steps'], history['valid_steps'],
        history['train_losses'], history['valid_losses'],
        'Training steps', 'Loss', title,
        save_path / f'loss_trainvalid_lr{lr_id}.png'
    )
    
    # Variance over epochs
    epoch_steps = list(range(len(history['valid_sigma'])))
    plot_variance(epoch_steps, history['valid_sigma'], title, save_path / f'variance_valid_lr{lr_id}.png')
    
    # MSE over epochs
    if benchmarks and history['model_kf_mse']:
        mse_kal = benchmarks['perf']
        if hasattr(mse_kal, 'mean'):
            mse_kal = mse_kal.mean()
        plot_mse(epoch_steps, history['valid_mse'], title, 
                 save_path / f'mse_valid_lr{lr_id}.png',
                 mse_kal=mse_kal, model_mse_kal=history['model_kf_mse'])
    else:
        plot_mse(epoch_steps, history['valid_mse'], title, save_path / f'mse_valid_lr{lr_id}.png')
    
    # Weight updates
    if history['weights_updates']:
        weights_updates = torch.stack(history['weights_updates'], dim=1)
        plot_weights(
            history['train_steps'][:len(history['weights_updates'])],
            weights_updates,
            [f'param_{i}' for i in range(weights_updates.shape[0])],
            title,
            save_path / f'weights_updates_lr{lr_id}.png'
        )


# =============================================================================
# Full Pipeline (combines train + test)
# =============================================================================

def run_single_config(
    config: RunConfig,
    benchmarks_train: Optional[Dict] = None,
    benchmarks_test: Optional[Dict] = None,
    train_only: bool = False,
    test_only: bool = False,
) -> Dict[str, Any]:
    """
    Run training and/or testing for a single configuration.
    
    This is the main entry point for running one experiment.
    """
    results = {}
    
    # Create model
    model = create_model(config)
    
    # Train
    if not test_only:
        history = train_model(model, config, benchmarks_train)
        results['training'] = history
    
    # Test
    if not train_only:
        test_results = test_model(model, config, benchmarks_test)
        results['testing'] = test_results
    
    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results
