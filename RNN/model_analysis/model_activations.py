import os
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# pipeline_core_v2 is in RNN/train/; evaluate_models is a sibling in model_analysis/
_here = os.path.abspath(os.path.dirname(__file__))
_train_dir = os.path.abspath(os.path.join(_here, '..', 'train'))
for _p in [_train_dir, _here]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluate_models as eval
from pipeline_core_v2 import get_model_predictions
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA as SklearnCCA

# CUE_LABELS = ['cue_1', 'cue_2']
CUE_LABELS = ['cue_2', 'cue_1']

# Minimum deviant position labelled in the experimental sequence files (generated with
# rules_dpos_set=[[2,3,4],[4,5,6]], so dpos in {2..6}). This is a property of the files,
# not of any model.
EXPERIMENTAL_DPOS_MIN = 2


def dpos_conventions(info, experimental_dpos_min=EXPERIMENTAL_DPOS_MIN):
    """dpos alignment offsets between a trained model and the experimental files.

    A model's dpos output class ``c`` encodes deviant position ``c + dpos_min``, where
    ``dpos_min = min(rules_dpos_set)`` is read from the model's saved config
    (``info.data_config_dict``). Returns ``(dpos_min, shift)`` with
    ``shift = dpos_min - experimental_dpos_min``: add ``shift`` to an experimental dpos
    to express it in the model's convention, and the class index of a model-convention
    dpos is ``dpos - dpos_min``.

    Self-adjusting: the old lr0 model (rules_dpos_set=[[3,4,5],[5,6,7]]) yields (3, 1);
    a model retrained on [[2,3,4],[4,5,6]] yields (2, 0).
    """
    rules_dpos_set = np.asarray(info.data_config_dict['rules_dpos_set'])
    dpos_min = int(rules_dpos_set.min())
    return dpos_min, dpos_min - experimental_dpos_min


def load_trial_sequence(filepath, return_hierarch=False):
    """Load a trial CSV and return observation array and one-hot cue array (T, 2).

    Parameters
    ----------
    filepath : str or Path
        Path to the trial sequence CSV.
    return_labels : bool
        If True, also return the ground-truth class labels for the categorical
        modules, inserted right after the cue array:
        (obs, cue, ctx, dpos, rule, lim_std, d, tau_std, trial_n).
        - ctx comes from the 'trial_type' column (the context label)
        - dpos comes from the 'dpos' column (raw deviant positions, not yet
          shifted to 0-based class indices)
        - rule comes from the 'rule' column
        Each is an int64 array of shape (T,).

    Returns
    -------
    By default: (obs, cue, lim_std, d, tau_std, trial_n).
    With return_labels=True: (obs, cue, ctx, dpos, rule, lim_std, d, tau_std, trial_n).
    """
    df = pd.read_csv(filepath)
    obs = df['observation'].to_numpy(dtype=np.float32)
    cue_raw = df['cue'].to_numpy()
    label_to_idx = {label: i for i, label in enumerate(CUE_LABELS)}
    cue_idx = np.vectorize(label_to_idx.get)(cue_raw)
    cue = np.eye(len(CUE_LABELS), dtype=np.float32)[cue_idx]  # (T, 2)
    trial_n = df['trial_n']
    lim_std = df['lim_std'].iloc[0]
    d = df['d'].iloc[0]
    tau_std = df['tau_std'].iloc[0]
    if return_hierarch:
        ctx = df['trial_type'].to_numpy(dtype=np.int64)   # context label
        dpos = df['dpos'].to_numpy(dtype=np.int64)        # raw deviant position
        rule = df['rule'].to_numpy(dtype=np.int64)
        return obs, cue, ctx, dpos, rule, lim_std, d, tau_std, trial_n
    return obs, cue, lim_std, d, tau_std, trial_n


def to_model_tensors(obs, cue):
    """Convert observation (T,) and one-hot cue (T, 2) arrays to (1, T, dim) float tensors."""
    y = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    q = torch.tensor(cue, dtype=torch.float32).unsqueeze(0)                # (1, T, 2)
    return y, q


def load_trial_params(filepath):
    """Extract scalar trial parameters from a sequence CSV file.

    Reads the first row (parameters are constant within a trial) and returns a
    dict compatible with extract_sample_parameters, including si_stat derived
    from d = |lim_std - lim_dev| / sqrt(si_stat^2 + si_r^2).

    Returns
    -------
    dict with keys: 'tau', 'lim', 'si_stat', 'si_r'
    """
    row = pd.read_csv(filepath, nrows=1).iloc[0]
    d       = float(row['d'])
    tau     = float(row['tau_std'])
    lim_std = float(row['lim_std'])
    lim_dev = float(row['lim_dev'])
    si_r    = float(row['sigma_r'])

    numerator = np.abs(lim_std - lim_dev)
    si_stat = np.sqrt(max(0.0, (numerator / (d + 1e-8)) ** 2 - si_r ** 2))

    return {
        'tau':     tau,
        'lim':     [lim_std, lim_dev],
        'si_stat': si_stat,
        'si_r':    si_r,
    }


def run_forward_pass(model, y, q):
    """Run a forward pass and return raw hidden states per module.

    Parameters
    ----------
    model : nn.Module
        Trained model that accepts return_hidden=True.
    y : torch.Tensor
        Observation sequences, shape (batch, seq_len, obs_dim).
    q : torch.Tensor
        Query sequences, shape (batch, seq_len, q_dim).

    Returns
    -------
    dict
        Module name → tensor of shape (seq_len, n_layers, batch, hidden_dim).
    """
    with torch.no_grad():
        obs_outputs, ctx_outputs, dpos_outputs, rule_outputs, \
            obs_hidden, ctx_hidden, dpos_hidden, rule_hidden = \
            model(y[:, :-1, :], q[:, :-1, :], return_hidden=True)
    
    prob_output = {
        'obs':  obs_outputs,
        'ctx':  ctx_outputs,
        'dpos': dpos_outputs,
        'rule': rule_outputs,
    }

    hidden_states ={
        'obs':  obs_hidden,
        'ctx':  ctx_hidden,
        'dpos': dpos_hidden,
        'rule': rule_hidden,
    }

    return prob_output, hidden_states


def compute_hidden_norms(hidden_states, layer_idx=-1):
    """Compute L2 norms across neuron units for one layer of each module.

    Parameters
    ----------
    hidden_states : dict
        Module name → tensor of shape (seq_len, n_layers, batch, hidden_dim).
    layer_idx : int
        Index of the layer to extract. Default: -1 (last layer).

    Returns
    -------
    dict
        Module name → ndarray of shape (seq_len, batch).
    """
    norms = {}
    for module_name, hidden in hidden_states.items():
        layer_hidden = hidden[:, layer_idx, :, :].detach().cpu().numpy()  # (seq_len, batch, hidden_dim)
        norms[module_name] = np.linalg.norm(layer_hidden, axis=2)         # (seq_len, batch)
    return norms


def get_module_output_and_activity(model, y, q, layer_idx=-1):
    """Return per-module hidden activity norms and their temporal derivatives.

    Runs a single forward pass with return_hidden=True, selects the requested
    layer, reduces across neuron units with an L2 norm, and differentiates.

    Parameters
    ----------
    model : nn.Module
        Trained model that accepts return_hidden=True.
    y : torch.Tensor
        Observation sequences, shape (batch, seq_len, obs_dim).
    q : torch.Tensor
        Query sequences, shape (batch, seq_len, q_dim).
    layer_idx : int
        Which RNN layer to extract. Default: -1 (last layer).

    Returns
    -------
    hidden_activity : dict
        Module name → ndarray of shape (seq_len, batch).
    hidden_derivatives : dict
        Module name → ndarray of shape (seq_len-1, batch).
    """
    prob_output, hidden_states = run_forward_pass(model, y, q)
    hidden_activity = compute_hidden_norms(hidden_states, layer_idx=layer_idx)
    hidden_derivatives = {name: compute_derivatives(norms) for name, norms in hidden_activity.items()}
    return prob_output, hidden_activity, hidden_derivatives


def get_module_probabilities(model, y, q):
    """Return per-module output probabilities/distribution parameters.

    Runs a standard forward pass and delegates the output post-processing to
    pipeline_core_v2.get_model_predictions, so the observation mean/variance and
    the class probabilities are computed exactly as in training and evaluation
    (single source of truth). In particular:

    - 'obs' is a regressor: the returned columns are (mean, variance), where the
      variance is get_model_predictions' var_estim = softplus(raw) + 1e-6. This
      is a *variance* (the same quantity passed to GaussianNLLLoss at training
      time), not a standard deviation.
    - the remaining modules are classifiers: softmax class probabilities.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    y : torch.Tensor
        Observation sequences, shape (batch, seq_len, obs_dim).
    q : torch.Tensor
        Query sequences, shape (batch, seq_len, q_dim).

    Returns
    -------
    dict
        Module name → ndarray of shape (seq_len, batch, dim):
        - 'obs':  dim = 2, columns are (mean, variance)
        - others: dim = n_classes, softmax class probabilities
    """
    with torch.no_grad():
        model_output = model(y[:, :-1, :], q[:, :-1, :])
    pred = get_model_predictions(model, model_output)

    # get_model_predictions returns batch-major arrays:
    #   mu_estim / var_estim : (batch, seq_len)
    #   *_prob               : (batch, seq_len, n_classes)
    # Stack obs into (batch, seq_len, 2) then move to the (seq_len, batch, dim)
    # convention used elsewhere (time-major, batch second).
    obs = np.stack([pred['mu_estim'], pred['var_estim']], axis=-1)  # (batch, seq_len, 2)

    probabilities = {
        'obs':  np.transpose(obs, (1, 0, 2)),
        'ctx':  np.transpose(pred['ctx_prob'], (1, 0, 2)),
        'dpos': np.transpose(pred['dpos_prob'], (1, 0, 2)),
        'rule': np.transpose(pred['rule_prob'], (1, 0, 2)),
    }
    return probabilities


def gaussian_likelihood(observations, mean, variance):
    """Likelihood (density) of each observation under a Gaussian.

    Evaluates N(observation | mean, variance) elementwise, i.e. the value of the
    Gaussian probability density at each observation. This is the likelihood of
    the ground-truth observation under the obs module's predicted distribution.

    Parameters
    ----------
    observations, mean, variance : array-like, same shape
        Ground-truth observations and the predicted Gaussian mean/variance.

    Returns
    -------
    np.ndarray
        Per-element likelihood, same shape as the inputs.
    """
    obs = np.asarray(observations, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    var = np.asarray(variance, dtype=np.float64)
    return np.exp(-0.5 * (obs - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def class_likelihood(class_probs, labels):
    """Likelihood of the true class label under predicted class probabilities.

    For a (K, C) matrix ``lambda`` of class probabilities and 0-based ground-truth
    labels c, returns ``lambda[k, labels[k]]`` for each row k — the probability the
    model assigns to the true class. For a categorical distribution this *is* the
    likelihood of the observed class, which is why no other transform is needed.

    Parameters
    ----------
    class_probs : np.ndarray, shape (K, C)
        Per-row class probabilities (softmax outputs).
    labels : array-like of int, shape (K,)
        Zero-based ground-truth class indices, one per row.

    Returns
    -------
    np.ndarray, shape (K,)
        Likelihood of the true class at each row.
    """
    class_probs = np.asarray(class_probs)
    labels = np.asarray(labels, dtype=int)
    if labels.min() < 0 or labels.max() >= class_probs.shape[1]:
        raise IndexError(
            f"labels out of range [0, {class_probs.shape[1] - 1}]: "
            f"got min={labels.min()}, max={labels.max()}. "
            "Did you forget to shift to 0-based class indices (e.g. dpos)?"
        )
    return class_probs[np.arange(class_probs.shape[0]), labels]


def compute_derivatives(norms):
    """Compute temporal derivative (finite differences) of activity norms.

    Parameters
    ----------
    norms : np.ndarray
        Shape (seq_len, batch)

    Returns
    -------
    derivatives : np.ndarray
        Shape (seq_len-1, batch) - derivative at each timestep
    """
    return np.diff(norms, axis=0)


def compute_pairwise_sample_correlations(norms):
    """Compute pairwise correlations between all samples in a module.
    
    Parameters
    ----------
    norms : np.ndarray
        Shape (seq_len, batch)
    
    Returns
    -------
    tuple of (min_corr, max_corr)
    """
    n_samples = norms.shape[1]
    correlations = []
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            act_i = norms[:, i]
            act_j = norms[:, j]
            
            # Normalize for correlation
            act_i_norm = (act_i - act_i.mean()) / (act_i.std() + 1e-8)
            act_j_norm = (act_j - act_j.mean()) / (act_j.std() + 1e-8)
            corr = np.mean(act_i_norm * act_j_norm)
            correlations.append(corr)
    
    if correlations:
        return min(correlations), max(correlations)
    return 0.0, 0.0


def compute_pairwise_module_correlations(module_norms_dict, use_derivatives=False):
    """Pearson correlation between every (unordered) pair of modules.

    Parameters
    ----------
    module_norms_dict : dict
        Keys are module names, values are norms arrays of shape (seq_len, batch)
        (or 1D). Flattened across time and samples before correlating.
    use_derivatives : bool
        If True, correlate the temporal derivatives instead of the raw activity.

    Returns
    -------
    dict mapping "<name_i>_<name_j>" -> Pearson r (float), for i < j in the
    order the modules appear in `module_norms_dict`. A pair with a constant
    (zero-variance) vector gets a correlation of 0.0 instead of NaN.
    """
    module_names = list(module_norms_dict.keys())
    pair_correlations = {}

    for i, name_i in enumerate(module_names):
        for j, name_j in enumerate(module_names):
            if i < j:
                activity_i = module_norms_dict[name_i]
                activity_j = module_norms_dict[name_j]

                if use_derivatives:
                    activity_i = compute_derivatives(activity_i)
                    activity_j = compute_derivatives(activity_j)

                flat_i = activity_i.reshape(-1)
                flat_j = activity_j.reshape(-1)

                # pearsonr is undefined for a constant input; treat that as no correlation.
                if flat_i.std() == 0 or flat_j.std() == 0:
                    corr = 0.0
                else:
                    corr = pearsonr(flat_i, flat_j)[0]

                pair_correlations[f"{name_i}_{name_j}"] = float(corr)

    return pair_correlations


def compute_intermodule_correlations(module_norms_dict, use_derivatives=False, absolute=False):
    """Average pairwise Pearson correlation between modules.

    Parameters
    ----------
    module_norms_dict : dict
        Keys are module names, values are norms arrays
    use_derivatives : bool
        If True, compute correlations on derivatives instead of raw activity
    absolute : bool
        If True, average the absolute value of each pairwise correlation (a
        redundancy/coupling-strength metric, where +0.8 and -0.8 both count as
        strong). If False (default), average the signed correlations, so
        positive and negative pairs can cancel.

    Returns
    -------
    float - average correlation between modules
    """
    pair_correlations = compute_pairwise_module_correlations(
        module_norms_dict, use_derivatives=use_derivatives
    )
    values = list(pair_correlations.values())
    if not values:
        return 0.0
    if absolute:
        values = [abs(v) for v in values]
    return float(np.mean(values))


def extract_sample_parameters(pars, sample_idx):
    """Extract sample parameters d, tau, and si_stat.
    
    Computes d = |lim[0] - lim[1]| / sqrt(si_stat^2 + si_r^2)
    where lim[0]=mu_std, lim[1]=mu_dvt
    
    Parameters
    ----------
    pars : dict or list
        Parameter structure from test_data with keys: 'tau', 'lim', 'si_stat', 'si_q', 'si_r'
        Values are lists of numpy arrays or scalars (one value per sample)
    sample_idx : int
        Sample index
    
    Returns
    -------
    tuple of (d, tau, si_stat) or (None, None, None) if not available
    """
    try:
        if isinstance(pars, dict):
            # Extract base parameters
            tau_val = pars.get('tau', [None] * 100)[sample_idx] if 'tau' in pars else None
            si_stat_val = pars.get('si_stat', [None] * 100)[sample_idx] if 'si_stat' in pars else None
            si_r_val = pars.get('si_r', [None] * 100)[sample_idx] if 'si_r' in pars else None
            lim_val = pars.get('lim', [None] * 100)[sample_idx] if 'lim' in pars else None
            
            # Compute d if we have all required parameters
            if tau_val is not None and si_stat_val is not None and si_r_val is not None and lim_val is not None:
                # Convert to float if numpy arrays/values - handle array elements or scalars
                if hasattr(tau_val, '__len__'):
                    tau = float(tau_val[0]) if len(tau_val) > 0 else float(tau_val)
                else:
                    tau = float(tau_val)
                    
                if hasattr(si_stat_val, '__len__'):
                    si_stat = float(si_stat_val[0]) if len(si_stat_val) > 0 else float(si_stat_val)
                else:
                    si_stat = float(si_stat_val)
                    
                if hasattr(si_r_val, '__len__'):
                    si_r = float(si_r_val[0]) if len(si_r_val) > 0 else float(si_r_val)
                else:
                    si_r = float(si_r_val)
                
                # Extract mu_std and mu_dvt from lim
                if hasattr(lim_val, '__len__') and len(lim_val) >= 2:
                    mu_std = float(lim_val[0])
                    mu_dvt = float(lim_val[1])
                else:
                    mu_std = float(lim_val)
                    mu_dvt = float(lim_val)
                
                denominator = np.sqrt(si_stat**2 + si_r**2)
                d = np.abs(mu_std - mu_dvt) / (denominator + 1e-8)
                return float(d), tau, si_stat, si_r
    except Exception as e:
        pass
    return None, None, None, None


def plot_individual_trajectories(module_norms_dict, module_titles, timesteps, output_dir, model_name, 
                                  include_derivatives=False, pars=None, sample_indices=None):
    """Plot individual sample trajectories for all modules.
    
    Parameters
    ----------
    module_norms_dict : dict
        Keys are module names, values are norms arrays of shape (seq_len, batch)
    module_titles : dict
        Keys are module names, values are display titles
    timesteps : np.ndarray
        Timestep indices
    output_dir : Path
        Output directory for saving
    model_name : str
        Model name for file and title
    include_derivatives : bool
        If True, plot derivatives instead of activity
    pars : dict or list, optional
        Parameter structure for sample info
    sample_indices : np.ndarray, optional
        Indices of selected samples
    """
    n_modules = len(module_norms_dict)
    
    fig, axes = plt.subplots(n_modules, 1, figsize=(8, 3 * n_modules), sharex=True)
    if n_modules == 1:
        axes = [axes]
    
    title_suffix = "derivatives " if include_derivatives else ""
    fig.suptitle(f'Hidden activity {title_suffix}for individual samples\n  ', fontsize=16)
    
    # Compute correlation ranges for each module
    corr_ranges = {}
    for module_name, norms in module_norms_dict.items():
        min_corr, max_corr = compute_pairwise_sample_correlations(norms)
        corr_ranges[module_name] = (min_corr, max_corr)
    
    ax_idx = 0
    for module_name, norms in module_norms_dict.items():
        title = module_titles[module_name]
        ax = axes[ax_idx]
        
        if include_derivatives:
            derivatives = compute_derivatives(norms)
            timesteps_plot = timesteps[:-1]
            data_to_plot = derivatives
            ylabel = 'dActivity/dt'
        else:
            timesteps_plot = timesteps
            data_to_plot = norms
            ylabel = 'Activity (L2 norm)'
        
        # Plot each sample's activity/derivatives across time
        for sample_idx in range(data_to_plot.shape[1]):
            ax.plot(timesteps_plot, data_to_plot[:, sample_idx], alpha=0.6, linewidth=1.5)
        
        if include_derivatives:
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        ax.set_ylabel(ylabel)
        
        # Add correlation range to subplot title
        min_corr, max_corr = corr_ranges[module_name]
        ax.set_title(f'{title} (min corr. between samples: {min_corr:.2f}, max: {max_corr:.2f})')
        ax.grid(True, alpha=0.3)
        
        # Add legend with sample parameters for obs module
        if module_name == 'obs' and pars is not None:
            legend_lines = []
            legend_labels = []
            for sample_idx in range(data_to_plot.shape[1]):
                d, tau, si_stat, si_r = extract_sample_parameters(pars, sample_idx)
                if d is not None and tau is not None and si_stat is not None and si_r is not None:
                    label = f'd={d:.1f}, τ={tau:>5.1f}, σ_stat={si_stat:.1f}, σ_r={si_r:.1f}'
                else:
                    label = f'Sample {sample_idx}'
                legend_labels.append(label)
                if sample_idx < len(ax.lines):
                    legend_lines.append(ax.lines[sample_idx])
            
            ax.legend(legend_lines, legend_labels, loc='lower right' if not include_derivatives else 'upper right', fontsize=6, framealpha=0.9)
        
        ax_idx += 1
    
    axes[-1].set_xlabel('Timestep')
    plt.tight_layout()
    return fig


def plot_averaged_activity(module_norms_dict, module_titles, timesteps, output_dir, model_name, n_samples, include_derivatives=False):
    """Plot averaged activity with uncertainty across samples for all modules.
    
    Parameters
    ----------
    module_norms_dict : dict
        Keys are module names, values are norms arrays of shape (seq_len, batch)
    module_titles : dict
        Keys are module names, values are display titles
    timesteps : np.ndarray
        Timestep indices
    output_dir : Path
        Output directory for saving
    model_name : str
        Model name for file and title
    include_derivatives : bool
        If True, plot derivatives instead of activity
    """
    n_modules = len(module_norms_dict)
    
    fig, axes = plt.subplots(n_modules, 1, figsize=(8, 3 * n_modules), sharex=True)
    if n_modules == 1:
        axes = [axes]
    
    title_suffix = "derivatives " if include_derivatives else ""
    
    # Compute redundancy as a measure of module independence
    # Apply derivatives if needed before computing independence
    independence_dict = module_norms_dict
    if include_derivatives:
        independence_dict = {name: compute_derivatives(norms) for name, norms in module_norms_dict.items()}
    
    independence_metrics = compute_module_independence(independence_dict)
    redundancy = independence_metrics['redundancy']  # Lower = more independent
    # Display redundancy as "avg corr." in the title for consistency
    
    fig.suptitle(f'Average hidden activity {title_suffix}across {n_samples} samples\n(avg corr. between modules: {redundancy:.2f})', fontsize=16)
    
    ax_idx = 0
    for module_name, norms in module_norms_dict.items():
        title = module_titles[module_name]
        ax = axes[ax_idx]
        
        if include_derivatives:
            derivatives = compute_derivatives(norms)
            timesteps_plot = timesteps[:-1]
            mean_data = derivatives.mean(axis=1)
            std_data = derivatives.std(axis=1)
            ylabel = 'dActivity/dt'
        else:
            timesteps_plot = timesteps
            mean_data = norms.mean(axis=1)
            std_data = norms.std(axis=1)
            ylabel = 'Activity (L2 norm)'
        
        ax.plot(timesteps_plot, mean_data, '-', linewidth=2, label='Mean')
        ax.fill_between(timesteps_plot, 
                         mean_data - std_data,
                         mean_data + std_data,
                         alpha=0.3, label='± STD')
        
        if include_derivatives:
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax_idx += 1
    
    axes[-1].set_xlabel('Timestep')
    plt.tight_layout()
    return fig


def reshape_norms_by_position(data, period=8):
    """Split a (seq_len, batch) activity array into within-trial positions.

    Every timestep ``t`` is assigned a within-trial position ``t % period`` and a
    trial index ``t // period``. Because every trial spans exactly ``period``
    consecutive timesteps (and the sequence starts at trial 0, position 0), this
    is equivalent to ``df.groupby('trial_n').cumcount()`` used elsewhere, but
    works directly on the batched norms array without needing the trial_n column.

    Parameters
    ----------
    data : np.ndarray
        Shape (seq_len, batch) — e.g. a module's activity norms or derivatives.
    period : int
        Number of timesteps per trial (within-trial positions). Default 8.

    Returns
    -------
    dict
        Maps a 0-based within-trial position to a dict with:
          - 'trials' : np.ndarray (n_trials_p,)        trial index per point (x-axis)
          - 'values' : np.ndarray (n_trials_p, batch)  activity at this position
        Trailing positions may cover one fewer trial when seq_len is not an exact
        multiple of period.
    """
    seq_len = data.shape[0]
    by_pos = {}
    for p in range(period):
        ts = np.arange(p, seq_len, period)
        if ts.size == 0:
            continue
        by_pos[p] = {
            'trials': ts // period,
            'values': data[ts, :],
        }
    return by_pos


def plot_averaged_activity_by_position(module_norms_dict, module_titles, output_dir,
                                       model_name, n_samples, period=8,
                                       include_derivatives=False, show_std=True):
    """Plot per-module activity averaged over sequences, split by within-trial position.

    Unlike :func:`plot_averaged_activity` (timestep on the x-axis), this puts the
    *trial* index on the x-axis and draws one series per within-trial position
    (0..period-1). Each point is the activity at a given (trial, position)
    averaged over all sequences/samples, so there are ``period`` dots above every
    trial x-tick. Plotting against the trial index keeps each position's curve
    continuous (consecutive trials are adjacent), which a timestep x-axis would
    not — there the same position only recurs every ``period`` steps.

    Parameters
    ----------
    module_norms_dict : dict
        Module name → norms array of shape (seq_len, batch). batch is the number
        of sequences being averaged over.
    module_titles : dict
        Module name → display title.
    output_dir : Path
        Output directory (kept for signature parity; saving is done by the caller).
    model_name : str
        Model name (used only for context; the caller handles file names).
    n_samples : int
        Number of sequences averaged over (shown in the title).
    period : int
        Timesteps per trial / number of within-trial positions. Default 8.
    include_derivatives : bool
        If True, plot temporal derivatives instead of raw activity.
    show_std : bool
        If True (default), shade ±STD across sequences around each position's
        mean, like :func:`plot_averaged_activity`. Set False if the per-position
        bands overlap too much to read.
    """
    n_modules = len(module_norms_dict)

    fig, axes = plt.subplots(n_modules, 1, figsize=(10, 3 * n_modules), sharex=True)
    if n_modules == 1:
        axes = [axes]

    title_suffix = "derivatives " if include_derivatives else ""
    fig.suptitle(
        f'Average hidden activity {title_suffix}across {n_samples} sequences\n'
        f'(separated by within-trial position; trial index on x-axis)',
        fontsize=16,
    )

    cmap = plt.get_cmap('tab10')

    for ax, (module_name, norms) in zip(axes, module_norms_dict.items()):
        if include_derivatives:
            data = compute_derivatives(norms)
            ylabel = 'dActivity/dt'
        else:
            data = norms
            ylabel = 'Activity (L2 norm)'

        by_pos = reshape_norms_by_position(data, period=period)
        for p in sorted(by_pos):
            trials = by_pos[p]['trials']
            values = by_pos[p]['values']          # (n_trials_p, batch)
            mean_data = values.mean(axis=1)
            color = cmap(p % 10)
            ax.plot(trials, mean_data, '-', linewidth=1.5,
                    color=color, label=f'pos {p + 1}')
            if show_std:
                std_data = values.std(axis=1)
                ax.fill_between(trials, mean_data - std_data, mean_data + std_data,
                                color=color, alpha=0.2)

        if include_derivatives:
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        ax.set_ylabel(ylabel)
        ax.set_title(module_titles[module_name])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Trial (trial_n)')

    # One shared legend to the right of the subplots.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='within-trial\nposition',
               loc='center left', bbox_to_anchor=(0.98, 0.5))
    plt.tight_layout(rect=[0, 0, 0.97, 1])
    return fig


def extract_deviant_activity(data, dev_pos, period=8):
    """Pick the activity at each trial's deviant position, grouped by deviant value.

    For every trial the deviant tone sits at within-trial position ``dev_pos`` (a
    0-based index, constant within the trial). The activity at that deviant is the
    timestep ``trial * period + dev_pos`` of ``data``. Trials are grouped by the
    *value* of their deviant position and, for each (trial, value) cell, averaged
    over the sequences whose deviant fell on that position at that trial.

    Parameters
    ----------
    data : np.ndarray
        Shape (seq_len, n_seq) — a module's activity norms or derivatives.
    dev_pos : np.ndarray
        Shape (n_seq, n_trials) — 0-based within-trial deviant position for every
        sequence and trial.
    period : int
        Timesteps per trial. Default 8.

    Returns
    -------
    dict
        deviant-position value (int) → dict of equal-length arrays:
          - 'trials' : trial index per point (x-axis)
          - 'mean'   : activity at the deviant, averaged over contributing sequences
          - 'std'    : std across those sequences
          - 'count'  : number of contributing sequences
        A (trial, value) point is dropped when no sequence has that deviant value
        at that trial, or when its timestep falls past ``seq_len`` (e.g. the very
        last position of the last trial after the ``y[:, :-1]`` slice).
    """
    seq_len = data.shape[0]
    n_trials = dev_pos.shape[1]
    trials = np.arange(n_trials)

    out = {}
    for v in np.unique(dev_pos):
        v = int(v)
        rec = {'trials': [], 'mean': [], 'std': [], 'count': []}
        for t in trials:
            g = t * period + v
            if g >= seq_len:
                continue
            mask = dev_pos[:, t] == v          # sequences with deviant at v in trial t
            if not mask.any():
                continue
            vals = data[g, mask]
            rec['trials'].append(t)
            rec['mean'].append(vals.mean())
            rec['std'].append(vals.std())
            rec['count'].append(int(mask.sum()))
        out[v] = {k: np.asarray(val) for k, val in rec.items()}
    return out


def plot_deviant_activity_by_position(module_norms_dict, dev_pos, module_titles,
                                      output_dir, model_name, n_samples, period=8,
                                      include_derivatives=False, show_std=True):
    """Plot per-module activity at each trial's deviant position, grouped by deviant value.

    Like :func:`plot_averaged_activity_by_position` the x-axis is the trial index,
    but instead of drawing every within-trial position this keeps only the single
    timestep that is the deviant in each trial. Trials are grouped (coloured) by
    the value of their deviant position, and each point is the activity at that
    deviant timestep averaged over the sequences whose deviant fell on that
    position at that trial — one dot per trial per deviant-position value. Colours
    match :func:`plot_averaged_activity_by_position` (same within-trial position →
    same colour), so the two figures are directly comparable.

    Parameters
    ----------
    module_norms_dict : dict
        Module name → norms array of shape (seq_len, n_seq).
    dev_pos : np.ndarray
        Shape (n_seq, n_trials) — 0-based within-trial deviant position per
        sequence and trial.
    module_titles : dict
        Module name → display title.
    output_dir : Path
        Output directory (kept for signature parity; saving is done by the caller).
    model_name : str
        Model name (used only for context; the caller handles file names).
    n_samples : int
        Number of sequences (shown in the title).
    period : int
        Timesteps per trial. Default 8.
    include_derivatives : bool
        If True, plot temporal derivatives instead of raw activity.
    show_std : bool
        If True (default), shade ±STD across the contributing sequences around
        each deviant-value series. Set False if the bands overlap too much.
    """
    n_modules = len(module_norms_dict)

    fig, axes = plt.subplots(n_modules, 1, figsize=(10, 3 * n_modules), sharex=True)
    if n_modules == 1:
        axes = [axes]

    title_suffix = "derivatives " if include_derivatives else ""
    fig.suptitle(
        f'Hidden activity {title_suffix}at the deviant position, across {n_samples} sequences\n'
        f'(grouped by deviant-position value; trial index on x-axis)',
        fontsize=16,
    )

    cmap = plt.get_cmap('tab10')

    for ax, (module_name, norms) in zip(axes, module_norms_dict.items()):
        if include_derivatives:
            data = compute_derivatives(norms)
            ylabel = 'dActivity/dt'
        else:
            data = norms
            ylabel = 'Activity (L2 norm)'

        by_dev = extract_deviant_activity(data, dev_pos, period=period)
        for v in sorted(by_dev):
            rec = by_dev[v]
            if rec['trials'].size == 0:
                continue
            color = cmap(v % 10)
            ax.plot(rec['trials'], rec['mean'], linestyle='-',
                    color=color, label=f'pos {v + 1}')
            if show_std:
                ax.fill_between(rec['trials'], rec['mean'] - rec['std'],
                                rec['mean'] + rec['std'], color=color, alpha=0.15)

        if include_derivatives:
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        ax.set_ylabel(ylabel)
        ax.set_title(module_titles[module_name])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Trial (trial_n)')

    # One shared legend to the right of the subplots.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='deviant\nposition',
               loc='center left', bbox_to_anchor=(0.98, 0.5))
    plt.tight_layout(rect=[0, 0, 0.97, 1])
    return fig


def compute_sample_difference_single_module(norms, sample_idx_1, sample_idx_2):
    """Compute difference in activity between two samples for a single module.
    
    Metrics:
    - L2 distance: Euclidean distance in activity space over time
    - Correlation: How synchronized are the two temporal patterns
    - Max absolute difference: Peak difference over time
    
    Parameters
    ----------
    norms : np.ndarray
        Shape (seq_len, batch)
    sample_idx_1, sample_idx_2 : int
        Sample indices to compare
    
    Returns
    -------
    dict with metrics
    """
    activity_1 = norms[:, sample_idx_1]
    activity_2 = norms[:, sample_idx_2]
    
    l2_distance = np.linalg.norm(activity_1 - activity_2)
    
    # Normalize for correlation to handle scale differences
    activity_1_norm = (activity_1 - activity_1.mean()) / (activity_1.std() + 1e-8)
    activity_2_norm = (activity_2 - activity_2.mean()) / (activity_2.std() + 1e-8)
    correlation = np.mean(activity_1_norm * activity_2_norm)  # cosine similarity
    
    max_diff = np.max(np.abs(activity_1 - activity_2))
    
    return {
        'l2_distance': l2_distance,
        'correlation': correlation,  # higher = more similar
        'max_difference': max_diff
    }


def compute_sample_difference_all_modules(module_norms_dict, sample_idx_1, sample_idx_2):
    """Compute difference in activity across all modules between two samples.
    
    Parameters
    ----------
    module_norms_dict : dict
        Keys are module names, values are norms arrays
    sample_idx_1, sample_idx_2 : int
        Sample indices to compare
    
    Returns
    -------
    dict with per-module and aggregate metrics
    """
    results = {}
    all_diffs = []
    all_corrs = []
    
    for module_name, norms in module_norms_dict.items():
        module_metrics = compute_sample_difference_single_module(norms, sample_idx_1, sample_idx_2)
        results[module_name] = module_metrics
        all_diffs.append(module_metrics['l2_distance'])
        all_corrs.append(module_metrics['correlation'])
    
    # Aggregate metrics
    results['aggregate'] = {
        'mean_l2_distance': np.mean(all_diffs),
        'mean_correlation': np.mean(all_corrs),
        'total_l2_distance': np.linalg.norm(all_diffs),  # Euclidean distance in module space
    }
    
    return results


def compute_module_independence(module_norms_dict):
    """Compute metrics for independence/separability of modules.
    
    Metrics:
    - Correlation matrix: Pairwise correlations between module activity patterns
    - Redundancy: Average absolute correlation (lower = more independent)
    - Explained variance ratio (PCA): How much variance each module explains
    
    Parameters
    ----------
    module_norms_dict : dict
        Keys are module names, values are norms arrays of shape (seq_len, batch)
    
    Returns
    -------
    dict with multiple independence metrics
    """
    module_names = list(module_norms_dict.keys())
    n_modules = len(module_names)
    
    # Flatten each module across time and samples
    module_activities = {}
    for name in module_names:
        norms = module_norms_dict[name]
        # Flatten to (seq_len * batch,) and normalize
        flat = norms.reshape(-1)
        flat_norm = (flat - flat.mean()) / (flat.std() + 1e-8)
        module_activities[name] = flat_norm
    
    # --- Correlation matrix ---
    correlation_matrix = np.zeros((n_modules, n_modules))
    for i, name_i in enumerate(module_names):
        for j, name_j in enumerate(module_names):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                corr, _ = pearsonr(module_activities[name_i], module_activities[name_j])
                correlation_matrix[i, j] = corr
    
    # Redundancy: average absolute correlation (excluding diagonal)
    off_diag = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
    redundancy = np.mean(np.abs(off_diag))
    independence_score = 1 - redundancy  # Higher = more independent
    
    # --- Explained variance (PCA approach) ---
    stacked_activities = np.column_stack([module_activities[name] for name in module_names])
    pca = PCA()
    pca.fit(stacked_activities)
    explained_var_ratio = pca.explained_variance_ratio_
    
    # Effective number of dimensions (entropy of explained variance)
    entropy = -np.sum(explained_var_ratio * np.log(explained_var_ratio + 1e-8))
    effective_dims = np.exp(entropy)  # e^entropy, bounded by n_modules
    
    # --- Linear regression redundancy ---
    # For each module, fit linear regression from others to predict it
    r_squared_values = []
    for i, target_name in enumerate(module_names):
        target = module_activities[target_name]
        feature_names = [name for j, name in enumerate(module_names) if j != i]
        features = np.column_stack([module_activities[name] for name in feature_names])
        
        # Simple linear regression R-squared
        from numpy.linalg import lstsq
        try:
            coeffs, residuals, _, _ = lstsq(features, target, rcond=None)
            ss_res = np.sum(residuals) if residuals.size > 0 else np.sum((target - features @ coeffs) ** 2)
            ss_tot = np.sum((target - target.mean()) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
            r_squared_values.append(r_squared)
        except:
            r_squared_values.append(0.0)
    
    mean_r_squared = np.mean(r_squared_values)
    
    return {
        'correlation_matrix': correlation_matrix,
        'module_names': module_names,
        'redundancy': redundancy,  # 0-1, lower = more independent
        'independence_score': independence_score,  # 0-1, higher = more independent
        'explained_variance_ratio': explained_var_ratio,
        'effective_dimensions': effective_dims,
        'mean_linear_predictability': mean_r_squared,  # 0-1, lower = more independent
    }


# ---------------------------------------------------------------------------
# Joining per-trial deviant activity with the likelihood at the deviant.
#
# These helpers consume the per-file CSVs produced by
#   - model_act_exp_trials_deviant.py  -> activations_deviant/*_deviant_trial.csv
#       one row per trial: <module>_norm / <module>_deriv sampled at the deviant
#   - model_prob_exp_trials.py         -> probabilities/*_probabilities.csv
#       one row per timestep, including lik_<module> (likelihood of the ground
#       truth under each module's predicted distribution).
# and study how module activity at the deviant relates to how (un)likely the
# model found that deviant.
# ---------------------------------------------------------------------------

MODULE_NAMES = ['obs', 'ctx', 'dpos', 'rule']


def load_deviant_activity(act_csv):
    """Load a per-trial deviant-activity CSV (model_act_exp_trials_deviant.py).

    Returns the DataFrame as written: one row per trial with columns
    trial_n, deviant_pos, <module>_norm, <module>_deriv, lim_std, d, tau_std.
    """
    return pd.read_csv(act_csv)


def load_deviant_likelihoods(prob_csv, context_label=1, module_names=MODULE_NAMES):
    """Per-trial likelihoods at the deviant position from a *_probabilities.csv.

    The probabilities file has one row per timestep, where the ``ctx`` column is
    the next-step ground-truth context label (1 marks the deviant tone). Filtering
    to ``ctx == context_label`` therefore keeps exactly the deviant of each trial,
    and ``lik_<module>`` on that row is the likelihood the model assigned to the
    deviant. There is one such row per trial (verified on the data), but if a file
    ever had more, duplicates are averaged within a trial.

    Parameters
    ----------
    prob_csv : str or Path
        Path to a *_probabilities.csv file.
    context_label : int
        Context label marking the deviant tone (default 1).
    module_names : list of str
        Modules whose likelihood columns (``lik_<module>``) to keep.

    Returns
    -------
    pandas.DataFrame
        Columns: trial_n, deviant_pos, lik_<module> for each module.
    """
    df = pd.read_csv(prob_csv)
    dev = df[df['ctx'] == context_label].copy()
    lik_cols = [f'lik_{m}' for m in module_names]
    keep = ['trial_n', 'dpos'] + lik_cols
    dev = dev[keep].rename(columns={'dpos': 'deviant_pos'})
    dev = dev.groupby(['trial_n', 'deviant_pos'], as_index=False)[lik_cols].mean()
    return dev


def join_activity_likelihood(act_df, lik_df, module_names=MODULE_NAMES):
    """Merge per-trial deviant activity and deviant likelihoods on trial_n.

    Adds a ``mean_norm`` column: the mean activity across modules at the deviant
    (the average module activity referred to in the analysis). The merge is an
    inner join on trial_n, so only trials present in both frames are kept.

    Parameters
    ----------
    act_df : pandas.DataFrame
        Output of :func:`load_deviant_activity`.
    lik_df : pandas.DataFrame
        Output of :func:`load_deviant_likelihoods`.
    module_names : list of str
        Modules to average over for ``mean_norm``.

    Returns
    -------
    pandas.DataFrame
        The merged per-trial frame with an added ``mean_norm`` column.
    """
    merged = pd.merge(act_df, lik_df, on='trial_n', how='inner',
                      suffixes=('', '_lik'))
    norm_cols = [f'{m}_norm' for m in module_names]
    merged['mean_norm'] = merged[norm_cols].mean(axis=1)
    return merged


def compute_activity_likelihood_correlations(df, activity_cols, likelihood_cols,
                                             method='pearson'):
    """Correlation of every activity column with every likelihood column.

    Parameters
    ----------
    df : pandas.DataFrame
        Joined per-trial frame.
    activity_cols, likelihood_cols : list of str
        Column names to cross-correlate (activity on rows, likelihood on columns).
    method : {'pearson', 'spearman'}
        Correlation coefficient to use.

    Returns
    -------
    pandas.DataFrame
        Correlation matrix indexed by activity_cols, columns likelihood_cols.
        Pairs with fewer than 3 finite points or a constant column are NaN.
    """
    func = pearsonr if method == 'pearson' else spearmanr
    corr = pd.DataFrame(index=activity_cols, columns=likelihood_cols, dtype=float)
    for a in activity_cols:
        for l in likelihood_cols:
            x = df[a].to_numpy(dtype=float)
            y = df[l].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if x.size >= 3 and x.std() > 0 and y.std() > 0:
                corr.loc[a, l] = float(func(x, y)[0])
            else:
                corr.loc[a, l] = np.nan
    return corr


def plot_activity_likelihood_grid(df, activity_cols, likelihood_cols,
                                  activity_labels=None, likelihood_labels=None,
                                  add_fit=True, color='steelblue'):
    """Scatter grid of activity (columns) vs likelihood (rows) with fit + r.

    Each panel scatters one activity column against one likelihood column, draws a
    least-squares line, and annotates Pearson r and Spearman rho. Deliberately
    plain: small markers, thin black fit line, no bold fonts.

    Returns
    -------
    matplotlib.figure.Figure
    """
    activity_labels = activity_labels or activity_cols
    likelihood_labels = likelihood_labels or likelihood_cols
    n_rows, n_cols = len(likelihood_cols), len(activity_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.8 * n_cols, 2.8 * n_rows),
                             squeeze=False)
    for i, lcol in enumerate(likelihood_cols):
        for j, acol in enumerate(activity_cols):
            ax = axes[i][j]
            x = df[acol].to_numpy(dtype=float)
            y = df[lcol].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            ax.scatter(x, y, s=10, alpha=0.45, color=color, edgecolors='none')

            if x.size >= 3 and x.std() > 0 and y.std() > 0:
                r = pearsonr(x, y)[0]
                rho = spearmanr(x, y)[0]
                if add_fit:
                    b1, b0 = np.polyfit(x, y, 1)
                    xs = np.array([x.min(), x.max()])
                    ax.plot(xs, b0 + b1 * xs, color='black', linewidth=1)
                ax.set_title(f'r={r:.2f}, ρ={rho:.2f}', fontsize=9)

            if i == n_rows - 1:
                ax.set_xlabel(activity_labels[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(likelihood_labels[i], fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr, title='', value_fmt='{:.2f}'):
    """Heatmap of an activity-vs-likelihood correlation matrix.

    Parameters
    ----------
    corr : pandas.DataFrame
        Output of :func:`compute_activity_likelihood_correlations`.
    title : str
        Axes title.
    value_fmt : str
        Format string for the per-cell annotations.

    Returns
    -------
    matplotlib.figure.Figure
    """
    values = corr.to_numpy(dtype=float)
    n_rows, n_cols = values.shape

    fig, ax = plt.subplots(figsize=(1.1 * n_cols + 2.5, 0.6 * n_rows + 2.0))
    im = ax.imshow(values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(list(corr.columns), rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(list(corr.index), fontsize=9)

    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, value_fmt.format(v), ha='center', va='center',
                        fontsize=8, color='black')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig