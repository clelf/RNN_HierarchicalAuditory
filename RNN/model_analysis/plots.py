"""Every figure drawn by the model_analysis scripts.

Contract
--------
A ``plot_*`` function BUILDS a figure and returns it; it does not save. Saving
goes through :func:`save_figure`, the single point in this package that touches
the filesystem for figures, so that each PNG gets an entry in the
``figure_metadoc.txt`` sitting next to it recording which script produced it.

Because every figure is saved from inside this module, ``__file__`` here would
attribute all of them to ``plots.py``, which is useless. Each entry point
therefore registers itself once, at the top of its ``__main__`` block::

    import plots
    plots.set_script_path(__file__)

and every figure saved afterwards is attributed to that script. Pass
``parallel=True`` when the entry point is one of several concurrent workers
writing into the same figure directory (a SLURM array element, or a run under
``--jobs``), so the metadata file is appended under a lock instead of rewritten.

The few functions that draw a whole family of figures in a loop
(``plot_metric_violins``, ``plot_calibration_curves``,
``plot_likelihood_distributions_per_model``) save internally, through the same
choke point.
"""

import os
import sys
from collections import Counter
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm as sp_norm, pearsonr, spearmanr

# figure_metadoc.py lives at the Workspace root, three levels above this file.
_workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _workspace not in sys.path:
    sys.path.append(_workspace)
from figure_metadoc import log_figure_metadata

from analysis_config import (
    DPI,
    LIKELIHOOD_LABELS,
    MODULE_NAMES as MODULES,
    PLOT_LIKELIHOOD_COLS,
)
from analysis_core import (
    compute_derivatives,
    compute_module_independence,
    compute_pairwise_sample_correlations,
    extract_deviant_activity,
    extract_sample_parameters,
    reshape_norms_by_position,
)
from metrics import model_predictions


# =============================================================================
# Constants
# =============================================================================

# Metrics that also get a Kalman-filter reference drawn on their panel.
KF_METRICS = ('mse', 'obs_loglik', 'ks_statistic')
KF_COLOR = '#9B59B6'   # dashed KF reference line only; violins are not coloured
KF_LABEL = 'Kalman Filter'

# Upper triangle of the 4x4 module matrix, with the always-empty left column
# (obs) and bottom row (rule) dropped: rows = obs/ctx/dpos, cols = ctx/dpos/rule.
ROW_IDXS = [0, 1, 2]
COL_IDXS = [1, 2, 3]


# =============================================================================
# Saving, and the figure-provenance record
# =============================================================================

# Entry point every figure of this run is attributed to; set by set_script_path.
_SCRIPT_PATH = None
_PARALLEL = False


def set_script_path(script_path, parallel=False):
    """Record which script the figures saved from now on should be credited to.

    Called once per run, from the entry point's ``__main__`` block, as
    ``plots.set_script_path(__file__)``. `parallel` marks the caller as one of
    several concurrent workers sharing a figure directory.
    """
    global _SCRIPT_PATH, _PARALLEL
    _SCRIPT_PATH = script_path
    _PARALLEL = parallel


def save_figure(fig, figures_dir, filename, script_path=None, parallel=None):
    """Save `fig`, log which script produced it, and close it.

    `script_path` and `parallel` default to whatever `set_script_path` recorded;
    pass them explicitly to override for one figure. When neither has been set,
    the figure is still saved and a warning is printed, so a forgotten
    `set_script_path` costs provenance rather than the figure itself.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / filename

    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    source = script_path if script_path is not None else _SCRIPT_PATH
    if source is None:
        print(f"  warning: no script path registered; {filename} saved without "
              f"a figure_metadoc entry (call plots.set_script_path(__file__))")
    else:
        log_figure_metadata(out_path, source,
                            _PARALLEL if parallel is None else parallel)

    print(f"  saved: {out_path.name}")
    return out_path


# =============================================================================
# Experimental-sequence activity figures
# =============================================================================
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


def plot_deviant_activity_by_position(dev_df, module_titles, output_dir, model_name,
                                      n_samples, include_derivatives=False, show_std=True):
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
    dev_df : pandas.DataFrame
        One row per (sequence, trial): the per-trial deviant-activity CSVs of the
        sequences to average over, concatenated, with '<module>_norm' /
        '<module>_deriv' columns sampled at the deviant, 'trial_n', and 'dev_pos'
        (0-based within-trial deviant position; see
        exp_sequence_analysis.load_deviant_activity_frame).
    module_titles : dict
        Module name → display title; one subplot per module, in this order.
    output_dir : Path
        Output directory (kept for signature parity; saving is done by the caller).
    model_name : str
        Model name (used only for context; the caller handles file names).
    n_samples : int
        Number of sequences (shown in the title).
    include_derivatives : bool
        If True, plot temporal derivatives instead of raw activity.
    show_std : bool
        If True (default), shade ±STD across the contributing sequences around
        each deviant-value series. Set False if the bands overlap too much.
    """
    n_modules = len(module_titles)

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

    for ax, module_name in zip(axes, module_titles):
        if include_derivatives:
            column = f'{module_name}_deriv'
            ylabel = 'dActivity/dt'
        else:
            column = f'{module_name}_norm'
            ylabel = 'Activity (L2 norm)'

        by_dev = extract_deviant_activity(dev_df, column)
        for v in sorted(by_dev):
            rec = by_dev[v]
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


# =============================================================================
# Activity / likelihood joint figures
# =============================================================================

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


# =============================================================================
# Calibration, and model display order
# =============================================================================

def plot_calibration_curve(
    y_true: np.ndarray,
    mu_pred: np.ndarray,
    var_pred: np.ndarray,
    save_path: Path = None,
    title: str = "KS Calibration Plot",
    ax: plt.Axes = None,
    color: str = "#1f77b4",
    label: str = None,
    alpha_band: float = 0.05,
):
    """
    Plot the empirical CDF of the Probability Integral Transform (PIT) values
    against the ideal uniform diagonal.

    For a perfectly calibrated model the PIT values are Uniform(0,1), so the
    empirical CDF should lie on the diagonal.  The maximum vertical distance
    from the diagonal is the Kolmogorov–Smirnov D-statistic.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples, seq_len)
        True observations.
    mu_pred : np.ndarray, shape (n_samples, seq_len)
        Predicted means.
    var_pred : np.ndarray, shape (n_samples, seq_len)
        Predicted variances.
    save_path : Path, optional
        If given, save the figure to this path.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        If provided, draw on this axes (useful for multi-panel figures).
    color : str
        Line colour for the empirical CDF.
    label : str, optional
        Legend label for the empirical CDF curve.
    alpha_band : float
        Significance level for the KS confidence band (default 0.05 → 95 %).

    Returns
    -------
    fig : matplotlib Figure or None
        The figure object (None when an external *ax* was supplied).
    ks_stat : float
        The pooled KS D-statistic.
    """

    # --- Compute PIT values (pooled across samples & time) ---
    sigma_pred = np.sqrt(var_pred)
    pit = sp_norm.cdf((y_true - mu_pred) / sigma_pred)
    pit_flat = pit.ravel()
    pit_flat = pit_flat[~np.isnan(pit_flat)]
    pit_flat.sort()

    n = len(pit_flat)
    ecdf = np.arange(1, n + 1) / n          # empirical CDF values
    F    = pit_flat                           # theoretical quantiles (sorted PITs)

    # KS statistic = max |ECDF(f) - f|
    ks_stat = np.max(np.abs(ecdf - F))

    # --- KS confidence band width ---
    # c(alpha) for two-sided KS test: 1.36 (alpha=0.05), 1.22 (0.10), 1.63 (0.01)
    c_alpha = {0.01: 1.63, 0.05: 1.36, 0.10: 1.22}.get(alpha_band, 1.36)
    band_half = c_alpha / np.sqrt(n)

    # --- Plot ---
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = None

    # Confidence band around the diagonal
    F_grid = np.linspace(0, 1, 500)
    ax.fill_between(
        F_grid,
        np.clip(F_grid - band_half, 0, 1),
        np.clip(F_grid + band_half, 0, 1),
        color="grey", alpha=0.75,
        label=f"{int((1-alpha_band)*100)}% KS band",
    )

    # Ideal diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal (Uniform)")

    # Empirical CDF of PITs
    ax.plot(F, ecdf, color=color, linewidth=1.5,
            label=label or f"Empirical CDF (D={ks_stat:.4f})")

    # Mark the point of maximum deviation
    idx_max = np.argmax(np.abs(ecdf - F))
    ax.plot([F[idx_max], F[idx_max]], [F[idx_max], ecdf[idx_max]],
            color="red", linewidth=1.5, linestyle="-",
            label=f"Max deviation = {ks_stat:.4f}")
    ax.plot(F[idx_max], ecdf[idx_max], "o", color="red", markersize=5)

    ax.set_xlabel("PIT value (theoretical quantile)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    if own_fig and save_path is not None:
        save_path = Path(save_path)
        save_figure(fig, save_path.parent, save_path.name)

    return fig, ks_stat


def get_desired_order(results):
    # Categorize models
    obs = [mod for mod in results if results[mod]['learning_objective'] == 'obs']
    obs_ctx = [mod for mod in results if results[mod]['learning_objective'] == 'obs_ctx']
    ctx = [mod for mod in results if results[mod]['learning_objective'] == 'ctx']
    # Handle any other objectives (e.g., 'all')
    other = [mod for mod in results if results[mod]['learning_objective'] not in ['obs', 'obs_ctx', 'ctx']]

    # Sort obs by bottleneck_dim
    obs_sorted = sorted(obs, key=lambda mod: results[mod]['bottleneck_dim'])
    # Sort obs_ctx by bottleneck_dim, then kappa (inverted order)
    obs_ctx_sorted = sorted(obs_ctx, key=lambda mod: (results[mod]['bottleneck_dim'], results[mod]['kappa']))
    # Sort ctx by bottleneck_dim
    ctx_sorted = sorted(ctx, key=lambda mod: results[mod]['bottleneck_dim'])
    # Sort other by bottleneck_dim
    other_sorted = sorted(other, key=lambda mod: results[mod]['bottleneck_dim'])

    # Concatenate in desired order
    return obs_sorted + obs_ctx_sorted + ctx_sorted + other_sorted


# =============================================================================
# Cross-model metric comparison (violins)
# =============================================================================

def make_display_names(results):
    """Short label per model: objective, kappa (obs_ctx only) and bottleneck dim.

    Labels end up as DataFrame column names, so they must be unique: models whose
    hyper-parameters give the same label fall back to their folder name.
    """
    labels = {}
    for mod, entry in results.items():
        obj = entry.get('learning_objective', '')
        kappa = f"(k={entry['kappa']}) " if obj == 'obs_ctx' else ''
        labels[mod] = f"Obj={obj} {kappa}bdim={entry.get('bottleneck_dim')}"

    counts = Counter(labels.values())
    return {mod: (mod if counts[label] > 1 else label) for mod, label in labels.items()}


def default_colors(n):
    """The colours seaborn gives n violins when no palette is specified.

    Only used to build legends that match what was drawn — the plots themselves
    never pass a palette.
    """
    return sns.color_palette(n_colors=n, desat=0.75)


def style_results(results):
    """Order the models and attach a 'disp_name' to each entry."""
    results_ord = {mod: results[mod] for mod in get_desired_order(results)}
    names = make_display_names(results_ord)
    for mod, entry in results_ord.items():
        entry['disp_name'] = names[mod]
    return results_ord


def metric_frame(results_ord, metric, kf_values=None, transform=None):
    """Stack the per-sample values of `metric` across models into a DataFrame.

    Models that do not have the metric are skipped. `kf_values` prepends a
    Kalman-filter column; `transform` is applied to each model array (used for
    the model / KF ratio figure).

    Returns the DataFrame, or None if no model has the metric.
    """
    arrays, names = [], []

    if kf_values is not None:
        arrays.append(np.asarray(kf_values))
        names.append(KF_LABEL)

    for entry in results_ord.values():
        if metric not in entry:
            continue
        values = np.asarray(entry[metric])
        arrays.append(transform(values) if transform is not None else values)
        names.append(entry['disp_name'])

    if not arrays:
        return None
    return pd.DataFrame(np.array(arrays).T, columns=names)


def violin_panel(ax, df, title, log_scale=False, symlog=False,
                 hline=None, hline_label=None):
    """Draw one violin panel: one violin per column, x tick labels stripped.

    No palette is passed, so the violins take seaborn's default colours.
    """
    sns.violinplot(data=df, ax=ax, cut=0, log_scale=log_scale)
    ax.set_title(title)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks=ticks, labels=[''] * len(ticks))
    if symlog:
        ax.set_yscale('symlog')
        ax.set_ylim(top=df.max().max())
    if hline is not None:
        ax.axhline(hline, color=KF_COLOR, linestyle='--', linewidth=2, label=hline_label)
    return ax


def legend_handles(names, colors, kf_label=None):
    """Legend handles: one thick colour bar per model, optionally a KF dashed line."""
    handles = []
    if kf_label is not None:
        handles.append(plt.Line2D([0], [0], color=KF_COLOR, linestyle='--',
                                  linewidth=2, label=kf_label))
    handles += [plt.Line2D([0], [0], color=color, linewidth=8, label=name)
                for name, color in zip(names, colors)]
    return handles


def draw_color_legend(ax, names, colors, fontsize=12, row_height=0.07):
    """Colour-patch legend filling a whole (otherwise unused) axes."""
    ax.axis('off')
    for i, (name, color) in enumerate(zip(names, colors)):
        ax.add_patch(plt.Rectangle((0, 1 - i * row_height), 0.2, 0.05, color=color,
                                   transform=ax.transAxes, clip_on=False))
        ax.text(0.25, 1 - i * row_height + 0.025, name, va='center', ha='left',
                transform=ax.transAxes, fontsize=fontsize)


def clip_ylim(ax, df, lo_pct, hi_pct, pad_frac=0.15):
    """Clip the y-axis to the median of the per-column [lo_pct, hi_pct] range.

    Keeps the well-behaved models readable: clipping each column separately and
    then taking the median of those bounds stops one outlier model from dragging
    the whole axis range.
    """
    col_los, col_his = [], []
    for col in df.columns:
        col_vals = df[col].dropna().values
        if col_vals.size == 0:
            continue
        col_los.append(np.percentile(col_vals, lo_pct))
        col_his.append(np.percentile(col_vals, hi_pct))
    if not col_los:
        return
    lo, hi = np.median(col_los), np.median(col_his)
    pad = (hi - lo) * pad_frac
    ax.set_ylim(lo - pad, hi + pad)


def plot_metric_violins(results_ord, metrics, output_dir, kf_metrics=None,
                        file_suffix='test', ncols=3, panel_size=(6.5, 5),
                        row_labels=None, log_metrics=('mse',),
                        symlog_metrics=('obs_loglik',), save_individual=True):
    """One violin panel per metric, all models side by side.

    The last panel of the grid holds the model colour legend. When `kf_metrics`
    is given, observation-level metrics also get a KF violin and a dashed line at
    the KF median. With `save_individual`, each metric is additionally saved as
    its own figure (subplot_<metric>_<file_suffix>.png).

    Violins take seaborn's default colours, which are assigned by column
    position: the shared legend is therefore only valid while every panel shows
    the same models (a warning is printed when they do not).
    """
    output_dir = Path(output_dir)
    metrics = [m for m in metrics if any(m in entry for entry in results_ord.values())]
    if not metrics:
        print("No model has any of the requested metrics — nothing to plot.")
        return None

    nrows = int(np.ceil((len(metrics) + 1) / ncols))  # +1 panel for the legend
    fig, axs = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True,
                            figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    flat_axs = axs.ravel()

    column_sets = set()
    for metric, ax in zip(metrics, flat_axs):
        kf_values = kf_metrics.get(metric) if kf_metrics is not None else None
        df = metric_frame(results_ord, metric, kf_values=kf_values)
        kf_median = np.median(kf_values) if kf_values is not None else None
        column_sets.add(tuple(df.columns))

        violin_panel(ax, df, f'{metric} across models',
                     log_scale=metric in log_metrics,
                     symlog=metric in symlog_metrics,
                     hline=kf_median, hline_label='KF median')

        if save_individual:
            fig_single, ax_single = plt.subplots(figsize=(7, 5))
            violin_panel(ax_single, df, f'{metric} across models',
                         log_scale=metric in log_metrics,
                         symlog=metric in symlog_metrics,
                         hline=kf_median, hline_label='KF median')
            ax_single.legend(
                handles=legend_handles(df.columns, default_colors(len(df.columns)),
                                       kf_label='KF median' if kf_median is not None else None),
                loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=11,
                frameon=False, ncol=3,
            )
            save_figure(fig_single, output_dir,
                        f"subplot_{metric}_{file_suffix}.png")

    # Optional row labels (e.g. 'tone process estimation' / 'context estimation')
    if row_labels:
        for row, label in enumerate(row_labels[:nrows]):
            axs[row][0].set_ylabel(label)

    # Legend in the first free panel; blank out whatever is left.
    if len(column_sets) > 1:
        print("  ! panels do not all show the same models, so the shared legend "
              "matches only the panels holding every column")
    legend_names = max(column_sets, key=len)
    draw_color_legend(flat_axs[len(metrics)], legend_names,
                      default_colors(len(legend_names)))
    for ax in flat_axs[len(metrics) + 1:]:
        ax.axis('off')

    return save_figure(fig, output_dir,
                       f"model_comparison_metrics_{file_suffix}.png")


def plot_kf_ratio_violins(results_ord, kf_metrics, output_dir, metrics=KF_METRICS,
                          panel_size=(6, 5), default_pct=(2, 98),
                          pct_by_metric=None):
    """Violin plots of each model's metric divided by the KF's, per sample.

    A ratio of 1 (dashed purple line) means the model matches the KF. The y-axis
    is clipped per metric (see clip_ylim) since the ratios can be very skewed —
    log-likelihood ratios especially, hence the tighter percentiles for them.
    """
    output_dir = Path(output_dir)
    pct_by_metric = {'obs_loglik': (10, 90), **(pct_by_metric or {})}
    metrics = [m for m in metrics if m in kf_metrics]

    fig, axs = plt.subplots(1, len(metrics), squeeze=False, constrained_layout=True,
                            figsize=(panel_size[0] * len(metrics), panel_size[1]))
    axs = axs.ravel()

    columns = ()
    for metric, ax in zip(metrics, axs):
        kf_values = np.asarray(kf_metrics[metric])
        df = metric_frame(results_ord, metric,
                          transform=lambda v: v / (kf_values + 1e-10))
        if df is None:
            ax.axis('off')
            continue
        columns = max(columns, tuple(df.columns), key=len)

        violin_panel(ax, df, f'{metric}  (model / KF)',
                     hline=1.0, hline_label='KF baseline')
        clip_ylim(ax, df, *pct_by_metric.get(metric, default_pct))

    fig.supylabel('model / KF ratio  (dashed purple line = KF baseline)', x=0.01)
    fig.legend(handles=legend_handles(columns, default_colors(len(columns)),
                                      kf_label='KF baseline (ratio = 1)'),
               loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=13, frameon=False)

    return save_figure(fig, output_dir, "model_comparison_ratio_vs_kf.png")


def plot_calibration_curves(models_info, results_ord, test_sets, output_dir,
                            device='cpu', min_obs_for_em=None):
    """Per-model calibration curve: empirical CDF of the PIT values vs uniform.

    Re-runs the forward pass because the raw predictions (not just the metrics)
    are needed, each model on its own entry of `test_sets`. `min_obs_for_em`
    applies the same KF-window restriction as evaluate_model.
    """
    output_dir = Path(output_dir)
    infos = {info.model_dir.name: info for info in models_info}

    saved = []
    for mod_name, entry in results_ord.items():
        info = infos.get(mod_name)
        if info is None:
            continue

        y_target, mu_pred, var_pred = model_predictions(
            info, test_sets[mod_name], device=device)
        if min_obs_for_em is not None:
            start = min_obs_for_em - 1  # prediction index aligned on y[min_obs_for_em:]
            y_target, mu_pred, var_pred = y_target[:, start:], mu_pred[:, start:], var_pred[:, start:]

        save_path = output_dir / f"calibration_curve_{mod_name}.png"
        plot_calibration_curve(
            y_target, mu_pred, var_pred,
            save_path=save_path,
            title=f"KS Calibration – {entry['disp_name']}",
        )
        saved.append(save_path)
    return saved


# =============================================================================
# Module-pair correlation figures
# =============================================================================

def tuple_hue(df, columns):
    """Hue Series grouping rows by the combination of `columns`.

    The notebook's ``df[['lim_std', 'd']].apply(tuple, axis=1)`` idiom: one hue
    level per observed combination of parameter values, index-aligned with `df`.
    """
    return df[list(columns)].apply(tuple, axis=1)


def resolve_hue(df, hue):
    """Accept either a single column name or a list of columns as the hue."""
    if isinstance(hue, str):
        return hue
    return tuple_hue(df, hue)


def plot_score_histogram(df, column, hue, figures_dir, filename,
                         title=None, bins=20, ax=None):
    """Histogram (+KDE) of one correlation-score column, split by `hue`.

    Shared by every distribution cell of part 1: only `column` and `hue` change
    between them. `hue` is a column name or a list of columns to combine.
    """
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(data=df, x=column, bins=bins, kde=True,
                 hue=resolve_hue(df, hue), ax=ax)
    if title:
        ax.set_title(title)

    if own_figure:
        fig.tight_layout()
        save_figure(fig, figures_dir, filename)
        return None
    return ax


def module_matrix_figure(draw_cell, figsize=(9, 9), legend_pair=('obs', 'ctx')):
    """Build the upper-triangle module-pair matrix and let `draw_cell` fill it.

    Shared frame for every scatter-matrix cell of part 2. `draw_cell` is called
    as ``draw_cell(ax, module_y, module_x, legend)`` for each populated panel;
    `legend` is True only for the single panel whose legend is later moved to
    the right of the figure (`legend_pair`, i.e. the top-left panel).

    Returns ``(fig, axs)``.
    """
    fig, axs = plt.subplots(len(ROW_IDXS), len(COL_IDXS), figsize=figsize)
    for ri, i in enumerate(ROW_IDXS):
        mi = MODULES[i]
        for ci, j in enumerate(COL_IDXS):
            mj = MODULES[j]
            ax = axs[ri, ci]
            if j > i:
                draw_cell(ax, mi, mj, (mi, mj) == legend_pair)
                ax.set_xlabel(mj)
                ax.set_ylabel(mi)
            else:
                ax.axis('off')

    # Row and column labels outside the panels
    for ri, i in enumerate(ROW_IDXS):
        axs[ri, 0].text(-0.3, 0.5, MODULES[i], transform=axs[ri, 0].transAxes,
                        fontsize=14, ha='right', va='center')
    for ci, j in enumerate(COL_IDXS):
        axs[-1, ci].text(0.5, -0.15, MODULES[j], transform=axs[-1, ci].transAxes,
                         fontsize=14, ha='center', va='top')
    return fig, axs


def legend_to_right(fig, axs, title, rect_right=0.84, anchor_x=0.85):
    """Move the single per-panel legend to a reserved strip on the right.

    Reserve the strip FIRST (``tight_layout(rect=...)``), then place the legend
    into it -- otherwise tight_layout re-expands the axes over the legend.
    """
    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend = axs[0, 0].get_legend()
    if legend is not None:
        legend.remove()
    fig.tight_layout(rect=[0, 0, rect_right, 1])
    fig.legend(handles, labels, title=title, loc='center left',
               bbox_to_anchor=(anchor_x, 0.5))


def plot_score_histograms(df, bins=30):
    """Build one histogram per score column of `df`, as subplots in one figure.

    Every column except 'sequence_name' is treated as a score column; each gets
    its own histogram over all sequences.
    """
    score_cols = [c for c in df.columns
                  if c.startswith('activity_') or c.startswith('derivative_')]
    n = len(score_cols)
    ncols = 3
    nrows = ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, col in zip(axes, score_cols):
        # ax.hist(df[col].dropna(), bins=bins, color='tab:orange' if "derivative" in col else 'tab:blue', edgecolor='black')
        sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax,
                    color='tab:orange' if "derivative" in col else 'tab:blue')
        ax.set_title(col, fontsize=12)
        ax.set_xlabel('correlation')
        ax.set_ylabel('count')

    # Hide any unused subplot slots
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


# =============================================================================
# Deviant likelihood distributions
# =============================================================================

# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_likelihood_distributions(summary_df, out_png=None, value_cols=None,
                                  hue="dpos", ncols=3, title=None):
    """Overlapping histograms of one model's likelihood averages.

    One subplot per likelihood column, the `hue` levels overlaid within each.
    Everything about the histograms themselves is left to seaborn's defaults.
    """
    if value_cols is None:
        value_cols = [f"mean_{col}" for col in PLOT_LIKELIHOOD_COLS]

    # Drop columns that are absent or all-NaN, they would plot empty.
    value_cols = [c for c in value_cols
                  if c in summary_df.columns and summary_df[c].notna().any()]
    if not value_cols:
        raise ValueError("None of the requested likelihood columns hold data.")

    plot_df = summary_df
    if hue is not None and pd.api.types.is_numeric_dtype(summary_df[hue]):
        # dpos is stored as an integer; as a category seaborn gives it discrete
        # hue levels instead of treating it as a continuous variable.
        plot_df = summary_df.assign(**{hue: summary_df[hue].astype("category")})

    nrows = int(np.ceil(len(value_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows),
                             squeeze=False)
    flat_axes = axes.ravel()

    for ax, col in zip(flat_axes, value_cols):
        sns.histplot(data=plot_df, x=col, hue=hue, ax=ax)

        module, kind = LIKELIHOOD_LABELS.get(
            col[len("mean_"):] if col.startswith("mean_") else col, ("", col))
        ax.set_title(f"{module} — {kind}" if module else kind)

    for ax in flat_axes[len(value_cols):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_likelihood_distributions_per_model(summaries, figure_dir, hue="dpos", **kwargs):
    """One figure per model, saved as <model>_likelihood_distributions_by_<hue>.png."""
    figure_dir = Path(figure_dir)
    saved = []
    for model_name, summary_df in summaries.items():
        fig = plot_likelihood_distributions(
            summary_df, hue=hue,
            title=f"Per-sequence mean likelihood at deviant positions\n{model_name}",
            **kwargs,
        )
        saved.append(save_figure(
            fig, figure_dir,
            f"{model_name}_likelihood_distributions_by_{hue}.png"))
    return saved


def plot_dpos_prob_hist(avg_df, model_name, value_col, bins=30):
    """Histogram of the per-sequence means, with the overall mean marked.

    `value_col` is the summariser's output column for the averaged quantity --
    'mean_lik_dpos' for the probability of the true deviant-position class.
    """
    means = avg_df[value_col].to_numpy(dtype=float)
    valid = means[~np.isnan(means)]
    n_nan = int(np.sum(np.isnan(means)))
    if n_nan:
        print(f"Note: {n_nan} sequence(s) had no valid rows and are excluded from the histogram.")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(valid, bins=bins, range=(0, 1))
    ax.axvline(valid.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f"mean = {valid.mean():.3f}")
    ax.set_xlabel('Per-sequence mean P(true deviant position)')
    ax.set_ylabel('Number of sequences')
    ax.set_title(f'dpos probability at true deviant\n{model_name}  (n = {len(valid)} sequences)')
    ax.legend()
    fig.tight_layout()

    print(f"Summary over {len(valid)} sequences: "
          f"mean={valid.mean():.4f}, median={np.median(valid):.4f}, "
          f"min={valid.min():.4f}, max={valid.max():.4f}")
    return fig


# =============================================================================
# Module-pair correlation figure set
# =============================================================================
#
# One function per figure of the module-correlation analysis. Each is a thin
# choice of column and hue over the shared helpers above; they save directly
# because a run draws the whole set in one go.

PAIR_COLUMNS = ['activity_obs_ctx', 'activity_obs_dpos', 'activity_obs_rule',
                'activity_ctx_dpos', 'activity_ctx_rule', 'activity_dpos_rule']

# 'activity_mean' is the aggregate score; the correlation table also carries
# 'activity_mean_abs' and 'activity_abs_mean'.
MEAN_COLUMN = 'activity_mean'

POSITION_CMAPS = ['Blues', 'Oranges', 'Greens', 'Reds',
                  'Purples', 'YlOrBr', 'RdPu', 'Greys']


def plot_obs_ctx_by_lim_std(act_df, figures_dir):
    """[cell 4] OBS-CTX correlations by lim_std.

    Effect of lim_std (no effect of tau_std or d taken individually, not shown;
    nor combined, see `plot_obs_ctx_by_d_tau_std`).
    """
    plot_score_histogram(act_df, 'activity_obs_ctx', 'lim_std', figures_dir,
                         "obs_ctx__by_lim_std.png",
                         title="OBS-CTX correlation by lim_std")


def plot_obs_ctx_by_d_tau_std(act_df, figures_dir):
    """[cell 5] OBS-CTX correlations by (d, tau_std): no effect of the pair."""
    plot_score_histogram(act_df, 'activity_obs_ctx', ['d', 'tau_std'], figures_dir,
                         "obs_ctx__by_d_tau_std.png",
                         title="OBS-CTX correlation by (d, tau_std)")


def plot_obs_dpos_by_lim_std(act_df, figures_dir):
    """[cell 7] OBS-DPOS correlations by lim_std. Effect of lim_std."""
    plot_score_histogram(act_df, 'activity_obs_dpos', 'lim_std', figures_dir,
                         "obs_dpos__by_lim_std.png",
                         title="OBS-DPOS correlation by lim_std")


def plot_ctx_dpos_by_all_params(act_df, figures_dir):
    """[cell 9] CTX-DPOS by (lim_std, d, tau_std).

    Different distribution profiles emerge when showing all 3-level
    combinations individually.
    """
    plot_score_histogram(act_df, 'activity_ctx_dpos', ['lim_std', 'd', 'tau_std'],
                         figures_dir, "ctx_dpos__by_lim_std_d_tau_std.png",
                         title="CTX-DPOS correlation by (lim_std, d, tau_std)")


def plot_ctx_dpos_by_lim_std_d(act_df, figures_dir):
    """[cell 10] CTX-DPOS by (lim_std, d).

    Two distinct profiles emerge:
    - group 1: (lim_std = -0.6, d = -1) & (lim_std = 0.3, d = 1)
    - group 2: (lim_std = -0.6, d = 1)  & (lim_std = 0.3, d = -1)
    """
    plot_score_histogram(act_df, 'activity_ctx_dpos', ['lim_std', 'd'],
                         figures_dir, "ctx_dpos__by_lim_std_d.png",
                         title="CTX-DPOS correlation by (lim_std, d)")


def plot_ctx_rule_by_all_params(act_df, figures_dir):
    """[cell 12] CTX-RULE by (lim_std, d, tau_std).

    Different distribution profiles emerge when showing all 3-level
    combinations individually.
    """
    plot_score_histogram(act_df, 'activity_ctx_rule', ['lim_std', 'd', 'tau_std'],
                         figures_dir, "ctx_rule__by_lim_std_d_tau_std.png",
                         title="CTX-RULE correlation by (lim_std, d, tau_std)")


def plot_ctx_rule_by_lim_std_d(act_df, figures_dir):
    """[cell 13] CTX-RULE by (lim_std, d).

    Two distinct profiles emerge:
    - group 1: (lim_std = -0.6, d = -1) & (lim_std = 0.3, d = 1)
    - group 2: (lim_std = -0.6, d = 1)  & (lim_std = 0.3, d = -1)
    """
    plot_score_histogram(act_df, 'activity_ctx_rule', ['lim_std', 'd'],
                         figures_dir, "ctx_rule__by_lim_std_d.png",
                         title="CTX-RULE correlation by (lim_std, d)")


def plot_dpos_rule_by_all_params(act_df, figures_dir):
    """[cell 15] DPOS-RULE by (lim_std, d, tau_std).

    No clear effect; some 3-level combinations even show bi-modal distributions
    (-0.6, -1, 160; -0.6, 1, 160?).
    """
    plot_score_histogram(act_df, 'activity_dpos_rule', ['lim_std', 'd', 'tau_std'],
                         figures_dir, "dpos_rule__by_lim_std_d_tau_std.png",
                         title="DPOS-RULE correlation by (lim_std, d, tau_std)")


def plot_mean_by_lim_std_d(act_df, figures_dir):
    """[cell 17] Mean correlation by (lim_std, d). Effect of lim_std + d."""
    plot_score_histogram(act_df, MEAN_COLUMN, ['lim_std', 'd'], figures_dir,
                         "mean__by_lim_std_d.png",
                         title="Mean pair correlation by (lim_std, d)")


def plot_mean_by_tau_std_per_condition(act_df, figures_dir):
    """[cell 18] One subplot per (lim_std, d) combination, tau_std as hue.

    tau_std has different effects in the different combinations of lim_std + d.
    """
    combinations = act_df[['lim_std', 'd']].drop_duplicates().values
    ncols = 2
    nrows = max(1, ceil(len(combinations) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axs = np.atleast_1d(axs).reshape(-1)

    for i, (lim_std, d) in enumerate(combinations):
        ax = axs[i]
        subset = act_df[(act_df['lim_std'] == lim_std) & (act_df['d'] == d)]
        sns.histplot(data=subset, x=MEAN_COLUMN, bins=20, kde=True,
                     hue='tau_std', ax=ax)
        ax.set_title(f'lim_std={lim_std}, d={d}')

    for ax in axs[len(combinations):]:
        ax.set_visible(False)

    fig.tight_layout()
    save_figure(fig, figures_dir, "mean__by_tau_std_per_lim_std_d.png")


def plot_mean_by_all_params(act_df, figures_dir):
    """[cell 19] Mean correlation by (lim_std, d, tau_std), all 16 conditions."""
    plot_score_histogram(act_df, MEAN_COLUMN, ['lim_std', 'd', 'tau_std'],
                         figures_dir, "mean__by_lim_std_d_tau_std.png",
                         title="Mean pair correlation by (lim_std, d, tau_std)")


def plot_pairs_vs_mean(act_df, figures_dir, filename, density=True):
    """[cells 21 & 22] 2D histogram of each module pair against the mean score.

    Checks whether average activities closest to zero also display activity
    closest to zero in as many module pairs as possible.

    `density=True` reproduces cell 21 (mako colormap, ``pthresh=.1`` to blank
    the sparsest bins); `density=False` reproduces cell 22 (seaborn defaults).
    """
    ncols = 3
    nrows = ceil(len(PAIR_COLUMNS) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    axs = np.atleast_1d(axs).reshape(-1)

    hue = tuple_hue(act_df, ['lim_std', 'd'])
    extra = dict(bins=20, pthresh=.1, cmap="mako") if density else {}

    for ax, col in zip(axs, PAIR_COLUMNS):
        sns.histplot(data=act_df, x=col, y=MEAN_COLUMN, hue=hue, ax=ax, **extra)

    for ax in axs[len(PAIR_COLUMNS):]:
        ax.set_visible(False)

    fig.tight_layout()
    save_figure(fig, figures_dir, filename)


def plot_sequence_means_matrix(seq_means, figures_dir, hue_columns, filename,
                               legend_title='params'):
    """[cells 25 & 27] Module-pair scatter matrix, one dot per sequence.

    Each dot is the association of the mean activity of two modules within one
    sequence, so there are as many dots per panel as sequences. Cell 25 colours
    by (lim_std, d), cell 27 by the full (lim_std, d, tau_std) condition.
    """
    hue = tuple_hue(seq_means, hue_columns)

    def draw_cell(ax, mi, mj, legend):
        sns.scatterplot(data=seq_means, x=mj, y=mi, hue=hue, ax=ax, s=15,
                        legend=legend)

    fig, axs = module_matrix_figure(draw_cell)
    legend_to_right(fig, axs, legend_title, rect_right=0.84, anchor_x=0.85)
    save_figure(fig, figures_dir, filename)


def plot_timestep_matrix(ts_df, figures_dir, filename, suptitle):
    """[cells 29 & 31] Module-pair scatter matrix, one dot per timestep.

    Cell 29 passes a single sequence, cell 31 the per-timestep mean over all
    sequences. Colour encodes the timestep (viridis).

    Progression through the sequence (increasing timestep) moves the association
    of activations up vertically / right horizontally, towards a centre in the
    upper right corner but not its extreme borders -- i.e. activity increases
    with progression through the sequence.
    """
    def draw_cell(ax, mi, mj, legend):
        sns.scatterplot(data=ts_df, x=f'{mj}_norm', y=f'{mi}_norm',
                        hue='timestep', palette='viridis', ax=ax, s=15,
                        legend=legend)

    fig, axs = module_matrix_figure(draw_cell)
    fig.suptitle(suptitle, fontsize=12)
    legend_to_right(fig, axs, 'timestep', rect_right=0.9, anchor_x=0.91)
    save_figure(fig, figures_dir, filename)


def plot_first_sequence_matrix(seq_files, figures_dir):
    """[cell 29] Per-timestep matrix for the first sequence only."""
    first_seq = pd.read_csv(seq_files[0])   # one row per timestep
    first_seq['timestep'] = range(len(first_seq))
    plot_timestep_matrix(first_seq, figures_dir,
                         "timestep_matrix__first_sequence.png",
                         os.path.basename(seq_files[0]))


def plot_timestep_mean_matrix(ts_means, figures_dir):
    """[cell 31] Per-timestep matrix, averaged over all sequences."""
    plot_timestep_matrix(ts_means, figures_dir,
                         "timestep_matrix__mean_over_sequences.png",
                         'Per-timestep mean over all sequences')


def plot_timestep_matrix_by_position(ts_means, pos_order, figures_dir):
    """[cell 33] Per-timestep matrix coloured by within-trial position.

    Same upper-triangle matrix as `plot_timestep_mean_matrix`, but one discrete
    colour per within-trial position (derived from ``trial_n``) instead of a
    timestep gradient.
    """
    def draw_cell(ax, mi, mj, legend):
        sns.scatterplot(data=ts_means, x=f'{mj}_norm', y=f'{mi}_norm',
                        hue='position', hue_order=pos_order, palette='tab10',
                        ax=ax, s=20, legend=legend)

    fig, axs = module_matrix_figure(draw_cell)
    fig.suptitle('Per-timestep mean over all sequences, '
                 'coloured by within-trial position (from trial_n)', fontsize=13)
    legend_to_right(fig, axs, 'position', rect_right=0.93, anchor_x=0.94)
    save_figure(fig, figures_dir, "timestep_matrix__by_position.png")


def plot_timestep_matrix_by_position_gradient(ts_means, pos_order, figures_dir):
    """[cell 34] Per-position matrix with a within-position timestep gradient.

    Same data as `plot_timestep_matrix_by_position`, but each position gets its
    own sequential colormap spanning that position's timestep range, so the
    progression across trials is readable within each position.
    """
    positions = pos_order

    def draw_cell(ax, mi, mj, legend):
        for p_idx, p in enumerate(positions):
            sub = ts_means[ts_means['position'] == p]
            t_min, t_max = sub['timestep'].min(), sub['timestep'].max()
            ax.scatter(sub[f'{mj}_norm'], sub[f'{mi}_norm'],
                       c=sub['timestep'], cmap=POSITION_CMAPS[p_idx % len(POSITION_CMAPS)],
                       s=20, vmin=t_min, vmax=t_max)

    fig, _ = module_matrix_figure(draw_cell)
    fig.suptitle('Module-pair activity association at deviant positions, per position, '
                 'and across trials (light: early trials, darker: late trials)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.88, 1])

    # Right-side legend: one stacked gradient bar per position (ax.scatter draws
    # no legend of its own, so `legend_to_right` does not apply here).
    cbar_x, cbar_w, cbar_h, cbar_gap = 0.895, 0.07, 0.028, 0.022
    n = len(positions)
    total_h = n * cbar_h + (n - 1) * cbar_gap
    top_y = 0.5 + total_h / 2
    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    for p_idx, p in enumerate(positions):
        y0 = top_y - p_idx * (cbar_h + cbar_gap) - cbar_h
        cax = fig.add_axes([cbar_x, y0, cbar_w, cbar_h])
        cax.imshow(gradient, aspect='auto',
                   cmap=POSITION_CMAPS[p_idx % len(POSITION_CMAPS)])
        cax.set_xticks([])
        cax.set_yticks([])
        fig.text(cbar_x - 0.005, y0 + cbar_h / 2, str(p),
                 va='center', ha='right', fontsize=10)

    fig.text(cbar_x + cbar_w / 2, top_y + 0.018, 'position',
             ha='center', va='bottom', fontsize=10)
    fig.text(cbar_x, top_y + 0.005, 'early t', ha='left', va='bottom', fontsize=7)
    fig.text(cbar_x + cbar_w, top_y + 0.005, 'late t', ha='right', va='bottom', fontsize=7)

    save_figure(fig, figures_dir, "timestep_matrix__by_position_gradient.png")

if __name__ == '__main__':
    pass
