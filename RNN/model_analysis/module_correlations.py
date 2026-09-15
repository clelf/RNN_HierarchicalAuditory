"""Module-pair correlation / association figures for experimental sequences.

Script version of ``act_exp_param_analaysis copy.ipynb``. The notebook has two
parts, each wrapped here in a single entry point:

1. ``run_module_pair_correlations`` -- distributions of the per-sequence
   correlation scores (one row per sequence, produced by
   ``compute_exp_trials_corr.py``), split by the generative parameters
   ``lim_std`` / ``d`` / ``tau_std``.
2. ``run_module_associations`` -- scatter matrices of the raw module activities
   (``*_norm`` columns of the per-sequence activation files), showing how two
   modules co-vary across sequences, across timesteps, and per within-trial
   position.

Each notebook cell maps to one function; the machinery shared between cells
(the histogram call, the upper-triangle matrix frame, the right-side legend)
lives in the helpers at the top. Figures are written as PNGs instead of being
displayed inline.
"""

from math import ceil
from pathlib import Path
import glob
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================================
# Constants
# =============================================================================

BASE_OUTPUT_DIR = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output")

# Module names as stored in the activation CSV files (see model_act_exp_trials.py)
MODULES = ['obs', 'ctx', 'dpos', 'rule']

# Upper triangle of the 4x4 module matrix, with the always-empty left column
# (obs) and bottom row (rule) dropped: rows = obs/ctx/dpos, cols = ctx/dpos/rule.
ROW_IDXS = [0, 1, 2]
COL_IDXS = [1, 2, 3]

# The six per-pair correlation columns of exp_trials_correlations.csv
PAIR_COLUMNS = ['activity_obs_ctx', 'activity_obs_dpos', 'activity_obs_rule',
                'activity_ctx_dpos', 'activity_ctx_rule', 'activity_dpos_rule']

# The notebook uses 'activity_mean' as the aggregate score; compute_exp_trials_corr.py
# also writes 'activity_mean_abs' and 'activity_abs_mean'.
MEAN_COLUMN = 'activity_mean'

DPI = 150


# =============================================================================
# Shared helpers
# =============================================================================

def model_paths(model_name, base_dir=BASE_OUTPUT_DIR):
    """Resolve the input/output paths for one model.

    Returns
    -------
    dict with keys
        'activations_dir' : per-sequence '*_activations.csv' files
        'correlation_csv' : the per-sequence correlation scores table
        'figures_dir'     : where this script writes its PNGs (created)
    """
    model_dir = Path(base_dir) / model_name
    activations_dir = model_dir / "activations"
    figures_dir = model_dir / "module_correlations"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        'activations_dir': activations_dir,
        'correlation_csv': activations_dir / "correlation_scores" / "exp_trials_correlations.csv",
        'figures_dir': figures_dir,
    }


def save_figure(fig, figures_dir, filename):
    """Save and close a figure. Replaces the notebook's inline display."""
    out_path = Path(figures_dir) / filename
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path.name}")
    return out_path


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


# =============================================================================
# Part 1 -- data loading
# =============================================================================

def load_correlation_scores(correlation_csv):
    """[cell 1] Load the per-sequence correlation scores table.

    One row per experimental sequence, written by ``compute_exp_trials_corr.py``.
    Sorted by the generative parameters so the hue levels come out in a stable
    order across figures.
    """
    act_df = pd.read_csv(correlation_csv, index_col=0)
    act_df.sort_values(by=['lim_std', 'd', 'tau_std'], inplace=True)
    return act_df


# =============================================================================
# Part 1 -- OBS-CTX
# =============================================================================

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


# =============================================================================
# Part 1 -- OBS-DPOS
# =============================================================================

def plot_obs_dpos_by_lim_std(act_df, figures_dir):
    """[cell 7] OBS-DPOS correlations by lim_std. Effect of lim_std."""
    plot_score_histogram(act_df, 'activity_obs_dpos', 'lim_std', figures_dir,
                         "obs_dpos__by_lim_std.png",
                         title="OBS-DPOS correlation by lim_std")


# =============================================================================
# Part 1 -- CTX-DPOS
# =============================================================================

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


# =============================================================================
# Part 1 -- CTX-RULE
# =============================================================================

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


# =============================================================================
# Part 1 -- DPOS-RULE
# =============================================================================

def plot_dpos_rule_by_all_params(act_df, figures_dir):
    """[cell 15] DPOS-RULE by (lim_std, d, tau_std).

    No clear effect; some 3-level combinations even show bi-modal distributions
    (-0.6, -1, 160; -0.6, 1, 160?).
    """
    plot_score_histogram(act_df, 'activity_dpos_rule', ['lim_std', 'd', 'tau_std'],
                         figures_dir, "dpos_rule__by_lim_std_d_tau_std.png",
                         title="DPOS-RULE correlation by (lim_std, d, tau_std)")


# =============================================================================
# Part 1 -- MEAN over module pairs
# =============================================================================

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


# =============================================================================
# Part 2 -- data loading
# =============================================================================

def list_sequence_files(activations_dir):
    """Sorted list of the per-sequence activation files ('*_activations.csv')."""
    return sorted(glob.glob(os.path.join(str(activations_dir), "*_activations.csv")))


def load_sequence_means(seq_files):
    """[cells 25 & 27] One row per sequence: mean activity of each module.

    One file = one sequence. Also carries the sequence's generative parameters
    (lim_std, d, tau_std), which are constant within a file.
    """
    records = []
    for f in seq_files:
        seq_df = pd.read_csv(f)
        rec = {m: seq_df[f'{m}_norm'].mean() for m in MODULES}
        rec['lim_std'] = seq_df['lim_std'].iloc[0]
        rec['d'] = seq_df['d'].iloc[0]
        rec['tau_std'] = seq_df['tau_std'].iloc[0]
        records.append(rec)
    return pd.DataFrame(records)


def load_timestep_means(seq_files, with_position=False):
    """[cells 31 & 33] One row per timestep: module activity averaged over sequences.

    With `with_position`, also returns the 1-based within-trial position of each
    timestep (derived from ``trial_n``), plus the sorted position order used as
    ``hue_order``. The position is consistent across sequences, so averaging it
    over sequences and rounding recovers the integer position.
    """
    per_ts = []
    for f in seq_files:
        seq_df = pd.read_csv(f)
        sub = seq_df[[f'{m}_norm' for m in MODULES]].copy()
        sub['timestep'] = range(len(sub))
        if with_position:
            # 0-based index of the timestep within its trial
            sub['position'] = seq_df.groupby('trial_n').cumcount()
        per_ts.append(sub)

    ts_means = pd.concat(per_ts).groupby('timestep').mean().reset_index()

    if not with_position:
        return ts_means, None

    ts_means['position'] = ts_means['position'].round().astype(int) + 1
    pos_order = [str(p) for p in sorted(ts_means['position'].unique())]
    # categorical -> discrete colours; column name -> legend title "position"
    ts_means['position'] = ts_means['position'].astype(str)
    return ts_means, pos_order


# =============================================================================
# Part 2 -- scatter matrices
# =============================================================================

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


# One sequential colormap per position, hue-matched to the tab10 colours used in
# `plot_timestep_matrix_by_position` (tab:blue, orange, green, red, purple, brown,
# pink, gray), so the gradient (light = early trial -> dark = late trial) reads as
# an extension of that same per-position colour coding.
POSITION_CMAPS = ['Blues', 'Oranges', 'Greens', 'Reds',
                  'Purples', 'YlOrBr', 'RdPu', 'Greys']


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


# =============================================================================
# Wrapping functions -- one per notebook part
# =============================================================================

def run_module_pair_correlations(model_name, base_dir=BASE_OUTPUT_DIR):
    """Part 1: 'Pairs of modules - correlations between modules' (cells 1-22).

    Reads the per-sequence correlation scores and plots how each pair's score
    distributes over the generative parameters.
    """
    paths = model_paths(model_name, base_dir)
    correlation_csv = paths['correlation_csv']
    figures_dir = paths['figures_dir']

    if not correlation_csv.exists():
        print(f"  SKIPPED part 1: no correlation table at {correlation_csv}\n"
              f"  (run compute_exp_trials_corr.py for this model first)")
        return None

    act_df = load_correlation_scores(correlation_csv)
    print(f"  loaded {len(act_df)} sequence(s) from {correlation_csv.name}")

    plot_obs_ctx_by_lim_std(act_df, figures_dir)
    plot_obs_ctx_by_d_tau_std(act_df, figures_dir)
    plot_obs_dpos_by_lim_std(act_df, figures_dir)
    plot_ctx_dpos_by_all_params(act_df, figures_dir)
    plot_ctx_dpos_by_lim_std_d(act_df, figures_dir)
    plot_ctx_rule_by_all_params(act_df, figures_dir)
    plot_ctx_rule_by_lim_std_d(act_df, figures_dir)
    plot_dpos_rule_by_all_params(act_df, figures_dir)
    plot_mean_by_lim_std_d(act_df, figures_dir)
    plot_mean_by_tau_std_per_condition(act_df, figures_dir)
    plot_mean_by_all_params(act_df, figures_dir)
    plot_pairs_vs_mean(act_df, figures_dir, "pairs_vs_mean__density.png", density=True)
    plot_pairs_vs_mean(act_df, figures_dir, "pairs_vs_mean__scatter.png", density=False)

    return act_df


def run_module_associations(model_name, base_dir=BASE_OUTPUT_DIR):
    """Part 2: 'Module-to-module associations within and across sequences' (cells 25-34).

    Reads the raw per-sequence activation files and plots the upper-triangle
    module-pair scatter matrices, aggregated per sequence, per timestep, and per
    within-trial position.
    """
    paths = model_paths(model_name, base_dir)
    activations_dir = paths['activations_dir']
    figures_dir = paths['figures_dir']

    seq_files = list_sequence_files(activations_dir)
    if not seq_files:
        print(f"  SKIPPED part 2: no '*_activations.csv' files in {activations_dir}")
        return None
    print(f"  found {len(seq_files)} sequence file(s) in {activations_dir.name}")

    # Per combination of (lim_std, d), then per condition (all params, 16 conditions)
    seq_means = load_sequence_means(seq_files)
    plot_sequence_means_matrix(seq_means, figures_dir, ['lim_std', 'd'],
                               "sequence_matrix__by_lim_std_d.png")
    plot_sequence_means_matrix(seq_means, figures_dir, ['lim_std', 'd', 'tau_std'],
                               "sequence_matrix__by_lim_std_d_tau_std.png")

    # Per timestep, for one sequence only
    plot_first_sequence_matrix(seq_files, figures_dir)

    # Per timestep, averaged over sequences
    ts_means, _ = load_timestep_means(seq_files, with_position=False)
    plot_timestep_mean_matrix(ts_means, figures_dir)

    # Per timestep, averaged over sequences, split by within-trial position
    ts_means_pos, pos_order = load_timestep_means(seq_files, with_position=True)
    plot_timestep_matrix_by_position(ts_means_pos, pos_order, figures_dir)
    plot_timestep_matrix_by_position_gradient(ts_means_pos, pos_order, figures_dir)

    return seq_means


if __name__ == '__main__':

    # =============================================================================
    # Configuration
    # =============================================================================

    # One entry per model to analyse; a single model is just a one-element list.
    MODEL_NAMES = [
        "population_network_all_bn8_trainh0_fixedsir0.02_epochs300_lr0.002",
        "population_network_all_bn8_trainh0_fixedsir0.05_epochs300_lr0.002",
        "population_network_all_bn8_trainh0_fixedsir0.1_epochs300_lr0.002",
    ]
    

    # =============================================================================
    # Run both parts for every model
    # =============================================================================
    for model_name in MODEL_NAMES:
        print(f"\n=== {model_name} ===")
        print("[part 1] Pairs of modules - correlations between modules")
        run_module_pair_correlations(model_name)

        print("[part 2] Module-to-module associations within and across sequences")
        run_module_associations(model_name)

        print(f"Figures written to {model_paths(model_name)['figures_dir']}")
