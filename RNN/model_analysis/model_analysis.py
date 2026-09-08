"""
A script that performs analysis of trained models. It computes:
- log probability (log likelihoods) of tone estimations
- log probability (log likelihoods) of context estimations
- calibration statistics

The models get tested on a test dataset generated in the scope of the script —
by default one per model, from that model's own training data config, so models
trained on different generative parameters (a pinned sigma_r, say) are each
scored in their own regime; SHARED_TEST_DATA puts them all on a single test set
instead. If provided, it also displays the Kalman filter derived likelihoods for
the tone level.

Figures produced (each one is a function, toggled in the SETTINGS block below):
    1. plot_metric_violins()      – one violin panel per metric, all models side
                                    by side (+ one standalone file per metric).
    2. plot_kf_ratio_violins()    – the same observation-level metrics expressed
                                    as a model / Kalman-filter ratio.
    3. plot_calibration_curves()  – per-model KS calibration curve (PIT ECDF).

Usage:
    Edit the SETTINGS block at the bottom of this file (BASE_DIR, OUTPUT_DIR,
    benchmark paths, metrics, which figures to draw), then run:

        python model_analysis.py

    Or import the functions below from a notebook and call them yourself.

Versions:
- was suited for 2-level model (ModuleNetwork), trained/tested
"""


import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import evaluate_models as eval


# Metrics that have a Kalman-filter counterpart. The KF metric dict is keyed with
# these exact names so that model and KF values can be looked up the same way.
KF_METRICS = ('mse', 'obs_loglik', 'ks_statistic')
KF_COLOR = '#9B59B6'   # dashed KF reference line only; violins are not coloured
KF_LABEL = 'Kalman Filter'


# =============================================================================
# Model discovery, evaluation and KF benchmark
# =============================================================================

def resolve_model_dir(model_dir, models_dir=None):
    """Resolve one entry of a model selection.

    Accepts an absolute path, a path relative to the working directory, or the
    bare name of a folder inside `models_dir`.
    """
    model_dir = Path(model_dir)
    if model_dir.is_absolute() or model_dir.exists() or models_dir is None:
        return model_dir
    return Path(models_dir) / model_dir


def load_models_info(models_dir=None, model_dirs=None, verbose=True, strict=False):
    """Load a ModelInfo for each model to compare.

    Pass `models_dir` to take every model sub-folder it contains, and/or
    `model_dirs` to select specific models — either full paths or bare folder
    names looked up inside `models_dir`. A model folder is any directory holding
    a .pth weights file; anything else in `models_dir` is skipped silently (an
    explicitly selected folder is reported instead).

    Folders whose config cannot be read (no config.json and an unrecognised
    directory name, or a config.json written by another version of config_v2)
    are reported and skipped — pass strict=True to get the traceback instead.
    """
    if model_dirs is not None:
        candidates = [resolve_model_dir(d, models_dir) for d in model_dirs]
    elif models_dir is not None:
        candidates = [Path(models_dir) / name for name in sorted(os.listdir(models_dir))]
    else:
        raise ValueError("Pass either models_dir (take every model in it) or model_dirs.")

    models_info = []
    for model_dir in candidates:
        if not model_dir.is_dir() or not list(model_dir.glob('*.pth')):
            if model_dirs is not None:  # explicitly asked for: say why it is dropped
                print(f"  ✗ {model_dir}: skipped (not a model folder with a .pth file)")
            continue
        try:
            models_info.append(eval.ModelInfo.from_path(model_dir))
        except Exception as err:
            if strict:
                raise
            print(f"  ✗ {model_dir.name}: skipped ({err})")
            continue
        if verbose:
            print(f"  ✓ {model_dir.name}")

    if not models_info:
        raise FileNotFoundError(
            f"No readable model folder in {model_dirs if model_dirs is not None else models_dir}")
    return models_info


def pinned_si_r(data_config_dict):
    """The observation noise a config pins, or None when sigma_r is sampled.

    DataConfig.to_gm_dict() encodes si_r_fixed as degenerate si_r_bounds
    (low == high), which is what the GM then draws sigma_r from, so a pinned
    sigma_r shows up here and under no other key.
    """
    bounds = (data_config_dict or {}).get('si_r_bounds') or {}
    low, high = bounds.get('low'), bounds.get('high')
    return low if low is not None and low == high else None


def check_shared_data_config(models_info, shared=False):
    """Report the models that were not all trained on the same data config.

    With `shared`, one test set (the first model's) is used for every model, so
    a mismatch matters twice over: a structural difference (a different number
    of cues, say) makes the other models fail in their forward pass with an
    opaque shape error, and a difference in the generative parameters — a fixed
    sigma_r above all, see pinned_si_r — silently scores them off their training
    regime. Without it every model gets its own test set, and the differences
    are only listed for the record.
    """
    reference = models_info[0].data_config_dict or {}
    ref_si_r = pinned_si_r(reference)
    for info in models_info[1:]:
        other = info.data_config_dict or {}
        differing = [k for k in set(reference) | set(other)
                     if k != 'N_samples' and repr(reference.get(k)) != repr(other.get(k))]
        if not differing:
            continue

        mark = '!' if shared else '-'
        print(f"  {mark} {info.model_dir.name}: data config differs from "
              f"{models_info[0].model_dir.name} on {sorted(differing)}")

        si_r = pinned_si_r(other)
        if 'si_r_bounds' in differing and si_r is not None and ref_si_r is not None:
            if shared:
                print(f"      trained with sigma_r fixed at {si_r:g}, evaluated at "
                      f"{ref_si_r:g} (test set follows the first model)")
            else:
                print(f"      sigma_r fixed at {si_r:g} (vs {ref_si_r:g}); "
                      f"evaluated on its own test set")


def generate_test_set(data_config_dict, n_samples, device='cpu', seed=None):
    """One freshly generated test set.

    With `seed` the generation becomes reproducible: the RNG is reset and the GM
    is forced to run sequentially, because the worker pool it uses otherwise
    (max_cores > 1) forks streams of its own. Two configs that then differ only
    in sigma_r give the same latents — rules, contexts, cues, tau — and differ
    only in the observation noise, which is what makes per-model test sets
    comparable to each other.
    """
    if seed is not None:
        data_config_dict = {**data_config_dict, 'max_cores': 1}
        np.random.seed(seed)
    return eval.generate_test_data(data_config_dict, n_samples=n_samples, device=device)


def build_test_sets(models_info, n_samples, shared=False, benchmark_data=None,
                    device='cpu', seed=None):
    """Test set of every model, as {model folder name: test_data}.

    By default each model gets a set generated from its own training data
    config, so a model trained at a fixed sigma_r is scored at that sigma_r;
    models whose config is identical share one generated set, which makes this
    mode collapse to the shared one when the configs agree.

    `shared` instead generates a single set from the first model's config and
    hands it to all of them — worth it only when the models really were trained
    on the same data and the point is to compare them on the very sequences.

    `benchmark_data` overrides both: the sequences the KF was run on are reused
    for every model, so model and KF metrics are computed on the same data.
    """
    names = [info.model_dir.name for info in models_info]

    if benchmark_data is not None:
        test_data = {
            'y': torch.tensor(benchmark_data.y, dtype=torch.float32).unsqueeze(-1).to(device),
            'y_np': benchmark_data.y,
            'contexts_np': benchmark_data.contexts,
            'pars': benchmark_data.pars,
        }
        return {name: test_data for name in names}

    if shared:
        test_data = generate_test_set(models_info[0].data_config_dict, n_samples,
                                      device=device, seed=seed)
        return {name: test_data for name in names}

    test_sets, by_config = {}, {}
    for info in tqdm(models_info, desc="Generating test data"):
        key = repr(info.data_config_dict)
        if key not in by_config:
            by_config[key] = generate_test_set(info.data_config_dict, n_samples,
                                               device=device, seed=seed)
        test_sets[info.model_dir.name] = by_config[key]
    return test_sets


def evaluate_all_models(models_info, test_sets, min_obs_for_em=None, device='cpu'):
    """Evaluate every model, keeping metrics as per-sample distributions.

    `test_sets` maps a model folder name to the test data that model is
    evaluated on (see build_test_sets); a shared test set is simply the same
    object under every key.

    Returns {model folder name: metrics dict}. `min_obs_for_em` restricts the
    evaluation to the KF window (y[min_obs_for_em:]) so per-sample model metrics
    stay comparable with the KF ones.
    """
    results = {}
    for info in tqdm(models_info, desc="Evaluating models"):
        model = eval.load_model(info, device=device)
        results[info.model_dir.name] = eval.evaluate_model(
            model, info, test_sets[info.model_dir.name], device=device, reduce=False,
            min_obs_for_em=min_obs_for_em,
        )
        del model
    return results


def compute_kf_metrics(benchmark_data):
    """Per-sample KF metrics, keyed like the model metrics (see KF_METRICS)."""
    var_kf = benchmark_data.std_kf ** 2
    y_target = benchmark_data.y[:, benchmark_data.min_obs_for_em:]
    return {
        'mse': eval.compute_mse(y_target, benchmark_data.mu_kf, reduce=False),
        'obs_loglik': eval.compute_log_likelihood(
            y_target, benchmark_data.mu_kf, var_kf, reduce=False),
        'ks_statistic': eval.compute_calibration_ks(
            y_target, benchmark_data.mu_kf, var_kf, reduce=False),
    }


def model_predictions(info, test_data, device='cpu'):
    """Forward pass of one model on the test set.

    Returns (y_target, mu_pred, var_pred), all aligned on y[1:] (the model
    predicts the next observation).
    """
    model = eval.load_model(info, device=device)
    y = test_data['y']
    with torch.no_grad():
        if info.model_type == 'population_network':
            q = test_data.get('q')
            if q is None:
                raise ValueError("PopulationNetwork requires the one-hot cues 'q' in test_data "
                                 "(not available when using benchmark data).")
            model_output = model(y[:, :-1, :], q[:, :-1, :])
        else:
            model_output = model(y[:, :-1, :])
        preds = eval.get_model_predictions(model, model_output)
    del model
    return test_data['y_np'][:, 1:], preds['mu_estim'], preds['var_estim']


# =============================================================================
# Naming, ordering and colours (shared by every figure)
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
    results_ord = {mod: results[mod] for mod in eval.get_desired_order(results)}
    names = make_display_names(results_ord)
    for mod, entry in results_ord.items():
        entry['disp_name'] = names[mod]
    return results_ord


# =============================================================================
# Shared plotting primitives
# =============================================================================

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


# =============================================================================
# Figure 1 – metric distributions across models (violin plots)
# =============================================================================

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
            single_path = output_dir / f"subplot_{metric}_{file_suffix}.png"
            fig_single.savefig(single_path, bbox_inches='tight')
            plt.close(fig_single)

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

    save_path = output_dir / f"model_comparison_metrics_{file_suffix}.png"
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved model comparison plot to: {save_path}")
    return save_path


# =============================================================================
# Figure 2 – model / KF ratio for observation-level metrics
# =============================================================================

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

    save_path = output_dir / "model_comparison_ratio_vs_kf.png"
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved model comparison ratio plot to: {save_path}")
    return save_path


# =============================================================================
# Figure 3 – KS calibration curves (one file per model)
# =============================================================================

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
        eval.plot_calibration_curve(
            y_target, mu_pred, var_pred,
            save_path=save_path,
            title=f"KS Calibration – {entry['disp_name']}",
        )
        saved.append(save_path)
    return saved


# =============================================================================
# Script entry point — edit the settings below, then: python model_analysis.py
# =============================================================================

if __name__ == "__main__":

    # ------------------------- SETTINGS (edit these) -------------------------
    RNN_DIR = Path(__file__).resolve().parent.parent   # .../RNN_paradigm/RNN

    # Models to compare: every sub-folder of BASE_DIR holding a .pth file.
    BASE_DIR = RNN_DIR / 'training_results/N_ctx_2/HierarchicalGM'
    # BASE_DIR = RNN_DIR / 'training_results/N_ctx_2/NonHierarchicalGM'

    # Restrict the comparison to specific models. If MODEL_DIRS is not None it
    # takes precedence over BASE_DIR; entries are full paths or bare folder
    # names looked up inside BASE_DIR.
    # MODEL_DIRS = None
    MODEL_DIRS = [
        BASE_DIR / "population_network_all_bn8_trainh0_fixedsir0.02_epochs300_lr0.002",
        BASE_DIR / "population_network_all_bn8_trainh0_fixedsir0.05_epochs300_lr0.002",
        BASE_DIR / "population_network_all_bn8_trainh0_fixedsir0.1_epochs300_lr0.002",
    ]
    

    OUTPUT_DIR = RNN_DIR / 'evaluation_results/model_comparison/HierarchicalGM'
    # OUTPUT_DIR = RNN_DIR / 'evaluation_results/model_comparison/NonHierarchicalGM'

    # Kalman-filter benchmark. When True, the test set AND the KF predictions are
    # loaded from BENCHMARK_PATH (instead of generating a fresh test set), and the
    # KF is overlaid on the figures.
    USE_BENCHMARK_DATA = False
    BENCHMARK_PATH = RNN_DIR / 'benchmarks/N_ctx_2/NonHierarchicalGM/benchmarks_1000_test.pkl'

    # Test data (ignored when USE_BENCHMARK_DATA is True).
    # SHARED_TEST_DATA=False generates one test set per model, from that model's
    # own training data config, so a model trained with sigma_r pinned to some
    # value is scored at that value instead of the first model's. Models sharing
    # a config still share one generated set. Set it True to force the single
    # first-model test set — every model on the very same sequences, only sound
    # when they really were trained on the same data config.
    SHARED_TEST_DATA = False
    # Seed of the test data generation. Keeps runs reproducible and, per model,
    # pairs the sets: configs differing only in sigma_r yield the same latents
    # (rules, contexts, cues, tau), so the comparison is not blurred by sampling
    # noise. None leaves the RNG alone and lets the GM generate in parallel.
    TEST_SEED = 0
    N_SAMPLES_TEST = 1000
    DEVICE = 'cpu'

    # One violin panel per metric. Available keys depend on the model type:
    # 'mse', 'obs_loglik', 'ks_statistic' (all models), 'context_accuracy',
    # 'context_loglik' (n_ctx > 1), 'dpos_loglik', 'dpos_accuracy',
    # 'rule_loglik', 'rule_accuracy' (population_network). Missing ones are
    # silently skipped per model.
    METRICS = ['obs_loglik', 'context_loglik', 'dpos_loglik', 'rule_loglik']
    # METRICS = ['mse', 'obs_loglik', 'ks_statistic', 'context_accuracy', 'context_loglik']

    # Which figures to produce.
    PLOT_METRIC_VIOLINS = True
    PLOT_KF_RATIOS = True       # only drawn when USE_BENCHMARK_DATA is True
    PLOT_CALIBRATION = False
    SAVE_INDIVIDUAL_PANELS = True   # one extra file per metric of figure 1
    ROW_LABELS = None               # e.g. ['tone process estimation', 'context estimation']
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_suffix = 'kfbm' if USE_BENCHMARK_DATA else 'test'

    print(f"\nLoading models from {BASE_DIR}")
    models_info = load_models_info(BASE_DIR, model_dirs=MODEL_DIRS)

    # Every model that disagrees with the first one's data config is listed.
    check_shared_data_config(models_info, shared=SHARED_TEST_DATA)

    benchmark_data = eval.load_benchmark_data(BENCHMARK_PATH) if USE_BENCHMARK_DATA else None
    eval_min_obs = benchmark_data.min_obs_for_em if USE_BENCHMARK_DATA else None
    test_sets = build_test_sets(models_info, N_SAMPLES_TEST, shared=SHARED_TEST_DATA,
                                benchmark_data=benchmark_data, device=DEVICE,
                                seed=TEST_SEED)

    results = evaluate_all_models(models_info, test_sets,
                                  min_obs_for_em=eval_min_obs, device=DEVICE)
    kf_metrics = compute_kf_metrics(benchmark_data) if USE_BENCHMARK_DATA else None
    results_ord = style_results(results)

    if PLOT_METRIC_VIOLINS:
        plot_metric_violins(results_ord, METRICS, OUTPUT_DIR, kf_metrics=kf_metrics,
                            file_suffix=file_suffix, row_labels=ROW_LABELS,
                            save_individual=SAVE_INDIVIDUAL_PANELS)

    if PLOT_KF_RATIOS and kf_metrics is not None:
        plot_kf_ratio_violins(results_ord, kf_metrics, OUTPUT_DIR)

    if PLOT_CALIBRATION:
        plot_calibration_curves(models_info, results_ord, test_sets, OUTPUT_DIR,
                                device=DEVICE, min_obs_for_em=eval_min_obs)
