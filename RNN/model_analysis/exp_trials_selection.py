"""
Summarise the likelihoods a trained model assigns at deviant positions, and show
how they are distributed across deviant positions.

For every model listed in ``MODEL_NAMES`` this script reads all
*_probabilities_deviant.csv files produced by model_prob_exp_trials.py and saves
per-file averages of the likelihood columns to a sub-folder called "average"
inside that model's deviant output directory:

  - <model>_likelihood_averages.csv          one row per input file
  - <model>_likelihood_averages_by_dpos.csv  one row per (input file, dpos)

Each row carries the source filename, the per-file scalar parameters (lim_std,
d, tau_std) and the mean of every likelihood column (lik_obs, lik_ctx, lik_dpos,
lik_rule, cdf_lik_obs, log_lik_obs).

One figure is then produced per model, with one subplot per module / kind of
likelihood and the deviant positions overlaid as seaborn ``hue`` histograms.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

# Columns summarised in the CSVs.
LIKELIHOOD_COLS = ["lik_obs", "cdf_lik_obs", "log_lik_obs", "lik_ctx", "lik_dpos", "lik_rule"]
PARAM_COLS      = ["lim_std", "d", "tau_std"]

# Subset drawn in the figures: cdf_lik_obs stays in the CSVs but is not plotted.
PLOT_LIKELIHOOD_COLS = ["lik_obs", "log_lik_obs", "lik_ctx", "lik_dpos", "lik_rule"]

# (module, kind of likelihood) shown in each subplot title. Ordering the subplots
# after PLOT_LIKELIHOOD_COLS keeps the obs-based quantities side by side.
LIKELIHOOD_LABELS = {
    "lik_obs":     ("obs",  "Gaussian likelihood"),
    "cdf_lik_obs": ("obs",  "Gaussian CDF"),
    "log_lik_obs": ("obs",  "log-likelihood"),
    "lik_ctx":     ("ctx",  "P(true class)"),
    "lik_dpos":    ("dpos", "P(true class)"),
    "lik_rule":    ("rule", "P(true class)"),
}


# ── Reading and summarising ───────────────────────────────────────────────────
def add_derived_likelihoods(df):
    """Add the likelihood columns model_prob_exp_trials.py does not store."""
    df["cdf_lik_obs"] = scipy.stats.norm.cdf(
        (df["observation"] - df["obs_mean"]) / np.sqrt(df["obs_var"])
    )
    df["log_lik_obs"] = np.log(df["lik_obs"] + 1e-8)  # Add small constant to avoid log(0)
    return df


def summarise_deviant_files(deviant_path, model_name, group_col=None):
    """Likelihood averages over one model's *_probabilities_deviant.csv files.

    Without `group_col`, one row per file. With it (e.g. 'dpos'), one row per
    (file, value of that column) so the averages can be split by deviant
    position — the file's rows are spread over dpos 2..6, so a per-file mean
    would mix them.

    Returns an empty DataFrame (with a warning) when the model has no deviant
    files, so a partially processed model list still yields the other figures.
    """
    deviant_files = sorted(deviant_path.glob("*_probabilities_deviant.csv"))
    if not deviant_files:
        print(f"  Warning: no *_probabilities_deviant.csv files in {deviant_path}; skipping.")
        return pd.DataFrame()

    print(f"Found {len(deviant_files)} deviant file(s) in {deviant_path}")

    rows = []
    for f in deviant_files:
        df = add_derived_likelihoods(pd.read_csv(f))

        base = {"model": model_name, "file": f.name}
        for col in PARAM_COLS:
            if col in df.columns:
                base[col] = df[col].iloc[0]

        # A single group holding every row reproduces the per-file average.
        groups = df.groupby(group_col, sort=True) if group_col else [(None, df)]
        for key, sub in groups:
            row = dict(base)
            if group_col:
                row[group_col] = key
            row["n_rows"] = len(sub)
            for col in LIKELIHOOD_COLS:
                # .mean() skips NaN: rows whose prediction is undefined are
                # stored as NaN and must not be counted as zeros.
                row[f"mean_{col}"] = sub[col].mean() if col in sub.columns else float("nan")
            rows.append(row)

    return pd.DataFrame(rows)


def write_summary_csv(summary_df, deviant_path, model_name, group_col=None):
    """Save one model's likelihood averages next to its deviant files."""
    average_path = deviant_path / "average"
    average_path.mkdir(parents=True, exist_ok=True)

    suffix = f"_by_{group_col}" if group_col else ""
    out_file = average_path / f"{model_name}_likelihood_averages{suffix}.csv"
    summary_df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")
    return out_file


def build_likelihood_summaries(base_output_root, model_names, group_col=None, save_csv=True):
    """{model name: likelihood averages}, one entry per model that has data.

    Models without deviant files are reported and left out; a FileNotFoundError
    is raised only when none of them has any.
    """
    summaries = {}
    for model_name in model_names:
        deviant_path = base_output_root / model_name / "probabilities_deviant"
        summary_df = summarise_deviant_files(deviant_path, model_name, group_col=group_col)
        if summary_df.empty:
            continue
        if save_csv:
            write_summary_csv(summary_df, deviant_path, model_name, group_col=group_col)
        summaries[model_name] = summary_df

    if not summaries:
        raise FileNotFoundError(
            f"No *_probabilities_deviant.csv files found for any of {model_names} "
            f"under {base_output_root}"
        )
    return summaries


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

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {out_png}")

    return fig


def plot_likelihood_distributions_per_model(summaries, figure_dir, hue="dpos", **kwargs):
    """One figure per model, saved as <model>_likelihood_distributions_by_<hue>.png."""
    figure_dir = Path(figure_dir)
    saved = []
    for model_name, summary_df in summaries.items():
        out_png = figure_dir / f"{model_name}_likelihood_distributions_by_{hue}.png"
        fig = plot_likelihood_distributions(
            summary_df, out_png=out_png, hue=hue,
            title=f"Per-sequence mean likelihood at deviant positions\n{model_name}",
            **kwargs,
        )
        plt.close(fig)
        saved.append(out_png)
    return saved


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Models to summarise; each must own an
    # exp_seq_act_output/<model>/probabilities_deviant folder filled by
    # model_prob_exp_trials.py. Kept in sync with MODEL_DIRS in model_analysis.py.
    MODEL_NAMES = [
        "population_network_all_bn8_trainh0_fixedsir0.02_epochs300_lr0.002",
        "population_network_all_bn8_trainh0_fixedsir0.05_epochs300_lr0.002",
        "population_network_all_bn8_trainh0_fixedsir0.1_epochs300_lr0.002",
    ]

    # Column whose values are overlaid as hue inside each subplot.
    HUE_COL = "dpos"

    base_output_root = Path(
        "/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output"
    )
    figure_dir = base_output_root / "likelihood_distributions"

    # Per-file averages (the original summary), then the same split by deviant
    # position, which is what the figures show.
    build_likelihood_summaries(base_output_root, MODEL_NAMES)
    dpos_summaries = build_likelihood_summaries(base_output_root, MODEL_NAMES,
                                                group_col=HUE_COL)

    plot_likelihood_distributions_per_model(dpos_summaries, figure_dir, hue=HUE_COL)
