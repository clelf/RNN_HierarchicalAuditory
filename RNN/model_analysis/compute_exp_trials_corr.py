from math import ceil
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_activations import compute_pairwise_module_correlations

# Module names as stored in the activation CSV files (see model_act_exp_trials.py)
MODULES = ['obs', 'ctx', 'dpos', 'rule']


def load_module_dict(df, suffix):
    """Build {module_name: 1D array} from columns named '<module>_<suffix>'.

    Rows with NaN (e.g. the padded last derivative timestep) are dropped so all
    modules keep the same length.
    """
    cols = [f'{m}_{suffix}' for m in MODULES]
    values = df[cols].dropna()
    return {m: values[f'{m}_{suffix}'].to_numpy() for m in MODULES}


def aggregate_scores(pair_correlations):
    """Three ways to summarise the per-pair correlations into one score.

    - mean_abs: mean of the absolute per-pair correlations (coupling strength;
      anti-correlated pairs still count).
    - abs_mean: absolute value of the mean of the signed correlations (lets
      opposite-sign pairs cancel, then drops the sign).
    - mean:     plain mean of the signed correlations (keeps the sign).
    """
    values = np.array(list(pair_correlations.values()))
    return {
        'mean_abs': float(np.mean(np.abs(values))),
        'abs_mean': float(abs(np.mean(values))),
        'mean': float(np.mean(values)),
    }


def plot_score_histograms(df, output_path, bins=30):
    """Plot one histogram per score column of `df`, as subplots in one figure.

    Every column except 'sequence_name' is treated as a score column; each gets
    its own histogram over all sequences.
    """
    score_cols = [c for c in df.columns if ('norm' in c or 'deriv' in c)]
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
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved histogram figure: {output_path}")


if __name__ == '__main__':

    # Specify the path to the directory containing already computed model activations files for exp trials sequences
    # activations_dir = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output/population_network_all_bn8_lr0/activations")
    activations_dir = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output/population_network_all_bn8_lr0.001_dposweight/activations")

    # Loop through the .CSV sequences. For each sequence, compute the Pearson correlation
    # between every pair of modules (excluding self-pairs) for both the activities and the
    # derivatives, store each per-pair score, and store three aggregate scores per quantity.
    results = []
    for csv_file in sorted(activations_dir.glob("*.csv")):
        df = pd.read_csv(csv_file)

        # Activities: the '*_norm' columns; derivatives are already stored in '*_deriv' columns.
        activity_dict = load_module_dict(df, 'norm')
        deriv_dict = load_module_dict(df, 'deriv')

        row = {'sequence_name': csv_file.stem,
               'lim_std': df['lim_std'].iloc[0],
               'd': df['d'].iloc[0],
               'tau_std': df['tau_std'].iloc[0]
        }
        for kind, module_dict in [('activity', activity_dict), ('derivative', deriv_dict)]:
            pair_correlations = compute_pairwise_module_correlations(module_dict, use_derivatives=False)
            # Per-pair scores, e.g. 'activity_obs_ctx', 'derivative_dpos_rule'
            for pair_name, corr in pair_correlations.items():
                row[f'{kind}_{pair_name}'] = corr
            # Aggregate scores, e.g. 'activity_mean_abs', 'activity_abs_mean', 'activity_mean'
            for agg_name, agg_val in aggregate_scores(pair_correlations).items():
                row[f'{kind}_{agg_name}'] = agg_val

        results.append(row)
        print(f"  {csv_file.name}: "
              f"activity_mean_abs={row['activity_mean_abs']:.4f}, "
              f"derivative_mean_abs={row['derivative_mean_abs']:.4f}")

    out_df = pd.DataFrame(results)
    out_file = activations_dir / "correlation_scores/exp_trials_correlations.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_file, index=False)
    print(f"\n{len(results)} sequence(s) processed. Saved: {out_file}")

    # Histogram of every score column across sequences
    plot_score_histograms(out_df, out_file.with_name("score_histograms.png"))
    pass
