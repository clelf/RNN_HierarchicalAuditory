"""
Study the relationship between per-module activity at the deviant position and
the likelihood the model assigned to that deviant.

It joins, per trial, the two sets of files produced earlier:
  - activations_deviant/*_deviant_trial.csv  (model_act_exp_trials_deviant.py)
        one row per trial: <module>_norm sampled at the trial's deviant timestep.
  - probabilities/*_probabilities.csv        (model_prob_exp_trials.py)
        the deviant row (ctx == 1) carries lik_<module>, the likelihood of the
        deviant under each module's predicted distribution.

The two are merged on trial_n (one deviant per trial). We then look at how the
module activity at the deviant — both per module and averaged across modules
(``mean_norm``) — covaries with the deviant likelihoods, via scatter grids and
correlation heatmaps pooled over all sequences.

Outputs (under exp_seq_act_output/<model>/act_likelihood_join/):
  - deviant_activity_likelihood_joined.csv : the pooled per-trial merged table.
  - scatter_activity_vs_likelihood.png     : activity (cols) vs likelihood (rows).
  - corr_heatmap_pearson.png / _spearman.png : correlation matrices.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from model_activations import (
    MODULE_NAMES,
    load_deviant_activity,
    load_deviant_likelihoods,
    join_activity_likelihood,
    compute_activity_likelihood_correlations,
    plot_activity_likelihood_grid,
    plot_correlation_heatmap,
)


if __name__ == '__main__':
    model_name = "population_network_all_bn8_lr0"
    base_output_path = Path(
        "/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output"
    ) / model_name

    activations_dir = base_output_path / 'activations_deviant'
    probabilities_dir = base_output_path / 'probabilities'
    output_path = base_output_path / 'act_likelihood_join'
    output_path.mkdir(parents=True, exist_ok=True)

    # Columns to study.
    activity_cols = [f'{m}_norm' for m in MODULE_NAMES] + ['mean_norm']
    activity_labels = [f'{m} act.' for m in MODULE_NAMES] + ['mean act.']
    likelihood_cols = [f'lik_{m}' for m in MODULE_NAMES]
    likelihood_labels = [f'lik {m}' for m in MODULE_NAMES]

    # Pair files by their original sequence stem (the part before the suffix that
    # each upstream script appended).
    ACT_SUFFIX = '_deviant_trial'
    act_files = sorted(activations_dir.glob(f'*{ACT_SUFFIX}.csv'))
    print(f"Found {len(act_files)} deviant-activity files in {activations_dir}")

    merged_frames = []
    n_missing = 0
    for act_file in act_files:
        stem = act_file.stem[: -len(ACT_SUFFIX)]  # original sequence stem
        prob_file = probabilities_dir / f'{stem}_probabilities.csv'
        if not prob_file.exists():
            n_missing += 1
            print(f"  [skip] no probabilities file for {stem}")
            continue

        act_df = load_deviant_activity(act_file)
        lik_df = load_deviant_likelihoods(prob_file, module_names=MODULE_NAMES)
        merged = join_activity_likelihood(act_df, lik_df, module_names=MODULE_NAMES)
        merged.insert(0, 'source', stem)
        merged_frames.append(merged)

    if not merged_frames:
        raise SystemExit("No files could be joined; nothing to analyse.")

    data = pd.concat(merged_frames, ignore_index=True)
    print(f"Joined {len(merged_frames)} file(s) "
          f"({n_missing} skipped) -> {len(data)} deviant trials pooled")

    joined_csv = output_path / 'deviant_activity_likelihood_joined.csv'
    data.to_csv(joined_csv, index=False)
    print(f"  Saved: {joined_csv.name}")

    # --- Scatter grid: activity (columns) vs likelihood (rows) ---
    fig = plot_activity_likelihood_grid(
        data, activity_cols, likelihood_cols,
        activity_labels=activity_labels, likelihood_labels=likelihood_labels,
    )
    fig.suptitle('Module activity at the deviant vs deviant likelihood',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    scatter_file = output_path / 'scatter_activity_vs_likelihood.png'
    fig.savefig(scatter_file, dpi=150)
    plt.close(fig)
    print(f"  Saved: {scatter_file.name}")

    # --- Correlation heatmaps (pooled over all trials) ---
    for method, fname in [('pearson', 'corr_heatmap_pearson.png'),
                          ('spearman', 'corr_heatmap_spearman.png')]:
        corr = compute_activity_likelihood_correlations(
            data, activity_cols, likelihood_cols, method=method
        )
        corr.index = activity_labels
        corr.columns = likelihood_labels
        fig = plot_correlation_heatmap(
            corr, title=f'{method.capitalize()} correlation: activity vs likelihood'
        )
        heatmap_file = output_path / fname
        fig.savefig(heatmap_file, dpi=150)
        plt.close(fig)
        print(f"  Saved: {heatmap_file.name}")
        print(f"\n{method} correlations (activity x likelihood):")
        print(corr.round(3).to_string())

    print(f"\nDone. Results in {output_path}")
