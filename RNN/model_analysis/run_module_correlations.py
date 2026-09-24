"""Module-pair correlations on the experimental sequences: scores, then figures.

Three stages, each toggled in the SETTINGS block:

  scores        for every sequence, the Pearson correlation between each pair of
                modules (activities and derivatives) plus three aggregate scores
                -> <model>/activations/correlation_scores/exp_trials_correlations.csv
                -> .../score_histograms.png
  pair figures  how each pair's score distributes over the generative parameters
                lim_std / d / tau_std
                -> <model>/module_correlations/*.png
  associations  upper-triangle scatter matrices of the raw module activities,
                aggregated per sequence, per timestep, and per within-trial
                position
                -> <model>/module_correlations/*.png

Replaces compute_exp_trials_corr.py and module_correlations.py, which were the
two halves of one handoff: the first wrote the score table, the second read it.
Running them together means a model can never be plotted from a stale table.
"""

import pandas as pd

import analysis_config as cfg
import exp_sequence_analysis as exp
import plots
from analysis_core import model_sequence_csvs, outputs_up_to_date

# Figures each figure stage writes to <model>/module_correlations/ (their names
# are set inside plots.py), used to tell whether the stage is up to date.
PAIR_FIGURES = [
    'obs_ctx__by_lim_std.png', 'obs_ctx__by_d_tau_std.png', 'obs_dpos__by_lim_std.png',
    'ctx_dpos__by_lim_std_d_tau_std.png', 'ctx_dpos__by_lim_std_d.png',
    'ctx_rule__by_lim_std_d_tau_std.png', 'ctx_rule__by_lim_std_d.png',
    'dpos_rule__by_lim_std_d_tau_std.png', 'mean__by_lim_std_d.png',
    'mean__by_tau_std_per_lim_std_d.png', 'mean__by_lim_std_d_tau_std.png',
    'pairs_vs_mean__density.png', 'pairs_vs_mean__scatter.png',
]
ASSOCIATION_FIGURES = [
    'sequence_matrix__by_lim_std_d.png', 'sequence_matrix__by_lim_std_d_tau_std.png',
    'timestep_matrix__first_sequence.png', 'timestep_matrix__mean_over_sequences.png',
    'timestep_matrix__by_position.png', 'timestep_matrix__by_position_gradient.png',
]


def build_correlation_scores(activations_dir):
    """One row per sequence: every module-pair correlation plus the aggregates.

    Both the activities ('*_norm') and the derivatives ('*_deriv') are scored,
    giving columns such as 'activity_obs_ctx' and 'derivative_dpos_rule'
    alongside 'activity_mean_abs' / 'activity_abs_mean' / 'activity_mean'.
    """
    rows = []
    for csv_file in sorted(activations_dir.glob("*.csv")):
        df = pd.read_csv(csv_file)
        row = {
            'sequence_name': csv_file.stem,
            'lim_std': df['lim_std'].iloc[0],
            'd': df['d'].iloc[0],
            'tau_std': df['tau_std'].iloc[0],
        }
        for kind, suffix in [('activity', 'norm'), ('derivative', 'deriv')]:
            module_dict = exp.load_module_dict(df, suffix)
            pair_correlations = exp.compute_pairwise_module_correlations(
                module_dict, use_derivatives=False)
            for pair_name, corr in pair_correlations.items():
                row[f'{kind}_{pair_name}'] = corr
            for agg_name, agg_val in exp.aggregate_scores(pair_correlations).items():
                row[f'{kind}_{agg_name}'] = agg_val
        rows.append(row)
        print(f"  {csv_file.name}: "
              f"activity_mean_abs={row['activity_mean_abs']:.4f}, "
              f"derivative_mean_abs={row['derivative_mean_abs']:.4f}")
    return pd.DataFrame(rows)


def draw_pair_correlation_figures(act_df, figures_dir):
    """Every distribution figure of the module-pair correlation scores."""
    plots.plot_obs_ctx_by_lim_std(act_df, figures_dir)
    plots.plot_obs_ctx_by_d_tau_std(act_df, figures_dir)
    plots.plot_obs_dpos_by_lim_std(act_df, figures_dir)
    plots.plot_ctx_dpos_by_all_params(act_df, figures_dir)
    plots.plot_ctx_dpos_by_lim_std_d(act_df, figures_dir)
    plots.plot_ctx_rule_by_all_params(act_df, figures_dir)
    plots.plot_ctx_rule_by_lim_std_d(act_df, figures_dir)
    plots.plot_dpos_rule_by_all_params(act_df, figures_dir)
    plots.plot_mean_by_lim_std_d(act_df, figures_dir)
    plots.plot_mean_by_tau_std_per_condition(act_df, figures_dir)
    plots.plot_mean_by_all_params(act_df, figures_dir)
    plots.plot_pairs_vs_mean(act_df, figures_dir, "pairs_vs_mean__density.png",
                             density=True)
    plots.plot_pairs_vs_mean(act_df, figures_dir, "pairs_vs_mean__scatter.png",
                             density=False)


def draw_association_figures(seq_files, figures_dir):
    """Upper-triangle module-pair scatter matrices, at three levels of aggregation."""
    # Per combination of (lim_std, d), then per condition (all params)
    seq_means = exp.load_sequence_means(seq_files)
    plots.plot_sequence_means_matrix(seq_means, figures_dir, ['lim_std', 'd'],
                                     "sequence_matrix__by_lim_std_d.png")
    plots.plot_sequence_means_matrix(seq_means, figures_dir,
                                     ['lim_std', 'd', 'tau_std'],
                                     "sequence_matrix__by_lim_std_d_tau_std.png")

    # Per timestep, for one sequence only
    plots.plot_first_sequence_matrix(seq_files, figures_dir)

    # Per timestep, averaged over sequences
    ts_means, _ = exp.load_timestep_means(seq_files, with_position=False)
    plots.plot_timestep_mean_matrix(ts_means, figures_dir)

    # Per timestep, averaged over sequences, split by within-trial position
    ts_means_pos, pos_order = exp.load_timestep_means(seq_files, with_position=True)
    plots.plot_timestep_matrix_by_position(ts_means_pos, pos_order, figures_dir)
    plots.plot_timestep_matrix_by_position_gradient(ts_means_pos, pos_order, figures_dir)


if __name__ == '__main__':
    plots.set_script_path(__file__)

    # ------------------------- SETTINGS (edit these) -------------------------
    # Each model must own an <output-root>/<model>/activations folder filled by
    # run_exp_trials_pipeline.py.
    MODEL_NAMES = cfg.FIXED_SIGMA_R_MODELS
    OUTPUT_ROOT = cfg.EXP_SEQ_OUTPUT_ROOT

    RUN_SCORES = True
    RUN_PAIR_FIGURES = True
    RUN_ASSOCIATIONS = True

    # A stage is skipped when all its outputs exist and are newer than its inputs.
    # Set True to redo every stage regardless.
    OVERWRITE = False
    # -------------------------------------------------------------------------

    for model_name in MODEL_NAMES:
        print(f"\n{'=' * 79}\n{model_name}\n{'=' * 79}")
        paths = exp.model_paths(model_name, OUTPUT_ROOT)
        activations_dir = paths['activations_dir']
        correlation_csv = paths['correlation_csv']
        figures_dir = paths['figures_dir']

        if not activations_dir.is_dir():
            print(f"  SKIPPED: no activations folder at {activations_dir}")
            continue
        activation_csvs = model_sequence_csvs(OUTPUT_ROOT, model_name, 'activations')
        score_outputs = [correlation_csv, correlation_csv.parent / "score_histograms.png"]

        if RUN_SCORES and not OVERWRITE and outputs_up_to_date(score_outputs, activation_csvs):
            print(f"[scores] up to date, skipped: {correlation_csv}")
        elif RUN_SCORES:
            print("[scores] per-sequence module-pair correlations")
            scores = build_correlation_scores(activations_dir)
            if scores.empty:
                print(f"  SKIPPED: no '*.csv' files in {activations_dir}")
            else:
                correlation_csv.parent.mkdir(parents=True, exist_ok=True)
                scores.to_csv(correlation_csv, index=False)
                print(f"  {len(scores)} sequence(s) -> {correlation_csv}")
                fig = plots.plot_score_histograms(scores)
                plots.save_figure(fig, correlation_csv.parent, "score_histograms.png")

        pair_outputs = [figures_dir / name for name in PAIR_FIGURES]
        if (RUN_PAIR_FIGURES and not OVERWRITE and correlation_csv.exists()
                and outputs_up_to_date(pair_outputs, [correlation_csv])):
            print("[pair figures] up to date, skipped")
        elif RUN_PAIR_FIGURES:
            print("[pair figures] score distributions over the generative parameters")
            if not correlation_csv.exists():
                print(f"  SKIPPED: no correlation table at {correlation_csv}")
            else:
                act_df = exp.load_correlation_scores(correlation_csv)
                print(f"  loaded {len(act_df)} sequence(s) from {correlation_csv.name}")
                draw_pair_correlation_figures(act_df, figures_dir)

        association_outputs = [figures_dir / name for name in ASSOCIATION_FIGURES]
        if (RUN_ASSOCIATIONS and not OVERWRITE
                and outputs_up_to_date(association_outputs, activation_csvs)):
            print("[associations] up to date, skipped")
        elif RUN_ASSOCIATIONS:
            print("[associations] module-to-module scatter matrices")
            seq_files = exp.list_sequence_files(activations_dir)
            if not seq_files:
                print(f"  SKIPPED: no '*_activations.csv' files in {activations_dir}")
            else:
                print(f"  found {len(seq_files)} sequence file(s)")
                draw_association_figures(seq_files, figures_dir)

        print(f"Figures written to {figures_dir}")
