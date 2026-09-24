"""Summarise the likelihoods a model assigns at deviant positions.

Three analyses over the ``probabilities_deviant/*.csv`` files that
run_exp_trials_pipeline.py writes, each toggled in the SETTINGS block:

  likelihood averages   one row per sequence (and per deviant position) holding
                        the mean of every likelihood column
                        -> <model>/probabilities_deviant/average/*.csv
                        -> <output-root>/likelihood_distributions/*.png
  dpos probability      the per-sequence mean probability assigned to the TRUE
                        deviant-position class, with the wider generative
                        parameters joined back from the original sequence files
                        -> <model>/probabilities_deviant_avg/*.csv + hist .png
  activity join         per trial, module activity at the deviant against the
                        likelihood the model gave that deviant
                        -> <model>/act_likelihood_join/*.csv + scatter/heatmaps

Replaces exp_trials_selection.py, dpos_prob_at_deviant_distribution.py and
exp_trial_join_act_proba.py. The first two ran the same averaging over the same
files and differed only in which columns they averaged and where the sequence
parameters came from; both are now one call to
exp_sequence_analysis.summarise_deviant_files.
"""

import pandas as pd

import analysis_config as cfg
import exp_sequence_analysis as exp
import plots
from analysis_core import model_sequence_csvs, outputs_up_to_date


if __name__ == '__main__':
    plots.set_script_path(__file__)

    # ------------------------- SETTINGS (edit these) -------------------------
    # Each model must own an <output-root>/<model>/probabilities_deviant folder
    # filled by run_exp_trials_pipeline.py; the others are reported and skipped.
    # MODEL_NAMES = cfg.FIXED_SIGMA_R_MODELS
    MODEL_NAMES = cfg.EVALUATION_MODELS
    OUTPUT_ROOT = cfg.EXP_SEQ_OUTPUT_ROOT

    # Column whose values are overlaid as hue inside each subplot.
    HUE_COL = 'dpos'

    RUN_LIKELIHOOD_AVERAGES = True
    RUN_DPOS_PROBABILITY = True
    RUN_ACTIVITY_JOIN = True

    # A model's stage is skipped when all its outputs exist and are newer than the
    # CSVs it reads. Set True to redo every stage regardless (e.g. after changing
    # the plotting code or the sequence files in TRIALS_PATH).
    OVERWRITE = False
    # -------------------------------------------------------------------------

    if RUN_LIKELIHOOD_AVERAGES:
        print(f"\n{'=' * 79}\nLikelihood averages\n{'=' * 79}")
        todo = []
        for model_name in MODEL_NAMES:
            average_dir = OUTPUT_ROOT / model_name / 'probabilities_deviant' / 'average'
            outputs = [
                average_dir / f"{model_name}_likelihood_averages.csv",
                average_dir / f"{model_name}_likelihood_averages_by_{HUE_COL}.csv",
                OUTPUT_ROOT / 'likelihood_distributions'
                / f"{model_name}_likelihood_distributions_by_{HUE_COL}.png",
            ]
            sources = model_sequence_csvs(OUTPUT_ROOT, model_name, 'probabilities_deviant')
            if not OVERWRITE and outputs_up_to_date(outputs, sources):
                print(f"=== {model_name}: up to date, skipped")
            else:
                todo.append(model_name)

        if todo:
            # Per-file averages (the original summary), then the same split by
            # deviant position, which is what the figures show.
            exp.build_likelihood_summaries(OUTPUT_ROOT, todo)
            dpos_summaries = exp.build_likelihood_summaries(
                OUTPUT_ROOT, todo, group_col=HUE_COL)

            plots.plot_likelihood_distributions_per_model(
                dpos_summaries, OUTPUT_ROOT / 'likelihood_distributions', hue=HUE_COL)

    if RUN_DPOS_PROBABILITY:
        print(f"\n{'=' * 79}\ndpos probability at the true deviant\n{'=' * 79}")
        found = []
        for model_name in MODEL_NAMES:
            print(f"\n=== {model_name}")
            model_root = OUTPUT_ROOT / model_name
            avg_path = model_root / 'probabilities_deviant_avg'
            outputs = [avg_path / 'mean_dpos_prob_per_sequence.csv',
                       avg_path / 'mean_dpos_prob_per_sequence_hist.png']
            sources = model_sequence_csvs(OUTPUT_ROOT, model_name, 'probabilities_deviant')
            if not OVERWRITE and outputs_up_to_date(outputs, sources):
                print("  up to date, skipped")
                found.append(model_name)
                continue
            # trials_path joins the wider generative parameters (sigma_r, run_n,
            # ...) that the extraction does not copy into the deviant files.
            avg_df = exp.summarise_deviant_files(
                model_root / 'probabilities_deviant', model_name,
                value_cols=['lik_dpos'], trials_path=cfg.TRIALS_PATH)
            if avg_df.empty:
                continue
            found.append(model_name)

            avg_path.mkdir(parents=True, exist_ok=True)
            avg_df.to_csv(outputs[0], index=False)
            print(f"Saved per-sequence means: {outputs[0]}")

            fig = plots.plot_dpos_prob_hist(avg_df, model_name, 'mean_lik_dpos')
            plots.save_figure(fig, avg_path, outputs[1].name)

        skipped = [m for m in MODEL_NAMES if m not in found]
        if skipped:
            print(f"\nSkipped {len(skipped)} model(s) with no deviant probability "
                  f"files: {', '.join(skipped)}")

    if RUN_ACTIVITY_JOIN:
        print(f"\n{'=' * 79}\nActivity against likelihood at the deviant\n{'=' * 79}")
        activity_cols = [f'{m}_norm' for m in cfg.MODULE_NAMES] + ['mean_norm']
        activity_labels = [f'{m} act.' for m in cfg.MODULE_NAMES] + ['mean act.']
        likelihood_cols = [f'lik_{m}' for m in cfg.MODULE_NAMES]
        likelihood_labels = [f'lik {m}' for m in cfg.MODULE_NAMES]

        for model_name in MODEL_NAMES:
            print(f"\n=== {model_name}")
            model_root = OUTPUT_ROOT / model_name
            act_dir = model_root / 'activations_deviant'
            prob_dir = model_root / 'probabilities'
            if not act_dir.is_dir() or not prob_dir.is_dir():
                print("  missing activations_deviant/ or probabilities/; skipping.")
                continue
            out_dir = model_root / 'act_likelihood_join'
            outputs = [out_dir / 'deviant_activity_likelihood_joined.csv',
                       out_dir / 'scatter_activity_vs_likelihood.png',
                       out_dir / 'corr_heatmap_pearson.png',
                       out_dir / 'corr_heatmap_spearman.png']
            sources = (model_sequence_csvs(OUTPUT_ROOT, model_name, 'activations_deviant')
                       + model_sequence_csvs(OUTPUT_ROOT, model_name, 'probabilities'))
            if not OVERWRITE and outputs_up_to_date(outputs, sources):
                print("  up to date, skipped")
                continue

            merged_rows = []
            for act_file in sorted(act_dir.glob('*_deviant_trial.csv')):
                sequence = act_file.name.replace('_deviant_trial.csv', '')
                prob_file = prob_dir / f'{sequence}_probabilities.csv'
                if not prob_file.exists():
                    continue
                act_df = exp.load_deviant_activity(act_file)
                lik_df = exp.load_deviant_likelihoods(
                    prob_file, module_names=cfg.MODULE_NAMES)
                merged = exp.join_activity_likelihood(
                    act_df, lik_df, module_names=cfg.MODULE_NAMES)
                merged.insert(0, 'source', sequence)
                merged_rows.append(merged)

            if not merged_rows:
                print("  no matching activation/probability pairs; skipping.")
                continue

            pooled = pd.concat(merged_rows, ignore_index=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            pooled.to_csv(outputs[0], index=False)
            print(f"  {len(pooled)} trials pooled -> {outputs[0].name}")

            fig = plots.plot_activity_likelihood_grid(
                pooled, activity_cols, likelihood_cols,
                activity_labels=activity_labels,
                likelihood_labels=likelihood_labels)
            fig.suptitle('Module activity at the deviant vs deviant likelihood',
                         fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            plots.save_figure(fig, out_dir, outputs[1].name)

            for method in ('pearson', 'spearman'):
                corr = exp.compute_activity_likelihood_correlations(
                    pooled, activity_cols, likelihood_cols, method=method)
                corr.index = activity_labels
                corr.columns = likelihood_labels
                fig = plots.plot_correlation_heatmap(
                    corr,
                    title=f'{method.capitalize()} correlation: activity vs likelihood')
                plots.save_figure(fig, out_dir, f'corr_heatmap_{method}.png')
                print(f"\n  {method} correlations (activity x likelihood):")
                print(corr.round(3).to_string())
