"""Assess ctx/dpos detection alignment, and collapse the result into one table per model.

Two stages, both toggled in the SETTINGS block below:

  1. per-trial   -- read each model's reports on the experimental sequences from
                    the probabilities CSVs run_exp_trials_pipeline.py writes, and
                    classify every trial into one dpos case and one ctx case
                    -> <output-root>/<model>/alignment/<model>_trial_alignment.csv
                    -> ..._alignment_summary.csv (that model's case counts)
  2. per-model   -- read those tables back and write one wide row per model, each
                    cell the proportion of that model's trials in that case
                    -> <output-root>/dpos_and_ctx_detection_case_proportions.csv

The case taxonomies, the classification logic and the table builders live in
alignment_cases.py; this file only chooses what to run it on.

Replaces assess_dpos_and_ctx_detection.py and its _summary companion. The latter
could not run at all: it imported a module name left behind by an earlier rename.
"""

import pandas as pd

import analysis_config as cfg
import dpos_ctx_detection_alignment_cases as align
from analysis_core import find_trial_files, outputs_up_to_date, select_files, sequence_csv_path


if __name__ == '__main__':

    # ------------------------- SETTINGS (edit these) -------------------------
    MODEL_NAMES = cfg.EVALUATION_MODELS
    TRIALS_PATH = cfg.TRIALS_PATH
    OUTPUT_ROOT = cfg.EXP_SEQ_OUTPUT_ROOT

    # Sequence files: None uses all of them; an int samples that many.
    N_SEQUENCES = None
    SEED = 0

    # Also write every model's trials into one combined CSV (None = don't).
    COMBINED_CSV = None

    # A model's per-trial tables are reused when newer than its probabilities CSVs.
    # Set True to redo them all, e.g. after changing N_SEQUENCES, SEED, or run_cfg.
    OVERWRITE = False

    RUN_PER_TRIAL = True    # stage 1
    RUN_CASE_TABLE = True   # stage 2

    # Stage 2 presentation.
    AS_PERCENT = False
    DECIMALS = None
    SHOW_LEGEND = True

    run_cfg = align.AlignmentConfig(
        model_dir=cfg.TRAINING_RESULTS_DIR,
        output_root=OUTPUT_ROOT,
        period=cfg.PERIOD,
        skip_trials=align.DEFAULT_SKIP_TRIALS,
        ctx_rule='argmax',
        ctx_threshold=align.DEFAULT_CTX_THRESHOLD,
        no_summary=False,
    )
    # -------------------------------------------------------------------------

    if RUN_PER_TRIAL:
        all_files = find_trial_files(TRIALS_PATH)
        if not all_files:
            raise FileNotFoundError(f"No .csv or .txt sequence files in {TRIALS_PATH}")
        sequences = [f.stem for f in select_files(all_files, N_SEQUENCES, seed=SEED)]
        print(f"Found {len(all_files)} trial sequence files in {TRIALS_PATH}; "
              f"using {len(sequences)}")
        print(f"Models ({len(MODEL_NAMES)}): {', '.join(MODEL_NAMES)}")
        print(f"ctx report rule: {run_cfg.ctx_rule}"
              + (f" (threshold {run_cfg.ctx_threshold})"
                 if run_cfg.ctx_rule == 'threshold' else ''))

        frames = []
        for i, model_name in enumerate(MODEL_NAMES, start=1):
            print(f"\n{'=' * 79}\n[{i}/{len(MODEL_NAMES)}] {model_name}\n{'=' * 79}")
            outputs = align.alignment_outputs(OUTPUT_ROOT, model_name, run_cfg.no_summary)
            sources = [sequence_csv_path(OUTPUT_ROOT, model_name, 'probabilities', name)
                       for name in sequences]
            if not OVERWRITE and outputs_up_to_date(outputs, sources):
                print(f"  up to date, skipped: {outputs[0].name}")
                if COMBINED_CSV:
                    frames.append(pd.read_csv(outputs[0]))
                continue
            df, summary = align.run_one_model(model_name, sequences, run_cfg)
            frames.append(df)

        if COMBINED_CSV:
            COMBINED_CSV.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_csv(COMBINED_CSV, index=False)
            print(f"\nSaved combined table: {COMBINED_CSV}")

    if RUN_CASE_TABLE:
        paths = align.find_trial_tables(OUTPUT_ROOT, MODEL_NAMES)
        if not paths:
            raise FileNotFoundError(
                f"no *_trial_alignment.csv under {OUTPUT_ROOT}/*/alignment/ — "
                "run the per-trial stage first")
        print(f"\nReading {len(paths)} per-trial table(s)")

        table = align.build_case_table(align.load_cases(paths), MODEL_NAMES)
        align.check_family_sums(table)

        case_columns = align.DPOS_COLUMNS + align.CTX_COLUMNS
        if AS_PERCENT:
            table[case_columns] *= 100
        if DECIMALS is not None:
            table[case_columns] = table[case_columns].round(DECIMALS)

        out_file = OUTPUT_ROOT / align.DEFAULT_OUTPUT_NAME
        out_file.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_file, index=False)

        unit = '%' if AS_PERCENT else 'fraction'
        with pd.option_context('display.width', 200, 'display.max_columns', None,
                               'display.float_format', lambda v: f'{v:.3f}'):
            print(f"\nProportion of trials per case ({unit}), one row per model:")
            print(table.to_string(index=False))
        if SHOW_LEGEND:
            align.print_legend()
        print(f"\nSaved: {out_file}")
