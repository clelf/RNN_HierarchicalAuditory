"""Unified driver for the experimental-sequence model analyses.

Runs, for each of a list of trained models and one directory of experimental
trial-sequence files, any subset of six stages (toggled in the SETTINGS block by
listing them in STAGES). Paths below are relative to ``<output-root>/<model name>/``,
one tree per model.

  CSV extraction (one CSV per sequence file and kind) -- the only stages that run
  a model; every other experimental-sequence analysis reads these CSVs back
    activations         per-timestep module activity norms + derivatives
                        -> activations/*_activations.csv
    activations_deviant one row per trial, sampled at the deviant timestep
                        -> activations_deviant/*_deviant_trial.csv
    probabilities       per-module predicted distributions + ground-truth
                        likelihoods, plus a deviant-only subset
                        -> probabilities/*_probabilities.csv
                        -> probabilities_deviant/*_probabilities_deviant.csv

  Figures (-> viz_examples/), drawn from those CSVs
    plot_trajectories   individual + averaged activity and derivatives for a
                        small random sample of sequences      <- activations
    plot_by_position    activity / derivatives averaged over sequences, split by
                        within-trial position                 <- activations
    plot_deviant        activity / derivatives at the deviant timestep, grouped
                        by deviant position                   <- activations_deviant

Existing outputs are reused: a sequence is only run through the model when one of
the CSVs its requested stages write is missing, and a model is only loaded when
at least one sequence needs it. The sequences that do need it go through one
batched forward pass per chunk, which serves all three extraction stages at once.
A figure stage is redrawn when one of its figures is missing or older than one of
the CSVs it is drawn from. Requesting a figure stage also makes sure
the CSVs it reads exist, extracting the missing ones.

The stage implementations live in exp_sequence_analysis.py and plots.py.
"""

import numpy as np
import torch

import analysis_config as cfg
import exp_sequence_analysis as exp
import plots
from analysis_config import MODULE_TITLES
from analysis_core import (
    ModelInfo,
    dpos_conventions,
    find_trial_files,
    get_module_output_and_activity,
    load_model,
    load_trial_params,
    load_trial_sequence,
    outputs_up_to_date,
    select_files,
    sequence_csv_path,
)

EXTRACTION_STAGES = ('activations', 'activations_deviant', 'probabilities')
PLOT_STAGES = ('plot_trajectories', 'plot_by_position', 'plot_deviant')
ALL_STAGES = EXTRACTION_STAGES + PLOT_STAGES

# CSV kinds (keys of analysis_config.SEQUENCE_CSV_SUFFIXES) each extraction stage
# writes per sequence.
STAGE_CSVS = {
    'activations':         ('activations',),
    'activations_deviant': ('activations_deviant',),
    'probabilities':       ('probabilities', 'probabilities_deviant'),
}

# The extraction stage whose CSVs each figure stage is drawn from.
FIGURE_SOURCES = {
    'plot_trajectories': 'activations',
    'plot_by_position':  'activations',
    'plot_deviant':      'activations_deviant',
}

# Figure names (after '<model>_') each figure stage writes, each in two versions.
FIGURE_STEMS = {
    'plot_trajectories': ('exp_activity_trajectories', 'exp_activity_averaged'),
    'plot_by_position':  ('exp_activity_averaged_by_position',),
    'plot_deviant':      ('exp_deviant_activity_by_position',),
}

# Every activity figure comes in two versions: the norms and their derivatives.
DERIVATIVE_SUFFIXES = ((False, ''), (True, '_derivatives'))


# =============================================================================
# Produce CSV of activations, probabilities
# =============================================================================

def pending_extraction(trial_files, stages, output_root, model_name, overwrite):
    """{trial file: extraction stages to (re)write}, for the files with any left to do."""
    pending = {}
    for trial_file in trial_files:
        todo = [s for s in stages
                if overwrite or not all(
                    sequence_csv_path(output_root, model_name, kind, trial_file.stem).exists()
                    for kind in STAGE_CSVS[s])]
        if todo:
            pending[trial_file] = todo
    return pending


def write_sequence_csvs(model, info, pending, output_root, model_name, period, cue_seed,
                        chunk_size, include_prior, include_next_stimulus):
    """Write the pending CSVs, running one batched forward pass per chunk of sequences.

    - activations: per-timestep module activity norms + derivatives
    - activations_deviant: one row per trial, sampled at the deviant timestep
    - probabilities: per-module predicted distributions + ground-truth likelihoods
    - probabilities_deviant: same as probabilities, but restricted to the deviant rows + optionally the immediate next stimulus row
    """
    kinds = {kind for todo in pending.values() for s in todo for kind in STAGE_CSVS[s]}
    for kind in kinds:
        (output_root / model_name / kind).mkdir(parents=True, exist_ok=True)

    # If the model was trained with a larger cue vocabulary than the two cues present
    # in the experimental sequences, re-encode the cues into the dimensionality it
    # expects.
    n_cue_classes = len(info.data_config_dict['cues_set'])

    # dpos alignment, read from the model's saved config (rules_dpos_set): dpos_min is
    # the model's class-index offset (class c <-> position c+dpos_min), and dpos_shift
    # maps the on-disk experimental dpos into the model's convention. Self-adjusting:
    # lr0 -> (3, 1); a model retrained on [[2,3,4],[4,5,6]] -> (2, 0). The sequence
    # files are never edited.
    dpos_min, dpos_shift = dpos_conventions(info)
    print(f"dpos class-index offset (model convention): {dpos_min}; "
          f"experimental dpos shift: +{dpos_shift}")

    files = list(pending)
    for start in range(0, len(files), chunk_size):
        chunk = files[start:start + chunk_size]
        # Each entry: obs, cue, ctx, dpos_raw, rule, lim_std, d, tau_std, trial_n.
        # ctx/dpos/rule are the ground-truth labels from the original sequence;
        # dpos_raw is the raw deviant position on disk (e.g. 2..6), not a class index.
        seqs = [load_trial_sequence(f, return_hierarch=True,
                                    n_cue_classes=n_cue_classes, cue_seed=cue_seed)
                for f in chunk]

        # Stacking requires every sequence of a chunk to have the same length.
        y = torch.tensor(np.stack([seq[0] for seq in seqs]),
                         dtype=torch.float32).unsqueeze(-1)             # (N, T, 1)
        q = torch.tensor(np.stack([seq[1] for seq in seqs]),
                         dtype=torch.float32)                           # (N, T, n_cue)

        # One forward pass serves every stage:
        #   probs:  dict module → (T-1, N, dim)
        #     'obs':  dim=2, columns are (mean, variance) of the predicted Gaussian
        #     others: dim=n_classes, softmax class probabilities
        #   norms:  dict module → (T-1, N);  derivs: dict module → (T-2, N)
        # With include_prior, the probabilities are also read from the prior
        # (first-call) readout of each module, i.e. before the feedback sweep.
        forward = get_module_output_and_activity(model, y, q, return_prior=include_prior)
        probs, norms, derivs = forward[:3]
        prior_probs = forward[3] if include_prior else None

        for j, (trial_file, seq) in enumerate(zip(chunk, seqs)):
            obs, cue, ctx, dpos_raw, rule, lim_std, d, tau_std, trial_n = seq
            todo = pending[trial_file]
            seq_norms = {name: arr[:, j] for name, arr in norms.items()}      # each: (T-1,)
            seq_derivs = {name: arr[:, j] for name, arr in derivs.items()}    # each: (T-2,)
            csv_path = {kind: sequence_csv_path(output_root, model_name, kind, trial_file.stem)
                        for kind in kinds}

            if 'activations' in todo:
                out_df = exp.build_activations_frame(obs, cue, seq_norms, seq_derivs,
                                                     lim_std, d, tau_std, trial_n)
                out_df.to_csv(csv_path['activations'], index=False)

            if 'activations_deviant' in todo:
                out_df = exp.build_deviant_activations_frame(
                    obs, dpos_raw, seq_norms, seq_derivs, lim_std, d, tau_std, trial_n,
                    period=period, dpos_shift=dpos_shift)
                out_df.to_csv(csv_path['activations_deviant'], index=False)

            if 'probabilities' in todo:
                # Shift the experimental deviant-position labels into the model's
                # training convention ({2..6} -> {3..7}); the on-disk files are left
                # untouched. Probabilities keep a batch dimension of one.
                out_df = exp.build_probabilities_frame(
                    obs, cue, ctx, dpos_raw + dpos_shift, rule,
                    {name: arr[:, j:j + 1] for name, arr in probs.items()},
                    lim_std, d, tau_std, trial_n, dpos_min,
                    prior_probs=(None if prior_probs is None else
                                 {name: arr[:, j:j + 1] for name, arr in prior_probs.items()}))
                out_df.to_csv(csv_path['probabilities'], index=False)

                deviant_df = exp.subset_deviant_rows(out_df, include_next_stimulus)
                deviant_df.to_csv(csv_path['probabilities_deviant'], index=False)

        print(f"  processed {min(start + chunk_size, len(files))}/{len(files)} sequences")

    for kind in sorted(kinds):
        print(f"  {kind}: {output_root / model_name / kind}")


# =============================================================================
# Figures, drawn from the CSVs
# =============================================================================

def figure_files(stage, model_name):
    """Every figure file name a figure stage writes."""
    return [f"{model_name}_{stem}{suffix}.png"
            for stem in FIGURE_STEMS[stage] for _, suffix in DERIVATIVE_SUFFIXES]


def run_plot_trajectories(trial_files, output_root, output_dir, model_name,
                          n_trajectories, seed):
    """Individual + averaged activity and derivatives.

    Draws one panel row per sequence, so it runs on a small random sample
    (`n_trajectories`) rather than the full set.
    """
    selected_files = select_files(trial_files, n_trajectories, seed=seed)
    n_select = len(selected_files)
    print(f"[plot_trajectories] using {n_select} trial sequence files")

    # Norms (T-1, N); sequences of unequal length are truncated to the shortest one.
    module_norms_dict = exp.load_activity_norms(
        [sequence_csv_path(output_root, model_name, 'activations', f.stem)
         for f in selected_files])

    # Build pars dict in the format expected by extract_sample_parameters
    params_list = [load_trial_params(f) for f in selected_files]
    pars = {key: [p[key] for p in params_list] for key in ('tau', 'lim', 'si_stat', 'si_r')}

    seq_len = next(iter(module_norms_dict.values())).shape[0]
    timesteps = np.arange(seq_len)

    for include_derivatives, suffix in DERIVATIVE_SUFFIXES:
        fig = plots.plot_individual_trajectories(
            module_norms_dict, MODULE_TITLES, timesteps,
            output_dir, model_name, include_derivatives=include_derivatives, pars=pars,
        )
        plots.save_figure(fig, output_dir, f"{model_name}_exp_activity_trajectories{suffix}.png")

        fig = plots.plot_averaged_activity(
            module_norms_dict, MODULE_TITLES, timesteps,
            output_dir, model_name, n_samples=n_select, include_derivatives=include_derivatives,
        )
        plots.save_figure(fig, output_dir, f"{model_name}_exp_activity_averaged{suffix}.png")


def run_plot_by_position(module_norms_dict, n_select, output_dir, model_name, period):
    """Activity / derivatives averaged over sequences, split by within-trial position."""
    for include_derivatives, suffix in DERIVATIVE_SUFFIXES:
        fig = plots.plot_averaged_activity_by_position(
            module_norms_dict, MODULE_TITLES, output_dir, model_name,
            n_samples=n_select, period=period, include_derivatives=include_derivatives,
        )
        plots.save_figure(fig, output_dir,
                          f"{model_name}_exp_activity_averaged_by_position{suffix}.png")


def run_plot_deviant(dev_df, n_select, output_dir, model_name):
    """Activity / derivatives at the deviant timestep, grouped by deviant position."""
    for include_derivatives, suffix in DERIVATIVE_SUFFIXES:
        fig = plots.plot_deviant_activity_by_position(
            dev_df, MODULE_TITLES, output_dir, model_name,
            n_samples=n_select, include_derivatives=include_derivatives,
        )
        plots.save_figure(fig, output_dir,
                          f"{model_name}_exp_deviant_activity_by_position{suffix}.png")


def run_plots(info, trial_files, stages, output_root, model_name,
              n_trajectories, n_sequences, seed, period):
    """Run the requested figure stages from the CSVs.

    plot_by_position and plot_deviant draw on the same (large) file selection.
    """
    output_dir = output_root / model_name / 'viz_examples'
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'plot_trajectories' in stages:
        run_plot_trajectories(trial_files, output_root, output_dir, model_name,
                              n_trajectories, seed)

    include_by_pos = 'plot_by_position' in stages
    include_dev = 'plot_deviant' in stages
    if include_by_pos or include_dev:
        selected_files = select_files(trial_files, n_sequences, seed=seed)
        n_select = len(selected_files)
        print(f"[plot_by_position/plot_deviant] using {n_select} trial sequence files")

        if include_by_pos:
            module_norms_dict = exp.load_activity_norms(
                [sequence_csv_path(output_root, model_name, 'activations', f.stem)
                 for f in selected_files])
            run_plot_by_position(module_norms_dict, n_select, output_dir, model_name, period)

        if include_dev:
            _, dpos_shift = dpos_conventions(info)
            dev_df = exp.load_deviant_activity_frame(
                [sequence_csv_path(output_root, model_name, 'activations_deviant', f.stem)
                 for f in selected_files], dpos_shift)
            run_plot_deviant(dev_df, n_select, output_dir, model_name)

    print(f"\nFigures saved to {output_dir}")


if __name__ == '__main__':
    plots.set_script_path(__file__)

    # ------------------------- SETTINGS (edit these) -------------------------
    # Any subset of ALL_STAGES; they always run in ALL_STAGES order.
    STAGES = list(ALL_STAGES)
    MODEL_NAMES = cfg.EVALUATION_MODELS
    MODEL_DIR = cfg.TRAINING_RESULTS_DIR
    TRIALS_PATH = cfg.TRIALS_PATH
    # Each model writes to OUTPUT_ROOT/<model name>/.
    OUTPUT_ROOT = cfg.EXP_SEQ_OUTPUT_ROOT

    # Existing CSVs and figures are trusted and reused. Set True to rewrite them
    # all, e.g. after retraining a model or changing CUE_SEED, INCLUDE_PRIOR,
    # INCLUDE_NEXT_STIMULUS, or the figure sampling below.
    OVERWRITE = False

    # Sequence files used by plot_by_position / plot_deviant (None: all of them),
    # and by plot_trajectories, which draws one panel row per sequence.
    N_SEQUENCES = None
    N_TRAJECTORIES = 2
    SEED = 0                        # random file sampling

    PERIOD = cfg.PERIOD             # timesteps per trial
    CUE_SEED = cfg.CUE_SEED         # cue re-encoding; keep it fixed across stages
    CHUNK_SIZE = cfg.CHUNK_SIZE     # sequences per batched forward pass

    # Deviant-only probabilities file: deviant rows plus the immediate next
    # stimulus (True), or the deviant rows alone (False).
    INCLUDE_NEXT_STIMULUS = True
    # Also write the prior (first-call) readout of every module to the probability
    # CSVs, as prior_-prefixed columns; the unprefixed columns stay the posterior.
    INCLUDE_PRIOR = False
    # -------------------------------------------------------------------------

    unknown_stages = [s for s in STAGES if s not in ALL_STAGES]
    if unknown_stages:
        raise ValueError(f"Unknown stage(s) {unknown_stages}; choose from {ALL_STAGES}")
    figure_stages = [s for s in PLOT_STAGES if s in STAGES]
    # Figure stages need the CSVs they are drawn from.
    extraction_stages = [s for s in EXTRACTION_STAGES
                         if s in STAGES or s in [FIGURE_SOURCES[f] for f in figure_stages]]

    trial_files = find_trial_files(TRIALS_PATH)
    if not trial_files:
        raise FileNotFoundError(f"No .csv or .txt sequence files in {TRIALS_PATH}")
    print(f"Found {len(trial_files)} trial sequence files in {TRIALS_PATH}")
    print(f"Extraction stages: {', '.join(extraction_stages)}")
    print(f"Figure stages: {', '.join(figure_stages)}")
    print(f"Models ({len(MODEL_NAMES)}): {', '.join(MODEL_NAMES)}")

    for i, model_name in enumerate(MODEL_NAMES, start=1):
        print(f"\n{'=' * 79}\n[{i}/{len(MODEL_NAMES)}] {model_name}\n{'=' * 79}")
        # Only the config is read here; the weights are loaded when a forward pass is due.
        info = ModelInfo.from_path(MODEL_DIR / model_name)

        pending = pending_extraction(trial_files, extraction_stages, OUTPUT_ROOT,
                                     model_name, OVERWRITE)
        print(f"{len(pending)}/{len(trial_files)} sequence file(s) need CSVs written")
        if pending:
            model = load_model(info)
            model.eval()
            print(f"Loaded model: {model_name}")
            write_sequence_csvs(model, info, pending, OUTPUT_ROOT, model_name, PERIOD,
                                CUE_SEED, CHUNK_SIZE, INCLUDE_PRIOR, INCLUDE_NEXT_STIMULUS)

        # A figure stage is stale when a figure is missing or older than a CSV it reads.
        viz_dir = OUTPUT_ROOT / model_name / 'viz_examples'
        stale_figures = [s for s in figure_stages
                         if OVERWRITE or not outputs_up_to_date(
                             [viz_dir / name for name in figure_files(s, model_name)],
                             [sequence_csv_path(OUTPUT_ROOT, model_name, FIGURE_SOURCES[s], f.stem)
                              for f in trial_files])]
        print(f"Figure stages to draw: {', '.join(stale_figures) or 'none'}")
        if stale_figures:
            run_plots(info, trial_files, stale_figures, OUTPUT_ROOT, model_name,
                      N_TRAJECTORIES, N_SEQUENCES, SEED, PERIOD)
