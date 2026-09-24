"""Run trained models on the experimental trial sequences, and plot their activity.

For each model in MODEL_NAMES and every sequence file in TRIALS_PATH, the script
can write three kinds of CSVs and draw three kinds of figures, each switched on
or off in the settings of the __main__ block. Paths are relative to
``<output-root>/<model name>/``.

  CSVs (one per sequence file) -- the only step that runs the model
    WRITE_ACTIVATIONS          per-timestep module activity norms + derivatives
                               -> activations/*_activations.csv
    WRITE_ACTIVATIONS_DEVIANT  one row per trial, sampled at the deviant timestep
                               -> activations_deviant/*_deviant_trial.csv
    WRITE_PROBABILITIES        per-module predicted distributions + ground-truth
                               likelihoods, plus a deviant-only subset
                               -> probabilities/*_probabilities.csv
                               -> probabilities_deviant/*_probabilities_deviant.csv

  Figures (-> viz_examples/), drawn from the CSVs
    PLOT_TRAJECTORIES          individual + averaged activity for a small random
                               sample of sequences          <- activations
    PLOT_BY_POSITION           activity averaged over sequences, split by
                               within-trial position        <- activations
    PLOT_DEVIANT               activity at the deviant timestep, grouped by
                               deviant position             <- activations_deviant

Every figure comes in two versions: the activity norms and their derivatives.

Existing results are reused. A CSV is only written when it does not exist yet,
and the model is only loaded when at least one CSV is missing. A figure is only
drawn when it does not exist yet or is older than the CSVs it is drawn from.
Asking for a figure also writes the CSVs it needs. Set OVERWRITE to True to
redo everything.
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


# =============================================================================
# CSVs of activations and probabilities
# =============================================================================

def csv_kinds_to_write(trial_file, csv_kinds, output_root, model_name, overwrite):
    """The CSV kinds, among `csv_kinds`, that still have to be written for one sequence file."""
    if overwrite:
        return csv_kinds
    return [kind for kind in csv_kinds
            if not sequence_csv_path(output_root, model_name, kind, trial_file.stem).exists()]


def write_sequence_csvs(model, info, trial_files, csv_kinds, output_root, model_name,
                        period, cue_seed, chunk_size, include_prior, include_next_stimulus,
                        overwrite):
    """Write the missing CSVs of `trial_files`, running one batched forward pass per chunk.

    - activations: per-timestep module activity norms + derivatives
    - activations_deviant: one row per trial, sampled at the deviant timestep
    - probabilities: per-module predicted distributions + ground-truth likelihoods
    - probabilities_deviant: same as probabilities, but restricted to the deviant rows + optionally the immediate next stimulus row
    """
    for kind in csv_kinds:
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

    for start in range(0, len(trial_files), chunk_size):
        chunk = trial_files[start:start + chunk_size]
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

        # One forward pass serves every kind of CSV:
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
            stem = trial_file.stem
            kinds = csv_kinds_to_write(trial_file, csv_kinds, output_root, model_name, overwrite)
            seq_norms = {name: arr[:, j] for name, arr in norms.items()}      # each: (T-1,)
            seq_derivs = {name: arr[:, j] for name, arr in derivs.items()}    # each: (T-2,)

            if 'activations' in kinds:
                out_df = exp.build_activations_frame(obs, cue, seq_norms, seq_derivs,
                                                     lim_std, d, tau_std, trial_n)
                out_df.to_csv(sequence_csv_path(output_root, model_name, 'activations', stem),
                              index=False)

            if 'activations_deviant' in kinds:
                out_df = exp.build_deviant_activations_frame(
                    obs, dpos_raw, seq_norms, seq_derivs, lim_std, d, tau_std, trial_n,
                    period=period, dpos_shift=dpos_shift)
                out_df.to_csv(sequence_csv_path(output_root, model_name, 'activations_deviant', stem),
                              index=False)

            # The two probability CSVs come from the same frame, so they are written together.
            if 'probabilities' in kinds or 'probabilities_deviant' in kinds:
                # Shift the experimental deviant-position labels into the model's
                # training convention ({2..6} -> {3..7}); the on-disk files are left
                # untouched. Probabilities keep a batch dimension of one.
                out_df = exp.build_probabilities_frame(
                    obs, cue, ctx, dpos_raw + dpos_shift, rule,
                    {name: arr[:, j:j + 1] for name, arr in probs.items()},
                    lim_std, d, tau_std, trial_n, dpos_min,
                    prior_probs=(None if prior_probs is None else
                                 {name: arr[:, j:j + 1] for name, arr in prior_probs.items()}))
                out_df.to_csv(sequence_csv_path(output_root, model_name, 'probabilities', stem),
                              index=False)

                deviant_df = exp.subset_deviant_rows(out_df, include_next_stimulus)
                deviant_df.to_csv(sequence_csv_path(output_root, model_name, 'probabilities_deviant', stem),
                                  index=False)

        print(f"  processed {min(start + chunk_size, len(trial_files))}/{len(trial_files)} sequences")

    for kind in csv_kinds:
        print(f"  {kind}: {output_root / model_name / kind}")


# =============================================================================
# Figures, drawn from the CSVs
# =============================================================================

def figure_name(model_name, name, include_derivatives):
    """File name of one figure, in its norms or derivatives version."""
    if include_derivatives:
        return f"{model_name}_{name}_derivatives.png"
    return f"{model_name}_{name}.png"


def figures_up_to_date(output_dir, model_name, names, csv_paths):
    """True when both versions of every figure in `names` exist and are newer than every CSV."""
    figure_paths = []
    for name in names:
        for include_derivatives in [False, True]:
            figure_paths.append(output_dir / figure_name(model_name, name, include_derivatives))
    return outputs_up_to_date(figure_paths, csv_paths)


def plot_trajectories(trial_files, output_root, model_name, n_trajectories, seed, overwrite):
    """Individual + averaged activity and derivatives.

    Draws one panel row per sequence, so it runs on a small random sample
    (`n_trajectories`) rather than the full set.
    """
    output_dir = output_root / model_name / 'viz_examples'
    csv_paths = [sequence_csv_path(output_root, model_name, 'activations', f.stem)
                 for f in trial_files]
    names = ['exp_activity_trajectories', 'exp_activity_averaged']
    if not overwrite and figures_up_to_date(output_dir, model_name, names, csv_paths):
        print("[plot_trajectories] figures up to date, skipped")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_files = select_files(trial_files, n_trajectories, seed=seed)
    n_select = len(selected_files)
    print(f"[plot_trajectories] using {n_select} trial sequence files")

    # Norms (T-1, N); sequences of unequal length are truncated to the shortest one.
    module_norms_dict = exp.load_activity_norms(
        [sequence_csv_path(output_root, model_name, 'activations', f.stem)
         for f in selected_files])

    # Build pars dict in the format expected by extract_sample_parameters
    params_list = [load_trial_params(f) for f in selected_files]
    pars = {key: [p[key] for p in params_list] for key in ['tau', 'lim', 'si_stat', 'si_r']}

    seq_len = next(iter(module_norms_dict.values())).shape[0]
    timesteps = np.arange(seq_len)

    for include_derivatives in [False, True]:
        fig = plots.plot_individual_trajectories(
            module_norms_dict, MODULE_TITLES, timesteps,
            output_dir, model_name, include_derivatives=include_derivatives, pars=pars,
        )
        plots.save_figure(fig, output_dir,
                          figure_name(model_name, 'exp_activity_trajectories', include_derivatives))

        fig = plots.plot_averaged_activity(
            module_norms_dict, MODULE_TITLES, timesteps,
            output_dir, model_name, n_samples=n_select, include_derivatives=include_derivatives,
        )
        plots.save_figure(fig, output_dir,
                          figure_name(model_name, 'exp_activity_averaged', include_derivatives))


def plot_by_position(trial_files, output_root, model_name, n_sequences, seed, period, overwrite):
    """Activity / derivatives averaged over sequences, split by within-trial position."""
    output_dir = output_root / model_name / 'viz_examples'
    csv_paths = [sequence_csv_path(output_root, model_name, 'activations', f.stem)
                 for f in trial_files]
    names = ['exp_activity_averaged_by_position']
    if not overwrite and figures_up_to_date(output_dir, model_name, names, csv_paths):
        print("[plot_by_position] figures up to date, skipped")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_files = select_files(trial_files, n_sequences, seed=seed)
    n_select = len(selected_files)
    print(f"[plot_by_position] using {n_select} trial sequence files")

    module_norms_dict = exp.load_activity_norms(
        [sequence_csv_path(output_root, model_name, 'activations', f.stem)
         for f in selected_files])

    for include_derivatives in [False, True]:
        fig = plots.plot_averaged_activity_by_position(
            module_norms_dict, MODULE_TITLES, output_dir, model_name,
            n_samples=n_select, period=period, include_derivatives=include_derivatives,
        )
        plots.save_figure(fig, output_dir,
                          figure_name(model_name, 'exp_activity_averaged_by_position',
                                      include_derivatives))


def plot_deviant(info, trial_files, output_root, model_name, n_sequences, seed, overwrite):
    """Activity / derivatives at the deviant timestep, grouped by deviant position."""
    output_dir = output_root / model_name / 'viz_examples'
    csv_paths = [sequence_csv_path(output_root, model_name, 'activations_deviant', f.stem)
                 for f in trial_files]
    names = ['exp_deviant_activity_by_position']
    if not overwrite and figures_up_to_date(output_dir, model_name, names, csv_paths):
        print("[plot_deviant] figures up to date, skipped")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_files = select_files(trial_files, n_sequences, seed=seed)
    n_select = len(selected_files)
    print(f"[plot_deviant] using {n_select} trial sequence files")

    _, dpos_shift = dpos_conventions(info)
    dev_df = exp.load_deviant_activity_frame(
        [sequence_csv_path(output_root, model_name, 'activations_deviant', f.stem)
         for f in selected_files], dpos_shift)

    for include_derivatives in [False, True]:
        fig = plots.plot_deviant_activity_by_position(
            dev_df, MODULE_TITLES, output_dir, model_name,
            n_samples=n_select, include_derivatives=include_derivatives,
        )
        plots.save_figure(fig, output_dir,
                          figure_name(model_name, 'exp_deviant_activity_by_position',
                                      include_derivatives))


if __name__ == '__main__':
    plots.set_script_path(__file__)

    # ------------------------- SETTINGS (edit these) -------------------------
    MODEL_NAMES = cfg.EVALUATION_MODELS
    MODEL_DIR = cfg.TRAINING_RESULTS_DIR
    TRIALS_PATH = cfg.TRIALS_PATH
    # Each model writes to OUTPUT_ROOT/<model name>/.
    OUTPUT_ROOT = cfg.EXP_SEQ_OUTPUT_ROOT

    # CSVs, one per sequence file (the only step that runs the model).
    WRITE_ACTIVATIONS = True
    WRITE_ACTIVATIONS_DEVIANT = True
    WRITE_PROBABILITIES = True

    # Figures, drawn from the CSVs.
    PLOT_TRAJECTORIES = True
    PLOT_BY_POSITION = True
    PLOT_DEVIANT = True

    # Existing CSVs and figures are trusted and reused. Set True to rewrite them
    # all, e.g. after retraining a model or changing CUE_SEED, INCLUDE_PRIOR,
    # INCLUDE_NEXT_STIMULUS, or the figure sampling below.
    OVERWRITE = False

    # Sequence files used by PLOT_BY_POSITION / PLOT_DEVIANT (None: all of them),
    # and by PLOT_TRAJECTORIES, which draws one panel row per sequence.
    N_SEQUENCES = None
    N_TRAJECTORIES = 2
    SEED = 0                        # random file sampling

    PERIOD = cfg.PERIOD             # timesteps per trial
    CUE_SEED = cfg.CUE_SEED         # cue re-encoding; keep it fixed across runs
    CHUNK_SIZE = cfg.CHUNK_SIZE     # sequences per batched forward pass

    # Deviant-only probabilities file: deviant rows plus the immediate next
    # stimulus (True), or the deviant rows alone (False).
    INCLUDE_NEXT_STIMULUS = True
    # Also write the prior (first-call) readout of every module to the probability
    # CSVs, as prior_-prefixed columns; the unprefixed columns stay the posterior.
    INCLUDE_PRIOR = False
    # -------------------------------------------------------------------------

    # CSV kinds to write; asking for a figure also asks for the CSVs it is drawn from.
    csv_kinds = []
    if WRITE_ACTIVATIONS or PLOT_TRAJECTORIES or PLOT_BY_POSITION:
        csv_kinds.append('activations')
    if WRITE_ACTIVATIONS_DEVIANT or PLOT_DEVIANT:
        csv_kinds.append('activations_deviant')
    if WRITE_PROBABILITIES:
        csv_kinds.append('probabilities')
        csv_kinds.append('probabilities_deviant')

    trial_files = find_trial_files(TRIALS_PATH)
    if not trial_files:
        raise FileNotFoundError(f"No .csv or .txt sequence files in {TRIALS_PATH}")
    print(f"Found {len(trial_files)} trial sequence files in {TRIALS_PATH}")
    print(f"CSVs: {', '.join(csv_kinds)}")
    print(f"Models ({len(MODEL_NAMES)}): {', '.join(MODEL_NAMES)}")

    for i, model_name in enumerate(MODEL_NAMES, start=1):
        print(f"\n{'=' * 79}\n[{i}/{len(MODEL_NAMES)}] {model_name}\n{'=' * 79}")
        # Only the config is read here; the weights are loaded when a forward pass is due.
        info = ModelInfo.from_path(MODEL_DIR / model_name)

        files_missing_csvs = [f for f in trial_files
                              if csv_kinds_to_write(f, csv_kinds, OUTPUT_ROOT, model_name, OVERWRITE)]
        print(f"{len(files_missing_csvs)}/{len(trial_files)} sequence file(s) need CSVs written")
        if files_missing_csvs:
            model = load_model(info)
            model.eval()
            print(f"Loaded model: {model_name}")
            write_sequence_csvs(model, info, files_missing_csvs, csv_kinds, OUTPUT_ROOT,
                                model_name, PERIOD, CUE_SEED, CHUNK_SIZE, INCLUDE_PRIOR,
                                INCLUDE_NEXT_STIMULUS, OVERWRITE)

        if PLOT_TRAJECTORIES:
            plot_trajectories(trial_files, OUTPUT_ROOT, model_name, N_TRAJECTORIES, SEED,
                              OVERWRITE)
        if PLOT_BY_POSITION:
            plot_by_position(trial_files, OUTPUT_ROOT, model_name, N_SEQUENCES, SEED, PERIOD,
                             OVERWRITE)
        if PLOT_DEVIANT:
            plot_deviant(info, trial_files, OUTPUT_ROOT, model_name, N_SEQUENCES, SEED,
                         OVERWRITE)
