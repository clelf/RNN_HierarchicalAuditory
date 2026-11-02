"""Run trained models on the experimental trial sequences, and plot their activity.

For each model in MODEL_NAMES and every sequence file in TRIALS_PATH, the script
can write three kinds of analysis output and plot three kinds of figures, each switched on
or off in the settings of the __main__ block. Paths are relative to
``<output-root>/<model name>/``.

  Outputs (one per sequence file)
    WRITE_ACTIVATIONS          per-timestep module activity norms + derivatives
                               -> activations/*_activations.csv
    WRITE_ACTIVATIONS_DEVIANT  one row per trial, sampled at the deviant timestep
                               -> activations_deviant/*_deviant_trial.csv
    WRITE_PROBABILITIES        per-module predicted distributions + ground-truth
                               likelihoods, plus a deviant-only subset
                               -> probabilities/*_probabilities.csv
                               -> probabilities_deviant/*_probabilities_deviant.csv

  Figures (-> viz_examples/), plot from the output CSVs
    PLOT_TRAJECTORIES          averaged activity (mean +/- STD) for a random
                               sample of sequences, plus the individual ones
                               when the sample is small     <- activations
    PLOT_BY_POSITION           activity averaged over sequences, split by
                               within-trial position        <- activations
    PLOT_DEVIANT               activity at the deviant timestep, grouped by
                               deviant position             <- activations_deviant

Every CSV row has a 'timestep' column, the absolute position in the sequence of the
observation of that row (observation == y[timestep]), so the outputs align by
joining on it (exp.merge_activations_and_probabilities): activations at t are
those after seeing y[t] (t = 0 ... T-2) and their derivatives the change on the
arrival of y[t] (t = 1 ... T-2, NaN at t = 0); probabilities at t are the prediction of y[t]
made after seeing y[t-1] (t = 1 ... T-1).

For every figure related to activations, we get two versions: one for activity norms and one for their derivatives.

With SEQUENTIAL_PRESENTATION = True, the sequences are presented in pools
of four consecutive sequences selected randomly (without replacement) as long as they
exhibit distinct tau_std value (analysis_core.make_sequence_pools): the four
sequences of a pool are run one after the other, each starting from the hidden
states the previous one ended with, instead of the trained initial hidden state.
One activations CSV is then written per pool instead of one per sequence, with
the same columns plus the sequence and its position in the pool at each timestep
  -> pool_activations/seed<POOL_SEED>_pool_*_pool_activations.csv
Only WRITE_ACTIVATIONS, PLOT_TRAJECTORIES and PLOT_BY_POSITION can be used with
SEQUENTIAL_PRESENTATION for now. The figures' names end in '_pools_seed<POOL_SEED>'.

Existing results are reused. A CSV is only written when it does not exist yet,
or is older than the sequence file(s) it comes from or the model's weights file
(e.g. after retraining a model under the same name), and the model is only
loaded when at least one CSV has to be written. A figure is only plot when it
does not exist yet or is older than the CSVs it is plot from.

Set OVERWRITE to True to recompute output files and figures.
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
    load_pool_sequence,
    load_trial_params,
    load_trial_sequence,
    make_sequence_pools,
    outputs_up_to_date,
    pool_name,
    select_files,
    sequence_csv_path,
)


# =============================================================================
# CSVs of activations and probabilities
# =============================================================================

def analysis_output_kinds_to_write(trial_file, analysis_output_kinds, output_root, model_name,
                                   weights_path, overwrite):
    """The kinds of analysis output, among `analysis_output_kinds`, that still have to be written for one sequence file.

    A CSV has to be written when it does not exist yet, or when it is older than
    the sequence file or the model's weights file `weights_path`.
    """
    if overwrite:
        return analysis_output_kinds
    kinds = []
    for kind in analysis_output_kinds:
        csv_path = sequence_csv_path(output_root, model_name, kind, trial_file.stem)
        if not outputs_up_to_date([csv_path], [trial_file, weights_path]):
            kinds.append(kind)
    return kinds


def write_proba_and_activ_csvs(model, info, trial_files, analysis_output_kinds, output_root,
                               model_name, period, cue_seed, chunk_size, include_prior,
                               include_next_stimulus, overwrite):
    """Write the missing CSVs of `trial_files`, running one batched forward pass per chunk.

    - activations: per-timestep module activity norms + derivatives
    - activations_deviant: one row per trial, sampled at the deviant timestep
    - probabilities: per-module predicted distributions + ground-truth likelihoods
    - probabilities_deviant: same as probabilities, but restricted to the deviant rows + optionally the immediate next stimulus row
    """
    for kind in analysis_output_kinds:
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

        # One forward pass serves every kind of analysis output:
        #   probs:  dict module → (T-1, N, dim)
        #     'obs':  dim=2, columns are (mean, variance) of the predicted Gaussian
        #     others: dim=n_classes, softmax class probabilities
        #   norms:  dict module → (T-1, N);  derivs: dict module → (T-1, N), row 0 NaN
        # With include_prior, the probabilities are also read from the prior
        # (first-call) readout of each module, i.e. before the feedback sweep, and
        # the norms and derivatives from the prior hidden states.
        forward = get_module_output_and_activity(model, y, q, return_prior=include_prior)
        probs, norms, derivs = forward[:3]
        prior_probs = forward[3] if include_prior else None
        prior_norms = forward[4] if include_prior else None
        prior_derivs = forward[5] if include_prior else None

        for j, (trial_file, seq) in enumerate(zip(chunk, seqs)):
            obs, cue, ctx, dpos_raw, rule, lim_std, d, tau_std, trial_n = seq
            stem = trial_file.stem
            kinds = analysis_output_kinds_to_write(trial_file, analysis_output_kinds, output_root,
                                                   model_name, info.weights_path, overwrite)
            seq_norms = {name: arr[:, j] for name, arr in norms.items()}      # each: (T-1,)
            seq_derivs = {name: arr[:, j] for name, arr in derivs.items()}    # each: (T-1,)
            seq_prior_norms = None
            seq_prior_derivs = None
            if include_prior:
                seq_prior_norms = {name: arr[:, j] for name, arr in prior_norms.items()}
                seq_prior_derivs = {name: arr[:, j] for name, arr in prior_derivs.items()}

            if 'activations' in kinds:
                out_df = exp.build_activations_frame(obs, cue, seq_norms, seq_derivs,
                                                     lim_std, d, tau_std, trial_n,
                                                     prior_norms=seq_prior_norms,
                                                     prior_derivs=seq_prior_derivs)
                out_df.to_csv(sequence_csv_path(output_root, model_name, 'activations', stem),
                              index=False)

            if 'activations_deviant' in kinds:
                out_df = exp.build_deviant_activations_frame(
                    obs, dpos_raw, seq_norms, seq_derivs, lim_std, d, tau_std, trial_n,
                    period=period, dpos_shift=dpos_shift,
                    prior_norms=seq_prior_norms, prior_derivs=seq_prior_derivs)
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

    for kind in analysis_output_kinds:
        print(f"  {kind}: {output_root / model_name / kind}")


def write_pool_activations_csvs(model, info, pools, pool_indices, pool_seed, output_root,
                                model_name, cue_seed, chunk_size, include_prior):
    """Write the activations CSVs of the pools in `pool_indices`, one batched forward pass per chunk.

    The sequences of a pool are joined end to end (analysis_core.load_pool_sequence)
    and the model is run once on the result, so that each sequence starts from the
    hidden states the previous one ended with. `pools` is the list of
    analysis_core.make_sequence_pools plot with `pool_seed`, and `chunk_size` a
    number of pools.
    """
    (output_root / model_name / 'pool_activations').mkdir(parents=True, exist_ok=True)

    # Cue re-encoding, as in write_proba_and_activ_csvs.
    n_cue_classes = len(info.data_config_dict['cues_set'])

    for start in range(0, len(pool_indices), chunk_size):
        chunk = pool_indices[start:start + chunk_size]
        # Each entry: obs, cue, timesteps, for the joined sequences of one pool.
        loaded_pools = [load_pool_sequence(pools[pool_index], n_cue_classes, cue_seed)
                        for pool_index in chunk]

        # Stacking requires every pool of a chunk to have the same length.
        y = torch.tensor(np.stack([loaded_pool[0] for loaded_pool in loaded_pools]),
                         dtype=torch.float32).unsqueeze(-1)             # (N, T_total, 1)
        q = torch.tensor(np.stack([loaded_pool[1] for loaded_pool in loaded_pools]),
                         dtype=torch.float32)                           # (N, T_total, n_cue)

        #   norms and derivs:  dict module → (T_total-1, N)
        # With include_prior, also the norms and derivatives of the prior hidden states.
        forward = get_module_output_and_activity(model, y, q, return_prior=include_prior)
        norms, derivs = forward[1:3]
        prior_norms = forward[4] if include_prior else None
        prior_derivs = forward[5] if include_prior else None

        for j, (pool_index, loaded_pool) in enumerate(zip(chunk, loaded_pools)):
            obs, cue, timesteps = loaded_pool
            pool_norms = {name: arr[:, j] for name, arr in norms.items()}      # each: (T_total-1,)
            pool_derivs = {name: arr[:, j] for name, arr in derivs.items()}    # each: (T_total-1,)
            pool_prior_norms = None
            pool_prior_derivs = None
            if include_prior:
                pool_prior_norms = {name: arr[:, j] for name, arr in prior_norms.items()}
                pool_prior_derivs = {name: arr[:, j] for name, arr in prior_derivs.items()}

            out_df = exp.build_pool_activations_frame(obs, cue, pool_norms, pool_derivs, timesteps,
                                                      pool_prior_norms, pool_prior_derivs)
            out_df.to_csv(sequence_csv_path(output_root, model_name, 'pool_activations',
                                            pool_name(pool_index, pool_seed)),
                          index=False)

        print(f"  processed {min(start + chunk_size, len(pool_indices))}/{len(pool_indices)} pools")

    print(f"  pool_activations: {output_root / model_name / 'pool_activations'}")


# =============================================================================
# Figures, plot from the CSVs
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


def plot_trajectories(act_csvs, trial_files, output_root, model_name, figure_suffix,
                      n_trajectories, max_individual_trajectories, seed, overwrite):
    """Averaged activity and derivatives (mean +/- STD), plus the individual ones.

    `act_csvs` are the activations CSVs the figures are plotted from: one per
    sequence file, or one per pool with sequential presentation. `trial_files` are
    the sequence files of `act_csvs`, in the same order, whose parameters label
    the lines of the individual figure, or None for pools: the sequences of a pool
    have different parameters, and the individual figure then has no legend.
    `figure_suffix` ends the names of the figures.

    The individual figure draws one line and one legend entry per sequence, so it
    is only plot when at most `max_individual_trajectories` sequences are selected.
    """
    # Positions, in `act_csvs` and `trial_files`, of the sequences (or pools) plotted.
    selected_positions = select_files(list(range(len(act_csvs))), n_trajectories, seed=seed)
    n_select = len(selected_positions)
    draw_individual = n_select <= max_individual_trajectories

    output_dir = output_root / model_name / 'viz_examples'
    averaged_name = 'exp_activity_averaged' + figure_suffix
    individual_name = 'exp_activity_trajectories' + figure_suffix
    names = [averaged_name]
    if draw_individual:
        names.append(individual_name)
    if not overwrite and figures_up_to_date(output_dir, model_name, names, act_csvs):
        print("[plot_trajectories] figures up to date, skipped")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot_trajectories] using {n_select} activations CSVs")
    if not draw_individual:
        print(f"[plot_trajectories] more than {max_individual_trajectories} sequences: "
              f"individual trajectories not plot, averaged figure only")

    # Norms (T-1, N); sequences of unequal length are truncated to the shortest one.
    module_norms_dict = exp.load_activity_norms([act_csvs[i] for i in selected_positions],
                                                prior=False)

    seq_len = next(iter(module_norms_dict.values())).shape[0]
    timesteps = np.arange(seq_len)

    if draw_individual:
        pars = None
        if trial_files is not None:
            # Build pars dict in the format expected by extract_sample_parameters
            params_list = [load_trial_params(trial_files[i]) for i in selected_positions]
            pars = {key: [p[key] for p in params_list] for key in ['tau', 'lim', 'si_stat', 'si_r']}

        for include_derivatives in [False, True]:
            fig = plots.plot_individual_trajectories(
                module_norms_dict, MODULE_TITLES, timesteps,
                output_dir, model_name, n_samples=n_select, include_derivatives=include_derivatives, pars=pars,
            )
            plots.save_figure(fig, output_dir,
                              figure_name(model_name, individual_name, include_derivatives))

    for include_derivatives in [False, True]:

        fig = plots.plot_averaged_activity(
            module_norms_dict, MODULE_TITLES, timesteps,
            output_dir, model_name, n_samples=n_select, include_derivatives=include_derivatives,
        )
        plots.save_figure(fig, output_dir,
                          figure_name(model_name, averaged_name, include_derivatives))


def plot_by_position(act_csvs, output_root, model_name, figure_suffix, sample_name,
                     n_sequences, seed, period, overwrite):
    """Activity / derivatives averaged over sequences, split by within-trial position.

    `act_csvs` are the activations CSVs the figures are plotted from: one per
    sequence file, or one per pool with sequential presentation, the trial index
    then running over the whole pool. `sample_name` names what `act_csvs` hold in
    the figure title ('sequences' or 'pools'), and `figure_suffix` ends the names
    of the figures.
    """
    output_dir = output_root / model_name / 'viz_examples'
    by_position_name = 'exp_activity_averaged_by_position' + figure_suffix
    if not overwrite and figures_up_to_date(output_dir, model_name, [by_position_name], act_csvs):
        print("[plot_by_position] figures up to date, skipped")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_csvs = select_files(act_csvs, n_sequences, seed=seed)
    n_select = len(selected_csvs)
    print(f"[plot_by_position] using {n_select} activations CSVs")

    module_norms_dict = exp.load_activity_norms(selected_csvs, prior=False)

    for include_derivatives in [False, True]:
        fig = plots.plot_averaged_activity_by_position(
            module_norms_dict, MODULE_TITLES, output_dir, model_name,
            n_samples=n_select, sample_name=sample_name, period=period,
            include_derivatives=include_derivatives,
        )
        plots.save_figure(fig, output_dir,
                          figure_name(model_name, by_position_name, include_derivatives))


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
    TRIALS_PATH = cfg.WORKSPACE_ROOT / 'Jasmin' / 'trialsequences_new'
    # TRIALS_PATH = cfg.WORKSPACE_ROOT / 'Jasmin' / 'sub-01'

    # Each model writes to OUTPUT_ROOT/<model name>/.
    OUTPUT_ROOT = cfg.RNN_DIR / 'exp_seq_act_output' / TRIALS_PATH.name


    # CSVs, one per sequence file (the only step that runs the model).
    WRITE_ACTIVATIONS = True
    WRITE_ACTIVATIONS_DEVIANT = False
    WRITE_PROBABILITIES = True

    # Figures, created using data from the CSVs.
    PLOT_TRAJECTORIES = True
    PLOT_BY_POSITION = True
    PLOT_DEVIANT = False

    # Existing CSVs and figures are reused when they are newer than what they come
    # from (sequence files and model weights for the CSVs, CSVs for the figures).
    # Set True to rewrite them all, e.g. after changing CUE_SEED, INCLUDE_PRIOR,
    # INCLUDE_NEXT_STIMULUS, or the figure sampling below.
    OVERWRITE = False

    # Sequence files used by PLOT_BY_POSITION / PLOT_DEVIANT (None: all of them),
    # and by PLOT_TRAJECTORIES. With SEQUENTIAL_PRESENTATION, numbers of pools.
    N_SEQUENCES = None
    # N_TRAJECTORIES = 2
    N_TRAJECTORIES = N_SEQUENCES
    # PLOT_TRAJECTORIES draws one line and one legend entry per sequence in its
    # individual figure. Above this number of sequences, only its averaged figure
    # (mean +/- STD) is plot: the legend no longer fits, and matplotlib's 10
    # default colors start repeating.
    MAX_INDIVIDUAL_TRAJECTORIES = 10
    SEED = 0                        # random file sampling

    PERIOD = cfg.PERIOD             # timesteps per trial
    CUE_SEED = cfg.CUE_SEED         # cue re-encoding; keep it fixed across runs
    CHUNK_SIZE = cfg.CHUNK_SIZE     # sequences per batched forward pass

    # Deviant-only probabilities file: deviant rows plus the immediate next
    # stimulus (True), or the deviant rows alone (False).
    INCLUDE_NEXT_STIMULUS = True
    # Also write the prior (first-call) readout of every module to the probability
    # CSVs, and the norm + derivative of its prior hidden state to the activation
    # CSVs, as prior_-prefixed columns; the unprefixed columns stay the posterior.
    # Needed by run_model_pair_correlations.py for its prior quantities.
    INCLUDE_PRIOR = True

    # Sequential presentation: the sequences are grouped into random pools of
    # four, one per tau_std value, and the four sequences of a pool are run one
    # after the other, each starting from the hidden states the previous one
    # ended with. One activations CSV is written per pool, to pool_activations/,
    # instead of one per sequence. Only WRITE_ACTIVATIONS, PLOT_TRAJECTORIES and
    # PLOT_BY_POSITION are available in this mode for now: the figures then run
    # over the four sequences of each pool, and their names end in
    # '_pools_seed<POOL_SEED>'. Read by run_model_pair_correlations.py with its
    # own SEQUENTIAL_PRESENTATION = True.
    SEQUENTIAL_PRESENTATION = False
    POOL_SEED = cfg.POOL_SEED       # random pools; keep it fixed across runs
    # Pools per batched forward pass: a pool is four sequences long, so this
    # keeps the memory of one forward pass the same as with CHUNK_SIZE sequences.
    POOL_CHUNK_SIZE = cfg.CHUNK_SIZE // 4
    # -------------------------------------------------------------------------

    # Kinds of analysis output to write; asking for a figure also asks for the analysis outputs it is plotted from.
    analysis_output_kinds = []
    if SEQUENTIAL_PRESENTATION:
        if WRITE_ACTIVATIONS_DEVIANT or WRITE_PROBABILITIES or PLOT_DEVIANT:
            raise ValueError("SEQUENTIAL_PRESENTATION = True only writes the activations "
                             "CSVs for now: set WRITE_ACTIVATIONS_DEVIANT, "
                             "WRITE_PROBABILITIES and PLOT_DEVIANT to False")
        if WRITE_ACTIVATIONS or PLOT_TRAJECTORIES or PLOT_BY_POSITION:
            analysis_output_kinds.append('pool_activations')
    else:
        if WRITE_ACTIVATIONS or PLOT_TRAJECTORIES or PLOT_BY_POSITION:
            analysis_output_kinds.append('activations')
        if WRITE_ACTIVATIONS_DEVIANT or PLOT_DEVIANT:
            analysis_output_kinds.append('activations_deviant')
        if WRITE_PROBABILITIES:
            analysis_output_kinds.append('probabilities')
            analysis_output_kinds.append('probabilities_deviant')

    trial_files = find_trial_files(TRIALS_PATH)
    if not trial_files:
        raise FileNotFoundError(f"No .csv or .txt sequence files in {TRIALS_PATH}")
    print(f"Found {len(trial_files)} trial sequence files in {TRIALS_PATH}")
    if SEQUENTIAL_PRESENTATION:
        pools = make_sequence_pools(trial_files, POOL_SEED)
        n_left_out = len(trial_files) - len(pools) * len(pools[0])
        print(f"{len(pools)} pools of {len(pools[0])} sequences, one per tau_std value; "
              f"{n_left_out} sequence file(s) left out")
        # The activity figures are plotted from the pool CSVs, and named after the pools.
        figure_suffix = f'_pools_seed{POOL_SEED}'
        sample_name = 'pools'
    else:
        figure_suffix = ''
        sample_name = 'sequences'
    print(f"Analysis outputs: {', '.join(analysis_output_kinds)}")
    print(f"Models ({len(MODEL_NAMES)}): {', '.join(MODEL_NAMES)}")

    for i, model_name in enumerate(MODEL_NAMES, start=1):
        print(f"\n{'=' * 79}\n[{i}/{len(MODEL_NAMES)}] {model_name}\n{'=' * 79}")
        # Only the config is read here; the weights are loaded when a forward pass is due.
        info = ModelInfo.from_path(MODEL_DIR / model_name)

        if SEQUENTIAL_PRESENTATION:
            # Activations CSVs of the pools, which the activity figures are plotted from.
            act_csvs = []
            pools_missing_csvs = []
            for pool_index in range(len(pools)):
                pool_csv = sequence_csv_path(OUTPUT_ROOT, model_name, 'pool_activations',
                                             pool_name(pool_index, POOL_SEED))
                act_csvs.append(pool_csv)
                # Rewritten when older than one of its sequence files or the weights.
                sources = pools[pool_index] + [info.weights_path]
                pool_csv_outdated = OVERWRITE or not outputs_up_to_date([pool_csv], sources)
                if 'pool_activations' in analysis_output_kinds and pool_csv_outdated:
                    pools_missing_csvs.append(pool_index)
            # The sequences of a pool have different parameters: no legend in the
            # figure of the individual trajectories.
            legend_trial_files = None
            print(f"{len(pools_missing_csvs)}/{len(pools)} pool(s) need CSVs written")
            if pools_missing_csvs:
                model = load_model(info)
                model.eval()
                print(f"Loaded model: {model_name}")
                write_pool_activations_csvs(model, info, pools, pools_missing_csvs, POOL_SEED,
                                            OUTPUT_ROOT, model_name, CUE_SEED, POOL_CHUNK_SIZE,
                                            INCLUDE_PRIOR)
        else:
            # Activations CSVs of the sequences, which the activity figures are plotted from.
            act_csvs = []
            for trial_file in trial_files:
                act_csvs.append(sequence_csv_path(OUTPUT_ROOT, model_name, 'activations',
                                                  trial_file.stem))
            legend_trial_files = trial_files
            files_missing_csvs = [f for f in trial_files
                                  if analysis_output_kinds_to_write(f, analysis_output_kinds,
                                                                    OUTPUT_ROOT, model_name,
                                                                    info.weights_path, OVERWRITE)]
            print(f"{len(files_missing_csvs)}/{len(trial_files)} sequence file(s) need CSVs written")
            if files_missing_csvs:
                model = load_model(info)
                model.eval()
                print(f"Loaded model: {model_name}")
                write_proba_and_activ_csvs(model, info, files_missing_csvs, analysis_output_kinds,
                                           OUTPUT_ROOT, model_name, PERIOD, CUE_SEED, CHUNK_SIZE,
                                           INCLUDE_PRIOR, INCLUDE_NEXT_STIMULUS, OVERWRITE)

        if PLOT_TRAJECTORIES:
            plot_trajectories(act_csvs, legend_trial_files, OUTPUT_ROOT, model_name,
                              figure_suffix, N_TRAJECTORIES, MAX_INDIVIDUAL_TRAJECTORIES, SEED,
                              OVERWRITE)
        if PLOT_BY_POSITION:
            plot_by_position(act_csvs, OUTPUT_ROOT, model_name, figure_suffix, sample_name,
                             N_SEQUENCES, SEED, PERIOD, OVERWRITE)
        if PLOT_DEVIANT:
            plot_deviant(info, trial_files, OUTPUT_ROOT, model_name, N_SEQUENCES, SEED,
                         OVERWRITE)
