"""Unified driver for the experimental-sequence model analyses.

Runs, for each of a list of trained models and one directory of experimental
trial-sequence files, any subset of six analyses. Paths below are relative to
``<output-root>/<model name>/``, one tree per model.

  CSV extraction (one row-set per sequence file)
    activations         per-timestep module activity norms + derivatives
                        -> activations/*_activations.csv
    activations_deviant one row per trial, sampled at the deviant timestep
                        -> activations_deviant/*_deviant_trial.csv
    probabilities       per-module predicted distributions + ground-truth
                        likelihoods, plus a deviant-only subset
                        -> probabilities/*_probabilities.csv
                        -> probabilities_deviant/*_probabilities_deviant.csv

  Figures (-> viz_examples/)
    plot_trajectories   individual + averaged activity and derivatives for a
                        small random sample of sequences
    plot_by_position    activity / derivatives averaged over sequences, split by
                        within-trial position
    plot_deviant        activity / derivatives at the deviant timestep, grouped
                        by deviant position

Each model is loaded once for all of its stages, each sequence file is read once
per group of stages, and the batched forward passes needed by plot_by_position
and plot_deviant are shared instead of being recomputed twice. Models are
independent, so a model that fails (a missing checkpoint, say) is reported at the
end without stopping the rest.

This is the one script here that keeps a command line, because it is the heavy
batch driver; every flag defaults to the corresponding value in analysis_config.
The stage implementations live in exp_sequence_analysis.py and plots.py.

Examples
--------
    # every stage, for every model in analysis_config.DEFAULT_MODEL_NAMES
    python run_exp_trials_pipeline.py

    # just the CSV extraction, for one model
    python run_exp_trials_pipeline.py \
        --stages activations activations_deviant probabilities \
        --model-names population_network_all_bn8_trainh0_fixedsir0.02_epochs300_lr0.002

    # just the figures, on 200 randomly sampled sequences
    python run_exp_trials_pipeline.py \
        --stages plot_trajectories plot_by_position plot_deviant --n-sequences 200
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import analysis_config as cfg_mod
import exp_sequence_analysis as exp
import plots
from analysis_config import MODULE_TITLES
from analysis_core import (
    ModelInfo,
    dpos_conventions,
    find_trial_files,
    get_module_output_and_activity,
    get_module_probabilities,
    load_model,
    load_trial_params,
    load_trial_sequence,
    select_files,
    to_model_tensors,
)

CSV_STAGES = ('activations', 'activations_deviant', 'probabilities')
PLOT_STAGES = ('plot_trajectories', 'plot_by_position', 'plot_deviant')
ALL_STAGES = CSV_STAGES + PLOT_STAGES

# plot_trajectories draws one panel row per sequence, so it uses a small sample.
DEFAULT_N_TRAJECTORIES = 2


# =============================================================================
# Stage group 1: per-file CSV extraction
# =============================================================================

def run_csv_stages(model, info, trial_files, stages, output_root, cfg):
    """Run the per-file CSV stages, reading each sequence file once.

    Any of 'activations', 'activations_deviant' and 'probabilities' present in
    `stages` are produced in the same loop: one activity forward pass (shared by
    the two activation stages) and one probability forward pass per file, only
    when the corresponding stage was requested.
    """
    want_act = 'activations' in stages
    want_act_dev = 'activations_deviant' in stages
    want_prob = 'probabilities' in stages
    need_activity = want_act or want_act_dev

    dirs = {}
    if want_act:
        dirs['activations'] = output_root / 'activations'
    if want_act_dev:
        dirs['activations_deviant'] = output_root / 'activations_deviant'
    if want_prob:
        dirs['probabilities'] = output_root / 'probabilities'
        # Deviant-only output: same information but restricted to rows where the
        # context label (trial_type) is 1.
        dirs['probabilities_deviant'] = output_root / 'probabilities_deviant'
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

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

    for trial_file in trial_files:
        # ctx/dpos/rule are the ground-truth labels from the original sequence.
        # dpos_raw is the raw deviant position on disk (e.g. 2..6), not yet a class index.
        obs, cue, ctx, dpos_raw, rule, lim_std, d, tau_std, trial_n = load_trial_sequence(
            trial_file, return_hierarch=True,
            n_cue_classes=n_cue_classes, cue_seed=cfg.cue_seed)
        y, q = to_model_tensors(obs, cue)

        if need_activity:
            # hidden_activity:    dict module → (T-1, 1)
            # hidden_derivatives: dict module → (T-2, 1)
            _, hidden_activity, hidden_derivatives = get_module_output_and_activity(model, y, q)
            # Squeeze batch dim (batch=1) from norms and derivatives
            norms = {name: arr[:, 0] for name, arr in hidden_activity.items()}      # each: (T-1,)
            derivs = {name: arr[:, 0] for name, arr in hidden_derivatives.items()}  # each: (T-2,)

        if want_act:
            out_df = exp.build_activations_frame(obs, cue, norms, derivs,
                                             lim_std, d, tau_std, trial_n)
            out_file = dirs['activations'] / (trial_file.stem + '_activations.csv')
            out_df.to_csv(out_file, index=False)
            print(f"  Saved: {out_file.name}")

        if want_act_dev:
            out_df = exp.build_deviant_activations_frame(
                obs, dpos_raw, norms, derivs, lim_std, d, tau_std, trial_n,
                period=cfg.period, dpos_shift=dpos_shift)
            out_file = dirs['activations_deviant'] / (trial_file.stem + '_deviant_trial.csv')
            out_df.to_csv(out_file, index=False)
            print(f"  Saved: {out_file.name}")

        if want_prob:
            # probs: dict module → (seq_len, batch, dim)
            #   'obs':  dim=2, columns are (mean, variance) of the predicted Gaussian
            #   others: dim=n_classes, softmax class probabilities
            # With --include-prior, the same quantities are also read from the prior
            # (first-call) readout of each module, i.e. before the feedback sweep.
            if cfg.include_prior:
                probs, prior_probs = get_module_probabilities(model, y, q, return_prior=True)
            else:
                probs = get_module_probabilities(model, y, q)
                prior_probs = None

            # Shift the experimental deviant-position labels into the model's training
            # convention ({2..6} -> {3..7}); the on-disk files are left untouched.
            dpos_model = dpos_raw + dpos_shift

            out_df = exp.build_probabilities_frame(obs, cue, ctx, dpos_model, rule, probs,
                                               lim_std, d, tau_std, trial_n, dpos_min,
                                               prior_probs=prior_probs)
            out_file = dirs['probabilities'] / (trial_file.stem + '_probabilities.csv')
            out_df.to_csv(out_file, index=False)
            print(f"  Saved: {out_file.name}")

            deviant_df = exp.subset_deviant_rows(out_df, cfg.include_next_stimulus)
            deviant_file = (dirs['probabilities_deviant']
                            / (trial_file.stem + '_probabilities_deviant.csv'))
            deviant_df.to_csv(deviant_file, index=False)
            print(f"  Saved: {deviant_file.name}")

    print(f"\n{len(trial_files)} file(s) processed for stages: "
          f"{', '.join(s for s in CSV_STAGES if s in stages)}")
    for name, path in dirs.items():
        print(f"  {name}: {path}")


def run_plot_trajectories(model, n_cue_classes, all_files, output_dir, model_name, cfg):
    """Individual + averaged activity and derivatives.

    Draws one panel row per sequence, so it runs on a small random sample
    (``--n-trajectories``) rather than the full set.
    """
    selected_files = select_files(all_files, cfg.n_trajectories, seed=cfg.seed)
    n_select = len(selected_files)
    print(f"[plot_trajectories] using {n_select} trial sequence files")

    # Sequences may differ in length; we truncate all to the shortest one.
    obs_list, cue_list, params_list = [], [], []
    for f in selected_files:
        obs, cue, lim_std, d, tau_std, trial_n = load_trial_sequence(
            f, n_cue_classes=n_cue_classes, cue_seed=cfg.cue_seed)
        obs_list.append(obs)
        cue_list.append(cue)
        params_list.append(load_trial_params(f))

    # Build pars dict in the format expected by extract_sample_parameters
    pars = {
        'tau':     [p['tau']     for p in params_list],
        'lim':     [p['lim']     for p in params_list],
        'si_stat': [p['si_stat'] for p in params_list],
        'si_r':    [p['si_r']    for p in params_list],
    }

    min_len = min(o.shape[0] for o in obs_list)
    if any(o.shape[0] != min_len for o in obs_list):
        print(f"  Warning: sequences have unequal lengths — truncating all to {min_len} timesteps")

    obs_stack = np.stack([o[:min_len] for o in obs_list], axis=0)        # (N, T)
    cue_stack = np.stack([c[:min_len, :] for c in cue_list], axis=0)     # (N, T, n_cue)

    y = torch.tensor(obs_stack, dtype=torch.float32).unsqueeze(-1)       # (N, T, 1)
    q = torch.tensor(cue_stack, dtype=torch.float32)                     # (N, T, n_cue)

    # Forward pass — returns norms (T-1, N) and derivatives (T-2, N)
    _, module_norms_dict, _ = get_module_output_and_activity(model, y, q)

    seq_len = next(iter(module_norms_dict.values())).shape[0]
    timesteps = np.arange(seq_len)

    # Figure 1: Individual trajectories (activity)
    fig = plots.plot_individual_trajectories(
        module_norms_dict, MODULE_TITLES, timesteps,
        output_dir, model_name, include_derivatives=False, pars=pars,
    )
    plots.save_figure(fig, output_dir, f"{model_name}_exp_activity_trajectories.png")

    # Figure 2: Averaged activity
    fig = plots.plot_averaged_activity(
        module_norms_dict, MODULE_TITLES, timesteps,
        output_dir, model_name, n_samples=n_select, include_derivatives=False,
    )
    plots.save_figure(fig, output_dir, f"{model_name}_exp_activity_averaged.png")

    # Figure 3: Individual trajectories (derivatives)
    fig = plots.plot_individual_trajectories(
        module_norms_dict, MODULE_TITLES, timesteps,
        output_dir, model_name, include_derivatives=True, pars=pars,
    )
    plots.save_figure(fig, output_dir, f"{model_name}_exp_activity_trajectories_derivatives.png")

    # Figure 4: Averaged derivatives
    fig = plots.plot_averaged_activity(
        module_norms_dict, MODULE_TITLES, timesteps,
        output_dir, model_name, n_samples=n_select, include_derivatives=True,
    )
    plots.save_figure(fig, output_dir, f"{model_name}_exp_activity_averaged_derivatives.png")


def run_plot_by_position(module_norms_dict, n_select, output_dir, model_name, cfg):
    """Activity / derivatives averaged over sequences, split by within-trial position."""
    # Figure 1: Averaged activity by within-trial position
    fig = plots.plot_averaged_activity_by_position(
        module_norms_dict, MODULE_TITLES, output_dir, model_name,
        n_samples=n_select, period=cfg.period, include_derivatives=False,
    )
    plots.save_figure(fig, output_dir, f"{model_name}_exp_activity_averaged_by_position.png")

    # Figure 2: Averaged derivatives by within-trial position
    fig = plots.plot_averaged_activity_by_position(
        module_norms_dict, MODULE_TITLES, output_dir, model_name,
        n_samples=n_select, period=cfg.period, include_derivatives=True,
    )
    plots.save_figure(fig, output_dir, f"{model_name}_exp_activity_averaged_by_position_derivatives.png")


def run_plot_deviant(module_norms_dict, dev_pos, n_select, output_dir, model_name, cfg):
    """Activity / derivatives at the deviant timestep, grouped by deviant position."""
    # Figure 1: Activity at the deviant position
    fig = plots.plot_deviant_activity_by_position(
        module_norms_dict, dev_pos, MODULE_TITLES, output_dir, model_name,
        n_samples=n_select, period=cfg.period, include_derivatives=False,
    )
    plots.save_figure(fig, output_dir, f"{model_name}_exp_deviant_activity_by_position.png")

    # Figure 2: Derivatives at the deviant position
    fig = plots.plot_deviant_activity_by_position(
        module_norms_dict, dev_pos, MODULE_TITLES, output_dir, model_name,
        n_samples=n_select, period=cfg.period, include_derivatives=True,
    )
    plots.save_figure(fig, output_dir, f"{model_name}_exp_deviant_activity_by_position_derivatives.png")


def run_plot_stages(model, info, all_files, stages, output_root, model_name, cfg):
    """Run the requested figure stages.

    plot_by_position and plot_deviant both need per-module norms over the same
    (large) file selection, so when both are requested the chunked forward passes
    are done once and shared.
    """
    output_dir = output_root / 'viz_examples'
    output_dir.mkdir(parents=True, exist_ok=True)

    # If the model was trained with a larger cue vocabulary than the two cues present
    # in the experimental sequences, re-encode the cues into the dimensionality it expects.
    n_cue_classes = len(info.data_config_dict['cues_set'])

    if 'plot_trajectories' in stages:
        run_plot_trajectories(model, n_cue_classes, all_files, output_dir, model_name, cfg)

    want_by_pos = 'plot_by_position' in stages
    want_dev = 'plot_deviant' in stages
    if want_by_pos or want_dev:
        selected_files = select_files(all_files, cfg.n_sequences, seed=cfg.seed)
        n_select = len(selected_files)
        print(f"[plot_by_position/plot_deviant] using {n_select} trial sequence files")

        module_norms_dict, dev_pos = exp.compute_norms_for_files(
            model, selected_files, period=cfg.period, chunk_size=cfg.chunk_size,
            n_cue_classes=n_cue_classes, cue_seed=cfg.cue_seed, return_devpos=want_dev)

        if want_by_pos:
            run_plot_by_position(module_norms_dict, n_select, output_dir, model_name, cfg)
        if want_dev:
            run_plot_deviant(module_norms_dict, dev_pos, n_select, output_dir, model_name, cfg)

    print(f"\nFigures saved to {output_dir}")


def run_one_model(model_name, trial_files, stages, cfg):
    """Run every requested stage for one model, into <output-root>/<model_name>/."""
    model_path = cfg.model_dir / model_name
    output_root = cfg.output_root / model_name
    output_root.mkdir(parents=True, exist_ok=True)

    # Load the model once for every stage.
    info = ModelInfo.from_path(model_path)
    model = load_model(info)
    model.eval()
    print(f"Loaded model: {model_path.name}")
    print(f"Output root: {output_root}")

    if any(s in stages for s in CSV_STAGES):
        run_csv_stages(model, info, trial_files, stages, output_root, cfg)

    if any(s in stages for s in PLOT_STAGES):
        run_plot_stages(model, info, trial_files, stages, output_root, model_name, cfg)


# =============================================================================
# Entry point
# =============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stages', nargs='+', choices=ALL_STAGES, default=list(ALL_STAGES),
                   metavar='STAGE',
                   help=f"stages to run, any of: {', '.join(ALL_STAGES)} (default: all)")
    p.add_argument('--model-names', nargs='+', default=list(cfg_mod.DEFAULT_MODEL_NAMES),
                   metavar='NAME',
                   help='trained model directory names, run in order (default: the '
                        f'{len(cfg_mod.DEFAULT_MODEL_NAMES)} models in '
                        'analysis_config.DEFAULT_MODEL_NAMES)')
    p.add_argument('--model-dir', type=Path, default=cfg_mod.TRAINING_RESULTS_DIR,
                   help='directory containing the trained models (default: %(default)s)')
    p.add_argument('--trials-path', type=Path, default=cfg_mod.TRIALS_PATH,
                   help='directory of experimental sequence files (default: %(default)s)')
    p.add_argument('--output-root', type=Path, default=cfg_mod.EXP_SEQ_OUTPUT_ROOT,
                   help='base output directory; each model writes to '
                        '<output-root>/<model-name>/ (default: %(default)s)')
    p.add_argument('--n-sequences', type=int, default=None,
                   help='sequence files used by plot_by_position / plot_deviant '
                        '(default: all of them)')
    p.add_argument('--n-trajectories', type=int, default=DEFAULT_N_TRAJECTORIES,
                   help='sequence files used by plot_trajectories (default: %(default)s)')
    p.add_argument('--seed', type=int, default=0,
                   help='seed for random file sampling (default: %(default)s)')
    p.add_argument('--period', type=int, default=cfg_mod.PERIOD,
                   help='timesteps per trial (default: %(default)s)')
    p.add_argument('--cue-seed', type=int, default=cfg_mod.CUE_SEED,
                   help='seed for cue re-encoding; keep it fixed across stages '
                        '(default: %(default)s)')
    p.add_argument('--chunk-size', type=int, default=cfg_mod.CHUNK_SIZE,
                   help='sequences per batched forward pass (default: %(default)s)')
    p.add_argument('--no-next-stimulus', dest='include_next_stimulus',
                   action='store_false',
                   help='restrict the deviant-only probabilities file to the deviant '
                        'rows, instead of the deviant plus the immediate next stimulus')
    p.add_argument('--include-prior', action='store_true',
                   help='also write the prior (first-call) readout of every module to '
                        'the probability CSVs, as prior_-prefixed columns; the '
                        'unprefixed columns stay the posterior readouts')
    p.add_argument('--parallel', action='store_true',
                   help='mark this run as one of several concurrent workers sharing a '
                        'figure directory, so figure provenance is appended under a lock')
    return p.parse_args(argv)


if __name__ == '__main__':
    cfg = parse_args()
    plots.set_script_path(__file__, parallel=cfg.parallel)

    stages = [s for s in ALL_STAGES if s in set(cfg.stages)]   # canonical order

    trial_files = find_trial_files(cfg.trials_path)
    if not trial_files:
        raise FileNotFoundError(f"No .csv or .txt sequence files in {cfg.trials_path}")
    print(f"Found {len(trial_files)} trial sequence files in {cfg.trials_path}")
    print(f"Stages: {', '.join(stages)}")
    print(f"Models ({len(cfg.model_names)}): {', '.join(cfg.model_names)}")

    # A failure on one model (a missing checkpoint, say) should not throw away the
    # models already done or block the ones still queued; report at the end instead.
    failures = []
    for i, model_name in enumerate(cfg.model_names, start=1):
        print(f"\n{'=' * 79}\n[{i}/{len(cfg.model_names)}] {model_name}\n{'=' * 79}")
        try:
            run_one_model(model_name, trial_files, stages, cfg)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failures.append((model_name, exc))

    n_ok = len(cfg.model_names) - len(failures)
    print(f"\nDone: {n_ok}/{len(cfg.model_names)} model(s) completed.")
    for model_name, exc in failures:
        print(f"  FAILED {model_name}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit(1)
