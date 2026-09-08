"""Unified driver for the experimental-sequence model analyses.

Runs, for each of a list of trained models and one directory of experimental
trial-sequence files, any subset of the six analyses that were previously six
separate scripts:

  (paths below are relative to <output-root>/<model name>/, one tree per model)

  CSV extraction (one row-set per sequence file)
    activations         <- model_act_exp_trials.py
                           per-timestep module activity norms + derivatives
                           -> activations/*_activations.csv
    activations_deviant <- model_act_exp_trials_deviant.py
                           one row per trial, sampled at the deviant timestep
                           -> activations_deviant/*_deviant_trial.csv
    probabilities       <- model_prob_exp_trials.py
                           per-module predicted distributions + ground-truth
                           likelihoods, plus a deviant-only subset
                           -> probabilities/*_probabilities.csv
                           -> probabilities_deviant/*_probabilities_deviant.csv

  Figures (-> viz_examples/)
    plot_trajectories   <- plot_exp_trial_activity.py
                           individual + averaged activity and derivatives for a
                           small random sample of sequences
    plot_by_position    <- plot_exp_trial_activity_by_position.py
                           activity / derivatives averaged over sequences, split
                           by within-trial position
    plot_deviant        <- plot_exp_trial_deviant_activity.py
                           activity / derivatives at the deviant timestep,
                           grouped by deviant position

The numerical content of every output is identical to the original scripts. What
changes is bookkeeping: each model is loaded once for all of its stages, each
sequence file is read once per group of stages, and the batched forward passes
needed by plot_by_position and plot_deviant are shared instead of being
recomputed twice. Models are independent of each other, so a model that fails
(a missing checkpoint, say) is reported at the end without stopping the rest.

Configuration lives in the DEFAULTS block below (same style as the original
scripts); every field can also be overridden on the command line.

Examples
--------
    # every stage, for every model in DEFAULT_MODEL_NAMES
    python run_exp_trials_pipeline.py

    # just the CSV extraction, for one model
    python run_exp_trials_pipeline.py --stages activations activations_deviant probabilities \
        --model-names population_network_all_bn8_trainh0_fixedsir_lr0.002_epochs200_lrsched

    # just the figures, on 200 randomly sampled sequences, for two models
    python run_exp_trials_pipeline.py --stages plot_trajectories plot_by_position plot_deviant \
        --n-sequences 200 \
        --model-names population_network_all_bn8_trainh0_fixedsir_lr0.002_epochs300 \
                      population_network_all_bn8_trainh0_fixedsir0.05_epochs300_lr0.002
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import evaluate_models as eval

from model_activations import (
    get_module_output_and_activity,
    get_module_probabilities,
    load_trial_sequence,
    load_trial_params,
    to_model_tensors,
    gaussian_likelihood,
    class_likelihood,
    dpos_conventions,
    plot_individual_trajectories,
    plot_averaged_activity,
    plot_averaged_activity_by_position,
    plot_deviant_activity_by_position,
)


# =============================================================================
# Defaults (override on the command line; see --help)
# =============================================================================

DEFAULT_MODEL_DIR = Path(
    "/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/training_results/N_ctx_2/HierarchicalGM")

# Models to run, in order. Each gets its own output tree under
# <output-root>/<model name>/, so the stages of one model never overwrite another's.
# The original scripts had drifted onto different checkpoints (epochs300 for the
# probabilities, epochs200_lrsched for the activations, lr0.001_dposweight for the
# figures); listing the models here is what makes that unnecessary. Matches the
# MODEL_DIRS list in evaluate_models.py.
DEFAULT_MODEL_NAMES = [
    # "population_network_all_bn8_trainh0_fixedsir_lr0.002_epochs200_lrsched",
    "population_network_all_bn8_trainh0_fixedsir0.02_epochs300_lr0.002",
    # "population_network_all_bn8_trainh0_fixedsir0.05_epochs300_lr0.002",
    # "population_network_all_bn8_trainh0_fixedsir0.005_epochs300_lr0.002",
    # "population_network_all_bn8_trainh0_fixedsir0.1_epochs300_lr0.002",
]

DEFAULT_TRIALS_PATH = Path("/home/clevyfidel/Documents/Workspace/Jasmin/trialsequences2clem")
DEFAULT_OUTPUT_ROOT = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output")

# Timesteps per trial. The within-trial position of timestep t is t % PERIOD and
# its trial index is t // PERIOD; the deviant of trial t sits at t * PERIOD + dpos.
DEFAULT_PERIOD = 8

# Cue re-encoding seed. Must be the same across stages so that the activity norms
# and the likelihoods refer to the same cue encoding (they are merged per trial
# downstream in exp_trial_join_act_proba.py).
DEFAULT_CUE_SEED = 11

# Sequences per batched forward pass in the plotting stages (bounds hidden-state memory).
DEFAULT_CHUNK_SIZE = 128

# plot_trajectories draws one panel row per sequence, so it uses a small sample.
DEFAULT_N_TRAJECTORIES = 2

MODULE_TITLES = {
    'obs':  'Observation module',
    'ctx':  'Type module',
    'dpos': 'Deviant position module',
    'rule': 'Rule module',
}

CSV_STAGES = ('activations', 'activations_deviant', 'probabilities')
PLOT_STAGES = ('plot_trajectories', 'plot_by_position', 'plot_deviant')
ALL_STAGES = CSV_STAGES + PLOT_STAGES


# =============================================================================
# Helpers
# =============================================================================

def find_trial_files(trials_path):
    """Sequence files in `trials_path`, preferring .csv and falling back to .txt."""
    files = sorted(trials_path.glob("*.csv"))
    if not files:
        files = sorted(trials_path.glob("*.txt"))
    return files


def select_files(all_files, n_sequences, seed=0):
    """All files, or `n_sequences` of them sampled without replacement."""
    if n_sequences is None or n_sequences >= len(all_files):
        return list(all_files)
    rng = np.random.default_rng(seed=seed)
    return rng.choice(all_files, size=n_sequences, replace=False).tolist()


def gather_at(arr, idx, length):
    """Gather arr[idx], returning NaN where idx falls outside [0, length).

    Used to sample one value per trial at the trial's deviant timestep; the very
    last within-trial position of the last trial can fall past the activity array
    (which is one step shorter than the raw sequence), so it becomes NaN.
    """
    out = np.full(len(idx), np.nan, dtype=float)
    valid = idx < length
    out[valid] = arr[idx[valid]]
    return out


# =============================================================================
# Stage group 1: per-file CSV extraction
# =============================================================================

def build_activations_frame(obs, cue, norms, derivs, lim_std, d, tau_std, trial_n):
    """Per-timestep activity CSV (model_act_exp_trials.py).

    Rows are the T-1 model output timesteps; the sequence columns are the raw
    (unshifted) observation and cue at those same timesteps.
    """
    n_out = next(iter(norms.values())).shape[0]

    obs_out = obs[:n_out]
    # `cue` may be a (T, n_cue_classes) one-hot where only the two cues used in this
    # sequence are ever active. Recover those two columns (ascending class order,
    # matching CUE_LABELS order) so the CSV keeps its original two-cue encoding.
    used_cols = np.flatnonzero(cue.any(axis=0))
    cue_out = cue[:n_out][:, used_cols]  # (n_out, 2)

    # Pad derivatives with NaN at the last position so all columns have length T-1
    n_deriv = next(iter(derivs.values())).shape[0]
    pad = np.full(n_out - n_deriv, np.nan)
    derivs_padded = {name: np.concatenate([arr, pad]) for name, arr in derivs.items()}

    out_df = pd.DataFrame({
        'observation':  obs_out,
        'cue_1':        cue_out[:, 0].astype(int),
        'cue_2':        cue_out[:, 1].astype(int),
        'obs_norm':     norms['obs'],
        'ctx_norm':     norms['ctx'],
        'dpos_norm':    norms['dpos'],
        'rule_norm':    norms['rule'],
        'obs_deriv':    derivs_padded['obs'],
        'ctx_deriv':    derivs_padded['ctx'],
        'dpos_deriv':   derivs_padded['dpos'],
        'rule_deriv':   derivs_padded['rule'],
    })
    out_df['lim_std'] = lim_std
    out_df['d'] = d
    out_df['tau_std'] = tau_std
    out_df['trial_n'] = trial_n.values[:n_out]
    return out_df


def build_deviant_activations_frame(obs, dpos_raw, norms, derivs, lim_std, d, tau_std,
                                    trial_n, period, dpos_shift):
    """One row per trial, sampled at the deviant timestep (model_act_exp_trials_deviant.py).

    Two distinct uses of dpos are kept separate:
      * the PHYSICAL within-trial timestep of the deviant uses the RAW experimental
        position (that is literally where the deviant tone sits in the sequence);
      * the stored deviant_pos LABEL is shifted into the model's convention so it
        matches the probabilities stage's dpos column and the model's class mapping.
    """
    n_out = next(iter(norms.values())).shape[0]     # T-1
    n_deriv = next(iter(derivs.values())).shape[0]  # T-2

    # dpos is constant within a trial, so take the first timestep of each trial.
    T = obs.shape[0]
    n_trials = T // period
    trial_ids = trial_n.to_numpy()[::period]                  # (n_trials,)
    # RAW physical within-trial position {2..6}: used to index the deviant's timestep.
    deviant_pos_phys = dpos_raw[::period]                     # (n_trials,)
    dev_idx = np.arange(n_trials) * period + deviant_pos_phys  # global timestep of deviant
    # Stored LABEL in the model convention {3..7} (raw + shift) for cross-stage consistency.
    deviant_pos = deviant_pos_phys + dpos_shift

    out_df = pd.DataFrame({
        'trial_n':      trial_ids,
        'deviant_pos':  deviant_pos,
        'obs_norm':     gather_at(norms['obs'],  dev_idx, n_out),
        'ctx_norm':     gather_at(norms['ctx'],  dev_idx, n_out),
        'dpos_norm':    gather_at(norms['dpos'], dev_idx, n_out),
        'rule_norm':    gather_at(norms['rule'], dev_idx, n_out),
        'obs_deriv':    gather_at(derivs['obs'],  dev_idx, n_deriv),
        'ctx_deriv':    gather_at(derivs['ctx'],  dev_idx, n_deriv),
        'dpos_deriv':   gather_at(derivs['dpos'], dev_idx, n_deriv),
        'rule_deriv':   gather_at(derivs['rule'], dev_idx, n_deriv),
    })
    out_df['lim_std'] = lim_std
    out_df['d'] = d
    out_df['tau_std'] = tau_std
    return out_df


def build_probabilities_frame(obs, cue, ctx, dpos_model, rule, probs,
                              lim_std, d, tau_std, trial_n, dpos_min):
    """Predicted distributions + ground-truth likelihoods (model_prob_exp_trials.py).

    `dpos_model` must already be in the model's convention (raw + shift), so that
    dpos_model - dpos_min is a valid 0-based class index.
    """
    # Number of timesteps in the output: T-1 (the model is run on y[:, :-1]).
    # The model output at index k predicts timestep k+1, so the outputs align
    # with the ground truth shifted by one: y[1:], q[1:], labels[1:].
    n_out = next(iter(probs.values())).shape[0]

    # Next-step prediction alignment: ground truth that each output row is predicting.
    obs_gt = obs[1:n_out + 1]           # (n_out,)
    cue_gt = cue[1:n_out + 1, :]        # (n_out, 2)
    ctx_gt = ctx[1:n_out + 1]           # (n_out,)
    dpos_gt = dpos_model[1:n_out + 1]   # (n_out,)
    rule_gt = rule[1:n_out + 1]         # (n_out,)

    # Squeeze batch dim (batch=1) from each module's probabilities
    probs = {name: arr[:, 0, :] for name, arr in probs.items()}  # each: (n_out, dim)

    # Sequence info columns (same fields as the activations stage), now carrying the
    # next-step (prediction-target) values so every column in a row refers to the
    # same timestep as that row's likelihoods.
    out_df = pd.DataFrame({
        'observation':  obs_gt,
        'cue_1':        cue_gt[:, 0].astype(int),
        'cue_2':        cue_gt[:, 1].astype(int),
        'ctx':          ctx_gt,
        'dpos':         dpos_gt,
        'rule':         rule_gt,
    })

    # Observation module is a regressor: mean and variance of the Gaussian
    out_df['obs_mean'] = probs['obs'][:, 0]
    out_df['obs_var'] = probs['obs'][:, 1]

    # Classifier modules: one column per class probability
    for module_name in ('ctx', 'dpos', 'rule'):
        module_probs = probs[module_name]  # (n_out, n_classes)
        for c in range(module_probs.shape[1]):
            out_df[f'{module_name}_p{c}'] = module_probs[:, c]

    # --- Likelihood of the ground truth under each module's distribution ---
    # obs: Gaussian density of the true observation under (mean, variance).
    out_df['lik_obs'] = gaussian_likelihood(obs_gt, probs['obs'][:, 0], probs['obs'][:, 1])

    # Classifiers: probability assigned to the true class. ctx and rule labels
    # are already 0-based; dpos must be shifted by its minimum to index into
    # the class-probability columns (matches the dpos_min convention used in
    # training/eval, e.g. pipeline_core_v2 get_model_predictions).
    out_df['lik_ctx'] = class_likelihood(probs['ctx'], ctx_gt)
    out_df['lik_dpos'] = class_likelihood(probs['dpos'], dpos_gt - dpos_min)
    out_df['lik_rule'] = class_likelihood(probs['rule'], rule_gt)

    out_df['lim_std'] = lim_std
    out_df['d'] = d
    out_df['tau_std'] = tau_std
    out_df['trial_n'] = trial_n.values[1:n_out + 1]
    return out_df


def subset_deviant_rows(out_df, include_next_stimulus):
    """Deviant-only subset of a probabilities frame (rows whose context label is 1)."""
    deviant_mask = out_df['ctx'] == 1
    if include_next_stimulus:
        # Deviants + the immediate next stimulus.
        deviant_mask = deviant_mask | deviant_mask.shift(1, fill_value=False)
    return out_df[deviant_mask]


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
            out_df = build_activations_frame(obs, cue, norms, derivs,
                                             lim_std, d, tau_std, trial_n)
            out_file = dirs['activations'] / (trial_file.stem + '_activations.csv')
            out_df.to_csv(out_file, index=False)
            print(f"  Saved: {out_file.name}")

        if want_act_dev:
            out_df = build_deviant_activations_frame(
                obs, dpos_raw, norms, derivs, lim_std, d, tau_std, trial_n,
                period=cfg.period, dpos_shift=dpos_shift)
            out_file = dirs['activations_deviant'] / (trial_file.stem + '_deviant_trial.csv')
            out_df.to_csv(out_file, index=False)
            print(f"  Saved: {out_file.name}")

        if want_prob:
            # probs: dict module → (seq_len, batch, dim)
            #   'obs':  dim=2, columns are (mean, variance) of the predicted Gaussian
            #   others: dim=n_classes, softmax class probabilities
            probs = get_module_probabilities(model, y, q)

            # Shift the experimental deviant-position labels into the model's training
            # convention ({2..6} -> {3..7}); the on-disk files are left untouched.
            dpos_model = dpos_raw + dpos_shift

            out_df = build_probabilities_frame(obs, cue, ctx, dpos_model, rule, probs,
                                               lim_std, d, tau_std, trial_n, dpos_min)
            out_file = dirs['probabilities'] / (trial_file.stem + '_probabilities.csv')
            out_df.to_csv(out_file, index=False)
            print(f"  Saved: {out_file.name}")

            deviant_df = subset_deviant_rows(out_df, cfg.include_next_stimulus)
            deviant_file = (dirs['probabilities_deviant']
                            / (trial_file.stem + '_probabilities_deviant.csv'))
            deviant_df.to_csv(deviant_file, index=False)
            print(f"  Saved: {deviant_file.name}")

    print(f"\n{len(trial_files)} file(s) processed for stages: "
          f"{', '.join(s for s in CSV_STAGES if s in stages)}")
    for name, path in dirs.items():
        print(f"  {name}: {path}")


# =============================================================================
# Stage group 2: batched forward passes for the figures
# =============================================================================

def compute_norms_for_files(model, files, period=8, chunk_size=128,
                            n_cue_classes=None, cue_seed=None, return_devpos=False):
    """Run the model over many sequence files and return per-module norms.

    Sequences are stacked into batches of at most ``chunk_size`` so the hidden
    states of one forward pass stay bounded in memory; the (small) norm arrays
    are concatenated along the batch axis afterwards. All sequences are assumed
    to share the same length (they are here: period * n_trials timesteps).

    ``n_cue_classes`` / ``cue_seed`` are forwarded to ``load_trial_sequence`` so the
    two experimental cues can be re-encoded into the model's cue vocabulary. With a
    fixed ``cue_seed`` every sequence uses the same class pair, keeping the batched
    cue width uniform.

    With ``return_devpos`` the 0-based within-trial deviant position of every trial
    is collected too, so plot_by_position and plot_deviant can share one pass.

    Returns
    -------
    module_norms_dict : dict
        Module name → ndarray of shape (seq_len, n_files).
    dev_pos : np.ndarray or None
        Shape (n_files, n_trials) — 0-based within-trial deviant position per trial.
        None unless ``return_devpos``.
    """
    norm_chunks = {}
    devpos_list = []
    for start in range(0, len(files), chunk_size):
        chunk = files[start:start + chunk_size]

        obs_list, cue_list = [], []
        for f in chunk:
            if return_devpos:
                obs, cue, ctx, dpos, rule, lim_std, d, tau_std, trial_n = \
                    load_trial_sequence(f, return_hierarch=True,
                                        n_cue_classes=n_cue_classes, cue_seed=cue_seed)
                # RAW physical within-trial position {2..6}: kept raw because it indexes
                # the deviant's actual timestep (t*period + pos) in extract_deviant_activity.
                # This is about where the deviant physically sits, independent of any model,
                # so it must NOT be shifted to a model's dpos convention (that would point
                # one tone past the real deviant). The legend renders pos+1 (1-indexed).
                devpos_list.append(dpos[::period])   # one deviant position per trial
            else:
                obs, cue, lim_std, d, tau_std, trial_n = load_trial_sequence(
                    f, n_cue_classes=n_cue_classes, cue_seed=cue_seed)
            obs_list.append(obs)
            cue_list.append(cue)

        min_len = min(o.shape[0] for o in obs_list)
        obs_stack = np.stack([o[:min_len] for o in obs_list], axis=0)        # (N, T)
        cue_stack = np.stack([c[:min_len, :] for c in cue_list], axis=0)     # (N, T, n_cue)

        y = torch.tensor(obs_stack, dtype=torch.float32).unsqueeze(-1)       # (N, T, 1)
        q = torch.tensor(cue_stack, dtype=torch.float32)                     # (N, T, n_cue)

        _, module_norms, _ = get_module_output_and_activity(model, y, q)
        for name, norms in module_norms.items():
            norm_chunks.setdefault(name, []).append(norms)                   # (seq_len, N_chunk)

        print(f"  processed {min(start + chunk_size, len(files))}/{len(files)} sequences")

    module_norms_dict = {name: np.concatenate(chunks, axis=1)
                         for name, chunks in norm_chunks.items()}
    dev_pos = np.stack(devpos_list, axis=0) if return_devpos else None       # (n_files, n_trials)
    return module_norms_dict, dev_pos


def _save(fig, output_dir, filename):
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def run_plot_trajectories(model, n_cue_classes, all_files, output_dir, model_name, cfg):
    """Individual + averaged activity and derivatives (plot_exp_trial_activity.py).

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
    fig = plot_individual_trajectories(
        module_norms_dict, MODULE_TITLES, timesteps,
        output_dir, model_name, include_derivatives=False, pars=pars,
    )
    _save(fig, output_dir, f"{model_name}_exp_activity_trajectories.png")

    # Figure 2: Averaged activity
    fig = plot_averaged_activity(
        module_norms_dict, MODULE_TITLES, timesteps,
        output_dir, model_name, n_samples=n_select, include_derivatives=False,
    )
    _save(fig, output_dir, f"{model_name}_exp_activity_averaged.png")

    # Figure 3: Individual trajectories (derivatives)
    fig = plot_individual_trajectories(
        module_norms_dict, MODULE_TITLES, timesteps,
        output_dir, model_name, include_derivatives=True, pars=pars,
    )
    _save(fig, output_dir, f"{model_name}_exp_activity_trajectories_derivatives.png")

    # Figure 4: Averaged derivatives
    fig = plot_averaged_activity(
        module_norms_dict, MODULE_TITLES, timesteps,
        output_dir, model_name, n_samples=n_select, include_derivatives=True,
    )
    _save(fig, output_dir, f"{model_name}_exp_activity_averaged_derivatives.png")


def run_plot_by_position(module_norms_dict, n_select, output_dir, model_name, cfg):
    """Activity / derivatives averaged over sequences, split by within-trial position."""
    # Figure 1: Averaged activity by within-trial position
    fig = plot_averaged_activity_by_position(
        module_norms_dict, MODULE_TITLES, output_dir, model_name,
        n_samples=n_select, period=cfg.period, include_derivatives=False,
    )
    _save(fig, output_dir, f"{model_name}_exp_activity_averaged_by_position.png")

    # Figure 2: Averaged derivatives by within-trial position
    fig = plot_averaged_activity_by_position(
        module_norms_dict, MODULE_TITLES, output_dir, model_name,
        n_samples=n_select, period=cfg.period, include_derivatives=True,
    )
    _save(fig, output_dir, f"{model_name}_exp_activity_averaged_by_position_derivatives.png")


def run_plot_deviant(module_norms_dict, dev_pos, n_select, output_dir, model_name, cfg):
    """Activity / derivatives at the deviant timestep, grouped by deviant position."""
    # Figure 1: Activity at the deviant position
    fig = plot_deviant_activity_by_position(
        module_norms_dict, dev_pos, MODULE_TITLES, output_dir, model_name,
        n_samples=n_select, period=cfg.period, include_derivatives=False,
    )
    _save(fig, output_dir, f"{model_name}_exp_deviant_activity_by_position.png")

    # Figure 2: Derivatives at the deviant position
    fig = plot_deviant_activity_by_position(
        module_norms_dict, dev_pos, MODULE_TITLES, output_dir, model_name,
        n_samples=n_select, period=cfg.period, include_derivatives=True,
    )
    _save(fig, output_dir, f"{model_name}_exp_deviant_activity_by_position_derivatives.png")


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

        module_norms_dict, dev_pos = compute_norms_for_files(
            model, selected_files, period=cfg.period, chunk_size=cfg.chunk_size,
            n_cue_classes=n_cue_classes, cue_seed=cfg.cue_seed, return_devpos=want_dev)

        if want_by_pos:
            run_plot_by_position(module_norms_dict, n_select, output_dir, model_name, cfg)
        if want_dev:
            run_plot_deviant(module_norms_dict, dev_pos, n_select, output_dir, model_name, cfg)

    print(f"\nFigures saved to {output_dir}")


# =============================================================================
# Entry point
# =============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stages', nargs='+', choices=ALL_STAGES, default=list(ALL_STAGES),
                   metavar='STAGE',
                   help=f"stages to run (default: all). Choices: {', '.join(ALL_STAGES)}")
    p.add_argument('--model-names', nargs='+', default=list(DEFAULT_MODEL_NAMES),
                   metavar='NAME',
                   help='trained model directory names, run in order (default: the '
                        f'{len(DEFAULT_MODEL_NAMES)} models listed in DEFAULT_MODEL_NAMES)')
    p.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR,
                   help='directory containing the trained models (default: %(default)s)')
    p.add_argument('--trials-path', type=Path, default=DEFAULT_TRIALS_PATH,
                   help='directory of experimental sequence files (default: %(default)s)')
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help='base output directory; each model writes to '
                        '<output-root>/<model-name>/ (default: %(default)s)')
    p.add_argument('--n-sequences', type=int, default=None,
                   help='sequences to use for plot_by_position/plot_deviant '
                        '(default: all files)')
    p.add_argument('--n-trajectories', type=int, default=DEFAULT_N_TRAJECTORIES,
                   help='sequences to draw in plot_trajectories (default: %(default)s)')
    p.add_argument('--period', type=int, default=DEFAULT_PERIOD,
                   help='timesteps per trial (default: %(default)s)')
    p.add_argument('--cue-seed', type=int, default=DEFAULT_CUE_SEED,
                   help='seed for cue re-encoding; keep it fixed across stages '
                        '(default: %(default)s)')
    p.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                   help='sequences per batched forward pass (default: %(default)s)')
    p.add_argument('--seed', type=int, default=0,
                   help='seed for random file sampling (default: %(default)s)')
    p.add_argument('--no-next-stimulus', dest='include_next_stimulus',
                   action='store_false',
                   help='restrict *_probabilities_deviant.csv to the deviant rows only, '
                        'instead of the deviant plus the immediate next stimulus')
    p.set_defaults(include_next_stimulus=True)
    return p.parse_args(argv)


def run_one_model(model_name, trial_files, stages, cfg):
    """Run every requested stage for one model, into <output-root>/<model_name>/."""
    model_path = cfg.model_dir / model_name
    output_root = cfg.output_root / model_name
    output_root.mkdir(parents=True, exist_ok=True)

    # Load the model once for every stage.
    info = eval.ModelInfo.from_path(model_path)
    model = eval.load_model(info)
    model.eval()
    print(f"Loaded model: {model_path.name}")
    print(f"Output root: {output_root}")

    if any(s in stages for s in CSV_STAGES):
        run_csv_stages(model, info, trial_files, stages, output_root, cfg)

    if any(s in stages for s in PLOT_STAGES):
        run_plot_stages(model, info, trial_files, stages, output_root, model_name, cfg)


def main(argv=None):
    cfg = parse_args(argv)
    stages = [s for s in ALL_STAGES if s in set(cfg.stages)]  # canonical order

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


if __name__ == '__main__':
    main()
