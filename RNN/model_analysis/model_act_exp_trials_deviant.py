import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import evaluate_models as eval

from model_activations import get_module_output_and_activity, load_trial_sequence, to_model_tensors, dpos_conventions


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


if __name__ == '__main__':

    # Number of timesteps per trial. The deviant of trial t sits at timestep
    # t * PERIOD + dpos, where dpos is the within-trial deviant position.
    PERIOD = 8

    # Specify model path (one model only), load model
    model_name = "population_network_all_bn8_lr0"
    model_dir = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/training_results/N_ctx_2/HierarchicalGM")
    model_path = model_dir / model_name

    # Specify generated sequences files directory
    trials_path = Path("/home/clevyfidel/Documents/Workspace/Jasmin/trialsequences2clem")

    # Specify output directory path
    output_path = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output") / model_name / 'activations_deviant'
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    info = eval.ModelInfo.from_path(model_path)
    model = eval.load_model(info)
    model.eval()
    print(f"Loaded model: {model_path.name}")

    # dpos convention from the model's config. Two distinct uses of dpos are kept
    # separate below:
    #   * the PHYSICAL within-trial timestep of the deviant uses the RAW experimental
    #     position (that is literally where the deviant tone sits in the sequence);
    #   * the stored deviant_pos LABEL is shifted into the model's convention so it
    #     matches model_prob_exp_trials.py's dpos column and the model's class mapping.
    # Self-adjusting via the config: lr0 -> shift +1; retrained on [[2,3,4],[4,5,6]] ->
    # shift 0. The on-disk sequence files are never edited.
    _dpos_min, EXPERIMENTAL_DPOS_SHIFT = dpos_conventions(info)

    trial_files = sorted(trials_path.glob("*.csv"))
    if not trial_files:
        trial_files = sorted(trials_path.glob("*.txt"))
    print(f"Found {len(trial_files)} trial sequence files in {trials_path}")

    for trial_file in trial_files:
        obs, cue, ctx, dpos, rule, lim_std, d, tau_std, trial_n = \
            load_trial_sequence(trial_file, return_hierarch=True)
        y, q = to_model_tensors(obs, cue)

        # hidden_activity:    dict module → (T-1, 1)
        # hidden_derivatives: dict module → (T-2, 1)
        prob_output, hidden_activity, hidden_derivatives = get_module_output_and_activity(model, y, q)

        # Squeeze batch dim (batch=1) from norms and derivatives
        norms = {name: arr[:, 0] for name, arr in hidden_activity.items()}      # each: (T-1,)
        derivs = {name: arr[:, 0] for name, arr in hidden_derivatives.items()}  # each: (T-2,)
        n_out = next(iter(norms.values())).shape[0]    # T-1
        n_deriv = next(iter(derivs.values())).shape[0]  # T-2

        # One value per trial: dpos is constant within a trial, so take the first
        # timestep of each trial. The deviant timestep is trial_start + dpos.
        T = obs.shape[0]
        n_trials = T // PERIOD
        trial_ids = trial_n.to_numpy()[::PERIOD]               # (n_trials,)
        # RAW physical within-trial position {2..6}: used to index the deviant's timestep.
        deviant_pos_phys = dpos[::PERIOD]                       # (n_trials,)
        dev_idx = np.arange(n_trials) * PERIOD + deviant_pos_phys  # (n_trials,) global timestep of deviant
        # Stored LABEL in the model convention {3..7} (raw + shift) for cross-script consistency.
        deviant_pos = deviant_pos_phys + EXPERIMENTAL_DPOS_SHIFT

        # One row per trial: module activity / derivative sampled at the deviant timestep.
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

        out_file = output_path / (trial_file.stem + '_deviant_trial.csv')
        out_df.to_csv(out_file, index=False)
        print(f"  Saved: {out_file.name}")

    print(f"\n{len(trial_files)} file(s) processed in {output_path}")
