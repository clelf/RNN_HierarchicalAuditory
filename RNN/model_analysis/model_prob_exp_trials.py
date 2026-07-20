"""
This script obtains the model's predicted probabilities for each module (obs, ctx, dpos, rule) on a set of trial sequences.
It also computes the likelihoods of the ground truth observations and associated ctx, dpos, rule labels under the predicted distriutions.
It returns two files for each sequence file:
- *_probabilities.csv: contains the predicted probabilities and likelihoods for all timesteps in the  sequence.
- *_probabilities_deviant.csv: contains the same information but restricted to rows where the context (trial_type) is 1 (deviant trials), or optionally the deviant trials plus the immediate next stimulus.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import evaluate_models as eval

from model_activations import (
    get_module_probabilities,
    load_trial_sequence,
    to_model_tensors,
    gaussian_likelihood,
    class_likelihood,
    dpos_conventions,
)


if __name__ == '__main__':
    # TODO: here, define whether the deviant-only file should include the immediate next stimulus or not. Currently, it includes the next stimulus.
    include_next_stimulus = True

    # Specify model path (one model only), load model
    # model_name = "population_network_all_bn8_lr0"
    model_name = "population_network_all_bn8_lr0.001_dposweight"
    model_dir = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/training_results/N_ctx_2/HierarchicalGM")
    model_path = model_dir / model_name

    # Specify generated sequences files directory
    trials_path = Path("/home/clevyfidel/Documents/Workspace/Jasmin/trialsequences2clem")

    # Specify output directory path
    base_output_path = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output") / model_name
    output_path = base_output_path / 'probabilities'
    output_path.mkdir(parents=True, exist_ok=True)

    # Deviant-only output: same information but restricted to rows where the
    # context label (trial_type) is 1.
    output_path_deviant = base_output_path / 'probabilities_deviant'
    output_path_deviant.mkdir(parents=True, exist_ok=True)

    # Load model
    info = eval.ModelInfo.from_path(model_path)
    model = eval.load_model(info)
    model.eval()
    print(f"Loaded model: {model_path.name}")

    trial_files = sorted(trials_path.glob("*.csv"))
    if not trial_files:
        trial_files = sorted(trials_path.glob("*.txt"))
    print(f"Found {len(trial_files)} trial sequence files in {trials_path}")

    # dpos alignment, read from the model's saved config (rules_dpos_set): dpos_min is
    # the model's class-index offset (class c <-> position c+dpos_min), and
    # EXPERIMENTAL_DPOS_SHIFT maps the on-disk experimental dpos into the model's
    # convention. Self-adjusting: lr0 -> (3, 1); a model retrained on [[2,3,4],[4,5,6]]
    # -> (2, 0). The sequence files are never edited.
    dpos_min, EXPERIMENTAL_DPOS_SHIFT = dpos_conventions(info)
    print(f"dpos class-index offset (model convention): {dpos_min}; "
          f"experimental dpos shift: +{EXPERIMENTAL_DPOS_SHIFT}")

    # If using a model that was trained with a larger cue set than the two cues present in
    # the experimental sequences. Read the trained cue count so the sequences can be
    # re-encoded into the cue dimensionality the model expects.
    n_cue_classes = len(info.data_config_dict['cues_set'])

    for trial_file in trial_files:
        # ctx/dpos/rule are the ground-truth labels from the original sequence.
        # dpos is the raw deviant position on disk (e.g. 2..6), not yet a 0-based class index.
        obs, cue, ctx, dpos, rule, lim_std, d, tau_std, trial_n = load_trial_sequence(
            trial_file, return_hierarch=True, n_cue_classes=n_cue_classes, cue_seed=11
        )

        # Shift the experimental deviant-position labels into the model's training
        # convention ({2..6} -> {3..7}); the on-disk files are left untouched. From here
        # on, dpos is in the model's convention and dpos-dpos_min is a valid 0-based class.
        dpos = dpos + EXPERIMENTAL_DPOS_SHIFT

        y, q = to_model_tensors(obs, cue)

        # probs: dict module → (seq_len, batch, dim)
        #   'obs':  dim=2, columns are (mean, variance) of the predicted Gaussian
        #   others: dim=n_classes, softmax class probabilities
        probs = get_module_probabilities(model, y, q)

        # Number of timesteps in the output: T-1 (the model is run on y[:, :-1]).
        # The model output at index k predicts timestep k+1, so the outputs align
        # with the ground truth shifted by one: y[1:], q[1:], labels[1:].
        n_out = next(iter(probs.values())).shape[0]

        # Next-step prediction alignment: ground truth that each output row is predicting.
        obs_gt = obs[1:n_out + 1]          # (n_out,)
        cue_gt = cue[1:n_out + 1, :]       # (n_out, 2)
        ctx_gt = ctx[1:n_out + 1]          # (n_out,)
        dpos_gt = dpos[1:n_out + 1]        # (n_out,)
        rule_gt = rule[1:n_out + 1]        # (n_out,)

        # Squeeze batch dim (batch=1) from each module's probabilities
        probs = {name: arr[:, 0, :] for name, arr in probs.items()}  # each: (n_out, dim)

        # Sequence info columns (same fields as model_act_exp_trials.py), now
        # carrying the next-step (prediction-target) values so every column in a
        # row refers to the same timestep as that row's likelihoods.
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
            n_classes = module_probs.shape[1]
            for c in range(n_classes):
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
        
        out_file = output_path / (trial_file.stem + '_probabilities.csv')
        out_df.to_csv(out_file, index=False)
        print(f"  Saved: {out_file.name}")

        if not include_next_stimulus:
            # Save the deviant-only subset (rows where the context label is 1).
            deviant_df = out_df[out_df['ctx'] == 1]
        else:
            # Or if deviants + immediate next stimulus:
            deviant_mask = out_df['ctx'] == 1
            extended_mask = deviant_mask | deviant_mask.shift(1, fill_value=False)
            deviant_df = out_df[extended_mask]

        deviant_file = output_path_deviant / (trial_file.stem + '_probabilities_deviant.csv')
        deviant_df.to_csv(deviant_file, index=False)
        print(f"  Saved: {deviant_file.name}")

    print(f"\n{len(trial_files)} file(s) processed in {output_path}")
    print(f"{len(trial_files)} deviant-only file(s) processed in {output_path_deviant}")
