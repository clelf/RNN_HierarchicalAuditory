import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import evaluate_models as eval

from model_activations import get_module_hidden_activity, load_trial_sequence, to_model_tensors






if __name__ == '__main__':


    # Specify model path (one model only), load model
    model_name = "population_network_all_bn8_lr0"
    model_dir = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/training_results/N_ctx_2/HierarchicalGM")
    model_path = model_dir / model_name

    # Specify generated sequences files directory
    trials_path = Path("/home/clevyfidel/Documents/Workspace/Jasmin/trialsequences2clem")

    # Specify output directory path
    output_path = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output") / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    info = eval.ModelInfo.from_path(model_path)
    model = eval.load_model(info)
    model.eval()
    print(f"Loaded model: {model_path.name}")

    trial_files = sorted(trials_path.glob("*.csv"))
    if not trial_files:
        trial_files = sorted(trials_path.glob("*.txt"))
    print(f"Found {len(trial_files)} trial sequence files in {trials_path}")

    for trial_file in trial_files:
        obs, cue = load_trial_sequence(trial_file)
        y, q = to_model_tensors(obs, cue)

        # hidden_activity:    dict module → (T-1, 1)
        # hidden_derivatives: dict module → (T-2, 1)
        hidden_activity, hidden_derivatives = get_module_hidden_activity(model, y, q)

        # Number of timesteps in the output: T-1 (matching norms length)
        n_out = next(iter(hidden_activity.values())).shape[0]

        # Align observation and cue to the first n_out timesteps
        obs_out = obs[:n_out]
        cue_out = cue[:n_out, :]  # (n_out, 2)

        # Squeeze batch dim (batch=1) from norms and derivatives
        norms = {name: arr[:, 0] for name, arr in hidden_activity.items()}      # each: (T-1,)
        derivs = {name: arr[:, 0] for name, arr in hidden_derivatives.items()}  # each: (T-2,)

        # Pad derivatives with NaN at the last position so all columns have length T-1
        n_deriv = next(iter(derivs.values())).shape[0]
        pad = np.full(n_out - n_deriv, np.nan)
        derivs_padded = {name: np.concatenate([d, pad]) for name, d in derivs.items()}

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

        out_file = output_path / (trial_file.stem + '_activations.csv')
        out_df.to_csv(out_file, index=False)
        print(f"  Saved: {out_file.name}")

    print(f"\n{len(trial_files)} file(s) processed in {output_path}")
