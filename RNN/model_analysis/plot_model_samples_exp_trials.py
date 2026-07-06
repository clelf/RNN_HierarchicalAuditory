"""
Plot model sample figures (obs + predictions, with ctx/dpos/rule tracks) on the
*experimental* trial sequences — the same sequence files and model used by
model_prob_exp_trials.py.

Unlike plot_model_samples.py (which generates fresh test data from the generative
model, or loads benchmark data), this script loads the recorded experimental
sequences, runs the model on them, and produces the same plot_samples figures.

One figure is written per selected sequence (the figure file name carries the
sample index). For a HierarchicalGM model, plot_samples internally produces both
a '_last8blocks' and a '_full' view per sample.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

# Set up sys.path before local imports (mirrors plot_model_samples.py):
#   model_analysis/ — sibling modules (evaluate_models, model_activations)
#   RNN/train/      — pipeline_core_v2
#   Workspace/      — Kalman package
_here = os.path.abspath(os.path.dirname(__file__))
_train_dir = os.path.abspath(os.path.join(_here, '..', 'train'))
_workspace = os.path.abspath(os.path.join(_here, '..', '..', '..'))
for _p in [_workspace, _train_dir, _here]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluate_models as eval
from pipeline_core_v2 import plot_samples, get_model_predictions
from model_activations import load_trial_sequence, to_model_tensors, load_trial_params, dpos_conventions

# =============================================================================
# Configuration (model + sequence files match model_prob_exp_trials.py)
# =============================================================================

# Model (one model only)
model_name = "population_network_all_bn8_lr0"
model_dir = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/training_results/N_ctx_2/HierarchicalGM")
model_path = model_dir / model_name

# Generated sequence files directory
trials_path = Path("/home/clevyfidel/Documents/Workspace/Jasmin/trialsequences2clem")

# Output directory for the sample figures
output_dir = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output") / model_name / "samples"
output_dir.mkdir(parents=True, exist_ok=True)

# dpos alignment is read from the model's config below, right after it is loaded.

# How many sequences to load and how many to actually plot.
N_load = 16        # number of sequence files to load
N_plots = 8        # number of those to plot (plot_samples selects this many)
SEED = 0           # reproducible file + sample selection

# =============================================================================
# Load model
# =============================================================================
info = eval.ModelInfo.from_path(model_path)
model = eval.load_model(info)
model.eval()
print(f"Loaded model: {model_path.name}")

data_config = info.data_config_dict  # provides gm_name, N_ctx, N_tones

# dpos alignment from the model's config: dpos_min is the class-index offset (model
# dpos_pred = argmax_class + dpos_min) and EXPERIMENTAL_DPOS_SHIFT maps the on-disk
# experimental dpos into the model's convention, so dpos_true and dpos_pred share
# coordinates. Self-adjusting: lr0 -> (3, 1); retrained on [[2,3,4],[4,5,6]] -> (2, 0).
dpos_min, EXPERIMENTAL_DPOS_SHIFT = dpos_conventions(info)

# =============================================================================
# Select and load experimental sequences
# =============================================================================
trial_files = sorted(trials_path.glob("*.csv"))
if not trial_files:
    trial_files = sorted(trials_path.glob("*.txt"))
print(f"Found {len(trial_files)} trial sequence files in {trials_path}")

rng = np.random.default_rng(seed=SEED)
n_pick = min(N_load, len(trial_files))
pick_idx = rng.choice(len(trial_files), size=n_pick, replace=False)
selected_files = [trial_files[i] for i in sorted(pick_idx)]
print(f"Loading {len(selected_files)} sequences for plotting")

obs_list, cue_list = [], []
ctx_list, dpos_list, rule_list = [], [], []
tau_list, lim_list, si_q_list, si_stat_list, si_r_list = [], [], [], [], []

for f in selected_files:
    obs, cue, ctx, dpos, rule, lim_std, d, tau_std, trial_n = load_trial_sequence(
        f, return_hierarch=True
    )
    # Relabel experimental dpos {2..6} -> model convention {3..7} (see above).
    dpos = dpos + EXPERIMENTAL_DPOS_SHIFT
    obs_list.append(obs)
    cue_list.append(cue)            # (T, 2) one-hot
    ctx_list.append(ctx)
    dpos_list.append(dpos)
    rule_list.append(rule)

    # Per-sequence scalar parameters for the plot title (n_ctx == 2 expects
    # std/dvt pairs for tau, lim and si_q).
    import pandas as pd
    row = pd.read_csv(f, nrows=1).iloc[0]
    p = load_trial_params(f)  # tau, lim=[std, dev], si_stat, si_r
    tau_list.append([float(row['tau_std']), float(row['tau_dev'])])
    lim_list.append([float(row['lim_std']), float(row['lim_dev'])])
    si_q_list.append([float(row['sigma_q_std']), float(row['sigma_q_dev'])])
    si_stat_list.append(p['si_stat'])
    si_r_list.append(p['si_r'])

# Stack into batched arrays (all sequences share the same length).
obs_np = np.stack(obs_list).astype(np.float32)       # (N, T)
cue_np = np.stack(cue_list).astype(np.float32)       # (N, T, 2) one-hot
ctx_np = np.stack(ctx_list)                          # (N, T)
dpos_np = np.stack(dpos_list)                        # (N, T) raw deviant positions
rule_np = np.stack(rule_list)                        # (N, T)

y = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(-1)  # (N, T, 1)
q = torch.tensor(cue_np, dtype=torch.float32)                # (N, T, 2)

params = {
    'tau': np.asarray(tau_list, dtype=float),       # (N, 2)
    'lim': np.asarray(lim_list, dtype=float),       # (N, 2)
    'si_q': np.asarray(si_q_list, dtype=float),     # (N, 2)
    'si_stat': np.asarray(si_stat_list, dtype=float),  # (N,)
    'si_r': np.asarray(si_r_list, dtype=float),     # (N,)
}

# =============================================================================
# Forward pass + predictions (model runs on y[:, :-1], q[:, :-1])
# =============================================================================
with torch.no_grad():
    model_output = model(y[:, :-1, :], q[:, :-1, :])
    predictions = get_model_predictions(model, model_output, dpos_min=dpos_min)

sample_metrics = {
    'y': obs_np,                              # (N, T)
    'mu_estim': predictions['mu_estim'],     # (N, T-1)
    'sigma_estim': predictions['sigma_estim'],
    'kalman_mu': None,
    'kalman_sigma': None,
    'contexts': ctx_np,                      # (N, T)
    'ctx_prob': predictions['ctx_prob'],
    'ctx_pred': predictions['ctx_pred'],
    'dpos_true': dpos_np,                    # (N, T), model-convention positions {3..7} (matches dpos_pred coords)
    'dpos_prob': predictions['dpos_prob'],
    'dpos_pred': predictions['dpos_pred'],
    'rule_true': rule_np,                    # (N, T)
    'rule_prob': predictions['rule_prob'],
    'rule_pred': predictions['rule_pred'],
    'cues': cue_np,                          # (N, T, 2) one-hot
}

# =============================================================================
# Plot
# =============================================================================
save_path = output_dir / f"{model_name}_samples"
np.random.seed(SEED)  # reproducible sample selection inside plot_samples
plot_samples(
    sample_metrics=sample_metrics,
    save_path=str(save_path),
    title=f"Samples (exp. trials) – {model_name}",
    N_plots=N_plots,
    seq_start=None,
    seq_end=None,
    params=params,
    data_config=data_config,
    min_obs_for_em=None,
    shared_ylim=True,
)

print(f"Saved sample figures to {output_dir}")