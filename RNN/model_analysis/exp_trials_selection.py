"""
Read all *_probabilities_deviant.csv files produced by model_prob_exp_trials.py,
compute per-file averages of the likelihood columns, and save the summary to a
sub-folder called "average" inside the deviant output directory.

One row per input file is written, carrying:
  - the source filename
  - per-file scalar parameters (lim_std, d, tau_std)
  - mean of each likelihood column (lik_obs, lik_ctx, lik_dpos, lik_rule)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import scipy

# ── Configuration ─────────────────────────────────────────────────────────────
model_name = "population_network_all_bn8_lr0"
base_output_path = Path(
    "/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output"
) / model_name

deviant_path = base_output_path / "probabilities_deviant"
average_path = deviant_path / "average"
average_path.mkdir(parents=True, exist_ok=True)

LIKELIHOOD_COLS = ["lik_obs", "lik_ctx", "lik_dpos", "lik_rule", "cdf_lik_obs", "log_lik_obs"]
PARAM_COLS      = ["lim_std", "d", "tau_std"]

# ── Main loop ─────────────────────────────────────────────────────────────────
deviant_files = sorted(deviant_path.glob("*_probabilities_deviant.csv"))
if not deviant_files:
    raise FileNotFoundError(f"No *_probabilities_deviant.csv files found in {deviant_path}")

print(f"Found {len(deviant_files)} deviant file(s) in {deviant_path}")

rows = []
for f in deviant_files:
    df = pd.read_csv(f)
    df['cdf_lik_obs'] = scipy.stats.norm.cdf((df['observation'] - df['obs_mean']) / np.sqrt(df['obs_var']))
    df['log_lik_obs'] = np.log(df['lik_obs'] + 1e-8)  # Add small constant to avoid log(0)

    row = {"file": f.name}

    for col in PARAM_COLS:
        if col in df.columns:
            row[col] = df[col].iloc[0]

    for col in LIKELIHOOD_COLS:
        if col in df.columns:
            row[f"mean_{col}"] = df[col].mean()
        else:
            row[f"mean_{col}"] = float("nan")

    rows.append(row)

summary_df = pd.DataFrame(rows)

out_file = average_path / f"{model_name}_likelihood_averages.csv"
summary_df.to_csv(out_file, index=False)
print(f"Saved: {out_file}")