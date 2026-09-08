"""
Distribution of the per-sequence mean dpos probability at the true deviant.

For every model listed in ``MODEL_NAMES`` and every ``*_probabilities_deviant.csv``
file produced by ``model_prob_exp_trials.py`` this script takes the ``lik_dpos``
column -- the probability the model assigns to the *true* deviant-position class
at each row (see model_prob_exp_trials.py, ``out_df['lik_dpos'] = class_likelihood(...)``)
-- and averages it over all rows of the file, giving one value per sequence.

Each sequence's generative parameters (the columns that are constant across its
``..._trials.csv`` file, e.g. lim_std/mu, tau_std, d, the sigmas) are joined on as
additional columns so the distribution can be split by parameter later.

Outputs, per model (in a ``probabilities_deviant_avg/`` folder next to that
model's ``probabilities_deviant/``):
- ``mean_dpos_prob_per_sequence.csv``: one row per sequence with the mean
  (and the row count) plus the sequence parameters, so the values can be
  retrieved later.
- ``mean_dpos_prob_per_sequence_hist.png``: histogram of those per-sequence means.

Models whose ``probabilities_deviant/`` folder is missing or empty are reported
and skipped -- run model_prob_exp_trials.py on them first.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Column holding the probability of the true deviant-position class.
DPOS_PROB_COL = 'lik_dpos'

# Sequence-level generative parameters to carry over. These are constant
# across every row of a ..._trials.csv file (unlike per-trial columns such as
# rule, dpos, trial_type). 'lim_std' is the mu shown in the file name.
PARAM_COLS = [
    'lim_std', 'lim_dev', 'tau_std', 'tau_dev', 'd',
    'sigma_q_std', 'sigma_q_dev', 'sigma_r',
    'duration_tones', 'ISI', 'run_n', 'session_n',
]


# =============================================================================
# One mean per sequence file
# =============================================================================

def summarise_dpos_prob(deviant_path, trials_path, dpos_prob_col=DPOS_PROB_COL,
                        param_cols=PARAM_COLS):
    """Per-sequence mean of `dpos_prob_col`, with the sequence parameters joined on.

    Returns None (with a warning) when the model has no deviant files, so one
    unprocessed model does not stop the others.
    """
    deviant_files = sorted(deviant_path.glob("*_probabilities_deviant.csv"))
    if not deviant_files:
        print(f"  Warning: no *_probabilities_deviant.csv files in {deviant_path}; skipping.")
        return None
    print(f"Found {len(deviant_files)} deviant probability files in {deviant_path}")

    records = []
    for f in deviant_files:
        df = pd.read_csv(f)
        values = df[dpos_prob_col].to_numpy(dtype=float)

        # nanmean: undefined/unpredictable rows are stored as NaN and must not be
        # counted as zeros. A file with no valid rows yields NaN (and a warning),
        # which we keep so it is visible rather than silently dropped here.
        if np.all(np.isnan(values)):
            mean_prob = np.nan
        else:
            mean_prob = np.nanmean(values)

        # Strip the trailing suffix to recover the sequence identifier. The
        # matching trial file is <sequence>.csv in trials_path.
        sequence = f.name.replace('_probabilities_deviant.csv', '')
        record = {
            'sequence': sequence,
            'mean_dpos_prob': mean_prob,
            'n_rows': len(values),
            'n_valid_rows': int(np.sum(~np.isnan(values))),
        }

        # Join the sequence's generative parameters from its trial file. Each
        # PARAM_COL is constant within the file, so take the first row's value;
        # warn (rather than silently pick one) if that assumption ever breaks.
        trial_file = trials_path / (sequence + '.csv')
        if trial_file.exists():
            tdf = pd.read_csv(trial_file, usecols=lambda c: c in param_cols)
            for col in param_cols:
                if col not in tdf.columns:
                    record[col] = np.nan
                    continue
                if tdf[col].nunique(dropna=False) > 1:
                    print(f"  Warning: '{col}' is not constant in {trial_file.name}; "
                          f"using first row.")
                record[col] = tdf[col].iloc[0]
        else:
            print(f"  Warning: no trial file for {sequence}; parameters set to NaN.")
            for col in param_cols:
                record[col] = np.nan

        records.append(record)

    return pd.DataFrame.from_records(records).sort_values('sequence').reset_index(drop=True)


# =============================================================================
# Histogram of the per-sequence means
# =============================================================================

def plot_dpos_prob_hist(avg_df, out_png, model_name, bins=30):
    """Histogram of the per-sequence means, with the overall mean marked."""
    means = avg_df['mean_dpos_prob'].to_numpy(dtype=float)
    valid = means[~np.isnan(means)]
    n_nan = int(np.sum(np.isnan(means)))
    if n_nan:
        print(f"Note: {n_nan} sequence(s) had no valid rows and are excluded from the histogram.")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(valid, bins=bins, range=(0, 1))
    ax.axvline(valid.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f"mean = {valid.mean():.3f}")
    ax.set_xlabel('Per-sequence mean P(true deviant position)')
    ax.set_ylabel('Number of sequences')
    ax.set_title(f'dpos probability at true deviant\n{model_name}  (n = {len(valid)} sequences)')
    ax.legend()
    fig.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved histogram: {out_png}")

    print(f"Summary over {len(valid)} sequences: "
          f"mean={valid.mean():.4f}, median={np.median(valid):.4f}, "
          f"min={valid.min():.4f}, max={valid.max():.4f}")
    return out_png


# =============================================================================
# Per-model driver
# =============================================================================

def process_model(model_name, base_output_root, trials_path):
    """CSV + histogram for one model. Returns the per-sequence frame, or None."""
    print(f"\n=== {model_name}")
    base_output_path = Path(base_output_root) / model_name
    deviant_path = base_output_path / 'probabilities_deviant'

    avg_df = summarise_dpos_prob(deviant_path, trials_path)
    if avg_df is None:
        return None

    avg_path = base_output_path / 'probabilities_deviant_avg'
    avg_path.mkdir(parents=True, exist_ok=True)

    avg_csv = avg_path / 'mean_dpos_prob_per_sequence.csv'
    avg_df.to_csv(avg_csv, index=False)
    print(f"Saved per-sequence means: {avg_csv}")

    plot_dpos_prob_hist(avg_df, avg_path / 'mean_dpos_prob_per_sequence_hist.png',
                        model_name)
    return avg_df


def process_models(model_names, base_output_root, trials_path):
    """Run process_model for each model; returns {model name: frame} for those with data."""
    results = {}
    for model_name in model_names:
        avg_df = process_model(model_name, base_output_root, trials_path)
        if avg_df is not None:
            results[model_name] = avg_df

    skipped = [m for m in model_names if m not in results]
    if skipped:
        print(f"\nSkipped {len(skipped)} model(s) with no deviant probability files: "
              + ", ".join(skipped))
    if not results:
        raise FileNotFoundError(
            f"No *_probabilities_deviant.csv files found for any of {model_names} "
            f"under {base_output_root}")
    return results


if __name__ == '__main__':

    # =============================================================================
    # Configuration
    # =============================================================================
    # Models to process. Each must own an
    # exp_seq_act_output/<model>/probabilities_deviant folder filled by
    # model_prob_exp_trials.py; the others are reported and skipped.
    MODEL_NAMES = [
        "population_network_all_bn8_trainh0_fixedsir0.02_epochs300_lr0.002",
        "population_network_all_bn8_trainh0_fixedsir0.05_epochs300_lr0.002",
        "population_network_all_bn8_trainh0_fixedsir0.1_epochs300_lr0.002",
    ]

    base_output_root = Path(
        "/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output"
    )

    # Original trial sequence files, source of the per-sequence parameters.
    trials_path = Path("/home/clevyfidel/Documents/Workspace/Jasmin/trialsequences2clem")

    process_models(MODEL_NAMES, base_output_root, trials_path)
