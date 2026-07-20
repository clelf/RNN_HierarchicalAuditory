"""
Distribution of the per-sequence mean dpos probability at the true deviant.

For every ``*_probabilities_deviant.csv`` file produced by
``model_prob_exp_trials.py`` this script takes the ``lik_dpos`` column -- the
probability the model assigns to the *true* deviant-position class at each row
(see model_prob_exp_trials.py, ``out_df['lik_dpos'] = class_likelihood(...)``) --
and averages it over all rows of the file, giving one value per sequence.

Each sequence's generative parameters (the columns that are constant across its
``..._trials.csv`` file, e.g. lim_std/mu, tau_std, d, the sigmas) are joined on as
additional columns so the distribution can be split by parameter later.

Outputs (in a ``probabilities_deviant_avg/`` folder next to
``probabilities_deviant/``):
- ``mean_dpos_prob_per_sequence.csv``: one row per sequence with the mean
  (and the row count) plus the sequence parameters, so the values can be
  retrieved later.
- ``mean_dpos_prob_per_sequence_hist.png``: histogram of those per-sequence means.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # =============================================================================
    # Configuration
    # =============================================================================
    # model_name = "population_network_all_bn8_lr0"
    model_name = "population_network_all_bn8_lr0.001_dposweight"

    base_output_path = Path(
        "/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output"
    ) / model_name

    deviant_path = base_output_path / 'probabilities_deviant'
    avg_path = base_output_path / 'probabilities_deviant_avg'
    avg_path.mkdir(parents=True, exist_ok=True)

    # Original trial sequence files, source of the per-sequence parameters.
    trials_path = Path("/home/clevyfidel/Documents/Workspace/Jasmin/trialsequences2clem")

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
    deviant_files = sorted(deviant_path.glob("*_probabilities_deviant.csv"))
    if not deviant_files:
        raise FileNotFoundError(f"No *_probabilities_deviant.csv files in {deviant_path}")
    print(f"Found {len(deviant_files)} deviant probability files in {deviant_path}")

    records = []
    for f in deviant_files:
        df = pd.read_csv(f)
        values = df[DPOS_PROB_COL].to_numpy(dtype=float)

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
            tdf = pd.read_csv(trial_file, usecols=lambda c: c in PARAM_COLS)
            for col in PARAM_COLS:
                if col not in tdf.columns:
                    record[col] = np.nan
                    continue
                if tdf[col].nunique(dropna=False) > 1:
                    print(f"  Warning: '{col}' is not constant in {trial_file.name}; "
                          f"using first row.")
                record[col] = tdf[col].iloc[0]
        else:
            print(f"  Warning: no trial file for {sequence}; parameters set to NaN.")
            for col in PARAM_COLS:
                record[col] = np.nan

        records.append(record)

    avg_df = pd.DataFrame.from_records(records).sort_values('sequence').reset_index(drop=True)

    avg_csv = avg_path / 'mean_dpos_prob_per_sequence.csv'
    avg_df.to_csv(avg_csv, index=False)
    print(f"Saved per-sequence means: {avg_csv}")

    # =============================================================================
    # Histogram of the per-sequence means
    # =============================================================================
    means = avg_df['mean_dpos_prob'].to_numpy(dtype=float)
    valid = means[~np.isnan(means)]
    n_nan = int(np.sum(np.isnan(means)))
    if n_nan:
        print(f"Note: {n_nan} sequence(s) had no valid rows and are excluded from the histogram.")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(valid, bins=30, range=(0, 1))
    ax.axvline(valid.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f"mean = {valid.mean():.3f}")
    ax.set_xlabel('Per-sequence mean P(true deviant position)')
    ax.set_ylabel('Number of sequences')
    ax.set_title(f'dpos probability at true deviant\n{model_name}  (n = {len(valid)} sequences)')
    ax.legend()
    fig.tight_layout()

    hist_png = avg_path / 'mean_dpos_prob_per_sequence_hist.png'
    fig.savefig(hist_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved histogram: {hist_png}")

    print(f"\nSummary over {len(valid)} sequences: "
          f"mean={valid.mean():.4f}, median={np.median(valid):.4f}, "
          f"min={valid.min():.4f}, max={valid.max():.4f}")
