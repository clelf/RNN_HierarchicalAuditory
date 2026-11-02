"""Running a model over the experimental sequences, and aggregating the results.

Two halves:

  1. Extraction -- build the per-sequence tables from a model's forward pass over
     the recorded trial sequences (activity norms per timestep, one row per trial
     at the deviant, per-module predicted distributions and ground-truth
     likelihoods), and the per-timestep activity of pools of sequences presented
     one after the other. These are what run_exp_trials_pipeline.py writes to
     CSV, and the only place a model is run on the experimental sequences.

  2. Aggregation -- read the generated CSVs and summarise them: mean likelihoods per
     sequence (optionally split by deviant position), module-pair correlation
     scores, the per-trial join of activity against likelihood, and the arrays
     the activity figures and per-sample figures are drawn from.

The ctx/dpos detection case taxonomy is large and self-contained, so it lives in
alignment_cases.py rather than here.
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr

from analysis_config import (
    DEVIANT_PARAM_COLS,
    EXP_SEQ_OUTPUT_ROOT as BASE_OUTPUT_DIR,
    LIKELIHOOD_COLS,
    MODULE_NAMES,
    MODULE_NAMES as MODULES,
    SEQUENCE_PARAM_COLS,
)
from analysis_core import (
    class_likelihood,
    class_probability_columns,
    compute_derivatives,
    read_trial_at,
    gaussian_likelihood,
)


# =============================================================================
# Extraction: per-sequence tables
# =============================================================================
def add_module_activity_columns(out_df, norms, derivs, n_out, prefix=''):
    """Append one block of per-module activity norms and derivatives to `out_df`.

    Called once per pass of a forward cycle: with prefix='' for the posterior
    hidden states (the state each module ends the timestep with) and with
    prefix='prior_' for the states their first call left them in.

    Parameters
    ----------
    out_df : pandas.DataFrame
        Frame to add the columns to, modified in place.
    norms : dict
        Module name → ndarray of shape (n_out,); row t holds the norm after
        seeing the observation of timestep t.
    derivs : dict
        Module name → ndarray of shape (n_out,), as returned by compute_derivatives:
        row t holds norms[t] - norms[t-1], the change in activity on the arrival
        of the observation of timestep t, and row 0 is NaN.
    n_out : int
        Number of rows of the frame, i.e. of model output timesteps.
    prefix : str
        Prepended to every column name written here.
    """
    for name in MODULES:
        out_df[f'{prefix}{name}_norm'] = norms[name]
    for name in MODULES:
        out_df[f'{prefix}{name}_deriv'] = derivs[name]


def get_two_cue_encoding(cue):
    """The two cues of a sequence as a (T, 2) one-hot, from the model's (T, n_cue_classes) cue input.

    `cue` may be a (T, n_cue_classes) one-hot where only the two cues used in this
    sequence are ever active (analysis_core.load_trial_sequence). The two active
    columns are recovered in ascending class order, matching CUE_LABELS order, so
    the CSVs keep their original two-cue encoding. `cue` must be the whole
    sequence, so that both cues are present in it.
    """
    used_cols = np.flatnonzero(cue.any(axis=0))
    return cue[:, used_cols]


def build_activity_columns_frame(obs, cue, norms, derivs, prior_norms, prior_derivs):
    """Observation, cue and module activity columns, one row per model output timestep.

    The part of the activations CSVs shared by single sequences
    (build_activations_frame) and pools of sequences (build_pool_activations_frame).
    Rows are the model output timesteps t = 0 ... T-2 of the 'timestep' column,
    the absolute position in the sequence of the observation `obs[t]` of the same
    row; the observation and cue columns are the raw (unshifted) ones at those
    timesteps. The norm of a row is the activity after seeing `obs[t]`, and its
    derivative the change in activity on the arrival of `obs[t]` (NaN at t = 0).
    The probabilities CSVs use the same 'timestep' column, so the two kinds of
    output are aligned by joining on it (merge_activations_and_probabilities).

    `prior_norms` / `prior_derivs`, when not None (get_module_output_and_activity(...,
    return_prior=True)), add the same columns computed on the prior hidden states,
    under a 'prior_' prefix. The unprefixed columns always hold the posterior pass.
    """
    n_out = next(iter(norms.values())).shape[0]

    obs_out = obs[:n_out]
    cue_out = get_two_cue_encoding(cue)[:n_out]  # (n_out, 2)

    out_df = pd.DataFrame({
        'timestep':     np.arange(n_out),
        'observation':  obs_out,
        'cue_1':        cue_out[:, 0].astype(int),
        'cue_2':        cue_out[:, 1].astype(int),
    })

    add_module_activity_columns(out_df, norms, derivs, n_out)

    if prior_norms is not None:
        add_module_activity_columns(out_df, prior_norms, prior_derivs, n_out,
                                    prefix='prior_')
    return out_df


def build_activations_frame(obs, cue, norms, derivs, lim_std, d, tau_std, trial_n,
                            prior_norms=None, prior_derivs=None):
    """Per-timestep activity CSV of one sequence: one row per model output timestep.

    Rows are the T-1 model output timesteps t = 0 ... T-2 ('timestep' column); the
    sequence columns are the raw (unshifted) observation and cue at those same
    timesteps. See build_activity_columns_frame for what the norms and
    derivatives of a row are.

    `prior_norms` / `prior_derivs`, when given (get_module_output_and_activity(...,
    return_prior=True)), add the same columns computed on the prior hidden states,
    under a 'prior_' prefix. The unprefixed columns always hold the posterior pass.
    """
    out_df = build_activity_columns_frame(obs, cue, norms, derivs, prior_norms, prior_derivs)
    n_out = len(out_df)
    out_df['lim_std'] = lim_std
    out_df['d'] = d
    out_df['tau_std'] = tau_std
    out_df['trial_n'] = trial_n.values[:n_out]
    return out_df


def build_pool_activations_frame(obs, cue, norms, derivs, timesteps, prior_norms, prior_derivs):
    """Per-timestep activity CSV of one pool of sequences presented one after the other.

    Same columns as build_activations_frame, from one forward pass over the joined
    sequences of the pool (analysis_core.load_pool_sequence). The sequence
    parameters change from one sequence of the pool to the next, so they are
    per-timestep columns here, taken from `timesteps`: 'sequence_name',
    'pool_position', 'lim_std', 'd', 'tau_std' and 'trial_n'. Rows are the T_total-1
    model output timesteps t = 0 ... T_total-2: the last timestep of the last
    sequence has none. 'timestep' runs over the joined sequences of the pool.
    """
    out_df = build_activity_columns_frame(obs, cue, norms, derivs, prior_norms, prior_derivs)
    n_out = len(out_df)
    for column in ['sequence_name', 'pool_position', 'lim_std', 'd', 'tau_std', 'trial_n']:
        out_df[column] = timesteps[column].to_numpy()[:n_out]
    return out_df


def add_module_deviant_activity_columns(out_df, norms, derivs, dev_idx, n_out, prefix=''):
    """Append per-module activity sampled at the deviant timestep of each trial.

    Same column block as add_module_activity_columns, but one row per trial: each
    module's norm and derivative are gathered at that trial's deviant timestep
    `dev_idx`: the norm is the activity after seeing the deviant, the derivative
    the change in activity on its arrival, derivs[dev_idx] (see compute_derivatives).
    """
    for name in MODULES:
        out_df[f'{prefix}{name}_norm'] = read_trial_at(norms[name], dev_idx, n_out)
    for name in MODULES:
        out_df[f'{prefix}{name}_deriv'] = read_trial_at(derivs[name], dev_idx, n_out)


def build_deviant_activations_frame(obs, dpos_raw, norms, derivs, lim_std, d, tau_std,
                                    trial_n, period, dpos_shift,
                                    prior_norms=None, prior_derivs=None):
    """One row per trial, sampled at the deviant timestep.

    Two distinct uses of dpos are kept separate:
      * the PHYSICAL within-trial timestep of the deviant uses the RAW experimental
        position (that is literally where the deviant tone sits in the sequence);
      * the stored deviant_pos LABEL is shifted into the model's convention so it
        matches the probabilities stage's dpos column and the model's class mapping.

    `prior_norms` / `prior_derivs`, when given, add the same columns computed on the
    prior hidden states, under a 'prior_' prefix.
    """
    n_out = next(iter(norms.values())).shape[0]     # T-1: norms and derivs at timesteps 0 ... T-2

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
        'timestep':     dev_idx,
        'trial_n':      trial_ids,
        'deviant_pos':  deviant_pos,
    })

    add_module_deviant_activity_columns(out_df, norms, derivs, dev_idx, n_out)

    if prior_norms is not None:
        add_module_deviant_activity_columns(out_df, prior_norms, prior_derivs, dev_idx,
                                            n_out, prefix='prior_')

    out_df['lim_std'] = lim_std
    out_df['d'] = d
    out_df['tau_std'] = tau_std
    return out_df


def add_module_probability_columns(out_df, probs, obs_gt, ctx_gt, dpos_gt, rule_gt,
                                   dpos_min, prefix=''):
    """Append one block of module distributions + ground-truth likelihoods to `out_df`.

    Called once per readout of a forward pass: with prefix='' for the posterior
    readouts (the ones supervised during training) and with prefix='prior_' for the
    prior readouts, so the two blocks sit side by side in the same row.

    Parameters
    ----------
    out_df : pandas.DataFrame
        Frame to add the columns to, modified in place.
    probs : dict
        Module name → ndarray of shape (n_out, dim), batch dimension already squeezed.
    obs_gt, ctx_gt, dpos_gt, rule_gt : ndarray
        Ground truth each output row is predicting, shape (n_out,). dpos_gt is in the
        model's convention, so dpos_gt - dpos_min is a valid 0-based class index.
    dpos_min : int
        Class-index offset of the dpos module.
    prefix : str
        Prepended to every column name written here.
    """
    # Observation module is a regressor: mean and variance of the Gaussian
    out_df[f'{prefix}obs_mean'] = probs['obs'][:, 0]
    out_df[f'{prefix}obs_var'] = probs['obs'][:, 1]

    # Classifier modules: one column per class probability
    for module_name in ('ctx', 'dpos', 'rule'):
        module_probs = probs[module_name]  # (n_out, n_classes)
        for c in range(module_probs.shape[1]):
            out_df[f'{prefix}{module_name}_p{c}'] = module_probs[:, c]

    # --- Likelihood of the ground truth under each module's distribution ---
    # obs: Gaussian density of the true observation under (mean, variance).
    out_df[f'{prefix}lik_obs'] = gaussian_likelihood(obs_gt, probs['obs'][:, 0], probs['obs'][:, 1])

    # Classifiers: probability assigned to the true class. ctx and rule labels
    # are already 0-based; dpos must be shifted by its minimum to index into
    # the class-probability columns (matches the dpos_min convention used in
    # training/eval, e.g. pipeline_core_v2 get_model_predictions).
    out_df[f'{prefix}lik_ctx'] = class_likelihood(probs['ctx'], ctx_gt)
    out_df[f'{prefix}lik_dpos'] = class_likelihood(probs['dpos'], dpos_gt - dpos_min)
    out_df[f'{prefix}lik_rule'] = class_likelihood(probs['rule'], rule_gt)


def build_probabilities_frame(obs, cue, ctx, dpos_model, rule, probs,
                              lim_std, d, tau_std, trial_n, dpos_min,
                              prior_probs=None):
    """Predicted distributions + ground-truth likelihoods, per timestep.

    Rows are the T-1 model outputs; output k is the prediction made after seeing
    `obs[k]`, of the observation of timestep k+1. The 'timestep' column is that
    predicted timestep, 1 ... T-1, so that the 'observation' of a row is
    `obs[timestep]`, as in the activations CSVs (build_activity_columns_frame),
    and the two align by joining on 'timestep'
    (merge_activations_and_probabilities).

    `dpos_model` must already be in the model's convention (raw + shift), so that
    dpos_model - dpos_min is a valid 0-based class index.

    `prior_probs`, when given (get_module_output_and_activity(..., return_prior=True)),
    adds the same columns computed on the prior readouts, under a 'prior_' prefix.
    The unprefixed columns always hold the posterior readouts, so files written
    without priors stay readable by the downstream summaries unchanged.
    """
    # Number of timesteps in the output: T-1 (the model is run on y[:, :-1]).
    # The model output at index k predicts timestep k+1, so the outputs align
    # with the ground truth shifted by one: y[1:], q[1:], labels[1:].
    n_out = next(iter(probs.values())).shape[0]

    # Next-step prediction alignment: ground truth that each output row is predicting.
    obs_gt = obs[1:n_out + 1]           # (n_out,)
    cue_gt = get_two_cue_encoding(cue)[1:n_out + 1]  # (n_out, 2)
    ctx_gt = ctx[1:n_out + 1]           # (n_out,)
    dpos_gt = dpos_model[1:n_out + 1]   # (n_out,)
    rule_gt = rule[1:n_out + 1]         # (n_out,)

    # Squeeze batch dim (batch=1) from each module's probabilities
    probs = {name: arr[:, 0, :] for name, arr in probs.items()}  # each: (n_out, dim)

    # Sequence info columns (same fields as the activations stage), now carrying the
    # next-step (prediction-target) values so every column in a row refers to the
    # same timestep as that row's likelihoods.
    out_df = pd.DataFrame({
        'timestep':     np.arange(1, n_out + 1),
        'observation':  obs_gt,
        'cue_1':        cue_gt[:, 0].astype(int),
        'cue_2':        cue_gt[:, 1].astype(int),
        'ctx':          ctx_gt,
        'dpos':         dpos_gt,
        'rule':         rule_gt,
    })

    add_module_probability_columns(out_df, probs, obs_gt, ctx_gt, dpos_gt, rule_gt, dpos_min)

    if prior_probs is not None:
        prior_probs = {name: arr[:, 0, :] for name, arr in prior_probs.items()}
        add_module_probability_columns(out_df, prior_probs, obs_gt, ctx_gt, dpos_gt,
                                       rule_gt, dpos_min, prefix='prior_')

    out_df['lim_std'] = lim_std
    out_df['d'] = d
    out_df['tau_std'] = tau_std
    out_df['trial_n'] = trial_n.values[1:n_out + 1]
    return out_df


def merge_activations_and_probabilities(activations_df, probabilities_df):
    """Activations and probabilities of one sequence, one row per timestep of the sequence.

    Outer join of the two CSVs on 'timestep', the absolute position in the
    sequence of the observation of the row (t = 0 ... T-1). The edge timesteps
    are kept with NaN where an output does not exist: the probabilities have no
    row at t = 0 (nothing precedes the first observation), the activations no
    row at t = T-1 (the model is not run on the last observation), and the
    derivatives none at t = 0. The columns shared by both CSVs (observation,
    cues, sequence parameters, trial_n) are filled from whichever side has the
    row, and must agree where both have it.

    Both CSVs must come from the same sequence (or pool) and model run.
    """
    merged_df = pd.merge(activations_df, probabilities_df, on='timestep', how='outer',
                         suffixes=('', '_probabilities'))
    shared_columns = [c for c in activations_df.columns
                      if c in probabilities_df.columns and c != 'timestep']
    for column in shared_columns:
        from_probabilities = merged_df.pop(f'{column}_probabilities')
        both = merged_df[column].notna() & from_probabilities.notna()
        if not np.allclose(merged_df.loc[both, column].to_numpy(dtype=float),
                           from_probabilities[both].to_numpy(dtype=float)):
            raise ValueError(f"Column '{column}' differs between the activations and "
                             f"probabilities rows of the same timestep")
        merged_df[column] = merged_df[column].combine_first(from_probabilities)
    return merged_df.sort_values('timestep').reset_index(drop=True)


def subset_deviant_rows(out_df, include_next_stimulus):
    """Deviant-only subset of a probabilities frame (rows whose context label is 1)."""
    deviant_mask = out_df['ctx'] == 1
    if include_next_stimulus:
        # Deviants + the immediate next stimulus.
        deviant_mask = deviant_mask | deviant_mask.shift(1, fill_value=False)
    return out_df[deviant_mask]


# =============================================================================
# Aggregation: deviant likelihood summaries
# =============================================================================

# ── Reading and summarising ───────────────────────────────────────────────────
def add_derived_likelihoods(df):
    """Add the likelihood columns the extraction stage does not store."""
    df["cdf_lik_obs"] = scipy.stats.norm.cdf(
        (df["observation"] - df["obs_mean"]) / np.sqrt(df["obs_var"])
    )
    df["log_lik_obs"] = np.log(df["lik_obs"] + 1e-8)  # Add small constant to avoid log(0)
    return df


def summarise_deviant_files(deviant_path, model_name, value_cols=None,
                            group_col=None, param_cols=None, trials_path=None):
    """Mean of each likelihood column over one model's deviant-probability files.

    This is the single summariser behind both figures that read
    ``probabilities_deviant/*.csv``: the likelihood-averages tables and the
    distribution of the dpos probability at the true deviant. They differed only
    in which columns they averaged and where the sequence parameters came from,
    so both are parameters here.

    Parameters
    ----------
    deviant_path : Path
        A model's ``probabilities_deviant/`` folder.
    model_name : str
        Carried into the ``model`` column.
    value_cols : list of str, optional
        Columns to average. Defaults to every likelihood column.
    group_col : str, optional
        When given (e.g. 'dpos'), one row per (file, value) instead of one row
        per file, so a per-file mean does not mix deviant positions.
    param_cols : list of str, optional
        Generative parameters to carry over. Defaults to the three scalars
        stored in the deviant files themselves, or, when `trials_path` is given,
        to the wider set held only by the original sequence files.
    trials_path : Path, optional
        When given, parameters are joined from ``<sequence>.csv`` there rather
        than read out of the deviant file. This is the only way to recover the
        parameters the extraction does not copy forward (sigma_r, run_n, ...).

    Returns
    -------
    DataFrame, empty (with a warning) when the model has no deviant files, so a
    partially processed model list still yields the other models' figures.

    Means skip NaN throughout: a row whose prediction is undefined is stored as
    NaN and must never be counted as a zero. ``n_valid_<col>`` reports how many
    rows actually contributed to each mean, so an all-NaN group is visible as a
    NaN mean over zero valid rows rather than silently dropped.
    """
    value_cols = list(LIKELIHOOD_COLS if value_cols is None else value_cols)
    if param_cols is None:
        param_cols = SEQUENCE_PARAM_COLS if trials_path is not None else DEVIANT_PARAM_COLS
    param_cols = list(param_cols)

    deviant_files = sorted(Path(deviant_path).glob("*_probabilities_deviant.csv"))
    if not deviant_files:
        print(f"  Warning: no *_probabilities_deviant.csv files in {deviant_path}; skipping.")
        return pd.DataFrame()

    print(f"Found {len(deviant_files)} deviant file(s) in {deviant_path}")

    rows = []
    for f in deviant_files:
        df = add_derived_likelihoods(pd.read_csv(f))
        sequence = f.name.replace('_probabilities_deviant.csv', '')

        base = {'model': model_name, 'file': f.name, 'sequence': sequence}
        base.update(_sequence_params(df, sequence, param_cols, trials_path))

        # A single group holding every row reproduces the per-file average.
        groups = df.groupby(group_col, sort=True) if group_col else [(None, df)]
        for key, sub in groups:
            row = dict(base)
            if group_col:
                row[group_col] = key
            row['n_rows'] = len(sub)
            for col in value_cols:
                values = (sub[col].to_numpy(dtype=float) if col in sub.columns
                          else np.full(len(sub), np.nan))
                valid = ~np.isnan(values)
                row[f'mean_{col}'] = values[valid].mean() if valid.any() else np.nan
                row[f'n_valid_{col}'] = int(valid.sum())
            rows.append(row)

    return pd.DataFrame(rows)


def _sequence_params(df, sequence, param_cols, trials_path):
    """The generative parameters of one sequence, from the deviant file or the trial file.

    Both sources hold one constant value per column across the sequence, so the
    first row is taken; a column that is not constant is reported rather than
    silently reduced.
    """
    if trials_path is None:
        return {col: df[col].iloc[0] for col in param_cols if col in df.columns}

    trial_file = Path(trials_path) / (sequence + '.csv')
    if not trial_file.exists():
        print(f"  Warning: no trial file for {sequence}; parameters set to NaN.")
        return {col: np.nan for col in param_cols}

    tdf = pd.read_csv(trial_file, usecols=lambda c: c in param_cols)
    params = {}
    for col in param_cols:
        if col not in tdf.columns:
            params[col] = np.nan
            continue
        if tdf[col].nunique(dropna=False) > 1:
            print(f"  Warning: '{col}' is not constant in {trial_file.name}; using first row.")
        params[col] = tdf[col].iloc[0]
    return params


def write_summary_csv(summary_df, deviant_path, model_name, group_col=None, suffix=''):
    """Save one model's averages to an ``average/`` folder beside its deviant files."""
    average_path = Path(deviant_path) / "average"
    average_path.mkdir(parents=True, exist_ok=True)

    group_suffix = f"_by_{group_col}" if group_col else ""
    out_file = average_path / f"{model_name}_likelihood_averages{suffix}{group_suffix}.csv"
    summary_df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")
    return out_file


def build_likelihood_summaries(base_output_root, model_names, group_col=None,
                               save_csv=True, **kwargs):
    """{model name: summary frame}, one entry per model that has deviant files.

    Models without data are reported and left out; FileNotFoundError is raised
    only when none of them has any. Extra keyword arguments are forwarded to
    summarise_deviant_files (value_cols, param_cols, trials_path).
    """
    summaries = {}
    for model_name in model_names:
        deviant_path = Path(base_output_root) / model_name / "probabilities_deviant"
        summary_df = summarise_deviant_files(deviant_path, model_name,
                                             group_col=group_col, **kwargs)
        if summary_df.empty:
            continue
        if save_csv:
            write_summary_csv(summary_df, deviant_path, model_name, group_col=group_col)
        summaries[model_name] = summary_df

    if not summaries:
        raise FileNotFoundError(
            f"No *_probabilities_deviant.csv files found for any of {model_names} "
            f"under {base_output_root}")
    return summaries


# =============================================================================
# Aggregation: module-pair correlation scores
# =============================================================================

def compute_pairwise_module_correlations(module_norms_dict, use_derivatives=False):
    """Pearson correlation between every (unordered) pair of modules.

    Parameters
    ----------
    module_norms_dict : dict
        Keys are module names, values are norms arrays of shape (seq_len, batch)
        (or 1D). Flattened across time and samples before correlating.
    use_derivatives : bool
        If True, correlate the temporal derivatives instead of the raw activity.

    Returns
    -------
    dict mapping "<name_i>_<name_j>" -> Pearson r (float), for i < j in the
    order the modules appear in `module_norms_dict`. A pair with a constant
    (zero-variance) vector gets a correlation of 0.0 instead of NaN.
    """
    module_names = list(module_norms_dict.keys())
    pair_correlations = {}

    for i, name_i in enumerate(module_names):
        for j, name_j in enumerate(module_names):
            if i < j:
                activity_i = module_norms_dict[name_i]
                activity_j = module_norms_dict[name_j]

                if use_derivatives:
                    # Row 0 of the derivatives is NaN, nothing precedes it.
                    activity_i = compute_derivatives(activity_i)[1:]
                    activity_j = compute_derivatives(activity_j)[1:]

                flat_i = activity_i.reshape(-1)
                flat_j = activity_j.reshape(-1)

                # pearsonr is undefined for a constant input; treat that as no correlation.
                if flat_i.std() == 0 or flat_j.std() == 0:
                    corr = 0.0
                else:
                    corr = pearsonr(flat_i, flat_j)[0]

                pair_correlations[f"{name_i}_{name_j}"] = float(corr)

    return pair_correlations


# Kept when model_activations.py was retired: this had no callers anywhere in
# the workspace, but it is the natural summary over the pairwise scores above,
# so it is preserved here rather than dropped.
def compute_intermodule_correlations(module_norms_dict, use_derivatives=False, absolute=False):
    """Average pairwise Pearson correlation between modules.

    Parameters
    ----------
    module_norms_dict : dict
        Keys are module names, values are norms arrays
    use_derivatives : bool
        If True, compute correlations on derivatives instead of raw activity
    absolute : bool
        If True, average the absolute value of each pairwise correlation (a
        redundancy/coupling-strength metric, where +0.8 and -0.8 both count as
        strong). If False (default), average the signed correlations, so
        positive and negative pairs can cancel.

    Returns
    -------
    float - average correlation between modules
    """
    pair_correlations = compute_pairwise_module_correlations(
        module_norms_dict, use_derivatives=use_derivatives
    )
    values = list(pair_correlations.values())
    if not values:
        return 0.0
    if absolute:
        values = [abs(v) for v in values]
    return float(np.mean(values))


def load_module_dict(df, suffix):
    """Build {module_name: 1D array} from columns named '<module>_<suffix>'.

    Rows with NaN (e.g. the padded first derivative timestep) are dropped so all
    modules keep the same length.
    """
    cols = [f'{m}_{suffix}' for m in MODULES]
    values = df[cols].dropna()
    return {m: values[f'{m}_{suffix}'].to_numpy() for m in MODULES}


def aggregate_scores(pair_correlations):
    """Three ways to summarise the per-pair correlations into one score.

    - mean_abs: mean of the absolute per-pair correlations (coupling strength;
      anti-correlated pairs still count).
    - abs_mean: absolute value of the mean of the signed correlations (lets
      opposite-sign pairs cancel, then drops the sign).
    - mean:     plain mean of the signed correlations (keeps the sign).
    """
    values = np.array(list(pair_correlations.values()))
    return {
        'mean_abs': float(np.mean(np.abs(values))),
        'abs_mean': float(abs(np.mean(values))),
        'mean': float(np.mean(values)),
    }


# =============================================================================
# Aggregation: activity against likelihood, per trial
# =============================================================================

def load_deviant_activity(act_csv):
    """Load a per-trial deviant-activity CSV.

    Returns the DataFrame as written: one row per trial with columns
    trial_n, deviant_pos, <module>_norm, <module>_deriv, lim_std, d, tau_std.
    """
    return pd.read_csv(act_csv)


def load_deviant_likelihoods(prob_csv, context_label=1, module_names=MODULE_NAMES):
    """Per-trial likelihoods at the deviant position from a *_probabilities.csv.

    The probabilities file has one row per timestep, where the ``ctx`` column is
    the next-step ground-truth context label (1 marks the deviant tone). Filtering
    to ``ctx == context_label`` therefore keeps exactly the deviant of each trial,
    and ``lik_<module>`` on that row is the likelihood the model assigned to the
    deviant. There is one such row per trial (verified on the data), but if a file
    ever had more, duplicates are averaged within a trial.

    Parameters
    ----------
    prob_csv : str or Path
        Path to a *_probabilities.csv file.
    context_label : int
        Context label marking the deviant tone (default 1).
    module_names : list of str
        Modules whose likelihood columns (``lik_<module>``) to keep.

    Returns
    -------
    pandas.DataFrame
        Columns: trial_n, deviant_pos, lik_<module> for each module.
    """
    df = pd.read_csv(prob_csv)
    dev = df[df['ctx'] == context_label].copy()
    lik_cols = [f'lik_{m}' for m in module_names]
    keep = ['trial_n', 'dpos'] + lik_cols
    dev = dev[keep].rename(columns={'dpos': 'deviant_pos'})
    dev = dev.groupby(['trial_n', 'deviant_pos'], as_index=False)[lik_cols].mean()
    return dev


def join_activity_likelihood(act_df, lik_df, module_names=MODULE_NAMES):
    """Merge per-trial deviant activity and deviant likelihoods on trial_n.

    Adds a ``mean_norm`` column: the mean activity across modules at the deviant
    (the average module activity referred to in the analysis). The merge is an
    inner join on trial_n, so only trials present in both frames are kept.

    Parameters
    ----------
    act_df : pandas.DataFrame
        Output of :func:`load_deviant_activity`.
    lik_df : pandas.DataFrame
        Output of :func:`load_deviant_likelihoods`.
    module_names : list of str
        Modules to average over for ``mean_norm``.

    Returns
    -------
    pandas.DataFrame
        The merged per-trial frame with an added ``mean_norm`` column.
    """
    merged = pd.merge(act_df, lik_df, on='trial_n', how='inner',
                      suffixes=('', '_lik'))
    norm_cols = [f'{m}_norm' for m in module_names]
    merged['mean_norm'] = merged[norm_cols].mean(axis=1)
    return merged


def compute_activity_likelihood_correlations(df, activity_cols, likelihood_cols,
                                             method='pearson'):
    """Correlation of every activity column with every likelihood column.

    Parameters
    ----------
    df : pandas.DataFrame
        Joined per-trial frame.
    activity_cols, likelihood_cols : list of str
        Column names to cross-correlate (activity on rows, likelihood on columns).
    method : {'pearson', 'spearman'}
        Correlation coefficient to use.

    Returns
    -------
    pandas.DataFrame
        Correlation matrix indexed by activity_cols, columns likelihood_cols.
        Pairs with fewer than 3 finite points or a constant column are NaN.
    """
    func = pearsonr if method == 'pearson' else spearmanr
    corr = pd.DataFrame(index=activity_cols, columns=likelihood_cols, dtype=float)
    for a in activity_cols:
        for l in likelihood_cols:
            x = df[a].to_numpy(dtype=float)
            y = df[l].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if x.size >= 3 and x.std() > 0 and y.std() > 0:
                corr.loc[a, l] = float(func(x, y)[0])
            else:
                corr.loc[a, l] = np.nan
    return corr


# =============================================================================
# Reading back what the pipeline wrote
# =============================================================================

def model_paths(model_name, base_dir=BASE_OUTPUT_DIR):
    """Resolve the input/output paths for one model.

    Returns
    -------
    dict with keys
        'activations_dir' : per-sequence '*_activations.csv' files
        'correlation_csv' : the per-sequence correlation scores table
        'figures_dir'     : where this script writes its PNGs (created)
    """
    model_dir = Path(base_dir) / model_name
    activations_dir = model_dir / "activations"
    figures_dir = model_dir / "module_correlations"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        'activations_dir': activations_dir,
        'correlation_csv': activations_dir / "correlation_scores" / "exp_trials_correlations.csv",
        'figures_dir': figures_dir,
    }


def load_correlation_scores(correlation_csv):
    """[cell 1] Load the per-sequence correlation scores table.

    One row per experimental sequence, written by run_module_correlations.py.
    Sorted by the generative parameters so the hue levels come out in a stable
    order across figures.
    """
    act_df = pd.read_csv(correlation_csv, index_col=0)
    act_df.sort_values(by=['lim_std', 'd', 'tau_std'], inplace=True)
    return act_df


def list_sequence_files(activations_dir):
    """Sorted list of the per-sequence activation files ('*_activations.csv')."""
    return sorted(glob.glob(os.path.join(str(activations_dir), "*_activations.csv")))


def load_sequence_means(seq_files):
    """[cells 25 & 27] One row per sequence: mean activity of each module.

    One file = one sequence. Also carries the sequence's generative parameters
    (lim_std, d, tau_std), which are constant within a file.
    """
    records = []
    for f in seq_files:
        seq_df = pd.read_csv(f)
        rec = {m: seq_df[f'{m}_norm'].mean() for m in MODULES}
        rec['lim_std'] = seq_df['lim_std'].iloc[0]
        rec['d'] = seq_df['d'].iloc[0]
        rec['tau_std'] = seq_df['tau_std'].iloc[0]
        records.append(rec)
    return pd.DataFrame(records)


def load_timestep_means(seq_files, with_position=False):
    """[cells 31 & 33] One row per timestep: module activity averaged over sequences.

    With `with_position`, also returns the 1-based within-trial position of each
    timestep (derived from ``trial_n``), plus the sorted position order used as
    ``hue_order``. The position is consistent across sequences, so averaging it
    over sequences and rounding recovers the integer position.
    """
    per_ts = []
    for f in seq_files:
        seq_df = pd.read_csv(f)
        sub = seq_df[[f'{m}_norm' for m in MODULES]].copy()
        sub['timestep'] = seq_df['timestep']
        if with_position:
            # 0-based index of the timestep within its trial
            sub['position'] = seq_df.groupby('trial_n').cumcount()
        per_ts.append(sub)

    ts_means = pd.concat(per_ts).groupby('timestep').mean().reset_index()

    if not with_position:
        return ts_means, None

    ts_means['position'] = ts_means['position'].round().astype(int) + 1
    pos_order = [str(p) for p in sorted(ts_means['position'].unique())]
    # categorical -> discrete colours; column name -> legend title "position"
    ts_means['position'] = ts_means['position'].astype(str)
    return ts_means, pos_order


def load_activity_norms(act_files, prior):
    """{module: (T-1, n_files) activity norms}, stacked from per-sequence activations CSVs.

    The norms of the posterior hidden states ('<module>_norm' columns), or with
    `prior` those of the prior hidden states ('prior_<module>_norm' columns,
    written by run_exp_trials_pipeline.py with INCLUDE_PRIOR = True).

    The layout get_module_output_and_activity returns for a batch, so the activity
    figures draw the same arrays whether they come from a forward pass or from
    disk. Sequences of unequal length are truncated to the shortest one.
    """
    if prior:
        prefix = 'prior_'
    else:
        prefix = ''
    norm_cols = [f'{prefix}{m}_norm' for m in MODULES]
    frames = [pd.read_csv(f, usecols=norm_cols) for f in act_files]
    min_len = min(len(df) for df in frames)
    if any(len(df) != min_len for df in frames):
        print(f"  Warning: sequences have unequal lengths — truncating all to {min_len} timesteps")
    return {m: np.stack([df[f'{prefix}{m}_norm'].to_numpy()[:min_len] for df in frames], axis=1)
            for m in MODULES}


def load_timestep_rules(act_files, sequence_files_by_csv):
    """(T-1, n_files) rule of the trial every timestep of the activations CSVs belongs to.

    The same layout and truncation as load_activity_norms, so that the rule of a
    timestep sits at the same place as its activity norm. The activations CSVs do
    not hold the rule, so it is read from the sequence files: for every CSV of
    `act_files`, `sequence_files_by_csv` gives the sequence files it was written
    from, in presentation order (the one file of a sequence, or the four files of
    a pool). The rows of a CSV are the timesteps of these files joined end to end,
    without the last one. A ValueError is raised when the observations of a CSV
    are not those of its sequence files.
    """
    rules_per_file = []
    for act_file, sequence_files in zip(act_files, sequence_files_by_csv):
        act_observations = pd.read_csv(act_file, usecols=['observation'])['observation'].to_numpy()
        sequence_dfs = []
        for sequence_file in sequence_files:
            sequence_dfs.append(pd.read_csv(sequence_file, usecols=['observation', 'rule']))
        joined_df = pd.concat(sequence_dfs, ignore_index=True).iloc[:len(act_observations)]
        if not np.allclose(joined_df['observation'].to_numpy(), act_observations):
            raise ValueError(f"The observations of {act_file.name} are not those of "
                             f"{[f.name for f in sequence_files]}")
        rules_per_file.append(joined_df['rule'].to_numpy())
    min_len = min(len(rules) for rules in rules_per_file)
    return np.stack([rules[:min_len] for rules in rules_per_file], axis=1)


def load_deviant_activity_frame(dev_files, dpos_shift):
    """Every per-trial deviant-activity CSV of `dev_files`, concatenated.

    Adds 'dev_pos', the 0-based within-trial position the deviant physically sits
    at: the stored deviant_pos label is in the model's convention (raw + shift),
    whereas grouping trials by where the deviant is must not depend on the model.
    """
    dev_df = pd.concat([pd.read_csv(f) for f in dev_files], ignore_index=True)
    dev_df['dev_pos'] = dev_df['deviant_pos'] - dpos_shift
    return dev_df


def predictions_from_probabilities(prob_df, dpos_min):
    """One sequence's model predictions, rebuilt from its probabilities CSV.

    Same keys and layout as pipeline_core_v2.get_model_predictions returns for a
    batch of one sequence: (1, T-1) for estimates and predicted labels, and
    (1, T-1, n_classes) for class probabilities. Predicted dpos labels are in the
    model's convention (class index + dpos_min), like the dpos column of the CSV.
    """
    pred = {'mu_estim': prob_df['obs_mean'].to_numpy()[None],
            'sigma_estim': np.sqrt(prob_df['obs_var'].to_numpy())[None]}
    for module in ('ctx', 'dpos', 'rule'):
        probs = prob_df[class_probability_columns(prob_df.columns, module)].to_numpy()[None]
        pred[f'{module}_prob'] = probs
        pred[f'{module}_pred'] = probs.argmax(axis=-1)
    pred['dpos_pred'] = pred['dpos_pred'] + dpos_min
    return pred


if __name__ == '__main__':
    pass
