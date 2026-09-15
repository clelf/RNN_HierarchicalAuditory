"""Scoring a trained model, and comparing it against the Kalman-filter benchmark.

Three groups:

  1. Metrics      -- one function per quantity (MSE, Gaussian log-likelihood,
                     calibration KS, and the accuracy / log-probability of the
                     ctx, dpos and rule modules).
  2. KF benchmark -- the hierarchical Kalman filter's marginal predictive
                     likelihood, used as the reference an RNN is scored against.
  3. Drivers      -- evaluate_model (one model) and evaluate_models (a list of
                     them, returning a tidy DataFrame), plus the benchmark
                     comparison that turns those into model/KF ratios.

Split out of evaluate_models.py; the bodies are unchanged. Model loading and
test-data generation live in analysis_core.py, plotting in plots.py.
"""

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats, special
from tqdm import tqdm

from analysis_core import (
    ModelInfo,
    load_model,
    generate_test_data,
    get_model_predictions,
)

# Kalman-filter benchmark machinery (config-agnostic numerics).
from Kalman.kalman import (
    kalman_online_fit_predict_multicontext,
    likelihood_observation,
    contexts_to_probabilities,
)
# The deviant-position conditional and its marginalization over rules are REUSED
# from PreProParadigm/model_RTs.py (single source of truth). They accept a
# `valid_positions` arg so the same math applies to the RNN HierarchicalGM's
# rules_dpos_set instead of the old hard-coded positions.
from PreProParadigm.model_RTs import (
    prior_dpos_given_prev_rule,
    pior_dpos_given_prev_rule_and_stds,
)


# =============================================================================
# Metrics
# =============================================================================
def compute_mse(y_true: np.ndarray, y_pred: np.ndarray, reduce: bool = True) -> Union[np.ndarray, float]:
    """
    Compute Mean Squared Error.
    
    Args:
        y_true: True observations (n_samples, seq_len)
        y_pred: Predicted observations (n_samples, seq_len)
        reduce: If True, return scalar mean. If False, return per-sample MSE (n_samples,)
    
    Returns:
        Scalar MSE if reduce=True, else array of per-sample MSE values.
    """
    squared_errors = (y_true - y_pred) ** 2
    if reduce:
        return float(np.mean(squared_errors))
    else:
        # Mean over sequence dimension, keep sample dimension
        return np.mean(squared_errors, axis=1)


def compute_log_likelihood(y_true: np.ndarray, mu_pred: np.ndarray, var_pred: np.ndarray, 
                           reduce: bool = True) -> Union[np.ndarray, float]:
    """
    Compute Gaussian log-likelihood.
    
    log p(y|mu, var) = -0.5 * [log(2*pi*var) + (y-mu)^2/var]
    
    Args:
        y_true: True observations (n_samples, seq_len)
        mu_pred: Predicted mean (n_samples, seq_len)
        var_pred: Predicted variance (n_samples, seq_len)
        reduce: If True, return scalar mean. If False, return per-sample log-likelihood (n_samples,)
    
    Returns:
        Scalar mean log-likelihood if reduce=True, else array of per-sample values.
    """
    log_prob = -0.5 * (np.log(2 * np.pi * var_pred) + (y_true - mu_pred) ** 2 / var_pred)
    if reduce:
        return float(np.mean(log_prob))
    else:
        # Mean over sequence dimension, keep sample dimension
        return np.mean(log_prob, axis=1)


def compute_calibration_ks(y_true: np.ndarray, mu_pred: np.ndarray, var_pred: np.ndarray,
                           reduce: bool = True) -> Union[Tuple[float, float], np.ndarray]:
    """
    Compute calibration using Kolmogorov-Smirnov statistics.
    
    For a well-calibrated model, the z-scores should be standard normal.
    z = (y - mu) / sigma
    
    Two modes:
    - reduce=True (default): Global KS test pooling all data. Returns (ks_stat, p_value).
      This is a formal hypothesis test with p-value.
    
    - reduce=False: Per-sample KS statistics. Returns array of shape (n_samples,).
      Useful for reporting mean ± SEM across samples, which captures sample-to-sample
      variability in calibration. This is a descriptive statistic, not a hypothesis test.
    
    Args:
        y_true: True observations (n_samples, seq_len)
        mu_pred: Predicted mean (n_samples, seq_len)
        var_pred: Predicted variance (n_samples, seq_len)
        reduce: If True, compute global KS test. If False, compute per-sample KS stats.
    
    Returns:
        If reduce=True: (ks_statistic, p_value) tuple
        If reduce=False: Array of per-sample KS statistics (n_samples,)
        
        Lower KS statistic = better calibration
    """
    sigma_pred = np.sqrt(var_pred)
    
    if reduce:
        z_scores = (y_true - mu_pred) / sigma_pred
        # Flatten for KS test - pool all samples and timesteps
        z_flat = z_scores.flatten()
        # KS test against standard normal
        ks_stat, p_value = stats.kstest(z_flat, 'norm')
        return float(ks_stat), float(p_value)
    else:
        # Per-sample KS statistic using measure_KS_stat
        x = y_true[:, :, np.newaxis] if y_true.ndim == 2 else y_true
        u = mu_pred[:, :, np.newaxis] if mu_pred.ndim == 2 else mu_pred
        s = sigma_pred[:, :, np.newaxis] if sigma_pred.ndim == 2 else sigma_pred
        return measure_KS_stat(x, u, s)


def measure_KS_stat(x, u, s):
	""" Measures the calibration (K-S statistic) of a prediction on a set of observations 

		Parameters
		----------
		x : np.array
			Input array enconding sequences of observations; three dimensions: dim0 runs across the
			batch, dim1 runs across timepoints (i.e., trials in the COIN jargon), dim2 set to one.
		u  : np.array
			Mean of the predictions; shifted (i.e., u[:, t] are the predictions for t). Same 
			dimensions as x.
		s  : np.array
			Standard deviation of the predictions; same shape and shifting as u.

		Returns
		----------
		KS : np.array
			One dimensional array with a scalar per sample in the batch. 
	"""

	F    = np.linspace(0, 1, 1000)
	cump = (0.5 * (1 + special.erf((x - u) / (np.sqrt(2) * s))))
	N    = (~np.isnan(cump)).sum(1)
	KS   = abs(np.array([(cump <= f).sum(1) / N for f in F]) - F[:, None, None]).max((0, 2))
	
	return KS


def compute_context_accuracy(contexts_true: np.ndarray, contexts_pred: np.ndarray,
                             reduce: bool = True) -> Union[np.ndarray, float]:
    """
    Compute context inference accuracy using predicted labels.
    
    Args:
        contexts_true: True context labels (n_samples, seq_len)
        contexts_pred: Model context predictions (n_samples, seq_len) - argmax already applied
        reduce: If True, return scalar mean. If False, return per-sample accuracy (n_samples,)
    
    Returns:
        Scalar accuracy if reduce=True, else array of per-sample accuracy values.
    """
    correct = (contexts_pred == contexts_true)
    
    if reduce:
        return float(np.mean(correct))
    else:
        # Mean over sequence dimension, keep sample dimension
        return np.mean(correct, axis=1)


def compute_context_log_likelihood(contexts_true: np.ndarray, context_probs: np.ndarray,
                                    reduce: bool = True) -> Union[np.ndarray, float]:
    """
    Compute log-likelihood of true contexts given predicted probabilities.
    
    This is the log-likelihood of the true context labels under the categorical
    distribution defined by the model's softmax probabilities.
    
    Args:
        contexts_true: True context labels (n_samples, seq_len)
        context_probs: Model context probabilities (n_samples, seq_len, n_ctx) - already softmax applied
        reduce: If True, return scalar mean. If False, return per-sample log-likelihood (n_samples,)
    
    Returns:
        Scalar mean log-likelihood if reduce=True, else array of per-sample values.
    """
    # Get probability of true context for each timestep
    n_samples, seq_len = contexts_true.shape
    true_probs = np.zeros((n_samples, seq_len))
    
    # Calculate minimum context index to handle non-zero-based indexing
    ctx_min = contexts_true.min()
    
    for i in range(n_samples):
        for t in range(seq_len):
            true_ctx = contexts_true[i, t]
            # Shift to 0-based index for probability array access
            true_probs[i, t] = context_probs[i, t, true_ctx - ctx_min]
    
    # Compute log probability
    log_probs = np.log(true_probs + 1e-10)  # Add small epsilon to avoid log(0)
    
    if reduce:
        return float(np.mean(log_probs))
    else:
        # Mean over sequence dimension, keep sample dimension
        return np.mean(log_probs, axis=1)


def compute_dpos_accuracy(dpos_true: np.ndarray, dpos_pred: np.ndarray,
                          reduce: bool = True) -> Union[np.ndarray, float]:
    """
    Compute deviant position inference accuracy using predicted labels.
    
    Args:
        dpos_true: True dpos labels (n_samples, seq_len)
        dpos_pred: Model dpos predictions (n_samples, seq_len) - argmax already applied
        reduce: If True, return scalar mean. If False, return per-sample accuracy (n_samples,)
    
    Returns:
        Scalar accuracy if reduce=True, else array of per-sample accuracy values.
    """
    correct = (dpos_pred == dpos_true)
    
    if reduce:
        return float(np.mean(correct))
    else:
        # Mean over sequence dimension, keep sample dimension
        return np.mean(correct, axis=1)


def compute_dpos_log_prob(dpos_true: np.ndarray, dpos_probs: np.ndarray,
                          reduce: bool = True) -> Union[np.ndarray, float]:
    """
    Compute log-probability of true deviant positions (dpos).
    
    Args:
        dpos_true: True dpos labels (n_samples, seq_len)
        dpos_probs: Model dpos probabilities (n_samples, seq_len, n_dpos) - already softmax applied
        reduce: If True, return scalar mean. If False, return per-sample log-prob (n_samples,)
    
    Returns:
        Scalar mean log-probability if reduce=True, else array of per-sample values.
    """
    # Get probability of true dpos for each timestep
    n_samples, seq_len = dpos_true.shape
    true_probs = np.zeros((n_samples, seq_len))
    
    # Calculate minimum dpos index to handle non-zero-based indexing
    # (e.g., dpos_true contains [3, 4, 5, 6, 7] but dpos_probs is indexed [0, 1, 2, 3, 4])
    dpos_min = dpos_true.min()
    
    for i in range(n_samples):
        for t in range(seq_len):
            true_dpos = dpos_true[i, t]
            # Shift to 0-based index for probability array access
            true_probs[i, t] = dpos_probs[i, t, int(true_dpos - dpos_min)]
    
    # Compute log probability
    log_probs = np.log(true_probs + 1e-10)  # Add small epsilon to avoid log(0)
    
    if reduce:
        return float(np.mean(log_probs))
    else:
        # Mean over sequence dimension, keep sample dimension
        return np.mean(log_probs, axis=1)


def compute_rule_accuracy(rules_true: np.ndarray, rules_pred: np.ndarray,
                          reduce: bool = True) -> Union[np.ndarray, float]:
    """
    Compute rule inference accuracy using predicted labels.
    
    Args:
        rules_true: True rule labels (n_samples, seq_len)
        rules_pred: Model rule predictions (n_samples, seq_len) - argmax already applied
        reduce: If True, return scalar mean. If False, return per-sample accuracy (n_samples,)
    
    Returns:
        Scalar accuracy if reduce=True, else array of per-sample accuracy values.
    """
    correct = (rules_pred == rules_true)
    
    if reduce:
        return float(np.mean(correct))
    else:
        # Mean over sequence dimension, keep sample dimension
        return np.mean(correct, axis=1)


def compute_rule_log_prob(rules_true: np.ndarray, rule_probs: np.ndarray,
                          reduce: bool = True) -> Union[np.ndarray, float]:
    """
    Compute log-probability of true rules.
    
    Args:
        rules_true: True rule labels (n_samples, seq_len)
        rule_probs: Model rule probabilities (n_samples, seq_len, n_rules) - already softmax applied
        reduce: If True, return scalar mean. If False, return per-sample log-prob (n_samples,)
    
    Returns:
        Scalar mean log-probability if reduce=True, else array of per-sample values.
    """
    # Get probability of true rule for each timestep
    n_samples, seq_len = rules_true.shape
    true_probs = np.zeros((n_samples, seq_len))
    
    # Calculate minimum rule index to handle non-zero-based indexing
    rule_min = rules_true.min()
    
    for i in range(n_samples):
        for t in range(seq_len):
            true_rule = rules_true[i, t]
            # Shift to 0-based index for probability array access
            true_probs[i, t] = rule_probs[i, t, int(true_rule - rule_min)]
    
    # Compute log probability
    log_probs = np.log(true_probs + 1e-10)  # Add small epsilon to avoid log(0)
    
    if reduce:
        return float(np.mean(log_probs))
    else:
        # Mean over sequence dimension, keep sample dimension
        return np.mean(log_probs, axis=1)


def compute_dev_probabilities(dpos_long, rules_long, n_tones, rules_dpos_set,
                              pi_rules=None, mu_rho_rules=None,
                              post_dev_standards=True):
    """Marginal probability of the deviant context, P(context_t = dev | H), at
    every timestep.

    This is the structural part of the benchmark: it depends only on the sequence
    labels (deviant position and rule), not on the observations or any KF
    estimation. It marginalizes the deviant-position and rule levels, reusing
    prior_dpos_given_prev_rule (the position conditional within a rule) and
    pior_dpos_given_prev_rule_and_stds (its marginalization over rules) from
    PreProParadigm.model_RTs.

    Index convention: index t corresponds to predicting y[t]; j = t % n_tones is
    the within-trial position.

    Parameters
    ----------
    dpos_long, rules_long : np.ndarray, shape (N, T) or (T,)
        Per-tone deviant position and rule (generate_test_data's 'dpos_np' /
        'rules_np', or the equivalent columns of the dataset).
    n_tones : int
        Tones per trial (e.g. 8).
    rules_dpos_set : list
        Rule -> valid deviant positions (the data config's rules_dpos_set), used
        directly as the {rule: positions} map.
    pi_rules : np.ndarray, optional
        Assumed (n_rules, n_rules) rule-transition matrix, pi_rules[r_prev, r]. If
        None, build it from `mu_rho_rules` (2 rules only;
        = compute_fixed_pi([mu_rho_rules])).
    post_dev_standards : bool
        If True, encode that the positions following a trial's deviant are
        standards (one deviant per trial), i.e. set P(dev|H)=0 there.

    Returns
    -------
    np.ndarray of P(dev|H), same leading shape as the inputs ((N, T) or (T,)).
    """
    dpos_long = np.asarray(dpos_long)
    rules_long = np.asarray(rules_long)
    single = dpos_long.ndim == 1
    if single:
        dpos_long, rules_long = dpos_long[None], rules_long[None]

    n_rules = len(rules_dpos_set)
    valid_positions = {r: list(positions) for r, positions in enumerate(rules_dpos_set)
                       if positions is not None}
    if pi_rules is None:
        if n_rules != 2:
            raise ValueError(f"Auto-built pi_rules only supports 2 rules; got {n_rules}. "
                             "Pass pi_rules explicitly.")
        if mu_rho_rules is None:
            raise ValueError("Provide either pi_rules or mu_rho_rules.")
        rho = float(mu_rho_rules)
        pi_rules = np.array([[rho, 1 - rho], [1 - rho, rho]])

    N, T = dpos_long.shape
    p_dev = np.zeros((N, T))
    for i in range(N):
        for t in range(T):
            trial, j = divmod(t, n_tones)
            if post_dev_standards and j > int(dpos_long[i, trial * n_tones]):
                continue  # one deviant per trial: positions after it are standards
            if trial == 0:
                # first trial: no previous rule observed -> uniform rule prior
                p_dev[i, t] = sum(prior_dpos_given_prev_rule(j, r, valid_positions)
                                  for r in valid_positions) / n_rules
            else:
                prev_rule = int(rules_long[i, trial * n_tones - 1])
                p_dev[i, t] = pior_dpos_given_prev_rule_and_stds(
                    j, pi_rules, prev_rule, valid_positions)
    return p_dev[0] if single else p_dev


# =============================================================================
# Hierarchical Kalman-filter benchmark: the marginal predictive likelihood
# =============================================================================

def marginal_obs_likelihood(y, p_dev, mu_std, var_std, mu_dev, var_dev):
    """Marginal likelihood of each observation under the two-context predictive
    mixture, combining precomputed conditional per-context predictions with the
    marginal context probabilities:

        P(y_t | H) = (1 - p_dev) N(y_t; mu_std, var_std) + p_dev N(y_t; mu_dev, var_dev)

    The entry point when conditional KF per-context estimations are already
    available for a dataset (this function does not fit the KF itself). All inputs
    are elementwise-broadcastable arrays (e.g. (N, T)). var_std / var_dev are
    VARIANCES (as returned by the multi-context KF and as consumed by
    likelihood_observation), NOT standard deviations.

    Each conditional per-context likelihood (lik_std = N(y; mu_std, var_std),
    lik_dev = N(y; mu_dev, var_dev)) is multiplied by its context probability
    (1 - p_dev for standard, p_dev for deviant). Where a context probability is
    zero the corresponding term is set to zero outright, so a NaN returned by the
    KF for a context it cannot yet estimate (too few observations of that context
    so far) does not propagate into P(y|H) when that context carries zero weight.
    A NaN that survives in lik_obs therefore flags a context that is both weighted
    and not yet estimable (mask it downstream, e.g. via reduce_benchmark_loglik).

    Returns
    -------
    dict with arrays:
      lik_obs  : marginal likelihood of the realized y_t, P(y_t | H)
      p_std    : standard-context probability, 1 - p_dev
      lik_std  : conditional likelihood P(y_t | std, H)
      lik_dev  : conditional likelihood P(y_t | dev, H)
    """
    y = np.asarray(y, dtype=float)
    p_dev = np.asarray(p_dev, dtype=float)
    p_std = 1.0 - p_dev
    lik_std = likelihood_observation(y=y, mu=mu_std, sigma=var_std)
    lik_dev = likelihood_observation(y=y, mu=mu_dev, sigma=var_dev)
    with np.errstate(invalid='ignore'):
        term_std = np.where(p_std > 0, p_std * lik_std, 0.0)
        term_dev = np.where(p_dev > 0, p_dev * lik_dev, 0.0)
    return {'lik_obs': term_std + term_dev, 'p_std': p_std,
            'lik_std': lik_std, 'lik_dev': lik_dev}


def kf_per_context_predictions(y, contexts, n_iter=5, observation_noise=None):
    """Conditional per-context one-step KF predictions for ONE sequence, estimated
    under the assumption that the context labels are known.

    The costly step (EM refit at every timestep); when a dataset already carries KF
    estimations, skip it and call marginal_obs_likelihood directly.

    Returns (mu_std, var_std, mu_dev, var_dev), each length T (NaN before the KF
    can be estimated; var_* are VARIANCES). Index t predicts y[t] from y[0:t-1].
    """
    y = np.asarray(y, dtype=float)
    ctx_prob = contexts_to_probabilities(np.asarray(contexts), n_ctx=2)  # (T, 2) one-hot
    _, _, _, per_ctx_mu, per_ctx_var = kalman_online_fit_predict_multicontext(
        y, ctx_prob, n_iter=n_iter, return_per_ctx=True,
        observation_noise=observation_noise,
    )
    return per_ctx_mu[:, 0], per_ctx_var[:, 0], per_ctx_mu[:, 1], per_ctx_var[:, 1]


def compute_kf_hierarchical_benchmark(test_data, data_config_dict, n_iter=5,
                                      observation_noise=None, post_dev_standards=True,
                                      max_samples=None, pi_rules=None, verbose=True):
    """From-scratch hierarchical KF benchmark of P(y|H) for a whole test set.

    Convenience wrapper that FITS the per-context KF itself (the costly step). It
    chains compute_dev_probabilities (marginal context probabilities) +
    kf_per_context_predictions (conditional KF estimation) + marginal_obs_likelihood
    (the mixture). When conditional KF estimations already exist for a dataset,
    call compute_dev_probabilities + marginal_obs_likelihood directly instead.

    Parameters
    ----------
    test_data : dict
        generate_test_data output for a HierarchicalGM; needs 'y_np',
        'contexts_np', 'dpos_np', 'rules_np'.
    data_config_dict : dict
        to_gm_dict output, providing 'N_tones', 'rules_dpos_set', 'mu_rho_rules'.
    max_samples : int, optional
        Process only the first `max_samples` sequences (the EM refit is slow).

    Returns
    -------
    dict of (N, T) arrays: p_dev, p_std, mu_std, var_std, mu_dev, var_dev,
    lik_std, lik_dev, lik_obs (N capped by max_samples).
    """
    y = test_data['y_np']
    contexts = test_data.get('contexts_np')
    dpos_long = test_data.get('dpos_np')
    rules_long = test_data.get('rules_np')
    if contexts is None or dpos_long is None or rules_long is None:
        raise ValueError("Hierarchical KF benchmark needs 'contexts_np', 'dpos_np' and "
                         "'rules_np' in test_data (a HierarchicalGM test set).")

    n_tones = int(data_config_dict['N_tones'])
    rules_dpos_set = data_config_dict['rules_dpos_set']

    N, T = y.shape
    if max_samples is not None:
        N = min(N, int(max_samples))

    # Marginal context probabilities for all (selected) sequences at once — no KF.
    p_dev = compute_dev_probabilities(
        dpos_long[:N], rules_long[:N], n_tones, rules_dpos_set,
        pi_rules=pi_rules, mu_rho_rules=data_config_dict.get('mu_rho_rules'),
        post_dev_standards=post_dev_standards,
    )

    keys = ['p_dev', 'p_std', 'mu_std', 'var_std', 'mu_dev', 'var_dev',
            'lik_std', 'lik_dev', 'lik_obs']
    out = {k: np.full((N, T), np.nan) for k in keys}

    iterator = tqdm(range(N), desc="KF hierarchical benchmark") if verbose else range(N)
    for i in iterator:
        mu_std, var_std, mu_dev, var_dev = kf_per_context_predictions(
            y[i], contexts[i], n_iter=n_iter, observation_noise=observation_noise)
        mix = marginal_obs_likelihood(y[i], p_dev[i], mu_std, var_std, mu_dev, var_dev)
        out['p_dev'][i] = p_dev[i]
        out['p_std'][i] = mix['p_std']
        out['mu_std'][i], out['var_std'][i] = mu_std, var_std
        out['mu_dev'][i], out['var_dev'][i] = mu_dev, var_dev
        out['lik_std'][i], out['lik_dev'][i] = mix['lik_std'], mix['lik_dev']
        out['lik_obs'][i] = mix['lik_obs']
    return out


def reduce_benchmark_loglik(lik_obs, start=1, min_obs_for_em=None, reduce=True):
    """Per-sample mean log marginal likelihood of the realized observations.

    Averages log P(y_t | H) over time, aligned to the model's targets y[1:]
    (start=1). Pass min_obs_for_em to start at the KF estimation boundary instead.
    NaN timesteps (no KF estimate yet) are ignored by the nanmean.

    Returns (N,) per-sample means (reduce=True) or the (N, T-start) log values.
    """
    s = min_obs_for_em if min_obs_for_em is not None else start
    with np.errstate(divide='ignore'):
        logp = np.log(np.maximum(lik_obs[:, s:], 1e-300))
    if reduce:
        return np.nanmean(logp, axis=1)
    return logp


# =============================================================================
# Evaluation drivers
# =============================================================================

def evaluate_model(
    model: nn.Module,
    info: ModelInfo,
    test_data: Dict[str, Any],
    device: str = 'cpu',
    reduce: bool = True,
    min_obs_for_em: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single model on the test data.
    
    Args:
        model: The model to evaluate
        info: ModelInfo with model metadata
        test_data: Dictionary with test data (y, y_np, contexts_np, etc.)
        device: Device to run evaluation on
        reduce: If True, return scalar metrics (mean over dataset).
                If False, return per-sample metric distributions as arrays (n_samples,).
                For calibration: reduce=True gives global KS test with p-value,
                reduce=False gives per-sample KS statistics (for mean ± SEM reporting).
        min_obs_for_em: If provided, restrict evaluation to the same timestep window
                        as the KF benchmark, i.e. y[min_obs_for_em:] only.
                        Model pred at index t predicts y[t+1], so we slice from
                        index min_obs_for_em-1 to get predictions for y[min_obs_for_em:].
                        Pass benchmark_data.min_obs_for_em when comparing against a KF
                        benchmark so that per-sample metrics are truly comparable.
    
    Returns:
        Dictionary with all computed metrics.
        If reduce=True: values are scalars (calibration includes ks_pvalue)
        If reduce=False: values are numpy arrays of shape (n_samples,)
    """
    y = test_data['y']
    y_np = test_data['y_np']
    contexts_np = test_data.get('contexts_np')
    dpos_np = test_data.get('dpos_np')
    rules_np = test_data.get('rules_np')
    q = test_data.get('q')

    # Forward pass
    with torch.no_grad():
        if info.model_type == 'population_network':
            if q is None:
                raise ValueError("PopulationNetwork requires 'q' field (one-hot encoded cues) in test_data")
            model_output = model(y[:, :-1, :], q[:, :-1, :])
        else:
            model_output = model(y[:, :-1, :])
        predictions = get_model_predictions(model, model_output)
        mu_pred = predictions['mu_estim']
        var_pred = predictions['var_estim']
        # Extract softmax-applied probabilities and argmax predictions
        ctx_prob = predictions['ctx_prob']  # (n_samples, seq_len, n_ctx) or None
        ctx_pred = predictions['ctx_pred']  # (n_samples, seq_len) or None
        dpos_prob = predictions['dpos_prob']  # (n_samples, seq_len, n_dpos) or None
        dpos_pred = predictions['dpos_pred']  # (n_samples, seq_len) or None
        rule_prob = predictions['rule_prob']  # (n_samples, seq_len, n_rules) or None
        rule_pred = predictions['rule_pred']  # (n_samples, seq_len) or None
    
    # Target is y[1:] (predicting next observation)
    y_target = y_np[:, 1:]

    # Optionally restrict to the KF evaluation window (y[min_obs_for_em:]) so that
    # per-sample model metrics and KF metrics cover exactly the same timesteps.
    # mu_pred[:, t] predicts y[t+1], so slicing from (min_obs_for_em-1) gives
    # predictions aligned with y[min_obs_for_em:].
    if min_obs_for_em is not None:
        start = min_obs_for_em - 1  # model-prediction index → targets y[min_obs_for_em:]
        mu_pred   = mu_pred[:, start:]
        var_pred  = var_pred[:, start:]
        y_target  = y_target[:, start:]   # y_target[:,start:] == y_np[:,min_obs_for_em:]
        if ctx_prob is not None:
            ctx_prob = ctx_prob[:, start:, :]
            ctx_pred = ctx_pred[:, start:]
        # contexts_np[:,1:] aligned; slice the same way for context targets
        contexts_np_aligned = contexts_np[:, min_obs_for_em:] if contexts_np is not None else None
        if dpos_prob is not None:
            dpos_prob = dpos_prob[:, start:, :]
            dpos_pred = dpos_pred[:, start:]
        # dpos_np[:,1:] aligned; slice the same way for dpos targets
        dpos_np_aligned = dpos_np[:, min_obs_for_em:] if dpos_np is not None else None
        if rule_prob is not None:
            rule_prob = rule_prob[:, start:, :]
            rule_pred = rule_pred[:, start:]
        # rules_np[:,1:] aligned; slice the same way for rule targets
        rules_np_aligned = rules_np[:, min_obs_for_em:] if rules_np is not None else None
    else:
        contexts_np_aligned = contexts_np[:, 1:] if contexts_np is not None else None
        dpos_np_aligned = dpos_np[:, 1:] if dpos_np is not None else None
        rules_np_aligned = rules_np[:, 1:] if rules_np is not None else None
    
    # Compute observation metrics
    results = {
        'model_dir': str(info.model_dir),
        'model_type': info.model_type,
        'hidden_dim': info.hidden_dim,
        'n_ctx': info.n_ctx,
        'gm_name': info.gm_name,
        'learning_objective': info.learning_objective,
        'kappa': info.kappa,
        'bottleneck_dim': info.bottleneck_dim,
    }
    
    # MSE
    results['mse'] = compute_mse(y_target, mu_pred, reduce=reduce)
    
    # Log-likelihood
    results['obs_loglik'] = compute_log_likelihood(y_target, mu_pred, var_pred, reduce=reduce)
    
    # Calibration (KS statistic)
    if reduce:
        # Global KS test with p-value
        ks_stat, ks_pval = compute_calibration_ks(y_target, mu_pred, var_pred, reduce=True)
        results['ks_statistic'] = ks_stat
        results['ks_pvalue'] = ks_pval
    else:
        # Per-sample KS statistics (for mean ± SEM reporting)
        results['ks_statistic'] = compute_calibration_ks(y_target, mu_pred, var_pred, reduce=False)
    
    # ModuleNetwork specific metrics
    if info.model_type in ['module_network', 'population_network'] and info.n_ctx > 1 and ctx_pred is not None:
        
        # Context predictions – use pre-sliced arrays so timestep window matches y_target
        contexts_target = contexts_np_aligned
        
        # Context accuracy (using predicted labels directly)
        results['context_accuracy'] = compute_context_accuracy(contexts_target, ctx_pred, reduce=reduce)
        
        # Context log-likelihood (using probabilities)
        results['context_loglik'] = compute_context_log_likelihood(contexts_target, ctx_prob, reduce=reduce)
    
    # PopulationNetwork specific metrics
    if info.model_type == 'population_network':
        # Context log-likelihood
        if ctx_prob is not None and contexts_np_aligned is not None:
            contexts_target = contexts_np_aligned
            results['context_loglik'] = compute_context_log_likelihood(contexts_target, ctx_prob, reduce=reduce)
        
        # Dpos accuracy (using predicted labels directly)
        if dpos_pred is not None and dpos_np_aligned is not None:
            dpos_target = dpos_np_aligned
            results['dpos_accuracy'] = compute_dpos_accuracy(dpos_target, dpos_pred, reduce=reduce)
        
        # Dpos log-likelihood (using probabilities)
        if dpos_prob is not None and dpos_np_aligned is not None:
            dpos_target = dpos_np_aligned
            results['dpos_loglik'] = compute_dpos_log_prob(dpos_target, dpos_prob, reduce=reduce)
        
        # Rule accuracy (using predicted labels directly)
        if rule_pred is not None and rules_np_aligned is not None:
            rules_target = rules_np_aligned
            results['rule_accuracy'] = compute_rule_accuracy(rules_target, rule_pred, reduce=reduce)
        
        # Rule log-likelihood (using probabilities)
        if rule_prob is not None and rules_np_aligned is not None:
            rules_target = rules_np_aligned
            results['rule_loglik'] = compute_rule_log_prob(rules_target, rule_prob, reduce=reduce)
    
    return results


def evaluate_models(
    model_dirs: List[Path] = None,
    n_samples: int = 1000,
    n_tones: int = 1000,
    output_path: Optional[Path] = None,
    verbose: bool = True,
    reduce: bool = True,
) -> Union[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Evaluate multiple models on a shared test dataset.
    
    Args:
        model_dirs: List of paths to model directories
        n_samples: Number of test samples to generate
        n_tones: Sequence length
        output_path: Optional path to save results CSV (only used if reduce=True)
        verbose: Print progress information
        reduce: If True, return DataFrame with scalar metrics (default behavior).
                If False, return dict of model_name -> dict of metric distributions.
    
    Returns:
        If reduce=True: DataFrame with evaluation results for all models (scalars)
        If reduce=False: Dict mapping model names to dicts of metric arrays (n_samples,)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Validate input
    if model_dirs is None or len(model_dirs) == 0:
        raise ValueError("Must provide model_dirs")
    
    if verbose:
        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Number of models: {len(model_dirs)}")
        print(f"Test samples: {n_samples}")
        print(f"Sequence length: {n_tones}")
        print("=" * 70)
    
    # Parse model information
    model_infos = []
    for model_dir in model_dirs:
        info = ModelInfo.from_path(Path(model_dir))
        model_infos.append(info)
        if verbose:
            print(f"  Found: {info.model_type} (h={info.hidden_dim}, n_ctx={info.n_ctx})")
    
    if not model_infos:
        raise ValueError("No valid models found!")
    
    # Check all models have same N_ctx and GM (for shared test data)
    n_ctx_values = set(info.n_ctx for info in model_infos)
    gm_values = set(info.gm_name for info in model_infos)
    
    if len(n_ctx_values) > 1:
        raise ValueError(f"Models have different N_ctx values: {n_ctx_values}. "
                         "Cannot generate shared test data.")
    if len(gm_values) > 1:
        raise ValueError(f"Models have different GM types: {gm_values}. "
                         "Cannot generate shared test data.")
    
    n_ctx = model_infos[0].n_ctx
    gm_name = model_infos[0].gm_name
    
    # Generate shared test data
    # Try to use saved config from first model with a config file for accurate params
    if verbose:
        print(f"\nGenerating test data (n_ctx={n_ctx}, gm={gm_name})...")
    
    # Check if any model has a saved data config
    saved_data_config = None
    for info in model_infos:
        if info.data_config_dict is not None:
            saved_data_config = info.data_config_dict
            if verbose:
                print(f"  Using saved data config from: {info.model_dir}")
                print(f"    - N_tones: {saved_data_config.get('N_tones', 'N/A')}")
                print(f"    - mu_tau_bounds: {saved_data_config.get('mu_tau_bounds', 'N/A')}")
                print(f"    - si_stat_bounds: {saved_data_config.get('si_stat_bounds', 'N/A')}")
                print(f"    - params_testing: {saved_data_config.get('params_testing', 'N/A')}")
            break
    
    if saved_data_config is not None:
        # Use saved config (ensures same params_testing bounds, etc.)
        test_data = generate_test_data(saved_data_config, n_samples=n_samples, device=device)
    else:
        # Fall back to defaults: create a minimal gm_dict
        if verbose:
            print(f"  No saved config found, using defaults")
        # Build a minimal config dict for fallback
        fallback_config = {
            'gm_name': gm_name,
            'N_ctx': n_ctx,
            'N_samples': n_samples,
            'N_blocks': 1,
            'N_tones': n_tones,
            'mu_rho_ctx': 0.9,
            'si_rho_ctx': 0.05,
            'si_lim': 5.0,
            'si_tau': 0.5,
            'params_testing': True,
            'mu_tau_bounds': {'low': 1, 'high': 250},
            'si_stat_bounds': {'low': 0.1, 'high': 2},
            'si_r_bounds': {'low': 0.1, 'high': 2},
        }
        if n_ctx > 1:
            fallback_config.update({
                'si_d_coef': 0.05,
                'd_bounds': {'high': 4, 'low': 0.1},
                'mu_d': 2.0,
            })
        test_data = generate_test_data(fallback_config, n_samples=n_samples, device=device)
    
    if verbose:
        print(f"  Generated {n_samples} sequences of length {n_tones}")
    
    # Evaluate each model
    results = []
    results_unreduced = {}  # For reduce=False mode
    
    for info in tqdm(model_infos, desc="Evaluating models", disable=not verbose):
        model = load_model(info, device=device)
        metrics = evaluate_model(model, info, test_data, device=device, reduce=reduce)
        
        if reduce:
            results.append(metrics)
        else:
            # Use model directory name as key
            model_name = info.model_dir.name
            results_unreduced[model_name] = metrics
        
        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Return based on reduce mode
    if not reduce:
        if verbose:
            print("\n" + "=" * 70)
            print("EVALUATION COMPLETE (unreduced mode)")
            print("=" * 70)
            print(f"Returned dict with {len(results_unreduced)} models")
            print(f"Each model has metric arrays of shape ({n_samples},)")
            print("=" * 70)
        return results_unreduced
    
    # Create results DataFrame (reduce=True mode)
    df = pd.DataFrame(results)
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        # Display key metrics
        display_cols = ['model_type', 'hidden_dim', 'mse', 'obs_loglik', 'ks_statistic']
        if 'context_accuracy' in df.columns:
            display_cols.extend(['context_accuracy', 'context_loglik'])
        
        available_cols = [c for c in display_cols if c in df.columns]
        print(df[available_cols].to_string(index=False))
        print("=" * 70)
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return df


# =============================================================================
# Benchmark comparison
# =============================================================================

@dataclass
class BenchmarkData:
    """Data loaded from benchmark files."""
    y: np.ndarray  # Observations (n_samples, seq_len)
    contexts: Optional[np.ndarray]  # Context labels (n_samples, seq_len) or None
    pars: Dict[str, Any]  # Generation parameters
    mu_kf: np.ndarray  # KF predicted mean (n_samples, seq_len_eval)
    std_kf: np.ndarray  # KF predicted std (n_samples, seq_len_eval)
    mse_kf: np.ndarray  # KF MSE per sample (n_samples,)
    min_obs_for_em: int  # Number of initial observations used for EM fitting
    
    @property
    def n_samples(self) -> int:
        return self.y.shape[0]
    
    @property
    def seq_len(self) -> int:
        return self.y.shape[1]
    
    @property
    def eval_seq_len(self) -> int:
        """Length of sequence used for evaluation (after EM warmup)."""
        return self.mu_kf.shape[1]


def load_benchmark_data(
    results_path: Path,
    input_data_path: Optional[Path] = None,
) -> BenchmarkData:
    """
    Load benchmark data from pickle files.
    
    The results file is required and must contain KF predictions. If it also
    contains the input data (y, contexts, pars), the separate input_data_path
    is not needed.
    
    Args:
        results_path: Path to results pickle containing:
            - 'y': observations (n_samples, seq_len)
            - 'mu_kal_pred': KF predicted mean (n_samples, eval_seq_len)
            - 'sigma_kal_pred': KF predicted std (n_samples, eval_seq_len)
            - 'min_obs_for_em': number of initial observations for EM
            - 'perf': KF MSE per sample (n_samples,)
            - 'contexts': context labels [optional]
            - 'pars': generation parameters [optional]
        input_data_path: Optional path to separate input data pickle containing:
            - 'y': observations (n_samples, seq_len)
            - 'contexts': context labels (n_samples, seq_len) [optional]
            - 'pars': generation parameters [optional]
            If provided, these values override those in results_path.
    
    Returns:
        BenchmarkData object with all loaded data.
    """
    import pickle
    
    # Load results (required)
    with open(results_path, 'rb') as f:
        results_data = pickle.load(f)
    
    # Load separate input data if provided
    if input_data_path is not None:
        with open(input_data_path, 'rb') as f:
            input_data = pickle.load(f)
    else:
        input_data = results_data  # Use results file as input data source
    
    # Extract input fields (prefer input_data, fall back to results_data)
    y = input_data.get('y', results_data.get('y'))
    if y is None:
        raise ValueError("Could not find 'y' in benchmark files")
    
    contexts = input_data.get('contexts', results_data.get('contexts', None))
    pars = input_data.get('pars', results_data.get('pars', {}))
    
    # Extract KF results
    mu_kf = results_data['mu_kal_pred']
    std_kf = results_data['sigma_kal_pred']
    min_obs_for_em = results_data['min_obs_for_em']
    mse_kf = results_data['perf']
    
    return BenchmarkData(
        y=y,
        contexts=contexts,
        pars=pars,
        mu_kf=mu_kf,
        std_kf=std_kf,
        mse_kf=mse_kf,
        min_obs_for_em=min_obs_for_em,
    )


def assess_model_against_benchmarks(
    model: nn.Module,
    info: ModelInfo,
    benchmark_data: BenchmarkData,
    device: str = 'cpu',
    reduce: bool = True,
) -> Dict[str, Any]:
    """
    Assess a model against Kalman Filter benchmarks.
    
    This function computes how difficult the problem was for the model relative
    to the Kalman Filter (KF) benchmark. It returns ratios of model metrics to
    KF metrics, allowing comparison across different difficulty levels.
    
    The KF predictions start from min_obs_for_em (after EM warmup), so model
    predictions are aligned to the same time range for fair comparison.
    
    Args:
        model: The trained model to evaluate
        info: ModelInfo with model metadata
        benchmark_data: BenchmarkData object containing:
            - y: observations
            - contexts: context labels (optional)
            - mu_kf, std_kf: KF predictions (mean and std)
            - mse_kf: KF MSE per sample
            - min_obs_for_em: EM warmup period
        device: Device to run evaluation on
        reduce: If True, return scalar metrics (mean over dataset).
                If False, return per-sample metric distributions as arrays (n_samples,).
    
    Returns:
        Dictionary with metrics including:
        - Model metrics (mse, log_likelihood, ks_statistic)
        - KF metrics (mse_kf, log_likelihood_kf, ks_statistic_kf)
        - Ratios (mse_ratio, log_likelihood_ratio, ks_ratio)
          - mse_ratio < 1 means model is better than KF
          - log_likelihood_ratio > 1 means model is better than KF
          - ks_ratio < 1 means model is better calibrated than KF
        - Context metrics (if applicable)
    """
    y = benchmark_data.y
    contexts = benchmark_data.contexts
    mu_kf = benchmark_data.mu_kf
    std_kf = benchmark_data.std_kf
    var_kf = std_kf ** 2
    mse_kf = benchmark_data.mse_kf
    min_obs_for_em = benchmark_data.min_obs_for_em
    
    # Convert observations to tensor for model
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)
    
    # Forward pass through model
    # Model predicts y[t+1] from y[0:t], so we feed y[:, :-1]
    with torch.no_grad():
        model_output = model(y_tensor[:, :-1, :])
        predictions = get_model_predictions(model, model_output)
        mu_model_full = predictions['mu_estim']
        var_model_full = predictions['var_estim']
        # Extract softmax-applied probabilities and argmax predictions
        ctx_prob = predictions['ctx_prob']
        ctx_pred = predictions['ctx_pred']
    
    # Model predictions correspond to y[:, 1:] (predicting next observation)
    # Align with KF which starts from min_obs_for_em
    # KF predicts y[min_obs_for_em:], model predicts y[1:]
    # So model predictions for y[min_obs_for_em:] are at indices [min_obs_for_em-1:]
    
    # Target observations (what we're predicting)
    y_target_kf = y[:, min_obs_for_em:]  # KF evaluation range
    
    # Model predictions aligned to KF range
    # Model output at index t predicts y[t+1]
    # So to predict y[min_obs_for_em:], we need model output [min_obs_for_em-1:]
    model_start_idx = min_obs_for_em - 1
    mu_model = mu_model_full[:, model_start_idx:]
    var_model = var_model_full[:, model_start_idx:]
    
    # Ensure shapes match
    eval_len = min(y_target_kf.shape[1], mu_model.shape[1], mu_kf.shape[1])
    y_target_kf = y_target_kf[:, :eval_len]
    mu_model = mu_model[:, :eval_len]
    var_model = var_model[:, :eval_len]
    mu_kf = mu_kf[:, :eval_len]
    var_kf = var_kf[:, :eval_len]
    
    # Initialize results
    results = {
        'model_dir': str(info.model_dir),
        'model_type': info.model_type,
        'hidden_dim': info.hidden_dim,
        'n_ctx': info.n_ctx,
        'gm_name': info.gm_name,
        'min_obs_for_em': min_obs_for_em,
        'eval_seq_len': eval_len,
    }
    
    # --- Compute Model Metrics ---
    mse_model = compute_mse(y_target_kf, mu_model, reduce=reduce)
    ll_model = compute_log_likelihood(y_target_kf, mu_model, var_model, reduce=reduce)
    
    # --- Compute KF Metrics ---
    mse_kf_computed = compute_mse(y_target_kf, mu_kf, reduce=reduce)
    ll_kf = compute_log_likelihood(y_target_kf, mu_kf, var_kf, reduce=reduce)
    
    # --- Calibration ---
    if reduce:
        ks_model, ks_pval_model = compute_calibration_ks(y_target_kf, mu_model, var_model, reduce=True)
        ks_kf, ks_pval_kf = compute_calibration_ks(y_target_kf, mu_kf, var_kf, reduce=True)
        
        results['mse'] = mse_model
        results['obs_loglik'] = ll_model
        results['ks_statistic'] = ks_model
        results['ks_pvalue'] = ks_pval_model
        
        results['mse_kf'] = mse_kf_computed
        results['obs_loglik_kf'] = ll_kf
        results['ks_statistic_kf'] = ks_kf
        results['ks_pvalue_kf'] = ks_pval_kf
        
        # Compute ratios (model / KF)
        # mse_ratio < 1 means model is better
        results['mse_ratio'] = mse_model / (mse_kf_computed + 1e-10)
        # ll_ratio > 1 means model is better (higher log-likelihood)
        # Use difference for log-likelihood since ratio of logs is less interpretable
        results['obs_loglik_diff'] = ll_model - ll_kf
        # ks_ratio < 1 means model is better calibrated
        results['ks_ratio'] = ks_model / (ks_kf + 1e-10)
    else:
        ks_model = compute_calibration_ks(y_target_kf, mu_model, var_model, reduce=False)
        ks_kf = compute_calibration_ks(y_target_kf, mu_kf, var_kf, reduce=False)
        
        results['mse'] = mse_model
        results['obs_loglik'] = ll_model
        results['ks_statistic'] = ks_model
        
        results['mse_kf'] = mse_kf_computed
        results['obs_loglik_kf'] = ll_kf
        results['ks_statistic_kf'] = ks_kf
        
        # Compute per-sample ratios
        results['mse_ratio'] = mse_model / (mse_kf_computed + 1e-10)
        results['obs_loglik_diff'] = ll_model - ll_kf
        results['ks_ratio'] = ks_model / (ks_kf + 1e-10)
    
    # --- Context Metrics (if applicable) ---
    if info.model_type in ['module_network', 'population_network'] and info.n_ctx > 1 and ctx_prob is not None and contexts is not None:
        results['learning_objective'] = info.learning_objective
        results['kappa'] = info.kappa
        results['bottleneck_dim'] = info.bottleneck_dim
        
        # Context probabilities and predictions aligned to evaluation range
        ctx_prob_aligned = ctx_prob[:, model_start_idx:model_start_idx + eval_len]
        ctx_pred_aligned = ctx_pred[:, model_start_idx:model_start_idx + eval_len]
        
        # True contexts for evaluation range
        # Context at time t corresponds to observation y[t]
        contexts_target = contexts[:, min_obs_for_em:min_obs_for_em + eval_len]
        
        # Context accuracy (using predicted labels directly)
        results['context_accuracy'] = compute_context_accuracy(contexts_target, ctx_pred_aligned, reduce=reduce)
        
        # Context log-likelihood (using probabilities)
        results['context_loglik'] = compute_context_log_likelihood(contexts_target, ctx_prob_aligned, reduce=reduce)
    
    return results


def assess_models_against_benchmarks(
    model_dirs: List[Path],
    results_path: Path,
    input_data_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose: bool = True,
    reduce: bool = True,
) -> Union[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Assess multiple models against Kalman Filter benchmarks.
    
    Args:
        model_dirs: List of paths to model directories
        results_path: Path to benchmark results pickle (must contain y and KF predictions)
        input_data_path: Optional path to separate input data pickle. If None,
            input data (y, contexts, pars) is loaded from results_path.
        output_path: Optional path to save results CSV (only used if reduce=True)
        verbose: Print progress information
        reduce: If True, return DataFrame with scalar metrics.
                If False, return dict of model_name -> dict of metric distributions.
    
    Returns:
        If reduce=True: DataFrame with evaluation results for all models
        If reduce=False: Dict mapping model names to dicts of metric arrays
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print("\n" + "=" * 70)
        print("MODEL VS BENCHMARK EVALUATION")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Number of models: {len(model_dirs)}")
        print(f"Benchmark results: {results_path}")
        if input_data_path:
            print(f"Benchmark input (separate): {input_data_path}")
        print("=" * 70)
    
    # Load benchmark data
    if verbose:
        print("\nLoading benchmark data...")
    benchmark_data = load_benchmark_data(results_path, input_data_path)
    if verbose:
        print(f"  Loaded {benchmark_data.n_samples} samples")
        print(f"  Sequence length: {benchmark_data.seq_len}")
        print(f"  Evaluation length (after EM): {benchmark_data.eval_seq_len}")
        print(f"  min_obs_for_em: {benchmark_data.min_obs_for_em}")
    
    # Parse model information
    model_infos = []
    for model_dir in model_dirs:
        info = ModelInfo.from_path(Path(model_dir))
        model_infos.append(info)
        if verbose:
            print(f"  Found: {info.model_type} (h={info.hidden_dim}, n_ctx={info.n_ctx})")
    
    if not model_infos:
        raise ValueError("No valid models found!")
    
    # Evaluate each model
    results = []
    results_unreduced = {}
    
    for info in tqdm(model_infos, desc="Evaluating models vs benchmarks", disable=not verbose):
        model = load_model(info, device=device)
        metrics = assess_model_against_benchmarks(
            model, info, benchmark_data, device=device, reduce=reduce
        )
        
        if reduce:
            results.append(metrics)
        else:
            model_name = info.model_dir.name
            results_unreduced[model_name] = metrics
        
        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Return based on reduce mode
    if not reduce:
        if verbose:
            print("\n" + "=" * 70)
            print("EVALUATION COMPLETE (unreduced mode)")
            print("=" * 70)
            print(f"Returned dict with {len(results_unreduced)} models")
            print(f"Each model has metric arrays of shape ({benchmark_data.n_samples},)")
            print("=" * 70)
        return results_unreduced
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK COMPARISON RESULTS")
        print("=" * 70)
        
        display_cols = ['model_type', 'hidden_dim', 'mse', 'mse_kf', 'mse_ratio', 
                        'obs_loglik', 'obs_loglik_kf', 'ks_statistic', 'ks_ratio']
        if 'context_accuracy' in df.columns:
            display_cols.append('context_accuracy')
        
        available_cols = [c for c in display_cols if c in df.columns]
        print(df[available_cols].to_string(index=False))
        print("=" * 70)
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return df


# =============================================================================
# Evaluation orchestration over a list of models
# =============================================================================
#
# Resolving model folders, building one test set per model (or one shared set),
# and collecting the per-model metric dictionaries the figures are drawn from.
# Previously the top half of model_analysis.py.

def resolve_model_dir(model_dir, models_dir=None):
    """Resolve one entry of a model selection.

    Accepts an absolute path, a path relative to the working directory, or the
    bare name of a folder inside `models_dir`.
    """
    model_dir = Path(model_dir)
    if model_dir.is_absolute() or model_dir.exists() or models_dir is None:
        return model_dir
    return Path(models_dir) / model_dir


def load_models_info(models_dir=None, model_dirs=None, verbose=True, strict=False):
    """Load a ModelInfo for each model to compare.

    Pass `models_dir` to take every model sub-folder it contains, and/or
    `model_dirs` to select specific models — either full paths or bare folder
    names looked up inside `models_dir`. A model folder is any directory holding
    a .pth weights file; anything else in `models_dir` is skipped silently (an
    explicitly selected folder is reported instead).

    Folders whose config cannot be read (no config.json and an unrecognised
    directory name, or a config.json written by another version of config_v2)
    are reported and skipped — pass strict=True to get the traceback instead.
    """
    if model_dirs is not None:
        candidates = [resolve_model_dir(d, models_dir) for d in model_dirs]
    elif models_dir is not None:
        candidates = [Path(models_dir) / name for name in sorted(os.listdir(models_dir))]
    else:
        raise ValueError("Pass either models_dir (take every model in it) or model_dirs.")

    models_info = []
    for model_dir in candidates:
        if not model_dir.is_dir() or not list(model_dir.glob('*.pth')):
            if model_dirs is not None:  # explicitly asked for: say why it is dropped
                print(f"  ✗ {model_dir}: skipped (not a model folder with a .pth file)")
            continue
        try:
            models_info.append(ModelInfo.from_path(model_dir))
        except Exception as err:
            if strict:
                raise
            print(f"  ✗ {model_dir.name}: skipped ({err})")
            continue
        if verbose:
            print(f"  ✓ {model_dir.name}")

    if not models_info:
        raise FileNotFoundError(
            f"No readable model folder in {model_dirs if model_dirs is not None else models_dir}")
    return models_info


def pinned_si_r(data_config_dict):
    """The observation noise a config pins, or None when sigma_r is sampled.

    DataConfig.to_gm_dict() encodes si_r_fixed as degenerate si_r_bounds
    (low == high), which is what the GM then draws sigma_r from, so a pinned
    sigma_r shows up here and under no other key.
    """
    bounds = (data_config_dict or {}).get('si_r_bounds') or {}
    low, high = bounds.get('low'), bounds.get('high')
    return low if low is not None and low == high else None


def check_shared_data_config(models_info, shared=False):
    """Report the models that were not all trained on the same data config.

    With `shared`, one test set (the first model's) is used for every model, so
    a mismatch matters twice over: a structural difference (a different number
    of cues, say) makes the other models fail in their forward pass with an
    opaque shape error, and a difference in the generative parameters — a fixed
    sigma_r above all, see pinned_si_r — silently scores them off their training
    regime. Without it every model gets its own test set, and the differences
    are only listed for the record.
    """
    reference = models_info[0].data_config_dict or {}
    ref_si_r = pinned_si_r(reference)
    for info in models_info[1:]:
        other = info.data_config_dict or {}
        differing = [k for k in set(reference) | set(other)
                     if k != 'N_samples' and repr(reference.get(k)) != repr(other.get(k))]
        if not differing:
            continue

        mark = '!' if shared else '-'
        print(f"  {mark} {info.model_dir.name}: data config differs from "
              f"{models_info[0].model_dir.name} on {sorted(differing)}")

        si_r = pinned_si_r(other)
        if 'si_r_bounds' in differing and si_r is not None and ref_si_r is not None:
            if shared:
                print(f"      trained with sigma_r fixed at {si_r:g}, evaluated at "
                      f"{ref_si_r:g} (test set follows the first model)")
            else:
                print(f"      sigma_r fixed at {si_r:g} (vs {ref_si_r:g}); "
                      f"evaluated on its own test set")


def generate_test_set(data_config_dict, n_samples, device='cpu', seed=None):
    """One freshly generated test set.

    With `seed` the generation becomes reproducible: the RNG is reset and the GM
    is forced to run sequentially, because the worker pool it uses otherwise
    (max_cores > 1) forks streams of its own. Two configs that then differ only
    in sigma_r give the same latents — rules, contexts, cues, tau — and differ
    only in the observation noise, which is what makes per-model test sets
    comparable to each other.
    """
    if seed is not None:
        data_config_dict = {**data_config_dict, 'max_cores': 1}
        np.random.seed(seed)
    return generate_test_data(data_config_dict, n_samples=n_samples, device=device)


def build_test_sets(models_info, n_samples, shared=False, benchmark_data=None,
                    device='cpu', seed=None):
    """Test set of every model, as {model folder name: test_data}.

    By default each model gets a set generated from its own training data
    config, so a model trained at a fixed sigma_r is scored at that sigma_r;
    models whose config is identical share one generated set, which makes this
    mode collapse to the shared one when the configs agree.

    `shared` instead generates a single set from the first model's config and
    hands it to all of them — worth it only when the models really were trained
    on the same data and the point is to compare them on the very sequences.

    `benchmark_data` overrides both: the sequences the KF was run on are reused
    for every model, so model and KF metrics are computed on the same data.
    """
    names = [info.model_dir.name for info in models_info]

    if benchmark_data is not None:
        test_data = {
            'y': torch.tensor(benchmark_data.y, dtype=torch.float32).unsqueeze(-1).to(device),
            'y_np': benchmark_data.y,
            'contexts_np': benchmark_data.contexts,
            'pars': benchmark_data.pars,
        }
        return {name: test_data for name in names}

    if shared:
        test_data = generate_test_set(models_info[0].data_config_dict, n_samples,
                                      device=device, seed=seed)
        return {name: test_data for name in names}

    test_sets, by_config = {}, {}
    for info in tqdm(models_info, desc="Generating test data"):
        key = repr(info.data_config_dict)
        if key not in by_config:
            by_config[key] = generate_test_set(info.data_config_dict, n_samples,
                                               device=device, seed=seed)
        test_sets[info.model_dir.name] = by_config[key]
    return test_sets


def evaluate_all_models(models_info, test_sets, min_obs_for_em=None, device='cpu'):
    """Evaluate every model, keeping metrics as per-sample distributions.

    `test_sets` maps a model folder name to the test data that model is
    evaluated on (see build_test_sets); a shared test set is simply the same
    object under every key.

    Returns {model folder name: metrics dict}. `min_obs_for_em` restricts the
    evaluation to the KF window (y[min_obs_for_em:]) so per-sample model metrics
    stay comparable with the KF ones.
    """
    results = {}
    for info in tqdm(models_info, desc="Evaluating models"):
        model = load_model(info, device=device)
        results[info.model_dir.name] = evaluate_model(
            model, info, test_sets[info.model_dir.name], device=device, reduce=False,
            min_obs_for_em=min_obs_for_em,
        )
        del model
    return results


def compute_kf_metrics(benchmark_data):
    """Per-sample KF metrics, keyed like the model metrics (see KF_METRICS)."""
    var_kf = benchmark_data.std_kf ** 2
    y_target = benchmark_data.y[:, benchmark_data.min_obs_for_em:]
    return {
        'mse': compute_mse(y_target, benchmark_data.mu_kf, reduce=False),
        'obs_loglik': compute_log_likelihood(
            y_target, benchmark_data.mu_kf, var_kf, reduce=False),
        'ks_statistic': compute_calibration_ks(
            y_target, benchmark_data.mu_kf, var_kf, reduce=False),
    }


def model_predictions(info, test_data, device='cpu'):
    """Forward pass of one model on the test set.

    Returns (y_target, mu_pred, var_pred), all aligned on y[1:] (the model
    predicts the next observation).
    """
    model = load_model(info, device=device)
    y = test_data['y']
    with torch.no_grad():
        if info.model_type == 'population_network':
            q = test_data.get('q')
            if q is None:
                raise ValueError("PopulationNetwork requires the one-hot cues 'q' in test_data "
                                 "(not available when using benchmark data).")
            model_output = model(y[:, :-1, :], q[:, :-1, :])
        else:
            model_output = model(y[:, :-1, :])
        preds = get_model_predictions(model, model_output)
    del model
    return test_data['y_np'][:, 1:], preds['mu_estim'], preds['var_estim']

if __name__ == '__main__':
    pass
