"""Model-pair correlations on the experimental sequences, one module at a time.

Where run_module_correlations.py correlates the modules of one model with each
other, this script compares several models with each other: for every sequence,
every module and each of the four quantities the forward pass yields for that
module, the Pearson correlation between the time courses of the quantity in
every pair of models, plus the mean over the pairs.

The four quantities, read from the activation CSVs of run_exp_trials_pipeline.py:

  posterior_norm    L2 norm of the posterior hidden state    '<module>_norm'
  posterior_deriv   its temporal derivative                  '<module>_deriv'
  prior_norm        L2 norm of the prior hidden state        'prior_<module>_norm'
  prior_deriv       its temporal derivative                  'prior_<module>_deriv'

The prior columns are only written when run_exp_trials_pipeline.py is run with
INCLUDE_PRIOR = True.

Three CSVs are written:
  - one row per (sequence, module, quantity): the sequence's generative
    parameters, one column per pair of models, named after the short model names
    (e.g. 'sir0.02 vs sirvar'), and 'mean_over_pairs'
  - per condition (lim_std, d, tau_std): the same score columns averaged over the
    sequences of each condition, one row per (condition, module, quantity)
  - over all conditions: the same score columns averaged over all sequences, one
    row per (module, quantity)
The two summaries also give the number of sequences averaged ('n_sequences').

With MEAN_ABS = True, every mean (over pairs, then over sequences) is taken over
absolute correlations, so that anti-correlated pairs or sequences do not cancel
out positively correlated ones, and the three file names end in '_meanabs'. The
per-pair correlations of the first CSV are not averages and stay signed.
"""

from itertools import combinations

import numpy as np
import pandas as pd

import analysis_config as cfg
import exp_sequence_analysis as exp
from analysis_config import MODULE_NAMES
from analysis_core import find_trial_files, sequence_csv_path


def quantity_column(quantity, module):
    """Column of an activations CSV holding one quantity of one module.

    The posterior columns are unprefixed ('obs_norm') and the prior ones carry a
    'prior_' prefix ('prior_obs_norm'), as written by
    exp_sequence_analysis.build_activations_frame.
    """
    hidden_state, measure = quantity.split('_')     # e.g. 'prior', 'norm'
    if hidden_state == 'prior':
        return f'prior_{module}_{measure}'
    return f'{module}_{measure}'


def short_model_name(model_name):
    """'sir<value>' for a model trained with a fixed sigma_r, 'sirvar' otherwise.

    The value is read from the 'fixedsir<value>' part of the model name, e.g.
    'population_network_all_bn8_trainh0_fixedsir0.02_epochs300_lr0.002' -> 'sir0.02'
    and 'population_network_all_bn8_trainh0_epochs300_lr0.002' -> 'sirvar'.
    """
    for part in model_name.split('_'):
        if part.startswith('fixedsir'):
            return 'sir' + part[len('fixedsir'):]
    return 'sirvar'


def pair_column(model_a, model_b):
    """Output column holding the correlation between two models, e.g. 'sir0.02 vs sirvar'."""
    return f'{short_model_name(model_a)} vs {short_model_name(model_b)}'


def correlate_models_on_sequence(sequence, model_names, output_root, quantities, mean_abs):
    """Output rows of one sequence: one per (module, quantity).

    For each module and quantity, the time course of every model is correlated
    with the time course of every other model (Pearson), and the mean over the
    pairs is added: of the absolute correlations if `mean_abs`, of the signed
    ones otherwise. Timesteps where a model has no value (the last row of the
    derivatives, padded with NaN in the CSVs) are left out of every pair.
    """
    frames = {}
    for model_name in model_names:
        frames[model_name] = pd.read_csv(
            sequence_csv_path(output_root, model_name, 'activations', sequence))
    # The generative parameters are constant within a sequence and the same for every model.
    first_frame = frames[model_names[0]]

    rows = []
    for module in MODULE_NAMES:
        for quantity in quantities:
            column = quantity_column(quantity, module)
            # One row per model, one column per timestep.
            values = np.stack([frames[model_name][column].to_numpy() for model_name in model_names])
            defined = ~np.isnan(values).any(axis=0)
            time_courses = {}
            for i, model_name in enumerate(model_names):
                time_courses[model_name] = values[i, defined]
            # Keys are '<model a>_<model b>', for every pair in the order of model_names.
            pair_correlations = exp.compute_pairwise_module_correlations(time_courses)

            row = {
                'sequence_name': sequence,
                'lim_std': first_frame['lim_std'].iloc[0],
                'd': first_frame['d'].iloc[0],
                'tau_std': first_frame['tau_std'].iloc[0],
                'module': module,
                'quantity': quantity,
            }
            for model_a, model_b in combinations(model_names, 2):
                row[pair_column(model_a, model_b)] = pair_correlations[f'{model_a}_{model_b}']
            aggregates = exp.aggregate_scores(pair_correlations)
            if mean_abs:
                row['mean_over_pairs'] = aggregates['mean_abs']
            else:
                row['mean_over_pairs'] = aggregates['mean']
            rows.append(row)
    return rows


def summarize_scores(scores, group_columns, score_columns, mean_abs):
    """Mean of every score over the sequences of each group, with the number of sequences.

    With `mean_abs`, the means are taken over the absolute scores. Groups keep
    the order in which they first appear in `scores`. NaN scores are left out of
    the means.
    """
    table = scores[group_columns + score_columns].copy()
    if mean_abs:
        table[score_columns] = table[score_columns].abs()
    grouped = table.groupby(group_columns, sort=False)
    summary = grouped[score_columns].mean()
    summary.insert(0, 'n_sequences', grouped.size())
    return summary.reset_index()


if __name__ == '__main__':

    # ------------------------- SETTINGS (edit these) -------------------------
    # Models to compare, every pair of them is correlated. Each must own an
    # <output-root>/<model>/activations folder filled by run_exp_trials_pipeline.py,
    # run with INCLUDE_PRIOR = True for the prior quantities.
    MODEL_NAMES = cfg.EVALUATION_MODELS
    OUTPUT_ROOT = cfg.EXP_SEQ_OUTPUT_ROOT
    # Every sequence file in TRIALS_PATH is used.
    TRIALS_PATH = cfg.TRIALS_PATH

    # Remove the two prior quantities to run on activation CSVs written without
    # INCLUDE_PRIOR.
    QUANTITIES = ['posterior_norm', 'posterior_deriv', 'prior_norm', 'prior_deriv']

    # True: every mean (over pairs, then over sequences) is taken over absolute
    # correlations, and the CSV names end in '_meanabs'. False: plain means.
    MEAN_ABS = True

    # The three CSVs are written to OUTPUT_DIR, their names starting with
    # OUTPUT_NAME. They are overwritten on every run: change OUTPUT_NAME when
    # comparing another list of models.
    OUTPUT_DIR = OUTPUT_ROOT / 'model_pair_correlations'
    OUTPUT_NAME = 'model_pair_correlations'
    # -------------------------------------------------------------------------

    if MEAN_ABS:
        suffix = '_meanabs'
    else:
        suffix = ''
    # One row per (sequence, module, quantity)
    sequences_csv = OUTPUT_DIR / f'{OUTPUT_NAME}{suffix}.csv'
    # Mean over the sequences of each condition: one row per (condition, module, quantity)
    conditions_csv = OUTPUT_DIR / f'{OUTPUT_NAME}_by_condition{suffix}.csv'
    # Mean over all sequences: one row per (module, quantity)
    all_conditions_csv = OUTPUT_DIR / f'{OUTPUT_NAME}_all_conditions{suffix}.csv'

    sequences = []
    for trial_file in find_trial_files(TRIALS_PATH):
        sequences.append(trial_file.stem)
    n_pairs = len(list(combinations(MODEL_NAMES, 2)))
    print(f"{len(sequences)} sequences, {len(MODEL_NAMES)} models, {n_pairs} pairs of models")

    # The pair columns are named after the short model names, so these must all
    # differ, otherwise the columns of different pairs would overwrite each other.
    short_names = []
    for model_name in MODEL_NAMES:
        short_names.append(short_model_name(model_name))
        print(f"  {short_model_name(model_name)}: {model_name}")
    if len(set(short_names)) < len(short_names):
        raise ValueError(f"Several models share the same short name: {short_names}")

    rows = []
    for i, sequence in enumerate(sequences, start=1):
        rows.extend(correlate_models_on_sequence(sequence, MODEL_NAMES, OUTPUT_ROOT,
                                                 QUANTITIES, MEAN_ABS))
        if i % 100 == 0 or i == len(sequences):
            print(f"  processed {i}/{len(sequences)} sequences")
    scores = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(sequences_csv, index=False)
    print(f"{len(scores)} rows -> {sequences_csv}")

    # Scores to summarize: every pair of models, then the mean over pairs.
    score_columns = []
    for model_a, model_b in combinations(MODEL_NAMES, 2):
        score_columns.append(pair_column(model_a, model_b))
    score_columns.append('mean_over_pairs')

    by_condition = summarize_scores(scores, ['lim_std', 'd', 'tau_std', 'module', 'quantity'],
                                    score_columns, MEAN_ABS)
    # Conditions in increasing order; the stable sort keeps the module and quantity
    # order within each condition.
    by_condition = by_condition.sort_values(['lim_std', 'd', 'tau_std'], kind='stable')
    by_condition.to_csv(conditions_csv, index=False)
    print(f"{len(by_condition)} rows -> {conditions_csv}")

    all_conditions = summarize_scores(scores, ['module', 'quantity'], score_columns, MEAN_ABS)
    all_conditions.to_csv(all_conditions_csv, index=False)
    print(f"{len(all_conditions)} rows -> {all_conditions_csv}")

    # A first look at the mean over pairs, over all conditions.
    print(all_conditions[['module', 'quantity', 'n_sequences', 'mean_over_pairs']])
