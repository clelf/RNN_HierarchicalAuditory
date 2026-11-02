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

With SEQUENTIAL_PRESENTATION = True, the models are evaluated once per pool of
four sequences instead of once per sequence: the pools of
analysis_core.make_sequence_pools, one sequence per tau_std value, which
run_exp_trials_pipeline.py (with its own SEQUENTIAL_PRESENTATION = True) runs one
after the other, each sequence starting from the hidden states the previous one
ended with. The time courses then run over the four sequences of a pool. Two CSVs
are written, their names containing '_pools_seed<POOL_SEED>': one row per (pool,
module, quantity), with the tau_std of the pool's sequences in presentation order
('tau_std_order'), and the mean over all pools ('n_pools'). A pool mixes four
conditions, so there is no per-condition summary.

With PLOT_CORRELATIONS = True, the correlations are also plotted, in one figure
per quantity, saved next to the CSVs and named '<name start>_<quantity>.png',
after the CSVs but without their '_meanabs': four rows, one per module, and one
column per pair of models. A panel scatters the two models' time courses of that
quantity against each other, one point per timestep of the first
FIGURE_SEQUENCES sequences (or pools), so that what the correlation of the CSVs
measures can be seen, and is annotated with the correlation of the points
plotted. With PLOT_BY_TRIAL_POS = True, every point takes the colour of the
position its timestep holds within its trial.

With SHOW_ALL_SEQ = True, a point is instead one timestep averaged over every
sequence (or pool) of the set, the aggregation of
exp_sequence_analysis.load_timestep_means, and the figures are named
'<name start>_<quantity>_allseq.png'. With PLOT_BY_TRIAL_POS, each position's
colour then shades from the early to the late trials, as
timestep_matrix__by_position_gradient.png does.

With COMPUTE_SCORES = False, the scores CSV of an earlier run is read back and
only the correlations are plotted, which saves recomputing them.

Every run that computes the scores also writes a '_settings.txt' file next to its
CSVs, with the same name start: TRIALS_PATH, the full names of the models
compared (the short names of the pair columns do not identify them), and the
other settings of the run.
"""

from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd

import analysis_config as cfg
import exp_sequence_analysis as exp
import plots
from analysis_config import MODULE_NAMES
from analysis_core import (
    find_trial_files,
    make_sequence_pools,
    pool_name,
    sequence_csv_path,
    short_model_name,
)




def quantity_column_name(quantity, module):
    """Name of the activations CSV column holding one quantity of one module.

    The posterior columns are unprefixed ('obs_norm') and the prior ones carry a
    'prior_' prefix ('prior_obs_norm'), as written by
    exp_sequence_analysis.build_activations_frame.
    """
    hidden_state, measure = quantity.split('_')     # e.g. 'prior', 'norm'
    if hidden_state == 'prior':
        return f'prior_{module}_{measure}'
    return f'{module}_{measure}'


def model_pair_column(model_a, model_b):
    """Name of the output column holding the correlation between two models, e.g. 'sir0.02 vs sirvar'."""
    return f'{short_model_name(model_a)} vs {short_model_name(model_b)}'


def figure_png_name(csv_name_start, quantity, all_sequences):
    """Name of the PNG file holding the correlations between models of one quantity.

    The same name start as the CSVs of the run, so that a figure sits next to
    the scores it was plotted from, then the quantity it shows, and '_allseq'
    when its points are means over every sequence rather than the timesteps of
    the first few. The '_meanabs' the CSV names end with is left out: MEAN_ABS
    is a choice of summary over the pairs and over the sequences, and the
    figures show neither.
    """
    if all_sequences:
        return f'{csv_name_start}_{quantity}_allseq.png'
    return f'{csv_name_start}_{quantity}.png'


def correlate_model_activation_quantities(dfs, model_names, quantities, mean_abs, row_start):
    """Output rows of one sequence or pool: one per (module, quantity).

    `dfs` holds the activations CSV of every model, read as a DataFrame. For
    each module and quantity, the time course of every model is correlated with
    the time course of every other model (Pearson), and the mean over the pairs
    is added: of the absolute correlations if `mean_abs`, of the signed ones
    otherwise. Timesteps where a model has no value (the first row of the
    derivatives, padded with NaN in the CSVs) are left out of every pair. Every
    row starts with the columns of `row_start`, which identify the sequence or pool.
    """
    rows = []
    for module in MODULE_NAMES:
        for quantity in quantities:
            column = quantity_column_name(quantity, module)
            # One row per model, one column per timestep.
            values = np.stack([dfs[model_name][column].to_numpy() for model_name in model_names])
            defined = ~np.isnan(values).any(axis=0)
            time_courses = {}
            for i, model_name in enumerate(model_names):
                time_courses[model_name] = values[i, defined]
            # Keys are '<model a>_<model b>', for every pair in the order of model_names.
            pair_correlations = exp.compute_pairwise_module_correlations(time_courses)

            row = dict(row_start)
            row['module'] = module
            row['quantity'] = quantity
            for model_a, model_b in combinations(model_names, 2):
                row[model_pair_column(model_a, model_b)] = pair_correlations[f'{model_a}_{model_b}']
            aggregates = exp.aggregate_scores(pair_correlations)
            if mean_abs:
                row['mean_over_pairs'] = aggregates['mean_abs']
            else:
                row['mean_over_pairs'] = aggregates['mean']
            rows.append(row)
    return rows


def get_trial_positions(df):
    """Within-trial position of every timestep of an activations CSV, 0-based.

    The timesteps of one trial are consecutive rows sharing the same 'trial_n',
    which restarts at 0 for every sequence of a pool CSV, so the trials are read
    as runs of equal 'trial_n' rather than grouped by its value. Position 0 is
    the first timestep of the trial, the convention the deviant positions dpos
    are expressed in (see analysis_config.PERIOD).
    """
    trials = (df['trial_n'] != df['trial_n'].shift()).cumsum()
    return df.groupby(trials).cumcount().to_numpy()


def get_sequence_activation_quantities(dfs, model_names, quantities, sequence_or_pool):
    """Activation quantities of one sequence or pool: one row per (module, quantity, timestep).

    The values the correlations are computed from, laid out for the figure: one
    column per model, named after its short name, so that a pair of models is a
    pair of columns to scatter against each other, plus the timestep the row
    comes from ('timestep') and its within-trial position ('trial_position'),
    both the same for every model since they are all given the same input.
    Timesteps where a model has no value (the first row of the derivatives,
    padded with NaN in the CSVs) are left out, as in
    correlate_model_activation_quantities, so that every row holds a value for
    every model.
    """
    trial_positions = get_trial_positions(dfs[model_names[0]])
    timesteps = dfs[model_names[0]]['timestep'].to_numpy()
    tables = []
    for module in MODULE_NAMES:
        for quantity in quantities:
            column = quantity_column_name(quantity, module)
            # One row per model, one column per timestep.
            values = np.stack([dfs[model_name][column].to_numpy() for model_name in model_names])
            defined = ~np.isnan(values).any(axis=0)
            table = {
                'sequence_or_pool': sequence_or_pool,
                'module': module,
                'quantity': quantity,
                'timestep': timesteps[defined],
                'trial_position': trial_positions[defined],
            }
            for i, model_name in enumerate(model_names):
                table[short_model_name(model_name)] = values[i, defined]
            tables.append(pd.DataFrame(table))
    return pd.concat(tables, ignore_index=True)


def add_to_timestep_sums(running_sums, table, model_names):
    """Running sum, over the sequences read so far, of their activation quantities.

    `table` is what get_sequence_activation_quantities returns for one sequence
    or pool, and `running_sums` what this function returned for the previous one
    (None before the first). The sums are indexed by (module, quantity,
    timestep) and hold every model's values summed over the sequences, the
    within-trial position summed the same way, and 'n_sequences', the number of
    sequences a (module, quantity, timestep) was seen in: a sequence shorter
    than the others contributes to the timesteps it reaches and to no other.
    """
    summed_columns = ['trial_position']
    for model_name in model_names:
        summed_columns.append(short_model_name(model_name))
    summed = table.set_index(['module', 'quantity', 'timestep'])[summed_columns].copy()
    summed['n_sequences'] = 1
    if running_sums is None:
        return summed
    return running_sums.add(summed, fill_value=0)


def get_timestep_mean_activation_quantities(running_sums, model_names):
    """Activation quantities averaged over the sequences: one row per (module, quantity, timestep).

    The columns the figure needs, as get_sequence_activation_quantities returns
    them for a single sequence, so that it is plotted from either. The
    within-trial position is averaged like the values and rounded back to an
    integer: it is a property of the timestep, the same in every sequence, so
    its mean recovers it (as exp_sequence_analysis.load_timestep_means does).
    """
    averaged_columns = ['trial_position']
    for model_name in model_names:
        averaged_columns.append(short_model_name(model_name))
    means = running_sums[averaged_columns].div(running_sums['n_sequences'], axis=0)
    means['trial_position'] = means['trial_position'].round().astype(int)
    return means.reset_index()


def read_sequence_dfs(sequence, model_names, output_root):
    """Activations CSV of one sequence, read as a DataFrame per model."""
    dfs = {}
    for model_name in model_names:
        dfs[model_name] = pd.read_csv(
            sequence_csv_path(output_root, model_name, 'activations', sequence))
    return dfs


def read_pool_dfs(pool, pool_files, model_names, output_root):
    """Pool activations CSV of one pool, read as a DataFrame per model.

    `pool` is the name of the pool (analysis_core.pool_name) and `pool_files` its
    sequence files, in presentation order. Comparing models only makes sense when
    they were all given the same input, so a ValueError is raised when the
    sequences of a model's CSV are not those of `pool_files`, in that order, e.g.
    for CSVs written with another POOL_SEED or another set of sequence files.
    """
    expected_sequences = []
    for trial_file in pool_files:
        expected_sequences.append(trial_file.stem)

    dfs = {}
    for model_name in model_names:
        df = pd.read_csv(sequence_csv_path(output_root, model_name, 'pool_activations', pool))
        # unique() keeps the order in which the sequences appear.
        written_sequences = list(df['sequence_name'].unique())
        if written_sequences != expected_sequences:
            raise ValueError(f"{pool} of {model_name} holds the sequences {written_sequences}, "
                             f"not {expected_sequences}: rewrite the pool CSVs with "
                             f"run_exp_trials_pipeline.py and the same POOL_SEED")
        dfs[model_name] = df
    return dfs


def correlate_models_on_sequence(sequence, model_names, output_root, quantities, mean_abs):
    """Output rows of one sequence, one per (module, quantity), from its activations CSVs."""
    dfs = read_sequence_dfs(sequence, model_names, output_root)
    # The generative parameters are constant within a sequence and the same for every model.
    first_df = dfs[model_names[0]]
    row_start = {
        'sequence_name': sequence,
        'lim_std': first_df['lim_std'].iloc[0],
        'd': first_df['d'].iloc[0],
        'tau_std': first_df['tau_std'].iloc[0],
    }
    return correlate_model_activation_quantities(dfs, model_names, quantities, mean_abs, row_start)


def correlate_models_on_pool(pool, pool_files, model_names, output_root, quantities, mean_abs):
    """Output rows of one pool of sequences, one per (module, quantity), from its pool activations CSVs."""
    dfs = read_pool_dfs(pool, pool_files, model_names, output_root)

    # tau_std of each sequence, in presentation order, e.g. '160-16-240-40'.
    first_timesteps = dfs[model_names[0]].drop_duplicates('pool_position')
    tau_std_values = []
    for tau_std in first_timesteps['tau_std']:
        tau_std_values.append(str(tau_std))
    row_start = {
        'pool_name': pool,
        'tau_std_order': '-'.join(tau_std_values),
    }
    return correlate_model_activation_quantities(dfs, model_names, quantities, mean_abs, row_start)


def summarize_scores(scores, group_columns, score_columns, mean_abs, count_column):
    """Mean of every score over the sequences (or pools) of each group, with their number.

    With `mean_abs`, the means are taken over the absolute scores. Groups keep
    the order in which they first appear in `scores`. NaN scores are left out of
    the means. The number of sequences or pools averaged goes to `count_column`.
    """
    table = scores[group_columns + score_columns].copy()
    if mean_abs:
        table[score_columns] = table[score_columns].abs()
    grouped = table.groupby(group_columns, sort=False)
    summary = grouped[score_columns].mean()
    summary.insert(0, count_column, grouped.size())
    return summary.reset_index()


if __name__ == '__main__':
    plots.set_script_path(__file__)

    # ------------------------- SETTINGS (edit these) -------------------------
    # Models to compare, every pair of them is correlated. Each must own an
    # <output-root>/<model>/activations folder (pool_activations with
    # SEQUENTIAL_PRESENTATION below) filled by run_exp_trials_pipeline.py, run with
    # INCLUDE_PRIOR = True for the prior quantities.
    MODEL_NAMES = cfg.EVALUATION_MODELS
    
    # OUTPUT_ROOT = cfg.EXP_SEQ_OUTPUT_ROOT
    # TRIALS_PATH = cfg.TRIALS_PATH
    # TRIALS_PATH = cfg.WORKSPACE_ROOT / 'Jasmin' / 'trialsequences_new'
    TRIALS_PATH = cfg.WORKSPACE_ROOT / 'Jasmin' / 'sub-01'
    OUTPUT_ROOT = cfg.RNN_DIR / 'exp_seq_act_output' / TRIALS_PATH.name

    # Remove the two prior quantities to run on activation CSVs written without
    # INCLUDE_PRIOR.
    QUANTITIES = ['posterior_norm', 'posterior_deriv', 'prior_norm', 'prior_deriv']

    # True: every mean (over pairs, then over sequences) is taken over absolute
    # correlations, and the CSV names end in '_meanabs'. False: plain means.
    MEAN_ABS = True

    # False: one evaluation per sequence, from the activations CSVs.
    # True: one evaluation per pool of four sequences (one per tau_std value) run
    # one after the other, each starting from the hidden states the previous one
    # ended with, from the pool_activations CSVs written by
    # run_exp_trials_pipeline.py with SEQUENTIAL_PRESENTATION = True. The CSV
    # names then contain '_pools'.
    SEQUENTIAL_PRESENTATION = True
    POOL_SEED = cfg.POOL_SEED       # the one the pool CSVs were written with

    # True: compute the correlations and write the CSVs. False: read the scores
    # CSV of an earlier run back instead, to plot the correlations again without
    # recomputing them; the CSVs and the settings file are then left as they are.
    COMPUTE_SCORES = True

    # True: also plot the correlations, in a figure saved next to the CSVs.
    PLOT_CORRELATIONS = True
    # Sequences (or pools) the points of the panels are taken from, from the
    # start of the list: every timestep of every one of them is one point of
    # every panel, so a handful of them already gives thousands of points.
    # Ignored with SHOW_ALL_SEQ below, which reads every sequence.
    FIGURE_SEQUENCES = 5
    # True: one point per timestep, its value averaged over every sequence (or
    # pool) of the set instead of one point per timestep of each of the first
    # FIGURE_SEQUENCES ones, the aggregation of
    # exp_sequence_analysis.load_timestep_means. The figures are then named
    # '..._allseq.png', and with PLOT_BY_TRIAL_POS each position's colour also
    # shades from the early to the late trials, as
    # timestep_matrix__by_position_gradient.png does. Every sequence is read, so
    # a run takes minutes rather than seconds.
    SHOW_ALL_SEQ = True
    # True: plot the panels as 2D histograms rather than as scatters of
    # individual points, which stays readable with many points (raise
    # FIGURE_SEQUENCES with it).
    FIGURE_DENSITY = False
    # True: colour every point after the position its timestep holds within its
    # trial (0-based, as the deviant positions dpos are), with one legend on the
    # right of the figure. A 2D histogram has no colour per point, so this and
    # FIGURE_DENSITY cannot both be on.
    PLOT_BY_TRIAL_POS = False

    # The CSVs and the figure are written to OUTPUT_DIR, their names starting
    # with OUTPUT_NAME. They are overwritten on every run: change OUTPUT_NAME
    # when comparing another list of models. The models compared are listed in
    # the accompanying '_settings.txt' file.
    OUTPUT_DIR = OUTPUT_ROOT / 'model_pair_correlations'
    OUTPUT_NAME = 'model_pair_correlations'
    # -------------------------------------------------------------------------

    if MEAN_ABS:
        suffix = '_meanabs'
    else:
        suffix = ''
    if SEQUENTIAL_PRESENTATION:
        csv_name_start = f'{OUTPUT_NAME}_pools_seed{POOL_SEED}'
    else:
        csv_name_start = OUTPUT_NAME
    # One row per (sequence or pool, module, quantity)
    scores_csv = OUTPUT_DIR / f'{csv_name_start}{suffix}.csv'
    # Mean over the sequences of each condition: one row per (condition, module, quantity)
    conditions_csv = OUTPUT_DIR / f'{csv_name_start}_by_condition{suffix}.csv'
    # Mean over all sequences or pools: one row per (module, quantity)
    all_conditions_csv = OUTPUT_DIR / f'{csv_name_start}_all_conditions{suffix}.csv'
    # The settings of the run, with the full model names
    settings_txt = OUTPUT_DIR / f'{csv_name_start}{suffix}_settings.txt'

    trial_files = find_trial_files(TRIALS_PATH)
    n_pairs = len(list(combinations(MODEL_NAMES, 2)))
    if SEQUENTIAL_PRESENTATION:
        # A pool mixes four conditions, so the pools are only summarized over all conditions.
        count_column = 'n_pools'
        sequences_or_pools = 'pools'
        pools = make_sequence_pools(trial_files, POOL_SEED)
        print(f"{len(pools)} pools of {len(pools[0])} sequences, {len(MODEL_NAMES)} models, "
              f"{n_pairs} pairs of models")
    else:
        count_column = 'n_sequences'
        sequences_or_pools = 'sequences'
        sequences = []
        for trial_file in trial_files:
            sequences.append(trial_file.stem)
        print(f"{len(sequences)} sequences, {len(MODEL_NAMES)} models, {n_pairs} pairs of models")

    # The pair columns are named after the short model names, so these must all
    # differ, otherwise the columns of different pairs would overwrite each other.
    short_names = []
    for model_name in MODEL_NAMES:
        short_names.append(short_model_name(model_name))
        print(f"  {short_model_name(model_name)}: {model_name}")
    if len(set(short_names)) < len(short_names):
        raise ValueError(f"Several models share the same short name: {short_names}")

    if COMPUTE_SCORES:
        rows = []
        if SEQUENTIAL_PRESENTATION:
            for pool_index, pool_files in enumerate(pools):
                rows.extend(correlate_models_on_pool(pool_name(pool_index, POOL_SEED), pool_files,
                                                     MODEL_NAMES, OUTPUT_ROOT, QUANTITIES, MEAN_ABS))
                if (pool_index + 1) % 100 == 0 or pool_index + 1 == len(pools):
                    print(f"  processed {pool_index + 1}/{len(pools)} pools")
        else:
            for i, sequence in enumerate(sequences, start=1):
                rows.extend(correlate_models_on_sequence(sequence, MODEL_NAMES, OUTPUT_ROOT,
                                                         QUANTITIES, MEAN_ABS))
                if i % 100 == 0 or i == len(sequences):
                    print(f"  processed {i}/{len(sequences)} sequences")
        scores = pd.DataFrame(rows)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        scores.to_csv(scores_csv, index=False)
        print(f"{len(scores)} rows -> {scores_csv}")

        # Scores to summarize: every pair of models, then the mean over pairs.
        score_columns = []
        for model_a, model_b in combinations(MODEL_NAMES, 2):
            score_columns.append(model_pair_column(model_a, model_b))
        score_columns.append('mean_over_pairs')

        if not SEQUENTIAL_PRESENTATION:
            by_condition = summarize_scores(scores, ['lim_std', 'd', 'tau_std', 'module', 'quantity'],
                                            score_columns, MEAN_ABS, count_column)
            # Conditions in increasing order; the stable sort keeps the module and quantity
            # order within each condition.
            by_condition = by_condition.sort_values(['lim_std', 'd', 'tau_std'], kind='stable')
            by_condition.to_csv(conditions_csv, index=False)
            print(f"{len(by_condition)} rows -> {conditions_csv}")

        all_conditions = summarize_scores(scores, ['module', 'quantity'], score_columns, MEAN_ABS,
                                          count_column)
        all_conditions.to_csv(all_conditions_csv, index=False)
        print(f"{len(all_conditions)} rows -> {all_conditions_csv}")

        # The settings these CSVs were computed with, written next to them.
        settings_lines = [
            f"Written by run_model_pair_correlations.py on {datetime.now():%Y-%m-%d %H:%M}",
            f"TRIALS_PATH: {TRIALS_PATH}",
            f"SEQUENTIAL_PRESENTATION: {SEQUENTIAL_PRESENTATION}",
        ]
        if SEQUENTIAL_PRESENTATION:
            settings_lines.append(f"POOL_SEED: {POOL_SEED}")
        settings_lines.append(f"MEAN_ABS: {MEAN_ABS}")
        settings_lines.append(f"QUANTITIES: {', '.join(QUANTITIES)}")
        settings_lines.append("MODEL_NAMES (short name: full name):")
        for model_name in MODEL_NAMES:
            settings_lines.append(f"  {short_model_name(model_name)}: {model_name}")
        settings_txt.write_text('\n'.join(settings_lines) + '\n')
        print(f"settings -> {settings_txt}")

        # A first look at the mean over pairs, over all conditions.
        print(all_conditions[['module', 'quantity', count_column, 'mean_over_pairs']])
    else:
        scores = pd.read_csv(scores_csv)
        print(f"{len(scores)} rows read back from {scores_csv}")

    if PLOT_CORRELATIONS and SHOW_ALL_SEQ:
        # Every sequence is read, and only their running sum is kept: holding
        # the timesteps of all of them at once would take gigabytes.
        running_sums = None
        if SEQUENTIAL_PRESENTATION:
            for pool_index, pool_files in enumerate(pools):
                pool = pool_name(pool_index, POOL_SEED)
                dfs = read_pool_dfs(pool, pool_files, MODEL_NAMES, OUTPUT_ROOT)
                table = get_sequence_activation_quantities(dfs, MODEL_NAMES, QUANTITIES, pool)
                running_sums = add_to_timestep_sums(running_sums, table, MODEL_NAMES)
                if (pool_index + 1) % 100 == 0 or pool_index + 1 == len(pools):
                    print(f"  read {pool_index + 1}/{len(pools)} pools")
            n_averaged = len(pools)
        else:
            for i, sequence in enumerate(sequences, start=1):
                dfs = read_sequence_dfs(sequence, MODEL_NAMES, OUTPUT_ROOT)
                table = get_sequence_activation_quantities(dfs, MODEL_NAMES, QUANTITIES, sequence)
                running_sums = add_to_timestep_sums(running_sums, table, MODEL_NAMES)
                if i % 100 == 0 or i == len(sequences):
                    print(f"  read {i}/{len(sequences)} sequences")
            n_averaged = len(sequences)
        activation_quantities = get_timestep_mean_activation_quantities(running_sums, MODEL_NAMES)
    elif PLOT_CORRELATIONS:
        # The time courses the points of the panels are taken from, read again
        # from the same CSVs the correlations were computed from.
        tables = []
        if SEQUENTIAL_PRESENTATION:
            for pool_index, pool_files in enumerate(pools[:FIGURE_SEQUENCES]):
                pool = pool_name(pool_index, POOL_SEED)
                dfs = read_pool_dfs(pool, pool_files, MODEL_NAMES, OUTPUT_ROOT)
                tables.append(get_sequence_activation_quantities(dfs, MODEL_NAMES, QUANTITIES, pool))
        else:
            for sequence in sequences[:FIGURE_SEQUENCES]:
                dfs = read_sequence_dfs(sequence, MODEL_NAMES, OUTPUT_ROOT)
                tables.append(get_sequence_activation_quantities(dfs, MODEL_NAMES, QUANTITIES, sequence))
        activation_quantities = pd.concat(tables, ignore_index=True)
        n_averaged = len(tables)

    if PLOT_CORRELATIONS:
        # Pair columns: the score column of the pair, then the models on x and y.
        pairs = []
        for model_a, model_b in combinations(MODEL_NAMES, 2):
            pairs.append([model_pair_column(model_a, model_b),
                          short_model_name(model_a), short_model_name(model_b)])

        # One figure per quantity, with one row per module. Every (module,
        # quantity) combination holds the points of one panel.
        n_module_quantities = len(MODULE_NAMES) * len(QUANTITIES)
        n_points = len(activation_quantities) // n_module_quantities
        if SHOW_ALL_SEQ:
            print(f"[figures] {n_points} points per panel, one per timestep, averaged over "
                  f"{n_averaged} {sequences_or_pools}")
        else:
            print(f"[figures] {n_points} points per panel, from {n_averaged} {sequences_or_pools}")
        for quantity in QUANTITIES:
            fig = plots.plot_model_pair_correlations(activation_quantities, scores, quantity, pairs,
                                                     mean_abs=MEAN_ABS,
                                                     point_density=FIGURE_DENSITY,
                                                     by_trial_position=PLOT_BY_TRIAL_POS,
                                                     timestep_gradient=SHOW_ALL_SEQ)
            plots.save_figure(fig, OUTPUT_DIR,
                              figure_png_name(csv_name_start, quantity, SHOW_ALL_SEQ))
