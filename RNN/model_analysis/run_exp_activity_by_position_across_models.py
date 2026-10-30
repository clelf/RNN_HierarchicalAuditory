"""Plot the by-position activity figures of several models side by side.

The figure of run_exp_trials_pipeline.py with PLOT_BY_POSITION = True (activity
averaged over sequences, split by within-trial position) shows one model, with
one row per module. This script plots the same figure for every model of
MODEL_NAMES at once: one column per model, one row per module, the y-axis shared
by the models within a row, and a single legend.

Two variants of the figure can be plotted:
  PLOT_BY_POSITION            one line per within-trial position, as in
                              run_exp_trials_pipeline.py
  PLOT_BY_POSITION_AND_RULE   every position also split by the rule of its trial,
                              read from the sequence files: one line per (rule,
                              position), rule 0 in shades of purple and rule 1 in
                              shades of orange, from light (position 1) to dark
                              (position 8). A point is then averaged over the
                              sequences whose trial has that rule at that trial
                              index only.

Each variant can be plotted for the posterior hidden states (PLOT_POSTERIOR, the
ones of the run_exp_trials_pipeline.py figure) and for the prior ones
(PLOT_PRIOR), read from the 'prior_<module>_norm' columns of the CSVs, and comes
in two versions, as in run_exp_trials_pipeline.py: the activity norms and their
derivatives. The figures are saved to <output-root>/model_comparison/ and named
after the short model names (see analysis_core.short_model_name), e.g.
  sir0.02_sir0.05_sir0.1_sirvar_exp_activity_averaged_by_position_pools_seed0.png
  sir0.02_sir0.05_sir0.1_sirvar_exp_prior_activity_averaged_by_position_and_rule_pools_seed0.png
each with a '_derivatives' counterpart.

No model is run here: the activations CSVs are read from
<output-root>/<model name>/, and must first be written by
run_exp_trials_pipeline.py with the same TRIALS_PATH, SEQUENTIAL_PRESENTATION and
POOL_SEED, and with INCLUDE_PRIOR = True for the prior figures. Every model is
given the same sequences (or pools).
"""

import analysis_config as cfg
import exp_sequence_analysis as exp
import plots
from analysis_config import MODULE_TITLES
from analysis_core import (
    find_trial_files,
    make_sequence_pools,
    pool_name,
    select_files,
    sequence_csv_path,
    short_model_name,
)


def plot_models_activity_by_position(act_csvs_by_model, model_labels, prior, timestep_rules,
                                     sample_name, period, output_dir, figure_suffix):
    """Plot and save the by-position activity figures of every model, side by side.

    `act_csvs_by_model` holds, for every model in column order, the activations
    CSVs averaged over, and `prior` selects the prior hidden states (True) or the
    posterior ones (False). `timestep_rules` (exp_sequence_analysis.load_timestep_rules)
    splits every position by the rule of its trial, or is None for one line per
    position. Two figures are saved to `output_dir`, the activity norms and their
    derivatives, named after the short model names `model_labels`, and ending in
    `figure_suffix`, then '_derivatives' for the second one.
    """
    model_norms_dicts = []
    for act_csvs in act_csvs_by_model:
        model_norms_dicts.append(exp.load_activity_norms(act_csvs, prior=prior))

    figure_name_start = '_'.join(model_labels) + '_exp'
    if prior:
        figure_name_start += '_prior'
    figure_name_start += '_activity_averaged_by_position'
    if timestep_rules is not None:
        figure_name_start += '_and_rule'
    figure_name_start += figure_suffix

    for include_derivatives in [False, True]:
        fig = plots.plot_averaged_activity_by_position_across_models(
            model_norms_dicts, model_labels, MODULE_TITLES,
            n_samples=len(act_csvs_by_model[0]), sample_name=sample_name, prior=prior,
            timestep_rules=timestep_rules, period=period,
            include_derivatives=include_derivatives,
        )
        figure_name = figure_name_start
        if include_derivatives:
            figure_name += '_derivatives'
        plots.save_figure(fig, output_dir, figure_name + '.png')


if __name__ == '__main__':
    plots.set_script_path(__file__)

    # ------------------------- SETTINGS (edit these) -------------------------
    # Models to plot, one column each, in this order.
    MODEL_NAMES = cfg.EVALUATION_MODELS
    TRIALS_PATH = cfg.WORKSPACE_ROOT / 'Jasmin' / 'sub-01'
    OUTPUT_ROOT = cfg.RNN_DIR / 'exp_seq_act_output' / TRIALS_PATH.name

    # Variants of the figure: one line per within-trial position, and one line
    # per (rule, within-trial position).
    PLOT_BY_POSITION = True
    PLOT_BY_POSITION_AND_RULE = True

    # Hidden states plotted, each variant in two figures (norms and derivatives).
    # The prior figures need the CSVs written by run_exp_trials_pipeline.py with
    # INCLUDE_PRIOR = True.
    PLOT_POSTERIOR = True
    PLOT_PRIOR = True

    # False: averaged over sequences, from the activations CSVs.
    # True: averaged over pools of four sequences presented one after the other,
    # from the pool_activations CSVs; the figure names then end in
    # '_pools_seed<POOL_SEED>'.
    SEQUENTIAL_PRESENTATION = True
    POOL_SEED = cfg.POOL_SEED       # the one the pool CSVs were written with

    # Sequences (or pools) averaged over (None: all of them), sampled with SEED.
    N_SEQUENCES = None
    SEED = 0

    PERIOD = cfg.PERIOD             # timesteps per trial

    OUTPUT_DIR = OUTPUT_ROOT / 'model_comparison'
    # -------------------------------------------------------------------------

    # The columns and the figure names use the short model names, so these must
    # all differ.
    model_labels = []
    for model_name in MODEL_NAMES:
        model_labels.append(short_model_name(model_name))
        print(f"  {short_model_name(model_name)}: {model_name}")
    if len(set(model_labels)) < len(model_labels):
        raise ValueError(f"Several models share the same short name: {model_labels}")

    trial_files = find_trial_files(TRIALS_PATH)
    # Names of the sequences (file stems), or of the pools, whose CSVs are read,
    # and the sequence files each CSV was written from, in presentation order.
    sequences_or_pools = []
    sequence_files_by_csv = []
    if SEQUENTIAL_PRESENTATION:
        pools = make_sequence_pools(trial_files, POOL_SEED)
        for pool_index in range(len(pools)):
            sequences_or_pools.append(pool_name(pool_index, POOL_SEED))
            sequence_files_by_csv.append(pools[pool_index])
        csv_kind = 'pool_activations'
        sample_name = 'pools'
        figure_suffix = f'_pools_seed{POOL_SEED}'
    else:
        for trial_file in trial_files:
            sequences_or_pools.append(trial_file.stem)
            sequence_files_by_csv.append([trial_file])
        csv_kind = 'activations'
        sample_name = 'sequences'
        figure_suffix = ''

    # Sampled once, so that every model is given the same sequences (or pools).
    selected_positions = select_files(list(range(len(sequences_or_pools))), N_SEQUENCES, seed=SEED)
    print(f"Averaging over {len(selected_positions)} {sample_name}, {len(MODEL_NAMES)} models")

    act_csvs_by_model = []
    for model_name in MODEL_NAMES:
        act_csvs = []
        for i in selected_positions:
            act_csvs.append(sequence_csv_path(OUTPUT_ROOT, model_name, csv_kind,
                                              sequences_or_pools[i]))
        missing_csvs = [csv for csv in act_csvs if not csv.exists()]
        if missing_csvs:
            raise FileNotFoundError(f"{len(missing_csvs)} {csv_kind} CSV(s) missing for {model_name}, "
                                    f"e.g. {missing_csvs[0]}: run run_exp_trials_pipeline.py "
                                    f"first, with the same settings")
        act_csvs_by_model.append(act_csvs)

    if PLOT_BY_POSITION_AND_RULE:
        # The same for every model, since they are all given the same sequences:
        # read along the CSVs of the first one.
        selected_sequence_files = []
        for i in selected_positions:
            selected_sequence_files.append(sequence_files_by_csv[i])
        timestep_rules = exp.load_timestep_rules(act_csvs_by_model[0], selected_sequence_files)

    if PLOT_POSTERIOR and PLOT_BY_POSITION:
        plot_models_activity_by_position(act_csvs_by_model, model_labels, prior=False,
                                         timestep_rules=None, sample_name=sample_name,
                                         period=PERIOD, output_dir=OUTPUT_DIR,
                                         figure_suffix=figure_suffix)
    if PLOT_POSTERIOR and PLOT_BY_POSITION_AND_RULE:
        plot_models_activity_by_position(act_csvs_by_model, model_labels, prior=False,
                                         timestep_rules=timestep_rules, sample_name=sample_name,
                                         period=PERIOD, output_dir=OUTPUT_DIR,
                                         figure_suffix=figure_suffix)
    if PLOT_PRIOR and PLOT_BY_POSITION:
        plot_models_activity_by_position(act_csvs_by_model, model_labels, prior=True,
                                         timestep_rules=None, sample_name=sample_name,
                                         period=PERIOD, output_dir=OUTPUT_DIR,
                                         figure_suffix=figure_suffix)
    if PLOT_PRIOR and PLOT_BY_POSITION_AND_RULE:
        plot_models_activity_by_position(act_csvs_by_model, model_labels, prior=True,
                                         timestep_rules=timestep_rules, sample_name=sample_name,
                                         period=PERIOD, output_dir=OUTPUT_DIR,
                                         figure_suffix=figure_suffix)
