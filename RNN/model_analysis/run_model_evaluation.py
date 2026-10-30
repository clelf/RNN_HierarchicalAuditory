"""Compare trained models on a generated test set, and plot the comparison figures.

Each model is scored on its own test set by default, generated from that model's
own training data config, so a model trained with sigma_r pinned to some value is
scored in its own regime; SHARED_TEST_DATA puts them all on a single set instead.
When a Kalman-filter benchmark pickle is supplied, the test set AND the KF
predictions come from it and the KF is overlaid on the figures.

With USE_EXP_SEQUENCES = True, every model is instead scored on the same
experimental sequences, the files of TRIALS_PATH that run_exp_trials_pipeline.py
and run_model_pair_correlations.py read, each sequence run on its own from the
trained initial hidden state. The figures then go to
exp_seq_act_output/<TRIALS_PATH name>/model_comparison, next to the
model_pair_correlations folder of the same sequences, and their names end in
'_expseq'. With SEQUENTIAL_PRESENTATION = True as well, the sequences are
instead run in the pools of four of run_exp_trials_pipeline.py, one after the
other, and every metric is one value per pool. With POOL_AVG = True on top, this
is repeated for N_POOL_SEEDS different pool seeds and the violins show the pools
of all seeds together, which is the average of the violins of the single seeds.

Figures (each toggled in the SETTINGS block):
    1. metric violins    one panel per metric, all models side by side, plus one
                         standalone file per metric
    2. KF ratio violins  the observation-level metrics as a model / KF ratio
    3. calibration       per-model PIT ECDF against the uniform

Replaces model_analysis.py and the __main__ block of evaluate_models.py, which
did the same evaluation with different defaults. The scoring itself lives in
metrics.py, the figures in plots.py.
"""

import analysis_config as cfg
import metrics
import plots
from analysis_core import find_trial_files


if __name__ == '__main__':
    plots.set_script_path(__file__)

    # ------------------------- SETTINGS (edit these) -------------------------
    BASE_DIR = cfg.TRAINING_RESULTS_DIR

    # Restrict the comparison to specific models. None evaluates every sub-folder
    # of BASE_DIR holding a .pth file; entries are full paths or bare folder names
    # looked up inside BASE_DIR.
    # MODEL_DIRS = cfg.FIXED_SIGMA_R_MODELS
    MODEL_DIRS = cfg.EVALUATION_MODELS

    # Experimental sequences. True: every model is evaluated on the same sequence
    # files, those of TRIALS_PATH, instead of on a generated test set. Cannot be
    # used together with USE_BENCHMARK_DATA below.
    USE_EXP_SEQUENCES = True
    # TRIALS_PATH = cfg.WORKSPACE_ROOT / 'Jasmin' / 'trialsequences_new'
    TRIALS_PATH = cfg.WORKSPACE_ROOT / 'Jasmin' / 'sub-01'
    CUE_SEED = cfg.CUE_SEED         # cue re-encoding, the one run_exp_trials_pipeline.py uses
    # Sequential presentation (only with USE_EXP_SEQUENCES), as in
    # run_exp_trials_pipeline.py: the sequences are grouped into random pools of
    # four, one per tau_std value, and the four sequences of a pool are run one
    # after the other, each starting from the hidden states the previous one ended
    # with. Every metric is then one value per pool, averaged over its four
    # sequences, and the figure names end in '_pools_seed<POOL_SEED>'.
    SEQUENTIAL_PRESENTATION = True
    # POOL_SEED = cfg.POOL_SEED       # random pools, the one run_exp_trials_pipeline.py uses
    POOL_SEED = 1
    # Average over pool seeds (only with SEQUENTIAL_PRESENTATION). True: the
    # evaluation above is repeated with the pool seeds 0, 1, ..., N_POOL_SEEDS - 1
    # (POOL_SEED is then not used), and every violin holds the pools of all these
    # seeds together, N_POOL_SEEDS times as many values as with a single seed. The
    # figure names end in '_pools_avg<N_POOL_SEEDS>seeds'. Cannot be used together
    # with PLOT_CALIBRATION below.
    POOL_AVG = True
    N_POOL_SEEDS = 10

    # The figures of the experimental sequences go to the folder of their dataset,
    # next to the model_pair_correlations folder of run_model_pair_correlations.py.
    if USE_EXP_SEQUENCES:
        OUTPUT_DIR = cfg.RNN_DIR / 'exp_seq_act_output' / TRIALS_PATH.name / 'model_comparison'
    else:
        OUTPUT_DIR = cfg.EVALUATION_RESULTS_DIR / 'model_comparison' / 'HierarchicalGM'

    # Kalman-filter benchmark. When True, the test set and the KF predictions are
    # loaded from BENCHMARK_PATH instead of a fresh test set being generated.
    USE_BENCHMARK_DATA = False
    BENCHMARK_PATH = (cfg.RNN_DIR / 'benchmarks' / 'N_ctx_2' / 'NonHierarchicalGM'
                      / 'benchmarks_1000_test.pkl')

    # Test data (ignored when USE_BENCHMARK_DATA or USE_EXP_SEQUENCES is True).
    # SHARED_TEST_DATA=False generates one test set per model, from that model's
    # own training data config. Models sharing a config still share one generated
    # set. True forces the single first-model test set onto everyone -- only sound
    # when they really were trained on the same data config.
    SHARED_TEST_DATA = True
    # Seed of the test data generation. Keeps runs reproducible and, per model,
    # pairs the sets: configs differing only in sigma_r yield the same latents
    # (rules, contexts, cues, tau), so the comparison is not blurred by sampling
    # noise. None leaves the RNG alone and lets the GM generate in parallel.
    TEST_SEED = 0
    N_SAMPLES_TEST = 1000
    DEVICE = 'cpu'

    # One violin panel per metric, all in one row. Available keys depend on the
    # model type: 'mse', 'obs_loglik', 'ks_statistic' (all models),
    # 'context_accuracy', 'context_loglik' (n_ctx > 1), 'dpos_loglik',
    # 'dpos_accuracy', 'rule_loglik', 'rule_accuracy' (population_network).
    # Missing ones are skipped per model; a metric no model has leaves its panel
    # empty.
    METRICS = ['obs_loglik', 'context_loglik', 'dpos_loglik', 'rule_loglik']
    # The metrics above are averaged over every timestep of a sequence. True: also
    # plot the context, dpos and rule log-likelihoods averaged over the timesteps
    # of the true deviants and the timesteps right after them only (the rows
    # run_exp_trials_summaries.py averages), in a second row of panels, each below
    # its whole-sequence version: 'context_loglik_at_dev', 'dpos_loglik_at_dev',
    # 'rule_loglik_at_dev'.
    AT_DEV = True

    PLOT_METRIC_VIOLINS = True
    PLOT_KF_RATIOS = True           # only created when USE_BENCHMARK_DATA is True
    # True: in the violins of figures 1 and 2, a thin gray line joins the values of
    # each sequence (each pool with SEQUENTIAL_PRESENTATION) across the models,
    # showing whether the models rank the sequences in the same order. Only
    # meaningful when every model is scored on the same sequences, i.e. with
    # USE_EXP_SEQUENCES, USE_BENCHMARK_DATA or SHARED_TEST_DATA, or with TEST_SEED
    # for models whose data configs differ only in sigma_r.
    LINK_SEQ = True
    PLOT_CALIBRATION = False
    SAVE_INDIVIDUAL_PANELS = False   # one extra file per metric of figure 1
    ROW_LABELS = None               # e.g. ['tone process estimation', 'context estimation']
    # -------------------------------------------------------------------------

    if USE_EXP_SEQUENCES and USE_BENCHMARK_DATA:
        raise ValueError("USE_EXP_SEQUENCES and USE_BENCHMARK_DATA cannot both be True: "
                         "the KF predictions only exist for the benchmark test set")
    average_over_pool_seeds = USE_EXP_SEQUENCES and SEQUENTIAL_PRESENTATION and POOL_AVG
    if average_over_pool_seeds and PLOT_CALIBRATION:
        raise ValueError("POOL_AVG and PLOT_CALIBRATION cannot both be True: the calibration "
                         "curves are computed on the pools of a single seed")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if USE_BENCHMARK_DATA:
        file_suffix = 'kfbm'
    elif average_over_pool_seeds:
        file_suffix = f'expseq_pools_avg{N_POOL_SEEDS}seeds'
    elif USE_EXP_SEQUENCES and SEQUENTIAL_PRESENTATION:
        file_suffix = f'expseq_pools_seed{POOL_SEED}'
    elif USE_EXP_SEQUENCES:
        file_suffix = 'expseq'
    else:
        file_suffix = 'test'

    print(f"\nLoading models from {BASE_DIR}")
    models_info = metrics.load_models_info(BASE_DIR, model_dirs=MODEL_DIRS)

    benchmark_data = (metrics.load_benchmark_data(BENCHMARK_PATH)
                      if USE_BENCHMARK_DATA else None)
    eval_min_obs = benchmark_data.min_obs_for_em if USE_BENCHMARK_DATA else None

    if USE_EXP_SEQUENCES:
        trial_files = find_trial_files(TRIALS_PATH)
        if not trial_files:
            raise FileNotFoundError(f"No .csv or .txt sequence files in {TRIALS_PATH}")
        print(f"\nEvaluating on {len(trial_files)} experimental sequence files in {TRIALS_PATH}")

    if average_over_pool_seeds:
        pool_seeds = list(range(N_POOL_SEEDS))
        results = metrics.evaluate_all_models_over_pool_seeds(models_info, trial_files, CUE_SEED,
                                                              pool_seeds, device=DEVICE,
                                                              at_deviant=AT_DEV)
    else:
        if USE_EXP_SEQUENCES:
            test_sets = metrics.build_exp_sequence_test_sets(models_info, trial_files, CUE_SEED,
                                                             SEQUENTIAL_PRESENTATION, POOL_SEED,
                                                             device=DEVICE)
        else:
            # Every model that disagrees with the first one's data config is listed.
            metrics.check_shared_data_config(models_info, shared=SHARED_TEST_DATA)
            test_sets = metrics.build_test_sets(models_info, N_SAMPLES_TEST,
                                                shared=SHARED_TEST_DATA,
                                                benchmark_data=benchmark_data,
                                                device=DEVICE, seed=TEST_SEED)
        results = metrics.evaluate_all_models(models_info, test_sets,
                                              min_obs_for_em=eval_min_obs, device=DEVICE,
                                              at_deviant=AT_DEV)
    kf_metrics = metrics.compute_kf_metrics(benchmark_data) if USE_BENCHMARK_DATA else None
    results_ord = plots.style_results(results)

    # One violin panel per metric: METRICS in the first row. With AT_DEV, a second
    # row holds the at-dev version of every context, dpos and rule log-likelihood
    # right below it, and an empty panel (None) below the other metrics.
    plotted_metrics = list(METRICS)
    if AT_DEV:
        for metric in METRICS:
            if metric in ['context_loglik', 'dpos_loglik', 'rule_loglik']:
                plotted_metrics.append(metric + '_at_dev')
            else:
                plotted_metrics.append(None)

    if PLOT_METRIC_VIOLINS:
        plots.plot_metric_violins(results_ord, plotted_metrics, OUTPUT_DIR, LINK_SEQ,
                                  ncols=len(METRICS),
                                  kf_metrics=kf_metrics, file_suffix=file_suffix,
                                  row_labels=ROW_LABELS, symlog_metrics=(),
                                  save_individual=SAVE_INDIVIDUAL_PANELS)

    if PLOT_KF_RATIOS and kf_metrics is not None:
        plots.plot_kf_ratio_violins(results_ord, kf_metrics, OUTPUT_DIR, LINK_SEQ)

    if PLOT_CALIBRATION:
        plots.plot_calibration_curves(models_info, results_ord, test_sets, OUTPUT_DIR,
                                      device=DEVICE, min_obs_for_em=eval_min_obs)
