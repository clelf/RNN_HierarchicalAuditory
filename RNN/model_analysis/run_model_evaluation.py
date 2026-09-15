"""Compare trained models on a generated test set, and draw the comparison figures.

Each model is scored on its own test set by default, generated from that model's
own training data config, so a model trained with sigma_r pinned to some value is
scored in its own regime; SHARED_TEST_DATA puts them all on a single set instead.
When a Kalman-filter benchmark pickle is supplied, the test set AND the KF
predictions come from it and the KF is overlaid on the figures.

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


if __name__ == '__main__':
    plots.set_script_path(__file__)

    # ------------------------- SETTINGS (edit these) -------------------------
    BASE_DIR = cfg.TRAINING_RESULTS_DIR

    # Restrict the comparison to specific models. None evaluates every sub-folder
    # of BASE_DIR holding a .pth file; entries are full paths or bare folder names
    # looked up inside BASE_DIR.
    MODEL_DIRS = cfg.SIGMA_R_SWEEP_MODELS

    OUTPUT_DIR = cfg.EVALUATION_RESULTS_DIR / 'model_comparison' / 'HierarchicalGM'

    # Kalman-filter benchmark. When True, the test set and the KF predictions are
    # loaded from BENCHMARK_PATH instead of a fresh test set being generated.
    USE_BENCHMARK_DATA = False
    BENCHMARK_PATH = (cfg.RNN_DIR / 'benchmarks' / 'N_ctx_2' / 'NonHierarchicalGM'
                      / 'benchmarks_1000_test.pkl')

    # Test data (ignored when USE_BENCHMARK_DATA is True).
    # SHARED_TEST_DATA=False generates one test set per model, from that model's
    # own training data config. Models sharing a config still share one generated
    # set. True forces the single first-model test set onto everyone -- only sound
    # when they really were trained on the same data config.
    SHARED_TEST_DATA = False
    # Seed of the test data generation. Keeps runs reproducible and, per model,
    # pairs the sets: configs differing only in sigma_r yield the same latents
    # (rules, contexts, cues, tau), so the comparison is not blurred by sampling
    # noise. None leaves the RNG alone and lets the GM generate in parallel.
    TEST_SEED = 0
    N_SAMPLES_TEST = 1000
    DEVICE = 'cpu'

    # One violin panel per metric. Available keys depend on the model type:
    # 'mse', 'obs_loglik', 'ks_statistic' (all models), 'context_accuracy',
    # 'context_loglik' (n_ctx > 1), 'dpos_loglik', 'dpos_accuracy',
    # 'rule_loglik', 'rule_accuracy' (population_network). Missing ones are
    # silently skipped per model.
    METRICS = ['obs_loglik', 'context_loglik', 'dpos_loglik', 'rule_loglik']

    PLOT_METRIC_VIOLINS = True
    PLOT_KF_RATIOS = True           # only drawn when USE_BENCHMARK_DATA is True
    PLOT_CALIBRATION = False
    SAVE_INDIVIDUAL_PANELS = True   # one extra file per metric of figure 1
    ROW_LABELS = None               # e.g. ['tone process estimation', 'context estimation']
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_suffix = 'kfbm' if USE_BENCHMARK_DATA else 'test'

    print(f"\nLoading models from {BASE_DIR}")
    models_info = metrics.load_models_info(BASE_DIR, model_dirs=MODEL_DIRS)

    # Every model that disagrees with the first one's data config is listed.
    metrics.check_shared_data_config(models_info, shared=SHARED_TEST_DATA)

    benchmark_data = (metrics.load_benchmark_data(BENCHMARK_PATH)
                      if USE_BENCHMARK_DATA else None)
    eval_min_obs = benchmark_data.min_obs_for_em if USE_BENCHMARK_DATA else None
    test_sets = metrics.build_test_sets(models_info, N_SAMPLES_TEST,
                                        shared=SHARED_TEST_DATA,
                                        benchmark_data=benchmark_data,
                                        device=DEVICE, seed=TEST_SEED)

    results = metrics.evaluate_all_models(models_info, test_sets,
                                          min_obs_for_em=eval_min_obs, device=DEVICE)
    kf_metrics = metrics.compute_kf_metrics(benchmark_data) if USE_BENCHMARK_DATA else None
    results_ord = plots.style_results(results)

    if PLOT_METRIC_VIOLINS:
        plots.plot_metric_violins(results_ord, METRICS, OUTPUT_DIR,
                                  kf_metrics=kf_metrics, file_suffix=file_suffix,
                                  row_labels=ROW_LABELS,
                                  save_individual=SAVE_INDIVIDUAL_PANELS)

    if PLOT_KF_RATIOS and kf_metrics is not None:
        plots.plot_kf_ratio_violins(results_ord, kf_metrics, OUTPUT_DIR)

    if PLOT_CALIBRATION:
        plots.plot_calibration_curves(models_info, results_ord, test_sets, OUTPUT_DIR,
                                      device=DEVICE, min_obs_for_em=eval_min_obs)
