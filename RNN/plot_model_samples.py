import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import evaluate_models as eval
from pipeline_next import plot_samples
import torch
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

# Toggle between generating new test data (False) and using benchmark data (True)
USE_BENCHMARK_DATA = True

# Directory containing models
models_dir = Path(__file__).parent.resolve() / Path('training_results/N_ctx_2/NonHierarchicalGM')

# Benchmark data path (only used when USE_BENCHMARK_DATA=True)
# Should point to a benchmark results pickle produced by compute_benchmarks / pipeline_next
benchmark_results_path = (
    Path(__file__).parent.resolve()
    / 'benchmarks' / 'N_ctx_2' / 'NonHierarchicalGM' / 'benchmarks_1000_test.pkl'
)
# Optional: separate input data file (set to None to load everything from results_path)
benchmark_input_data_path = None

N_samples = 5
N_plots_long = 5   # samples to plot for long (full-sequence) view
N_plots_short = 5  # samples to plot for short (seq_start=-100) view

# =============================================================================
# Load model info
# =============================================================================
models_info = [eval.ModelInfo.from_path(models_dir / model_dir) for model_dir in os.listdir(models_dir)]

# =============================================================================
# Prepare data (benchmark or freshly generated)
# =============================================================================
if USE_BENCHMARK_DATA:
    # Load pre-computed benchmark data (observations + Kalman filter estimates)
    benchmark_data = eval.load_benchmark_data(benchmark_results_path, benchmark_input_data_path)
    print(f"Loaded benchmark data: {benchmark_data.n_samples} samples, "
          f"seq_len={benchmark_data.seq_len}, eval_len={benchmark_data.eval_seq_len}, "
          f"min_obs_for_em={benchmark_data.min_obs_for_em}")

    y_np = benchmark_data.y                     # (N, T)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)  # (N, T, 1)
    pars = benchmark_data.pars
    contexts_np = benchmark_data.contexts        # (N, T) or None
    kalman_mu = benchmark_data.mu_kf             # (N, T - min_obs_for_em)
    kalman_sigma = benchmark_data.std_kf         # (N, T - min_obs_for_em)
    min_obs_for_em = benchmark_data.min_obs_for_em

    # Build a data_config dict for plot_samples (uses gm_name, N_ctx, N_tones)
    first_info = models_info[0]
    data_config = {
        'gm_name': first_info.gm_name,
        'N_ctx': first_info.n_ctx,
        'N_tones': y_np.shape[1],
    }
    # Restrict N_plots to available samples
    N_plots_long = min(N_plots_long, benchmark_data.n_samples)
    N_plots_short = min(N_plots_short, benchmark_data.n_samples)
else:
    # Generate fresh test data from the generative model
    data_info = models_info[0].data_config_dict
    test_data_config = eval.TestDataConfig.from_saved_config(data_info, n_samples=N_samples)
    kalman_mu = None
    kalman_sigma = None
    min_obs_for_em = None

# =============================================================================
# Plot samples for each model
# =============================================================================
output_dir = Path(__file__).parent.resolve() / Path('evaluation_results/example_samples')
output_dir.mkdir(parents=True, exist_ok=True)

# Seeded RNG so index selection is reproducible but distinct per model
rng = np.random.default_rng(seed=0)

for info in models_info: # [3:4]

    if not USE_BENCHMARK_DATA:
        # Generate fresh test data for this model
        test_data = eval.generate_test_data(test_data_config)
        data_config = test_data_config.to_gm_dict()
        y = test_data['y']
        y_np = test_data['y_np']
        pars = test_data['pars']
        contexts_np = test_data['contexts_np']

    # ------------------------------------------------------------------
    # Per-model random sample selection (different for every model)
    # ------------------------------------------------------------------
    n_available = y_np.shape[0]
    n_sel = max(N_plots_long, N_plots_short)
    sample_indices = rng.choice(n_available, size=min(n_sel, n_available), replace=False)
    # Draw a per-model seed (from the same rng, so it advances and stays unique per model)
    model_seed = int(rng.integers(0, 2**31))

    # Slice all data arrays down to the selected samples
    y_sel    = y[sample_indices]
    y_np_sel = y_np[sample_indices]
    pars_sel = (
        [pars[i] for i in sample_indices]
        if isinstance(pars, list)
        else {k: (v[sample_indices] if isinstance(v, np.ndarray) else v)
              for k, v in pars.items()}
        if isinstance(pars, dict)
        else pars[sample_indices]
    )
    contexts_np_sel = contexts_np[sample_indices] if contexts_np is not None else None
    kalman_mu_sel   = kalman_mu[sample_indices]   if kalman_mu   is not None else None
    kalman_sigma_sel = kalman_sigma[sample_indices] if kalman_sigma is not None else None

    model = eval.load_model(info)
    model_name = info.model_dir.name
    print(f"Plotting samples for model: {model_name}")

    # Forward pass (only on the selected samples)
    with torch.no_grad():
        model_output = model(y_sel[:, :-1, :])
        mu_pred, var_pred, context_output = eval.extract_model_predictions(model, model_output)

    # Context predictions (aligned with y[1:])
    contexts_logits = context_output.detach().cpu().numpy()
    contexts_probs = torch.softmax(torch.from_numpy(contexts_logits), dim=-1).numpy()
    contexts_preds = np.argmax(contexts_probs, axis=-1)
    contexts_target = contexts_np_sel[:, 1:] if contexts_np_sel is not None else None

    common_kwargs = dict(
        params=pars_sel,
        data_config=data_config,
        contexts=contexts_np_sel,
        shared_ylim=True,
        contexts_probs=contexts_probs,
        contexts_preds=contexts_preds,
        kalman_mu=kalman_mu_sel,
        kalman_sigma=kalman_sigma_sel,
        min_obs_for_em=min_obs_for_em,
    )

    # ── Long view: full sequence ──────────────────────────────────────
    save_path_long = output_dir / f"{model_name}_samples_long"
    np.random.seed(model_seed)  # fix global state so both views pick the same samples
    plot_samples(
        y_np_sel, mu_pred, np.sqrt(var_pred), save_path_long,
        title=f"Samples (long) – {model_name}",
        N_plots=N_plots_long,
        seq_start=None,
        seq_end=None,
        **common_kwargs,
    )

    # ── Short view: last 100 observations (same samples as long view) ─
    save_path_short = output_dir / f"{model_name}_samples_short"
    np.random.seed(model_seed)  # same seed → same selected_indices inside plot_samples
    plot_samples(
        y_np_sel, mu_pred, np.sqrt(var_pred), save_path_short,
        title=f"Samples (short) – {model_name}",
        N_plots=N_plots_short,
        seq_start=-100,
        seq_end=None,
        **common_kwargs,
    )