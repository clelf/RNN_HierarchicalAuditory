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
models_dir = Path(__file__).parent.resolve() / Path('training_results_CORRECT/N_ctx_2/NonHierarchicalGM_selected')

# Benchmark data path (only used when USE_BENCHMARK_DATA=True)
# Should point to a benchmark results pickle produced by compute_benchmarks / pipeline_next
benchmark_results_path = (
    Path(__file__).parent.resolve()
    / 'benchmarks_CORRECT' / 'N_ctx_2' / 'NonHierarchicalGM' / 'benchmarks_1000_test.pkl'
)
# Optional: separate input data file (set to None to load everything from results_path)
benchmark_input_data_path = None

N_samples = 5
N_plots = 5

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
    N_plots = min(N_plots, benchmark_data.n_samples)
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

for info in models_info[3:4]: # [3:4]

    if not USE_BENCHMARK_DATA:
        # Generate fresh test data for this model
        test_data = eval.generate_test_data(test_data_config)
        data_config = test_data_config.to_gm_dict()
        y = test_data['y']
        y_np = test_data['y_np']
        pars = test_data['pars']
        contexts_np = test_data['contexts_np']

    model = eval.load_model(info)
    model_name = info.model_dir.name
    print(f"Plotting samples for model: {model_name}")

    # Forward pass
    with torch.no_grad():
        model_output = model(y[:, :-1, :])
        mu_pred, var_pred, context_output = eval.extract_model_predictions(model, model_output)

    # Context predictions (aligned with y[1:])
    contexts_logits = context_output.detach().cpu().numpy()
    contexts_probs = torch.softmax(torch.from_numpy(contexts_logits), dim=-1).numpy()
    contexts_preds = np.argmax(contexts_probs, axis=-1)
    contexts_target = contexts_np[:, 1:] if contexts_np is not None else None
    
    title = f"Samples for {model_name}"
    save_path = output_dir / f"{model_name}_samples"
    plot_samples(
        y_np, mu_pred, np.sqrt(var_pred), save_path,
        title=title,
        params=pars,
        N_plots=N_plots,
        data_config=data_config,
        contexts=contexts_np,
        seq_start=-100,
        seq_end=None,
        shared_ylim=True,
        contexts_probs=contexts_probs,
        contexts_preds=contexts_preds,
        # Kalman filter estimates from benchmarks (None when generating fresh data)
        # kalman_mu/sigma shape: (N, T - min_obs_for_em), aligned to obs[:, min_obs_for_em:]
        kalman_mu=kalman_mu,
        kalman_sigma=kalman_sigma,
        min_obs_for_em=min_obs_for_em,
    )