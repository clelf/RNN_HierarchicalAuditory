import os
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import evaluate_models as eval
import torch
import numpy as np

# Local imports from pipeline_core_v2
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from pipeline_core_v2 import plot_samples, get_model_predictions

# Import Kalman constants for benchmark data
from Kalman.kalman import MIN_OBS_FOR_EM

# =============================================================================
# Configuration
# =============================================================================

# Toggle between generating new test data (False) and using benchmark data (True)
# USE_BENCHMARK_DATA = True
USE_BENCHMARK_DATA = False

# Directory containing models
# models_dir = Path(__file__).parent.resolve() / Path('training_results/N_ctx_2/NonHierarchicalGM')
models_dir = Path(__file__).parent.resolve() / Path('training_results/N_ctx_2/HierarchicalGM')


# Benchmark data path (only used when USE_BENCHMARK_DATA=True)
# Should point to a benchmark results pickle produced by compute_benchmarks / pipeline_next
# benchmark_results_path = (
#     Path(__file__).parent.resolve()
#     / 'benchmarks' / 'N_ctx_2' / 'NonHierarchicalGM' / 'benchmarks_1000_test.pkl'
# )
benchmark_results_path = None
# Optional: separate input data file (set to None to load everything from results_path)
benchmark_input_data_path = None

N_samples = 16
N_plots_long = 8   # samples to plot for long (full-sequence) view
N_plots_short = 8  # samples to plot for short (seq_start=-100) view

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
    
    # For benchmark data, dpos, rules, and cues are not available
    dpos_np = None
    rules_np = None
    cues_np = None
    q = None
else:
    # Generate fresh test data from the generative model
    data_config_dict = models_info[0].data_config_dict
    kalman_mu = None
    kalman_sigma = None
    min_obs_for_em = None
    dpos_np = None
    rules_np = None
    cues_np = None
    q = None

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
        test_data = eval.generate_test_data(data_config_dict, n_samples=N_samples)
        y = test_data['y']
        y_np = test_data['y_np']
        pars = test_data['pars']
        contexts_np = test_data['contexts_np']
        
        # Extract hierarchical fields if available (for HierarchicalGM)
        dpos_np = test_data['dpos_np'] if 'dpos_np' in test_data else None
        rules_np = test_data['rules_np'] if 'rules_np' in test_data else None
        cues_np = test_data['q_np'] if 'q_np' in test_data else None  # cues from q_np field
        q = test_data['q'] if 'q' in test_data else None  # one-hot encoded cues tensor
        
        # Build data_config for plot_samples
        data_config = info.data_config_dict
    else:
        # For benchmark data, q is not available
        q = None

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
    dpos_np_sel = dpos_np[sample_indices] if dpos_np is not None else None
    rules_np_sel = rules_np[sample_indices] if rules_np is not None else None
    cues_np_sel = cues_np[sample_indices] if cues_np is not None else None
    q_sel = q[sample_indices] if q is not None else None  # select from q tensor
    kalman_mu_sel   = kalman_mu[sample_indices]   if kalman_mu   is not None else None
    kalman_sigma_sel = kalman_sigma[sample_indices] if kalman_sigma is not None else None

    model = eval.load_model(info)
    model_name = info.model_dir.name
    print(f"Plotting samples for model: {model_name}")

    # Forward pass (only on the selected samples)
    # Handle different model types and data modes
    with torch.no_grad():
        if q_sel is not None:
            # HierarchicalGM case: model expects (y, q) as input
            model_output = model(y_sel[:, :-1, :], q_sel[:, :-1, :])
        else:
            # NonHierarchicalGM case: model only expects y as input
            model_output = model(y_sel[:, :-1, :])
        
        # Calculate dpos_min to shift dpos predictions back to original coordinate system
        # (e.g., if dpos_true contains [3, 4, 5, 6, 7], dpos_min=3)
        # This ensures predictions are in the same space as true labels for plotting
        dpos_min = dpos_np_sel.min() if dpos_np_sel is not None else 0
        predictions = get_model_predictions(model, model_output, dpos_min=dpos_min)

    # Extract predictions from the get_model_predictions output
    mu_pred = predictions['mu_estim']
    sigma_pred = predictions['sigma_estim']
    contexts_probs = predictions['ctx_prob']
    contexts_preds = predictions['ctx_pred']
    dpos_probs = predictions['dpos_prob']
    dpos_preds = predictions['dpos_pred']
    rule_probs = predictions['rule_prob']
    rule_preds = predictions['rule_pred']

    # Build sample_metrics dictionary for plot_samples function
    sample_metrics = {
        'y': y_np_sel,
        'mu_estim': mu_pred,
        'sigma_estim': sigma_pred,
        'kalman_mu': kalman_mu_sel,
        'kalman_sigma': kalman_sigma_sel,
        'contexts': contexts_np_sel,
        'ctx_prob': contexts_probs,
        'ctx_pred': contexts_preds,
        'dpos_true': dpos_np_sel,
        'dpos_prob': dpos_probs,
        'dpos_pred': dpos_preds,
        'rule_true': rules_np_sel,
        'rule_prob': rule_probs,
        'rule_pred': rule_preds,
        'cues': cues_np_sel,
    }

    # ── Single unified plot call (matches _validate_epoch behavior) ──────
    # Internal logic of plot_samples handles multiple passes:
    # - For HierarchicalGM: full sequence + last 8 blocks
    # - For NonHierarchicalGM: full sequence only
    save_path = output_dir / f"{model_name}_samples"
    np.random.seed(model_seed)  # reproducible sample selection
    plot_samples(
        sample_metrics=sample_metrics,
        save_path=str(save_path),
        title=f"Samples – {model_name}",
        N_plots=max(N_plots_long, N_plots_short),
        seq_start=None,  # Full sequence (plot_samples handles windowing internally)
        seq_end=None,
        params=pars_sel,
        data_config=data_config,
        min_obs_for_em=min_obs_for_em,
        shared_ylim=True,
    )