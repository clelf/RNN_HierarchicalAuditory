import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import evaluate_models as eval
import numpy as np

from model_activations import get_module_hidden_activity, plot_individual_trajectories, plot_averaged_activity, compute_sample_difference_all_modules, compute_module_independence


if __name__ == "__main__":
    # =============================================================================
    # Configuration
    # =============================================================================

    # Directory containing models
    models_dir = Path(__file__).parent.resolve() / Path('training_results/N_ctx_2/HierarchicalGM')

    # Number of samples to visualize
    N_samples = 8

    # =============================================================================
    # Load model info
    # =============================================================================
    # Load all models from directory:
    # models_info = [eval.ModelInfo.from_path(models_dir / model_dir) for model_dir in os.listdir(models_dir)]

    # Or load a single specific model:
    model_name = 'population_network_all_bn8_lr0'  # Change this to desired model name
    models_info = [eval.ModelInfo.from_path(models_dir / model_name)]

    # =============================================================================
    # Plot hidden activity for each model
    # =============================================================================
    output_dir = Path(__file__).parent.resolve() / Path('evaluation_results/hidden_activity')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seeded RNG so index selection is reproducible but distinct per model
    rng = np.random.default_rng(seed=0)

    for info in models_info:
        # Generate fresh test data for this model
        data_config_dict = info.data_config_dict
        test_data = eval.generate_test_data(data_config_dict, n_samples=N_samples)
        
        y = test_data['y']
        y_np = test_data['y_np']
        q = test_data['q'] if 'q' in test_data else None
        q_np = test_data['q_np'] if 'q_np' in test_data else None
        pars = test_data['pars'] if 'pars' in test_data else None
        
        # ------------------------------------------------------------------
        # Per-model random sample selection
        # ------------------------------------------------------------------
        n_available = y_np.shape[0]
        sample_indices = rng.choice(n_available, size=min(N_samples, n_available), replace=False)
        model_seed = int(rng.integers(0, 2**31))
        
        # Slice data down to selected samples
        y_sel = y[sample_indices]
        q_sel = q[sample_indices] if q is not None else None
        
        # Slice pars to match selected sample indices
        if pars is not None:
            pars_sel = {}
            for key, values in pars.items():
                if isinstance(values, list) or (hasattr(values, '__getitem__') and not isinstance(values, str)):
                    pars_sel[key] = [values[i] for i in sample_indices]
                else:
                    pars_sel[key] = values
            pars = pars_sel
        
        model = eval.load_model(info)
        model_name = info.model_dir.name
        print(f"Plotting hidden activity for model: {model_name}")

        module_norms_dict, module_derivatives_dict = get_module_hidden_activity(model, y_sel, q_sel)

        module_titles = {
            'obs':  'Observation module',
            'ctx':  'Type module',
            'dpos': 'Deviant position module',
            'rule': 'Rule module',
        }

        seq_len = next(iter(module_norms_dict.values())).shape[0]
        timesteps = np.arange(seq_len)
        
        # --- Plot 1: Individual trajectories (activity only) ---
        fig = plot_individual_trajectories(module_norms_dict, module_titles, timesteps, 
                                            output_dir, model_name, include_derivatives=False, pars=pars)
        save_path = output_dir / f"{model_name}_hidden_activity_trajectories.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectories figure to {save_path}")
        plt.close()
        
        # --- Plot 2: Averaged activity (activity only) ---
        fig = plot_averaged_activity(module_norms_dict, module_titles, timesteps,
                                    output_dir, model_name, n_samples=N_samples, include_derivatives=False)
        save_path_avg = output_dir / f"{model_name}_hidden_activity_averaged.png"
        plt.savefig(save_path_avg, dpi=150, bbox_inches='tight')
        print(f"Saved averaged figure to {save_path_avg}")
        plt.close()
        
        # --- Plot 3: Individual trajectories (derivatives only) ---
        fig = plot_individual_trajectories(module_norms_dict, module_titles, timesteps, 
                                            output_dir, model_name, include_derivatives=True, pars=pars)
        save_path_deriv = output_dir / f"{model_name}_hidden_activity_trajectories_derivatives.png"
        plt.savefig(save_path_deriv, dpi=150, bbox_inches='tight')
        print(f"Saved trajectories derivatives figure to {save_path_deriv}")
        plt.close()
        
        # --- Plot 4: Averaged activity (derivatives only) ---
        fig = plot_averaged_activity(module_norms_dict, module_titles, timesteps,
                                    output_dir, model_name, n_samples=N_samples, include_derivatives=True)
        save_path_avg_deriv = output_dir / f"{model_name}_hidden_activity_averaged_derivatives.png"
        plt.savefig(save_path_avg_deriv, dpi=150, bbox_inches='tight')
        print(f"Saved averaged derivatives figure to {save_path_avg_deriv}")
        plt.close()
        
        # --- Compute and display metrics ---
        print(f"\n{'='*60}")
        print(f"METRICS FOR MODEL: {model_name}")
        print(f"{'='*60}\n")
        
        # Sample-to-sample differences (example: first two samples)
        print("Sample-to-Sample Differences (Sample 0 vs Sample 1):")
        print("-" * 60)
        diff_metrics = compute_sample_difference_all_modules(module_norms_dict, 0, 1)
        for module_name in module_titles.keys():
            if module_name in diff_metrics:
                print(f"\n{module_titles[module_name]}:")
                for metric_name, value in diff_metrics[module_name].items():
                    print(f"  {metric_name}: {value:.4f}")
        print(f"\nAggregate metrics:")
        for metric_name, value in diff_metrics['aggregate'].items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Module independence metrics
        print(f"\n{'='*60}")
        print("Module Independence / Separability Metrics:")
        print("-" * 60)
        indep_metrics = compute_module_independence(module_norms_dict)
        
        print(f"\nCorrelation Matrix:")
        print(indep_metrics['correlation_matrix'].round(3))
        
        print(f"\nRedundancy Score: {indep_metrics['redundancy']:.4f} (0-1, lower = more independent)")
        print(f"Independence Score: {indep_metrics['independence_score']:.4f} (0-1, higher = more independent)")
        print(f"Mean Linear Predictability: {indep_metrics['mean_linear_predictability']:.4f} (0-1, lower = more independent)")
        print(f"Effective Dimensions: {indep_metrics['effective_dimensions']:.2f}/{len(module_titles)} modules")
        
        print(f"\nExplained Variance Ratio (PCA):")
        for i, var in enumerate(indep_metrics['explained_variance_ratio']):
            print(f"  PC{i+1}: {var:.4f}")
        
        print(f"\n{'='*60}\n")
        
        # Save metrics to file
        metrics_file = output_dir / f"{model_name}_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(f"METRICS FOR MODEL: {model_name}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write("Sample-to-Sample Differences (Sample 0 vs Sample 1):\n")
            f.write("-" * 60 + "\n")
            for module_name in module_titles.keys():
                if module_name in diff_metrics:
                    f.write(f"\n{module_titles[module_name]}:\n")
                    for metric_name, value in diff_metrics[module_name].items():
                        f.write(f"  {metric_name}: {value:.4f}\n")
            f.write(f"\nAggregate metrics:\n")
            for metric_name, value in diff_metrics['aggregate'].items():
                f.write(f"  {metric_name}: {value:.4f}\n")
            
            f.write(f"\n{'='*60}\n")
            f.write("Module Independence / Separability Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"\nCorrelation Matrix:\n")
            np.savetxt(f, indep_metrics['correlation_matrix'], fmt='%.3f')
            f.write(f"\nRedundancy Score: {indep_metrics['redundancy']:.4f}\n")
            f.write(f"Independence Score: {indep_metrics['independence_score']:.4f}\n")
            f.write(f"Mean Linear Predictability: {indep_metrics['mean_linear_predictability']:.4f}\n")
            f.write(f"Effective Dimensions: {indep_metrics['effective_dimensions']:.2f}/{len(module_titles)}\n")
            f.write(f"\nExplained Variance Ratio:\n")
            for i, var in enumerate(indep_metrics['explained_variance_ratio']):
                f.write(f"  PC{i+1}: {var:.4f}\n")
        
        print(f"Saved metrics to {metrics_file}")

    print("\nAll visualizations and metric computations complete!")
