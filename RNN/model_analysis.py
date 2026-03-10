import pandas as pd
import numpy as np
import seaborn as sns
import os
import torch
from pathlib import Path
import evaluate_models as eval
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from tqdm import tqdm
from scipy.stats import norm as sp_norm



def plot_calibration_curve(
    y_true: np.ndarray,
    mu_pred: np.ndarray,
    var_pred: np.ndarray,
    save_path: Path = None,
    title: str = "KS Calibration Plot",
    ax: plt.Axes = None,
    color: str = "#1f77b4",
    label: str = None,
    alpha_band: float = 0.05,
):
    """
    Plot the empirical CDF of the Probability Integral Transform (PIT) values
    against the ideal uniform diagonal.

    For a perfectly calibrated model the PIT values are Uniform(0,1), so the
    empirical CDF should lie on the diagonal.  The maximum vertical distance
    from the diagonal is the Kolmogorov–Smirnov D-statistic.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples, seq_len)
        True observations.
    mu_pred : np.ndarray, shape (n_samples, seq_len)
        Predicted means.
    var_pred : np.ndarray, shape (n_samples, seq_len)
        Predicted variances.
    save_path : Path, optional
        If given, save the figure to this path.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        If provided, draw on this axes (useful for multi-panel figures).
    color : str
        Line colour for the empirical CDF.
    label : str, optional
        Legend label for the empirical CDF curve.
    alpha_band : float
        Significance level for the KS confidence band (default 0.05 → 95 %).

    Returns
    -------
    fig : matplotlib Figure or None
        The figure object (None when an external *ax* was supplied).
    ks_stat : float
        The pooled KS D-statistic.
    """

    # --- Compute PIT values (pooled across samples & time) ---
    sigma_pred = np.sqrt(var_pred)
    pit = sp_norm.cdf((y_true - mu_pred) / sigma_pred)
    pit_flat = pit.ravel()
    pit_flat = pit_flat[~np.isnan(pit_flat)]
    pit_flat.sort()

    n = len(pit_flat)
    ecdf = np.arange(1, n + 1) / n          # empirical CDF values
    F    = pit_flat                           # theoretical quantiles (sorted PITs)

    # KS statistic = max |ECDF(f) - f|
    ks_stat = np.max(np.abs(ecdf - F))

    # --- KS confidence band width ---
    # c(alpha) for two-sided KS test: 1.36 (alpha=0.05), 1.22 (0.10), 1.63 (0.01)
    c_alpha = {0.01: 1.63, 0.05: 1.36, 0.10: 1.22}.get(alpha_band, 1.36)
    band_half = c_alpha / np.sqrt(n)

    # --- Plot ---
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = None

    # Confidence band around the diagonal
    F_grid = np.linspace(0, 1, 500)
    ax.fill_between(
        F_grid,
        np.clip(F_grid - band_half, 0, 1),
        np.clip(F_grid + band_half, 0, 1),
        color="grey", alpha=0.75,
        label=f"{int((1-alpha_band)*100)}% KS band",
    )

    # Ideal diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal (Uniform)")

    # Empirical CDF of PITs
    ax.plot(F, ecdf, color=color, linewidth=1.5,
            label=label or f"Empirical CDF (D={ks_stat:.4f})")

    # Mark the point of maximum deviation
    idx_max = np.argmax(np.abs(ecdf - F))
    ax.plot([F[idx_max], F[idx_max]], [F[idx_max], ecdf[idx_max]],
            color="red", linewidth=1.5, linestyle="-",
            label=f"Max deviation = {ks_stat:.4f}")
    ax.plot(F[idx_max], ecdf[idx_max], "o", color="red", markersize=5)

    ax.set_xlabel("PIT value (theoretical quantile)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    if own_fig and save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved calibration plot to: {save_path}")
        plt.close(fig)

    return fig, ks_stat


def get_desired_order(results):
    # Categorize models
    obs = [mod for mod in results if results[mod]['learning_objective'] == 'obs']
    obs_ctx = [mod for mod in results if results[mod]['learning_objective'] == 'obs_ctx']
    ctx = [mod for mod in results if results[mod]['learning_objective'] == 'ctx']

    # Sort obs by bottleneck_dim
    obs_sorted = sorted(obs, key=lambda mod: results[mod]['bottleneck_dim'])
    # Sort obs_ctx by bottleneck_dim, then kappa (inverted order)
    obs_ctx_sorted = sorted(obs_ctx, key=lambda mod: (results[mod]['bottleneck_dim'], results[mod]['kappa']))
    # Sort ctx by bottleneck_dim
    ctx_sorted = sorted(ctx, key=lambda mod: results[mod]['bottleneck_dim'])

    # Concatenate in desired order
    return obs_sorted + obs_ctx_sorted + ctx_sorted


if __name__ == "__main__":

    # Toggle: when True, load KF benchmark predictions and overlay them on the plots
    USE_BENCHMARK_DATA = True

    models_dir = Path(__file__).parent.resolve() / Path('training_results/N_ctx_2/NonHierarchicalGM')
    output_dir = Path(__file__).parent.resolve() / Path('evaluation_results/model_comparison/')
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_results_path = (
        Path(__file__).parent.resolve()
        / 'benchmarks' / 'N_ctx_2' / 'NonHierarchicalGM' / 'benchmarks_1000_test.pkl'
    )

   # Loading models info
    models_info = []
    for model_dir in os.listdir(models_dir):
        info = eval.ModelInfo.from_path(Path(models_dir / model_dir))
        models_info.append(info)

    # Load data info (all models should have the same data info, so we can just take the first one)
    data_info = models_info[0].data_config_dict
    N_samples_test = 1000
    data_config = eval.TestDataConfig.from_saved_config(data_info, n_samples=N_samples_test)


    # Choose test data source: generate or load from benchmark
    if USE_BENCHMARK_DATA:
        benchmark_data = eval.load_benchmark_data(benchmark_results_path)
        # Build test_data dict compatible with evaluate_model
        test_data = {
            'y': torch.tensor(benchmark_data.y, dtype=torch.float32).unsqueeze(-1),
            'y_np': benchmark_data.y,
            'contexts_np': benchmark_data.contexts if benchmark_data.contexts is not None else None,
            'pars': benchmark_data.pars,
        }
    else:
        test_data = eval.generate_test_data(data_config)

    # For each model, test the model on the dataset and compute metrics
    # metrics should not be averaged for one model but should be stored as a distribution of metrics for each model
    # use load_model and evaluate_model functions from evaluate_models.py to load the model and compute metrics
    # When benchmark data is used, restrict model evaluation to the same timestep
    # window as the KF (y[min_obs_for_em:]) so per-sample metrics are comparable.
    eval_min_obs = benchmark_data.min_obs_for_em if USE_BENCHMARK_DATA else None

    results = {}
    for info in tqdm(models_info, desc="Evaluating models"):
        model = eval.load_model(info)
        metrics = eval.evaluate_model(model, info, test_data, reduce=False,
                                      min_obs_for_em=eval_min_obs)
        model_name = info.model_dir.name
        results[model_name] = metrics

    # -------------------------------------------------------------------------
    # Load KF benchmark data and compute KF metrics (per-sample distributions)
    # -------------------------------------------------------------------------
    kf_metrics = None
    if USE_BENCHMARK_DATA:
        benchmark_data = eval.load_benchmark_data(benchmark_results_path)
        y_kf = benchmark_data.y                        # (N, T)
        mu_kf = benchmark_data.mu_kf                   # (N, T_eval)
        std_kf = benchmark_data.std_kf                 # (N, T_eval)
        var_kf = std_kf ** 2
        min_obs_for_em = benchmark_data.min_obs_for_em
        y_target_kf = y_kf[:, min_obs_for_em:]        # aligned targets

        kf_mse = eval.compute_mse(y_target_kf, mu_kf, reduce=False)
        kf_ll  = eval.compute_log_likelihood(y_target_kf, mu_kf, var_kf, reduce=False)
        kf_ks  = eval.compute_calibration_ks(y_target_kf, mu_kf, var_kf, reduce=False)
        kf_metrics = {'mse': kf_mse, 'log_likelihood': kf_ll, 'ks_statistic': kf_ks}

    # Metrics names
    metrics_names = ['mse', 'log_likelihood', 'ks_statistic', 'context_accuracy', 'context_log_prob']

    # Shorter name for model
    names_short = ['Obj=' + results[mod]['learning_objective'] + ' ' + (f"(k={results[mod]['kappa']}) " if results[mod]['learning_objective']=='obs_ctx' else '') + 'bdim=' + f"{results[mod]['bottleneck_dim']}" for mod in results.keys()]

    # Re-order dictionary
    desired_order_list = get_desired_order(results)
    results_ord = {k: results[k] for k in desired_order_list}

    # Define color maps for each learning objective
    cmap_blues = plt.get_cmap('Blues')
    cmap_reds = plt.get_cmap('Greens')
    cmap_oranges = plt.get_cmap('Oranges')

    # Count how many models per objective to space colors
    objectives = [results_ord[mod]['learning_objective'] for mod in results_ord.keys()]
    counts = {obj: objectives.count(obj) for obj in set(objectives)}
    indices = {obj: 0 for obj in set(objectives)}

    # Assign short name and color to models dict
    # For obs_ctx, span Oranges colormap per kappa value
    obs_ctx_kappas = sorted(set([results_ord[mod]['kappa'] for mod in results_ord.keys() if results_ord[mod]['learning_objective'] == 'obs_ctx']))
    obs_ctx_counts = {kappa: sum((results_ord[mod]['learning_objective'] == 'obs_ctx' and results_ord[mod]['kappa'] == kappa) for mod in results_ord.keys()) for kappa in obs_ctx_kappas}
    obs_ctx_indices = {kappa: 0 for kappa in obs_ctx_kappas}

    for mod in results_ord.keys():
        obj = results_ord[mod]['learning_objective']
        if obj == 'obs':
            color = to_hex(cmap_blues(0.3 + 0.7 * indices[obj] / max(counts[obj]-1,1)))
            indices[obj] += 1
        elif obj == 'ctx':
            color = to_hex(cmap_reds(0.3 + 0.7 * indices[obj] / max(counts[obj]-1,1)))
            indices[obj] += 1
        elif obj == 'obs_ctx':
            kappa = results_ord[mod]['kappa']
            n_kappa = obs_ctx_counts[kappa]
            idx_kappa = obs_ctx_indices[kappa]
            color = to_hex(cmap_oranges(0.3 + 0.7 * idx_kappa / max(n_kappa-1,1)))
            obs_ctx_indices[kappa] += 1
        else:
            color = '#888888'
        results_ord[mod]['color'] = color
        results_ord[mod]["disp_name"] = 'Obj=' + obj + ' ' + (f"(k={results_ord[mod]['kappa']}) " if obj=='obs_ctx' else '') + 'bdim=' + f"{results_ord[mod]['bottleneck_dim']}"

    

    # =========================================================================
    # Figure 3 – KS Calibration curves for all models
    # =========================================================================
    for mod_name in results_ord:
        # Find corresponding ModelInfo
        mod_info = next((info for info in models_info if info.model_dir.name == mod_name), None)
        if mod_info is None:
            continue
        # Re-run forward pass to get raw predictions (needed for the plot)
        model = eval.load_model(mod_info)
        with torch.no_grad():
            model_output = model(test_data['y'][:, :-1, :])
            mu_pred, var_pred, _ = eval.extract_model_predictions(model, model_output)

        y_target = test_data['y_np'][:, 1:]   # same alignment as evaluate_model

        disp = results_ord[mod_name]['disp_name']
        cal_save = output_dir / f"calibration_curve_{mod_name}.png"
        plot_calibration_curve(
            y_target, mu_pred, var_pred,
            save_path=cal_save,
            title=f"KS Calibration – {disp}",
            color=results_ord[mod_name].get('color', '#1f77b4'),
        )
    
    #========================================================================
    # Figure 1 – Comparison of metrics across models (violin plots)
    #===========================================================================
    # Plot comparison of the different metrics across models (one plot per metric)

    # fig, axd = plt.subplot_mosaic([['mse', 'log_likelihood', 'ks_statistic'],
    #                                ['context_accuracy', 'context_log_prob']],
    #                               figsize=(5.5, 3.5), layout="constrained")

    fig = plt.figure(figsize=(20, 10))
    subfigs = fig.subfigures(2, 1, wspace=0.05, hspace=0.25)

    axs0 = subfigs[0].subplots(1, 3)
    axs1 = subfigs[1].subplots(1, 3)  # Now 3 subplots in second row
    axs = [ax for ax in axs0] + [ax for ax in axs1[:2]]  # Only use first 2 for metrics

    mod_names = [results_ord[mod]['disp_name'] for mod in results_ord.keys()]
    mod_colors = [results_ord[mod]['color'] for mod in results_ord.keys()]

    # Metrics that are shared between models and KF (observation-level metrics only)
    kf_obs_metrics = {'mse', 'log_likelihood', 'ks_statistic'}

    # for metric in metrics_names:
    for id, (metric, ax) in enumerate(zip(metrics_names, axs)):
        # Build model columns
        model_arrays = [results_ord[mod][metric] for mod in results_ord.keys()]
        col_names    = [results_ord[mod]['disp_name'] for mod in results_ord.keys()]
        palette      = list(mod_colors)

        # Prepend KF column for observation-level metrics
        if USE_BENCHMARK_DATA and kf_metrics is not None and metric in kf_obs_metrics:
            model_arrays = [kf_metrics[metric]] + model_arrays
            col_names    = ['Kalman Filter'] + col_names
            palette      = ['#9B59B6'] + palette  # purple


        df_metric = pd.DataFrame(
            np.array(model_arrays).T,
            columns=col_names,
        )
        # Use palette to set colors for each model
        sns.violinplot(
            data=df_metric,
            log_scale=True if metric == 'mse' else False,
            ax=ax,
            cut=0,
            palette=palette,
        )
        ax.set_title(f'{metric} across models')
        ticks = ax.get_xticks()
        ax.set_xticks(ticks=ticks, labels=['']*len(ticks))
        if metric == 'log_likelihood':
            ax.set_yscale('symlog')
            ax.set_ylim(top=df_metric.max().max())

        # Add purple dashed line for KF median (top 3 metrics only)
        if USE_BENCHMARK_DATA and kf_metrics is not None and metric in kf_obs_metrics:
            kf_median = np.median(kf_metrics[metric])
            ax.axhline(kf_median, color='#9B59B6', linestyle='--', linewidth=2, label='KF median')

        # Save each subplot individually
        fig_single, ax_single = plt.subplots(figsize=(7, 5))
        sns.violinplot(
            data=df_metric,
            log_scale=True if metric == 'mse' else False,
            ax=ax_single,
            cut=0,
            palette=palette,
        )
        ax_single.set_title(f'{metric} across models')
        ax_single.set_xticks(ticks=ticks, labels=['']*len(ticks))
        if metric == 'log_likelihood':
            ax_single.set_yscale('symlog')
            ax_single.set_ylim(top=df_metric.max().max())
        if USE_BENCHMARK_DATA and kf_metrics is not None and metric in kf_obs_metrics:
            ax_single.axhline(kf_median, color='#9B59B6', linestyle='--', linewidth=2, label='KF median')
        # Add legend to individual subplot (outside, 3 columns)
        handles = [plt.Line2D([0], [0], color='#9B59B6', linestyle='--', linewidth=2, label='KF median')]
        handles += [plt.Line2D([0], [0], color=color, linewidth=8, label=name) for name, color in zip(col_names, palette)]
        ax_single.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=11, frameon=False, ncol=3)
        # fig_single.tight_layout()
        single_save_path = output_dir / f"subplot_{metric}_kfbm.png" if USE_BENCHMARK_DATA else output_dir / f"subplot_{metric}_test.png"
        fig_single.savefig(single_save_path, bbox_inches='tight')
        plt.close(fig_single)

    subfigs[0].supylabel('tone process estimation')
    subfigs[1].supylabel('context estimation')

    # Add legend as a color box in the third subplot of the second row
    ax_legend = axs1[2]
    ax_legend.axis('off')
    # KF entry at the top of the legend (if applicable)
    legend_names   = mod_names
    legend_colors  = mod_colors
    if USE_BENCHMARK_DATA and kf_metrics is not None:
        legend_names  = ['Kalman Filter'] + legend_names
        legend_colors = ['#9B59B6']       + legend_colors
    for i, (name, color) in enumerate(zip(legend_names, legend_colors)):
        ax_legend.add_patch(plt.Rectangle((0, 1-i*0.07), 0.2, 0.05, color=color, transform=ax_legend.transAxes, clip_on=False))
        ax_legend.text(0.25, 1-i*0.07+0.025, name, va='center', ha='left', transform=ax_legend.transAxes, fontsize=12)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.25, wspace=0.05)
    save_path = output_dir / (f"model_comparison_metrics_kfbm.png" if USE_BENCHMARK_DATA else f"model_comparison_metrics_test.png")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # =========================================================================
    # Figure 2 – Model / KF ratio for observation-level metrics
    # =========================================================================
    if USE_BENCHMARK_DATA and kf_metrics is not None:
        # Observation-level metrics (those with a KF counterpart)
        ratio_metrics = ['mse', 'log_likelihood', 'ks_statistic']

        # Single row of 3 metric plots
        fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        for metric, ax in zip(ratio_metrics, axs2):
            kf_vals = kf_metrics[metric]  # (N,)

            ratio_arrays = []
            ratio_col_names = []
            for mod in results_ord.keys():
                model_vals = results_ord[mod][metric]  # (N,)
                ratio = model_vals / (kf_vals + 1e-10)
                ratio_arrays.append(ratio)
                ratio_col_names.append(results_ord[mod]['disp_name'])

            df_ratio = pd.DataFrame(
                np.array(ratio_arrays).T,
                columns=ratio_col_names,
            )

            sns.violinplot(
                data=df_ratio,
                ax=ax,
                cut=0,
                palette=mod_colors,
            )
            ax.axhline(1.0, color='#9B59B6', linestyle='--', linewidth=1.5, label='KF baseline')
            ax.set_title(f'{metric}  (model / KF)')
            ticks = ax.get_xticks()
            ax.set_xticks(ticks=ticks, labels=['']*len(ticks))

            # Clip y-axis to keep the well-behaved models readable.
            # For log-likelihood the ratio can be extremely skewed, so we use
            # tighter per-column percentiles: clip each model's own distribution,
            # then take the union of those inner bounds across all "good" models
            # (i.e. ignore columns whose median is far from 1).
            lo_pct, hi_pct = (10, 90) if metric == 'log_likelihood' else (2, 98)
            col_los, col_his = [], []
            for col in df_ratio.columns:
                col_vals = df_ratio[col].dropna().values
                col_los.append(np.percentile(col_vals, lo_pct))
                col_his.append(np.percentile(col_vals, hi_pct))
            # Use the median of per-column bounds so one outlier column
            # cannot drag the axis range
            lo = np.median(col_los)
            hi = np.median(col_his)
            pad = (hi - lo) * 0.15
            ax.set_ylim(lo - pad, hi + pad)

        fig2.supylabel('model / KF ratio  (dashed purple line = KF baseline)', x=0.01)

        # Legend outside main axes
        handles = [plt.Line2D([0], [0], color='#9B59B6', linestyle='--', linewidth=2, label='KF baseline (ratio = 1)')]
        handles += [plt.Line2D([0], [0], color=color, linewidth=8, label=name) for name, color in zip(mod_names, mod_colors)]
        fig2.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=13, frameon=False)

        save_path2 = output_dir / "model_comparison_ratio_vs_kf.png"
        fig2.savefig(save_path2, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved model comparison ratio plot to: {save_path2}")


