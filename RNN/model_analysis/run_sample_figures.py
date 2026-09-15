"""Per-sample figures: model predictions on synthetic and experimental sequences.

Three stages, each toggled in the SETTINGS block:

  synthetic samples     plot_samples on freshly generated test data (or on a
                        benchmark pickle, which also supplies the KF overlay)
                        -> evaluation_results/example_samples/
  experimental samples  the same figure on the recorded trial sequences, one
                        figure per sequence so each is named after its file
                        -> <output-root>/<model>/samples/
  hidden activity       per-module activity trajectories on synthetic data, plus
                        the module-independence metrics written as a .txt
                        -> evaluation_results/hidden_activity/

Replaces plot_model_samples.py, plot_model_samples_exp_trials.py and
plot_hidden_activity.py. The first of those had no ``__main__`` block at all:
importing it ran the whole analysis as a side effect.

plot_samples names and writes its own files, so those figures are not routed
through plots.save_figure and get no figure_metadoc entry; the hidden-activity
figures, which this script saves itself, do.
"""

import numpy as np
import pandas as pd
import torch

import analysis_config as cfg
import metrics
import plots
from analysis_core import (
    ModelInfo,
    compute_module_independence,
    compute_sample_difference_all_modules,
    dpos_conventions,
    find_trial_files,
    generate_test_data,
    get_module_output_and_activity,
    load_model,
    load_trial_params,
    load_trial_sequence,
)

# analysis_core has already put RNN/train on sys.path.
from pipeline_core_v2 import plot_samples, get_model_predictions


def select_rows(value, indices):
    """Slice one entry of a sample bundle down to `indices`, whatever its container."""
    if value is None:
        return None
    if isinstance(value, list):
        return [value[i] for i in indices]
    if isinstance(value, dict):
        return {k: (v[indices] if isinstance(v, np.ndarray) else v)
                for k, v in value.items()}
    return value[indices]


def draw_synthetic_samples(info, output_dir, n_samples, n_plots, rng,
                           benchmark_data):
    """plot_samples for one model on generated test data, or on the benchmark set."""
    model_name = info.model_dir.name

    if benchmark_data is not None:
        y_np = benchmark_data.y
        y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)
        bundle = {
            'pars': benchmark_data.pars,
            'contexts_np': benchmark_data.contexts,
            'dpos_np': None, 'rules_np': None, 'cues_np': None, 'q': None,
            'kalman_mu': benchmark_data.mu_kf,
            'kalman_sigma': benchmark_data.std_kf,
        }
        min_obs_for_em = benchmark_data.min_obs_for_em
        # plot_samples needs gm_name, N_ctx and N_tones only.
        data_config = {'gm_name': info.gm_name, 'N_ctx': info.n_ctx,
                       'N_tones': y_np.shape[1]}
        n_plots = min(n_plots, benchmark_data.n_samples)
    else:
        test_data = generate_test_data(info.data_config_dict, n_samples=n_samples)
        y, y_np = test_data['y'], test_data['y_np']
        bundle = {
            'pars': test_data['pars'],
            'contexts_np': test_data['contexts_np'],
            'dpos_np': test_data.get('dpos_np'),
            'rules_np': test_data.get('rules_np'),
            'cues_np': test_data.get('q_np'),
            'q': test_data.get('q'),
            'kalman_mu': None, 'kalman_sigma': None,
        }
        min_obs_for_em = None
        data_config = info.data_config_dict

    # Per-model selection, drawn from a shared rng so it differs between models
    # yet stays reproducible across runs.
    n_available = y_np.shape[0]
    indices = rng.choice(n_available, size=min(n_plots, n_available), replace=False)
    model_seed = int(rng.integers(0, 2 ** 31))

    y_sel = y[indices]
    sel = {k: select_rows(v, indices) for k, v in bundle.items()}
    q_sel = sel['q']

    model = load_model(info)
    model.eval()
    print(f"Plotting samples for model: {model_name}")

    with torch.no_grad():
        if q_sel is not None:
            model_output = model(y_sel[:, :-1, :], q_sel[:, :-1, :])
        else:
            model_output = model(y_sel[:, :-1, :])
        # dpos predictions are shifted back into the coordinates dpos_true uses.
        dpos_min = sel['dpos_np'].min() if sel['dpos_np'] is not None else 0
        predictions = get_model_predictions(model, model_output, dpos_min=dpos_min)

    sample_metrics = {
        'y': y_np[indices],
        'mu_estim': predictions['mu_estim'],
        'sigma_estim': predictions['sigma_estim'],
        'kalman_mu': sel['kalman_mu'],
        'kalman_sigma': sel['kalman_sigma'],
        'contexts': sel['contexts_np'],
        'ctx_prob': predictions['ctx_prob'],
        'ctx_pred': predictions['ctx_pred'],
        'dpos_true': sel['dpos_np'],
        'dpos_prob': predictions['dpos_prob'],
        'dpos_pred': predictions['dpos_pred'],
        'rule_true': sel['rules_np'],
        'rule_prob': predictions['rule_prob'],
        'rule_pred': predictions['rule_pred'],
        'cues': sel['cues_np'],
    }

    # plot_samples handles the windowing itself: for a HierarchicalGM it writes
    # both a '_full' and a '_last8blocks' view per sample.
    np.random.seed(model_seed)   # reproducible sample selection inside plot_samples
    plot_samples(
        sample_metrics=sample_metrics,
        save_path=str(output_dir / f"{model_name}_samples"),
        title=f"Samples – {model_name}",
        N_plots=n_plots,
        seq_start=None,
        seq_end=None,
        params=sel['pars'],
        data_config=data_config,
        min_obs_for_em=min_obs_for_em,
        shared_ylim=True,
    )


def draw_experimental_samples(info, model_name, selected_files, output_dir, seed):
    """plot_samples on the recorded sequences, one figure per sequence file."""
    model = load_model(info)
    model.eval()
    data_config = info.data_config_dict
    n_cue_classes = len(data_config['cues_set'])
    # dpos_min is the class-index offset; dpos_shift maps the on-disk experimental
    # dpos into the model's convention so dpos_true and dpos_pred share coordinates.
    dpos_min, dpos_shift = dpos_conventions(info)

    obs_list, cue_list, ctx_list, dpos_list, rule_list = [], [], [], [], []
    tau_list, lim_list, si_q_list, si_stat_list, si_r_list = [], [], [], [], []

    for f in selected_files:
        obs, cue, ctx, dpos, rule, lim_std, d, tau_std, trial_n = load_trial_sequence(
            f, return_hierarch=True, n_cue_classes=n_cue_classes,
            cue_seed=cfg.CUE_SEED)
        obs_list.append(obs)
        cue_list.append(cue)
        ctx_list.append(ctx)
        dpos_list.append(dpos + dpos_shift)
        rule_list.append(rule)

        # Per-sequence scalars for the figure title (n_ctx == 2 expects std/dev
        # pairs for tau, lim and si_q).
        row = pd.read_csv(f, nrows=1).iloc[0]
        params = load_trial_params(f)
        tau_list.append([float(row['tau_std']), float(row['tau_dev'])])
        lim_list.append([float(row['lim_std']), float(row['lim_dev'])])
        si_q_list.append([float(row['sigma_q_std']), float(row['sigma_q_dev'])])
        si_stat_list.append(params['si_stat'])
        si_r_list.append(params['si_r'])

    obs_np = np.stack(obs_list).astype(np.float32)
    cue_np = np.stack(cue_list).astype(np.float32)
    ctx_np, dpos_np, rule_np = np.stack(ctx_list), np.stack(dpos_list), np.stack(rule_list)

    y = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(-1)
    q = torch.tensor(cue_np, dtype=torch.float32)

    all_params = {
        'tau': np.asarray(tau_list, dtype=float),
        'lim': np.asarray(lim_list, dtype=float),
        'si_q': np.asarray(si_q_list, dtype=float),
        'si_stat': np.asarray(si_stat_list, dtype=float),
        'si_r': np.asarray(si_r_list, dtype=float),
    }

    with torch.no_grad():
        model_output = model(y[:, :-1, :], q[:, :-1, :])
        predictions = get_model_predictions(model, model_output, dpos_min=dpos_min)

    sample_metrics = {
        'y': obs_np,
        'mu_estim': predictions['mu_estim'],
        'sigma_estim': predictions['sigma_estim'],
        'kalman_mu': None,
        'kalman_sigma': None,
        'contexts': ctx_np,
        'ctx_prob': predictions['ctx_prob'],
        'ctx_pred': predictions['ctx_pred'],
        'dpos_true': dpos_np,
        'dpos_prob': predictions['dpos_prob'],
        'dpos_pred': predictions['dpos_pred'],
        'rule_true': rule_np,
        'rule_prob': predictions['rule_prob'],
        'rule_pred': predictions['rule_pred'],
        'cues': cue_np,
    }

    # plot_samples names figures {save_path}_s{id}.png and picks the batch rows
    # itself, so a single batched call cannot tie a figure back to its source
    # file. One length-1 batch per call instead, named after that sequence.
    for j, src in enumerate(selected_files):
        seq_metrics = {k: (None if v is None else v[j:j + 1])
                       for k, v in sample_metrics.items()}
        seq_params = {k: v[j:j + 1] for k, v in all_params.items()}
        np.random.seed(seed)
        plot_samples(
            sample_metrics=seq_metrics,
            save_path=str(output_dir / src.stem),
            title=f"Samples (exp. trials) – {model_name} – {src.stem}",
            N_plots=1,
            seq_start=None,
            seq_end=None,
            params=seq_params,
            data_config=data_config,
            min_obs_for_em=None,
            shared_ylim=True,
        )


def write_independence_metrics(path, model_name, diff_metrics, indep_metrics,
                               module_titles):
    """Save the sample-difference and module-independence numbers as plain text."""
    with open(path, 'w') as fh:
        fh.write(f"METRICS FOR MODEL: {model_name}\n{'=' * 60}\n\n")

        fh.write("Sample-to-Sample Differences (Sample 0 vs Sample 1):\n")
        fh.write("-" * 60 + "\n")
        for module_name, title in module_titles.items():
            if module_name in diff_metrics:
                fh.write(f"\n{title}:\n")
                for metric_name, value in diff_metrics[module_name].items():
                    fh.write(f"  {metric_name}: {value:.4f}\n")
        fh.write("\nAggregate metrics:\n")
        for metric_name, value in diff_metrics['aggregate'].items():
            fh.write(f"  {metric_name}: {value:.4f}\n")

        fh.write(f"\n{'=' * 60}\n")
        fh.write("Module Independence / Separability Metrics:\n")
        fh.write("-" * 60 + "\n\nCorrelation Matrix:\n")
        np.savetxt(fh, indep_metrics['correlation_matrix'], fmt='%.3f')
        fh.write(f"\nRedundancy Score: {indep_metrics['redundancy']:.4f}\n")
        fh.write(f"Independence Score: {indep_metrics['independence_score']:.4f}\n")
        fh.write("Mean Linear Predictability: "
                 f"{indep_metrics['mean_linear_predictability']:.4f}\n")
        fh.write(f"Effective Dimensions: {indep_metrics['effective_dimensions']:.2f}"
                 f"/{len(module_titles)}\n")
        fh.write("\nExplained Variance Ratio:\n")
        for i, var in enumerate(indep_metrics['explained_variance_ratio']):
            fh.write(f"  PC{i + 1}: {var:.4f}\n")
    print(f"Saved metrics to {path}")


if __name__ == '__main__':
    plots.set_script_path(__file__)

    # ------------------------- SETTINGS (edit these) -------------------------
    MODEL_NAMES = cfg.SIGMA_R_SWEEP_MODELS
    MODEL_DIR = cfg.TRAINING_RESULTS_DIR

    RUN_SYNTHETIC = True
    RUN_EXPERIMENTAL = True
    RUN_HIDDEN_ACTIVITY = True

    # Synthetic stage. BENCHMARK_PATH=None generates fresh test data; a pickle
    # produced by the benchmark pipeline supplies the data and the KF overlay.
    N_SAMPLES = 16
    N_PLOTS = 8
    BENCHMARK_PATH = None
    BENCHMARK_INPUT_PATH = None

    # Experimental stage. SEQUENCE_NAMES=None samples N_SEQUENCES files at
    # random; otherwise it names the sequences explicitly.
    SEQUENCE_NAMES = [
        'sub-seq75_cond13_tau240_mu-0.6_dev1_ses-1_run-1_trials.csv',
        'sub-seq79_cond14_tau240_mu-0.6_dev-1_ses-1_run-1_trials.csv',
    ]
    N_SEQUENCES = 2

    # Hidden-activity stage.
    N_SAMPLES_HIDDEN = 8

    SEED = 0
    # -------------------------------------------------------------------------

    models_info = [ModelInfo.from_path(MODEL_DIR / name) for name in MODEL_NAMES]

    if RUN_SYNTHETIC:
        print(f"\n{'=' * 79}\nSynthetic samples\n{'=' * 79}")
        out_dir = cfg.EVALUATION_RESULTS_DIR / 'example_samples'
        out_dir.mkdir(parents=True, exist_ok=True)
        benchmark_data = (metrics.load_benchmark_data(BENCHMARK_PATH,
                                                      BENCHMARK_INPUT_PATH)
                          if BENCHMARK_PATH is not None else None)
        rng = np.random.default_rng(seed=SEED)
        for info in models_info:
            draw_synthetic_samples(info, out_dir, N_SAMPLES, N_PLOTS, rng,
                                   benchmark_data)

    if RUN_EXPERIMENTAL:
        print(f"\n{'=' * 79}\nExperimental samples\n{'=' * 79}")
        if SEQUENCE_NAMES:
            selected_files = [cfg.TRIALS_PATH / name for name in SEQUENCE_NAMES]
        else:
            all_files = find_trial_files(cfg.TRIALS_PATH)
            rng = np.random.default_rng(seed=SEED)
            picked = rng.choice(len(all_files), size=min(N_SEQUENCES, len(all_files)),
                                replace=False)
            selected_files = [all_files[i] for i in sorted(picked)]
        print(f"Plotting {len(selected_files)} sequence(s)")

        for info, model_name in zip(models_info, MODEL_NAMES):
            out_dir = cfg.EXP_SEQ_OUTPUT_ROOT / model_name / 'samples'
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== {model_name}")
            draw_experimental_samples(info, model_name, selected_files, out_dir, SEED)

    if RUN_HIDDEN_ACTIVITY:
        print(f"\n{'=' * 79}\nHidden activity\n{'=' * 79}")
        out_dir = cfg.EVALUATION_RESULTS_DIR / 'hidden_activity'
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed=SEED)

        for info in models_info:
            model_name = info.model_dir.name
            print(f"\n=== {model_name}")
            test_data = generate_test_data(info.data_config_dict,
                                           n_samples=N_SAMPLES_HIDDEN)
            y, q = test_data['y'], test_data.get('q')

            model = load_model(info)
            model.eval()
            _, module_norms, _ = get_module_output_and_activity(model, y, q)

            timesteps = np.arange(next(iter(module_norms.values())).shape[0])
            # Both plot functions take the NORMS and differentiate internally when
            # include_derivatives is set, so module_derivs is not passed to them.
            for include_derivatives, tag in ((False, ''), (True, '_derivatives')):
                fig = plots.plot_individual_trajectories(
                    module_norms, cfg.MODULE_TITLES, timesteps, out_dir, model_name,
                    include_derivatives=include_derivatives,
                    pars=test_data['pars'])
                plots.save_figure(
                    fig, out_dir,
                    f"{model_name}_hidden_activity_trajectories{tag}.png")

                fig = plots.plot_averaged_activity(
                    module_norms, cfg.MODULE_TITLES, timesteps, out_dir, model_name,
                    n_samples=N_SAMPLES_HIDDEN,
                    include_derivatives=include_derivatives)
                plots.save_figure(
                    fig, out_dir,
                    f"{model_name}_hidden_activity_averaged{tag}.png")

            diff_metrics = compute_sample_difference_all_modules(module_norms, 0, 1)
            indep_metrics = compute_module_independence(module_norms)
            print(f"  redundancy {indep_metrics['redundancy']:.4f}; "
                  f"independence {indep_metrics['independence_score']:.4f}; "
                  f"effective dims {indep_metrics['effective_dimensions']:.2f}"
                  f"/{len(cfg.MODULE_TITLES)}")
            write_independence_metrics(out_dir / f"{model_name}_metrics.txt",
                                       model_name, diff_metrics, indep_metrics,
                                       cfg.MODULE_TITLES)
