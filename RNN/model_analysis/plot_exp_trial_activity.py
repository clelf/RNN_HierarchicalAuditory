from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import evaluate_models as eval

from model_activations import (
    get_module_output_and_activity,
    load_trial_sequence,
    load_trial_params,
    to_model_tensors,
    plot_individual_trajectories,
    plot_averaged_activity,
)
import torch


if __name__ == '__main__':

    # =============================================================================
    # Configuration
    # =============================================================================

    # Number of trial sequences to use (only applies to Option B random sampling below)
    N_sequences = 2

    # model_name = "population_network_all_bn8_lr0"
    model_name = "population_network_all_bn8_lr0.001_dposweight"
    model_dir = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/training_results/N_ctx_2/HierarchicalGM")
    model_path = model_dir / model_name

    trials_path = Path("/home/clevyfidel/Documents/Workspace/Jasmin/trialsequences2clem")   

    output_dir = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output") / model_name / "viz_examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # Load model
    # =============================================================================
    info = eval.ModelInfo.from_path(model_path)
    model = eval.load_model(info)
    model.eval()
    print(f"Loaded model: {model_name}")

    # If the model was trained with a larger cue vocabulary than the two cues present in
    # the experimental sequences, re-encode the cues into the dimensionality it expects.
    n_cue_classes = len(info.data_config_dict['cues_set'])

    # =============================================================================
    # Select trial files
    # =============================================================================
    all_files = sorted(trials_path.glob("*.csv"))
    if not all_files:
        all_files = sorted(trials_path.glob("*.txt"))

    # Option A: specify files by name
    # selected_files = [trials_path / name for name in [
    #     "sequence_001.csv",
    #     "sequence_002.csv",
    # ]]

    # Option B: sample N_sequences at random
    rng = np.random.default_rng(seed=0)
    selected_files = rng.choice(all_files, size=N_sequences, replace=False).tolist()

    n_select = len(selected_files)
    print(f"Using {n_select} trial sequence files")

    # =============================================================================
    # Load sequences and build a batched tensor
    # Sequences may differ in length; we truncate all to the shortest one.
    # =============================================================================
    obs_list, cue_list, params_list = [], [], []
    for f in selected_files:
        obs, cue, lim_std, d, tau_std, trial_n = load_trial_sequence(
            f, n_cue_classes=n_cue_classes, cue_seed=11)
        obs_list.append(obs)
        cue_list.append(cue)
        params_list.append(load_trial_params(f))

    # Build pars dict in the format expected by extract_sample_parameters
    pars = {
        'tau':     [p['tau']     for p in params_list],
        'lim':     [p['lim']     for p in params_list],
        'si_stat': [p['si_stat'] for p in params_list],
        'si_r':    [p['si_r']    for p in params_list],
    }

    min_len = min(o.shape[0] for o in obs_list)
    if any(o.shape[0] != min_len for o in obs_list):
        print(f"  Warning: sequences have unequal lengths — truncating all to {min_len} timesteps")

    obs_stack = np.stack([o[:min_len] for o in obs_list], axis=0)        # (N, T)
    cue_stack = np.stack([c[:min_len, :] for c in cue_list], axis=0)     # (N, T, n_cue)

    y = torch.tensor(obs_stack, dtype=torch.float32).unsqueeze(-1)       # (N, T, 1)
    q = torch.tensor(cue_stack, dtype=torch.float32)                     # (N, T, n_cue)

    # =============================================================================
    # Run forward pass — returns norms (T-1, N) and derivatives (T-2, N)
    # =============================================================================
    prob_output, module_norms_dict, module_derivatives_dict = get_module_output_and_activity(model, y, q)

    module_titles = {
        'obs':  'Observation module',
        'ctx':  'Type module',
        'dpos': 'Deviant position module',
        'rule': 'Rule module',
    }

    seq_len = next(iter(module_norms_dict.values())).shape[0]
    timesteps = np.arange(seq_len)

    # =============================================================================
    # Generate the 4 figures
    # =============================================================================

    # Figure 1: Individual trajectories (activity)
    fig = plot_individual_trajectories(
        module_norms_dict, module_titles, timesteps,
        output_dir, model_name, include_derivatives=False, pars=pars,
    )
    save_path = output_dir / f"{model_name}_exp_activity_trajectories.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path.name}")
    plt.close()

    # Figure 2: Averaged activity
    fig = plot_averaged_activity(
        module_norms_dict, module_titles, timesteps,
        output_dir, model_name, n_samples=n_select, include_derivatives=False,
    )
    save_path = output_dir / f"{model_name}_exp_activity_averaged.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path.name}")
    plt.close()

    # Figure 3: Individual trajectories (derivatives)
    fig = plot_individual_trajectories(
        module_norms_dict, module_titles, timesteps,
        output_dir, model_name, include_derivatives=True, pars=pars,
    )
    save_path = output_dir / f"{model_name}_exp_activity_trajectories_derivatives.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path.name}")
    plt.close()

    # Figure 4: Averaged derivatives
    fig = plot_averaged_activity(
        module_norms_dict, module_titles, timesteps,
        output_dir, model_name, n_samples=n_select, include_derivatives=True,
    )
    save_path = output_dir / f"{model_name}_exp_activity_averaged_derivatives.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path.name}")
    plt.close()

    print(f"\nAll 4 figures saved to {output_dir}")
