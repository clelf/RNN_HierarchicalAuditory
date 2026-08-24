from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import evaluate_models as eval

from model_activations import (
    get_module_output_and_activity,
    load_trial_sequence,
    plot_averaged_activity_by_position,
)
import torch


def compute_norms_for_files(model, files, period=8, chunk_size=128,
                            n_cue_classes=None, cue_seed=None):
    """Run the model over many sequence files and return per-module norms.

    Sequences are stacked into batches of at most ``chunk_size`` so the hidden
    states of one forward pass stay bounded in memory; the (small) norm arrays
    are concatenated along the batch axis afterwards. All sequences are assumed
    to share the same length (they are here: 8 * n_trials timesteps).

    ``n_cue_classes`` / ``cue_seed`` are forwarded to ``load_trial_sequence`` so the
    two experimental cues can be re-encoded into the model's cue vocabulary (see
    that function). With a fixed ``cue_seed`` every sequence uses the same class pair,
    keeping the batched cue width uniform.

    Returns
    -------
    module_norms_dict : dict
        Module name → ndarray of shape (seq_len, n_files).
    """
    norm_chunks = {}
    for start in range(0, len(files), chunk_size):
        chunk = files[start:start + chunk_size]

        obs_list, cue_list = [], []
        for f in chunk:
            obs, cue, lim_std, d, tau_std, trial_n = load_trial_sequence(
                f, n_cue_classes=n_cue_classes, cue_seed=cue_seed)
            obs_list.append(obs)
            cue_list.append(cue)

        min_len = min(o.shape[0] for o in obs_list)
        obs_stack = np.stack([o[:min_len] for o in obs_list], axis=0)        # (N, T)
        cue_stack = np.stack([c[:min_len, :] for c in cue_list], axis=0)     # (N, T, n_cue)

        y = torch.tensor(obs_stack, dtype=torch.float32).unsqueeze(-1)       # (N, T, 1)
        q = torch.tensor(cue_stack, dtype=torch.float32)                     # (N, T, n_cue)

        _, module_norms, _ = get_module_output_and_activity(model, y, q)
        for name, norms in module_norms.items():
            norm_chunks.setdefault(name, []).append(norms)                   # (seq_len, N_chunk)

        print(f"  processed {min(start + chunk_size, len(files))}/{len(files)} sequences")

    return {name: np.concatenate(chunks, axis=1) for name, chunks in norm_chunks.items()}


if __name__ == '__main__':

    # =============================================================================
    # Configuration
    # =============================================================================

    # Timesteps per trial (number of within-trial positions). The within-trial
    # position of timestep t is t % PERIOD and its trial index is t // PERIOD.
    PERIOD = 8

    # Set to an int to randomly sample that many sequences, or None to use all.
    N_sequences = None

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

    if N_sequences is not None:
        rng = np.random.default_rng(seed=0)
        selected_files = rng.choice(all_files, size=N_sequences, replace=False).tolist()
    else:
        selected_files = all_files

    n_select = len(selected_files)
    print(f"Using {n_select} trial sequence files")

    # =============================================================================
    # Run forward passes (chunked) and collect per-module norms (seq_len, n_select)
    # =============================================================================
    module_norms_dict = compute_norms_for_files(
        model, selected_files, period=PERIOD, n_cue_classes=n_cue_classes, cue_seed=11)

    module_titles = {
        'obs':  'Observation module',
        'ctx':  'Type module',
        'dpos': 'Deviant position module',
        'rule': 'Rule module',
    }

    # =============================================================================
    # Figures: activity / derivatives averaged over sequences, split by position
    # =============================================================================

    # Figure 1: Averaged activity by within-trial position
    fig = plot_averaged_activity_by_position(
        module_norms_dict, module_titles, output_dir, model_name,
        n_samples=n_select, period=PERIOD, include_derivatives=False,
    )
    save_path = output_dir / f"{model_name}_exp_activity_averaged_by_position.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path.name}")
    plt.close()

    # Figure 2: Averaged derivatives by within-trial position
    fig = plot_averaged_activity_by_position(
        module_norms_dict, module_titles, output_dir, model_name,
        n_samples=n_select, period=PERIOD, include_derivatives=True,
    )
    save_path = output_dir / f"{model_name}_exp_activity_averaged_by_position_derivatives.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path.name}")
    plt.close()

    print(f"\nAll figures saved to {output_dir}")
