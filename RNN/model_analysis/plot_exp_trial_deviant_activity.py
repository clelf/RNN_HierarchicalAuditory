from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import evaluate_models as eval

from model_activations import (
    get_module_output_and_activity,
    load_trial_sequence,
    plot_deviant_activity_by_position,
)
import torch


def compute_norms_and_devpos_for_files(model, files, period=8, chunk_size=128):
    """Run the model over many sequence files; return per-module norms and deviant positions.

    Mirrors ``compute_norms_for_files`` (chunked forward passes to bound memory)
    but also collects, for every sequence, the 0-based within-trial deviant
    position of each trial. ``dpos`` is constant within a trial, so one value per
    trial is taken (``dpos[::period]``).

    Returns
    -------
    module_norms_dict : dict
        Module name → ndarray of shape (seq_len, n_files).
    dev_pos : np.ndarray
        Shape (n_files, n_trials) — 0-based within-trial deviant position per trial.
    """
    norm_chunks = {}
    devpos_list = []
    for start in range(0, len(files), chunk_size):
        chunk = files[start:start + chunk_size]

        obs_list, cue_list = [], []
        for f in chunk:
            obs, cue, ctx, dpos, rule, lim_std, d, tau_std, trial_n = \
                load_trial_sequence(f, return_hierarch=True)
            obs_list.append(obs)
            cue_list.append(cue)
            # RAW physical within-trial position {2..6}: kept raw because it indexes
            # the deviant's actual timestep (t*period + pos) in extract_deviant_activity.
            # This is about where the deviant physically sits, independent of any model,
            # so it must NOT be shifted to a model's dpos convention (that would point
            # one tone past the real deviant). The legend renders pos+1 (1-indexed).
            devpos_list.append(dpos[::period])   # one deviant position per trial

        min_len = min(o.shape[0] for o in obs_list)
        obs_stack = np.stack([o[:min_len] for o in obs_list], axis=0)        # (N, T)
        cue_stack = np.stack([c[:min_len, :] for c in cue_list], axis=0)     # (N, T, 2)

        y = torch.tensor(obs_stack, dtype=torch.float32).unsqueeze(-1)       # (N, T, 1)
        q = torch.tensor(cue_stack, dtype=torch.float32)                     # (N, T, 2)

        _, module_norms, _ = get_module_output_and_activity(model, y, q)
        for name, norms in module_norms.items():
            norm_chunks.setdefault(name, []).append(norms)                   # (seq_len, N_chunk)

        print(f"  processed {min(start + chunk_size, len(files))}/{len(files)} sequences")

    module_norms_dict = {name: np.concatenate(chunks, axis=1) for name, chunks in norm_chunks.items()}
    dev_pos = np.stack(devpos_list, axis=0)                                  # (n_files, n_trials)
    return module_norms_dict, dev_pos


if __name__ == '__main__':

    # =============================================================================
    # Configuration
    # =============================================================================

    # Timesteps per trial. The deviant of trial t sits at timestep t*PERIOD + dpos.
    PERIOD = 8

    # Set to an int to randomly sample that many sequences, or None to use all.
    N_sequences = None

    model_name = 'population_network_all_bn8_lr0'
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
    # Run forward passes (chunked); collect per-module norms and deviant positions
    # =============================================================================
    module_norms_dict, dev_pos = compute_norms_and_devpos_for_files(model, selected_files, period=PERIOD)

    module_titles = {
        'obs':  'Observation module',
        'ctx':  'Type module',
        'dpos': 'Deviant position module',
        'rule': 'Rule module',
    }

    # =============================================================================
    # Figures: activity / derivatives at the deviant position, grouped by dpos value
    # =============================================================================

    # Figure 1: Activity at the deviant position
    fig = plot_deviant_activity_by_position(
        module_norms_dict, dev_pos, module_titles, output_dir, model_name,
        n_samples=n_select, period=PERIOD, include_derivatives=False,
    )
    save_path = output_dir / f"{model_name}_exp_deviant_activity_by_position.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path.name}")
    plt.close()

    # Figure 2: Derivatives at the deviant position
    fig = plot_deviant_activity_by_position(
        module_norms_dict, dev_pos, module_titles, output_dir, model_name,
        n_samples=n_select, period=PERIOD, include_derivatives=True,
    )
    save_path = output_dir / f"{model_name}_exp_deviant_activity_by_position_derivatives.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path.name}")
    plt.close()

    print(f"\nAll figures saved to {output_dir}")
