import os
import numpy as np
from multiprocessing import cpu_count

def get_model_config(unit_test=False, training_phase=True):
    if training_phase:
        lr_vals = [0.05, 0.01, 0.005, 0.001, 0.0005]
        hu_vals = [8, 16, 32, 64]
        lr_vals_test = [0.01, 0.005]
        hu_vals_test = [8, 16]
    else:
        lr_vals = [0.05]
        hu_vals = [16]
        lr_vals_test = [0.05]
        hu_vals_test = [16]

    return {
        "use_minmax": False,
        "model": ['rnn'], # , 'vrnn'
        "learning_rates": lr_vals if not unit_test else lr_vals_test,
        "input_dim": 1,
        "output_dim": 2,
        "rnn_hidden_dim": hu_vals if not unit_test else hu_vals_test,
        "rnn_n_layers": 1,
        "num_epochs": 100 if not unit_test else 10,
        "epoch_res": 20 if not unit_test else 10,
        "batch_res": 16 if not unit_test else 2,
        "batch_size": 1000 if not unit_test else 64,
        "n_batches": 32 if not unit_test else 2,
        "weight_decay": 1e-5,
        "n_trials": 1000,
        "batch_size_test": 1000 if not unit_test else 64,
        "seq_len_viz": 125,
    }

def get_context_module_config():
    return {
        "input_dim": 2,
        "output_dim": 2,
        "rnn_hidden_dim": 16,
        "rnn_n_layers": 1,
        "num_epochs": 100,
        "epoch_res": 20,
        "batch_res": 16,
        "batch_size": 1000,
        "n_batches": 32,
        "weight_decay": 1e-5,
        "n_trials": 1000,
        "batch_size_test": 1000,
        "seq_len_viz": 125,
    }

def get_data_config(model_config, gm_name, N_ctx, params_testing=True, unit_test=False):
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        max_cores = int(slurm_cpus)
    else:
        max_cores = max(1, cpu_count() // 2) if not unit_test else 1
    data_config = {
        "gm_name": gm_name,
        "N_ctx": N_ctx,
        "N_samples": model_config['batch_size'],
        "N_blocks": 1,
        "N_tones": model_config['n_trials'],
        "mu_rho_ctx": 0.9,
        "si_rho_ctx": 0.05,
        "si_lim": 5,
        "si_tau": 0.5,
        "max_cores": max_cores,
        "params_testing": params_testing,
    }
    
    # Add baseline testing params if needed
    if data_config["params_testing"]:
        data_config.update(get_add_data_params_baseline())

    # Add multi-context params if needed
    if data_config["N_ctx"] > 1:
        data_config.update(get_add_multi_context_params())
    
    # Add hierarchical GM params if needed
    if data_config["gm_name"] == "HierarchicalGM":
        data_config.update(get_add_hierarchical_params())
    return data_config

def get_add_multi_context_params():
    return {
        "si_d_coef": 0.05,
        "d_bounds": {"high": 4, "low": 0.1},
        "mu_d": 2
    }

def get_add_hierarchical_params(model_config):
    hierarch_config = {
        "N_tones": 8,
        "rules_dpos_set": np.array([[3, 4, 5], [5, 6, 7]]),
        "mu_rho_rules": 0.9,
        "si_rho_rules": 0.05,
    }
    hierarch_config["N_blocks"] = model_config["N_trials"] / hierarch_config["N_tones"]
    return hierarch_config

def get_add_data_params_baseline():
    return {
        "si_lim": 5,
        "mu_tau_bounds": {'low': 1, 'high': 250},
        "si_stat_bounds": {'low': 0.1, 'high': 2},
        "si_r_bounds": {'low': 0.1, 'high': 2},
    }
