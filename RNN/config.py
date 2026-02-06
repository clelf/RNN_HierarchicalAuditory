import os
import numpy as np
from multiprocessing import cpu_count

def get_base_model_config(unit_test=False, training_phase=True):
    if training_phase:
        lr_vals = [0.05, 0.01, 0.005, 0.001, 0.0005]
        hu_vals = [8, 16, 32, 64]
        lr_vals_test = [0.01, 0.005]
        hu_vals_test = [8, 16]
    else:
        lr_vals = [0.05]
        hu_vals = 16
        lr_vals_test = [0.05]
        hu_vals_test = 16

    return {
        "use_minmax": False,
        "model": ['rnn'], # , 'vrnn'
        "learning_rates": lr_vals if not unit_test else lr_vals_test,
        "input_dim": 1,
        "output_dim": 2,
        "rnn_hidden_dims": hu_vals if not unit_test else hu_vals_test,  # scalar or list of hidden dims to try
        "rnn_n_layers": 1,
        "num_epochs": 100 if not unit_test else 10,
        "epoch_res": 20 if not unit_test else 10,
        "batch_res": 16 if not unit_test else 2,
        "batch_size": 1000 if not unit_test else 8,
        "n_batches": 32 if not unit_test else 2,
        "weight_decay": 1e-5,
        "n_trials": 1000,
        "batch_size_test": 1000 if not unit_test else 8,
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
        data_config.update(get_add_hierarchical_params(model_config))
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
    hierarch_config["N_blocks"] = int(model_config["n_trials"] / hierarch_config["N_tones"])
    return hierarch_config


def get_add_data_params_baseline():
    return {
        "si_lim": 5,
        "mu_tau_bounds": {'low': 1, 'high': 250},
        "si_stat_bounds": {'low': 0.1, 'high': 2},
        "si_r_bounds": {'low': 0.1, 'high': 2},
    }


def get_module_network_config(model_config, data_config, bottleneck_dim=24):
    """
    Build configuration for ModuleNetwork that learns from both observations and contexts.
    
    This creates a config with two sub-modules:
    - observation_module: Processes observations, outputs (mu, var) for next observation prediction
    - context_module: Processes compressed observation outputs to infer context
    
    Parameters
    ----------
    model_config : dict
        Base model configuration (learning rates, epochs, etc.)
    data_config : dict  
        Data configuration containing N_ctx (number of contexts)
    bottleneck_dim : int
        Dimension of the bottleneck layer between modules (default: 24)
    
    Returns
    -------
    dict
        Configuration dictionary with 'observation_module' and 'context_module' sub-configs
    """
    
    observation_module_config = {
        "input_dim": model_config.get('input_dim', 1),
        "output_dim": model_config.get('output_dim', 2),  # (mu, var)
        "rnn_hidden_dim": 64, # or model_config.get('rnn_hidden_dim', 64) model # Will be overridden per-run by h_dim in pipeline_single_config # TODO: why and where overriden?
        "rnn_n_layers": model_config.get('rnn_n_layers', 1),
        "bottleneck_dim": bottleneck_dim,
    }
    
    context_module_config = {
        "input_dim": 2,  # Receives compressed observation output (typically 2D)
        "output_dim": data_config.get('N_ctx', 2),  # Number of contexts to classify
        "rnn_hidden_dim": 32, # or model_config.get('rnn_hidden_dim', 32)  # Will be overridden per-run by h_dim in pipeline_single_config
        "rnn_n_layers": model_config.get('rnn_n_layers', 1),
        "bottleneck_dim": bottleneck_dim,
    }
    
    return {
        "kappa": 0.5,  # Balance between observation and context losses # TODO: study best value
        "observation_module": observation_module_config,
        "context_module": context_module_config,
        **{k: v for k, v in model_config.items() if k not in ['input_dim', 'output_dim', 'rnn_hidden_dims', 'rnn_n_layers']},
    }
