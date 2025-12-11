import torch
import os
import numpy as np
from pipeline_next import pipeline_model


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FREQ_MIN = 1400
FREQ_MAX = 1650

if __name__=='__main__':

    unit_test = False

    # DEFINE MODEL AND TRAINING PARAMETERS
    model_config = {
        "use_minmax": False,  # whether to use min-max normalization
        
        # Define models to train
        "model": ['rnn', 'vrnn'],  #: 'rnn', 'vrnn'

        # Learning rates to test
        "learning_rates":  [0.05, 0.01, 0.005, 0.001, 0.0005] if not unit_test else [0.01, 0.005],  # list of learning rates to try # [0.01, 0.005], #

        # Input and output dimensions
        "input_dim": 1,  # number of features in each observation: 1 --> need to write y_norm = y_norm.unsqueeze(-1) at one point
        "output_dim": 2,  # learn the sufficient statistics mu and var

        # RNN configuration
        "rnn_hidden_dim": [8, 16, 32, 64] if not unit_test else [8, 16],  # prev: 8  # [16], #
        "rnn_n_layers": 1,  # number of RNN layers

        # Training parameters
        "num_epochs": 100 if not unit_test else 10, # TODO: 250 / 100,  # number of epochs (TEST: 2)
        "epoch_res": 20 if not unit_test else 10,  # report results every epoch_res epochs
        "batch_res": 16 if not unit_test else 2,  # store and report loss every batch_res batches
        "batch_size": 1000 if not unit_test else 64, # 5, # TODO: 1000,  # batch size (TEST: 5)
        "n_batches": 32 if not unit_test else 2,  # number of batches # TODO: 32
        "weight_decay": 1e-5,  # weight decay for optimizer

        # Experiment parameters (non-hierarchical)
        "n_trials": 1000,  # single tones

        # Testing parameters
        "batch_size_test": 1000 if not unit_test else 64, # 5 # batch size during testing # TODO: 1000 (TEST: 10)

        # Visualization parameters
        "seq_len_viz": 125,  # or None
    }
    
    # DEFINE GENERATIVE MODEL PARAMETERS
    
    # Parallel processing configuration:
    # - SLURM: automatically uses allocated CPUs (SLURM_CPUS_PER_TASK)
    # - Local: uses half of available cores to keep machine responsive
    # - Set to 1 to disable parallelization
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        max_cores = int(slurm_cpus)  # Use all allocated SLURM CPUs
    else:
        from multiprocessing import cpu_count
        max_cores = max(1, cpu_count() // 2) if not unit_test else 1 # Use half of local CPUs
    
    data_config = {
        "gm_name": "NonHierarchicalGM",
        "N_ctx": 1,
        # "N_batch": n_batches,
        "N_samples": model_config['batch_size'],
        "N_blocks": 1,
        "N_tones": model_config['n_trials'],
        "mu_rho_ctx": 0.9,
        "si_rho_ctx": 0.05,
        "tones_values": [1455, 1500, 1600], # ~ lim
        "si_lim": 5,
        # "mu_tau": 4,
        "si_tau": 0.5,
        # "si_q": 2,  # process noise # Obsolete
        # "si_stat": 2,  # stationary processes variance
        # "si_r": 2,  # measurement noise
        "max_cores": max_cores,  # Parallel data generation

        # TESTING PARAMETERS
        "params_testing": True,
    }

    # MULTIPLE CONEXTS
    add_multi_context_params = {
        "si_d_coef": 0.05,
        "d_bounds": {"high": 4, "low": 0.1},
        "mu_d": 2
    }
    

    # HIERARCHICAL PARAMETERS
    add_hierarchical_params = {
        "N_blocks": 125,
        "N_tones": 8,
        "rules_dpos_set": np.array([[3, 4, 5], [5, 6, 7]]),
        "mu_rho_rules": 0.9,
        "si_rho_rules": 0.05,
    }
    
    
    # PARAMETERS TESTING
    add_data_params_baseline = {
        "si_lim": 5,
        "mu_tau_bounds": {'low': 1, 'high': 250},
        "si_stat_bounds": {'low': 0.1, 'high': 2},
        "si_r_bounds": {'low': 0.1, 'high': 2},  # measurement noise
    }
    if data_config["params_testing"]:
        data_config.update(add_data_params_baseline)

    
    # TRAINING MODEL WITH NON HIERARCHICAL GM AND SINGLE CONTEXT
    print("Running N_ctx = 1")
    pipeline_model(model_config, data_config)


    # #### RUNNING DIFFERENT GMS (OBSOLETE)

    # for N_ctx in [1, 2]:
    #     print("Running N_ctx =", N_ctx)
    #     data_config["N_ctx"] = N_ctx  # Update N_ctx in data_config
    #     if data_config["N_ctx"] > 1:
    #         data_config.update(add_multi_context_params)

    #     for gm_name in ['NonHierarchicalGM', 'HierarchicalGM']:
    #         if gm_name == 'HierarchicalGM' and N_ctx == 1:
    #             continue
            
    #         print("Running GM = ", gm_name)
    #         data_config['gm_name'] = gm_name
            
    #         if data_config["gm_name"] == "HierarchicalGM":
    #             data_config.update(add_hierarchical_params)
            
    #         pipeline_model(model_config, data_config)



