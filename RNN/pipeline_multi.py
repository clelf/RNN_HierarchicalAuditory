import torch
import os
from pipeline_next import pipeline_multi_param


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FREQ_MIN = 1400
FREQ_MAX = 1650

if __name__=='__main__':

    unit_test = False

    # Define model and training parameters
    model_config = {
        "use_minmax": False,  # whether to use min-max normalization

        # Define model's objectives to test
        "kalman_on": [True], # [True, False], depends on number of contexts
        
        # Define models to train
        "model": ['rnn'],  #: 'rnn', 'vrnn'

        # Learning rates to test
        "learning_rates": [0.05, 0.01, 0.005, 0.001, 0.0005] if not unit_test else [0.01, 0.005],  # list of learning rates to try

        # Input and output dimensions
        "input_dim": 1,  # number of features in each observation: 1 --> need to write y_norm = y_norm.unsqueeze(-1) at one point
        "output_dim": 2,  # learn the sufficient statistics mu and var

        # RNN configuration
        "rnn_hidden_dim": [8, 16, 32, 64] if not unit_test else [8, 16],  # prev: 8
        "rnn_n_layers": 1,  # number of RNN layers

        # Training parameters
        "num_epochs": 250 if not unit_test else 10, # TODO: 250,  # number of epochs (TEST: 2)
        "epoch_res": 10,  # report results every epoch_res epochs
        "batch_res": 10,  # store and report loss every batch_res batches
        "batch_size": 1000 if not unit_test else 5, # TODO: 1000,  # batch size (TEST: 5)
        "n_batches": 32 if not unit_test else 2,  # number of batches # TODO: 32
        "weight_decay": 1e-5,  # weight decay for optimizer

        # Experiment parameters (non-hierarchical)
        "n_trials": 1000,  # single tones

        # Testing parameters
        "batch_size_test": 1000 if not unit_test else 5 # batch size during testing # TODO: 1000 (TEST: 10)
    }

    vrnn_specific ={
        # VRNN specific configuration
        "latent_dim": model_config["rnn_hidden_dim"],  # needs to be < input_dim + rnn_hidden_dim
        "phi_x_dim": model_config["rnn_hidden_dim"],  # same as rnn_hidden_dim
        "phi_z_dim": model_config["rnn_hidden_dim"],  # same as rnn_hidden_dim
        "phi_prior_dim": model_config["rnn_hidden_dim"],  # same as rnn_hidden_dim
    }
    model_config.update(vrnn_specific)
    
    # Define data parameters
    gm_name = 'NonHierarchicalGM'
    
    # Parallel processing configuration:
    # - SLURM: automatically uses allocated CPUs (SLURM_CPUS_PER_TASK)
    # - Local: uses half of available cores to keep machine responsive
    # - Set to 1 to disable parallelization
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        max_cores = int(slurm_cpus)  # Use all allocated SLURM CPUs
    else:
        from multiprocessing import cpu_count
        max_cores = max(1, cpu_count() // 2)  # Use half of local CPUs
    
    config_NH = {
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
    }

    add_data_params_baseline = {
        "params_testing": True,
        "si_lim": 5,
        "mu_tau_bounds": {'low': 1, 'high': 250},
        "si_stat_bounds": {'low': 0.1, 'high': 2},
        "si_r_bounds": {'low': 0.1, 'high': 2},  # measurement noise
    }


    config_NH.update(add_data_params_baseline)

    
    # ========================================
    # STEP-BY-STEP WORKFLOW
    # ========================================
    
    # STEP 1: Pre-compute benchmarks and visualize parameter distributions
    # This computes Kalman filter estimates on training and test data, and saves:
    #   - benchmarks/benchmarks_<N>_<GM>_train.pkl
    #   - benchmarks/benchmarks_<N>_<GM>_test.pkl
    #   - benchmarks/visualizations/param_distribution_*.png
    #   - benchmarks/visualizations/binned_metrics_kalman.csv
    
    print("\n" + "="*60)
    print("STEP 1: Computing benchmarks (Kalman filter baseline)")
    print("="*60)
    pipeline_multi_param(model_config, config_NH, gm_name, benchmark_only=True)
    
    
    # STEP 2: Train the RNN model with different learning rates and number of hidden units
    # This will train the model using the pre-computed benchmarks for validation
    # Uncomment the following line to run training:
    
    print("\n" + "="*60)
    print("STEP 2: Training RNN models")
    print("="*60)
    pipeline_multi_param(model_config, config_NH, gm_name, benchmark_only=False)



