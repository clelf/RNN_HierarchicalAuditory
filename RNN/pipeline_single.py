import torch
from pipeline_next import pipeline_single_param


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FREQ_MIN = 1400
FREQ_MAX = 1650

if __name__=='__main__':


    # Define model and training parameters
    model_config = {
        # Input and output dimensions
        "input_dim": 1,  # number of features in each observation: 1 --> need to write y_norm = y_norm.unsqueeze(-1) at one point
        "output_dim": 2,  # learn the sufficient statistics mu and var

        # RNN configuration
        "rnn_hidden_dim": 8,  # comparable number of units as number of data parameters
        "rnn_n_layers": 1,  # number of RNN layers

        # VRNN specific configuration
        "latent_dim": 8,  # needs to be < input_dim + rnn_hidden_dim
        "phi_x_dim": 8,  # same as rnn_hidden_dim
        "phi_z_dim": 8,  # same as rnn_hidden_dim
        "phi_prior_dim": 8,  # same as rnn_hidden_dim

        # Training parameters
        "num_epochs": 2, # TODO: 250,  # number of epochs
        "epoch_res": 10,  # report results every epoch_res epochs
        "batch_res": 10,  # store and report loss every batch_res batches
        "batch_size": 4, # TODO: 1000,  # batch size
        "n_batches": 32,  # number of batches
        "weight_decay": 1e-5,  # weight decay for optimizer

        # Experiment parameters (non-hierarchical)
        "n_trials": 1000,  # single tones
    }
    
    # Define data parameters
    gm_name = 'NonHierarchicalGM'
    config_NH = {
        "N_ctx": 1,
        # "N_batch": n_batches,
        "N_samples": model_config['batch_size'],
        "N_blocks": 1,
        "N_tones": model_config['n_trials'],
        "mu_rho_ctx": 0.9,
        "si_rho_ctx": 0.05,
        # "tones_values": [1455, 1500, 1600], # ~ lim
        "si_lim": 5,
        # "mu_tau": 4,
        "si_tau": 1,
        # "si_q": 2,  # process noise # Obsolete
        # "si_stat": 2,  # stationary processes variance
        "si_r": 2,  # measurement noise
    }

    # Train model on one parameter configuration
    add_data_params_baseline = {
        "tones_values": [1455, 1500, 1600], # ~ lim
        "mu_tau": 64, # before used to be 4. Now 64 to describe a representative log-normal distribution
        # "si_q": 2,  # process noise # Obsolete
        "si_stat": 2  # stationary processes variance
    }

    config_NH.update(add_data_params_baseline)

    pipeline_single_param(model_config, config_NH, gm_name)
