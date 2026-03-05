import torch.nn as nn # import RNN, Module, Sequential, GRU, LSTM
import torch
from torch.nn import functional as F
from functools import partial
import numpy as np
from model import SimpleRNN, VRNN
from objectives import Objective
from config import get_base_model_config, get_data_config

from model import ModuleNetwork

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from PreProParadigm.audit_gm import NonHierachicalAuditGM, HierarchicalAuditGM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__=='__main__':
    unit_test = False

    # Training config
    train_config = {# Learning rates to test
        "learning_rates":  [0.05],

        # Training parameters
        "num_epochs": 100 if not unit_test else 10, # TODO: 250 / 100,  # number of epochs (TEST: 2)
        "epoch_res": 20 if not unit_test else 10,  # report results every epoch_res epochs
        "batch_res": 16 if not unit_test else 2,  # store and report loss every batch_res batches
        "batch_size": 1000 if not unit_test else 64, # 5, # TODO: 1000,  # batch size (TEST: 5)
        "n_batches": 32 if not unit_test else 2,  # number of batches # TODO: 32
        "weight_decay": 1e-5,  # weight decay for optimizer

        # Experiment parameters (non-hierarchical)
        "n_trials": 1000
    }

    # Generate data (N_samples samples)
    data_config = get_data_config(train_config, gm_name='NonHierarchicalGM', N_ctx=2, unit_test=unit_test)

    if data_config['gm_name'] == 'NonHierarchicalGM':
        gm = NonHierachicalAuditGM(data_config)
        ctx, _, y = gm.generate_batch(N_samples=10, return_pars=False)
    elif data_config['gm_name'] == 'HierarchicalGM':
        gm = HierarchicalAuditGM(data_config)
        _, _, _, _, _, ctx, _, y = gm.generate_batch(N_samples=10, return_pars=False)

    y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(DEVICE)

    # Model config
    model_obs_config = {
        # Input and output dimensions
        "input_dim": 1,  # number of features in each observation: 1 --> need to write y = y.unsqueeze(-1) at one point
        "output_dim": 2,  # learn the sufficient statistics mu and var

        # RNN configuration
        "rnn_hidden_dim": 64,
        "rnn_n_layers": 1,  # number of RNN layers

        # Bottleneck dimension
        "bottleneck_dim": 24,
    }

    model_ctx_config = {
        # Input and output dimensions
        "input_dim": 2, 
        "output_dim": data_config['N_ctx'],  # number of contexts to learn

        # RNN configuration
        "rnn_hidden_dim": 32,
        "rnn_n_layers": 1,  # number of RNN layers

        # Bottleneck dimension
        "bottleneck_dim": model_obs_config['bottleneck_dim'],  # match observation module
    }

    model_config = {"observation_module": model_obs_config, "context_module": model_ctx_config, **train_config}
    

    RNN2modules = ModuleNetwork(model_config)

    # Always train to predict the next observation --> Input: y[0:T-1], Target: y[1:T]
    model_output_oneloop = RNN2modules(y[:, :-1, :])

    pass
    
