"""
Script for training ModuleNetwork with different learning objectives.

This script demonstrates how to use the extended pipeline to train a ModuleNetwork
that learns from both observations and context labels. The ModuleNetwork has two
sub-modules:
1. Observation module: Predicts the next observation (like SimpleRNN/VRNN)
2. Context module: Classifies the current context based on observation patterns

Learning objectives for multi-context scenarios (N_ctx > 1):
- 'obs': Train observation module only (hidden process prediction)
- 'ctx': Train context module only (context inference)  
- 'obs_and_ctx': Train both modules with combined loss (weighted by kappa)

Usage:
    python pipeline_module_network.py
"""

import torch
import os
import numpy as np
from pipeline_next import pipeline_train_valid, pipeline_test, VALID_LEARNING_OBJECTIVES
from config import get_base_model_config, get_data_config, get_module_network_config



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    
    unit_test = True  # Set to False for full training
    skip_benchmarks = True

    # BASE MODULES CONFIG
    
    # Get base model config (can also define manually as in pipeline_multi.py)
    model_config = get_base_model_config(unit_test=unit_test)
    
    # Override model list to include 'module_network'
    model_config["model"] = ['module_network']  # Options: 'rnn', 'vrnn', 'module_network'
    
    # Note: ModuleNetwork uses fixed hidden dims set in module_config below,
    # not the rnn_hidden_dims from model_config (no iteration over hidden dims)

    
    # DATA CONFIG
    N_ctx = 2  # Number of contexts (must be > 1 for ModuleNetwork)
    gm_name = 'NonHierarchicalGM'  # or 'HierarchicalGM'
    
    data_config = get_data_config(
        model_config, 
        gm_name=gm_name, 
        N_ctx=N_ctx, 
        params_testing=True, 
        unit_test=unit_test
    )

    # WHOLE MODEL CONFIG
    # ModuleNetwork hidden dims are set here directly (not from rnn_hidden_dims)
    # - observation_module: default 64 (or customize below)
    # - context_module: default 32 (or customize below)
    whole_model_config = get_module_network_config(model_config, data_config)


    # TRAIN ALL VARIANTS
    # Train 3 different model variants with different learning objectives:
    # - 'obs': observation module only (hidden process prediction)
    # - 'ctx': context module only (context inference)
    # - 'obs_and_ctx': both modules (combined loss)
    
    print("\n" + "="*60)
    print("Training ModuleNetwork with all learning objectives")
    print(f"N_ctx={N_ctx}, gm_name={gm_name}")
    print(f"Learning objectives: {VALID_LEARNING_OBJECTIVES}")
    print("="*60 + "\n")
    
    # Key parameters:
    # - data_mode='multi_ctx': Use multi-context data (N_ctx > 1)
    # - learning_objective: Can be a single value or a list to train all variants
    # - kappa: Balance between observation loss (kappa=1.0) and context loss (kappa=0.0)
    #          Default kappa=0.5 gives equal weight to both (only used for 'obs_ctx')
    
    # When learning_objective is a list, pipeline_multi_config iterates over each objective
    # and trains a separate model for each, saved to different directories:
    #   - training_results/N_ctx_2/<gm_name>/module_network_obs/
    #   - training_results/N_ctx_2/<gm_name>/module_network_ctx/
    #   - training_results/N_ctx_2/<gm_name>/module_network_obs_ctx/
    
    pipeline_train_valid(
        whole_model_config, 
        data_config, 
        data_mode='multi_ctx',  # Multi-context data mode
        learning_objective=VALID_LEARNING_OBJECTIVES,  # Train all 3 variants: ['obs', 'ctx', 'obs_ctx']
        skip_benchmarks=skip_benchmarks,  # Skip slow KF benchmark computation; set False to include KF comparison
    )
    
    # TEST ALL VARIANTS
    
    print("\n" + "="*60)
    print("Testing all trained ModuleNetwork variants")
    print("="*60 + "\n")
    
    pipeline_test(
        whole_model_config, 
        data_config,
        data_mode='multi_ctx',
        learning_objective=VALID_LEARNING_OBJECTIVES,  # Test all 3 variants
        skip_benchmarks=skip_benchmarks,  # Match training setting
    )
