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
    python pipeline_module_network.py --unit_test --skip_benchmarks
    python pipeline_module_network.py --use_run_id
    python pipeline_module_network.py --use_run_id my_experiment
    python pipeline_module_network.py --bottleneck_dims 2 4 8 16
"""

import torch
import os
import argparse
from datetime import datetime
import numpy as np
from pipeline_next import pipeline_train_valid, pipeline_test, VALID_LEARNING_OBJECTIVES
from config import get_base_model_config, get_data_config, get_module_network_config


# Default bottleneck dimensions to explore
DEFAULT_BOTTLENECK_DIMS = [2, 4, 8, 16]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train ModuleNetwork models')
    parser.add_argument('--unit_test', action='store_true', help='Run in unit test mode')
    parser.add_argument('--skip_benchmarks', action='store_true', help='Skip benchmark computation')
    parser.add_argument('--use_run_id', type=str, nargs='?', const='', default=None,
                        help='Create a run subfolder. If name provided, uses that; otherwise uses timestamp.')
    parser.add_argument('--bottleneck_dims', type=int, nargs='+', default=DEFAULT_BOTTLENECK_DIMS,
                        help='Bottleneck dimensions to train (default: 2 4 8 16)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Fixed learning rate (default: 0.01)')
    args = parser.parse_args()

    unit_test = args.unit_test
    skip_benchmarks = args.skip_benchmarks
    bottleneck_dims = args.bottleneck_dims
    learning_rate = args.learning_rate
    
    # Determine run_id: None by default, timestamp if --use_run_id alone, or custom value if --use_run_id <name>
    if args.use_run_id is not None:
        run_id = args.use_run_id if args.use_run_id else datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        run_id = None

    # BASE MODULES CONFIG
    
    # Get base model config (can also define manually as in pipeline_multi.py)
    model_config = get_base_model_config(unit_test=unit_test)
    
    # Override model list to include 'module_network'
    model_config["model"] = ['module_network']  # Options: 'rnn', 'vrnn', 'module_network'
    
    # Fix learning rate to specified value (default: 0.01)
    model_config["learning_rates"] = [learning_rate]
    
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
    # Pass bottleneck_dims as a list to iterate over different values
    whole_model_config = get_module_network_config(model_config, data_config, bottleneck_dim=bottleneck_dims)


    # TRAIN ALL VARIANTS
    # Train different model variants with different learning objectives and bottleneck dimensions:
    # - Learning objectives: 'obs', 'ctx', 'obs_ctx'
    # - Bottleneck dimensions: specified via --bottleneck_dims (default: 2, 4, 8, 16)
    
    print("\n" + "="*60)
    print("Training ModuleNetwork with all learning objectives")
    print(f"N_ctx={N_ctx}, gm_name={gm_name}")
    print(f"Learning rate: {learning_rate}")
    print(f"Bottleneck dimensions: {bottleneck_dims}")
    print(f"Learning objectives: {VALID_LEARNING_OBJECTIVES}")
    if run_id:
        print(f"Run ID: {run_id}")
    print("="*60 + "\n")
    
    # Key parameters:
    # - data_mode='multi_ctx': Use multi-context data (N_ctx > 1)
    # - learning_objective: Can be a single value or a list to train all variants
    # - kappa: Balance between observation loss (kappa=1.0) and context loss (kappa=0.0)
    #          Default kappa=0.5 gives equal weight to both (only used for 'obs_ctx')
    # - bottleneck_dims: List of bottleneck dimensions to iterate over
    
    # When learning_objective is a list, pipeline_multi_config iterates over each objective
    # and trains a separate model for each. Combined with bottleneck_dims, results are saved to:
    #   - training_results/<run_id>/N_ctx_2/<gm_name>/module_network_obs_bn2/
    #   - training_results/<run_id>/N_ctx_2/<gm_name>/module_network_obs_bn4/
    #   - training_results/<run_id>/N_ctx_2/<gm_name>/module_network_ctx_bn2/
    #   - training_results/<run_id>/N_ctx_2/<gm_name>/module_network_obs_ctx_kappa0.5_bn2/
    #   - ... etc.
    
    pipeline_train_valid(
        whole_model_config, 
        data_config, 
        data_mode='multi_ctx',  # Multi-context data mode
        learning_objective=VALID_LEARNING_OBJECTIVES,  # Train all 3 variants: ['obs', 'ctx', 'obs_ctx']
        skip_benchmarks=skip_benchmarks,  # Skip slow KF benchmark computation; set False to include KF comparison
        run_id=run_id,
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
        run_id=run_id,
    )
