"""
Script for computing ONLY testing benchmark datasets.

This script computes Kalman filter benchmarks for testing data only (no training benchmarks).
Useful for preparing test datasets before training, or when you want to evaluate models
against pre-computed KF baselines.

Computes:
- N_ctx=2 (NonHierarchicalGM): Multi-context test benchmarks
- N_ctx=1 (NonHierarchicalGM): Single-context test benchmarks

Usage:
    python pipeline_benchmarks_only.py
    python pipeline_benchmarks_only.py --unit_test
"""

import torch
import os
import argparse
import numpy as np
from pipeline_next import load_or_compute_benchmarks
from config import get_base_model_config, get_data_config, get_module_network_config



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Compute Kalman filter benchmarks only')
    parser.add_argument('--unit_test', action='store_true', help='Run in unit test mode')
    args = parser.parse_args()

    unit_test = args.unit_test

    # =========================================================================
    # N_ctx=2, NonHierarchicalGM (for ModuleNetwork)
    # =========================================================================
    print("\n" + "="*60)
    print("Computing TEST benchmarks for N_ctx=2 (NonHierarchicalGM)")
    print("="*60 + "\n")

    # BASE MODULES CONFIG
    model_config_2 = get_base_model_config(unit_test=unit_test)
    model_config_2["model"] = ['module_network']
  
    # DATA CONFIG
    N_ctx = 2
    gm_name = 'NonHierarchicalGM'
    
    data_config_2 = get_data_config(
        model_config_2, 
        gm_name=gm_name, 
        N_ctx=N_ctx, 
        params_testing=True, 
        unit_test=unit_test
    )

    # COMPUTE TESTING BENCHMARKS ONLY 
    _, benchmarks_test_2 = load_or_compute_benchmarks(
        data_config=data_config_2,
        model_config=model_config_2,
        N_ctx=N_ctx,
        gm_name=gm_name,
        visualize=True,
        max_cores=data_config_2.get("max_cores", None),
        benchmark_mode='test_only',  # Only compute test benchmarks
    )
    
    print(f"✓ N_ctx=2 test benchmarks computed: {benchmarks_test_2['y'].shape[0]} samples")


    # =========================================================================
    # N_ctx=1 (for SimpleRNN/VRNN)
    # =========================================================================
    print("\n" + "="*60)
    print("Computing TEST benchmarks for N_ctx=1 (NonHierarchicalGM)")
    print("="*60 + "\n")

    # BASE MODEL CONFIG
    model_config_1 = get_base_model_config(unit_test=unit_test)
    model_config_1["model"] = ['rnn', 'vrnn']
  
    # DATA CONFIG
    N_ctx = 1
    gm_name = 'NonHierarchicalGM'
    
    data_config_1 = get_data_config(
        model_config_1, 
        gm_name=gm_name, 
        N_ctx=N_ctx, 
        params_testing=True, 
        unit_test=unit_test
    )

    # COMPUTE TESTING BENCHMARKS ONLY 
    _, benchmarks_test_1 = load_or_compute_benchmarks(
        data_config=data_config_1,
        model_config=model_config_1,
        N_ctx=N_ctx,
        gm_name=gm_name,
        visualize=True,
        max_cores=data_config_1.get("max_cores", None),
        benchmark_mode='test_only',  # Only compute test benchmarks
    )
    
    print(f"✓ N_ctx=1 test benchmarks computed: {benchmarks_test_1['y'].shape[0]} samples")


    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("BENCHMARK COMPUTATION COMPLETE")
    print("="*60)
    print(f"  N_ctx=1: {benchmarks_test_1['y'].shape[0]} test samples")
    print(f"  N_ctx=2: {benchmarks_test_2['y'].shape[0]} test samples")
    print("="*60)