"""
Refactored pipeline runner - the clean entry point.

This replaces pipeline_multi.py and pipeline_module_network.py with a unified,
simple interface that:
1. Expands hyperparameter grid upfront (no nested loops in training)
2. Runs each configuration sequentially with clear progress
3. Provides sensible defaults with easy customization

Usage:
    # Train all default configurations
    python pipeline_runner_v2.py
    
    # Unit test mode
    python pipeline_runner_v2.py --unit_test
    
    # Skip slow benchmarks
    python pipeline_runner_v2.py --skip_benchmarks
    
    # Use a run ID for output organization
    python pipeline_runner_v2.py --run_id my_experiment
    
    # Test only (load existing models)
    python pipeline_runner_v2.py --test_only
    
    # Compute benchmarks only (no training/testing)
    python pipeline_runner_v2.py --benchmark_only
"""

import argparse
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import numpy as np
import numpy as np
import torch

# Add current directory to path for local imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Local imports
from config_v2 import (
    RunConfig,
    TrainingConfig,
    ModelArchConfig, 
    DataConfig,
    HyperparameterGrid,
)
from pipeline_core_v2 import (
    run_single_config,
    load_or_compute_benchmarks,
    create_generative_model,
    run_benchmarks,  # re-exported for backward compatibility (defined in pipeline_core_v2)
)

# # Import benchmark functions from original pipeline (reusing, not rewriting)
# from pipeline_next import get_ctx_gm_subpath, compute_benchmarks, benchmarks_pars_viz


# =============================================================================
# Main Pipeline Runner
# =============================================================================

def run_pipeline(
    hp_grid: HyperparameterGrid,
    skip_benchmarks: bool = False,
    train_only: bool = False,
    test_only: bool = False,
    verbose: bool = True,
):
    """
    Run the full training/testing pipeline for all configurations in the grid.
    
    1. Expand grid into flat list of RunConfigs
    2. Compute/load benchmarks (once, shared by all runs)
    3. Loop over configs and run each one
    
    Args:
        grid: Hyperparameter grid specification
        skip_benchmarks: If True, skip Kalman filter benchmark computation
        train_only: If True, only train (skip testing)
        test_only: If True, only test (skip training)
        verbose: If True, print progress information
    """
    # ========== STEP 1: EXPAND GRID ==========
    configs = hp_grid.expand()
    
    if verbose:
        print("\n" + "=" * 70)
        print("PIPELINE RUNNER")
        print("=" * 70)
        print(f"Total configurations to run: {len(configs)}")
        print(f"Models: {hp_grid.model_types}")
        print(f"Learning rates: {hp_grid.learning_rates}")
        print(f"Hidden dims: {hp_grid.hidden_dims}")
        print(f"N_ctx: {hp_grid.data.N_ctx}, GM: {hp_grid.data.gm_name}")
        if hp_grid.run_id:
            print(f"Run ID: {hp_grid.run_id}")
        print("=" * 70)
    
    # ========== STEP 2: COMPUTE/LOAD BENCHMARKS ==========
    if skip_benchmarks:
        if verbose:
            print("\nSkipping benchmarks (skip_benchmarks=True)")
        benchmarks_test = None
    else:
        if verbose:
            print("\n" + "-" * 40)
            print("STEP 1: Computing/loading test benchmarks")
            print("-" * 40)
        
        # Build model_config and data_config dicts for backward compatibility
        # with load_or_compute_benchmarks
        model_config = {
            'batch_size': hp_grid.training.batch_size,
            'batch_size_test': hp_grid.training.batch_size_test,
        }
        data_config = hp_grid.data.to_gm_dict(hp_grid.training.batch_size)
        
        benchmarks_test = load_or_compute_benchmarks(
            data_config,
            model_config,
            hp_grid.data.N_ctx,
            hp_grid.data.gm_name,
            visualize=True,
            max_cores=hp_grid.data.max_cores,
        )
    
    # ========== STEP 3: RUN EACH CONFIG ==========
    if verbose:
        print("\n" + "-" * 40)
        print(f"STEP 2: Running {len(configs)} configurations")
        print("-" * 40)
    
    results = {}
    for i, config in enumerate(configs, 1):
        if verbose:
            print(f"\n[{i}/{len(configs)}] Running: {config.name}")
            print(f"         Save dir: {config.save_dir}")
        
        result = run_single_config(
            config,
            benchmarks_train=None,
            benchmarks_test=benchmarks_test,
            train_only=train_only,
            test_only=test_only,
        )
        results[config.name] = result
            
    if verbose:
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print(f"Successful: {sum(1 for r in results.values() if 'error' not in r)}/{len(configs)}")
        print("=" * 70)
    
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and test RNN models with clean pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default training (RNN + VRNN, multiple LRs and hidden dims)
    python pipeline_runner_v2.py
    
    # Quick unit test
    python pipeline_runner_v2.py --unit_test
    
    # Compute benchmarks only (no training)
    python pipeline_runner_v2.py --benchmark_only
    
    # Train ModuleNetwork with different objectives
    python pipeline_runner_v2.py --models module_network --objectives obs ctx obs_ctx
    
    # Custom learning rates
    python pipeline_runner_v2.py --learning_rates 0.01 0.001
        """
    )
    
    # Mode flags
    parser.add_argument('--unit_test', action='store_true',
                        help='Run minimal config for testing')
    parser.add_argument('--skip_benchmarks', action='store_true',
                        help='Skip Kalman filter benchmark computation')
    parser.add_argument('--benchmark_only', action='store_true',
                        help='Only compute benchmarks (no training/testing)')
    parser.add_argument('--benchmark_tag', type=str, default='',
                        help='Optional tag appended to benchmark file/dir names to avoid '
                             'overwriting existing files (e.g. "unit_test")')
    parser.add_argument('--train_only', action='store_true',
                        help='Only train, skip testing')
    parser.add_argument('--test_only', action='store_true',
                        help='Only test (requires existing trained models)')
    
    # Output organization
    parser.add_argument('--run_id', type=str, default=None,
                        help='Run identifier for output folder. Use "auto" for timestamp.')
    
    # Model selection
    parser.add_argument('--models', type=str, nargs='+',
                        default=None,
                        choices=['rnn', 'vrnn', 'module_network', 'population_network'],
                        help='Models to train (default: rnn vrnn)')
    
    # Hyperparameters
    parser.add_argument('--learning_rates', type=float, nargs='+',
                        default=None,
                        help='Learning rates to try (default: 0.05 0.01 0.005 0.001)')
    parser.add_argument('--hidden_dims', type=int, nargs='+',
                        default=None,
                        help='Hidden dimensions to try (default: 16 32 64)')
    
    # ModuleNetwork specific
    parser.add_argument('--objectives', type=str, nargs='+',
                        default=None,
                        choices=['obs', 'ctx', 'all'],
                        help='Learning objectives for ModuleNetwork')
    parser.add_argument('--bottleneck_dims', type=int, nargs='+',
                        default=None,
                        choices=[8, 16],
                        help='Bottleneck dimensions for ModuleNetwork')
    parser.add_argument('--kappa_values', type=float, nargs='+',
                        default=[0.3, 0.5, 0.7],
                        help='Kappa values for obs_ctx objective')
    
    # Data configuration
    parser.add_argument('--n_ctx', type=int, default=1,
                        help='Number of contexts (default: 1)')
    parser.add_argument('--gm_name', type=str, default='NonHierarchicalGM',
                        choices=['NonHierarchicalGM', 'HierarchicalGM'],
                        help='Generative model name')

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=100,
                            help='Number of training epochs (default: 100)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle run_id
    run_id = args.run_id
    if run_id == 'auto':
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Build training config
    if args.unit_test:
        training = TrainingConfig.for_unit_test()
    else:
        training = TrainingConfig(num_epochs=args.num_epochs)
    
    # Build data config
    data = DataConfig(
        gm_name=args.gm_name,
        N_ctx=args.n_ctx,
        N_tones=training.batch_size,  # Sequence length matches batch size # TODO: What???
    )
    
    # Apply HierarchicalGM-specific defaults if needed.
    # Uses the shared single source of truth so this path stays in sync with
    # experiment.py (training) and benchmark_experiment.py (KF benchmarks).
    if args.gm_name == 'HierarchicalGM':
        data = DataConfig.for_hierarchical_experiment(N_ctx=args.n_ctx)
    
    # =========================================================================
    # BENCHMARK-ONLY MODE: Early exit, no model grid needed
    # =========================================================================
    if args.benchmark_only:
        run_benchmarks(
            data_config=data,
            training_config=training,
            visualize=True,
            suffix_tag=args.benchmark_tag,
        )
        return
    
    # =========================================================================
    # TRAINING/TESTING MODE: Build full hyperparameter grid
    # =========================================================================
    # Set default hyperparameters based on mode
    if args.unit_test:
        default_lrs = [0.01]
        default_hdims = [16]
    else:
        default_lrs = [0.05, 0.01, 0.005, 0.001, 0.0005]
        default_hdims = [16, 32, 64]
    
    # Build hyperparameter grid
    hp_grid = HyperparameterGrid(
        model_types=args.models,
        learning_rates=args.learning_rates or default_lrs,
        hidden_dims=args.hidden_dims or default_hdims,
        learning_objectives=args.objectives,
        kappa_values=args.kappa_values,
        bottleneck_dims=args.bottleneck_dims,
        training=training,
        model_arch=ModelArchConfig(),
        data=data,
        run_id=run_id,
    )
    
    # Run full pipeline
    run_pipeline(
        hp_grid,
        skip_benchmarks=args.skip_benchmarks,
        train_only=args.train_only,
        test_only=args.test_only,
    )


if __name__ == '__main__':
    main()
