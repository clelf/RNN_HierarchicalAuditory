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
)

# # Import benchmark functions from original pipeline (reusing, not rewriting)
# from pipeline_next import get_ctx_gm_subpath, compute_benchmarks, benchmarks_pars_viz


# =============================================================================
# Benchmark-Only Pipeline
# =============================================================================

def run_benchmarks(
    grid: HyperparameterGrid,
    benchmark_mode: str = 'both',
    visualize: bool = True,
    verbose: bool = True,
    suffix_tag: str = '',
):
    """
    Compute Kalman filter benchmarks only (no model training/testing).
    
    This is useful for:
    - Pre-computing expensive benchmarks before running training
    - Analyzing Kalman filter performance on different data configurations
    - Generating benchmark visualizations
    
    Args:
        grid: Hyperparameter grid (only data config is used)
        benchmark_mode: 'both', 'train_only', or 'test_only'
        visualize: If True, generate parameter distribution plots
        verbose: If True, print progress information
    
    Returns:
        Tuple of (benchmarks_train, benchmarks_test) dictionaries
    """
    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK COMPUTATION")
        print("=" * 70)
        print(f"N_ctx: {grid.data.N_ctx}, GM: {grid.data.gm_name}")
        print(f"Batch size (train): {grid.training.batch_size}")
        print(f"Batch size (test): {grid.training.batch_size_test}")
        print(f"Mode: {benchmark_mode}")
        print(f"Max cores: {grid.data.max_cores}")
        print("=" * 70)
    
    # Build config dicts for backward compatibility
    model_config = {
        'batch_size': grid.training.batch_size,
        'batch_size_test': grid.training.batch_size_test,
    }
    data_config = grid.data.to_gm_dict(grid.training.batch_size)
    
    benchmarks_train, benchmarks_test = load_or_compute_benchmarks(
        data_config,
        model_config,
        grid.data.N_ctx,
        grid.data.gm_name,
        visualize=visualize,
        max_cores=grid.data.max_cores,
        benchmark_mode=benchmark_mode,
        suffix_tag=suffix_tag,
    )
    
    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK COMPUTATION COMPLETE")
        if benchmarks_train is not None:
            print(f"Train benchmarks: {benchmarks_train['y'].shape[0]} samples")
            print(f"  KF MSE (mean): {benchmarks_train['perf'].mean():.4f}")
        if benchmarks_test is not None:
            print(f"Test benchmarks: {benchmarks_test['y'].shape[0]} samples")
            print(f"  KF MSE (mean): {benchmarks_test['perf'].mean():.4f}")
        print("=" * 70)
    
    return benchmarks_train, benchmarks_test


# =============================================================================
# Main Pipeline Runner
# =============================================================================

def run_pipeline(
    grid: HyperparameterGrid,
    skip_benchmarks: bool = False,
    train_only: bool = False,
    test_only: bool = False,
    verbose: bool = True,
):
    """
    Run the full training/testing pipeline for all configurations in the grid.
    
    This is the MAIN ENTRY POINT. The flow is:
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
    configs = grid.expand()
    
    if verbose:
        print("\n" + "=" * 70)
        print("PIPELINE RUNNER")
        print("=" * 70)
        print(f"Total configurations to run: {len(configs)}")
        print(f"Models: {grid.model_types}")
        print(f"Learning rates: {grid.learning_rates}")
        print(f"Hidden dims: {grid.hidden_dims}")
        print(f"N_ctx: {grid.data.N_ctx}, GM: {grid.data.gm_name}")
        if grid.run_id:
            print(f"Run ID: {grid.run_id}")
        print("=" * 70)
    
    # ========== STEP 2: COMPUTE BENCHMARKS ==========
    if skip_benchmarks:
        if verbose:
            print("\nSkipping benchmarks (skip_benchmarks=True)")
        benchmarks_train = None
        benchmarks_test = None
    else:
        if verbose:
            print("\n" + "-" * 40)
            print("STEP 1: Computing/loading benchmarks")
            print("-" * 40)
        
        # Build model_config and data_config dicts for backward compatibility
        # with load_or_compute_benchmarks
        model_config = {
            'batch_size': grid.training.batch_size,
            'batch_size_test': grid.training.batch_size_test,
        }
        data_config = grid.data.to_gm_dict(grid.training.batch_size)
        
        benchmarks_train, benchmarks_test = load_or_compute_benchmarks(
            data_config,
            model_config,
            grid.data.N_ctx,
            grid.data.gm_name,
            visualize=True,
            max_cores=grid.data.max_cores,
            benchmark_mode='both',
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
        
        try:
            result = run_single_config(
                config,
                benchmarks_train=benchmarks_train,
                benchmarks_test=benchmarks_test,
                train_only=train_only,
                test_only=test_only,
            )
            results[config.name] = result
            
        except Exception as e:
            print(f"ERROR running {config.name}: {e}")
            results[config.name] = {'error': str(e)}
            
            # Continue with other configs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
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
    parser.add_argument('--benchmark_mode', type=str, default='both',
                        choices=['both', 'train_only', 'test_only'],
                        help='Which benchmarks to compute (default: both)')
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
                        default=['rnn', 'vrnn'],
                        choices=['rnn', 'vrnn', 'module_network'],
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
                        default=['obs'],
                        choices=['obs', 'ctx', 'obs_ctx'],
                        help='Learning objectives for ModuleNetwork')
    parser.add_argument('--bottleneck_dims', type=int, nargs='+',
                        default=[8, 16],
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
        N_tones=training.batch_size,  # Sequence length matches batch size
    )
    
    # Set default hyperparameters based on mode
    if args.unit_test:
        default_lrs = [0.01]
        default_hdims = [16]
    else:
        default_lrs = [0.05, 0.01, 0.005, 0.001, 0.0005]
        default_hdims = [16, 32, 64]
    
    # Build hyperparameter grid
    grid = HyperparameterGrid(
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
    
    # Handle benchmark_only mode
    if args.benchmark_only:
        run_benchmarks(
            grid,
            benchmark_mode=args.benchmark_mode,
            visualize=True,
            suffix_tag=args.benchmark_tag,
        )
        return
    
    # Run full pipeline
    run_pipeline(
        grid,
        skip_benchmarks=args.skip_benchmarks,
        train_only=args.train_only,
        test_only=args.test_only,
    )


if __name__ == '__main__':
    main()
