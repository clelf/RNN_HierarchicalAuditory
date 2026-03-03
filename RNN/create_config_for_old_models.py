"""
Generate config files for models trained before config saving was implemented.

This script helps you add config files to existing trained models, improving
reproducibility for models trained with older pipeline versions.

Usage:
    # Generate config for a single model
    python create_config_for_old_models.py --model_dir training_results/N_ctx_1/rnn_h16
    
    # Generate configs for all models in a directory
    python create_config_for_old_models.py --scan_dir training_results/N_ctx_1
    
    # Dry run to see what would be created
    python create_config_for_old_models.py --scan_dir training_results --dry_run
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List

# Add current directory to path for local imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Local imports
from evaluate_models import ModelInfo
from config_v2 import RunConfig, TrainingConfig, ModelArchConfig, DataConfig


def create_config_from_inference(model_dir: Path,
                                  training_config: TrainingConfig = None,
                                  dry_run: bool = False,
                                  force: bool = False) -> bool:
    """
    Create a config file for an old model by inferring settings from directory structure.
    
    Args:
        model_dir: Path to model directo    ry
        training_config: Optional TrainingConfig to use (defaults to standard config)
        dry_run: If True, only print what would be done without creating files
        force: If True, overwrite existing config files
    
    Returns:
        True if config was created/would be created, False otherwise
    """
    config_path = model_dir / 'config.json'
    
    # Find any weights file to infer architecture from
    weights_files = list(model_dir.glob('lr*_weights.pth'))
    if not weights_files:
        print(f"  ✗ Skipping {model_dir}: No weights found")
        return False
    weights_path = weights_files[0]  # Use first available weights file
    
    # Check if config already exists
    if config_path.exists() and not force:
        print(f"  Skipping {model_dir}: Config already exists (use --force to overwrite)")
        return False
    
    try:
        # Infer model info from directory structure
        info = ModelInfo._from_directory_structure(model_dir, weights_path)
        
        # For ModuleNetwork, verify bottleneck_dim from weights file
        if info.model_type == 'module_network':
            from evaluate_models import infer_bottleneck_dim_from_weights
            inferred_bn = infer_bottleneck_dim_from_weights(weights_path)
            if inferred_bn != info.bottleneck_dim:
                print(f"    Note: Correcting bottleneck_dim from {info.bottleneck_dim} to {inferred_bn} (from weights)")
                info = ModelInfo(
                    model_dir=info.model_dir,
                    model_type=info.model_type,
                    hidden_dim=info.hidden_dim,
                    n_ctx=info.n_ctx,
                    gm_name=info.gm_name,
                    weights_path=info.weights_path,
                    lr_id=info.lr_id,
                    learning_objective=info.learning_objective,
                    kappa=info.kappa,
                    bottleneck_dim=inferred_bn,
                )
        
        # Use provided training config or default
        if training_config is None:
            training_config = TrainingConfig()
        
        # Create DataConfig
        data_config = DataConfig(
            gm_name=info.gm_name,
            N_ctx=info.n_ctx,
            N_tones=training_config.batch_size,
        )
        
        # Build RunConfig (lr_id is not stored in config since architecture is lr-independent)
        run_config = RunConfig(
            name=f"{info.model_type}_h{info.hidden_dim}",
            save_dir=model_dir,
            model_type=info.model_type,
            hidden_dim=info.hidden_dim,
            learning_rate=0.0,  # Not stored - varies per training run
            lr_id=0,  # Not stored - varies per training run
            learning_objective=info.learning_objective,
            kappa=info.kappa,
            bottleneck_dim=info.bottleneck_dim,
            training=training_config,
            model_arch=ModelArchConfig(),
            data=data_config,
        )
        
        if dry_run:
            print(f"  ✓ Would create: {config_path}")
            print(f"    - Model: {info.model_type}, Hidden: {info.hidden_dim}")
            print(f"    - N_ctx: {info.n_ctx}, GM: {info.gm_name}")
        else:
            # Save the config
            saved_path = run_config.save(config_path)
            print(f"  ✓ Created: {saved_path}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {model_dir}: {e}")
        return False


def scan_and_create_configs(base_dir: Path,
                             training_config: TrainingConfig = None,
                             dry_run: bool = False,
                             force: bool = False) -> dict:
    """
    Scan a directory tree and create configs for all models without configs.
    
    Args:
        base_dir: Base directory to scan
        training_config: Optional TrainingConfig to use
        dry_run: If True, only print what would be done
        force: If True, overwrite existing config files
    
    Returns:
        Dictionary with counts of created, skipped, and failed configs
    """
    base_dir = Path(__file__).parent.resolve() / Path(base_dir)
    if not base_dir.exists():
        raise ValueError(f"Base directory {base_dir} does not exist.")
        
    
    # Find all directories with weight files (any lr*_weights.pth)
    weight_files = list(base_dir.rglob('lr*_weights.pth'))
    
    # Get unique model directories
    model_dirs = sorted(set(wf.parent for wf in weight_files))
    
    if not model_dirs:
        print(f"No model weights found in {base_dir}")
        return {'created': 0, 'skipped': 0, 'failed': 0}
    
    print(f"\nFound {len(model_dirs)} models in {base_dir}")
    print("=" * 70)
    
    created = 0
    skipped = 0
    failed = 0
    
    for model_dir in model_dirs:
        print(f"\nProcessing: {model_dir}")
        
        result = create_config_from_inference(
            model_dir, training_config, dry_run, force
        )
        
        if result:
            created += 1
        elif (model_dir / 'config.json').exists():
            skipped += 1
        else:
            failed += 1
    
    return {'created': created, 'skipped': skipped, 'failed': failed}


def main():
    parser = argparse.ArgumentParser(
        description='Create config files for models trained before config saving',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate config for a single model
    python create_config_for_old_models.py --model_dir training_results/N_ctx_1/rnn_h16
    
    # Generate configs for all models in a directory
    python create_config_for_old_models.py --scan_dir training_results/N_ctx_1
    
    # Dry run to preview changes
    python create_config_for_old_models.py --scan_dir training_results --dry_run
    
    # Specify custom training parameters (if you remember them)
    python create_config_for_old_models.py --scan_dir training_results_CORRECT \\
        --num_epochs 100 --batch_size 1000
        """
    )
    
    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model_dir', type=str,
                       help='Path to a single model directory')
    group.add_argument('--scan_dir', type=str, default='RNN_paradigm/RNN/training_results_CORRECT/N_ctx_2/NonHierarchicalGM_selected',
                       help='Scan directory tree for all models')
    
    # Options
    parser.add_argument('--dry_run', action='store_true',
                        help='Preview changes without creating files')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing config files')
    
    # Training config options (for reconstructing settings)
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs used in training (default: 100)')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size used in training (default: 1000)')
    parser.add_argument('--n_batches', type=int, default=32,
                        help='Number of batches per epoch (default: 32)')
    
    args = parser.parse_args()
    
    # Build training config from arguments
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        batch_size_test=args.batch_size,
        n_batches=args.n_batches,
    )
    
    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN MODE - No files will be created")
        print("=" * 70)
    
    if args.force:
        print("\n" + "=" * 70)
        print("FORCE MODE - Existing config files will be overwritten")
        print("=" * 70)
    
    # Process models
    if args.model_dir:
        # Single model
        model_dir = Path(args.model_dir)
        print(f"\nProcessing single model: {model_dir}")
        print("=" * 70)
        create_config_from_inference(model_dir, training_config, args.dry_run, args.force)
        
    else:
        # Scan directory
        results = scan_and_create_configs(
            Path(args.scan_dir),
            training_config,
            args.dry_run,
            args.force
        )
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        action = "Would create" if args.dry_run else "Created"
        print(f"{action}: {results['created']}")
        print(f"Skipped (already exist): {results['skipped']}")
        print(f"Failed: {results['failed']}")
        
        if args.dry_run and results['created'] > 0:
            print("\nRun without --dry_run to create the config files.")


if __name__ == '__main__':
    # Set to True to use hardcoded arguments (useful for debugging in editor)
    USE_HARDCODED_ARGS = True
    
    if USE_HARDCODED_ARGS:
        # Manually specify arguments here for running directly from editor/debugger
        # Note: paths are relative to CWD, not to this script's location
        sys.argv = [
            'create_config_for_old_models.py',
            '--scan_dir', '../training_results_CORRECT/N_ctx_2/NonHierarchicalGM_selected',
            # '--model_dir', 'path/to/specific/model',  # Use this instead of --scan_dir for single model
            # '--dry_run',  # Uncomment to preview without creating files
            # '--force',    # Uncomment to overwrite existing configs
            # '--num_epochs', '100',
            # '--batch_size', '1000',
            # '--n_batches', '32',
        ]
    
    main()
