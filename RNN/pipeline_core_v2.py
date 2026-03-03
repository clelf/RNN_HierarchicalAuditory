"""
Refactored core pipeline functions with clear, linear flow.

This module provides the essential building blocks:
- Model creation
- Training loop (single config)
- Testing loop (single config)
- Data provision

Key design principles:
1. Each function has a SINGLE responsibility
2. No nested loops for hyperparameter iteration (that's in the runner)
3. Clear input/output contracts
4. Reuse existing utility functions from pipeline_next.py
"""

import os
import sys
import time
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# Plotting backend
import matplotlib.pyplot as plt

# Local imports - reuse existing code (files are now in the same directory)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from model import SimpleRNN, VRNN, ModuleNetwork
from objectives import Objective

# Import from original pipeline (reusing, not rewriting)
from pipeline_next import (
    # Data utilities
    prepare_batch_data,
    prepare_benchmark_data,
    contexts_to_responsibilities,
    # Loss and prediction extraction
    compute_model_loss,
    extract_model_predictions,
    # Benchmark computation
    compute_benchmarks,
    load_or_compute_benchmarks,
    benchmark_filename,
    # Metrics and binning
    bin_params,
    map_binned_params_2_metrics,
    # Plotting
    plot_weights,
    plot_mse,
    plot_variance,
    plot_losses,
    plot_samples,
    # Constants
    VALID_DATA_MODES,
    VALID_LEARNING_OBJECTIVES,
    MIN_OBS_FOR_EM,
)

# Generative models (PreProParadigm is one level up from RNN/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PreProParadigm.audit_gm import NonHierachicalAuditGM, HierarchicalAuditGM

# Local config (config_v2 is now in the same directory)
from config_v2 import RunConfig, run_config_to_model_dict, run_config_to_training_dict


# =============================================================================
# Model Factory
# =============================================================================

def create_model(config: RunConfig) -> nn.Module:
    """
    Create a model instance from a RunConfig.
    
    This is the SINGLE place where model instantiation happens.
    No more scattered conditionals across the codebase.
    """
    model_dict = run_config_to_model_dict(config)
    
    if config.model_type == 'rnn':
        model = SimpleRNN(model_dict)
        
    elif config.model_type == 'vrnn':
        model = VRNN(model_dict)
        
    elif config.model_type == 'module_network':
        # ModuleNetwork needs the full config with module sub-configs
        module_config = _build_module_network_config(config)
        model = ModuleNetwork(module_config)
        
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    return model.to(config.device)


def _build_module_network_config(config: RunConfig) -> dict:
    """Build the specialized config dict for ModuleNetwork."""
    return {
        'kappa': config.kappa,
        'observation_module': {
            'input_dim': config.model_arch.input_dim,
            'output_dim': config.model_arch.output_dim,
            'rnn_hidden_dim': 64,
            'rnn_n_layers': config.model_arch.rnn_n_layers,
            'bottleneck_dim': config.bottleneck_dim or 16,
        },
        'context_module': {
            'input_dim': 2,
            'output_dim': config.data.N_ctx,
            'rnn_hidden_dim': 32,
            'rnn_n_layers': config.model_arch.rnn_n_layers,
            'bottleneck_dim': config.bottleneck_dim or 16,
        },
    }


# =============================================================================
# Generative Model Factory
# =============================================================================

def create_generative_model(config: RunConfig, batch_size: Optional[int] = None):
    """Create the appropriate generative model for data generation."""
    gm_dict = config.data.to_gm_dict(batch_size or config.training.batch_size)
    
    if config.data.gm_name == 'NonHierarchicalGM':
        return NonHierachicalAuditGM(gm_dict)
    elif config.data.gm_name == 'HierarchicalGM':
        return HierarchicalAuditGM(gm_dict)
    else:
        raise ValueError(f"Unknown GM name: {config.data.gm_name}")


# =============================================================================
# Training Loop
# =============================================================================

def train_model(
    model: nn.Module,
    config: RunConfig,
    benchmarks: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Train a model for one configuration.
    
    This is the clean, linear training loop:
    1. Setup (optimizer, loss, data generator)
    2. Epoch loop
       - Batch loop (train)
       - Validation step
       - Logging
    3. Save model and plots
    
    Args:
        model: The model to train (already on correct device)
        config: Complete run configuration
        benchmarks: Optional benchmark data for validation comparison
    
    Returns:
        Dict with training history (losses, metrics, etc.)
    """
    # ========== SETUP ==========
    device = config.device
    save_path = config.save_dir
    os.makedirs(save_path / 'samples', exist_ok=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Loss function
    loss_fn = torch.nn.GaussianNLLLoss(reduction='mean')
    if config.data.data_mode == 'multi_ctx' and config.learning_objective in ['ctx', 'obs_ctx']:
        loss_fn_ctx = torch.nn.CrossEntropyLoss(reduction='mean')
        objective = Objective(loss_fn, loss_func_ctx=loss_fn_ctx)
    else:
        objective = Objective(loss_fn)
    
    # Data generator
    gm = create_generative_model(config)
    data_mode = config.data.data_mode
    
    # Training config shortcuts
    num_epochs = config.training.num_epochs
    n_batches = config.training.n_batches
    batch_res = config.training.batch_res
    epoch_res = config.training.epoch_res
    
    # History tracking
    history = {
        'train_losses': [],
        'valid_losses': [],
        'train_steps': [],
        'valid_steps': [],
        'valid_mse': [],
        'valid_sigma': [],
        'model_kf_mse': [] if benchmarks else None,
        'weights_updates': [],
    }
    param_names = list(model.state_dict().keys())
    
    # Logging setup
    lr_title = f"Model: {config.model_type} | LR: {config.learning_rate:>6.0e} | #units: {config.hidden_dim}"
    
    print(f"\n{'='*60}")
    print(f"TRAINING: {config.name}")
    print(f"{'='*60}")
    
    # ========== TRAINING LOOP ==========
    for epoch in tqdm(range(num_epochs), desc="Epochs", leave=False):
        epoch_losses = []
        t_start = time.time()
        model.train()
        
        # ----- BATCH LOOP -----
        for batch_idx in range(n_batches):
            optimizer.zero_grad()
            
            # Generate batch
            batch_data = prepare_batch_data(gm, config.data.gm_name, data_mode, device)
            y = batch_data['y']
            contexts = batch_data['contexts']
            
            # Forward pass (predict y[1:T] from y[0:T-1])
            model_output = model(y[:, :-1, :])
            
            # Compute loss
            loss = compute_model_loss(
                model, objective, y, model_output, data_mode,
                learning_objective=config.learning_objective,
                contexts_tensor=contexts,
                kappa=config.kappa
            )
            
            # Track weights before update (for logging)
            if batch_idx % batch_res == batch_res - 1:
                weights_before = [w.detach().clone() for w in model.parameters()]
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track weights after update
            if batch_idx % batch_res == batch_res - 1:
                weights_after = [w.detach().clone() for w in model.parameters()]
                weights_update = torch.stack([
                    torch.mean((wa - wb)**2) 
                    for wb, wa in zip(weights_before, weights_after)
                ])
                history['weights_updates'].append(weights_update)
            
            epoch_losses.append(loss.item())
            
            # Batch logging
            if batch_idx % batch_res == batch_res - 1:
                history['train_steps'].append(epoch * n_batches + batch_idx)
                history['train_losses'].append(loss.item())
                
                _log_batch(
                    save_path, config.lr_id, config.learning_rate,
                    epoch, batch_idx, loss.item(),
                    np.mean(epoch_losses[-batch_res:]),
                    time.time() - t_start,
                    epoch * n_batches + batch_idx
                )
                t_start = time.time()
        
        # ----- VALIDATION -----
        model.eval()
        with torch.no_grad():
            valid_metrics = _validate_epoch(
                model, config, benchmarks, gm, objective
            )
        
        # Store validation metrics
        history['valid_losses'].append(valid_metrics['loss'])
        history['valid_mse'].append(valid_metrics['mse'])
        history['valid_sigma'].append(valid_metrics['sigma'])
        history['valid_steps'].append((epoch + 1) * n_batches)
        
        if benchmarks and 'mse_model2kal' in valid_metrics:
            history['model_kf_mse'].append(valid_metrics['mse_model2kal'])
        
        # Save sample plots periodically
        if epoch % epoch_res == epoch_res - 1:
            _save_validation_samples(
                valid_metrics, config, epoch, lr_title, benchmarks
            )
        
        # Epoch logging
        _log_epoch(
            save_path, config.lr_id, config.learning_rate,
            epoch, valid_metrics, time.time() - t_start,
            (epoch + 1) * n_batches, benchmarks is not None
        )
    
    # ========== SAVE RESULTS ==========
    print(f"Final - Train Loss: {np.mean(epoch_losses):.4f}, Valid Loss: {valid_metrics['loss']:.4f}")
    
    # Save model weights
    torch.save(model.state_dict(), save_path / f'lr{config.lr_id}_weights.pth')
    
    # Save config for reproducibility and easy loading
    config_path = config.save()
    print(f"Config saved to: {config_path}")
    
    # Save plots
    _save_training_plots(history, config, lr_title, benchmarks)
    
    return history


def _validate_epoch(
    model: nn.Module,
    config: RunConfig,
    benchmarks: Optional[Dict],
    gm,
    objective: Objective,
) -> Dict[str, Any]:
    """Run validation for one epoch and return metrics."""
    device = config.device
    data_mode = config.data.data_mode
    
    # Get validation data
    if benchmarks:
        bench_data = prepare_benchmark_data(benchmarks, data_mode, device)
        y = bench_data['y']
        contexts = bench_data['contexts']
        mu_kal = bench_data['mu_kal_pred']
        pars = bench_data['pars']
        min_obs = bench_data['min_obs_for_em']
    else:
        batch_data = prepare_batch_data(gm, config.data.gm_name, data_mode, device, return_pars=True)
        y = batch_data['y']
        contexts = batch_data['contexts']
        pars = batch_data['pars']
        mu_kal = None
        min_obs = None
    
    # Forward pass
    model_output = model(y[:, :-1, :])
    
    # Compute loss
    loss = compute_model_loss(
        model, objective, y, model_output, data_mode,
        learning_objective=config.learning_objective,
        contexts_tensor=contexts,
        kappa=config.kappa
    ).item()
    
    # Extract predictions
    mu_estim, var_estim, _ = extract_model_predictions(model, model_output)
    sigma_estim = np.sqrt(var_estim)
    
    # Compute MSE
    y_np = y.detach().cpu().numpy().squeeze()
    mse = ((mu_estim - y_np[:, 1:])**2).mean()
    
    metrics = {
        'loss': loss,
        'mse': mse,
        'sigma': sigma_estim.mean(),
        'mu_estim': mu_estim,
        'sigma_estim': sigma_estim,
        'y': y_np,
        'pars': pars,
    }
    
    # Compare with Kalman filter if available
    if mu_kal is not None:
        mu_estim_aligned = mu_estim[:, min_obs - 1:]
        mse_model2kal = ((mu_estim_aligned - mu_kal)**2).mean()
        metrics['mse_model2kal'] = mse_model2kal
        metrics['mu_kal'] = mu_kal
        metrics['min_obs'] = min_obs
    
    return metrics


# =============================================================================
# Testing
# =============================================================================

def test_model(
    model: nn.Module,
    config: RunConfig,
    benchmarks: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Test a trained model on held-out data.
    
    Args:
        model: The model to test
        config: Complete run configuration  
        benchmarks: Optional benchmark data for comparison
    
    Returns:
        Dict with test metrics
    """
    print(f"\n{'='*60}")
    print(f"TESTING: {config.name}")
    print(f"{'='*60}")
    
    save_path = config.save_dir
    device = config.device
    
    # Check model exists
    weights_path = save_path / f'lr{config.lr_id}_weights.pth'
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    # Load model
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()
    
    # Get test data
    data_mode = config.data.data_mode
    if benchmarks:
        bench_data = prepare_benchmark_data(benchmarks, data_mode, device)
        y = bench_data['y']
        contexts = bench_data['contexts']
        mu_kal = bench_data['mu_kal_pred']
        pars = bench_data['pars']
        mse_kal = bench_data['mse_kal']
        min_obs = bench_data['min_obs_for_em']
    else:
        gm = create_generative_model(config, config.training.batch_size_test)
        batch_data = prepare_batch_data(gm, config.data.gm_name, data_mode, device, return_pars=True)
        y = batch_data['y']
        contexts = batch_data['contexts']
        pars = batch_data['pars']
        mu_kal = None
        min_obs = None
    
    # Forward pass
    with torch.no_grad():
        model_output = model(y[:, :-1, :])
        mu_estim, var_estim, _ = extract_model_predictions(model, model_output)
        sigma_estim = np.sqrt(var_estim)
    
    # Compute metrics
    y_np = y.detach().cpu().numpy().squeeze()
    mse = ((mu_estim - y_np[:, 1:])**2).mean()
    
    results = {
        'mse': mse,
        'sigma': sigma_estim.mean(),
    }
    
    if mu_kal is not None:
        mu_estim_aligned = mu_estim[:, min_obs - 1:]
        results['mse_model2kal'] = ((mu_estim_aligned - mu_kal)**2).mean()
        results['mse_kal'] = mse_kal.mean() if hasattr(mse_kal, 'mean') else mse_kal
    
    # Save binned metrics if parameter testing enabled
    if config.data.params_testing:
        param_bins = bin_params(config.data.to_gm_dict(config.training.batch_size))
        if mu_kal is not None:
            mu_estim_aligned = mu_estim[:, min_obs - 1:]
            y_aligned = y_np[:, min_obs:]
            binned_df = map_binned_params_2_metrics(param_bins, y_aligned, mu_estim_aligned, pars, mu_kal=mu_kal)
        else:
            binned_df = map_binned_params_2_metrics(param_bins, y_np[:, 1:], mu_estim, pars)
        binned_df.to_csv(save_path / f'test_binned_metrics_lr{config.lr_id}.csv', index=False)
    
    print(f"Test MSE: {results['mse']:.4f}")
    if 'mse_model2kal' in results:
        print(f"Model-KF MSE: {results['mse_model2kal']:.4f}")
    
    return results


# =============================================================================
# Logging Helpers
# =============================================================================

def _log_batch(save_path, lr_id, lr, epoch, batch, loss, batch_loss, elapsed, step):
    """Log batch-level training info."""
    msg = f'LR: {lr:>6.0e}; epoch: {epoch:0>3}; batch: {batch:>3}; loss: {loss:>7.4f}; batch loss: {batch_loss:7.4f}; time: {elapsed:>.2f}; step: {step}'
    with open(save_path / f'training_loss_lr{lr_id}.txt', 'a') as f:
        f.write(f'{msg}\n')


def _log_epoch(save_path, lr_id, lr, epoch, metrics, elapsed, step, has_benchmarks):
    """Log epoch-level validation info."""
    msg = f"LR: {lr:>6.0e}; epoch: {epoch:>3}; mean var: {metrics['sigma']:>7.2f}; mean MSE: {metrics['mse']:>7.2f}; time: {elapsed:>.2f}; step: {step}"
    if has_benchmarks and 'mse_model2kal' in metrics:
        msg += f"; Model-KF MSE: {metrics['mse_model2kal']:>7.2f}"
    with open(save_path / f'training_log_lr{lr_id}.txt', 'a') as f:
        f.write(f'{msg}\n')


def _save_validation_samples(metrics, config, epoch, title, benchmarks):
    """Save validation sample plots."""
    save_path = config.save_dir / 'samples' / f'lr{config.lr_id}-epoch-{epoch:0>3}_samples'
    
    kwargs = {
        'params': metrics['pars'],
        'title': title,
        'data_config': config.data.to_gm_dict(config.training.batch_size),
        'seq_len': config.seq_len_viz,
    }
    
    if benchmarks and 'mu_kal' in metrics:
        kwargs['kalman_mu'] = metrics['mu_kal']
        kwargs['min_obs_for_em'] = metrics['min_obs']
    
    plot_samples(metrics['y'], metrics['mu_estim'], metrics['sigma_estim'], save_path, **kwargs) # TODO: also pass contexts


def _save_training_plots(history, config, title, benchmarks):
    """Save end-of-training plots."""
    save_path = config.save_dir
    lr_id = config.lr_id
    
    # Loss curves
    plot_losses(
        history['train_steps'], history['valid_steps'],
        history['train_losses'], history['valid_losses'],
        'Training steps', 'Loss', title,
        save_path / f'loss_trainvalid_lr{lr_id}.png'
    )
    
    # Variance over epochs
    epoch_steps = list(range(len(history['valid_sigma'])))
    plot_variance(epoch_steps, history['valid_sigma'], title, save_path / f'variance_valid_lr{lr_id}.png')
    
    # MSE over epochs
    if benchmarks and history['model_kf_mse']:
        mse_kal = benchmarks.get('perf', None)
        if hasattr(mse_kal, 'mean'):
            mse_kal = mse_kal.mean()
        plot_mse(epoch_steps, history['valid_mse'], title, 
                 save_path / f'mse_valid_lr{lr_id}.png',
                 mse_kal=mse_kal, model_mse_kal=history['model_kf_mse'])
    else:
        plot_mse(epoch_steps, history['valid_mse'], title, save_path / f'mse_valid_lr{lr_id}.png')
    
    # Weight updates
    if history['weights_updates']:
        weights_updates = torch.stack(history['weights_updates'], dim=1)
        param_names = list(config.model_type)  # Simplified
        plot_weights(
            history['train_steps'][:len(history['weights_updates'])],
            weights_updates,
            [f'param_{i}' for i in range(weights_updates.shape[0])],
            title,
            save_path / f'weights_updates_lr{lr_id}.png'
        )


# =============================================================================
# Full Pipeline (combines train + test)
# =============================================================================

def run_single_config(
    config: RunConfig,
    benchmarks_train: Optional[Dict] = None,
    benchmarks_test: Optional[Dict] = None,
    train_only: bool = False,
    test_only: bool = False,
) -> Dict[str, Any]:
    """
    Run training and/or testing for a single configuration.
    
    This is the main entry point for running one experiment.
    """
    results = {}
    
    # Create model
    model = create_model(config)
    
    # Train
    if not test_only:
        history = train_model(model, config, benchmarks_train)
        results['training'] = history
    
    # Test
    if not train_only:
        test_results = test_model(model, config, benchmarks_test)
        results['testing'] = test_results
    
    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results
