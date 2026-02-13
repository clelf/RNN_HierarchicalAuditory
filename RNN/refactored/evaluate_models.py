"""
Unified model evaluation script.

This script loads multiple trained models and evaluates them on the same
generated test dataset, computing:
- MSE (Mean Squared Error)
- Log-likelihood (Gaussian log-probability)
- Calibration (Kolmogorov-Smirnov test for predicted variance)
- Context inference accuracy (for ModuleNetwork with N_ctx > 1)
- Context log-probability (for ModuleNetwork with N_ctx > 1)

Usage:
    # Evaluate all models in a directory
    python evaluate_models.py --model_dirs training_results/N_ctx_1/rnn_h16 training_results/N_ctx_1/vrnn_h16
    
    # Specify number of test samples
    python evaluate_models.py --model_dirs ... --n_samples 1000
    
    # Custom output path
    python evaluate_models.py --model_dirs ... --output results/evaluation.csv
"""

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy import stats
from tqdm import tqdm

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import SimpleRNN, VRNN, ModuleNetwork
from pipeline_next import extract_model_predictions, prepare_batch_data

# Local config (for loading saved configs)
from config_v2 import RunConfig, TrainingConfig, ModelArchConfig, DataConfig as TrainDataConfig

# Generative models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from PreProParadigm.audit_gm import NonHierachicalAuditGM, HierarchicalAuditGM


# =============================================================================
# Model Configuration Inference
# =============================================================================

@dataclass
class ModelInfo:
    """Information about a trained model extracted from its directory or config file."""
    model_dir: Path
    model_type: str  # 'rnn', 'vrnn', 'module_network'
    hidden_dim: int
    n_ctx: int
    gm_name: str
    weights_path: Path
    lr_id: int = 0
    
    # ModuleNetwork specific
    learning_objective: str = 'obs'
    kappa: float = 0.5
    bottleneck_dim: int = 16
    
    # Full config if loaded from file
    run_config: Optional[RunConfig] = None
    
    # Data config for test data generation (parsed from run_config or defaults)
    data_config_dict: Optional[dict] = None
    
    @classmethod
    def from_path(cls, model_dir: Path, lr_id: int = 0) -> 'ModelInfo':
        """
        Load model configuration, preferring saved config file if available.
        
        Priority:
        1. Load from lr{lr_id}_config.json if it exists
        2. Fall back to inferring from directory structure
        """
        model_dir = Path(model_dir)
        config_path = model_dir / f'lr{lr_id}_config.json'
        weights_path = model_dir / f'lr{lr_id}_weights.pth'
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        
        # Try to load from saved config first
        if config_path.exists():
            return cls._from_config_file(config_path, model_dir, weights_path, lr_id)
        else:
            # Fall back to inferring from directory structure
            return cls._from_directory_structure(model_dir, weights_path, lr_id)
    
    @classmethod
    def _from_config_file(cls, config_path: Path, model_dir: Path, 
                          weights_path: Path, lr_id: int) -> 'ModelInfo':
        """Load ModelInfo from a saved config JSON file."""
        run_config = RunConfig.load(config_path)
        
        return cls(
            model_dir=model_dir,
            model_type=run_config.model_type,
            hidden_dim=run_config.hidden_dim,
            n_ctx=run_config.data.N_ctx,
            gm_name=run_config.data.gm_name,
            weights_path=weights_path,
            lr_id=lr_id,
            learning_objective=run_config.learning_objective,
            kappa=run_config.kappa,
            bottleneck_dim=run_config.bottleneck_dim or 16,
            run_config=run_config,
            data_config_dict=run_config.data.to_gm_dict(run_config.training.batch_size_test),
        )
    
    @classmethod
    def _from_directory_structure(cls, model_dir: Path, weights_path: Path, 
                                   lr_id: int) -> 'ModelInfo':
        """
        Infer model configuration from directory structure (legacy support).
        
        Expected structures:
        - N_ctx_1/rnn_h16/
        - N_ctx_1/vrnn_h32/
        - N_ctx_2/NonHierarchicalGM/module_network_obs_ctx_kappa0.5_bn16/
        """
        # Parse directory structure
        parts = model_dir.parts
        dir_name = model_dir.name
        
        # Find N_ctx from path
        n_ctx = 1
        gm_name = 'NonHierarchicalGM'
        for part in parts:
            if part.startswith('N_ctx_'):
                n_ctx = int(part.split('_')[-1])
            if part in ['NonHierarchicalGM', 'HierarchicalGM']:
                gm_name = part
        
        # Parse model type and hyperparameters from directory name
        if dir_name.startswith('rnn_h'):
            model_type = 'rnn'
            hidden_dim = int(dir_name.split('_h')[1])
            return cls(
                model_dir=model_dir,
                model_type=model_type,
                hidden_dim=hidden_dim,
                n_ctx=n_ctx,
                gm_name=gm_name,
                weights_path=weights_path,
                lr_id=lr_id,
            )
        
        elif dir_name.startswith('vrnn_h'):
            model_type = 'vrnn'
            hidden_dim = int(dir_name.split('_h')[1])
            return cls(
                model_dir=model_dir,
                model_type=model_type,
                hidden_dim=hidden_dim,
                n_ctx=n_ctx,
                gm_name=gm_name,
                weights_path=weights_path,
                lr_id=lr_id,
            )
        
        elif dir_name.startswith('module_network'):
            model_type = 'module_network'
            hidden_dim = 64  # Fixed for ModuleNetwork
            
            # Parse learning objective
            if '_obs_ctx_' in dir_name:
                learning_objective = 'obs_ctx'
            elif '_ctx_' in dir_name:
                learning_objective = 'ctx'
            else:
                learning_objective = 'obs'
            
            # Parse kappa
            kappa = 0.5
            if 'kappa' in dir_name:
                kappa_str = dir_name.split('kappa')[1].split('_')[0]
                kappa = float(kappa_str)
            
            # Parse bottleneck dimension
            bottleneck_dim = 16
            if '_bn' in dir_name:
                bn_str = dir_name.split('_bn')[1]
                bottleneck_dim = int(bn_str)
            
            return cls(
                model_dir=model_dir,
                model_type=model_type,
                hidden_dim=hidden_dim,
                n_ctx=n_ctx,
                gm_name=gm_name,
                weights_path=weights_path,
                lr_id=lr_id,
                learning_objective=learning_objective,
                kappa=kappa,
                bottleneck_dim=bottleneck_dim,
            )
        
        else:
            raise ValueError(f"Cannot parse model type from directory: {dir_name}")


# =============================================================================
# Model Loading
# =============================================================================

def load_model(info: ModelInfo, device: str = 'cpu') -> nn.Module:
    """Load a trained model from its weights file."""
    if info.model_type == 'rnn':
        config = {
            'input_dim': 1,
            'output_dim': 2,
            'hidden_dim': info.hidden_dim,
            'n_layers': 1,
            'device': device,
        }
        model = SimpleRNN(config)
    
    elif info.model_type == 'vrnn':
        config = {
            'input_dim': 1,
            'output_dim': 2,
            'latent_dim': info.hidden_dim,
            'phi_x_dim': info.hidden_dim,
            'phi_z_dim': info.hidden_dim,
            'phi_prior_dim': info.hidden_dim,
            'rnn_hidden_states_dim': info.hidden_dim,
            'rnn_n_layers': 1,
            'device': device,
        }
        model = VRNN(config)
    
    elif info.model_type == 'module_network':
        config = {
            'kappa': info.kappa,
            'observation_module': {
                'input_dim': 1,
                'output_dim': 2,
                'rnn_hidden_dim': 64,
                'rnn_n_layers': 1,
                'bottleneck_dim': info.bottleneck_dim,
            },
            'context_module': {
                'input_dim': 2,
                'output_dim': info.n_ctx,
                'rnn_hidden_dim': 32,
                'rnn_n_layers': 1,
                'bottleneck_dim': info.bottleneck_dim,
            },
            'device': device,
        }
        model = ModuleNetwork(config)
    
    else:
        raise ValueError(f"Unknown model type: {info.model_type}")
    
    # Load weights
    model.load_state_dict(torch.load(info.weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


# =============================================================================
# Test Data Generation
# =============================================================================

@dataclass
class TestDataConfig:
    """Configuration for test data generation."""
    n_samples: int = 1000
    n_tones: int = 1000  # Sequence length
    n_ctx: int = 1
    gm_name: str = 'NonHierarchicalGM'
    
    # Parameter bounds (same as training)
    mu_tau_bounds: dict = field(default_factory=lambda: {'low': 1, 'high': 250})
    si_stat_bounds: dict = field(default_factory=lambda: {'low': 0.1, 'high': 2})
    si_r_bounds: dict = field(default_factory=lambda: {'low': 0.1, 'high': 2})
    
    # Context parameters
    mu_rho_ctx: float = 0.9
    si_rho_ctx: float = 0.05
    si_lim: float = 5.0
    si_tau: float = 0.5
    
    # Multi-context
    si_d_coef: float = 0.05
    d_bounds: dict = field(default_factory=lambda: {'high': 4, 'low': 0.1})
    mu_d: float = 2.0
    
    def to_gm_dict(self) -> dict:
        """Convert to dict expected by GenerativeModel."""
        d = {
            'gm_name': self.gm_name,
            'N_ctx': self.n_ctx,
            'N_samples': self.n_samples,
            'N_blocks': 1,
            'N_tones': self.n_tones,
            'mu_rho_ctx': self.mu_rho_ctx,
            'si_rho_ctx': self.si_rho_ctx,
            'si_lim': self.si_lim,
            'si_tau': self.si_tau,
            'params_testing': True,
            'mu_tau_bounds': self.mu_tau_bounds,
            'si_stat_bounds': self.si_stat_bounds,
            'si_r_bounds': self.si_r_bounds,
        }
        
        if self.n_ctx > 1:
            d.update({
                'si_d_coef': self.si_d_coef,
                'd_bounds': self.d_bounds,
                'mu_d': self.mu_d,
            })
        
        return d
    
    @classmethod
    def from_saved_config(cls, data_config_dict: dict, n_samples: int) -> 'TestDataConfig':
        """Create TestDataConfig from a saved training data config dict."""
        return cls(
            n_samples=n_samples,
            n_tones=data_config_dict.get('N_tones', 1000),
            n_ctx=data_config_dict.get('N_ctx', 1),
            gm_name=data_config_dict.get('gm_name', 'NonHierarchicalGM'),
            mu_tau_bounds=data_config_dict.get('mu_tau_bounds', {'low': 1, 'high': 250}),
            si_stat_bounds=data_config_dict.get('si_stat_bounds', {'low': 0.1, 'high': 2}),
            si_r_bounds=data_config_dict.get('si_r_bounds', {'low': 0.1, 'high': 2}),
            mu_rho_ctx=data_config_dict.get('mu_rho_ctx', 0.9),
            si_rho_ctx=data_config_dict.get('si_rho_ctx', 0.05),
            si_lim=data_config_dict.get('si_lim', 5.0),
            si_tau=data_config_dict.get('si_tau', 0.5),
            si_d_coef=data_config_dict.get('si_d_coef', 0.05),
            d_bounds=data_config_dict.get('d_bounds', {'high': 4, 'low': 0.1}),
            mu_d=data_config_dict.get('mu_d', 2.0),
        )


def generate_test_data(config: TestDataConfig, device: str = 'cpu') -> Dict[str, Any]:
    """
    Generate a shared test dataset.
    
    Returns:
        Dictionary with:
        - 'y': Observations tensor (n_samples, seq_len, 1)
        - 'contexts': Context labels (n_samples, seq_len) if n_ctx > 1
        - 'pars': Generation parameters
    """
    gm_dict = config.to_gm_dict()
    
    if config.gm_name == 'NonHierarchicalGM':
        gm = NonHierachicalAuditGM(gm_dict)
        contexts, _, y, pars = gm.generate_batch(return_pars=True)
    elif config.gm_name == 'HierarchicalGM':
        gm = HierarchicalAuditGM(gm_dict)
        _, _, _, _, _, contexts, _, y, pars = gm.generate_batch(return_pars=True)
    else:
        raise ValueError(f"Unknown GM: {config.gm_name}")
    
    # Convert to tensors
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)
    
    result = {
        'y': y_tensor,
        'y_np': y,
        'pars': pars,
    }
    
    if config.n_ctx > 1:
        contexts_tensor = torch.tensor(contexts, dtype=torch.long).to(device)
        result['contexts'] = contexts_tensor
        result['contexts_np'] = contexts
    else:
        result['contexts'] = None
        result['contexts_np'] = None
    
    return result


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def compute_log_likelihood(y_true: np.ndarray, mu_pred: np.ndarray, var_pred: np.ndarray) -> float:
    """
    Compute mean Gaussian log-likelihood.
    
    log p(y|mu, var) = -0.5 * [log(2*pi*var) + (y-mu)^2/var]
    """
    log_prob = -0.5 * (np.log(2 * np.pi * var_pred) + (y_true - mu_pred) ** 2 / var_pred)
    return float(np.mean(log_prob))


def compute_calibration_ks(y_true: np.ndarray, mu_pred: np.ndarray, var_pred: np.ndarray) -> Tuple[float, float]:
    """
    Compute calibration using Kolmogorov-Smirnov test.
    
    For a well-calibrated model, the z-scores should be standard normal.
    z = (y - mu) / sigma
    
    Returns:
        (ks_statistic, p_value)
        - Lower KS statistic = better calibration
        - Higher p-value = better calibration (cannot reject null hypothesis that z~N(0,1))
    """
    sigma_pred = np.sqrt(var_pred)
    z_scores = (y_true - mu_pred) / sigma_pred
    
    # Flatten for KS test
    z_flat = z_scores.flatten()
    
    # KS test against standard normal
    ks_stat, p_value = stats.kstest(z_flat, 'norm')
    
    return float(ks_stat), float(p_value)


def compute_context_accuracy(contexts_true: np.ndarray, context_logits: np.ndarray) -> float:
    """
    Compute context inference accuracy.
    
    Args:
        contexts_true: True context labels (n_samples, seq_len)
        context_logits: Model context output logits (n_samples, seq_len, n_ctx)
    
    Returns:
        Accuracy as fraction of correctly predicted contexts.
    """
    # Get predicted contexts (argmax over context dimension)
    contexts_pred = np.argmax(context_logits, axis=-1)
    
    return float(np.mean(contexts_pred == contexts_true))


def compute_context_log_prob(contexts_true: np.ndarray, context_logits: np.ndarray) -> float:
    """
    Compute mean log-probability of true contexts.
    
    Args:
        contexts_true: True context labels (n_samples, seq_len)
        context_logits: Model context output logits (n_samples, seq_len, n_ctx)
    
    Returns:
        Mean log-probability of true context under softmax distribution.
    """
    # Apply softmax to get probabilities
    context_probs = F.softmax(torch.tensor(context_logits), dim=-1).numpy()
    
    # Get probability of true context for each timestep
    n_samples, seq_len = contexts_true.shape
    true_probs = np.zeros((n_samples, seq_len))
    
    for i in range(n_samples):
        for t in range(seq_len):
            true_ctx = contexts_true[i, t]
            true_probs[i, t] = context_probs[i, t, true_ctx]
    
    # Compute mean log probability
    log_probs = np.log(true_probs + 1e-10)  # Add small epsilon to avoid log(0)
    
    return float(np.mean(log_probs))


# =============================================================================
# Model Evaluation
# =============================================================================

def evaluate_model(
    model: nn.Module,
    info: ModelInfo,
    test_data: Dict[str, Any],
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Evaluate a single model on the test data.
    
    Returns:
        Dictionary with all computed metrics.
    """
    y = test_data['y']
    y_np = test_data['y_np']
    contexts_np = test_data.get('contexts_np')
    
    # Forward pass
    with torch.no_grad():
        model_output = model(y[:, :-1, :])
        mu_pred, var_pred, context_output = extract_model_predictions(model, model_output)
    
    # Target is y[1:] (predicting next observation)
    y_target = y_np[:, 1:]
    
    # Compute observation metrics
    results = {
        'model_dir': str(info.model_dir),
        'model_type': info.model_type,
        'hidden_dim': info.hidden_dim,
        'n_ctx': info.n_ctx,
        'gm_name': info.gm_name,
        'lr_id': info.lr_id,
    }
    
    # MSE
    results['mse'] = compute_mse(y_target, mu_pred)
    
    # Log-likelihood
    results['log_likelihood'] = compute_log_likelihood(y_target, mu_pred, var_pred)
    
    # Calibration (KS test)
    ks_stat, ks_pval = compute_calibration_ks(y_target, mu_pred, var_pred)
    results['ks_statistic'] = ks_stat
    results['ks_pvalue'] = ks_pval
    
    # ModuleNetwork specific metrics
    if info.model_type == 'module_network' and info.n_ctx > 1 and context_output is not None:
        results['learning_objective'] = info.learning_objective
        results['kappa'] = info.kappa
        results['bottleneck_dim'] = info.bottleneck_dim
        
        # Context predictions (aligned with y[1:])
        context_logits = context_output.detach().cpu().numpy()
        contexts_target = contexts_np[:, 1:]  # Align with predictions
        
        # Context accuracy
        results['context_accuracy'] = compute_context_accuracy(contexts_target, context_logits)
        
        # Context log-probability
        results['context_log_prob'] = compute_context_log_prob(contexts_target, context_logits)
    
    return results


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def evaluate_models(
    model_dirs: List[Path],
    n_samples: int = 1000,
    n_tones: int = 1000,
    lr_id: int = 0,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate multiple models on a shared test dataset.
    
    Args:
        model_dirs: List of paths to model directories
        n_samples: Number of test samples to generate
        n_tones: Sequence length
        lr_id: Learning rate index (for loading weights)
        output_path: Optional path to save results CSV
        verbose: Print progress information
    
    Returns:
        DataFrame with evaluation results for all models
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Number of models: {len(model_dirs)}")
        print(f"Test samples: {n_samples}")
        print(f"Sequence length: {n_tones}")
        print("=" * 70)
    
    # Parse model information
    model_infos = []
    for model_dir in model_dirs:
        try:
            info = ModelInfo.from_path(Path(model_dir), lr_id=lr_id)
            model_infos.append(info)
            if verbose:
                print(f"  Found: {info.model_type} (h={info.hidden_dim}, n_ctx={info.n_ctx})")
        except Exception as e:
            print(f"  ERROR parsing {model_dir}: {e}")
    
    if not model_infos:
        raise ValueError("No valid models found!")
    
    # Check all models have same N_ctx and GM (for shared test data)
    n_ctx_values = set(info.n_ctx for info in model_infos)
    gm_values = set(info.gm_name for info in model_infos)
    
    if len(n_ctx_values) > 1:
        raise ValueError(f"Models have different N_ctx values: {n_ctx_values}. "
                         "Cannot generate shared test data.")
    if len(gm_values) > 1:
        raise ValueError(f"Models have different GM types: {gm_values}. "
                         "Cannot generate shared test data.")
    
    n_ctx = model_infos[0].n_ctx
    gm_name = model_infos[0].gm_name
    
    # Generate shared test data
    # Try to use saved config from first model with a config file for accurate params
    if verbose:
        print(f"\nGenerating test data (n_ctx={n_ctx}, gm={gm_name})...")
    
    # Check if any model has a saved data config
    saved_data_config = None
    for info in model_infos:
        if info.data_config_dict is not None:
            saved_data_config = info.data_config_dict
            if verbose:
                print(f"  Using data config from: {info.model_dir}")
            break
    
    if saved_data_config is not None:
        # Use saved config (ensures same params_testing bounds, etc.)
        data_config = TestDataConfig.from_saved_config(saved_data_config, n_samples=n_samples)
    else:
        # Fall back to defaults
        data_config = TestDataConfig(
            n_samples=n_samples,
            n_tones=n_tones,
            n_ctx=n_ctx,
            gm_name=gm_name,
        )
    test_data = generate_test_data(data_config, device=device)
    
    if verbose:
        print(f"  Generated {n_samples} sequences of length {n_tones}")
    
    # Evaluate each model
    results = []
    for info in tqdm(model_infos, desc="Evaluating models", disable=not verbose):
        try:
            model = load_model(info, device=device)
            metrics = evaluate_model(model, info, test_data, device=device)
            results.append(metrics)
            
            # Cleanup
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"ERROR evaluating {info.model_dir}: {e}")
            results.append({
                'model_dir': str(info.model_dir),
                'model_type': info.model_type,
                'error': str(e),
            })
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        # Display key metrics
        display_cols = ['model_type', 'hidden_dim', 'mse', 'log_likelihood', 'ks_statistic']
        if 'context_accuracy' in df.columns:
            display_cols.extend(['context_accuracy', 'context_log_prob'])
        
        available_cols = [c for c in display_cols if c in df.columns]
        print(df[available_cols].to_string(index=False))
        print("=" * 70)
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return df


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate multiple trained models on shared test data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate RNN and VRNN models
    python evaluate_models.py --model_dirs training_results/N_ctx_1/rnn_h16 training_results/N_ctx_1/vrnn_h16
    
    # Evaluate ModuleNetwork models with different objectives
    python evaluate_models.py --model_dirs training_results/N_ctx_2/NonHierarchicalGM/module_network_obs_bn16 \\
                                           training_results/N_ctx_2/NonHierarchicalGM/module_network_obs_ctx_kappa0.5_bn16
    
    # Custom test configuration
    python evaluate_models.py --model_dirs ... --n_samples 2000 --n_tones 500
    
    # Save results
    python evaluate_models.py --model_dirs ... --output results/evaluation.csv
        """
    )
    
    parser.add_argument('--model_dirs', type=str, nargs='+', required=True,
                        help='Paths to model directories to evaluate')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of test samples (default: 1000)')
    parser.add_argument('--n_tones', type=int, default=1000,
                        help='Sequence length (default: 1000)')
    parser.add_argument('--lr_id', type=int, default=0,
                        help='Learning rate index for loading weights (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results CSV')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    evaluate_models(
        model_dirs=[Path(d) for d in args.model_dirs],
        n_samples=args.n_samples,
        n_tones=args.n_tones,
        lr_id=args.lr_id,
        output_path=Path(args.output) if args.output else None,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
