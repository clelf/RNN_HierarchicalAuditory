"""
Unified model evaluation script.

This script loads multiple trained models and evaluates them on the same
generated test dataset, computing:
- MSE (Mean Squared Error)
- Log-likelihood (Gaussian log-probability)
- Calibration (Kolmogorov-Smirnov test for predicted variance)
- Context inference accuracy (for ModuleNetwork with N_ctx > 1)
- Context log-probability (for ModuleNetwork with N_ctx > 1)

Benchmark Comparison:
    The assess_model_against_benchmarks() function allows comparing model 
    performance against Kalman Filter (KF) benchmarks. Given pre-computed KF
    results, it computes:
    - Model vs KF MSE ratio (< 1 means model is better)
    - Model vs KF log-likelihood difference (> 0 means model is better)
    - Model vs KF calibration ratio (< 1 means model is better calibrated)
    
    This helps assess how difficult the problem was for the model relative
    to the optimal Bayesian estimator (KF).

Configuration Loading:
    Models trained with the updated pipeline automatically save a config JSON file
    (config.json) alongside the weights. This script will:
    1. Load the saved config if available (recommended for reproducibility)
    2. Use the exact training data parameters for test data generation
    3. Fall back to inferring config from directory structure if needed
    
    Each model folder should contain a single .pth weights file, which is
    auto-detected during model loading.

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
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy import stats, special
from tqdm import tqdm

# Local imports (files are now in the same directory)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from model import SimpleRNN, VRNN, ObsCtxModuleNetwork
from pipeline_next import extract_model_predictions, prepare_batch_data

# Local config (for loading saved configs)
from config_v2 import RunConfig, TrainingConfig, ModelArchConfig, DataConfig as TrainDataConfig

# Generative models (PreProParadigm is one level up from RNN/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PreProParadigm.audit_gm import NonHierarchicalAuditGM, HierarchicalAuditGM


# =============================================================================
# Learning Rate Detection
# =============================================================================

def find_lr_id_for_learning_rate(model_dir: Path, target_lr: float = 0.01, tolerance: float = 1e-6) -> Optional[int]:
    """
    Find the lr_id that corresponds to a specific learning rate.
    
    Parses training log files (training_log_lr{id}.txt) to find the lr_id
    that was trained with the target learning rate.
    
    Args:
        model_dir: Path to the model directory
        target_lr: Target learning rate to find (default: 0.01)
        tolerance: Tolerance for float comparison
    
    Returns:
        The lr_id that matches, or None if not found
    """
    model_dir = Path(model_dir)
    
    # Find all training log files
    log_files = list(model_dir.glob("training_log_lr*.txt"))
    
    for log_file in log_files:
        # Extract lr_id from filename
        filename = log_file.name  # e.g., "training_log_lr0.txt"
        try:
            lr_id = int(filename.split("_lr")[1].split(".txt")[0])
        except (IndexError, ValueError):
            continue
        
        # Read first line to get the learning rate
        try:
            with open(log_file, 'r') as f:
                first_line = f.readline().strip()
            
            # Parse "LR:  1e-02; epoch: ..." format
            if "LR:" in first_line:
                lr_str = first_line.split("LR:")[1].split(";")[0].strip()
                actual_lr = float(lr_str)
                
                if abs(actual_lr - target_lr) < tolerance:
                    return lr_id
        except (IOError, ValueError) as e:
            print(f"  Warning: Could not parse {log_file}: {e}")
            continue
    
    return None


def discover_models_with_lr(
    base_dir: Path, 
    target_lr: float = 0.01,
    verbose: bool = True
) -> List[Tuple[Path, int]]:
    """
    Discover all model directories and find the lr_id for a specific learning rate.
    
    DEPRECATED: Use discover_models() for folders with single weights files.
    
    Args:
        base_dir: Base directory containing model subdirectories
        target_lr: Target learning rate to find
        verbose: Print discovery information
    
    Returns:
        List of (model_dir, lr_id) tuples for models trained with target_lr
    """
    base_dir = Path(base_dir)
    results = []
    
    if verbose:
        print(f"\nDiscovering models trained with lr={target_lr}...")
    
    # Find all model directories (those containing weights files)
    for weights_file in base_dir.rglob("*_weights.pth"):
        model_dir = weights_file.parent
        
        # Skip if we've already processed this directory
        if any(d == model_dir for d, _ in results):
            continue
        
        lr_id = find_lr_id_for_learning_rate(model_dir, target_lr)
        
        if lr_id is not None:
            weights_path = model_dir / f"lr{lr_id}_weights.pth"
            if weights_path.exists():
                results.append((model_dir, lr_id))
                if verbose:
                    print(f"  ✓ {model_dir.name}: lr_id={lr_id}")
            else:
                if verbose:
                    print(f"  ✗ {model_dir.name}: Found lr_id={lr_id} but weights file missing")
        else:
            if verbose:
                print(f"  ✗ {model_dir.name}: No training with lr={target_lr} found")
    
    if verbose:
        print(f"  Found {len(results)} models trained with lr={target_lr}")
    
    return results


def discover_models(
    base_dir: Path,
    verbose: bool = True
) -> List[Path]:
    """
    Discover all model directories (those containing a .pth weights file).
    
    Each model directory is expected to contain a single weights file (.pth)
    and optionally a config.json file.
    
    Args:
        base_dir: Base directory to search for models
        verbose: Print discovery information
    
    Returns:
        List of model directory paths
    """
    base_dir = Path(base_dir)
    results = []
    
    if verbose:
        print(f"\nDiscovering models in {base_dir}...")
    
    # Find all directories containing weights files
    for weights_file in base_dir.rglob("*.pth"):
        model_dir = weights_file.parent
        
        # Skip if we've already processed this directory
        if model_dir in results:
            continue
        
        results.append(model_dir)
        if verbose:
            print(f"  ✓ {model_dir.name}")
    
    if verbose:
        print(f"  Found {len(results)} models")
    
    return results


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
    
    # ModuleNetwork specific
    learning_objective: str
    kappa: float
    bottleneck_dim: int
    
    # Full config if loaded from file
    run_config: Optional[RunConfig] = None
    
    # Data config for test data generation (parsed from run_config or defaults)
    data_config_dict: Optional[dict] = None
    
    @classmethod
    def from_path(cls, model_dir: Path) -> 'ModelInfo':
        """
        Load model configuration, preferring saved config file if available.
        
        Automatically finds the weights file in the directory (expects single .pth file).
        
        Priority:
        1. Load from config.json if it exists
        2. Load from legacy lr*_config.json as fallback
        3. Fall back to inferring from directory structure
        """
        model_dir = Path(model_dir)
        config_path = model_dir / 'config.json'
        
        # Find the weights file (expect single .pth file)
        weights_files = list(model_dir.glob('*.pth'))
        if not weights_files:
            raise FileNotFoundError(f"No weights file found in: {model_dir}")
        if len(weights_files) > 1:
            # If multiple, prefer the one without 'lr' prefix or just take first
            weights_path = weights_files[0]
        else:
            weights_path = weights_files[0]
        
        # Try to load from config.json first
        if config_path.exists():
            return cls._from_config_file(config_path, model_dir, weights_path)
        
        # Try to find legacy lr*_config.json files
        config_files = list(model_dir.glob('lr*_config.json'))
        if config_files:
            # Use the first available config file
            fallback_config = config_files[0]
            return cls._from_config_file(fallback_config, model_dir, weights_path)
        
        # Fall back to inferring from directory structure
        return cls._from_directory_structure(model_dir, weights_path)
    
    @classmethod
    def _from_config_file(cls, config_path: Path, model_dir: Path, 
                          weights_path: Path) -> 'ModelInfo':
        """Load ModelInfo from a saved config JSON file."""
        run_config = RunConfig.load(config_path)
        
        return cls(
            model_dir=model_dir,
            model_type=run_config.model_type,
            hidden_dim=run_config.hidden_dim,
            n_ctx=run_config.data.N_ctx,
            gm_name=run_config.data.gm_name,
            weights_path=weights_path,
            learning_objective=run_config.learning_objective,
            kappa=run_config.kappa,
            bottleneck_dim=run_config.bottleneck_dim or 16,
            run_config=run_config,
            data_config_dict=run_config.data.to_gm_dict(run_config.training.batch_size_test),
        )
    
    @classmethod
    def _from_directory_structure(cls, model_dir: Path, weights_path: Path) -> 'ModelInfo':
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
            )
        
        elif dir_name.startswith('module_network'):
            model_type = 'module_network'
            hidden_dim = 64  # Fixed for ModuleNetwork
            
            # Parse learning objective
            if 'obs_ctx' in dir_name:
                learning_objective = 'obs_ctx'
            elif 'ctx' in dir_name and 'obs' not in dir_name:
                learning_objective = 'ctx'
            else:
                learning_objective = 'obs'
            
            # Parse kappa
            kappa = 0.5
            if 'kappa' in dir_name:
                kappa_str = dir_name.split('kappa')[1].split('_')[0]
                kappa = float(kappa_str)
            
            # Parse bottleneck dimension
            # Default is 24 to match get_module_network_config() in config.py
            bottleneck_dim = 24
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
                learning_objective=learning_objective,
                kappa=kappa,
                bottleneck_dim=bottleneck_dim,
            )
        
        else:
            raise ValueError(f"Cannot parse model type from directory: {dir_name}")


# =============================================================================
# Model Loading
# =============================================================================

def infer_bottleneck_dim_from_weights(weights_path: Path) -> int:
    """
    Infer the bottleneck dimension from saved ModuleNetwork weights.
    
    The bottleneck dimension can be determined from the readout layer shapes.
    For ModuleNetwork, readout_obs2ctx is:
        nn.Linear(in_dim=2, bottleneck_dim)  -> weight shape (bottleneck_dim, 2)
        nn.ReLU()
        nn.Linear(bottleneck_dim, out_dim)   -> weight shape (out_dim, bottleneck_dim)
    
    So readout_obs2ctx.0.weight has shape (bottleneck_dim, 2).
    """
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # readout_obs2ctx.0.weight shape is (bottleneck_dim, in_dim)
    if 'readout_obs2ctx.0.weight' in state_dict:
        bottleneck_dim = state_dict['readout_obs2ctx.0.weight'].shape[0]
        return bottleneck_dim
    
    # Default fallback
    return 16


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
        # Use bottleneck_dim from ModelInfo (parsed from dir name or config file)
        # Default is 24 to match get_module_network_config() in config.py
        bottleneck_dim = info.bottleneck_dim
        
        # Safety: infer from weights if there's a mismatch (shouldn't happen normally)
        try:
            inferred_dim = infer_bottleneck_dim_from_weights(info.weights_path)
            if inferred_dim != bottleneck_dim:
                # The weights use a different dimension than expected
                print(f"  Warning: bottleneck_dim mismatch for {info.model_dir.name}: "
                      f"expected {bottleneck_dim}, weights have {inferred_dim}. Using inferred value.")
                bottleneck_dim = inferred_dim
        except Exception as e:
            # Fall back to parsed value
            pass
            pass
        
        config = {
            'kappa': info.kappa,
            'observation_module': {
                'input_dim': 1,
                'output_dim': 2,
                'rnn_hidden_dim': 64,
                'rnn_n_layers': 1,
                'bottleneck_dim': bottleneck_dim,
            },
            'context_module': {
                'input_dim': 2,
                'output_dim': info.n_ctx,
                'rnn_hidden_dim': 32,
                'rnn_n_layers': 1,
                'bottleneck_dim': bottleneck_dim,
            },
            'device': device,
        }
        model = ObsCtxModuleNetwork(config)
    
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
        gm = NonHierarchicalAuditGM(gm_dict)
    elif config.gm_name == 'HierarchicalGM':
        gm = HierarchicalAuditGM(gm_dict)
    else:
        raise ValueError(f"Unknown GM: {config.gm_name}")
    
    batch = gm.generate_batch(return_pars=True)
    y = batch['obs']
    contexts = batch['contexts']
    pars = batch.get('pars', None)
    
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

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray, reduce: bool = True) -> Union[np.ndarray, float]:
    """
    Compute Mean Squared Error.
    
    Args:
        y_true: True observations (n_samples, seq_len)
        y_pred: Predicted observations (n_samples, seq_len)
        reduce: If True, return scalar mean. If False, return per-sample MSE (n_samples,)
    
    Returns:
        Scalar MSE if reduce=True, else array of per-sample MSE values.
    """
    squared_errors = (y_true - y_pred) ** 2
    if reduce:
        return float(np.mean(squared_errors))
    else:
        # Mean over sequence dimension, keep sample dimension
        return np.mean(squared_errors, axis=1)


def compute_log_likelihood(y_true: np.ndarray, mu_pred: np.ndarray, var_pred: np.ndarray, 
                           reduce: bool = True) -> Union[np.ndarray, float]:
    """
    Compute Gaussian log-likelihood.
    
    log p(y|mu, var) = -0.5 * [log(2*pi*var) + (y-mu)^2/var]
    
    Args:
        y_true: True observations (n_samples, seq_len)
        mu_pred: Predicted mean (n_samples, seq_len)
        var_pred: Predicted variance (n_samples, seq_len)
        reduce: If True, return scalar mean. If False, return per-sample log-likelihood (n_samples,)
    
    Returns:
        Scalar mean log-likelihood if reduce=True, else array of per-sample values.
    """
    log_prob = -0.5 * (np.log(2 * np.pi * var_pred) + (y_true - mu_pred) ** 2 / var_pred)
    if reduce:
        return float(np.mean(log_prob))
    else:
        # Mean over sequence dimension, keep sample dimension
        return np.mean(log_prob, axis=1)


def compute_calibration_ks(y_true: np.ndarray, mu_pred: np.ndarray, var_pred: np.ndarray,
                           reduce: bool = True) -> Union[Tuple[float, float], np.ndarray]:
    """
    Compute calibration using Kolmogorov-Smirnov statistics.
    
    For a well-calibrated model, the z-scores should be standard normal.
    z = (y - mu) / sigma
    
    Two modes:
    - reduce=True (default): Global KS test pooling all data. Returns (ks_stat, p_value).
      This is a formal hypothesis test with p-value.
    
    - reduce=False: Per-sample KS statistics. Returns array of shape (n_samples,).
      Useful for reporting mean ± SEM across samples, which captures sample-to-sample
      variability in calibration. This is a descriptive statistic, not a hypothesis test.
    
    Args:
        y_true: True observations (n_samples, seq_len)
        mu_pred: Predicted mean (n_samples, seq_len)
        var_pred: Predicted variance (n_samples, seq_len)
        reduce: If True, compute global KS test. If False, compute per-sample KS stats.
    
    Returns:
        If reduce=True: (ks_statistic, p_value) tuple
        If reduce=False: Array of per-sample KS statistics (n_samples,)
        
        Lower KS statistic = better calibration
    """
    sigma_pred = np.sqrt(var_pred)
    
    if reduce:
        z_scores = (y_true - mu_pred) / sigma_pred
        # Flatten for KS test - pool all samples and timesteps
        z_flat = z_scores.flatten()
        # KS test against standard normal
        ks_stat, p_value = stats.kstest(z_flat, 'norm')
        return float(ks_stat), float(p_value)
    else:
        # Per-sample KS statistic using measure_KS_stat
        x = y_true[:, :, np.newaxis] if y_true.ndim == 2 else y_true
        u = mu_pred[:, :, np.newaxis] if mu_pred.ndim == 2 else mu_pred
        s = sigma_pred[:, :, np.newaxis] if sigma_pred.ndim == 2 else sigma_pred
        return measure_KS_stat(x, u, s)


def measure_KS_stat(x, u, s):
	""" Measures the calibration (K-S statistic) of a prediction on a set of observations 

		Parameters
		----------
		x : np.array
			Input array enconding sequences of observations; three dimensions: dim0 runs across the
			batch, dim1 runs across timepoints (i.e., trials in the COIN jargon), dim2 set to one.
		u  : np.array
			Mean of the predictions; shifted (i.e., u[:, t] are the predictions for t). Same 
			dimensions as x.
		s  : np.array
			Standard deviation of the predictions; same shape and shifting as u.

		Returns
		----------
		KS : np.array
			One dimensional array with a scalar per sample in the batch. 
	"""

	F    = np.linspace(0, 1, 1000)
	cump = (0.5 * (1 + special.erf((x - u) / (np.sqrt(2) * s))))
	N    = (~np.isnan(cump)).sum(1)
	KS   = abs(np.array([(cump <= f).sum(1) / N for f in F]) - F[:, None, None]).max((0, 2))
	
	return KS



def compute_context_accuracy(contexts_true: np.ndarray, context_logits: np.ndarray,
                             reduce: bool = True) -> Union[np.ndarray, float]:
    """
    Compute context inference accuracy.
    
    Args:
        contexts_true: True context labels (n_samples, seq_len)
        context_logits: Model context output logits (n_samples, seq_len, n_ctx)
        reduce: If True, return scalar mean. If False, return per-sample accuracy (n_samples,)
    
    Returns:
        Scalar accuracy if reduce=True, else array of per-sample accuracy values.
    """
    # Get predicted contexts (argmax over context dimension)
    contexts_pred = np.argmax(context_logits, axis=-1)
    correct = (contexts_pred == contexts_true)
    
    if reduce:
        return float(np.mean(correct))
    else:
        # Mean over sequence dimension, keep sample dimension
        return np.mean(correct, axis=1)


def compute_context_log_prob(contexts_true: np.ndarray, context_logits: np.ndarray,
                             reduce: bool = True) -> Union[np.ndarray, float]:
    """
    Compute log-probability of true contexts.
    
    Args:
        contexts_true: True context labels (n_samples, seq_len)
        context_logits: Model context output logits (n_samples, seq_len, n_ctx)
        reduce: If True, return scalar mean. If False, return per-sample log-prob (n_samples,)
    
    Returns:
        Scalar mean log-probability if reduce=True, else array of per-sample values.
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
    
    # Compute log probability
    log_probs = np.log(true_probs + 1e-10)  # Add small epsilon to avoid log(0)
    
    if reduce:
        return float(np.mean(log_probs))
    else:
        # Mean over sequence dimension, keep sample dimension
        return np.mean(log_probs, axis=1)


# =============================================================================
# Model Evaluation
# =============================================================================

def evaluate_model(
    model: nn.Module,
    info: ModelInfo,
    test_data: Dict[str, Any],
    device: str = 'cpu',
    reduce: bool = True,
    min_obs_for_em: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single model on the test data.
    
    Args:
        model: The model to evaluate
        info: ModelInfo with model metadata
        test_data: Dictionary with test data (y, y_np, contexts_np, etc.)
        device: Device to run evaluation on
        reduce: If True, return scalar metrics (mean over dataset).
                If False, return per-sample metric distributions as arrays (n_samples,).
                For calibration: reduce=True gives global KS test with p-value,
                reduce=False gives per-sample KS statistics (for mean ± SEM reporting).
        min_obs_for_em: If provided, restrict evaluation to the same timestep window
                        as the KF benchmark, i.e. y[min_obs_for_em:] only.
                        Model pred at index t predicts y[t+1], so we slice from
                        index min_obs_for_em-1 to get predictions for y[min_obs_for_em:].
                        Pass benchmark_data.min_obs_for_em when comparing against a KF
                        benchmark so that per-sample metrics are truly comparable.
    
    Returns:
        Dictionary with all computed metrics.
        If reduce=True: values are scalars (calibration includes ks_pvalue)
        If reduce=False: values are numpy arrays of shape (n_samples,)
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

    # Optionally restrict to the KF evaluation window (y[min_obs_for_em:]) so that
    # per-sample model metrics and KF metrics cover exactly the same timesteps.
    # mu_pred[:, t] predicts y[t+1], so slicing from (min_obs_for_em-1) gives
    # predictions aligned with y[min_obs_for_em:].
    if min_obs_for_em is not None:
        start = min_obs_for_em - 1  # model-prediction index → targets y[min_obs_for_em:]
        mu_pred   = mu_pred[:, start:]
        var_pred  = var_pred[:, start:]
        y_target  = y_target[:, start:]   # y_target[:,start:] == y_np[:,min_obs_for_em:]
        if context_output is not None:
            # context_output shape: (N, T-1, n_ctx); same slicing as predictions
            context_output_np = context_output.detach().cpu().numpy()[:, start:, :]
        # contexts_np[:,1:] aligned; slice the same way for context targets
        contexts_np_aligned = contexts_np[:, min_obs_for_em:] if contexts_np is not None else None
    else:
        if context_output is not None:
            context_output_np = context_output.detach().cpu().numpy()
        contexts_np_aligned = contexts_np[:, 1:] if contexts_np is not None else None
    
    # Compute observation metrics
    results = {
        'model_dir': str(info.model_dir),
        'model_type': info.model_type,
        'hidden_dim': info.hidden_dim,
        'n_ctx': info.n_ctx,
        'gm_name': info.gm_name,
    }
    
    # MSE
    results['mse'] = compute_mse(y_target, mu_pred, reduce=reduce)
    
    # Log-likelihood
    results['log_likelihood'] = compute_log_likelihood(y_target, mu_pred, var_pred, reduce=reduce)
    
    # Calibration (KS statistic)
    if reduce:
        # Global KS test with p-value
        ks_stat, ks_pval = compute_calibration_ks(y_target, mu_pred, var_pred, reduce=True)
        results['ks_statistic'] = ks_stat
        results['ks_pvalue'] = ks_pval
    else:
        # Per-sample KS statistics (for mean ± SEM reporting)
        results['ks_statistic'] = compute_calibration_ks(y_target, mu_pred, var_pred, reduce=False)
    
    # ModuleNetwork specific metrics
    if info.model_type == 'module_network' and info.n_ctx > 1 and context_output is not None:
        results['learning_objective'] = info.learning_objective
        results['kappa'] = info.kappa
        results['bottleneck_dim'] = info.bottleneck_dim
        
        # Context predictions – use pre-sliced arrays so timestep window matches y_target
        context_logits = context_output_np
        contexts_target = contexts_np_aligned
        
        # Context accuracy
        results['context_accuracy'] = compute_context_accuracy(contexts_target, context_logits, reduce=reduce)
        
        # Context log-probability
        results['context_log_prob'] = compute_context_log_prob(contexts_target, context_logits, reduce=reduce)
    
    return results


# =============================================================================
# Benchmark Comparison
# =============================================================================

@dataclass
class BenchmarkData:
    """Data loaded from benchmark files."""
    y: np.ndarray  # Observations (n_samples, seq_len)
    contexts: Optional[np.ndarray]  # Context labels (n_samples, seq_len) or None
    pars: Dict[str, Any]  # Generation parameters
    mu_kf: np.ndarray  # KF predicted mean (n_samples, seq_len_eval)
    std_kf: np.ndarray  # KF predicted std (n_samples, seq_len_eval)
    mse_kf: np.ndarray  # KF MSE per sample (n_samples,)
    min_obs_for_em: int  # Number of initial observations used for EM fitting
    
    @property
    def n_samples(self) -> int:
        return self.y.shape[0]
    
    @property
    def seq_len(self) -> int:
        return self.y.shape[1]
    
    @property
    def eval_seq_len(self) -> int:
        """Length of sequence used for evaluation (after EM warmup)."""
        return self.mu_kf.shape[1]


def load_benchmark_data(
    results_path: Path,
    input_data_path: Optional[Path] = None,
) -> BenchmarkData:
    """
    Load benchmark data from pickle files.
    
    The results file is required and must contain KF predictions. If it also
    contains the input data (y, contexts, pars), the separate input_data_path
    is not needed.
    
    Args:
        results_path: Path to results pickle containing:
            - 'y': observations (n_samples, seq_len)
            - 'mu_kal_pred': KF predicted mean (n_samples, eval_seq_len)
            - 'sigma_kal_pred': KF predicted std (n_samples, eval_seq_len)
            - 'min_obs_for_em': number of initial observations for EM
            - 'perf': KF MSE per sample (n_samples,)
            - 'contexts': context labels [optional]
            - 'pars': generation parameters [optional]
        input_data_path: Optional path to separate input data pickle containing:
            - 'y': observations (n_samples, seq_len)
            - 'contexts': context labels (n_samples, seq_len) [optional]
            - 'pars': generation parameters [optional]
            If provided, these values override those in results_path.
    
    Returns:
        BenchmarkData object with all loaded data.
    """
    import pickle
    
    # Load results (required)
    with open(results_path, 'rb') as f:
        results_data = pickle.load(f)
    
    # Load separate input data if provided
    if input_data_path is not None:
        with open(input_data_path, 'rb') as f:
            input_data = pickle.load(f)
    else:
        input_data = results_data  # Use results file as input data source
    
    # Extract input fields (prefer input_data, fall back to results_data)
    y = input_data.get('y', results_data.get('y'))
    if y is None:
        raise ValueError("Could not find 'y' in benchmark files")
    
    contexts = input_data.get('contexts', results_data.get('contexts', None))
    pars = input_data.get('pars', results_data.get('pars', {}))
    
    # Extract KF results
    mu_kf = results_data['mu_kal_pred']
    std_kf = results_data['sigma_kal_pred']
    min_obs_for_em = results_data['min_obs_for_em']
    mse_kf = results_data['perf']
    
    return BenchmarkData(
        y=y,
        contexts=contexts,
        pars=pars,
        mu_kf=mu_kf,
        std_kf=std_kf,
        mse_kf=mse_kf,
        min_obs_for_em=min_obs_for_em,
    )


def assess_model_against_benchmarks(
    model: nn.Module,
    info: ModelInfo,
    benchmark_data: BenchmarkData,
    device: str = 'cpu',
    reduce: bool = True,
) -> Dict[str, Any]:
    """
    Assess a model against Kalman Filter benchmarks.
    
    This function computes how difficult the problem was for the model relative
    to the Kalman Filter (KF) benchmark. It returns ratios of model metrics to
    KF metrics, allowing comparison across different difficulty levels.
    
    The KF predictions start from min_obs_for_em (after EM warmup), so model
    predictions are aligned to the same time range for fair comparison.
    
    Args:
        model: The trained model to evaluate
        info: ModelInfo with model metadata
        benchmark_data: BenchmarkData object containing:
            - y: observations
            - contexts: context labels (optional)
            - mu_kf, std_kf: KF predictions (mean and std)
            - mse_kf: KF MSE per sample
            - min_obs_for_em: EM warmup period
        device: Device to run evaluation on
        reduce: If True, return scalar metrics (mean over dataset).
                If False, return per-sample metric distributions as arrays (n_samples,).
    
    Returns:
        Dictionary with metrics including:
        - Model metrics (mse, log_likelihood, ks_statistic)
        - KF metrics (mse_kf, log_likelihood_kf, ks_statistic_kf)
        - Ratios (mse_ratio, log_likelihood_ratio, ks_ratio)
          - mse_ratio < 1 means model is better than KF
          - log_likelihood_ratio > 1 means model is better than KF
          - ks_ratio < 1 means model is better calibrated than KF
        - Context metrics (if applicable)
    """
    y = benchmark_data.y
    contexts = benchmark_data.contexts
    mu_kf = benchmark_data.mu_kf
    std_kf = benchmark_data.std_kf
    var_kf = std_kf ** 2
    mse_kf = benchmark_data.mse_kf
    min_obs_for_em = benchmark_data.min_obs_for_em
    
    # Convert observations to tensor for model
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)
    
    # Forward pass through model
    # Model predicts y[t+1] from y[0:t], so we feed y[:, :-1]
    with torch.no_grad():
        model_output = model(y_tensor[:, :-1, :])
        mu_model_full, var_model_full, context_output = extract_model_predictions(model, model_output)
    
    # Model predictions correspond to y[:, 1:] (predicting next observation)
    # Align with KF which starts from min_obs_for_em
    # KF predicts y[min_obs_for_em:], model predicts y[1:]
    # So model predictions for y[min_obs_for_em:] are at indices [min_obs_for_em-1:]
    
    # Target observations (what we're predicting)
    y_target_kf = y[:, min_obs_for_em:]  # KF evaluation range
    
    # Model predictions aligned to KF range
    # Model output at index t predicts y[t+1]
    # So to predict y[min_obs_for_em:], we need model output [min_obs_for_em-1:]
    model_start_idx = min_obs_for_em - 1
    mu_model = mu_model_full[:, model_start_idx:]
    var_model = var_model_full[:, model_start_idx:]
    
    # Ensure shapes match
    eval_len = min(y_target_kf.shape[1], mu_model.shape[1], mu_kf.shape[1])
    y_target_kf = y_target_kf[:, :eval_len]
    mu_model = mu_model[:, :eval_len]
    var_model = var_model[:, :eval_len]
    mu_kf = mu_kf[:, :eval_len]
    var_kf = var_kf[:, :eval_len]
    
    # Initialize results
    results = {
        'model_dir': str(info.model_dir),
        'model_type': info.model_type,
        'hidden_dim': info.hidden_dim,
        'n_ctx': info.n_ctx,
        'gm_name': info.gm_name,
        'min_obs_for_em': min_obs_for_em,
        'eval_seq_len': eval_len,
    }
    
    # --- Compute Model Metrics ---
    mse_model = compute_mse(y_target_kf, mu_model, reduce=reduce)
    ll_model = compute_log_likelihood(y_target_kf, mu_model, var_model, reduce=reduce)
    
    # --- Compute KF Metrics ---
    mse_kf_computed = compute_mse(y_target_kf, mu_kf, reduce=reduce)
    ll_kf = compute_log_likelihood(y_target_kf, mu_kf, var_kf, reduce=reduce)
    
    # --- Calibration ---
    if reduce:
        ks_model, ks_pval_model = compute_calibration_ks(y_target_kf, mu_model, var_model, reduce=True)
        ks_kf, ks_pval_kf = compute_calibration_ks(y_target_kf, mu_kf, var_kf, reduce=True)
        
        results['mse'] = mse_model
        results['log_likelihood'] = ll_model
        results['ks_statistic'] = ks_model
        results['ks_pvalue'] = ks_pval_model
        
        results['mse_kf'] = mse_kf_computed
        results['log_likelihood_kf'] = ll_kf
        results['ks_statistic_kf'] = ks_kf
        results['ks_pvalue_kf'] = ks_pval_kf
        
        # Compute ratios (model / KF)
        # mse_ratio < 1 means model is better
        results['mse_ratio'] = mse_model / (mse_kf_computed + 1e-10)
        # ll_ratio > 1 means model is better (higher log-likelihood)
        # Use difference for log-likelihood since ratio of logs is less interpretable
        results['log_likelihood_diff'] = ll_model - ll_kf
        # ks_ratio < 1 means model is better calibrated
        results['ks_ratio'] = ks_model / (ks_kf + 1e-10)
    else:
        ks_model = compute_calibration_ks(y_target_kf, mu_model, var_model, reduce=False)
        ks_kf = compute_calibration_ks(y_target_kf, mu_kf, var_kf, reduce=False)
        
        results['mse'] = mse_model
        results['log_likelihood'] = ll_model
        results['ks_statistic'] = ks_model
        
        results['mse_kf'] = mse_kf_computed
        results['log_likelihood_kf'] = ll_kf
        results['ks_statistic_kf'] = ks_kf
        
        # Compute per-sample ratios
        results['mse_ratio'] = mse_model / (mse_kf_computed + 1e-10)
        results['log_likelihood_diff'] = ll_model - ll_kf
        results['ks_ratio'] = ks_model / (ks_kf + 1e-10)
    
    # --- Context Metrics (if applicable) ---
    if info.model_type == 'module_network' and info.n_ctx > 1 and context_output is not None and contexts is not None:
        results['learning_objective'] = info.learning_objective
        results['kappa'] = info.kappa
        results['bottleneck_dim'] = info.bottleneck_dim
        
        # Context predictions aligned to evaluation range
        context_logits_full = context_output.detach().cpu().numpy()
        context_logits = context_logits_full[:, model_start_idx:model_start_idx + eval_len]
        
        # True contexts for evaluation range
        # Context at time t corresponds to observation y[t]
        contexts_target = contexts[:, min_obs_for_em:min_obs_for_em + eval_len]
        
        # Context accuracy
        results['context_accuracy'] = compute_context_accuracy(contexts_target, context_logits, reduce=reduce)
        
        # Context log-probability
        results['context_log_prob'] = compute_context_log_prob(contexts_target, context_logits, reduce=reduce)
    
    return results


def assess_models_against_benchmarks(
    model_dirs: List[Path],
    results_path: Path,
    input_data_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose: bool = True,
    reduce: bool = True,
) -> Union[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Assess multiple models against Kalman Filter benchmarks.
    
    Args:
        model_dirs: List of paths to model directories
        results_path: Path to benchmark results pickle (must contain y and KF predictions)
        input_data_path: Optional path to separate input data pickle. If None,
            input data (y, contexts, pars) is loaded from results_path.
        output_path: Optional path to save results CSV (only used if reduce=True)
        verbose: Print progress information
        reduce: If True, return DataFrame with scalar metrics.
                If False, return dict of model_name -> dict of metric distributions.
    
    Returns:
        If reduce=True: DataFrame with evaluation results for all models
        If reduce=False: Dict mapping model names to dicts of metric arrays
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print("\n" + "=" * 70)
        print("MODEL VS BENCHMARK EVALUATION")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Number of models: {len(model_dirs)}")
        print(f"Benchmark results: {results_path}")
        if input_data_path:
            print(f"Benchmark input (separate): {input_data_path}")
        print("=" * 70)
    
    # Load benchmark data
    if verbose:
        print("\nLoading benchmark data...")
    benchmark_data = load_benchmark_data(results_path, input_data_path)
    if verbose:
        print(f"  Loaded {benchmark_data.n_samples} samples")
        print(f"  Sequence length: {benchmark_data.seq_len}")
        print(f"  Evaluation length (after EM): {benchmark_data.eval_seq_len}")
        print(f"  min_obs_for_em: {benchmark_data.min_obs_for_em}")
    
    # Parse model information
    model_infos = []
    for model_dir in model_dirs:
        try:
            info = ModelInfo.from_path(Path(model_dir))
            model_infos.append(info)
            if verbose:
                print(f"  Found: {info.model_type} (h={info.hidden_dim}, n_ctx={info.n_ctx})")
        except Exception as e:
            print(f"  ERROR parsing {model_dir}: {e}")
    
    if not model_infos:
        raise ValueError("No valid models found!")
    
    # Evaluate each model
    results = []
    results_unreduced = {}
    
    for info in tqdm(model_infos, desc="Evaluating models vs benchmarks", disable=not verbose):
        try:
            model = load_model(info, device=device)
            metrics = assess_model_against_benchmarks(
                model, info, benchmark_data, device=device, reduce=reduce
            )
            
            if reduce:
                results.append(metrics)
            else:
                model_name = info.model_dir.name
                results_unreduced[model_name] = metrics
            
            # Cleanup
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"ERROR evaluating {info.model_dir}: {e}")
            import traceback
            traceback.print_exc()
            if reduce:
                results.append({
                    'model_dir': str(info.model_dir),
                    'model_type': info.model_type,
                    'error': str(e),
                })
            else:
                model_name = info.model_dir.name
                results_unreduced[model_name] = {'error': str(e)}
    
    # Return based on reduce mode
    if not reduce:
        if verbose:
            print("\n" + "=" * 70)
            print("EVALUATION COMPLETE (unreduced mode)")
            print("=" * 70)
            print(f"Returned dict with {len(results_unreduced)} models")
            print(f"Each model has metric arrays of shape ({benchmark_data.n_samples},)")
            print("=" * 70)
        return results_unreduced
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK COMPARISON RESULTS")
        print("=" * 70)
        
        display_cols = ['model_type', 'hidden_dim', 'mse', 'mse_kf', 'mse_ratio', 
                        'log_likelihood', 'log_likelihood_kf', 'ks_statistic', 'ks_ratio']
        if 'context_accuracy' in df.columns:
            display_cols.append('context_accuracy')
        
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
# Main Evaluation Pipeline
# =============================================================================

def evaluate_models(
    model_dirs: List[Path] = None,
    n_samples: int = 1000,
    n_tones: int = 1000,
    output_path: Optional[Path] = None,
    verbose: bool = True,
    reduce: bool = True,
) -> Union[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Evaluate multiple models on a shared test dataset.
    
    Args:
        model_dirs: List of paths to model directories
        n_samples: Number of test samples to generate
        n_tones: Sequence length
        output_path: Optional path to save results CSV (only used if reduce=True)
        verbose: Print progress information
        reduce: If True, return DataFrame with scalar metrics (default behavior).
                If False, return dict of model_name -> dict of metric distributions.
    
    Returns:
        If reduce=True: DataFrame with evaluation results for all models (scalars)
        If reduce=False: Dict mapping model names to dicts of metric arrays (n_samples,)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Validate input
    if model_dirs is None or len(model_dirs) == 0:
        raise ValueError("Must provide model_dirs")
    
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
            info = ModelInfo.from_path(Path(model_dir))
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
                print(f"  Using saved data config from: {info.model_dir}")
                print(f"    - N_tones: {saved_data_config.get('N_tones', 'N/A')}")
                print(f"    - mu_tau_bounds: {saved_data_config.get('mu_tau_bounds', 'N/A')}")
                print(f"    - si_stat_bounds: {saved_data_config.get('si_stat_bounds', 'N/A')}")
                print(f"    - params_testing: {saved_data_config.get('params_testing', 'N/A')}")
            break
    
    if saved_data_config is not None:
        # Use saved config (ensures same params_testing bounds, etc.)
        data_config = TestDataConfig.from_saved_config(saved_data_config, n_samples=n_samples)
    else:
        # Fall back to defaults
        if verbose:
            print(f"  No saved config found, using defaults")
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
    results_unreduced = {}  # For reduce=False mode
    
    for info in tqdm(model_infos, desc="Evaluating models", disable=not verbose):
        try:
            model = load_model(info, device=device)
            metrics = evaluate_model(model, info, test_data, device=device, reduce=reduce)
            
            if reduce:
                results.append(metrics)
            else:
                # Use model directory name as key
                model_name = info.model_dir.name
                results_unreduced[model_name] = metrics
            
            # Cleanup
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"ERROR evaluating {info.model_dir}: {e}")
            if reduce:
                results.append({
                    'model_dir': str(info.model_dir),
                    'model_type': info.model_type,
                    'error': str(e),
                })
            else:
                model_name = info.model_dir.name
                results_unreduced[model_name] = {'error': str(e)}
    
    # Return based on reduce mode
    if not reduce:
        if verbose:
            print("\n" + "=" * 70)
            print("EVALUATION COMPLETE (unreduced mode)")
            print("=" * 70)
            print(f"Returned dict with {len(results_unreduced)} models")
            print(f"Each model has metric arrays of shape ({n_samples},)")
            print("=" * 70)
        return results_unreduced
    
    # Create results DataFrame (reduce=True mode)
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
    # Evaluate all models in a base directory (auto-discovers .pth files)
    python evaluate_models.py --base_dir training_results_CORRECT/N_ctx_2/NonHierarchicalGM_selected
    
    # Evaluate specific model directories
    python evaluate_models.py --model_dirs training_results/N_ctx_1/rnn_h16 training_results/N_ctx_1/vrnn_h16
    
    # Evaluate ModuleNetwork models with different objectives
    python evaluate_models.py --model_dirs training_results/N_ctx_2/NonHierarchicalGM/module_network_obs_bn16 \\
                                           training_results/N_ctx_2/NonHierarchicalGM/module_network_obs_ctx_kappa0.5_bn16
    
    # Custom test configuration
    python evaluate_models.py --base_dir training_results/N_ctx_2 --n_samples 2000 --n_tones 500
    
    # Save results
    python evaluate_models.py --base_dir training_results/N_ctx_2 --output results/evaluation.csv
        """
    )
    
    # Model selection
    parser.add_argument('--base_dir', type=str, default='../training_results_CORRECT/N_ctx_2/NonHierarchicalGM_selected',
                        help='Base directory to search for models (recursively finds all .pth files)')
    parser.add_argument('--model_dirs', type=str, nargs='+', default=None,
                        help='Paths to specific model directories to evaluate')
    
    # Test configuration
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of test samples (default: 1000)')
    parser.add_argument('--n_tones', type=int, default=1000,
                        help='Sequence length (default: 1000)')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results CSV')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet
    
    # Determine model selection method
    if args.model_dirs is not None:
        # Use explicit model directories
        model_dirs = [Path(d) for d in args.model_dirs]
    elif args.base_dir is not None:
        # Discover models in base directory
        model_dirs = discover_models(
            Path(args.base_dir),
            verbose=verbose
        )
        
        if not model_dirs:
            print(f"ERROR: No models found in {args.base_dir}")
            sys.exit(1)
    else:
        print("ERROR: Must provide either --base_dir or --model_dirs")
        sys.exit(1)
    
    evaluate_models(
        model_dirs=model_dirs,
        n_samples=args.n_samples,
        n_tones=args.n_tones,
        output_path=Path(args.output) if args.output else None,
        verbose=verbose,
    )


if __name__ == '__main__':
    main()
