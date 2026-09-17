"""Core primitives shared by every model_analysis script.

Four groups, in order:

  1. Model access      -- ModelInfo (config discovery + loading), load_model,
                          discover_models, infer_bottleneck_dim_from_weights.
  2. Synthetic data    -- generate_test_data, from a model's own training config.
  3. Sequence I/O and  -- find_trial_files, select_files, load_trial_sequence,
     the forward pass     to_model_tensors, load_trial_params, dpos_conventions,
                          run_forward_pass, get_module_output_and_activity,
                          get_module_probabilities,
                          module_probabilities_from_output.
  4. Numerics          -- gaussian_likelihood, class_likelihood,
                          compute_derivatives, gather_at.

Everything here was previously split between evaluate_models.py and
model_activations.py; the bodies are unchanged. Anything that draws a figure
lives in plots.py, anything that scores a model lives in metrics.py, and anything
specific to the experimental sequences lives in exp_sequence_analysis.py.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

# Set up sys.path before the local imports:
#   RNN/        -- model.py
#   RNN/train/  -- pipeline_core_v2.py, config_v2.py
#   Workspace/  -- PreProParadigm and Kalman packages
_here = os.path.abspath(os.path.dirname(__file__))
_rnn_dir = os.path.abspath(os.path.join(_here, '..'))
_train_dir = os.path.abspath(os.path.join(_here, '..', 'train'))
_workspace = os.path.abspath(os.path.join(_here, '..', '..', '..'))
for _p in [_workspace, _train_dir, _rnn_dir, _here]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import SimpleRNN, VRNN, ObsCtxModuleNetwork, PopulationNetwork
from pipeline_core_v2 import (
    get_model_predictions,
    _create_module_network_config,
    _create_population_network_config,
)
from config_v2 import RunConfig, DataConfig as TrainDataConfig
from PreProParadigm.audit_gm import NonHierarchicalAuditGM, HierarchicalAuditGM

from analysis_config import CUE_LABELS, EXPERIMENTAL_DPOS_MIN


# =============================================================================
# Model access
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
        config_files = list(model_dir.glob('*config*.json'))
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
        # Use config from file if available, otherwise fall back to inference
        if info.run_config is not None:
            # Build config using the standard function from pipeline_core_v2
            config = _create_module_network_config(info.run_config)
            config['device'] = device
        else:
            # Fall back to inferring from directory structure
            # This path is for legacy models without config files
            bottleneck_dim = info.bottleneck_dim
            
            # Safety: infer from weights if there's a mismatch
            inferred_dim = infer_bottleneck_dim_from_weights(info.weights_path)
            if inferred_dim != bottleneck_dim:
                print(f"  Warning: bottleneck_dim mismatch for {info.model_dir.name}: "
                        f"expected {bottleneck_dim}, weights have {inferred_dim}. Using inferred value.")
                bottleneck_dim = inferred_dim
            
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
    
    
    elif info.model_type == 'population_network':
        # Build config 
        config = _create_population_network_config(info.run_config)
        config['device'] = device
        # Create model
        model = PopulationNetwork(config)


    else:
        raise ValueError(f"Unknown model type: {info.model_type}")
    
    # Load weights
    model.load_state_dict(torch.load(info.weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


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
# Synthetic test data
# =============================================================================

def generate_test_data(data_config: Union[TrainDataConfig, dict], n_samples: int, 
                       device: str = 'cpu') -> Dict[str, Any]:
    """
    Generate a shared test dataset using training data config.
    
    Harmonized with training data generation (prepare_batch_data) to ensure consistent
    tensor types, shapes, and field naming for both forward passes and evaluation.
    
    Parameters
    ----------
    data_config : TrainDataConfig or dict
        Either a DataConfig object from config_v2, or a dict (from to_gm_dict()).
        Using TrainDataConfig is recommended as it handles all GM types (including HierarchicalGM).
    n_samples : int
        Number of samples to generate for the test set.
    device : str
        Device to place tensors on ('cpu' or 'cuda').
    
    Returns
    -------
    dict
        Dictionary containing (aligned with prepare_batch_data):
        
        Core fields (always present):
        - 'y': Observations tensor (n_samples, seq_len, 1) [float32]
        - 'y_np': Observations as numpy array (n_samples, seq_len)
        - 'pars': Generation parameters dict
        
        Context fields (n_ctx > 1):
        - 'contexts': Context labels tensor (n_samples, seq_len) [long]
        - 'contexts_np': Context labels as numpy array
        
        HierarchicalGM-specific fields (HierarchicalGM only):
        - 'rules': Active rules unsqueezed (n_samples, seq_len, 1) [long] ← used in forward pass
        - 'rules_np': Same as numpy array
        - 'dpos': Deviant positions unsqueezed (n_samples, seq_len, 1) [long] ← used in loss computation  
        - 'dpos_np': Same as numpy array
        - 'q': Cues converted to one-hot encoding (n_samples, seq_len, n_cues) [float32] ← used in forward pass
        - 'q_np': Original cue indices as numpy array
        - 'timbres': Object identities (n_samples, seq_len)
        - 'timbres_np': Same as numpy array
        - 'pi_rules': Rule transition probabilities
        - (and other hierarchical fields as generated by HierarchicalAuditGM)
    
    Notes
    -----
    CRITICAL ALIGNMENT WITH TRAINING:
    This function ensures test data matches training data structure exactly:
    1. Fields used in forward pass (y, q) are float32 to work with GRU layers
    2. Integer fields (rules, dpos) are unsqueezed to (batch, seq_len, 1) shape
    3. Cues are one-hot encoded into 'q' field (not raw integers)
    4. Both tensor and numpy versions provided for all fields
    
    For HierarchicalGM, all required parameters (rules_dpos_set, mu_rho_rules, 
    si_rho_rules, p_cues, cues_set) must be present in the data config.
    """
    # Convert DataConfig to gm_dict, overriding sample count
    if isinstance(data_config, TrainDataConfig):
        gm_dict = data_config.to_gm_dict(n_samples)
    elif isinstance(data_config, dict):
        # Assume it's already a gm_dict; update sample count
        gm_dict = data_config.copy()
        gm_dict['N_samples'] = n_samples
    else:
        raise TypeError(f"data_config must be TrainDataConfig or dict, got {type(data_config)}")
    
    gm_name = gm_dict['gm_name']
    
    # Instantiate the appropriate generative model
    if gm_name == 'NonHierarchicalGM':
        gm = NonHierarchicalAuditGM(gm_dict)
    elif gm_name == 'HierarchicalGM':
        gm = HierarchicalAuditGM(gm_dict)
    else:
        raise ValueError(f"Unknown GM: {gm_name}")
    
    # Generate batch
    batch = gm.generate_batch(return_pars=True)
    
    # Extract observations and convert to tensor
    y = batch['obs']
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)
    
    # Initialize result with observations and parameters
    result = {
        'y': y_tensor,
        'y_np': y,
        'pars': batch['pars'],
    }
    
    # Handle contexts (present in both NonHierarchicalGM and HierarchicalGM)
    contexts = batch['contexts'] if 'contexts' in batch.keys() else None
    n_ctx = gm_dict['N_ctx']
    if n_ctx > 1 and contexts is not None:
        contexts_tensor = torch.tensor(contexts, dtype=torch.long).to(device)
        result['contexts'] = contexts_tensor
        result['contexts_np'] = contexts
    else:
        result['contexts'] = None
        result['contexts_np'] = None
    
    # ========== HierarchicalGM-specific field processing ==========
    # Align with prepare_batch_data() to ensure consistent tensor types and shapes
    # for both training and testing
    if gm_name == 'HierarchicalGM':
        # 1) Process rules_long: convert to long, unsqueeze, and store as 'rules'
        #    (matches training where only rules_long is used and unsqueezed)
        if 'rules_long' in batch:
            rules_data = batch['rules_long']
            result['rules'] = torch.tensor(rules_data, dtype=torch.long, requires_grad=False).unsqueeze(2).to(device)
            result['rules_np'] = rules_data
        
        # 2) Process dpos_long: convert to long, unsqueeze, and store as 'dpos'
        #    (matches training where only dpos_long is used and unsqueezed)
        if 'dpos_long' in batch:
            dpos_data = batch['dpos_long']
            result['dpos'] = torch.tensor(dpos_data, dtype=torch.long, requires_grad=False).unsqueeze(2).to(device)
            result['dpos_np'] = dpos_data
        
        # 3) Process cues_long: convert to one-hot encoding (float32) and store as 'q'
        #    (matches training where cues_long is converted to one-hot and stored as 'q')
        if 'cues_long' in batch:
            cues_data = batch['cues_long']
            q_onehot = torch.nn.functional.one_hot(
                torch.tensor(cues_data, dtype=torch.long, requires_grad=False),
                num_classes=gm.N_cues
            ).float().to(device)
            result['q'] = q_onehot  # Shape: (n_samples, seq_len, n_cues)
            result['q_np'] = cues_data
        
        # 4) Store other hierarchical fields for analysis (numpy + tensor versions)
        #    These are not used in forward pass but may be needed for evaluation
        other_fields = ['timbres', 'timbres_long', 'pi_rules']
        for field in other_fields:
            if field in batch:
                field_data = batch[field]
                # Keep as appropriate dtype
                if field_data.dtype in [np.int32, np.int64]:
                    result[field] = torch.tensor(field_data, dtype=torch.long).to(device)
                else:
                    result[field] = torch.tensor(field_data, dtype=torch.float32).to(device)
                result[f'{field}_np'] = field_data
    
    return result


# =============================================================================
# Experimental sequence files
# =============================================================================

def find_trial_files(trials_path):
    """Sequence files in `trials_path`, preferring .csv and falling back to .txt."""
    files = sorted(trials_path.glob("*.csv"))
    if not files:
        files = sorted(trials_path.glob("*.txt"))
    return files


def select_files(all_files, n_sequences, seed=0):
    """All files, or `n_sequences` of them sampled without replacement."""
    if n_sequences is None or n_sequences >= len(all_files):
        return list(all_files)
    rng = np.random.default_rng(seed=seed)
    return sorted(rng.choice(all_files, size=n_sequences, replace=False).tolist())


# =============================================================================
# Sequence I/O and model conventions
# =============================================================================

def load_trial_sequence(filepath, return_hierarch=False, n_cue_classes=None, cue_seed=None):
    """Load a trial CSV and return observation array and one-hot cue array (T, n).

    Parameters
    ----------
    filepath : str or Path
        Path to the trial sequence CSV.
    return_labels : bool
        If True, also return the ground-truth class labels for the categorical
        modules, inserted right after the cue array:
        (obs, cue, ctx, dpos, rule, lim_std, d, tau_std, trial_n).
        - ctx comes from the 'trial_type' column (the context label)
        - dpos comes from the 'dpos' column (raw deviant positions, not yet
          shifted to 0-based class indices)
        - rule comes from the 'rule' column
        Each is an int64 array of shape (T,).
    n_cue_classes : int, optional
        Number of cue classes to encode. The experimental files only ever use the
        two cues in ``CUE_LABELS``, but some models were trained with a larger cue
        set (e.g. cues_set = 0,...,11). When ``n_cue_classes`` is larger than
        ``len(CUE_LABELS)``, the two cues in the sequence are each assigned a
        distinct, randomly chosen class out of ``range(n_cue_classes)`` (assignment
        preserves ``CUE_LABELS`` order), and ``cue`` is returned as a
        (T, n_cue_classes) one-hot the model can consume. Defaults to the native
        two-class encoding.
    cue_seed : int, optional
        Seed for the random class assignment when ``n_cue_classes`` is set. ``None``
        (default) draws a fresh assignment each call.

    Returns
    -------
    By default: (obs, cue, lim_std, d, tau_std, trial_n).
    With return_labels=True: (obs, cue, ctx, dpos, rule, lim_std, d, tau_std, trial_n).
    """
    df = pd.read_csv(filepath)
    obs = df['observation'].to_numpy(dtype=np.float32)
    cue_raw = df['cue'].to_numpy()
    label_to_idx = {label: i for i, label in enumerate(CUE_LABELS)}
    cue_idx = np.vectorize(label_to_idx.get)(cue_raw)

    n_native = len(CUE_LABELS)
    if n_cue_classes is None or n_cue_classes == n_native:
        cue = np.eye(n_native, dtype=np.float32)[cue_idx]  # (T, n_native)
    else:
        if n_cue_classes < n_native:
            raise ValueError(
                f"n_cue_classes ({n_cue_classes}) must be >= number of cue labels "
                f"({n_native})")
        # Map each native cue to a distinct random class; sort so the smaller class
        # index goes to CUE_LABELS[0], keeping the column ordering recoverable.
        rng = np.random.default_rng(cue_seed)
        sel = np.sort(rng.choice(n_cue_classes, size=n_native, replace=False))
        cue = np.zeros((len(cue_idx), n_cue_classes), dtype=np.float32)  # (T, n_cue_classes)
        cue[np.arange(len(cue_idx)), sel[cue_idx]] = 1.0
    trial_n = df['trial_n']
    lim_std = df['lim_std'].iloc[0]
    d = df['d'].iloc[0]
    tau_std = df['tau_std'].iloc[0]
    if return_hierarch:
        ctx = df['trial_type'].to_numpy(dtype=np.int64)   # context label
        dpos = df['dpos'].to_numpy(dtype=np.int64)        # raw deviant position
        rule = df['rule'].to_numpy(dtype=np.int64)
        return obs, cue, ctx, dpos, rule, lim_std, d, tau_std, trial_n
    return obs, cue, lim_std, d, tau_std, trial_n


def to_model_tensors(obs, cue):
    """Convert observation (T,) and one-hot cue (T, 2) arrays to (1, T, dim) float tensors."""
    y = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    q = torch.tensor(cue, dtype=torch.float32).unsqueeze(0)                # (1, T, 2)
    return y, q


def load_trial_params(filepath):
    """Extract scalar trial parameters from a sequence CSV file.

    Reads the first row (parameters are constant within a trial) and returns a
    dict compatible with extract_sample_parameters, including si_stat derived
    from d = |lim_std - lim_dev| / sqrt(si_stat^2 + si_r^2).

    Returns
    -------
    dict with keys: 'tau', 'lim', 'si_stat', 'si_r'
    """
    row = pd.read_csv(filepath, nrows=1).iloc[0]
    d       = float(row['d'])
    tau     = float(row['tau_std'])
    lim_std = float(row['lim_std'])
    lim_dev = float(row['lim_dev'])
    si_r    = float(row['sigma_r'])

    numerator = np.abs(lim_std - lim_dev)
    si_stat = np.sqrt(max(0.0, (numerator / (d + 1e-8)) ** 2 - si_r ** 2))

    return {
        'tau':     tau,
        'lim':     [lim_std, lim_dev],
        'si_stat': si_stat,
        'si_r':    si_r,
    }


def dpos_conventions(info, experimental_dpos_min=EXPERIMENTAL_DPOS_MIN):
    """dpos alignment offsets between a trained model and the experimental files.

    A model's dpos output class ``c`` encodes deviant position ``c + dpos_min``, where
    ``dpos_min = min(rules_dpos_set)`` is read from the model's saved config
    (``info.data_config_dict``). Returns ``(dpos_min, shift)`` with
    ``shift = dpos_min - experimental_dpos_min``: add ``shift`` to an experimental dpos
    to express it in the model's convention, and the class index of a model-convention
    dpos is ``dpos - dpos_min``.

    Self-adjusting: the old lr0 model (rules_dpos_set=[[3,4,5],[5,6,7]]) yields (3, 1);
    a model retrained on [[2,3,4],[4,5,6]] yields (2, 0).
    """
    rules_dpos_set = np.asarray(info.data_config_dict['rules_dpos_set'])
    dpos_min = int(rules_dpos_set.min())
    return dpos_min, dpos_min - experimental_dpos_min


# =============================================================================
# Forward pass
# =============================================================================

def run_forward_pass(model, y, q, return_prior=False):
    """Run a forward pass and return raw module outputs and hidden states per module.

    Parameters
    ----------
    model : nn.Module
        Trained model that accepts return_hidden=True.
    y : torch.Tensor
        Observation sequences, shape (batch, seq_len, obs_dim).
    q : torch.Tensor
        Query sequences, shape (batch, seq_len, q_dim).
    return_prior : bool
        If True, also return the prior (first-call) module readouts, i.e. the
        outputs produced before the feedback sweep has reached each module.

    Returns
    -------
    prob_output : dict
        Module name → posterior readout tensor, shape (batch, seq_len, out_dim).
    hidden_states : dict
        Module name → tensor of shape (seq_len, n_layers, batch, hidden_dim).
    prior_output : dict
        Module name → prior readout tensor, same shape as prob_output. Returned
        only when return_prior=True.
    """
    with torch.no_grad():
        forward_output = model(y[:, :-1, :], q[:, :-1, :],
                               return_hidden=True, return_prior=return_prior)

    # The forward output is made of groups of four (one entry per module), in the
    # order: posterior readouts, prior readouts (if asked for), hidden states.
    obs_outputs, ctx_outputs, dpos_outputs, rule_outputs = forward_output[:4]
    obs_hidden, ctx_hidden, dpos_hidden, rule_hidden = forward_output[-4:]

    prob_output = {
        'obs':  obs_outputs,
        'ctx':  ctx_outputs,
        'dpos': dpos_outputs,
        'rule': rule_outputs,
    }

    hidden_states ={
        'obs':  obs_hidden,
        'ctx':  ctx_hidden,
        'dpos': dpos_hidden,
        'rule': rule_hidden,
    }

    if return_prior:
        prior_obs, prior_ctx, prior_dpos, prior_rule = forward_output[4:8]
        prior_output = {
            'obs':  prior_obs,
            'ctx':  prior_ctx,
            'dpos': prior_dpos,
            'rule': prior_rule,
        }
        return prob_output, hidden_states, prior_output

    return prob_output, hidden_states


def compute_hidden_norms(hidden_states, layer_idx=-1):
    """Compute L2 norms across neuron units for one layer of each module.

    Parameters
    ----------
    hidden_states : dict
        Module name → tensor of shape (seq_len, n_layers, batch, hidden_dim).
    layer_idx : int
        Index of the layer to extract. Default: -1 (last layer).

    Returns
    -------
    dict
        Module name → ndarray of shape (seq_len, batch).
    """
    norms = {}
    for module_name, hidden in hidden_states.items():
        layer_hidden = hidden[:, layer_idx, :, :].detach().cpu().numpy()  # (seq_len, batch, hidden_dim)
        norms[module_name] = np.linalg.norm(layer_hidden, axis=2)         # (seq_len, batch)
    return norms


def get_module_output_and_activity(model, y, q, layer_idx=-1, return_prior=False):
    """Return per-module hidden activity norms and their temporal derivatives.

    Runs a single forward pass with return_hidden=True, selects the requested
    layer, reduces across neuron units with an L2 norm, and differentiates.

    Parameters
    ----------
    model : nn.Module
        Trained model that accepts return_hidden=True.
    y : torch.Tensor
        Observation sequences, shape (batch, seq_len, obs_dim).
    q : torch.Tensor
        Query sequences, shape (batch, seq_len, q_dim).
    layer_idx : int
        Which RNN layer to extract. Default: -1 (last layer).
    return_prior : bool
        If True, also return the prior (first-call) module readouts as a fourth
        element.

    Returns
    -------
    prob_output : dict
        Module name → posterior readout tensor.
    hidden_activity : dict
        Module name → ndarray of shape (seq_len, batch).
    hidden_derivatives : dict
        Module name → ndarray of shape (seq_len-1, batch).
    prior_output : dict
        Module name → prior readout tensor. Returned only when return_prior=True.
    """
    if return_prior:
        prob_output, hidden_states, prior_output = run_forward_pass(model, y, q, return_prior=True)
    else:
        prob_output, hidden_states = run_forward_pass(model, y, q)
    hidden_activity = compute_hidden_norms(hidden_states, layer_idx=layer_idx)
    hidden_derivatives = {name: compute_derivatives(norms) for name, norms in hidden_activity.items()}
    if return_prior:
        return prob_output, hidden_activity, hidden_derivatives, prior_output
    return prob_output, hidden_activity, hidden_derivatives


def module_probabilities_from_output(model, model_output, prior=False):
    """Post-process one forward output into per-module probabilities.

    Shared by get_module_probabilities for the posterior and the prior readouts;
    the post-processing itself is delegated to pipeline_core_v2.get_model_predictions
    so both passes are read exactly as in training and evaluation.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    model_output : tuple
        Output of the model's forward pass.
    prior : bool
        If True, read the prior (first-call) readouts, which requires model_output
        to come from forward(..., return_prior=True).

    Returns
    -------
    dict
        Module name → ndarray of shape (seq_len, batch, dim).
    """
    pred = get_model_predictions(model, model_output, prior=prior)

    # get_model_predictions returns batch-major arrays:
    #   mu_estim / var_estim : (batch, seq_len)
    #   *_prob               : (batch, seq_len, n_classes)
    # Stack obs into (batch, seq_len, 2) then move to the (seq_len, batch, dim)
    # convention used elsewhere (time-major, batch second).
    obs = np.stack([pred['mu_estim'], pred['var_estim']], axis=-1)  # (batch, seq_len, 2)

    return {
        'obs':  np.transpose(obs, (1, 0, 2)),
        'ctx':  np.transpose(pred['ctx_prob'], (1, 0, 2)),
        'dpos': np.transpose(pred['dpos_prob'], (1, 0, 2)),
        'rule': np.transpose(pred['rule_prob'], (1, 0, 2)),
    }


def get_module_probabilities(model, y, q, return_prior=False):
    """Return per-module output probabilities/distribution parameters.

    Runs a standard forward pass and delegates the output post-processing to
    pipeline_core_v2.get_model_predictions, so the observation mean/variance and
    the class probabilities are computed exactly as in training and evaluation
    (single source of truth). In particular:

    - 'obs' is a regressor: the returned columns are (mean, variance), where the
      variance is get_model_predictions' var_estim = softplus(raw) + 1e-6. This
      is a *variance* (the same quantity passed to GaussianNLLLoss at training
      time), not a standard deviation.
    - the remaining modules are classifiers: softmax class probabilities.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    y : torch.Tensor
        Observation sequences, shape (batch, seq_len, obs_dim).
    q : torch.Tensor
        Query sequences, shape (batch, seq_len, q_dim).
    return_prior : bool
        If True, also return the same quantities computed on the prior
        (first-call) readouts, as a second dict.

    Returns
    -------
    probabilities : dict
        Module name → ndarray of shape (seq_len, batch, dim), from the posterior
        readouts:
        - 'obs':  dim = 2, columns are (mean, variance)
        - others: dim = n_classes, softmax class probabilities
    prior_probabilities : dict
        Same, from the prior readouts. Returned only when return_prior=True.
    """
    with torch.no_grad():
        model_output = model(y[:, :-1, :], q[:, :-1, :], return_prior=return_prior)

    probabilities = module_probabilities_from_output(model, model_output, prior=False)
    if return_prior:
        return probabilities, module_probabilities_from_output(model, model_output, prior=True)
    return probabilities


def extract_sample_parameters(pars, sample_idx):
    """Extract sample parameters d, tau, and si_stat.
    
    Computes d = |lim[0] - lim[1]| / sqrt(si_stat^2 + si_r^2)
    where lim[0]=mu_std, lim[1]=mu_dvt
    
    Parameters
    ----------
    pars : dict or list
        Parameter structure from test_data with keys: 'tau', 'lim', 'si_stat', 'si_q', 'si_r'
        Values are lists of numpy arrays or scalars (one value per sample)
    sample_idx : int
        Sample index
    
    Returns
    -------
    tuple of (d, tau, si_stat) or (None, None, None) if not available
    """
    try:
        if isinstance(pars, dict):
            # Extract base parameters
            tau_val = pars.get('tau', [None] * 100)[sample_idx] if 'tau' in pars else None
            si_stat_val = pars.get('si_stat', [None] * 100)[sample_idx] if 'si_stat' in pars else None
            si_r_val = pars.get('si_r', [None] * 100)[sample_idx] if 'si_r' in pars else None
            lim_val = pars.get('lim', [None] * 100)[sample_idx] if 'lim' in pars else None
            
            # Compute d if we have all required parameters
            if tau_val is not None and si_stat_val is not None and si_r_val is not None and lim_val is not None:
                # Convert to float if numpy arrays/values - handle array elements or scalars
                if hasattr(tau_val, '__len__'):
                    tau = float(tau_val[0]) if len(tau_val) > 0 else float(tau_val)
                else:
                    tau = float(tau_val)
                    
                if hasattr(si_stat_val, '__len__'):
                    si_stat = float(si_stat_val[0]) if len(si_stat_val) > 0 else float(si_stat_val)
                else:
                    si_stat = float(si_stat_val)
                    
                if hasattr(si_r_val, '__len__'):
                    si_r = float(si_r_val[0]) if len(si_r_val) > 0 else float(si_r_val)
                else:
                    si_r = float(si_r_val)
                
                # Extract mu_std and mu_dvt from lim
                if hasattr(lim_val, '__len__') and len(lim_val) >= 2:
                    mu_std = float(lim_val[0])
                    mu_dvt = float(lim_val[1])
                else:
                    mu_std = float(lim_val)
                    mu_dvt = float(lim_val)
                
                denominator = np.sqrt(si_stat**2 + si_r**2)
                d = np.abs(mu_std - mu_dvt) / (denominator + 1e-8)
                return float(d), tau, si_stat, si_r
    except Exception as e:
        pass
    return None, None, None, None


# =============================================================================
# Numerics
# =============================================================================

def gaussian_likelihood(observations, mean, variance):
    """Likelihood (density) of each observation under a Gaussian.

    Evaluates N(observation | mean, variance) elementwise, i.e. the value of the
    Gaussian probability density at each observation. This is the likelihood of
    the ground-truth observation under the obs module's predicted distribution.

    Parameters
    ----------
    observations, mean, variance : array-like, same shape
        Ground-truth observations and the predicted Gaussian mean/variance.

    Returns
    -------
    np.ndarray
        Per-element likelihood, same shape as the inputs.
    """
    obs = np.asarray(observations, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    var = np.asarray(variance, dtype=np.float64)
    return np.exp(-0.5 * (obs - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def class_likelihood(class_probs, labels):
    """Likelihood of the true class label under predicted class probabilities.

    For a (K, C) matrix ``lambda`` of class probabilities and 0-based ground-truth
    labels c, returns ``lambda[k, labels[k]]`` for each row k — the probability the
    model assigns to the true class. For a categorical distribution this *is* the
    likelihood of the observed class, which is why no other transform is needed.

    Parameters
    ----------
    class_probs : np.ndarray, shape (K, C)
        Per-row class probabilities (softmax outputs).
    labels : array-like of int, shape (K,)
        Zero-based ground-truth class indices, one per row.

    Returns
    -------
    np.ndarray, shape (K,)
        Likelihood of the true class at each row.
    """
    class_probs = np.asarray(class_probs)
    labels = np.asarray(labels, dtype=int)
    if labels.min() < 0 or labels.max() >= class_probs.shape[1]:
        raise IndexError(
            f"labels out of range [0, {class_probs.shape[1] - 1}]: "
            f"got min={labels.min()}, max={labels.max()}. "
            "Did you forget to shift to 0-based class indices (e.g. dpos)?"
        )
    return class_probs[np.arange(class_probs.shape[0]), labels]


def compute_derivatives(norms):
    """Compute temporal derivative (finite differences) of activity norms.

    Parameters
    ----------
    norms : np.ndarray
        Shape (seq_len, batch)

    Returns
    -------
    derivatives : np.ndarray
        Shape (seq_len-1, batch) - derivative at each timestep
    """
    return np.diff(norms, axis=0)


def gather_at(arr, idx, length):
    """Gather arr[idx], returning NaN where idx falls outside [0, length).

    Used to sample one value per trial at the trial's deviant timestep; the very
    last within-trial position of the last trial can fall past the activity array
    (which is one step shorter than the raw sequence), so it becomes NaN.
    """
    out = np.full(len(idx), np.nan, dtype=float)
    valid = idx < length
    out[valid] = arr[idx[valid]]
    return out


# =============================================================================
# Module-activity statistics
# =============================================================================
#
# Reshaping activity by within-trial position or at the deviant, and the
# correlation / independence summaries computed over it. These live here rather
# than in exp_sequence_analysis.py because plots.py needs them, and importing
# that module from plots.py would be circular.

def reshape_norms_by_position(data, period=8):
    """Split a (seq_len, batch) activity array into within-trial positions.

    Every timestep ``t`` is assigned a within-trial position ``t % period`` and a
    trial index ``t // period``. Because every trial spans exactly ``period``
    consecutive timesteps (and the sequence starts at trial 0, position 0), this
    is equivalent to ``df.groupby('trial_n').cumcount()`` used elsewhere, but
    works directly on the batched norms array without needing the trial_n column.

    Parameters
    ----------
    data : np.ndarray
        Shape (seq_len, batch) — e.g. a module's activity norms or derivatives.
    period : int
        Number of timesteps per trial (within-trial positions). Default 8.

    Returns
    -------
    dict
        Maps a 0-based within-trial position to a dict with:
          - 'trials' : np.ndarray (n_trials_p,)        trial index per point (x-axis)
          - 'values' : np.ndarray (n_trials_p, batch)  activity at this position
        Trailing positions may cover one fewer trial when seq_len is not an exact
        multiple of period.
    """
    seq_len = data.shape[0]
    by_pos = {}
    for p in range(period):
        ts = np.arange(p, seq_len, period)
        if ts.size == 0:
            continue
        by_pos[p] = {
            'trials': ts // period,
            'values': data[ts, :],
        }
    return by_pos


def extract_deviant_activity(data, dev_pos, period=8):
    """Pick the activity at each trial's deviant position, grouped by deviant value.

    For every trial the deviant tone sits at within-trial position ``dev_pos`` (a
    0-based index, constant within the trial). The activity at that deviant is the
    timestep ``trial * period + dev_pos`` of ``data``. Trials are grouped by the
    *value* of their deviant position and, for each (trial, value) cell, averaged
    over the sequences whose deviant fell on that position at that trial.

    Parameters
    ----------
    data : np.ndarray
        Shape (seq_len, n_seq) — a module's activity norms or derivatives.
    dev_pos : np.ndarray
        Shape (n_seq, n_trials) — 0-based within-trial deviant position for every
        sequence and trial.
    period : int
        Timesteps per trial. Default 8.

    Returns
    -------
    dict
        deviant-position value (int) → dict of equal-length arrays:
          - 'trials' : trial index per point (x-axis)
          - 'mean'   : activity at the deviant, averaged over contributing sequences
          - 'std'    : std across those sequences
          - 'count'  : number of contributing sequences
        A (trial, value) point is dropped when no sequence has that deviant value
        at that trial, or when its timestep falls past ``seq_len`` (e.g. the very
        last position of the last trial after the ``y[:, :-1]`` slice).
    """
    seq_len = data.shape[0]
    n_trials = dev_pos.shape[1]
    trials = np.arange(n_trials)

    out = {}
    for v in np.unique(dev_pos):
        v = int(v)
        rec = {'trials': [], 'mean': [], 'std': [], 'count': []}
        for t in trials:
            g = t * period + v
            if g >= seq_len:
                continue
            mask = dev_pos[:, t] == v          # sequences with deviant at v in trial t
            if not mask.any():
                continue
            vals = data[g, mask]
            rec['trials'].append(t)
            rec['mean'].append(vals.mean())
            rec['std'].append(vals.std())
            rec['count'].append(int(mask.sum()))
        out[v] = {k: np.asarray(val) for k, val in rec.items()}
    return out


def compute_pairwise_sample_correlations(norms):
    """Compute pairwise correlations between all samples in a module.
    
    Parameters
    ----------
    norms : np.ndarray
        Shape (seq_len, batch)
    
    Returns
    -------
    tuple of (min_corr, max_corr)
    """
    n_samples = norms.shape[1]
    correlations = []
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            act_i = norms[:, i]
            act_j = norms[:, j]
            
            # Normalize for correlation
            act_i_norm = (act_i - act_i.mean()) / (act_i.std() + 1e-8)
            act_j_norm = (act_j - act_j.mean()) / (act_j.std() + 1e-8)
            corr = np.mean(act_i_norm * act_j_norm)
            correlations.append(corr)
    
    if correlations:
        return min(correlations), max(correlations)
    return 0.0, 0.0


def compute_module_independence(module_norms_dict):
    """Compute metrics for independence/separability of modules.
    
    Metrics:
    - Correlation matrix: Pairwise correlations between module activity patterns
    - Redundancy: Average absolute correlation (lower = more independent)
    - Explained variance ratio (PCA): How much variance each module explains
    
    Parameters
    ----------
    module_norms_dict : dict
        Keys are module names, values are norms arrays of shape (seq_len, batch)
    
    Returns
    -------
    dict with multiple independence metrics
    """
    module_names = list(module_norms_dict.keys())
    n_modules = len(module_names)
    
    # Flatten each module across time and samples
    module_activities = {}
    for name in module_names:
        norms = module_norms_dict[name]
        # Flatten to (seq_len * batch,) and normalize
        flat = norms.reshape(-1)
        flat_norm = (flat - flat.mean()) / (flat.std() + 1e-8)
        module_activities[name] = flat_norm
    
    # --- Correlation matrix ---
    correlation_matrix = np.zeros((n_modules, n_modules))
    for i, name_i in enumerate(module_names):
        for j, name_j in enumerate(module_names):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                corr, _ = pearsonr(module_activities[name_i], module_activities[name_j])
                correlation_matrix[i, j] = corr
    
    # Redundancy: average absolute correlation (excluding diagonal)
    off_diag = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
    redundancy = np.mean(np.abs(off_diag))
    independence_score = 1 - redundancy  # Higher = more independent
    
    # --- Explained variance (PCA approach) ---
    stacked_activities = np.column_stack([module_activities[name] for name in module_names])
    pca = PCA()
    pca.fit(stacked_activities)
    explained_var_ratio = pca.explained_variance_ratio_
    
    # Effective number of dimensions (entropy of explained variance)
    entropy = -np.sum(explained_var_ratio * np.log(explained_var_ratio + 1e-8))
    effective_dims = np.exp(entropy)  # e^entropy, bounded by n_modules
    
    # --- Linear regression redundancy ---
    # For each module, fit linear regression from others to predict it
    r_squared_values = []
    for i, target_name in enumerate(module_names):
        target = module_activities[target_name]
        feature_names = [name for j, name in enumerate(module_names) if j != i]
        features = np.column_stack([module_activities[name] for name in feature_names])
        
        # Simple linear regression R-squared
        from numpy.linalg import lstsq
        try:
            coeffs, residuals, _, _ = lstsq(features, target, rcond=None)
            ss_res = np.sum(residuals) if residuals.size > 0 else np.sum((target - features @ coeffs) ** 2)
            ss_tot = np.sum((target - target.mean()) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
            r_squared_values.append(r_squared)
        except:
            r_squared_values.append(0.0)
    
    mean_r_squared = np.mean(r_squared_values)
    
    return {
        'correlation_matrix': correlation_matrix,
        'module_names': module_names,
        'redundancy': redundancy,  # 0-1, lower = more independent
        'independence_score': independence_score,  # 0-1, higher = more independent
        'explained_variance_ratio': explained_var_ratio,
        'effective_dimensions': effective_dims,
        'mean_linear_predictability': mean_r_squared,  # 0-1, lower = more independent
    }

def compute_sample_difference_single_module(norms, sample_idx_1, sample_idx_2):
    """Compute difference in activity between two samples for a single module.
    
    Metrics:
    - L2 distance: Euclidean distance in activity space over time
    - Correlation: How synchronized are the two temporal patterns
    - Max absolute difference: Peak difference over time
    
    Parameters
    ----------
    norms : np.ndarray
        Shape (seq_len, batch)
    sample_idx_1, sample_idx_2 : int
        Sample indices to compare
    
    Returns
    -------
    dict with metrics
    """
    activity_1 = norms[:, sample_idx_1]
    activity_2 = norms[:, sample_idx_2]
    
    l2_distance = np.linalg.norm(activity_1 - activity_2)
    
    # Normalize for correlation to handle scale differences
    activity_1_norm = (activity_1 - activity_1.mean()) / (activity_1.std() + 1e-8)
    activity_2_norm = (activity_2 - activity_2.mean()) / (activity_2.std() + 1e-8)
    correlation = np.mean(activity_1_norm * activity_2_norm)  # cosine similarity
    
    max_diff = np.max(np.abs(activity_1 - activity_2))
    
    return {
        'l2_distance': l2_distance,
        'correlation': correlation,  # higher = more similar
        'max_difference': max_diff
    }


def compute_sample_difference_all_modules(module_norms_dict, sample_idx_1, sample_idx_2):
    """Compute difference in activity across all modules between two samples.
    
    Parameters
    ----------
    module_norms_dict : dict
        Keys are module names, values are norms arrays
    sample_idx_1, sample_idx_2 : int
        Sample indices to compare
    
    Returns
    -------
    dict with per-module and aggregate metrics
    """
    results = {}
    all_diffs = []
    all_corrs = []
    
    for module_name, norms in module_norms_dict.items():
        module_metrics = compute_sample_difference_single_module(norms, sample_idx_1, sample_idx_2)
        results[module_name] = module_metrics
        all_diffs.append(module_metrics['l2_distance'])
        all_corrs.append(module_metrics['correlation'])
    
    # Aggregate metrics
    results['aggregate'] = {
        'mean_l2_distance': np.mean(all_diffs),
        'mean_correlation': np.mean(all_corrs),
        'total_l2_distance': np.linalg.norm(all_diffs),  # Euclidean distance in module space
    }
    
    return results

if __name__ == '__main__':
    pass
