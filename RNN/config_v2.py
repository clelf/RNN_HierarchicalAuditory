"""
Refactored configuration module using dataclasses for clarity and type safety.

This replaces nested dictionaries with structured, documented config objects.
The configs are designed to be:
1. Self-documenting (with type hints and docstrings)
2. Immutable after creation (frozen dataclasses)
3. Easy to serialize/deserialize
4. Compatible with the original config dicts (via to_dict/from_dict)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Union, Dict, Tuple
from pathlib import Path
import json
import os
from multiprocessing import cpu_count
import numpy as np


# =============================================================================
# Type Aliases for Clarity
# =============================================================================

DataMode = Literal['single_ctx', 'multi_ctx']
LearningObjective = Literal['obs', 'ctx', 'all']
ModelType = Literal['rnn', 'vrnn', 'module_network']
GMType = Literal['NonHierarchicalGM', 'HierarchicalGM']


# =============================================================================
# Base Configuration Dataclasses
# =============================================================================

@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters - constant across all runs."""
    num_epochs: int = 100
    n_batches: int = 32
    batch_size: int = 1000
    batch_size_test: int = 1000
    weight_decay: float = 1e-5
    epoch_res: int = 20      # Report every N epochs
    batch_res: int = 16      # Report every N batches
    
    @classmethod
    def for_unit_test(cls) -> 'TrainingConfig':
        """Create a minimal config for unit testing."""
        return cls(
            num_epochs=10,
            n_batches=2,
            batch_size=8,
            batch_size_test=8,
            epoch_res=10,
            batch_res=2
        )


@dataclass(frozen=True)
class ModelArchConfig:
    """Model architecture parameters - varies per model type.
    This is the default 1-module: observations.
    """
    input_dim: int = 1
    output_dim: int = 2  # (mu, var) for observation prediction
    rnn_n_layers: int = 1


@dataclass(frozen=True)
class DataConfig:
    """Data generation parameters."""
    gm_name: GMType = 'NonHierarchicalGM'
    N_ctx: int = 1
    N_tones: int = 1000  # Sequence length
    N_blocks: int = 1
    
    # Context parameters
    mu_rho_ctx: float = 0.9
    si_rho_ctx: float = 0.05
    
    # Process parameters
    si_lim: float = 5.0
    si_tau: float = 0.5
    
    # Parameter testing bounds
    params_testing: bool = True
    mu_tau_bounds: dict = field(default_factory=lambda: {'low': 1, 'high': 250})
    si_stat_bounds: dict = field(default_factory=lambda: {'low': 0.1, 'high': 2})
    si_r_bounds: dict = field(default_factory=lambda: {'low': 0.1, 'high': 2})
    
    # Multi-context parameters (only used when N_ctx > 1)
    si_d_coef: float = 0.05
    d_bounds: dict = field(default_factory=lambda: {'high': 4, 'low': 0.1})
    mu_d: float = 2.0
    
    # Hierarchical GM parameters (only used when gm_name == 'HierarchicalGM')
    rules_dpos_set: Optional[list] = None
    mu_rho_rules: float = 0.9
    si_rho_rules: float = 0.05

    # Optional hierarchical GM parameters (only used when gm_name == 'HierarchicalGM')
    p_cues: Optional[np.ndarray] = None
    cues_set: Optional[list] = None

    # Processing
    max_cores: Optional[int] = None
    
    def __post_init__(self):
        # Auto-detect max_cores if not set
        if self.max_cores is None:
            slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
            object.__setattr__(self, 'max_cores', 
                int(slurm_cpus) if slurm_cpus else max(1, cpu_count() // 2))
    
    @property
    def data_mode(self) -> DataMode:
        """Derive data_mode from N_ctx."""
        return 'multi_ctx' if self.N_ctx > 1 else 'single_ctx'
    
    @property
    def N_rules(self):
        """Number of rules (only for HierarchicalGM)."""
        if self.rules_dpos_set is None:
            return None
        return len(self.rules_dpos_set)
    
    @property
    def N_dpos(self):
        """Number of unique deviant positions (only for HierarchicalGM)."""
        if self.rules_dpos_set is None:
            return None
        # Flatten all position sets and count unique values
        all_positions = set()
        for positions in self.rules_dpos_set:
            if positions is not None:  # Handle None entries in rules_dpos_set
                all_positions.update(positions)
        return len(all_positions)

    @property
    def N_cues(self):
        """Number of cues (only for HierarchicalGM)."""
        if self.cues_set is None:
            return None
        return len(self.cues_set)
    
    def to_gm_dict(self, batch_size: int) -> dict:
        """Convert to dictionary format expected by GenerativeModel classes."""
        d = {
            'gm_name': self.gm_name,
            'N_ctx': self.N_ctx,
            'N_samples': batch_size,
            'N_blocks': self.N_blocks,
            'N_tones': self.N_tones,
            'mu_rho_ctx': self.mu_rho_ctx,
            'si_rho_ctx': self.si_rho_ctx,
            'si_lim': self.si_lim,
            'si_tau': self.si_tau,
            'max_cores': self.max_cores,
            'params_testing': self.params_testing,
        }
        
        if self.params_testing:
            d.update({
                'mu_tau_bounds': self.mu_tau_bounds,
                'si_stat_bounds': self.si_stat_bounds,
                'si_r_bounds': self.si_r_bounds,
            })
        
        if self.N_ctx > 1:
            d.update({
                'si_d_coef': self.si_d_coef,
                'd_bounds': self.d_bounds,
                'mu_d': self.mu_d,
            })
        
        if self.gm_name == 'HierarchicalGM':
            d.update({
                'rules_dpos_set': self.rules_dpos_set,
                'mu_rho_rules': self.mu_rho_rules,
                'si_rho_rules': self.si_rho_rules,
            })
            if self.p_cues is not None and self.cues_set is not None:
                d.update({
                    'p_cues': self.p_cues,
                    'cues_set': self.cues_set
                })
        
        # Convert non-JSON-serializable objects
        return _serialize_for_json(d)


# =============================================================================
# Run Configuration - One per training run
# =============================================================================

@dataclass(frozen=True)
class RunConfig:
    """
    Complete configuration for a single training/testing run.
    
    This is the "flattened" config - each RunConfig represents exactly one
    point in the hyperparameter space.
    """
    # Identity
    name: str                          # Human-readable name for logging
    save_dir: Path                     # Where to save results
    
    # Model specification
    model_type: ModelType
    hidden_dim: Optional[int] = None   # RNN hidden dimension (for RNN/VRNN, None for ModuleNetwork/PopulationNetwork)
    learning_rate: float = 0.01
    lr_id: int = 0                     # Index for file naming
    
    # ModuleNetwork/PopulationNetwork-specific
    learning_objective: LearningObjective = 'all'
    kappa: float = 0.5                 # Weight for obs_ctx objective
    bottleneck_dim: Optional[int] = None
    module_hidden_dims: Optional[Dict[str, int]] = None  # Per-module hidden dims for ModuleNetwork/PopulationNetwork
    
    # References to shared configs
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model_arch: ModelArchConfig = field(default_factory=ModelArchConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Visualization
    seq_len_viz: Optional[int] = 125
    
    def __post_init__(self):
        # Ensure save_dir is a Path
        if not isinstance(self.save_dir, Path):
            object.__setattr__(self, 'save_dir', Path(self.save_dir))
    
    @property
    def device(self) -> str:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def to_dict(self) -> dict:
        """
        Serialize RunConfig to a JSON-compatible dictionary.
        
        This captures ALL settings needed to reproduce training and testing.
        Handles numpy arrays and other non-JSON-serializable types.
        """
        data_dict = asdict(self.data)
        data_dict['max_cores'] = self.data.max_cores
        
        result = {
            # Identity
            'name': self.name,
            'save_dir': str(self.save_dir),
            
            # Model specification
            'model_type': self.model_type,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'lr_id': self.lr_id,
            
            # ModuleNetwork/PopulationNetwork-specific
            'learning_objective': self.learning_objective,
            'kappa': self.kappa,
            'bottleneck_dim': self.bottleneck_dim,
            'module_hidden_dims': self.module_hidden_dims,
            
            # Nested configs (as dicts)
            'training': asdict(self.training),
            'model_arch': asdict(self.model_arch),
            'data': data_dict,
            
            # Visualization
            'seq_len_viz': self.seq_len_viz,
        }
        
        # Convert non-JSON-serializable objects (e.g., numpy arrays)
        return _serialize_for_json(result)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'RunConfig':
        """
        Deserialize RunConfig from a dictionary (e.g., loaded from JSON).
        """
        # Filter DataConfig fields to only include valid constructor args
        import inspect
        data_config_fields = {f.name for f in DataConfig.__dataclass_fields__.values()}
        data_dict = {k: v for k, v in d.get('data', {}).items() 
                     if k in data_config_fields}
        
        return cls(
            name=d['name'],
            save_dir=Path(d['save_dir']),
            model_type=d['model_type'],
            hidden_dim=d['hidden_dim'],
            learning_rate=d['learning_rate'],
            lr_id=d['lr_id'],
            learning_objective=d.get('learning_objective', 'obs'),
            kappa=d.get('kappa', 0.5),
            bottleneck_dim=d.get('bottleneck_dim'),
            module_hidden_dims=d.get('module_hidden_dims'),
            training=TrainingConfig(**d.get('training', {})),
            model_arch=ModelArchConfig(**d.get('model_arch', {})),
            data=DataConfig(**data_dict),
            seq_len_viz=d.get('seq_len_viz', 125),
        )
    
    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save config to JSON file.
        
        Args:
            path: Optional path. If None, saves to save_dir/lr{lr_id}_config.json
        
        Returns:
            Path to saved config file.
        """
        if path is None:
            path = self.save_dir / f'lr{self.lr_id}_config.json'
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return path
    
    @classmethod
    def load(cls, path: Path) -> 'RunConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            d = json.load(f)
        return cls.from_dict(d)


# =============================================================================
# Hyperparameter Grid Expansion
# =============================================================================

@dataclass
class HyperparameterGrid:
    """
    Defines the search space for hyperparameters.
    
    This replaces the nested loops in pipeline_multi_config with a declarative
    specification that gets expanded into a flat list of RunConfigs.
    """
    # Model types to train
    model_types: List[ModelType] = field(default_factory=lambda: ['rnn', 'vrnn'])
    
    # Hyperparameter lists
    learning_rates: List[float] = field(default_factory=lambda: [0.01, 0.005, 0.001])
    # hidden_dims can be List[int] for RNN/VRNN or Dict[str, List[int]] for ModuleNetwork/PopulationNetwork
    hidden_dims: Union[List[int], Dict[str, List[int]]] = field(default_factory=lambda: [16, 32, 64])
    
    # ModuleNetwork-specific
    learning_objectives: List[LearningObjective] = field(default_factory=lambda: ['all'])
    kappa_values: List[float] = field(default_factory=lambda: [0.5])
    bottleneck_dims: List[int] = field(default_factory=lambda: [8, 16])
    
    # Shared configs
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model_arch: ModelArchConfig = field(default_factory=ModelArchConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Output settings
    base_output_dir: Path = field(default_factory=lambda: Path(os.path.dirname(__file__)) / 'training_results')
    run_id: Optional[str] = None
    
    def _get_objectives_and_kappas(self, model_type: ModelType) -> List[Tuple[str, Optional[float]]]:
        """
        Determine which (learning_objective, kappa) pairs to use based on model type.
        
        Returns:
            List of (objective, kappa) tuples to iterate over
            - ModuleNetwork: All combinations of learning_objectives and kappas
                (only uses kappas for 'obs_ctx' objective, uses None for others)
            - PopulationNetwork: Only ('all', None) - no iteration over objectives
            - RNN/VRNN: Only ('obs', None) - not used for these models
        """
        if model_type == 'module_network':
            # ModuleNetwork: iterate over all objectives, with kappas only for 'obs_ctx'
            pairs = []
            for obj in self.learning_objectives:
                if obj == 'all':  # 'all' is alias for 'obs_ctx'
                    for kappa in self.kappa_values:
                        pairs.append((obj, kappa))
                else:
                    pairs.append((obj, None))
            return pairs
        elif model_type == 'population_network':
            # PopulationNetwork: only 'all' learning objective, no kappa variation
            return [('all', None)]
        else:
            # RNN/VRNN: 'obs' only, no kappa
            return [('obs', None)]
    
    def _build_network_config(
        self,
        model_type: str,
        lr_id: int,
        lr: float,
        bn_dim: int,
        objective: str,
        kappa: Optional[float],
        module_hidden_dims: dict,
        ctx_subpath: str,
        output_base: Path,
    ) -> RunConfig:
        """Create a RunConfig for ModuleNetwork or PopulationNetwork."""
        # Build folder and name parts
        folder_parts = [model_type, objective]
        name_parts = [model_type, f"lr{lr}", f"obj{objective}", f"bn{bn_dim}"]
        
        if objective == 'obs_ctx':
            if kappa is not None:
                folder_parts.append(f"kappa{kappa}")
                name_parts.append(f"kappa{kappa}")
        
        folder_parts.append(f"bn{bn_dim}")
        
        # Build name with module dims
        dims_str = '_'.join(f"{k}{v}" for k, v in sorted(module_hidden_dims.items()))
        name_parts.append(dims_str)
        
        folder_name = '_'.join(folder_parts)
        config_name = '_'.join(name_parts)
        
        return RunConfig(
            name=config_name,
            save_dir=output_base / ctx_subpath / folder_name,
            model_type=model_type,
            hidden_dim=None,  # Not used for module networks
            learning_rate=lr,
            lr_id=lr_id,
            learning_objective=objective,
            kappa=kappa,
            bottleneck_dim=bn_dim,
            module_hidden_dims=module_hidden_dims,
            training=self.training,
            model_arch=self.model_arch,
            data=self.data,
        )
    
    def expand(self) -> List[RunConfig]:
        """
        Expand the grid into a flat list of RunConfigs.
        
        Handles three cases:
        - RNN/VRNN: hidden_dims is List[int], iterates over different hidden dimensions
        - ModuleNetwork: hidden_dims is Dict[str, List[int]], iterates over learning 
          objectives and kappas (if applicable)
        - PopulationNetwork: hidden_dims is Dict[str, List[int]], no iteration over 
          objectives (always 'all')
        """
        configs = []
        
        output_base = self.base_output_dir
        if self.run_id:
            output_base = output_base / self.run_id
        
        # Build context/GM subpath
        if self.data.N_ctx == 1:
            ctx_subpath = f"N_ctx_{self.data.N_ctx}"
        else:
            ctx_subpath = f"N_ctx_{self.data.N_ctx}/{self.data.gm_name}"
        
        # Detect if hidden_dims is a dict (for module or population networks) or list (for RNN/VRNN)
        is_dict_hidden_dims = isinstance(self.hidden_dims, dict)
        
        for model_type in self.model_types:
            for lr_id, lr in enumerate(self.learning_rates):
                
                if model_type in ['module_network', 'population_network']:
                    # Validate hidden_dims format
                    if not is_dict_hidden_dims:
                        raise ValueError(
                            f"For {model_type}, hidden_dims must be a Dict[str, List[int]], "
                            f"got {type(self.hidden_dims)}. Use format like {{'obs': [64], 'ctx': [32]}}"
                        )
                    
                    # Extract single values from the dict of lists (use first value in each list)
                    module_hidden_dims = {
                        k: v[0] if isinstance(v, list) else v
                        for k, v in self.hidden_dims.items()
                    }
                    
                    # Get objectives and kappas to iterate over
                    obj_kappa_pairs = self._get_objectives_and_kappas(model_type)
                    
                    for bn_dim in self.bottleneck_dims:
                        for objective, kappa in obj_kappa_pairs:
                            configs.append(
                                self._build_network_config(
                                    model_type=model_type,
                                    lr_id=lr_id,
                                    lr=lr,
                                    bn_dim=bn_dim,
                                    objective=objective,
                                    kappa=kappa,
                                    module_hidden_dims=module_hidden_dims,
                                    ctx_subpath=ctx_subpath,
                                    output_base=output_base,
                                )
                            )
                
                else:
                    # RNN/VRNN: iterate over hidden dimensions only
                    # Validate hidden_dims format
                    if is_dict_hidden_dims:
                        raise ValueError(
                            f"For {model_type}, hidden_dims must be a List[int], "
                            f"got {type(self.hidden_dims)}. Use format like [16, 32, 64]"
                        )
                    
                    for h_dim in self.hidden_dims:
                        configs.append(RunConfig(
                            name=f"{model_type}_h{h_dim}_lr{lr}",
                            save_dir=output_base / ctx_subpath / f"{model_type}_h{h_dim}",
                            model_type=model_type,
                            hidden_dim=h_dim,
                            learning_rate=lr,
                            lr_id=lr_id,
                            learning_objective='obs',  # RNN/VRNN only support 'obs'
                            training=self.training,
                            model_arch=self.model_arch,
                            data=self.data,
                        ))
        
        return configs
    
    @classmethod
    def for_unit_test(cls, model_types: List[ModelType] = None) -> 'HyperparameterGrid':
        """Create a minimal grid for unit testing."""
        model_types = model_types or ['rnn']
        # Use appropriate hidden_dims format based on model type
        if all(mt in ['module_network', 'population_network'] for mt in model_types):
            hidden_dims = {'obs': [64], 'ctx': [32]}
        else:
            hidden_dims = [16]
        
        return cls(
            model_types=model_types,
            learning_rates=[0.01],
            hidden_dims=hidden_dims,
            learning_objectives=['obs'],
            bottleneck_dims=[8],
            training=TrainingConfig.for_unit_test(),
        )


# =============================================================================
# Conversion Utilities (for backward compatibility)
# =============================================================================

def _serialize_for_json(obj):
    """
    Convert non-JSON-serializable objects (e.g., numpy arrays) to JSON-compatible types.
    
    Handles:
    - numpy arrays -> lists
    - numpy types -> Python native types
    - dicts, lists -> recursively processed
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    return obj


def run_config_to_model_dict(config: RunConfig) -> dict:
    """
    Convert RunConfig to the dictionary format expected by current model code.
    
    This allows gradual migration - new code uses RunConfig, but can still
    instantiate models using the existing dictionary-based interface.
    """
    return {
        'input_dim': config.model_arch.input_dim,
        'output_dim': config.model_arch.output_dim,
        'hidden_dim': config.hidden_dim,
        'n_layers': config.model_arch.rnn_n_layers,
        'device': config.device,
        # VRNN-specific (uses same dim for all latent spaces)
        'latent_dim': config.hidden_dim,
        'phi_x_dim': config.hidden_dim,
        'phi_z_dim': config.hidden_dim,
        'phi_prior_dim': config.hidden_dim,
        'rnn_hidden_states_dim': config.hidden_dim,
        'rnn_n_layers': config.model_arch.rnn_n_layers,
    }


def run_config_to_training_dict(config: RunConfig) -> dict:
    """Convert RunConfig to the model_config dict format for training functions."""
    return {
        'use_minmax': False,
        'input_dim': config.model_arch.input_dim,
        'output_dim': config.model_arch.output_dim,
        'rnn_n_layers': config.model_arch.rnn_n_layers,
        'num_epochs': config.training.num_epochs,
        'n_batches': config.training.n_batches,
        'batch_size': config.training.batch_size,
        'batch_size_test': config.training.batch_size_test,
        'weight_decay': config.training.weight_decay,
        'epoch_res': config.training.epoch_res,
        'batch_res': config.training.batch_res,
        'seq_len_viz': config.seq_len_viz,
        'n_trials': config.data.N_tones,
    }
