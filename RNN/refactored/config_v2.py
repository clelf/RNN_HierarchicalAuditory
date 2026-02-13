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
from typing import List, Optional, Literal
from pathlib import Path
import json
import os
from multiprocessing import cpu_count


# =============================================================================
# Type Aliases for Clarity
# =============================================================================

DataMode = Literal['single_ctx', 'multi_ctx']
LearningObjective = Literal['obs', 'ctx', 'obs_ctx']
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
    """Model architecture parameters - varies per model type."""
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
        
        return d


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
    hidden_dim: int                    # RNN hidden dimension
    learning_rate: float
    lr_id: int                         # Index for file naming
    
    # ModuleNetwork-specific
    learning_objective: LearningObjective = 'obs'
    kappa: float = 0.5                 # Weight for obs_ctx objective
    bottleneck_dim: Optional[int] = None
    
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
        """
        return {
            # Identity
            'name': self.name,
            'save_dir': str(self.save_dir),
            
            # Model specification
            'model_type': self.model_type,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'lr_id': self.lr_id,
            
            # ModuleNetwork-specific
            'learning_objective': self.learning_objective,
            'kappa': self.kappa,
            'bottleneck_dim': self.bottleneck_dim,
            
            # Nested configs (as dicts)
            'training': asdict(self.training),
            'model_arch': asdict(self.model_arch),
            'data': {
                **asdict(self.data),
                # Ensure max_cores is captured (computed in __post_init__)
                'max_cores': self.data.max_cores,
            },
            
            # Visualization
            'seq_len_viz': self.seq_len_viz,
        }
    
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
    hidden_dims: List[int] = field(default_factory=lambda: [16, 32, 64])
    
    # ModuleNetwork-specific
    learning_objectives: List[LearningObjective] = field(default_factory=lambda: ['obs'])
    kappa_values: List[float] = field(default_factory=lambda: [0.5])
    bottleneck_dims: List[int] = field(default_factory=lambda: [8, 16])
    
    # Shared configs
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model_arch: ModelArchConfig = field(default_factory=ModelArchConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Output settings
    base_output_dir: Path = field(default_factory=lambda: Path('training_results'))
    run_id: Optional[str] = None
    
    def expand(self) -> List[RunConfig]:
        """
        Expand the grid into a flat list of RunConfigs.
        
        This is the KEY simplification - instead of nested loops scattered
        across functions, we have ONE place that generates all configs.
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
        
        for model_type in self.model_types:
            for lr_id, lr in enumerate(self.learning_rates):
                
                if model_type == 'module_network':
                    # ModuleNetwork: iterate over objectives, kappas, bottleneck dims
                    # But hidden_dim is fixed (not iterated)
                    for bn_dim in self.bottleneck_dims:
                        for obj in self.learning_objectives:
                            kappas = self.kappa_values if obj == 'obs_ctx' else [0.5]
                            for kappa in kappas:
                                folder_name = f"{model_type}_{obj}"
                                if obj == 'obs_ctx':
                                    folder_name += f"_kappa{kappa}"
                                folder_name += f"_bn{bn_dim}"
                                
                                configs.append(RunConfig(
                                    name=f"{model_type}_lr{lr}_obj{obj}_bn{bn_dim}",
                                    save_dir=output_base / ctx_subpath / folder_name,
                                    model_type=model_type,
                                    hidden_dim=64,  # Fixed for ModuleNetwork
                                    learning_rate=lr,
                                    lr_id=lr_id,
                                    learning_objective=obj,
                                    kappa=kappa,
                                    bottleneck_dim=bn_dim,
                                    training=self.training,
                                    model_arch=self.model_arch,
                                    data=self.data,
                                ))
                else:
                    # RNN/VRNN: iterate over hidden dimensions
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
        return cls(
            model_types=model_types or ['rnn'],
            learning_rates=[0.01],
            hidden_dims=[16],
            learning_objectives=['obs'],
            bottleneck_dims=[8],
            training=TrainingConfig.for_unit_test(),
        )


# =============================================================================
# Conversion Utilities (for backward compatibility)
# =============================================================================

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
