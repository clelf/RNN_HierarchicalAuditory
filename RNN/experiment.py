import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config_v2 import (
    HyperparameterGrid, 
    TrainingConfig, 
    DataConfig,
)
from pipeline_runner_v2 import run_pipeline, run_benchmarks

# Set to True for quick testing, False for full training
UNIT_TEST = True

# Custom training config
if UNIT_TEST:
    training = TrainingConfig.for_unit_test()
else:
    training = TrainingConfig(
        num_epochs=200,        # More epochs
        batch_size=500,        # Smaller batches
        weight_decay=1e-4,     # More regularization
    )

# Custom data config
data = DataConfig(
    gm_name='NonHierarchicalGM',
    N_ctx=2,
    N_tones=500,
    mu_tau_bounds={'low': 5, 'high': 100},  # Narrower tau range
)

# Custom hyperparameter grid
grid = HyperparameterGrid(
    model_types=['module_network'],
    learning_rates=[0.01],
    hidden_dims=[64],  # Fixed for ModuleNetwork anyway
    learning_objectives=['obs', 'ctx', 'obs_ctx'],
    kappa_values=[0.3, 0.5, 0.7],
    bottleneck_dims=[4], # , 8, 16, 32
    training=training,
    data=data
)

# Run!
run_pipeline(grid, skip_benchmarks=True)