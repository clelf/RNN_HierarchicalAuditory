import numpy as np
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
# Can override with --unit_test flag or UNIT_TEST environment variable
UNIT_TEST = os.getenv('UNIT_TEST', 'True').lower() == 'true'
if '--unit_test' in sys.argv:
    UNIT_TEST = True
    sys.argv.remove('--unit_test')
if '--no-unit-test' in sys.argv:
    UNIT_TEST = False
    sys.argv.remove('--no-unit-test')

# Custom training config
# constrained_dpos_response_window: only supervise the dpos module from the start of
# each trial up to one timestep after the ground-truth deviant (set False to supervise
# dpos at every timestep like the other modules).
if UNIT_TEST:
    training = TrainingConfig.for_unit_test(constrained_dpos_response_window=True)
else:
    training = TrainingConfig(num_epochs=200, constrained_dpos_response_window=True)


# Data config (single source of truth shared with benchmark_experiment.py).
# HierarchicalGM (default). Tune hyperparameters in the factory itself.
data = DataConfig.for_hierarchical_experiment(
    N_ctx=2,
    max_cores=1 if UNIT_TEST else None,  # Disable parallel processing during tests
)
# --- To use NonHierarchicalGM instead, comment the block above and uncomment: ---
# data = DataConfig.for_nonhierarchical_experiment(
#     N_ctx=2,
#     max_cores=1 if UNIT_TEST else None,
# )


# Custom hyperparameter grid
param_grid = HyperparameterGrid(
    model_types=['population_network'], # population_network , module_network
    learning_rates=[0.005], #, 0.01, 0.001] if not UNIT_TEST else [0.01], # Removed 0.05 and 0.1 to prevent gradient explosion
    # hidden_dims=[64],  # Fixed for ModuleNetwork anyway # TODO: look into this
    hidden_dims={'obs': [64], 'ctx': [64], 'dpos': [64], 'rule': [64]}, # TODO: look into this, possibly no need to decrease hidden dims...
    # learning_objectives=['all'], # 'obs', 'ctx', 'all'
    # kappa_values=[0.3, 0.5, 0.7],
    bottleneck_dims=[8], # 4, 8, 16, 32
    training=training,
    data=data
)

# Run!
run_pipeline(param_grid, skip_benchmarks=True, train_only=True)