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
# constrained_dpos_response_window: supervise the dpos module at every timestep but
# emphasize (add weight to) the deviant timestep and the one just after it, biasing it to
# commit to the correct position as soon as it is knowable (tune w_dev/w_next in
# TrainingConfig). To be set to False to supervise dpos uniformly like the other modules.
# train_h0: when True, each RNN module learns its initial hidden state h0 (a trainable
# parameter, broadcast across the batch) instead of always starting from zeros. Off by
# default to preserve existing behaviour; flip to True to enable.
if UNIT_TEST:
    training = TrainingConfig.for_unit_test(constrained_dpos_response_window=True, train_h0=False)
else:
    training = TrainingConfig(num_epochs=200, constrained_dpos_response_window=True, train_h0=False)


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
    learning_rates=[0.001], #, 0.01, 0.001] if not UNIT_TEST else [0.01], # Removed 0.05 and 0.1 to prevent gradient explosion
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