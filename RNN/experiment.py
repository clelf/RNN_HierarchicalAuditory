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
if UNIT_TEST:
    training = TrainingConfig.for_unit_test()
else:
    training = TrainingConfig(num_epochs=200)


# Custom data config
# data = DataConfig(
#     gm_name='NonHierarchicalGM',
#     N_ctx=2,
#     N_tones=500,
#     mu_tau_bounds={'low': 5, 'high': 100},  # Narrower tau range
# )
data = DataConfig(
    gm_name='HierarchicalGM',
    N_ctx=2,
    # Need extra parameters for hierarchical GM
    si_d_coef= 0.05,
    d_bounds = {"high": 4, "low": 0.1},
    mu_d = 2, # TODO: also test with [1, 2]
    N_blocks = 125,
    N_tones = 8,
    rules_dpos_set = np.array([[3, 4, 5], [5, 6, 7]]),
    mu_rho_rules = 0.9,
    si_rho_rules = 0.05,
    si_lim = 5,
    mu_tau_bounds = {'low': 1, 'high': 250},
    si_stat_bounds = {'low': 0.1, 'high': 2},
    si_r_bounds = {'low': 0.1, 'high': 2},
    p_cues = np.array([0.8, 0.2]),
    cues_set = [0, 1],
    max_cores=1 if UNIT_TEST else None,  # Disable parallel processing during tests

)


# Custom hyperparameter grid
param_grid = HyperparameterGrid(
    model_types=['population_network'], # population_network , module_network
    learning_rates=[0.01],
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