from refactored import (
    HyperparameterGrid, 
    TrainingConfig, 
    DataConfig,
    run_pipeline,
    run_benchmarks,
)

# Custom training config
training = TrainingConfig(
    num_epochs=200,        # More epochs
    batch_size=500,        # Smaller batches
    weight_decay=1e-4,     # More regularization
)

# Custom data config
data = DataConfig(
    gm_name='HierarchicalGM',
    N_ctx=2,
    N_tones=500,
    mu_tau_bounds={'low': 5, 'high': 100},  # Narrower tau range
)

# Custom hyperparameter grid
grid = HyperparameterGrid(
    model_types=['module_network'],
    learning_rates=[0.01, 0.005],
    hidden_dims=[64],  # Fixed for ModuleNetwork anyway
    learning_objectives=['obs', 'ctx', 'obs_ctx'],
    kappa_values=[0.3, 0.5, 0.7],
    bottleneck_dims=[4, 8, 16, 32],
    training=training,
    data=data
)

# Run!
run_pipeline(grid, skip_benchmarks=False)