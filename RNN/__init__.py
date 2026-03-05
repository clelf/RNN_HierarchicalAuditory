"""
RNN package — includes the refactored pipeline (previously under refactored/).

Quick start:
    from RNN.config_v2 import HyperparameterGrid
    from RNN.pipeline_runner_v2 import run_pipeline

    grid = HyperparameterGrid.for_unit_test()
    run_pipeline(grid, skip_benchmarks=True)

Or from command line:
    cd RNN/
    python pipeline_runner_v2.py --unit_test --skip_benchmarks
"""

from .config_v2 import (
    RunConfig,
    TrainingConfig,
    ModelArchConfig,
    DataConfig,
    HyperparameterGrid,
)

from .pipeline_core_v2 import (
    create_model,
    train_model,
    test_model,
    run_single_config,
)

from .pipeline_runner_v2 import run_pipeline, run_benchmarks

__all__ = [
    # Config
    'RunConfig',
    'TrainingConfig',
    'ModelArchConfig',
    'DataConfig',
    'HyperparameterGrid',
    # Core
    'create_model',
    'train_model',
    'test_model',
    'run_single_config',
    # Runner
    'run_pipeline',
    'run_benchmarks',
]
