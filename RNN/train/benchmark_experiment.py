"""
Dedicated entry point for computing Kalman-filter (KF) benchmarks.

Mirrors experiment.py but computes ONLY the test benchmarks (no training/testing),
and reuses the exact same data configuration via
DataConfig.for_hierarchical_experiment(). This guarantees the KF benchmarks are
computed on the same data distribution the models are trained on.

This deliberately does NOT go through pipeline_runner_v2: it calls run_benchmarks
directly from pipeline_core_v2.

Usage:
    # Full benchmarks (default; what the SLURM job runs)
    python benchmark_experiment.py

    # Quick smoke test (tiny sequences, single core)
    python benchmark_experiment.py --unit_test

    # Write to a separately-tagged file to avoid reusing/overwriting an existing
    # benchmark computed with different hyperparameters
    python benchmark_experiment.py --tag si_bounds_v2
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config_v2 import TrainingConfig, DataConfig
from pipeline_core_v2 import run_benchmarks

# =============================================================================
# Mode / CLI flags (kept minimal, matching experiment.py's lightweight style)
# =============================================================================
# Benchmarks default to a FULL run (unlike experiment.py, which defaults to unit
# test) because this script is normally launched as a long SLURM job.
UNIT_TEST = os.getenv('UNIT_TEST', 'False').lower() == 'true'
if '--unit_test' in sys.argv:
    UNIT_TEST = True
    sys.argv.remove('--unit_test')
if '--no-unit-test' in sys.argv:
    UNIT_TEST = False
    sys.argv.remove('--no-unit-test')

# Optional tag appended to benchmark file/dir names, e.g. --tag si_bounds_v2.
# Use this when data hyperparameters change, since benchmark filenames are keyed
# only by (N_ctx, gm_name, N_samples) and would otherwise silently reuse a stale
# file / resume a stale partial computation.
benchmark_tag = ''
if '--tag' in sys.argv:
    i = sys.argv.index('--tag')
    benchmark_tag = sys.argv[i + 1]
    del sys.argv[i:i + 2]

# =============================================================================
# Configs
# =============================================================================
# Only batch sizes matter for benchmarks (batch_size_test = number of KF samples).
if UNIT_TEST:
    training = TrainingConfig.for_unit_test()
else:
    training = TrainingConfig()

# Same source of truth as experiment.py (training). In unit-test mode, shrink the
# sequence length so the O(T^2) KF benchmark finishes quickly; the full run keeps
# the canonical size (seq length 1000).
# HierarchicalGM (default): shrink via N_blocks (seq length = N_tones * N_blocks).
data = DataConfig.for_hierarchical_experiment(
    N_ctx=2,
    N_blocks=5 if UNIT_TEST else 125,
    max_cores=1 if UNIT_TEST else None,  # Disable parallel processing during tests
)
# --- To benchmark NonHierarchicalGM instead, comment the block above and uncomment.
# NonHierarchicalGM forces N_blocks=1, so shrink via N_tones for the smoke test: ---
# data = DataConfig.for_nonhierarchical_experiment(
#     N_ctx=2,
#     N_tones=40 if UNIT_TEST else 1000,
#     max_cores=1 if UNIT_TEST else None,
# )

# =============================================================================
# Run KF benchmarks only
# =============================================================================
if __name__ == '__main__':
    run_benchmarks(
        data_config=data,
        training_config=training,
        visualize=True,
        suffix_tag=benchmark_tag,
    )
