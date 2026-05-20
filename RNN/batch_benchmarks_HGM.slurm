#!/bin/bash
#SBATCH --qos=xlong
#SBATCH --job-name=hgm_bench
#SBATCH --nodes=1              # Use single node to avoid distributed complexity
#SBATCH --ntasks=1             # Single task (your Python script)
#SBATCH --cpus-per-task=8      # CPUs for parallel data generation
#SBATCH --output=/scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/outputs_terminal/hgm_bench_%j.out
#SBATCH --error=/scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/outputs_terminal/hgm_bench_%j.err
#SBATCH --mail-type=END,FAIL           # Notifications for job done or fail
#SBATCH --mail-user=c.levyfidel@bcbl.eu  # Email address for notifications
#SBATCH --mem=64G              # Increased memory for large batch sizes


# Load modules needed
module load Miniforge3

# Activate conda environment
conda activate /scratch/clevyfidel/conda_env/rnn2

# ============ CONFIGURABLE PARAMETERS ============
# Computing HierarchicalGM benchmarks for test data (N_ctx=2)
# 
# Test data generation matches the configuration in model_analysis.py:
# - N_samples_test: 1000 samples
# - Data configuration: HierarchicalGM with N_ctx=2
# - Only test benchmarks (no training benchmarks)
#
# This generates Kalman filter predictions that can be used to compare
# against trained models in model_analysis.py by setting USE_BENCHMARK_DATA=True
# =================================================

# Navigate to the RNN directory
cd /scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/

# Compute HierarchicalGM test benchmarks (1000 samples)
# For benchmark computation, only data configuration is needed
/scratch/clevyfidel/conda_env/rnn2/bin/python pipeline_runner_v2.py \
    --n_ctx 2 \
    --gm_name HierarchicalGM \
    --benchmark_only


# Verify Python version and activated environment
echo "Python version: $(python --version)"
echo "Which Python: $(which python)"
