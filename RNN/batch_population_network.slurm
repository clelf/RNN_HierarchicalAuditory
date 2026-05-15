#!/bin/bash
#SBATCH --qos=xlong
#SBATCH --job-name=popnet_train
#SBATCH --nodes=1              # Use single node to avoid distributed complexity
#SBATCH --ntasks=1             # Single task (your Python script)
#SBATCH --cpus-per-task=8      # CPUs for parallel data generation
#SBATCH --output=/scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/outputs_terminal/popnet_%j.out
#SBATCH --error=/scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/outputs_terminal/popnet_%j.err
#SBATCH --mail-type=END,FAIL           # Notifications for job done or fail
#SBATCH --mail-user=c.levyfidel@bcbl.eu  # Email address for notifications
#SBATCH --mem=64G              # Increased memory for large batch sizes


# Load modules needed
module load Miniforge3

# Activate conda environment
conda activate /scratch/clevyfidel/conda_env/rnn2

# ============ Configuration ============
# This script runs the population_network + HierarchicalGM training
# as configured in experiment.py
#
# The experiment.py script contains:
# - model_types: population_network
# - learning_rate: 0.01
# - hidden_dims: 64 (per module)
# - bottleneck_dims: 8
# - num_epochs: 200 (when not in UNIT_TEST)
# - skip_benchmarks: True
# - train_only: True
#
# For unit testing, use: UNIT_TEST=True python experiment.py
# For full training, use: UNIT_TEST=False python experiment.py (or just python experiment.py)
# ========================================

# Navigate to the refactored directory
cd /scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/refactored

# Run experiment.py with full training (UNIT_TEST=False)
# This uses all the HierarchicalGM parameters configured in experiment.py
export UNIT_TEST=False
/scratch/clevyfidel/conda_env/rnn2/bin/python experiment.py

# Verify Python version and activated environment
echo "Python version: $(python --version)"
echo "Which Python: $(which python)"
echo "UNIT_TEST mode: $UNIT_TEST"
