#!/bin/bash
#SBATCH --qos=xlong
#SBATCH --job-name=rnn_benchm
#SBATCH --nodes=1              # Use single node to avoid distributed complexity
#SBATCH --ntasks=1             # Single task (your Python script)
#SBATCH --cpus-per-task=8      # CPUs for parallel data generation
#SBATCH --output=/scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/outputs_terminal/benchm_%j.out
#SBATCH --error=/scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/outputs_terminal/benchm_%j.err
#SBATCH --mail-type=END,FAIL           # Notifications for job done or fail
#SBATCH --mail-user=c.levyfidel@bcbl.eu  # Email address for notifications
#SBATCH --mem=64G              # Increased memory for large batch sizes


# Load modules needed
# module load Python
module load Miniforge3

# Delete the existing environment
# conda env remove -p /scratch/clevyfidel/conda_env/rnn

# Create and activate conda env
# conda env create -f /scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/rnn.yml -p /scratch/clevyfidel/conda_env/rnn
# conda env create -f /scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/rnn.yml -p /scratch/clevyfidel/conda_env/rnn2


# To update existing env with new packages from yml:
# conda env update -p /scratch/clevyfidel/conda_env/rnn -f /scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/rnn.yml --prune
# conda activate /scratch/clevyfidel/conda_env/rnn 
conda activate /scratch/clevyfidel/conda_env/rnn2

# source ~/.bashrcq

ARGS=""

# Run the Python script (benchmarks only)
/scratch/clevyfidel/conda_env/rnn2/bin/python /scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/pipeline_benchmarks_only.py $ARGS


# Verify Python version and activated environment
echo "Python version: $(python --version)"
echo "Which Python: $(which python)"
