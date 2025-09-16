#!/bin/bash
#SBATCH --qos=xlong
#SBATCH --job-name=rnn
#SBATCH --nodes=48              # nodes per instance
#SBATCH --ntasks=4            # tasks per instance
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/outputs_terminal/test_opt_coin_%j.out
#SBATCH --error=/scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/outputs_terminal/test_opt_coin_%j.err
#SBATCH --mail-type=END,FAIL           # Notifications for job done or fail
#SBATCH --mail-user=c.levyfidel@bcbl.eu  # Email address for notifications
#SBATCH --mem=32G

#cd /scratch/clevyfidel/Workspace/goin/opt_coin

# Load modules needed
# module load Python
module load Miniforge3

# Create and activate conda env
conda env create -f /scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/rnn.yml -p /scratch/clevyfidel/conda_env/rnn
conda activate /scratch/clevyfidel/conda_env/rnn

# Add Python path
export LD_LIBRARY_PATH=/scratch/clevyfidel/conda_env/coin/rnn:$LD_LIBRARY_PATH


# source ~/.bashrcq


# Run the Python script
/scratch/clevyfidel/conda_env/rnn/bin/python /scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/pipeline.py

# Verify Python version and activated environment
echo "Python version: $(python --version)"
echo "Which Python: $(which python)"
