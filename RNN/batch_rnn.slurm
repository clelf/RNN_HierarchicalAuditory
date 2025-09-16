#!/bin/bash
#SBATCH --qos=xlong
#SBATCH --job-name=coin_comp
#SBATCH --nodes=1              # nodes per instance
#SBATCH --ntasks=1            # tasks per instance
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/clevyfidel/Workspace/goin/opt_coin/outputs_terminal/test_opt_coin_%j.out
#SBATCH --error=/scratch/clevyfidel/Workspace/goin/opt_coin/outputs_terminal/test_opt_coin_%j.err
#SBATCH --mail-type=END,FAIL           # Notifications for job done or fail
#SBATCH --mail-user=c.levyfidel@bcbl.eu  # Email address for notifications
#SBATCH --mem=32G

#cd /scratch/clevyfidel/Workspace/goin/opt_coin

# Load modules needed
module load Python


# Create and activate conda env
conda env create -f /scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/rnn.yml
conda activate /scratch/clevyfidel/conda_env/coin

# Add Python path
export LD_LIBRARY_PATH=/scratch/clevyfidel/conda_env/coin/lib:$LD_LIBRARY_PATH


# source ~/.bashrcq


# Install matlabengine
#/scratch/clevyfidel/conda_env/coin/bin/python -m pip install matlabengine==24.2.1


OUTPUT="inference_results_512_mse_parallel"
N_SAMPLES=512
N_TRIALS=512


# Run the Python script
/scratch/clevyfidel/conda_env/coin/bin/python /scratch/clevyfidel/Workspace/goin/opt_coin/test_opt_coin_inf.py --output ${OUTPUT} --samples ${N_SAMPLES} --trials ${N_TRIALS}

# Verify Python version and activated environment
echo "Python version: $(python --version)"
echo "Which Python: $(which python)"
