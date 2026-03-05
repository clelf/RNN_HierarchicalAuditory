#!/bin/bash
#SBATCH --qos=xlong
#SBATCH --job-name=onectx_rnn
#SBATCH --nodes=1              # Use single node to avoid distributed complexity
#SBATCH --ntasks=1             # Single task (your Python script)
#SBATCH --cpus-per-task=8      # CPUs for parallel data generation
#SBATCH --output=/scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/outputs_terminal/modnet_bn_%j.out
#SBATCH --error=/scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/outputs_terminal/modnet_bn_%j.err
#SBATCH --mail-type=END,FAIL           # Notifications for job done or fail
#SBATCH --mail-user=c.levyfidel@bcbl.eu  # Email address for notifications
#SBATCH --mem=64G              # Increased memory for large batch sizes


# Load modules needed
module load Miniforge3

# Activate conda environment
conda activate /scratch/clevyfidel/conda_env/rnn2

# ============ CONFIGURABLE PARAMETERS ============
# Training ModuleNetwork models with different bottleneck dimensions
# Fixed learning rate: 0.01
# Bottleneck dims: 4, 8, 16, 32 (from experiment.py)
# 
# Add --use_run_id for timestamped subfolder
# Add --objectives to specify learning objectives (obs, ctx, obs_ctx)
# Add --kappa_values for obs_ctx objective weighting
# =================================================

# Navigate to the refactored directory
cd /scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/



# 1 context
/scratch/clevyfidel/conda_env/rnn2/bin/python pipeline_runner_v2.py \
    --models rnn vrnn \
    --learning_rates 0.01 \
    --hidden_dims 32 64 \
    --objectives obs ctx \
    --n_ctx 1 \
    --gm_name NonHierarchicalGM \
    --skip_benchmarks \
    --train_only



# Verify Python version and activated environment
echo "Python version: $(python --version)"
echo "Which Python: $(which python)"
