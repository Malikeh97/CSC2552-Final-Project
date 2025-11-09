#!/bin/bash

#SBATCH --job-name=preference_collapse
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/tulu_runs/pref_collapse/run_logs/%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/tulu_runs/pref_collapse/run_logs/%x_%j.err
#SBATCH --partition=gpubase_l40s_b2
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --account=aip-craffel
#SBATCH --time=12:00:00

# Load any necessary modules (adjust based on your cluster setup)
# module load python/3.11

# Activate your virtual environment
source /project/6104653/ehghaghi/CSC2552-Final-Project/myenv/bin/activate

# Print some info for debugging
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"

# Navigate to your project directory
cd /project/6104653/ehghaghi/CSC2552-Final-Project/src/diagnosis

# Run the Python script
python preference_collapse_metrics.py

echo "Job completed at: $(date)"