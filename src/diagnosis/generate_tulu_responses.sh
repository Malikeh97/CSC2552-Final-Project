#!/bin/bash

#SBATCH --job-name=tulu_generate
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/tulu_runs/run_logs/%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/tulu_runs/run_logs/%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --account=aip-craffel
#SBATCH --time=23:00:00

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU resources: $CUDA_VISIBLE_DEVICES"

echo "Running on node: $(hostname)"
nvidia-smi -q -d ECC

# Load modules
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11
source /home/ehghaghi/projects/aip-craffel/ehghaghi/c-btm-distillation/uv-x86_64-unknown-linux-gnu/distill_env/bin/activate

pip install protobuf sentencepiece

# Export cache dirs
export SCRATCH="/home/ehghaghi/scratch/ehghaghi"
export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache
export TMPDIR="$SCRATCH/tmp"
mkdir -p $SCRATCH/tulu_runs/run_logs $SCRATCH/tmp

# ====================
# Configuration
# ====================
INPUT_CSV="/home/ehghaghi/projects/aip-craffel/ehghaghi/CSC2552-Final-Project/data/prompt_controversy.csv"  # UPDATE THIS
OUTPUT_DIR="$SCRATCH/tulu_runs/outputs"
MODEL_CHECKPOINT="meta-llama/Llama-3.1-8B"  # UPDATE THIS
MODEL_NAME="Llama-3.1-8B" 
N_RESPONSES=10
MAX_NEW_TOKENS=512
TEMPERATURE=1.0
TOP_P=0.9

# Create output directory
mkdir -p $OUTPUT_DIR

# Set output filename based on model type
OUTPUT_CSV="$OUTPUT_DIR/prompt_controversy_quality_variance_${MODEL_NAME}_responses_${SLURM_JOB_ID}.csv"

echo "================================"
echo "Configuration:"
echo "Input CSV: $INPUT_CSV"
echo "Output CSV: $OUTPUT_CSV"
echo "Model Checkpoint: $MODEL_CHECKPOINT"
echo "N Responses: $N_RESPONSES"
echo "================================"

# Build the command
CMD="python -u generate_tulu_responses_v2.py \
    --input_csv $INPUT_CSV \
    --output_csv $OUTPUT_CSV \
    --model_checkpoint $MODEL_CHECKPOINT \
    --n_responses $N_RESPONSES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P"

# Run the script
eval $CMD

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) finished at $(date)"