#!/bin/bash
#SBATCH --job-name=rebuild_memory
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rebuild_memory_%j.log
#SBATCH --error=logs/rebuild_memory_%j.log

# Rebuild hierarchical skill memory with local Llama 3.3 70B Instruct

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load environment
source ~/.bashrc
conda activate habitat-llm

BASE_DIR="/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner"
MODEL_PATH="/home/a5l/shuqing.a5l/models/Llama-3.3-70B-Instruct"

echo "Using local model: $MODEL_PATH"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd "$BASE_DIR"

# Run memory rebuild with local model
echo ""
echo "Running memory rebuild with local Llama 3.3 70B..."
cd "$BASE_DIR/methods/skill-based(our_method)"
bash rebuild_all_memories.sh

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
