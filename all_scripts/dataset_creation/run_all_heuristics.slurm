#!/bin/bash
#SBATCH --job-name=heur-all-tasks
#SBATCH --output=logs/heuristic_all_tasks_%j.log
#SBATCH --error=logs/heuristic_all_tasks_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:2

# Run all heuristic trajectory generations for R, H, S, T task types

set -e
echo "=== Starting All Heuristic Generations on $(hostname) at $(date) ==="

source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export HF_HOME=/home/a5l/shuqing.a5l/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

cd /home/a5l/shuqing.a5l/partnr-planner

bash all_scripts/dataset_creation/run_all_heuristics.sh

echo "=== Completed at $(date) ==="
