#!/bin/bash
#SBATCH --job-name=memento-all
#SBATCH --output=logs/memento_all_%j.log
#SBATCH --error=logs/memento_all_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:2

# Create all MEMENTO memory datasets from heuristic trajectories

set -e
echo "=== Starting All MEMENTO Memory Creation on $(hostname) at $(date) ==="

source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export HF_HOME=/home/a5l/shuqing.a5l/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

cd /home/a5l/shuqing.a5l/partnr-planner

bash all_scripts/dataset_creation/create_all_memento.sh

echo "=== Completed at $(date) ==="
