#!/bin/bash
#SBATCH --job-name=traj-rs-r
#SBATCH --output=logs/traj_rs_mem_r_%j.log
#SBATCH --error=logs/traj_rs_mem_r_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:2

# Traj-based Method: Transfer R+S, Memory R only

set -e

echo "=== Starting Traj-based (R+S, mem R) on $(hostname) at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

# Activate conda environment
source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

# Configure headless EGL rendering
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

# HuggingFace cache
export HF_HOME=/home/a5l/shuqing.a5l/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

PROJECT_DIR=/home/a5l/shuqing.a5l/partnr-planner
cd $PROJECT_DIR

echo "=== Starting experiment at $(date) ==="

bash all_scripts/bash_files/run_traj_rs_mem_r.sh

EXIT_STATUS=$?
echo "=== Experiment completed with status $EXIT_STATUS at $(date) ==="
exit $EXIT_STATUS
