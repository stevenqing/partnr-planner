#!/bin/bash
#SBATCH --job-name=traj-rht-t
#SBATCH --output=logs/traj_rht_mem_t_%j.log
#SBATCH --error=logs/traj_rht_mem_t_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:2

# Traj-based Method: Transfer R+H+T, Memory T only

set -e

echo "=== Starting Traj-based (R+H+T, mem T) on $(hostname) at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"

source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export HF_HOME=/home/a5l/shuqing.a5l/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

cd /home/a5l/shuqing.a5l/partnr-planner

bash all_scripts/bash_files/run_traj_rht_mem_t.sh

EXIT_STATUS=$?
echo "=== Experiment completed with status $EXIT_STATUS at $(date) ==="
exit $EXIT_STATUS
