#!/bin/bash
#SBATCH --job-name=traj-rhst-h
#SBATCH --output=logs/traj_rhst_mem_h_%j.log
#SBATCH --error=logs/traj_rhst_mem_h_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:2

set -e
echo "=== Starting Traj-based (R+H+S+T, mem H) on $(hostname) at $(date) ==="
source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export HF_HOME=/home/a5l/shuqing.a5l/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
cd /home/a5l/shuqing.a5l/partnr-planner
bash all_scripts/bash_files/run_traj_rhst_mem_h.sh
exit $?
