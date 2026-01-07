#!/bin/bash
#SBATCH --job-name=ours-rs-r-video
#SBATCH --output=logs/ours_rs_mem_r_video_%j.log
#SBATCH --error=logs/ours_rs_mem_r_video_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --gres=gpu:1

# Video recording script for Ours Method (R+S eval, R memory)
#
# To run specific episodes, set EPISODE_INDICES before sbatch:
#   export EPISODE_INDICES="[0,1,2,3,4]"
#   sbatch run_ours_rs_mem_r_video.slurm
#
# Or modify the EPISODE_INDICES variable below directly.

set -e

# Top 10 most reliable successful episodes (episode IDs -> indices):
# 1312->64, 1317->65, 1327->66, 1378->67, 1425->69, 212->15, 214->17, 866->38, 1471->71, 153->3
# Change this to run different episodes, e.g., "[10,11,12]" or "" for all
export EPISODE_INDICES="${EPISODE_INDICES:-[64,65,66,67,69,15,17,38,71,3]}"

# Single process for video recording (recommended)
export NUM_PROC=1

source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export HF_HOME=/home/a5l/shuqing.a5l/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

cd /home/a5l/shuqing.a5l/partnr-planner

echo "============================================"
echo "Starting video recording run"
echo "Episode indices: ${EPISODE_INDICES}"
echo "Number of processes: ${NUM_PROC}"
echo "============================================"

bash all_scripts/bash_files/run_ours_rs_mem_r_video.sh

echo "============================================"
echo "Video recording complete!"
echo "Videos saved to: outputs/habitat_llm/<run_dir>/results/.../videos/"
echo "============================================"

exit $?
