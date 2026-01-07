#!/bin/bash
#SBATCH --job-name=ours-rule-hrt-r
#SBATCH --output=logs/ours_rule_hrt_mem_r_%j.log
#SBATCH --error=logs/ours_rule_hrt_mem_r_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

set -e
source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export HF_HOME=/home/a5l/shuqing.a5l/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
cd /home/a5l/shuqing.a5l/partnr-planner
export PYTHONPATH=/home/a5l/shuqing.a5l/partnr-planner:$PYTHONPATH
bash all_scripts/bash_files/rule_based/run_ours_hrt_mem_r_rule.sh
exit $?
