#!/bin/bash
#SBATCH --job-name=topdown-viz
#SBATCH --output=logs/topdown_viz_%j.log
#SBATCH --error=logs/topdown_viz_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Generate top-down task visualization with 3D object highlights
# Episode 1312 (index 64)

set -e

source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

cd /home/a5l/shuqing.a5l/partnr-planner

echo "Generating top-down visualization for episode 1312..."

python generate_topdown_task_viz.py \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    +episode_indices=[64] \
    habitat.dataset.data_path="task_classification_datasets/rerange+spatial_matched_subtasks.json.gz" \
    num_proc=1 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=/home/a5l/shuqing.a5l/models/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=/home/a5l/shuqing.a5l/models/Llama-3.1-8B-Instruct

echo "Done!"
