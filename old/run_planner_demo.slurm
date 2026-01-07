#!/bin/bash
#SBATCH --job-name=planner-demo
#SBATCH --output=planner_demo_%j.log
#SBATCH --error=planner_demo_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2

# PartnR Planner Demo - SLURM Script
# Uses Qwen2.5-7B-Instruct model with RAG-enhanced prompts

set -e

echo "=== Starting Planner Demo on $(hostname) at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

# Activate conda environment
source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

# Configure headless EGL rendering with NVIDIA GPU
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Set Habitat to use headless rendering
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

# HuggingFace cache directory (use local scratch if available)
export HF_HOME=/home/a5l/shuqing.a5l/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# Project paths
PROJECT_DIR=/home/a5l/shuqing.a5l/partnr-planner
DATA_DIR=$PROJECT_DIR/data

# Dataset path - use val_mini for quick testing, or val for full evaluation
DATASET_PATH="$DATA_DIR/datasets/partnr_episodes/v0_0/val_mini.json.gz"

# Verify dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

echo "Using dataset: $DATASET_PATH"
echo "Python: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Change to project directory
cd $PROJECT_DIR

echo "=== Starting planner demo at $(date) ==="

# Run the planner demo
python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="$DATASET_PATH" \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=qwen \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=qwen \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=800 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=800 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=Qwen/Qwen2.5-7B-Instruct \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=Qwen/Qwen2.5-7B-Instruct \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=zero_shot_prompt_qwen_rag \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=zero_shot_prompt_qwen_rag \
    evaluation.agents.agent_0.planner.plan_config.constrained_generation=False \
    evaluation.agents.agent_1.planner.plan_config.constrained_generation=False

EXIT_STATUS=$?

echo "=== Planner demo completed with status $EXIT_STATUS at $(date) ==="

exit $EXIT_STATUS
