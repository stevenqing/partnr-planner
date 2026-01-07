#!/bin/bash
#SBATCH --job-name=rag-traj-8b-hrst
#SBATCH --output=logs/planner_rag_traj_8b_%j.log
#SBATCH --error=logs/planner_rag_traj_8b_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:2

# PartnR Planner Demo with RAG - SLURM Script
# Uses Meta-Llama-3.1-8B-Instruct model with RAG-enhanced prompts
# Dataset: heterogeneous+rerange+spatial+temporal (2 tasks)
# Parallel execution with 2 processes

set -e

echo "=== Starting Planner Demo with RAG on $(hostname) at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Number of tasks: $SLURM_NTASKS"

# Activate conda environment
source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

# Configure headless EGL rendering with NVIDIA GPU
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Set Habitat to use headless rendering
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

# HuggingFace cache directory
export HF_HOME=/home/a5l/shuqing.a5l/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# Project paths
PROJECT_DIR=/home/a5l/shuqing.a5l/partnr-planner
DATA_DIR=$PROJECT_DIR/data

# Dataset path
DATASET_PATH="$PROJECT_DIR/task_classification_datasets/heterogeneous+rerange+spatial+temporal.json.gz"

# Verify dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

# Verify RAG dataset exists
RAG_DATASET_DIR="$DATA_DIR/rag_datasets/rerange_only_organized_by_skills/"
if [ ! -d "$RAG_DATASET_DIR" ]; then
    echo "ERROR: RAG dataset directory not found at $RAG_DATASET_DIR"
    exit 1
fi

echo "Using dataset: $DATASET_PATH"
echo "Using RAG dataset: $RAG_DATASET_DIR"
echo "Python: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Change to project directory
cd $PROJECT_DIR

echo "=== Starting planner demo with RAG at $(date) ==="

# Run the planner demo with 2 parallel processes
python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    evaluation.sequential_execution=True \
    +evaluation.num_proc=2 \
    habitat.dataset.data_path="$DATASET_PATH" \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=1200 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=1200 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=meta-llama/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_0.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_1.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir=[data/rag_datasets/rerange_only_organized_by_skills/] \
    evaluation.agents.agent_1.planner.plan_config.rag_dataset_dir=[data/rag_datasets/rerange_only_organized_by_skills/] \
    evaluation.agents.agent_0.planner.plan_config.rag_data_source_name=[react_trajectories] \
    evaluation.agents.agent_1.planner.plan_config.rag_data_source_name=[react_trajectories] \
    evaluation.agents.agent_0.planner.plan_config.example_type=skills \
    evaluation.agents.agent_1.planner.plan_config.example_type=skills \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=rag_prompt_sequential_state_reflection \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=rag_prompt_sequential_state_reflection \
    evaluation.agents.agent_0.planner.plan_config.constrained_generation=False \
    evaluation.agents.agent_1.planner.plan_config.constrained_generation=False

EXIT_STATUS=$?

echo "=== Planner demo completed with status $EXIT_STATUS at $(date) ==="

exit $EXIT_STATUS
