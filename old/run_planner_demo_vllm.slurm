#!/bin/bash
#SBATCH --job-name=planner-vllm
#SBATCH --output=planner_vllm_%j.log
#SBATCH --error=planner_vllm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# PartnR Planner Demo with vLLM - SLURM Script
# Uses vLLM for faster inference with tensor parallelism

set -e

echo "=== Starting Planner Demo with vLLM on $(hostname) at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

# Load CUDA 12.6 for vLLM compatibility
module load cuda/12.6

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

# vLLM configuration
export VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export TENSOR_PARALLEL_SIZE=4
export VLLM_PORT=8000
export VLLM_HOST="localhost"

# Fix CUDA multiprocessing issue - use spawn instead of fork
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Dataset path
DATASET_PATH="$DATA_DIR/datasets/partnr_episodes/v0_0/val_mini.json.gz"

if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

echo "Using dataset: $DATASET_PATH"
echo "Python: $(which python)"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

cd $PROJECT_DIR

# Start vLLM server in background with optimizations for throughput
echo "=== Starting vLLM server at $(date) ==="
python -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 512 \
    --enable-chunked-prefill \
    --disable-log-requests \
    --uvicorn-log-level warning &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to be ready..."
for i in {1..60}; do
    if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: vLLM server failed to start"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "Waiting... ($i/60)"
    sleep 5
done

echo "=== Starting planner demo at $(date) ==="

# Run the planner demo with vLLM inference mode
# num_proc=12: 12 processes × 2 agents = 24 concurrent vLLM requests for good GPU utilization
python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="$DATASET_PATH" \
    num_proc=12 \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=qwen \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=qwen \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=800 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=800 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=vllm \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=vllm \
    evaluation.agents.agent_0.planner.plan_config.llm.vllm_host=$VLLM_HOST \
    evaluation.agents.agent_1.planner.plan_config.llm.vllm_host=$VLLM_HOST \
    evaluation.agents.agent_0.planner.plan_config.llm.vllm_port=$VLLM_PORT \
    evaluation.agents.agent_1.planner.plan_config.llm.vllm_port=$VLLM_PORT \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=Qwen/Qwen2.5-7B-Instruct \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=Qwen/Qwen2.5-7B-Instruct \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=zero_shot_prompt_qwen_rag \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=zero_shot_prompt_qwen_rag \
    evaluation.agents.agent_0.planner.plan_config.constrained_generation=False \
    evaluation.agents.agent_1.planner.plan_config.constrained_generation=False

EXIT_STATUS=$?

# Cleanup: kill vLLM server
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true

echo "=== Planner demo completed with status $EXIT_STATUS at $(date) ==="

exit $EXIT_STATUS
