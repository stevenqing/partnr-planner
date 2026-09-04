#!/bin/bash
#SBATCH --job-name=build_skill_memory
#SBATCH --output=/home/a5l/shuqing.a5l/partnr-planner/logs/build_memory_%j.out
#SBATCH --error=/home/a5l/shuqing.a5l/partnr-planner/logs/build_memory_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --partition=workq

# =============================================================================
# SLURM Script: Build Hierarchical Skill Memory from All Heuristic Results
#
# This script:
# 1. Starts a vLLM server with Llama 3.3 70B
# 2. Processes all heuristic trajectory results
# 3. Builds hierarchical skill memory and RAG datasets
#
# Usage:
#   sbatch run_build_all_memories.slurm
#
# Or with custom options:
#   sbatch --export=USE_LLM=true,MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct run_build_all_memories.slurm
# =============================================================================

set -e

echo "=== Build Hierarchical Skill Memory Job Started at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Configuration
PROJECT_DIR="/home/a5l/shuqing.a5l/partnr-planner"
OUTPUTS_DIR="/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner/outputs/habitat_llm"
DATA_DIR="${PROJECT_DIR}/data"
LOG_DIR="${PROJECT_DIR}/logs"

# LLM Configuration (can be overridden via --export)
USE_LLM=${USE_LLM:-true}
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.3-70B-Instruct"}
VLLM_PORT=${VLLM_PORT:-8000}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}

# HuggingFace cache
export HF_HOME=${HF_HOME:-/home/a5l/shuqing.a5l/.cache/huggingface}
export TRANSFORMERS_CACHE=$HF_HOME

# Fix CUDA multiprocessing issue for tensor parallelism
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${DATA_DIR}/hierarchical_skill_memory"
mkdir -p "${DATA_DIR}/rag_datasets"

cd "${PROJECT_DIR}"

# Activate conda environment (adjust as needed)
source ~/.bashrc
conda activate habitat-llm 2>/dev/null || echo "Conda environment 'habitat-llm' not found, using current environment"

# =============================================================================
# Start vLLM Server (if using LLM extraction)
# =============================================================================

VLLM_PID=""

if [ "$USE_LLM" = "true" ]; then
    echo ""
    echo "=== Starting vLLM Server ==="
    echo "Model: $MODEL_NAME"
    echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
    echo "Port: $VLLM_PORT"

    # Start vLLM server in background
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --host "0.0.0.0" \
        --port "$VLLM_PORT" \
        --trust-remote-code \
        --dtype float16 \
        > "${LOG_DIR}/vllm_server_${SLURM_JOB_ID}.log" 2>&1 &

    VLLM_PID=$!
    echo "vLLM server started with PID: $VLLM_PID"

    # Wait for server to be ready
    echo "Waiting for vLLM server to be ready..."
    MAX_WAIT=300  # 5 minutes
    WAITED=0
    while ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "ERROR: vLLM server failed to start within ${MAX_WAIT} seconds"
            kill $VLLM_PID 2>/dev/null || true
            exit 1
        fi
        echo "  Waiting... (${WAITED}s / ${MAX_WAIT}s)"
    done
    echo "vLLM server is ready!"
fi

# =============================================================================
# Define Results Directories
# =============================================================================

# All heuristic results directories (sorted by size, largest first)
RESULTS_DIRS=(
    "2025-12-30_17-02-27-rerange_only.json/results"
    "2025-12-30_21-10-23-heterogeneous+rerange.json/results"
    "2025-12-30_23-44-10-heterogeneous+rerange+spatial+temporal.json/results"
    "2025-12-30_23-42-08-spatial_only.json/results"
    "2025-12-30_23-43-27-heterogeneous+temporal.json/results"
)

# =============================================================================
# Process Each Results Directory
# =============================================================================

echo ""
echo "=== Processing Heuristic Results ==="
echo "Total directories to process: ${#RESULTS_DIRS[@]}"
echo ""

PROCESSED=0
FAILED=0

for rel_path in "${RESULTS_DIRS[@]}"; do
    RESULTS_DIR="${OUTPUTS_DIR}/${rel_path}"

    # Extract dataset name from path
    DATASET_NAME=$(echo "$rel_path" | sed 's|/results||' | sed 's|\.json||' | sed 's|^[0-9_-]*||')

    echo "----------------------------------------"
    echo "Processing: $DATASET_NAME"
    echo "Results dir: $RESULTS_DIR"

    # Check if directory exists
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "  SKIPPED: Directory not found"
        continue
    fi

    # Check if episode log exists
    if [ ! -f "${RESULTS_DIR}/episode_result_log.csv" ]; then
        echo "  SKIPPED: No episode_result_log.csv found"
        continue
    fi

    # Count episodes
    EPISODE_COUNT=$(wc -l < "${RESULTS_DIR}/episode_result_log.csv")
    EPISODE_COUNT=$((EPISODE_COUNT - 1))  # Subtract header
    echo "  Episodes: $EPISODE_COUNT"

    # Skip if too few episodes
    if [ $EPISODE_COUNT -lt 1 ]; then
        echo "  SKIPPED: No episodes to process"
        continue
    fi

    # Output directories
    MEMORY_OUTPUT="${DATA_DIR}/hierarchical_skill_memory/${DATASET_NAME}"
    RAG_OUTPUT="${DATA_DIR}/rag_datasets/${DATASET_NAME}_rag"

    echo "  Memory output: $MEMORY_OUTPUT"
    echo "  RAG output: $RAG_OUTPUT"

    # Build command
    CMD="python our_method/run_build_memory.py \
        --results-dir \"${RESULTS_DIR}\" \
        --memory-output-dir \"${MEMORY_OUTPUT}\" \
        --rag-output-dir \"${RAG_OUTPUT}\""

    if [ "$USE_LLM" = "true" ]; then
        CMD="$CMD --use-llm --vllm-host localhost --vllm-port ${VLLM_PORT} --model-name \"${MODEL_NAME}\""
    fi

    # Run the command
    echo "  Running..."
    if eval $CMD 2>&1 | tee "${LOG_DIR}/build_memory_${DATASET_NAME}_${SLURM_JOB_ID}.log"; then
        echo "  SUCCESS"
        PROCESSED=$((PROCESSED + 1))
    else
        echo "  FAILED"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

# =============================================================================
# Cleanup
# =============================================================================

# Stop vLLM server
if [ -n "$VLLM_PID" ]; then
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    echo "vLLM server stopped"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "Processed: $PROCESSED directories"
echo "Failed: $FAILED directories"
echo "Output location: ${DATA_DIR}"
echo ""
echo "Memory outputs:"
ls -la "${DATA_DIR}/hierarchical_skill_memory/" 2>/dev/null || echo "  (none)"
echo ""
echo "RAG outputs:"
ls -la "${DATA_DIR}/rag_datasets/" 2>/dev/null || echo "  (none)"
echo ""
echo "=== Job Completed at $(date) ==="
