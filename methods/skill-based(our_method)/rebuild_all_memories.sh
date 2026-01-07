#!/bin/bash
# Rebuild all hierarchical skill memories with action sequences
# Uses vLLM server with Llama 3.3 70B Instruct for FAST skill extraction
# Includes failed episodes and uses LLM to patch/analyze them

BASE_DIR="/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner"
HEURISTIC_DIR="$BASE_DIR/heuristic_dataset"
OUTPUT_BASE="$BASE_DIR/data/hierarchical_skill_memory"
SCRIPT_DIR="$BASE_DIR/methods/skill-based(our_method)"

# Model and vLLM settings
MODEL_PATH="/home/a5l/shuqing.a5l/models/Llama-3.3-70B-Instruct"
VLLM_HOST="localhost"
VLLM_PORT=8000
TENSOR_PARALLEL=4

# Fix CUDA multiprocessing issue with vLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "$SCRIPT_DIR"

echo "=========================================="
echo "Rebuilding Hierarchical Skill Memories"
echo "Using vLLM Server: $MODEL_PATH"
echo "Tensor Parallel: $TENSOR_PARALLEL GPUs"
echo "Including failed episodes with LLM patching"
echo "=========================================="

# Start vLLM server in background
echo ""
echo "=== Starting vLLM Server ==="
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "meta-llama/Llama-3.3-70B-Instruct" \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --host $VLLM_HOST \
    --port $VLLM_PORT \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    &
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=300  # 5 minutes max
WAIT_TIME=0
while ! curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; do
    sleep 5
    WAIT_TIME=$((WAIT_TIME + 5))
    echo "  Waiting... ($WAIT_TIME seconds)"
    if [ $WAIT_TIME -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done
echo "vLLM server is ready!"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null
    echo "vLLM server stopped."
}
trap cleanup EXIT

# Rerange Only
echo ""
echo "=== Building hierarchical_rerange_only ==="
python build_hierarchical_skill_memory.py \
    --results-dir "$HEURISTIC_DIR/2025-12-30_17-02-27-rerange_only.json/results" \
    --output-dir "$OUTPUT_BASE/hierarchical_rerange_only" \
    --include-failed \
    --use-llm \
    --use-api \
    --vllm-host $VLLM_HOST \
    --vllm-port $VLLM_PORT \
    --patch-failed

# Heterogeneous + Rerange
echo ""
echo "=== Building hierarchical_heterogeneous_rerange ==="
python build_hierarchical_skill_memory.py \
    --results-dir "$HEURISTIC_DIR/2025-12-30_21-10-23-heterogeneous+rerange.json/results" \
    --output-dir "$OUTPUT_BASE/hierarchical_heterogeneous_rerange" \
    --include-failed \
    --use-llm \
    --use-api \
    --vllm-host $VLLM_HOST \
    --vllm-port $VLLM_PORT \
    --patch-failed

# Spatial Only
echo ""
echo "=== Building hierarchical_spatial_only ==="
python build_hierarchical_skill_memory.py \
    --results-dir "$HEURISTIC_DIR/2025-12-30_23-42-08-spatial_only.json/results" \
    --output-dir "$OUTPUT_BASE/hierarchical_spatial_only" \
    --include-failed \
    --use-llm \
    --use-api \
    --vllm-host $VLLM_HOST \
    --vllm-port $VLLM_PORT \
    --patch-failed

# Heterogeneous + Temporal
echo ""
echo "=== Building hierarchical_heterogeneous_temporal ==="
python build_hierarchical_skill_memory.py \
    --results-dir "$HEURISTIC_DIR/2025-12-30_23-43-27-heterogeneous+temporal.json/results" \
    --output-dir "$OUTPUT_BASE/hierarchical_heterogeneous_temporal" \
    --include-failed \
    --use-llm \
    --use-api \
    --vllm-host $VLLM_HOST \
    --vllm-port $VLLM_PORT \
    --patch-failed

# Heterogeneous + Rerange + Spatial + Temporal (Full)
echo ""
echo "=== Building hierarchical_heterogeneous_rerange_spatial_temporal ==="
python build_hierarchical_skill_memory.py \
    --results-dir "$HEURISTIC_DIR/2025-12-30_23-44-10-heterogeneous+rerange+spatial+temporal.json/results" \
    --output-dir "$OUTPUT_BASE/hierarchical_heterogeneous_rerange_spatial_temporal" \
    --include-failed \
    --use-llm \
    --use-api \
    --vllm-host $VLLM_HOST \
    --vllm-port $VLLM_PORT \
    --patch-failed

echo ""
echo "=========================================="
echo "All memories rebuilt!"
echo "=========================================="
