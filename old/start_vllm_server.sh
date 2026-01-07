#!/bin/bash

# vLLM Server Launch Script for Qwen2.5-7B-Instruct
# This script starts a vLLM server with tensor parallelism for faster inference

set -e

# Configuration
MODEL=${VLLM_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}  # Number of GPUs for tensor parallelism
PORT=${VLLM_PORT:-8000}
HOST=${VLLM_HOST:-"0.0.0.0"}

# HuggingFace cache
export HF_HOME=${HF_HOME:-/home/a5l/shuqing.a5l/.cache/huggingface}
export TRANSFORMERS_CACHE=$HF_HOME

echo "=== Starting vLLM Server ==="
echo "Model: $MODEL"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Host: $HOST"
echo "Port: $PORT"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --dtype float16
