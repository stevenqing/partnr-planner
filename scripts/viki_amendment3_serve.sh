#!/usr/bin/env bash

set -euo pipefail

candidate=${1:?Usage: viki_amendment3_serve.sh CANDIDATE [PORT]}
port=${2:-8050}

case "$candidate" in
  qwen2_5_vl_72b)
    model="Qwen/Qwen2.5-VL-72B-Instruct"
    revision="89c86200743eec961a297729e7990e8f2ddbc4c5"
    served_model="qwen2.5-vl-72b-amendment3-f1"
    tensor_parallel_size=4
    ;;
  qwen2_5_vl_72b_f2)
    model="Qwen/Qwen2.5-VL-72B-Instruct"
    revision="89c86200743eec961a297729e7990e8f2ddbc4c5"
    served_model="qwen2.5-vl-72b-amendment3-f2"
    tensor_parallel_size=4
    ;;
  qwen3_vl_32b)
    model="Qwen/Qwen3-VL-32B-Instruct"
    revision="0cfaf48183f594c314753d30a4c4974bc75f3ccb"
    served_model="qwen3-vl-32b-amendment3-f1"
    tensor_parallel_size=4
    ;;
  qwen3_vl_30b_a3b)
    model="Qwen/Qwen3-VL-30B-A3B-Instruct"
    revision="9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
    served_model="qwen3-vl-30b-a3b-amendment3-f1"
    tensor_parallel_size=4
    ;;
  *)
    echo "Unknown or unavailable candidate: $candidate" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES=0,1,3,5
export OMP_NUM_THREADS=1

exec .venv-amendment3/bin/vllm serve "$model" \
  --revision "$revision" \
  --served-model-name "$served_model" \
  --host 127.0.0.1 \
  --port "$port" \
  --tensor-parallel-size "$tensor_parallel_size" \
  --max-model-len 16384 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.70 \
  --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"use_fast":false}' \
  --mm-processor-cache-gb 0 \
  --async-scheduling \
  --generation-config vllm \
  --seed 0
