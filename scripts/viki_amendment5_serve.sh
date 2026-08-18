#!/usr/bin/env bash

set -euo pipefail

backbone=${1:?Usage: viki_amendment5_serve.sh BACKBONE [PORT]}
port=${2:-8060}

case "$backbone" in
  qwen2_5_vl_7b_stock)
    model="Qwen/Qwen2.5-VL-7B-Instruct"
    revision="cc594898137f460bfe9f0759e9844b3ce807cfb5"
    served_model="qwen2.5-vl-7b-amendment5"
    tensor_parallel_size=1
    default_gpus=2
    ;;
  qwen3_vl_30b_a3b)
    model="Qwen/Qwen3-VL-30B-A3B-Instruct"
    revision="9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
    served_model="qwen3-vl-30b-a3b-amendment5"
    tensor_parallel_size=4
    default_gpus=2,4,6,7
    ;;
  *)
    echo "Unknown Amendment 5 backbone: $backbone" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES=${AMENDMENT5_GPUS:-$default_gpus}
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
