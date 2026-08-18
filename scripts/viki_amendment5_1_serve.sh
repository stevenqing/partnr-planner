#!/usr/bin/env bash

set -euo pipefail

backbone=${1:?Usage: viki_amendment5_1_serve.sh rl_7b [PORT]}
port=${2:-8060}

if [[ "$backbone" != "rl_7b" ]]; then
  echo "Unknown Amendment 5.1 backbone: $backbone" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=${AMENDMENT5_1_GPUS:-2}
export OMP_NUM_THREADS=1

exec /home/aiscuser/VIKI-R/.venv-eval/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model /home/aiscuser/VIKI-R/models/Qwen2.5VL-7B-Instruct-VIKI-R-2 \
  --served-model-name viki-r-7b-l2-amendment1 \
  --host 127.0.0.1 \
  --port "$port" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.65 \
  --max-num-seqs 4 \
  --limit-mm-per-prompt image=1,video=0 \
  --disable-log-requests \
  --seed 0
