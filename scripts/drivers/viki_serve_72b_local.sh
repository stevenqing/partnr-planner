#!/usr/bin/env bash
# Qwen2.5-VL-72B-Instruct on this box, replacing the archived endpoint (192.168.32.40:8050, down
# since 2026-09-14). Same checkpoint revision, served name, TP, context length, dtype, multimodal
# and generation-config flags as scripts/viki_amendment3_serve.sh qwen2_5_vl_72b_f2.
# Differences, recorded here and in the paper README: this box's vLLM (/root/venvs/vllm, 0.11.2)
# instead of .venv-amendment3; GPUs ${GPUS}; --max-num-seqs ${MAX_SEQS} (archived 16) for throughput,
# which changes batch composition but not the request; bound to 127.0.0.1.
#
#   GPUS=1,2,3,5 PORT=8050 setsid nohup bash scripts/drivers/viki_serve_72b_local.sh \
#     > /root/vllm-72b-local-0914.log 2>&1 < /dev/null &
set -eu
GPUS=${GPUS:-1,2,3,5}
PORT=${PORT:-8050}
MAX_SEQS=${MAX_SEQS:-64}
GPU_UTIL=${GPU_UTIL:-0.70}
export HF_HOME=${HF_HOME:-/mnt/pfs/devs/pn5wp/shishuqing/hf}
export CUDA_VISIBLE_DEVICES=$GPUS
export OMP_NUM_THREADS=1
exec /root/venvs/vllm/bin/vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --revision 89c86200743eec961a297729e7990e8f2ddbc4c5 \
  --served-model-name qwen2.5-vl-72b-amendment3-f2 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs "$MAX_SEQS" \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"use_fast":false}' \
  --mm-processor-cache-gb 0 \
  --async-scheduling \
  --generation-config vllm \
  --seed 0
