#!/bin/bash
# vLLM endpoint for the Qwen2.5-7B cells, mirroring the archived Isambard Qwen serving
# (old/start_vllm_server.sh, old/run_planner_demo_vllm.slurm: vllm api server, --dtype float16,
# --max-model-len 32768, planner llm=qwen inference_mode=vllm, constrained_generation=False).
# Resource-only differences: one GPU (TP=1), port 8090, gpu-memory-utilization 0.55 (habitat shares the card).
# Usage: serve_qwen7b.sh <gpu>
PFS=/mnt/pfs/devs/pn5wp/shishuqing
export HF_HOME=$PFS/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${1:?gpu}
exec /root/venvs/vllm/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --dtype float16 --tensor-parallel-size 1 --max-model-len 32768 \
  --seed 0 --gpu-memory-utilization 0.55 --host 127.0.0.1 --port 8090
