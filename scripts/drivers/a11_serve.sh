#!/bin/bash
# Serve the two extra models on this box; the 72B stays on its own endpoint.
#
# Eight cards are free here, so the 7B takes two and the 30B takes four; both are given
# the same context length the archived runs used, and both are addressed by a served
# name that says which model it is, so a result file can never be traced to the wrong one.
export HF_HOME=/mnt/pfs/devs/pn5wp/shishuqing/hf
V=/root/venvs/vllm/bin/python

start() {  # name port gpus tp model
  local name=$1 port=$2 gpus=$3 tp=$4 model=$5
  if curl -s -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    echo "$name already up on $port"; return
  fi
  echo "starting $name on port $port (GPUs $gpus)"
  CUDA_VISIBLE_DEVICES=$gpus nohup $V -m vllm.entrypoints.openai.api_server \
    --model "$model" --served-model-name "$name" \
    --port "$port" --tensor-parallel-size "$tp" \
    --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":1}' \
    > "/root/vllm-$name.log" 2>&1 &
}

start qwen2.5-vl-7b   8061 0,1     2 Qwen/Qwen2.5-VL-7B-Instruct
start qwen3-vl-30b    8062 2,3,4,5 4 Qwen/Qwen3-VL-30B-A3B-Instruct
echo "launched; watch /root/vllm-*.log"
