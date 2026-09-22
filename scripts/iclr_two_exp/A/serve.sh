#!/usr/bin/env bash
# Part A re-ask services. Archived commands, only CUDA_VISIBLE_DEVICES and max_num_seqs changed
# (deviations A2-2). Usage: serve.sh 72B|30B|7B   -> prints the server PID.
set -u
case "$1" in
  72B)
    export HF_HOME=/mnt/pfs/devs/pn5wp/shishuqing/hf_home HF_HUB_OFFLINE=1
    CUDA_VISIBLE_DEVICES=2,3,4,5 setsid nohup /root/venvs/vllm/bin/vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
      --revision 89c86200743eec961a297729e7990e8f2ddbc4c5 --served-model-name qwen2.5-vl-72b-amendment3-f2 \
      --dtype bfloat16 --tensor-parallel-size 4 --max-model-len 16384 --max-num-seqs 64 --async-scheduling \
      --seed 0 --gpu-memory-utilization 0.72 --host 0.0.0.0 --port 8050 > /root/vllm-iclr2exp-72b.log 2>&1 &
    echo $! ;;
  30B)
    export HF_HOME=/mnt/pfs/devs/pn5wp/shishuqing/hf
    CUDA_VISIBLE_DEVICES=2,3,4,5 setsid nohup /root/venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-VL-30B-A3B-Instruct --served-model-name qwen3-vl-30b --port 8062 \
      --tensor-parallel-size 4 --max-model-len 16384 --gpu-memory-utilization 0.80 --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image":1}' > /root/vllm-iclr2exp-30b.log 2>&1 &
    echo $! ;;
  7B)
    export HF_HOME=/mnt/pfs/devs/pn5wp/shishuqing/hf
    CUDA_VISIBLE_DEVICES=2,3 setsid nohup /root/venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-VL-7B-Instruct --served-model-name qwen2.5-vl-7b --port 8061 \
      --tensor-parallel-size 2 --max-model-len 16384 --gpu-memory-utilization 0.80 --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image":1}' > /root/vllm-iclr2exp-7b.log 2>&1 &
    echo $! ;;
esac
