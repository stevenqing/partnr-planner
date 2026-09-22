#!/bin/bash
# Serve Llama-3.1-8B-Instruct (snapshot 0e9e39f, the one the HF path loads) with vLLM on one GPU, fp16 like the HF
# path (torch.float16). Usage: serve_8b.sh <gpu> <port> <gpu_mem_util> [max_num_seqs]
# Writes $W/vllm8b_<gpu>.{log,pid}. Refuses if the port already answers.
set -u
GPU=$1; PORT=$2; UTIL=$3; MAXSEQ=${4:-256}; MAXLEN=${5:-65536}
P=/mnt/pfs/devs/pn5wp/shishuqing
W=$P/iclr_two_exp_B_run
mkdir -p $W
MODEL=$P/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
curl -s -o /dev/null -m 3 http://127.0.0.1:$PORT/health && { echo "refuse: port $PORT answers"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1
setsid nohup $HOME/venvs/vllm/bin/vllm serve $MODEL --served-model-name llama31-8b \
  --dtype float16 --max-model-len $MAXLEN --gpu-memory-utilization $UTIL --max-num-seqs $MAXSEQ \
  --enable-prefix-caching --host 127.0.0.1 --port $PORT > $W/vllm8b_$GPU.log 2>&1 < /dev/null &
echo $! > $W/vllm8b_$GPU.pid
echo "started pid $(cat $W/vllm8b_$GPU.pid) gpu $GPU port $PORT"
