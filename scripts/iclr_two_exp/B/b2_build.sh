#!/bin/bash
# Part B2: build the fold-A and fold-B H_R libraries with the archived builder and proposing model, one fold after
# the other on one Llama-3.3-70B endpoint (one generation job per endpoint). Same resource settings as the 09-22
# route-Q build (scripts/iclr_master/C/build_heldout_q.sh): TP 2 on GPUs 0,1, util 0.80, enforce-eager, port 8090.
# Builder arguments are those of the archived H_R block of our_method/rebuild_all_memories.sh.
# Input per fold: the heuristic H_R log rows of that fold only (B/fold<k>_build_log.csv) + the original traces.
# The wrapper build_with_provenance.py adds provenance.json (source episode per instance); it changes no call.
# Usage (remote): setsid nohup bash b2_build.sh > $W/driver.log 2>&1 < /dev/null &
set -u
P=/mnt/pfs/devs/pn5wp/shishuqing
W=$P/iclr_two_exp_B_build
REPO=$P/partnr-isambard-C
RES=$P/partnr-planner/results/iclr_two_exp_2026-09-22/B
S=$P/partnr-planner/scripts/iclr_two_exp/B
MODEL=$P/models/Llama-3.3-70B-Instruct
PORT=8090
PY=/root/venvs/partnr/bin/python
mkdir -p $W
grep -q "DONE fail=0" $P/models/llama33_70b_download.log || { echo "model download not verified"; exit 1; }
for k in A B; do [ -e $W/lib_$k ] && { echo "refuse: $W/lib_$k exists"; exit 1; }; done
curl -s -o /dev/null http://127.0.0.1:$PORT/health && { echo "refuse: port $PORT already serving"; exit 1; }

export CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1
setsid $HOME/venvs/vllm/bin/vllm serve $MODEL \
  --served-model-name meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.80 --enforce-eager --host 127.0.0.1 --port $PORT > $W/vllm.log 2>&1 < /dev/null &
VPID=$!
echo $VPID > $W/vllm.pid
stop_vllm() { kill -TERM -- -$VPID 2>/dev/null; sleep 20; kill -KILL -- -$VPID 2>/dev/null; echo "$(date '+%F %T') vllm stopped"; }
for i in $(seq 1 180); do
  curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200 && break
  kill -0 $VPID 2>/dev/null || { echo "vllm died"; exit 1; }
  sleep 10
done
curl -s http://127.0.0.1:$PORT/v1/models | grep -q Llama-3.3-70B || { echo "endpoint not up"; stop_vllm; exit 1; }
echo "$(date '+%F %T') vllm up"

for k in A B; do
  mkdir -p $W/in_$k/results/heterogeneous+rerange.json.gz
  ln -sfn $P/iclr_master_C_build/src/traces $W/in_$k/results/heterogeneous+rerange.json.gz/traces
  cp $RES/fold${k}_build_log.csv $W/in_$k/results/episode_result_log.csv
  n0=$(grep -c 'POST /v1/chat/completions HTTP/1.1" 200' $W/vllm.log)
  echo "$(date '+%F %T') build fold $k start ($(($(wc -l < $W/in_$k/results/episode_result_log.csv) - 1)) episodes)"
  (cd $REPO/our_method && timeout 21600 $PY $S/build_with_provenance.py \
      --results-dir $W/in_$k/results --output-dir $W/lib_$k \
      --include-failed --use-llm --use-api --vllm-host 127.0.0.1 --vllm-port $PORT --patch-failed \
      > $W/build_$k.log 2>&1 < /dev/null)
  rc=$?
  n1=$(grep -c 'POST /v1/chat/completions HTTP/1.1" 200' $W/vllm.log)
  echo "$(date '+%F %T') build fold $k rc=$rc requests_200=$((n1 - n0)) api_failed=$(grep -c 'API call failed' $W/build_$k.log)"
done
stop_vllm
echo "B2 DONE"
