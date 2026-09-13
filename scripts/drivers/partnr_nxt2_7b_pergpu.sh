#!/usr/bin/env bash
# The 7B arm on gate_S3, one queue per GPU instead of waves.
#
# One cell saturates a GPU (habitat rendering at 97%) while the endpoint sits idle, so the
# waves left GPU 0 empty whenever one cell of a pair finished early. Each GPU now takes its
# next cell the moment its last one is gone. A and D (cand0 / cand3) come first: they are
# the informative pair, D being the negative control.
#
#   bash scripts/drivers/partnr_nxt2_7b_pergpu.sh <pid of the running cand0 driver>
#
# The running cand0 cell is waited on BY PID. A GPU is taken only once its memory is back
# under a floor, because a killed habitat runner leaves forkserver orphans holding VRAM
# (GPU 1 keeps ~30G for the 7B endpoint itself).
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
CAND0_DRIVER=${1:?pid of the cand0 driver}
say () { echo "[$(date +%H:%M:%S)] $*"; }

wait_gpu () {  # gpu floor_mib
  while :; do
    used=$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "$used" -lt "$2" ] && return 0
    sleep 30
  done
}

run () {  # cell gpu floor_mib
  wait_gpu "$2" "$3"
  if ! curl -sf -m 5 http://127.0.0.1:8061/v1/models | grep -q qwen2.5-vl-7b; then
    say "endpoint 8061 not answering; refusing $1"; return 1
  fi
  say "start $1 on GPU $2"
  LLM_MODEL=qwen2.5-vl-7b VLLM_BASE_URL=http://127.0.0.1:8061/v1 \
  CONFIG=baselines/skill_memory_v2_vllm.yaml \
  OPERATORS=results/partnr_gate/nxt2/$1.json GPU=$2 PROCS=60 \
  HARD_TIMEOUT=7200 STALL_SECONDS=1800 \
  bash scripts/drivers/partnr_gate_cell.sh gate_S3 "$1" outputs/gate/nxt2_7b +resume=True
  say "end $1 ($(ls outputs/gate/nxt2_7b/$1/results/gate_S3.json.gz/stats 2>/dev/null | wc -l) episodes)"
}

( run cand3 0 10000; run cand2 0 10000 ) &
( while kill -0 "$CAND0_DRIVER" 2>/dev/null; do sleep 20; done
  say "cand0 driver $CAND0_DRIVER exited"
  run cand1 1 40000 ) &
wait
say "ALL CELLS DONE"
