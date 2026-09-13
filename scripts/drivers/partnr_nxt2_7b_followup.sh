#!/usr/bin/env bash
# What the 7B arm on gate_S3 still needs before its adjudication means anything.
#
#   1. D (cand3) crashed on 25857 with `Object handbag_0 has no parent`
#      (world_graph.py:211), as A/C/D did on the oracle arm. One episode, one process:
#      is the crash deterministic?
#   2. A same-config self-repeat of base. band = 0 was measured on the privileged arm;
#      the model arm runs at temperature 0 but vLLM batching is not bit-exact, so its band
#      is unknown. Started now on GPU 0 beside C -- memory fits (33G + ~33G of 98G) and
#      the endpoint is idle, so the only cost is shared rendering.
#   3. Once the per-GPU queue (B, C) and both of the above are gone: compare B and C,
#      compare base against its repeat, then the four-criterion adjudication.
#
#   bash scripts/drivers/partnr_nxt2_7b_followup.sh <pid of partnr_nxt2_7b_pergpu.sh>
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
QUEUE=${1:?pid of the per-GPU queue}
PY=/root/venvs/partnr/bin/python
say () { echo "[$(date +%H:%M:%S)] $*"; }

curl -sf -m 5 http://127.0.0.1:8061/v1/models | grep -q qwen2.5-vl-7b \
  || { say "endpoint 8061 not answering; refusing to start"; exit 1; }

cell () {  # outbase cell operators gpu procs [overrides...]
  local outbase=$1 name=$2 ops=$3 gpu=$4 procs=$5; shift 5
  LLM_MODEL=qwen2.5-vl-7b VLLM_BASE_URL=http://127.0.0.1:8061/v1 \
  CONFIG=baselines/skill_memory_v2_vllm.yaml \
  OPERATORS=$ops GPU=$gpu PROCS=$procs HARD_TIMEOUT=7200 STALL_SECONDS=1800 \
  bash scripts/drivers/partnr_gate_cell.sh gate_S3 "$name" "$outbase" "$@"
}

say "start retry cand3/25857 on GPU 1"
cell outputs/gate/nxt2_7b_retry cand3 results/partnr_gate/nxt2/cand3.json 1 1 \
  "+episode_id_filter=[25857]" &
retry=$!
say "start base self-repeat on GPU 0"
cell outputs/gate/nxt2_7b_r2 base results/partnr_gate/nxt2/base.json 0 60 &
repeat=$!

wait "$retry"
say "retry done: $(python3 -c "
import json
b = json.load(open('outputs/gate/nxt2_7b_retry/cand3/results/gate_S3.json.gz/stats/25857.json'))
print('crashed: ' + b['info'].strip().splitlines()[-1] if 'stats' not in b
      else 'scored %s' % json.loads(b['stats'])['task_percent_complete'])
" 2>&1)"

wait "$repeat"
say "self-repeat done"
while kill -0 "$QUEUE" 2>/dev/null; do sleep 30; done
say "per-GPU queue done"

for c in cand1 cand2; do
  $PY scripts/partnr_gate_compare.py --pool gate_S3 --a outputs/gate/nxt2_7b/base \
    --b outputs/gate/nxt2_7b/$c --key is_next_to --json outputs/gate/nxt2_7b/$c/compare.json > /dev/null
  say "compared $c"
done
mkdir -p outputs/gate/nxt2_7b_r2/report
$PY scripts/partnr_gate_compare.py --pool gate_S3 --a outputs/gate/nxt2_7b/base \
  --b outputs/gate/nxt2_7b_r2/base --key is_next_to \
  --json outputs/gate/nxt2_7b_r2/report/band.json > /dev/null
say "compared base against its repeat"

$PY scripts/partnr_gate_adjudicate.py --tag nxt2_7b --pool gate_S3 \
  --candidates results/partnr_candidates_next_to_handwritten.json \
  --library results/partnr_operators_iir1.json > outputs/gate/nxt2_7b/adjudicate.stdout
say "adjudicated -> outputs/gate/nxt2_7b/ADJUDICATION.json"
say "ALL DONE"
