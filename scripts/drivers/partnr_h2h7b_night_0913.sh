#!/usr/bin/env bash
# Overnight tail of the 7B head-to-head: the two memory baselines that are still missing,
# G-Memory and MEMENTO, queued behind tonight's three cells, then one report over all five.
#
# MEMENTO's first step names the entities an instruction relies on, precomputed per
# instruction. The archived table was made by qwen3-vl-30b (builder default, 09-02, when only
# 30B was served). At test time that step belongs to the arm's own model, so it is rebuilt
# with 7B into a new file; the offline stores (G-Memory's graph, MEMENTO's nodes) stay as
# archived, as our operators do.
#
# Both arms keep their memory on the CPU (memory_device: cpu), so a worker is ~0.72 G like
# ReAct's. They start only when the ReAct and ours cells are gone and GPU 0 is back under
# 55 G (endpoint 30 G + the retrieval cell's 21 G), waited on BY PID.
#
#   bash scripts/drivers/partnr_h2h7b_night_0913.sh <react runner> <ours runner> <h2h driver>
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
REACT_PID=${1:?react runner pid}; OURS_PID=${2:?ours runner pid}; H2H_PID=${3:?h2h driver pid}
PY=/root/venvs/partnr/bin/python
OUT=outputs/headtohead_0913/val_mini
POOL=val_mini
MODEL=qwen2.5-vl-7b
URL=http://127.0.0.1:8061/v1
EXTRACT7=results/partnr_memento_extractions_val_mini_7b.json
export VLLM_BASE_URL=$URL MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "$URL/models" | grep -q "^200$"; }

# ---- stage 0: MEMENTO extractions with the arm's own model
if [ ! -s "$EXTRACT7" ]; then
  probe || { say "REFUSING stage 0: endpoint down"; exit 4; }
  say "stage 0: MEMENTO extractions with $MODEL"
  "$PY" scripts/partnr_build_baseline_memories.py --what extractions --eval-split $POOL \
      --base-url "$URL" --model $MODEL --workers 8 --extractions-out "$EXTRACT7" \
      > outputs/headtohead_0913/memento_extractions_7b.log 2>&1
  n=$(python3 -c "import json; print(len(json.load(open('$EXTRACT7'))))" 2>/dev/null || echo 0)
  say "stage 0 done: $n instructions (30B table had 368)"
  [ "$n" -ge 360 ] || { say "REFUSING: 7B extraction table incomplete"; exit 5; }
fi

# ---- stage 1: wait for the ReAct and ours cells, then the memory
while kill -0 "$REACT_PID" 2>/dev/null || kill -0 "$OURS_PID" 2>/dev/null; do sleep 60; done
say "react and ours runners gone"
until [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 55000 ]; do sleep 60; done
probe || { say "REFUSING stage 1: endpoint down"; exit 4; }

both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

cell () {  # name config gpu procs [overrides...]   -- same guards as partnr_h2h7b_0913.sh
  local name=$1 config=$2 gpu=$3 procs=$4; shift 4
  local out=$OUT/$name stats=$OUT/$name/results/$POOL.json.gz/stats
  mkdir -p "$out"
  [ -f "$out/DONE" ] && { say "skip $name"; return 0; }
  local started; started=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -m habitat_llm.examples.planner_demo \
      --config-name "$config" \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$POOL.json.gz" \
      num_proc="$procs" evaluation.save_video=False +resume=True hydra.run.dir="$out" \
      $(both llm.generation_params.model $MODEL) "$@" >> "$out/run.log" 2>&1 &
  local runner=$!
  echo "$runner" > "$out/PID"
  say "$name pid=$runner config=$config gpu=$gpu procs=$procs"
  local last=-1 changed=$started misses=0 why=""
  while kill -0 "$runner" 2>/dev/null; do
    sleep 30
    local now count; now=$(date +%s); count=$(ls "$stats" 2>/dev/null | wc -l)
    [ "$count" -ne "$last" ] && { last=$count; changed=$now; }
    if probe; then misses=0
    else
      misses=$((misses + 1)); say "endpoint miss $misses/2 for $name"
      [ "$misses" -ge 2 ] && { why=endpoint_dead; kill -9 "$runner"; break; }
    fi
    [ $((now - changed)) -ge 1800 ] && { why=stall; kill -9 "$runner"; break; }
    [ $((now - started)) -ge 28800 ] && { why=timeout; kill -9 "$runner"; break; }
  done
  wait "$runner" 2>/dev/null; local status=$?
  local count fake
  count=$(ls "$stats" 2>/dev/null | wc -l)
  fake=$("$PY" - "$stats" <<'PYEOF'
import glob, json, sys
n = 0
for f in glob.glob(sys.argv[1] + "/*.json"):
    st = json.load(open(f)).get("stats")
    st = json.loads(st) if isinstance(st, str) else st
    n += bool(st and st.get("sim_step_count") == 0)
print(n)
PYEOF
)
  printf '{"cell":"%s","config":"%s","model":"%s","status":%d,"episodes":%d,"fake_episodes":%d,"killed":"%s","seconds":%d}\n' \
      "$name" "$config" "$MODEL" "$status" "$count" "$fake" "$why" "$(( $(date +%s) - started ))" > "$out/CELL.json"
  [ "$status" -eq 0 ] && [ -z "$why" ] && touch "$out/DONE"
  say "end $name: $(cat "$out/CELL.json")"
}

cell gmemory_7b baselines/react_gmemory_vllm.yaml 0 24 &
cell memento_7b baselines/react_memento_vllm.yaml 0 24 $(both memory_extractions $EXTRACT7) &
wait

# ---- stage 2: one report over all five arms, after the retrieval cell is done too
while kill -0 "$H2H_PID" 2>/dev/null; do sleep 60; done
"$PY" scripts/partnr_v2_report.py --sweep "$OUT" --dataset "$POOL.json.gz" --split "$POOL" \
    --baseline react_7b --out "$OUT/report_all.json" > "$OUT/report_all.txt" 2>&1
say "report over all five arms -> $OUT/report_all.txt"
say "ALL DONE"
