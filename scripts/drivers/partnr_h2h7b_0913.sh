#!/usr/bin/env bash
# Head-to-head at 7B on val_mini, on today's harness: ReAct, ReAct + trajectory retrieval
# (rearrange-only train_mini, the same experience the operators are induced from), and ours
# with the 22-operator library.
#
# The 09-05 table has these arms, but at 7B only react and ours, react_7b with 52 crashed
# episodes and a 16k context, and on code from before the step-0 and world_graph fixes.
# So all three are re-run together here: same endpoint (now 32k context, so a long few-shot
# prompt is not cut short on the baselines' side), same code, fresh directories.
#
# Layout: GPU 1 serves the 7B (0.30) and renders ours; GPU 0 renders the two ReAct arms.
# 3 x 60 workers is what the 180 cores hold. Every cell is guarded the same way as
# partnr_model_cell.sh: endpoint probed for the whole run, stall, hard timeout, and the
# dead-endpoint signature (sim_step_count == 0) counted at the end.
#
#   bash scripts/drivers/partnr_h2h7b_0913.sh
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PY=/root/venvs/partnr/bin/python
OUT=outputs/headtohead_0913/val_mini
POOL=val_mini
MODEL=qwen2.5-vl-7b
URL=http://127.0.0.1:8061/v1
export VLLM_BASE_URL=$URL MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
mkdir -p "$OUT"

code=$(curl -s -m 10 -o /dev/null -w "%{http_code}" "$URL/models")
[ "$code" = "200" ] || { say "REFUSING: $URL returned $code"; exit 4; }

both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

cell () {  # name config gpu procs [overrides...]
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
    if curl -s -m 10 -o /dev/null -w "%{http_code}" "$URL/models" | grep -q "^200$"; then misses=0
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

# GPU 1 was taken by another user's habitat eval at 22:12, so everything is on GPU 0:
# the endpoint (16k context, 0.30 = 30 G, the archived settings -- react_7b's 09-05 crashes
# were an assert in evaluation_runner.py:517, not context length) and 3 x 24 workers.
# 3 x 36 reached 97 of 98 G within eight minutes and workers died of CUDA OOM (22:29);
# that run was killed and its directory removed. 3 x 24 OOMed again (22:41): every retrieval
# worker loads all-mpnet-base-v2 onto the GPU (rag.py:52), 1.75 G against 0.72 G for the
# other two arms. The embedding stays on the GPU as archived; the retrieval arm gets fewer
# workers instead: 30 + 17 + 17.5 + 12 x 1.75 = ~86 G of 95.
cell react_7b       baselines/react_vllm.yaml          0 24 &
cell react_rag_R_7b baselines/react_rag_R_vllm.yaml    0 12 &
cell v2_accepted_7b baselines/skill_memory_v2_vllm.yaml 0 24 $(both operators results/partnr_operators_iir1.json) &
wait

# Report on what every arm scored, per type -- the same reporter as 09-05, written to disk first.
"$PY" scripts/partnr_v2_report.py --sweep "$OUT" --dataset "$POOL.json.gz" --split "$POOL" \
    --baseline react_7b --out "$OUT/report.json" > "$OUT/report.txt" 2>&1
say "report -> $OUT/report.txt"
say "ALL DONE"
