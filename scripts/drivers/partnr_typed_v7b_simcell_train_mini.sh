#!/usr/bin/env bash
# One more typed cell on the simpair pool, with the v7b projection.
#
# The first typed cell (v5 projection) emptied 15 of its 120 episodes -- the projection dropped
# every line, the planner took no step. v7b (kind-aware place collapse, snap to the first
# same-kind furniture in the named room) empties 3/395 offline. Same 120 train_mini episodes
# (read from the first cell's episodes.json), same endpoint and GPU the first typed cell used,
# so it pairs with both typed_7b (what the projection fix is worth in the simulator) and
# intent_7b (what the interface is worth).
#
#   setsid nohup bash scripts/drivers/partnr_typed_v7b_simcell_train_mini.sh > /tmp/v7b.log 2>&1 &
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PY=/root/venvs/partnr/bin/python
PAIR=outputs/cand_iface_0914/simpair_train_mini
ARM=typed_v7b_7b
OUT=$PAIR/$ARM
POOL=train_mini
MODEL=qwen2.5-vl-7b
OPS=results/partnr_operators_iir1.json
GPU=1
PORT=8063
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "$1/models" | grep -q "^200$"; }

[ -e "$OUT" ] && { say "REFUSING: $OUT exists"; exit 3; }
[ -s "$PAIR/episodes.json" ] || { say "REFUSING: $PAIR/episodes.json missing"; exit 5; }
[ -e "$PAIR/typed_7b/CELL.json" ] || { say "REFUSING: first typed cell not finished, :$PORT may be busy"; exit 7; }
grep -q "return sorted(in_named)\[0\]" our_method/skill_memory_v2/partnr_typed_goals.py \
  && grep -q "collapsed: another place of the same kind in the same room" our_method/skill_memory_v2/partnr_typed_goals.py \
  || { say "REFUSING: v7b projection not in partnr_typed_goals.py"; exit 6; }
probe "http://127.0.0.1:$PORT/v1" || { say "REFUSING: :$PORT not up"; exit 4; }

IDS=$("$PY" -c "import json; print(','.join(json.load(open('$PAIR/episodes.json'))['ids']))")
WANT=$(echo "$IDS" | tr ',' '\n' | grep -c .)
both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

url=http://127.0.0.1:$PORT/v1
stats=$OUT/results/$POOL.json.gz/stats
mkdir -p "$OUT"
started=$(date +%s)
VLLM_BASE_URL=$url CUDA_VISIBLE_DEVICES=$GPU "$PY" -m habitat_llm.examples.planner_demo \
    --config-name baselines/skill_memory_v2_typed_vllm.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$POOL.json.gz" \
    num_proc=24 evaluation.save_video=False +resume=True hydra.run.dir="$PWD/$OUT" \
    "+episode_id_filter=[$IDS]" \
    $(both llm.generation_params.model $MODEL) $(both operators $OPS) >> "$OUT/run.log" 2>&1 &
runner=$!
echo "$runner" > "$OUT/PID"
say "$ARM pid=$runner gpu=$GPU url=$url episodes=$WANT"
last=-1; changed=$started; misses=0; why=""
while kill -0 "$runner" 2>/dev/null; do
  sleep 30
  now=$(date +%s); count=$(ls "$stats" 2>/dev/null | wc -l)
  [ "$count" -ne "$last" ] && { last=$count; changed=$now; }
  if probe "$url"; then misses=0
  else
    misses=$((misses + 1)); say "endpoint miss $misses/2"
    [ "$misses" -ge 2 ] && { why=endpoint_dead; kill -9 "$runner"; break; }
  fi
  [ $((now - changed)) -ge 1800 ] && { why=stall; kill -9 "$runner"; break; }
  [ $((now - started)) -ge 14400 ] && { why=timeout; kill -9 "$runner"; break; }
done
wait "$runner" 2>/dev/null; status=$?
count=$(ls "$stats" 2>/dev/null | wc -l)
fake=$("$PY" - "$stats" <<'PY'
import glob, json, sys
n = 0
for f in glob.glob(sys.argv[1] + "/*.json"):
    s = json.load(open(f)).get("stats")
    s = json.loads(s) if isinstance(s, str) else (s or {})
    n += int(s.get("sim_step_count", 1) == 0 and (s.get("runtime") or 0) < 60)
print(n)
PY
)
printf '{"cell":"%s","config":"baselines/skill_memory_v2_typed_vllm.yaml","url":"%s","status":%d,"episodes":%d,"wanted":%d,"fake_episodes":%s,"killed":"%s","seconds":%d}\n' \
    "$ARM" "$url" "$status" "$count" "$WANT" "${fake:-null}" "$why" "$(( $(date +%s) - started ))" > "$OUT/CELL.json"
[ "$status" -eq 0 ] && [ -z "$why" ] && [ "$count" -eq "$WANT" ] && touch "$OUT/DONE"
say "end $ARM: $(cat "$OUT/CELL.json")"
