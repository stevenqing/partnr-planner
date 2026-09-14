#!/usr/bin/env bash
# The typed requirement interface on val_mini, once, to pair with the 7B head-to-head.
#
# Tuned on train_mini only (offline recall and a 120-episode sim pairing: percent_complete
# 0.577 vs 0.210, state_success 0.383 vs 0.067 against the free-form intent arm). This cell is
# the val_mini report: the whole split, same model, library (iir1), runner and agents as the
# arms in outputs/headtohead_0913/val_mini_fixed/, so it pairs per episode with all of them.
# Do not tune on its result.
#
#   setsid nohup bash scripts/drivers/partnr_typed_val_mini.sh > outputs/cand_iface_0914/typed_val_mini.log 2>&1 &
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PY=/root/venvs/partnr/bin/python
# EXAMPLES picks which kinds of task the prompt's examples show: R (single-stage rearrangement
# only) or RS (plus "next to", no ordering) keep the compositional protocol -- everything the
# interface saw is elementary; RST is the train-tuned set whose third example is two-stage.
EXAMPLES=${EXAMPLES:?set EXAMPLES=R or RS or RST}
case "$EXAMPLES" in R|RS|RST) ;; *) echo "EXAMPLES must be R, RS or RST"; exit 2 ;; esac
ARM=typed_v7b_${EXAMPLES}_7b
OUT=outputs/cand_iface_0914/val_mini/$ARM
POOL=val_mini
MODEL=qwen2.5-vl-7b
OPS=results/partnr_operators_iir1.json
PRIOR=results/partnr_inside_prior_train_R_only.json   # R-only train; same decisions as the all-type prior
GPU=${GPU:-1}
PORT=${PORT:-8063}
PROCS=${PROCS:-24}
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "$1/models" | grep -q "^200$"; }

[ -e "$OUT" ] && { say "REFUSING: $OUT exists, not running on residue"; exit 3; }
grep -q "return sorted(in_named)\[0\]" our_method/skill_memory_v2/partnr_typed_goals.py \
  && grep -q "collapsed: another place of the same kind in the same room" our_method/skill_memory_v2/partnr_typed_goals.py \
  && grep -q "_requirements_from_typed" our_method/skill_memory_v2/partnr_planner.py \
  || { say "REFUSING: v7b typed interface not in the code"; exit 6; }
[ "$("$PY" -c "import json; print(len(json.load(open('results/partnr_object_kinds_train.json'))['kinds']))")" = 106 ] \
  || { say "REFUSING: object kinds file is not the 106-kind union"; exit 6; }
[ -s "$PRIOR" ] || { say "REFUSING: $PRIOR missing"; exit 6; }
grep -q "EXAMPLE_SETS = {" our_method/skill_memory_v2/partnr_typed_goals.py \
  && grep -q "typed_examples" our_method/skill_memory_v2/partnr_planner.py \
  || { say "REFUSING: typed_examples switch not in the code"; exit 6; }
probe "http://127.0.0.1:$PORT/v1" || { say "REFUSING: :$PORT not up"; exit 4; }

WANT=$("$PY" -c "import gzip, json; print(len(json.load(gzip.open('data/datasets/partnr_episodes/v0_0/$POOL.json.gz'))['episodes']))")
both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

url=http://127.0.0.1:$PORT/v1
stats=$OUT/results/$POOL.json.gz/stats
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/COMMIT"
started=$(date +%s)
VLLM_BASE_URL=$url CUDA_VISIBLE_DEVICES=$GPU "$PY" -m habitat_llm.examples.planner_demo \
    --config-name baselines/skill_memory_v2_typed_vllm.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$POOL.json.gz" \
    num_proc="$PROCS" evaluation.save_video=False +resume=True hydra.run.dir="$PWD/$OUT" \
    $(both llm.generation_params.model $MODEL) $(both operators $OPS) \
    $(both inside_prior $PRIOR) $(both typed_examples $EXAMPLES) >> "$OUT/run.log" 2>&1 &
runner=$!
echo "$runner" > "$OUT/PID"
say "$ARM pid=$runner gpu=$GPU url=$url procs=$PROCS episodes=$WANT commit=$(cat "$OUT/COMMIT")"
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
  # The model is called only at episode start, so running=0 is normal; judge by stats.
  [ $((now - changed)) -ge 1800 ] && { why=stall; kill -9 "$runner"; break; }
  [ $((now - started)) -ge 28800 ] && { why=timeout; kill -9 "$runner"; break; }
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
printf '{"cell":"%s","config":"baselines/skill_memory_v2_typed_vllm.yaml","url":"%s","status":%d,"episodes":%d,"wanted":%d,"fake_episodes":%s,"killed":"%s","seconds":%d,"commit":"%s"}\n' \
    "$ARM" "$url" "$status" "$count" "$WANT" "${fake:-null}" "$why" "$(( $(date +%s) - started ))" "$(cat "$OUT/COMMIT")" > "$OUT/CELL.json"
[ "$status" -eq 0 ] && [ -z "$why" ] && [ "$count" -eq "$WANT" ] && touch "$OUT/DONE"
say "end $ARM: $(cat "$OUT/CELL.json")"
