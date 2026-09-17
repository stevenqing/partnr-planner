#!/usr/bin/env bash
# One cell of the arm where the model reads the instruction, kept in its own file because
# the privileged-arm driver is usually running when this one is edited, and bash re-reads a
# script it is executing by byte offset -- editing one mid-run has already broken a cell here.
#
#   MODEL=Qwen3-8B URL=http://127.0.0.1:8101/v1 NO_THINK=1 \
#     OPERATORS=results/partnr_operators_iir1.json \
#     bash scripts/drivers/partnr_model_cell.sh <pool> <cell> <outdir>
#
# NO_THINK is not a tuning knob. A reasoning model answers this planner's prompt with a
# `<think>` block and the planner stops at a blank line, so the requirement list comes back
# empty and the arm scores zero for a harness reason. Checked against both endpoints before
# any cell was run: 7B parses 2 predicates, 8B parses 0 with thinking on and 3 with it off.
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
PROCS=${PROCS:-45}
OPERATORS=${OPERATORS:-results/partnr_operators.json}
# CONFIG picks the planner config; the default is the free-form arm this driver was written for,
# and the typed arm passes baselines/skill_memory_v2_typed_vllm.yaml.
CONFIG=${CONFIG:-baselines/skill_memory_v2_vllm.yaml}
HARD_TIMEOUT=${HARD_TIMEOUT:-14400}
STALL_SECONDS=${STALL_SECONDS:-1800}
MODEL=${MODEL:?set MODEL}
URL=${URL:?set URL}
export VLLM_BASE_URL="$URL"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false

# Preflight. The endpoint has to answer before 45 processes are started against it: the
# Qwen3-8B server died mid-afternoon and two cells ran to completion against nothing,
# 418 `APIConnectionError`s answered with the empty string, 183 episodes ending at step 0
# with a score of zero that looked exactly like a model that cannot plan.
if ! curl -s -m 10 -o /dev/null -w "%{http_code}" "$URL/models" | grep -q "^200$"; then
  echo "[$(date +%H:%M:%S)] REFUSING: $URL does not answer; start the server first"
  exit 4
fi

pool=$1; cell=$2; outbase=$3; shift 3
out="$outbase/$cell"
cd "$ROOT" || exit 1
mkdir -p "$out"
stats="$out/results/$pool.json.gz/stats"
[ -f "$out/DONE" ] && { echo "skip $cell"; exit 0; }

both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }
think=""
if [ "${NO_THINK:-0}" = "1" ]; then
  think="+evaluation.agents.agent_0.planner.plan_config.llm.extra_body.chat_template_kwargs.enable_thinking=False"
  think="$think +evaluation.agents.agent_1.planner.plan_config.llm.extra_body.chat_template_kwargs.enable_thinking=False"
fi

started=$(date +%s); endpoint_dead=0
"$PY" -m habitat_llm.examples.planner_demo \
    --config-name "$CONFIG" \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$pool.json.gz" \
    num_proc="$PROCS" evaluation.save_video=False hydra.run.dir="$out" \
    $(both operators "$OPERATORS") $(both llm.generation_params.model "$MODEL") $think \
    "$@" >> "$out/run.log" 2>&1 &
runner=$!
echo "$runner" > "$out/PID"
echo "[$(date +%H:%M:%S)] $cell model=$MODEL url=$URL no_think=${NO_THINK:-0} pid=$runner operators=$OPERATORS config=$CONFIG"

# The preflight above only proves the endpoint was alive at launch. On 09-07 the 8101 server
# was shut down at 22:08 with the cell four hours in, and the cell kept going for another nine
# minutes writing episodes that ended with `sim_step_count == 0` and `runtime ~13s` -- a whole
# column of zeros that reads exactly like a model that cannot plan. So the endpoint is probed
# for the whole run, not just before it. Two consecutive misses (one minute) is the trigger;
# a single miss during a long generation is not evidence of anything.
last=-1; changed=$started; misses=0
while kill -0 "$runner" 2>/dev/null; do
  sleep 30
  now=$(date +%s); count=$(ls "$stats" 2>/dev/null | wc -l)
  [ "$count" -ne "$last" ] && { last=$count; changed=$now; }
  if curl -s -m 10 -o /dev/null -w "%{http_code}" "$URL/models" | grep -q "^200$"; then
    misses=0
  else
    misses=$((misses + 1))
    echo "[$(date +%H:%M:%S)] endpoint miss $misses/2 for $cell ($URL)"
    if [ "$misses" -ge 2 ]; then
      echo "[$(date +%H:%M:%S)] ENDPOINT_DEAD $cell at $count -- killing $runner"
      kill -9 "$runner" 2>/dev/null; endpoint_dead=1; break
    fi
  fi
  if [ $((now - changed)) -ge "$STALL_SECONDS" ]; then
    echo "[$(date +%H:%M:%S)] STALL $cell at $count -- killing $runner"; kill -9 "$runner" 2>/dev/null; break
  fi
  if [ $((now - started)) -ge "$HARD_TIMEOUT" ]; then
    echo "[$(date +%H:%M:%S)] TIMEOUT $cell at $count -- killing $runner"; kill -9 "$runner" 2>/dev/null; break
  fi
done
wait "$runner" 2>/dev/null; status=$?
count=$(ls "$stats" 2>/dev/null | wc -l)
# An episode that never stepped the simulator is the dead-endpoint signature, and counting it
# here is more reliable than counting `APIConnectionError` in run.log (46 fake episodes on
# 09-07 produced only 28 logged errors).
fake=$("$PY" - "$stats" <<'PYEOF'
import glob, json, sys
n = 0
for f in glob.glob(sys.argv[1] + "/*.json"):
    try:
        st = json.load(open(f)).get("stats")
        if isinstance(st, str):
            st = json.loads(st)
        if st and st.get("sim_step_count") == 0:
            n += 1
    except Exception:
        pass
print(n)
PYEOF
)
printf '{"cell":"%s","pool":"%s","operators":"%s","model":"%s","no_think":"%s","status":%d,"episodes":%d,"fake_episodes":%d,"endpoint_dead":%d,"seconds":%d}\n' \
    "$cell" "$pool" "$OPERATORS" "$MODEL" "${NO_THINK:-0}" "$status" "$count" "$fake" "${endpoint_dead:-0}" "$(( $(date +%s) - started ))" > "$out/CELL.json"
[ "$status" -eq 0 ] && [ "${endpoint_dead:-0}" -eq 0 ] && touch "$out/DONE"
cat "$out/CELL.json"
