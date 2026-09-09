#!/usr/bin/env bash
# One PARTNR cell on one pool, with the two guards this box has needed.
#
# Hard timeout AND a stall guard: a hung habitat process keeps burning CPU, so "is it
# busy" says nothing. The judge is whether the results directory grew.  The runner is
# waited on BY PID and killed by PID -- `pgrep -f` matches this script's own command line
# and has killed the wrong thing six times here.
#
#   bash scripts/drivers/partnr_gate_cell.sh <pool> <cell-name> <outdir> [overrides...]
#
# Env: OPERATORS (library json), CONFIG, PROCS, HARD_TIMEOUT, STALL_SECONDS, GPU.
#
# Two arms run through here and they answer different questions. The default config reads
# the episode's own propositions and calls no model: deterministic, an upper bound, and the
# right place for an acceptance gate, which needs to attribute a change to the operator and
# nothing else. Setting LLM_MODEL switches to the arm where the model reads the instruction
# and says what must become true -- that is the arm a reported number has to come from, and
# it is where an added operator also changes the MENU the model is shown, since the menu is
# built from the effects the memory can bring about.
#
#   LLM_MODEL=qwen2.5-vl-7b VLLM_BASE_URL=http://127.0.0.1:8061/v1 \
#     CONFIG=baselines/skill_memory_v2_vllm.yaml bash ... <pool> <cell> <outdir>
#
# One generation job per endpoint at a time.
set -u

ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
PROCS=${PROCS:-40}
CONFIG=${CONFIG:-baselines/skill_memory_v2_oracle_goals.yaml}
OPERATORS=${OPERATORS:-results/partnr_operators.json}
HARD_TIMEOUT=${HARD_TIMEOUT:-3600}
STALL_SECONDS=${STALL_SECONDS:-1200}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false

pool=$1; cell=$2; outbase=$3; shift 3
out="$outbase/$cell"
cd "$ROOT" || exit 1
mkdir -p "$out"
stats="$out/results/$pool.json.gz/stats"

if [ -f "$out/DONE" ]; then echo "skip $cell (DONE)"; exit 0; fi

both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

llm_overrides=""
if [ -n "${LLM_MODEL:-}" ]; then
  export VLLM_BASE_URL=${VLLM_BASE_URL:?LLM_MODEL needs VLLM_BASE_URL}
  llm_overrides=$(both llm.generation_params.model "$LLM_MODEL")
  echo "[$(date +%H:%M:%S)] model arm: $LLM_MODEL at $VLLM_BASE_URL"
fi

started=$(date +%s)
# shellcheck disable=SC2046
"$PY" -m habitat_llm.examples.planner_demo \
    --config-name "$CONFIG" \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$pool.json.gz" \
    num_proc="$PROCS" \
    evaluation.save_video=False \
    hydra.run.dir="$out" \
    $(both operators "$OPERATORS") \
    $llm_overrides \
    "$@" >> "$out/run.log" 2>&1 &
runner=$!
echo "$runner" > "$out/PID"
echo "[$(date +%H:%M:%S)] $cell pid=$runner pool=$pool operators=$OPERATORS"

last_count=-1; last_change=$started
while kill -0 "$runner" 2>/dev/null; do
  sleep 30
  now=$(date +%s)
  count=$(ls "$stats" 2>/dev/null | wc -l)
  if [ "$count" -ne "$last_count" ]; then last_count=$count; last_change=$now; fi
  if [ $((now - last_change)) -ge "$STALL_SECONDS" ]; then
    echo "[$(date +%H:%M:%S)] STALL $cell: $count episodes for ${STALL_SECONDS}s -- killing $runner"
    kill -9 "$runner" 2>/dev/null; echo stalled > "$out/VERDICT"; exit 2
  fi
  if [ $((now - started)) -ge "$HARD_TIMEOUT" ]; then
    echo "[$(date +%H:%M:%S)] TIMEOUT $cell after ${HARD_TIMEOUT}s ($count episodes) -- killing $runner"
    kill -9 "$runner" 2>/dev/null; echo timeout > "$out/VERDICT"; exit 3
  fi
done
wait "$runner"; status=$?
elapsed=$(( $(date +%s) - started ))
count=$(ls "$stats" 2>/dev/null | wc -l)
printf '{"cell":"%s","pool":"%s","operators":"%s","config":"%s","model":"%s","status":%d,"episodes":%d,"seconds":%d}\n' \
    "$cell" "$pool" "$OPERATORS" "$CONFIG" "${LLM_MODEL:-none}" "$status" "$count" "$elapsed" > "$out/CELL.json"
if [ "$status" -eq 0 ]; then touch "$out/DONE"; fi
cat "$out/CELL.json"
