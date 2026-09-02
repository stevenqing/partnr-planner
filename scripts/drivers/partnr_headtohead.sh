#!/usr/bin/env bash
# The comparison the ablation sweep cannot make: three arms that all read the instruction.
#
# Every cell in partnr_v2_sweep.sh is handed the episode's own propositions, so that sweep
# says how much of the privileged ceiling our composer recovers and nothing about whether
# the method beats anything. These three arms differ only in what the memory is asked for,
# with the same model, the same split, the same partial observation and the same
# decentralization:
#
#   react        PARTNR's own decentralized ReAct. No memory. The floor.
#   react_rag    the same, with PARTNR's own trajectory retrieval over successful
#                rollouts. Prompt-shaped memory -- the thing skill memory v2 argues with.
#   v2_intent    ours: the model states what must become true, and the memory picks the
#                body, the container, the order and the agent.
#
# It waits for the ablation sweep to finish before starting, because both want the same
# cores and the served model is shared. Start it now and leave it queued.

set -u

ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
SPLIT=${SPLIT:-val_mini}
PROCS=${PROCS:-12}
SWEEP=${SWEEP:-outputs/headtohead}
MODEL=${MODEL:-qwen3-vl-30b}
BASE_URL=${BASE_URL:-http://127.0.0.1:8062/v1}
CELL_TIMEOUT=${CELL_TIMEOUT:-28800}
WAIT_FOR=${WAIT_FOR:-outputs/sweep/progress.log}

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export VLLM_BASE_URL="$BASE_URL"

cd "$ROOT" || exit 1
mkdir -p "$SWEEP"
PROGRESS="$SWEEP/progress.log"
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$PROGRESS"; }

if [ "${WAIT_FOR}" != "none" ]; then
  say "waiting for the ablation sweep to finish ($WAIT_FOR)"
  while ! grep -q "sweep finished" "$WAIT_FOR" 2>/dev/null; do sleep 120; done
  say "ablation sweep finished; starting"
fi

# The endpoint has to be up before an eight-hour cell is launched against it.
if ! curl -sf "$BASE_URL/models" > /dev/null; then
  say "ABORT: no model endpoint at $BASE_URL"
  exit 1
fi

run () {
  local name=$1 config=$2; shift 2
  local out="$SWEEP/$SPLIT/$name"
  local data="data/datasets/partnr_episodes/v0_0/$SPLIT.json.gz"
  if [ -f "$out/DONE" ]; then say "skip   $name"; return 0; fi
  mkdir -p "$out"
  local attempt
  for attempt in 1 2; do
    say "start  $name (attempt $attempt)"
    timeout "$CELL_TIMEOUT" "$PY" -m habitat_llm.examples.planner_demo \
        --config-name "$config" \
        habitat.dataset.data_path="$data" \
        num_proc="$PROCS" \
        evaluation.save_video=False \
        +resume=True \
        hydra.run.dir="$out" \
        "$@" >> "$out/run.log" 2>&1
    local status=$?
    local n; n=$(ls "$out/results/$SPLIT.json.gz/stats" 2>/dev/null | wc -l)
    if [ "$status" -eq 0 ]; then touch "$out/DONE"; say "done   $name  ($n episodes)"; return 0; fi
    say "FAILED $name status=$status ($n episodes) -- retrying"
  done
  say "GIVEUP $name"
  return 1
}

model () { echo "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.model=$MODEL evaluation.agents.agent_1.planner.plan_config.llm.generation_params.model=$MODEL"; }

run react     baselines/react_vllm.yaml           $(model)
run react_rag baselines/react_rag_vllm.yaml       $(model)
run v2_intent baselines/skill_memory_v2_vllm.yaml $(model)

say "report"
"$PY" scripts/partnr_v2_report.py \
    --sweep "$SWEEP/$SPLIT" --dataset "$SPLIT.json.gz" --split "$SPLIT" \
    --baseline react --out "$SWEEP/$SPLIT/report.json" > "$SWEEP/$SPLIT/report.txt" 2>&1
say "head-to-head finished"
