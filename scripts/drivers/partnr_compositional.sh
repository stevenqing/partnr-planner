#!/usr/bin/env bash
# The comparison the head-to-head could not make: four memories, one training pile.
#
# outputs/headtohead already holds `react` (no memory) and `v2_intent` (ours), and its
# react_rag arm retrieves over every task type -- so it has seen the spatial and temporal
# demonstrations our operators never did. That arm answers "is retrieval enough". It does
# not answer the paper's question, which is whether this memory composes, because the
# arms were not given the same experience.
#
# These three fix that. Every one is built from the same 161 rearrange-only train_mini
# rollouts skill memory v2 induces its operators from, and every one is rendered into the
# same `{rag_examples}` slot of the same ReAct planner, so they differ from each other in
# the representation of the memory and in nothing else:
#
#   react_rag_R  PARTNR's own trajectory retrieval, pool restricted to those episodes
#   gmemory      G-Memory's trajectory graph plus its distilled insights
#   memento      the MEMENTO-style type-separated knowledge graph
#
# What they do NOT share with `v2_intent` is the control structure: these three replan
# every step inside ReAct, ours composes a chain once. That difference is the method and
# cannot be removed, which is why the report reads the compositional axis as each arm's
# degradation from its own R score rather than as absolute scores.
#
# Results land beside the existing arms so one report covers all six.

set -u

ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
SPLIT=${SPLIT:-val_mini}
PROCS=${PROCS:-12}
# Which arms this invocation runs. The three are independent and share nothing but the
# served model, so one process per arm on its own simulation GPU finishes in a third of
# the wall clock. `report` alone re-reads whatever is on disk.
ARMS=${ARMS:-react_rag_R gmemory memento report}
SWEEP=${SWEEP:-outputs/headtohead}
MODEL=${MODEL:-qwen3-vl-30b}
BASE_URL=${BASE_URL:-http://127.0.0.1:8062/v1}
CELL_TIMEOUT=${CELL_TIMEOUT:-28800}

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export VLLM_BASE_URL="$BASE_URL"

cd "$ROOT" || exit 1
mkdir -p "$SWEEP"
PROGRESS="$SWEEP/compositional-$(echo $ARMS | tr ' ' '-').log"
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$PROGRESS"; }

if ! curl -sf "$BASE_URL/models" > /dev/null; then
  say "ABORT: no model endpoint at $BASE_URL"
  exit 1
fi

for store in results/rag_source_train_mini_R/episode_result_log.csv \
             results/partnr_gmemory_train_mini_R.json \
             results/partnr_memento_train_mini_R.json \
             results/partnr_memento_extractions_val_mini.json; do
  if [ ! -e "$store" ]; then
    say "ABORT: missing $store -- run scripts/partnr_build_baseline_memories.py first"
    exit 1
  fi
done

run () {
  local name=$1 config=$2; shift 2
  # Every cell gets its own hydra.run.dir: paths.results_dir hangs off it, and a shared
  # one silently overwrites another cell's episodes.
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

declare -A CONFIG=(
  [react_rag_R]=baselines/react_rag_R_vllm.yaml
  [gmemory]=baselines/react_gmemory_vllm.yaml
  [memento]=baselines/react_memento_vllm.yaml
)

say "arms='$ARMS' procs=$PROCS sim gpu=$CUDA_VISIBLE_DEVICES"
for arm in $ARMS; do
  [ "$arm" = "report" ] && continue
  run "$arm" "${CONFIG[$arm]}" $(model)
done

case " $ARMS " in *" report "*) ;; *) say "arms finished (no report asked for)"; exit 0;; esac

say "report"
"$PY" scripts/partnr_v2_report.py \
    --sweep "$SWEEP/$SPLIT" --dataset "$SPLIT.json.gz" --split "$SPLIT" \
    --baseline react --out "$SWEEP/$SPLIT/report.json" > "$SWEEP/$SPLIT/report.txt" 2>&1
"$PY" scripts/partnr_compositional_report.py \
    --report "$SWEEP/$SPLIT/report.json" \
    > "$SWEEP/$SPLIT/compositional.txt" 2>&1
say "compositional comparison finished"
