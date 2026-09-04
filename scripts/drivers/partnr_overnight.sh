#!/usr/bin/env bash
# What to run after the compositional comparison lands, without anyone watching.
#
# Queued in the order of what the paper is missing, and laid out one queue per simulation
# GPU so three cells run at once. Everything waits for the three arms already running to
# write their DONE files first -- they own the GPUs until then.
#
#   v2_prompt        our own operators rendered as prompt text, consumed by the same ReAct
#                    planner as the baselines. THE MISSING CELL: without it a win for
#                    skill memory v2 confounds "operators are a better representation"
#                    with "executing a memory beats reading one". This separates them.
#   react_rag_full   PARTNR's own retrieval over every task type, with the slot actually
#                    wired this time. Restores the claim withdrawn on 09-02 and, read
#                    against react_rag_R, prices the task types the pool was allowed.
#   react_replicate  react run a second time, unchanged. The null band currently rests on
#                    one accident (react vs the retrieval arm that never retrieved); this
#                    makes it two points, deliberately. Decoding is greedy, so whatever
#                    separates them is the simulator.
#   v2_retry         the privileged composer with its repair budget raised. Tests whether
#                    the composer's -0.10 against ReAct is a structural cost of open-loop
#                    execution or an artifact of retry_limit=2 / repair_limit=3. Zero LLM.
#   *_7b             the head-to-head on Qwen2.5-VL-7B, asked for on 09-02 and never run.
#                    Last, because the prior is weak: 7B scored 21.75% on VIKI-L2 and the
#                    bottleneck here is exactly the step it is worst at. It is a point on
#                    the delegation curve, not a contender. Needs its own endpoint, which
#                    this script starts on the one free card.
#
# Run:  setsid nohup bash scripts/drivers/partnr_overnight.sh > outputs/overnight.log 2>&1 < /dev/null &
# Check: tail -f outputs/overnight.log   or   outputs/headtohead/overnight.log

set -u

ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
VLLM=${VLLM:-/root/venvs/vllm/bin/python}
SPLIT=${SPLIT:-val_mini}
PROCS=${PROCS:-16}
SWEEP=${SWEEP:-outputs/headtohead}
MODEL=${MODEL:-qwen3-vl-30b}
BASE_URL=${BASE_URL:-http://127.0.0.1:8062/v1}
MODEL_7B=${MODEL_7B:-qwen2.5-vl-7b}
BASE_URL_7B=${BASE_URL_7B:-http://127.0.0.1:8061/v1}
GPU_7B=${GPU_7B:-7}
CELL_TIMEOUT=${CELL_TIMEOUT:-28800}
WAIT_FOR=${WAIT_FOR:-react_rag_R gmemory memento}
SKIP_7B=${SKIP_7B:-0}

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/pfs/devs/pn5wp/shishuqing/hf

cd "$ROOT" || exit 1
PROGRESS="$SWEEP/overnight.log"
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$PROGRESS"; }

# ---------------------------------------------------------------- wait for the GPUs

say "waiting for: $WAIT_FOR"
# `WAIT_FOR=none` starts immediately -- for re-running a cell after the queue has already
# drained, when nothing owns the GPUs.
[ "$WAIT_FOR" = "none" ] && WAIT_FOR=""
while [ -n "$WAIT_FOR" ]; do
  pending=""
  for arm in $WAIT_FOR; do
    [ -f "$SWEEP/$SPLIT/$arm/DONE" ] || pending="$pending $arm"
  done
  [ -z "$pending" ] && break
  sleep 120
done
say "the compositional arms are done; starting the overnight queue"

# ---------------------------------------------------------------- the 7B endpoint

start_7b () {
  if curl -s -m 3 "$BASE_URL_7B/models" > /dev/null 2>&1; then
    say "7B already up"; return 0
  fi
  say "starting 7B on GPU $GPU_7B"
  # One card, not the two a11_serve.sh asks for: the others are running simulation now,
  # and a 7B at fp16 fits in a single 97 GiB card several times over.
  CUDA_VISIBLE_DEVICES=$GPU_7B nohup "$VLLM" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct --served-model-name "$MODEL_7B" \
    --port 8061 --tensor-parallel-size 1 \
    --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":1}' > /root/vllm-$MODEL_7B.log 2>&1 &
  for _ in $(seq 1 60); do
    curl -s -m 3 "$BASE_URL_7B/models" > /dev/null 2>&1 && { say "7B up"; return 0; }
    sleep 20
  done
  say "7B did NOT come up -- its cells will be skipped"
  return 1
}

# ---------------------------------------------------------------- one cell

# VLLM_BASE_URL wins over `plan_config.llm.base_url` in VLLMChat.__init__:
#   base_url = os.getenv("VLLM_BASE_URL", "") or self.llm_conf.base_url
# So a per-cell endpoint has to be set in the environment. Passing it only as a hydra
# override looks right, runs, and silently talks to whatever the exported variable says --
# which is how both 7B cells spent an hour asking the 30B endpoint for a model it does not
# serve and 404ing on every call.
run () {
  local name=$1 config=$2 gpu=$3 out_root=$4 url=$5; shift 5
  # Each cell gets its own hydra.run.dir: paths.results_dir hangs off it and a shared one
  # silently overwrites another cell's episodes.
  local out="$out_root/$SPLIT/$name"
  local data="data/datasets/partnr_episodes/v0_0/$SPLIT.json.gz"
  if [ -f "$out/DONE" ]; then say "skip   $name"; return 0; fi
  mkdir -p "$out"
  local attempt
  for attempt in 1 2; do
    say "start  $name (gpu $gpu, attempt $attempt)"
    CUDA_VISIBLE_DEVICES=$gpu VLLM_BASE_URL="$url" \
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
    if [ "$status" -eq 0 ]; then touch "$out/DONE"; say "done   $name ($n episodes)"; return 0; fi
    say "FAILED $name status=$status ($n episodes) -- retrying"
  done
  say "GIVEUP $name"
  return 1
}

model () { echo "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.model=$MODEL evaluation.agents.agent_1.planner.plan_config.llm.generation_params.model=$MODEL"; }
model_7b () { echo "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.model=$MODEL_7B evaluation.agents.agent_1.planner.plan_config.llm.generation_params.model=$MODEL_7B"; }
both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

start_7b && HAVE_7B=1 || HAVE_7B=0
[ "$SKIP_7B" = "1" ] && HAVE_7B=0

# ---------------------------------------------------------------- three queues

export VLLM_BASE_URL="$BASE_URL"

queue_a () {   # the missing cell first, then a 7B point
  run v2_prompt baselines/react_v2prompt_vllm.yaml 0 "$SWEEP" "$BASE_URL" $(model)
  [ "$HAVE_7B" = "1" ] && run v2_intent_7b baselines/skill_memory_v2_vllm.yaml 0 "$SWEEP" "$BASE_URL_7B" $(model_7b)
}

queue_b () {   # the withdrawn claim, made testable
  run react_rag_full baselines/react_rag_full_vllm.yaml 1 "$SWEEP" "$BASE_URL" $(model)
  [ "$HAVE_7B" = "1" ] && run react_7b baselines/react_vllm.yaml 1 "$SWEEP" "$BASE_URL_7B" $(model_7b)
}

queue_c () {   # the noise line, then the cheap execution-layer test
  run react_replicate baselines/react_vllm.yaml 6 "$SWEEP" "$BASE_URL" $(model)
  # Privileged goals: no model is called at all, so this one is free and fast.
  # `repair_limit` is a code default rather than a config key, hence the `+`.
  run v2_retry baselines/skill_memory_v2_oracle_goals.yaml 6 outputs/sweep "$BASE_URL" \
      $(both operators results/partnr_operators.json) \
      $(both retry_limit 6) \
      +evaluation.agents.agent_0.planner.plan_config.repair_limit=10 \
      +evaluation.agents.agent_1.planner.plan_config.repair_limit=10
}

queue_a & A=$!
queue_b & B=$!
queue_c & C=$!
say "queues launched (a=$A b=$B c=$C); 7B available=$HAVE_7B"
wait $A $B $C
say "all overnight cells finished"

# ---------------------------------------------------------------- report

"$PY" scripts/partnr_v2_report.py \
    --sweep "$SWEEP/$SPLIT" --dataset "$SPLIT.json.gz" --split "$SPLIT" \
    --baseline react --out "$SWEEP/$SPLIT/report.json" > "$SWEEP/$SPLIT/report.txt" 2>&1
"$PY" scripts/partnr_compositional_report.py \
    --sweep "$SWEEP/$SPLIT" --split "$SPLIT" \
    > "$SWEEP/$SPLIT/compositional.txt" 2>&1
say "report written to $SWEEP/$SPLIT/{report.txt,compositional.txt}"
tail -40 "$SWEEP/$SPLIT/compositional.txt" | tee -a "$PROGRESS"
say "overnight finished"
