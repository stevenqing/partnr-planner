#!/usr/bin/env bash
# Overnight sweep: skill memory v2 against PARTNR's own privileged ceiling.
#
# Every cell is driven by the episode's own propositions, so nothing here needs a model
# to be up and the whole grid is reproducible from an operator library and a dataset. It
# is meant to be started and left: cells are independent, each writes to its own hydra run
# directory, and a cell that has finished is skipped on a re-run, so the script can be
# killed and restarted at any point without losing or duplicating work.
#
#   ceiling        PARTNR's scripted composer -- centralized, fully observed, reading the
#                  ground-truth propositions. The upper bound the benchmark itself defines.
#   v2_memory_R    our composer -- decentralized, partially observed -- with a memory
#                  induced from rearrangement-only episodes. The paper's first cell.
#   v2_memory_all  the same, with a memory induced from every training task type. The
#                  difference between the two is what the memory's coverage buys.
#   v2_R_nofold    v2_memory_R without the spatial fold: the control for the one piece of
#                  schema knowledge the memory asserts rather than induces.
#   v2_R_noorder   v2_memory_R without reading the episode's temporal DAG. This measures
#                  what an ordering is worth, and so what the mined ordering rules will
#                  have to recover in the arm that has no episode to read.
#
# Run as:  PROCS=40 CUDA_VISIBLE_DEVICES=1 nohup bash scripts/drivers/partnr_v2_sweep.sh &

set -u

ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
PROCS=${PROCS:-40}
SWEEP=${SWEEP:-outputs/sweep}
SPLITS=${SPLITS:-"val_mini val"}
CELL_TIMEOUT=${CELL_TIMEOUT:-10800}
R_MEMORY=${R_MEMORY:-results/partnr_operators.json}
ALL_MEMORY=${ALL_MEMORY:-results/partnr_operators_train_mini_all.json}

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
# Leave GPU 0 to whatever is already recording rollouts there.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

cd "$ROOT" || exit 1
mkdir -p "$SWEEP"
PROGRESS="$SWEEP/progress.log"

say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$PROGRESS"; }

# The `operators` and ablation overrides go to both planners: they are separate instances
# and there is deliberately nothing shared between them at run time.
both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

run () {
  local split=$1 name=$2 config=$3; shift 3
  local out="$SWEEP/$split/$name"
  local data="data/datasets/partnr_episodes/v0_0/$split.json.gz"
  if [ -f "$out/DONE" ]; then say "skip   $split/$name"; return 0; fi
  mkdir -p "$out"
  local attempt
  for attempt in 1 2; do
    say "start  $split/$name (attempt $attempt)"
    # `resume` is passed on every attempt, not only after a failure. It skips any episode
    # that already has a planner log, so it costs nothing on a fresh cell and means the
    # sweep can be stopped and restarted at will -- to change the parallelism, say --
    # without throwing away the hours already in the directory.
    timeout "$CELL_TIMEOUT" "$PY" -m habitat_llm.examples.planner_demo \
        --config-name "$config" \
        habitat.dataset.data_path="$data" \
        num_proc="$PROCS" \
        evaluation.save_video=False \
        +resume=True \
        hydra.run.dir="$out" \
        "$@" >> "$out/run.log" 2>&1
    local status=$?
    local done_count
    done_count=$(ls "$out/results/$split.json.gz/stats" 2>/dev/null | wc -l)
    if [ "$status" -eq 0 ]; then
      touch "$out/DONE"
      say "done   $split/$name  ($done_count episodes)"
      return 0
    fi
    say "FAILED $split/$name status=$status ($done_count episodes) -- retrying"
  done
  say "GIVEUP $split/$name"
  return 1
}

say "sweep starting: splits='$SPLITS' procs=$PROCS gpu=$CUDA_VISIBLE_DEVICES"
for split in $SPLITS; do
  run "$split" ceiling       baselines/heuristic_full_obs.yaml
  run "$split" v2_memory_R   baselines/skill_memory_v2_oracle_goals.yaml $(both operators "$R_MEMORY")
  run "$split" v2_memory_all baselines/skill_memory_v2_oracle_goals.yaml $(both operators "$ALL_MEMORY")
  run "$split" v2_R_nofold   baselines/skill_memory_v2_oracle_goals.yaml $(both operators "$R_MEMORY") $(both allow_spatial_composition False)
  run "$split" v2_R_noorder  baselines/skill_memory_v2_oracle_goals.yaml $(both operators "$R_MEMORY") $(both use_episode_order False)

  say "report $split"
  "$PY" scripts/partnr_v2_report.py \
      --sweep "$SWEEP/$split" --dataset "$split.json.gz" --split "$split" \
      --out "$SWEEP/$split/report.json" > "$SWEEP/$split/report.txt" 2>&1
done
say "sweep finished"
