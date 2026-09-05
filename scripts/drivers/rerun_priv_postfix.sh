#!/usr/bin/env bash
# The privileged sweep again, after the step-0 death was fixed.
#
# `outputs/sweep_remeasured` is the pre-fix table: in it, 70 of 366 `val_mini` episodes
# ended on the first planner call with both agents reporting `Done`, every one scoring
# zero, because privileged requirements arrive pre-bound and so never reached the explore
# fallback inside `_bind`. Those numbers are not wrong -- they are what the planner did --
# but they are a lower bound on it, and the compositional gap read off them is inflated by
# however much of that 19% was recoverable.
#
# The fix changes what an idle agent does, so it cannot be assumed to touch only the dead
# episodes: an agent that used to stand down while its partner worked now goes and looks.
# That is why all six cells are re-measured rather than patching the 70 rows into the old
# table. Privileged goals mean no model is called; this costs GPU time only.
#
# Writes to a new directory. `sweep_remeasured` is the evidence for what the bug cost and
# is not written over.
set -u

ROOT=/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
SPLIT=val_mini
PROCS=${PROCS:-20}
# 0/1/6 were free when this was written; 2-5 are the 30B and 7 is the 7B. Check nvidia-smi.
GPU=${GPU:-0}
OUT=$ROOT/outputs/sweep_postfix/$SPLIT

# Sized off the pre-fix run: six cells at 42-48 minutes each with 70 episodes costing
# nothing. Those 70 now run for real, so the per-cell budget is roughly doubled rather
# than copied across from the old script -- copying a neighbouring cell's timeout is how
# the ID cell got killed every round on the VIKI side.
HARD=${HARD:-9000}
# Hard timeouts cannot tell slow from hung. The runner appends as it goes; 20 minutes of
# no growth in run.log is dead.
STALL=${STALL:-1200}

cd "$ROOT" || exit 1
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false

both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

run () {             # name config extra...
  local name=$1 config=$2; shift 2
  local out="$OUT/$name"
  [ -f "$out/DONE" ] && { say "skip   $name"; return 0; }
  mkdir -p "$out"
  local log="$out/run.log"
  say "start  $name"
  CUDA_VISIBLE_DEVICES=$GPU timeout $HARD $PY -m habitat_llm.examples.planner_demo \
      --config-name "$config" \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$SPLIT.json.gz" \
      num_proc="$PROCS" evaluation.save_video=False +resume=True \
      hydra.run.dir="$out" "$@" >> "$log" 2>&1 &
  local runner=$! last=0 quiet=0 now
  # Wait by PID, never by `pgrep -f`: the pattern appears in this script's own command
  # line and in the ssh that launched it, which has killed a night's work six times.
  while kill -0 "$runner" 2>/dev/null; do
      sleep 60
      now=$(wc -c < "$log" 2>/dev/null || echo 0)
      if [ "$now" -gt "$last" ]; then last=$now; quiet=0; else quiet=$((quiet + 60)); fi
      if [ "$quiet" -ge "$STALL" ]; then
          say "STALL  $name: run.log flat for ${quiet}s -- killing $runner"
          kill -9 "$runner" 2>/dev/null
          pkill -9 -f "$PY -m habitat_llm.examples.planner_demo" 2>/dev/null
          break
      fi
  done
  wait "$runner"; local status=$? n
  n=$(ls "$out/results/$SPLIT.json.gz/stats" 2>/dev/null | wc -l)
  [ "$status" -eq 0 ] && touch "$out/DONE"
  say "$([ "$status" -eq 0 ] && echo done || echo FAILED)  $name ($n episodes)"
}

R=results/partnr_operators.json
ALL=results/partnr_operators_train_mini_all.json

# `ceiling` is the scripted full-observability planner and never went through the code
# that was fixed. It is re-run anyway so the whole table comes from one batch -- the 09-02
# sweep is unquotable precisely because it was quoted a cell at a time.
run ceiling        baselines/heuristic_full_obs.yaml
run v2_memory_R    baselines/skill_memory_v2_oracle_goals.yaml $(both operators $R)
run v2_memory_all  baselines/skill_memory_v2_oracle_goals.yaml $(both operators $ALL)
run v2_R_nofold    baselines/skill_memory_v2_oracle_goals.yaml $(both operators $R) $(both allow_spatial_composition False)
run v2_R_noorder   baselines/skill_memory_v2_oracle_goals.yaml $(both operators $R) $(both use_episode_order False)
# num_proc matches the rest of the batch, as it did in the pre-fix re-measurement: what
# this row has to be comparable with is the table it sits in.
run v2_retry       baselines/skill_memory_v2_oracle_goals.yaml $(both operators $R) \
    $(both retry_limit 6) \
    +evaluation.agents.agent_0.planner.plan_config.repair_limit=10 \
    +evaluation.agents.agent_1.planner.plan_config.repair_limit=10

say "post-fix sweep finished -> $OUT"
