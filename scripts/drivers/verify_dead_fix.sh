#!/usr/bin/env bash
# Does the explore fallback in `_claim` revive the 70 episodes that died at step 0?
#
# Before this fix, `priv:v2_memory_R` abandoned 70 of 366 `val_mini` episodes on the very
# first planner call: privileged requirements arrive already bound, so the explore branch
# inside `_bind` never fired for them, an agent that could not yet see any of its objects
# found nothing to ground, and both agents reported `Done` at once. All 70 scored exactly
# zero. The same 70 die under `retry_limit=6`, which is the tell -- the episode ends
# before a retry can happen.
#
# This runs only those 70, against the same config and operator library as the archived
# cell, so the comparison is paired and the archived scores (all 0.0) are the `before`.
# Privileged goals mean no model is called: this costs GPU time and nothing else.
#
# Results go to their own directory. The archived cell is evidence about what the bug did
# and is not written over.
set -u

ROOT=/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
SPLIT=val_mini
PROCS=${PROCS:-20}
# GPU 0/1/6 were all free when this was written. Check `nvidia-smi` before launching.
GPU=${GPU:-0}
OUT=$ROOT/outputs/dead_fix/$SPLIT/v2_memory_R_deadonly
DEAD=$ROOT/outputs/partnr_dead_sets.json

# 90 minutes. The archived cell did 369 episodes in 42 minutes at this `num_proc`, but 70
# of those were the step-0 deaths that cost nothing; these 70 now run for real and may run
# to the step budget, so the guard is sized off the slow case, not off that average.
HARD=${HARD:-5400}
# And a stall guard, because a hard timeout alone cannot tell a slow run from a hung one.
# The runner appends to run.log as it goes; 20 minutes without growth is dead.
STALL=${STALL:-1200}

cd "$ROOT" || exit 1
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false

say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

IDS=$($PY -c "
import json
print('[' + ','.join(json.load(open('$DEAD'))['v2_memory_R']) + ']')
") || { say "could not read $DEAD"; exit 1; }
say "episodes: $IDS"

mkdir -p "$OUT"
LOG=$OUT/run.log
: > "$LOG"

R=results/partnr_operators.json
both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

say "start  v2_memory_R_deadonly (gpu $GPU, num_proc $PROCS)"
CUDA_VISIBLE_DEVICES=$GPU timeout $HARD $PY -m habitat_llm.examples.planner_demo \
    --config-name baselines/skill_memory_v2_oracle_goals.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$SPLIT.json.gz" \
    num_proc="$PROCS" evaluation.save_video=False \
    +episode_id_filter="$IDS" \
    hydra.run.dir="$OUT" $(both operators $R) >> "$LOG" 2>&1 &
RUNNER=$!

# Wait by PID and watch the log for growth. Never `pgrep -f` here: the pattern would match
# this script's own command line, which has cost six sessions already.
LAST=0; QUIET=0
while kill -0 "$RUNNER" 2>/dev/null; do
    sleep 60
    NOW=$(wc -c < "$LOG" 2>/dev/null || echo 0)
    if [ "$NOW" -gt "$LAST" ]; then LAST=$NOW; QUIET=0; else QUIET=$((QUIET + 60)); fi
    if [ "$QUIET" -ge "$STALL" ]; then
        say "STALL: run.log has not grown in ${QUIET}s -- killing $RUNNER"
        kill -9 "$RUNNER" 2>/dev/null
        pkill -9 -f "$PY -m habitat_llm.examples.planner_demo" 2>/dev/null
        break
    fi
done
wait "$RUNNER"; STATUS=$?

N=$(ls "$OUT/results/$SPLIT.json.gz/stats" 2>/dev/null | wc -l)
[ "$STATUS" -eq 0 ] && touch "$OUT/DONE"
say "$([ "$STATUS" -eq 0 ] && echo done || echo FAILED\ \(status\ $STATUS\))  $N episodes -> $OUT"
