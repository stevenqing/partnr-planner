#!/usr/bin/env bash
# The baseline half of the sibling-grouped held-out column, on the 72B.
#
# A9_SIBLINGS=1 makes viki_amendment9_folds.held_out_ids hide the whole pre-registered
# group, so every memory-holding arm loses the sibling as well as the family. Removing the
# sibling for G-Memory alone would be the mirror image of the bias being corrected, so all
# three memory arms are rerun. zero_shot holds no memory and is unaffected: its existing
# fold rows are reused, not rerun.
#
# Only three evaluation folds have a sibling (see results/sibling_folds_preregistration.json),
# so this is 383 rows per arm rather than 924. Cells land beside the originals as
# `<arm>.sibgrp.jsonl` -- the archived single-family runs are never overwritten.
#
# Run:  setsid nohup bash scripts/drivers/viki_sibling_folds_baselines.sh \
#         > outputs/sibfolds_baselines.log 2>&1 < /dev/null &
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
BASE=${BASE:-http://192.168.32.40:8050/v1}
WORKERS=${WORKERS:-8}
STALL_MIN=${STALL_MIN:-20}
VARIANT=sibgrp
cd "$ROOT" || exit 1
export TOKENIZERS_PARALLELISM=false
export A9_SIBLINGS=1
export A8B_SKILL_TOPK=8 A9_ROLE_AWARE=0 A9_PATTERN_SLOTS=0 A9_MODE="" A9_ACTION_CAP=0
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

if ! curl -sf -m 10 "$BASE/models" | grep -q qwen2.5-vl-72b-amendment3-f2; then
  say "ABORT: $BASE is not serving the frozen 72B"; exit 1
fi

# Wait on the PID, never on a command line: `pgrep -f` also matches the shell that carries
# the pattern in its own argv, which has wedged a supervisor here four times.
stall_watch () {     # out pid
  local out=$1 pid=$2 last=-1 same=0 now
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    now=$(stat -c %s "$out" 2>/dev/null || echo 0)
    if [ "$now" = "$last" ]; then same=$((same + 1)); else same=0; last=$now; fi
    if [ "$same" -ge "$STALL_MIN" ]; then
      say "STALL  $out has not grown in $STALL_MIN min -- killing $pid"
      pkill -9 -P "$pid" 2>/dev/null; kill -9 "$pid" 2>/dev/null; return
    fi
  done
}

cell () {            # arm family rows
  local arm=$1 family=$2 rows=$3
  local out="$ROOT/results/viki_memory_experiments/amendment8b/folds/$family/${arm}.${VARIANT}.jsonl"
  local have; have=$(wc -l < "$out" 2>/dev/null || echo 0)
  if [ "$have" -eq "$rows" ]; then say "skip   $family/$arm -- $rows rows"; return 0; fi
  [ "$have" -gt 0 ] && { say "clear  $family/$arm -- $have rows is not $rows"; rm -f "$out" "$out.run.json"; }
  say "start  $family/$arm ($rows rows)"
  timeout -k 60 10800 "$PY" scripts/viki_amendment8b.py run-arm --arm "$arm" \
      --base-url "$BASE" --workers "$WORKERS" --fold "$family" --variant "$VARIANT" \
      >> "$ROOT/outputs/sibfolds_baselines.detail.log" 2>&1 &
  local job=$!
  stall_watch "$out" "$job" & local watcher=$!
  wait "$job"; local rc=$?
  kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
  [ "$rc" -eq 0 ] && say "done   $family/$arm -> $(wc -l < "$out" 2>/dev/null || echo 0) rows" \
                  || say "FAILED $family/$arm rc=$rc"
}

for arm in gmemory skill_memory trajectory_rag; do
  cell "$arm" cut_fruit_on_board 189
  cell "$arm" cut_two_fruits_on_board 108
  cell "$arm" parallel_human_dual_asset_to_plate_or_bowl 86
done
say "sibling-grouped baseline folds finished; nothing here was scored"
