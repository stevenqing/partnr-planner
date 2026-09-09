#!/usr/bin/env bash
# The held-out column's baselines on a model other than the 72B.
#
# `baseline_comparison.json` reports both OOD columns for the 72B only, because the fold
# runs exist only there -- for 30B and 7B every baseline row is declared missing rather
# than filled from another model. This driver produces them.
#
# Two passes over the same eight folds:
#   single    one family hidden          -- the column the archived 72B rows measure
#   sibgrp    the whole pre-registered sibling group hidden (A9_SIBLINGS=1), and only the
#             three folds that have a sibling; the other five are the same experiment
#             either way and are reused when the column is assembled.
#
# Every cell is namespaced by `--variant <TAG>` so it lands beside, and never on top of,
# the 72B archives: folds/<family>/<arm>.<TAG>.jsonl.
#
# Run:  MODEL=30B setsid nohup bash scripts/drivers/viki_folds_crossmodel.sh \
#         > outputs/folds_30b.log 2>&1 < /dev/null &
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
MODEL=${MODEL:?set MODEL to 30B or 7B}
WORKERS=${WORKERS:-8}
STALL_MIN=${STALL_MIN:-20}
cd "$ROOT" || exit 1
export TOKENIZERS_PARALLELISM=false
# v1's configuration, matching the archived 72B fold cells (a8b: A8B_SKILL_TOPK=8, cap off).
export A8B_SKILL_TOPK=8 A9_ROLE_AWARE=0 A9_PATTERN_SLOTS=0 A9_MODE="" A9_ACTION_CAP=0

case "$MODEL" in
  30B) BASE=http://127.0.0.1:8062/v1; SERVED=qwen3-vl-30b;  BACKBONE=qwen3_vl_30b;  TAG=m30 ;;
  7B)  BASE=http://127.0.0.1:8061/v1; SERVED=qwen2.5-vl-7b; BACKBONE=qwen2_5_vl_7b; TAG=m7  ;;
  *)   echo "MODEL must be 30B or 7B"; exit 1 ;;
esac
export VIKI_BACKBONE=$BACKBONE VIKI_SERVED_MODEL=$SERVED
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

if ! curl -sf -m 10 "$BASE/models" | grep -q "$SERVED"; then
  say "ABORT: $BASE is not serving $SERVED"; exit 1
fi

ALL_FOLDS=$($PY -c 'import sys;sys.path.insert(0,"scripts");sys.path.insert(0,".");
import viki_amendment9_folds as F;print(" ".join(F.folds()))' 2>/dev/null)
SIB_FOLDS=$($PY -c 'import json;print(" ".join(json.load(open("results/sibling_folds_preregistration.json"))["affected_eval_folds"]))')
[ -z "$ALL_FOLDS" ] && { say "ABORT: could not list folds"; exit 1; }
say "model=$MODEL tag=$TAG folds=$(echo $ALL_FOLDS | wc -w) sibling folds=$(echo $SIB_FOLDS | wc -w)"

rows_of () { $PY -c "import sys;sys.path.insert(0,'scripts');sys.path.insert(0,'.');
import viki_amendment9_folds as F;print(len(F.rows_of('$1')))" 2>/dev/null; }

# Wait on the PID, never on a command line: `pgrep -f` also matches the shell carrying the
# pattern in its own argv, which has wedged a supervisor here four separate times.
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

cell () {            # arm family variant
  local arm=$1 family=$2 variant=$3
  local out="$ROOT/results/viki_memory_experiments/amendment8b/folds/$family/${arm}.${variant}.jsonl"
  local want; want=$(rows_of "$family")
  local have; have=$(wc -l < "$out" 2>/dev/null || echo 0)
  if [ "$have" = "$want" ]; then say "skip   $variant $family/$arm -- $want rows"; return 0; fi
  [ "$have" -gt 0 ] && { say "clear  $variant $family/$arm -- $have is not $want"; rm -f "$out" "$out.run.json"; }
  say "start  $variant $family/$arm ($want rows)"
  timeout -k 60 10800 "$PY" scripts/viki_amendment8b.py run-arm --arm "$arm" \
      --base-url "$BASE" --workers "$WORKERS" --fold "$family" --variant "$variant" \
      >> "$ROOT/outputs/folds_${TAG}.detail.log" 2>&1 &
  local job=$!; stall_watch "$out" "$job" & local watcher=$!
  wait "$job"; local rc=$?
  kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
  [ "$rc" -eq 0 ] && say "done   $variant $family/$arm -> $(wc -l < "$out" 2>/dev/null || echo 0) rows" \
                  || say "FAILED $variant $family/$arm rc=$rc"
}

memento_cell () {    # family variant_suffix
  local family=$1 suffix=$2
  local tag="memento_fold${suffix}_${family}"
  local out="$ROOT/results/viki_memory_experiments/amendment11/${tag}.jsonl"
  local want; want=$(rows_of "$family")
  [ "$(wc -l < "$out" 2>/dev/null || echo 0)" = "$want" ] && { say "skip   $tag"; return 0; }
  say "start  $tag ($want rows)"
  timeout -k 60 10800 "$PY" scripts/viki_eval_memento.py --model "$SERVED" --base-url "$BASE" \
      --split id --workers "$WORKERS" --exclude-family "$family" --only-family "$family" \
      --tag "$tag" >> "$ROOT/outputs/folds_${TAG}.detail.log" 2>&1 \
      && say "done   $tag -> $(wc -l < "$out" 2>/dev/null || echo 0) rows" || say "FAILED $tag"
}

# ---- pass 1: single-family folds, all eight
unset A9_SIBLINGS
for arm in gmemory skill_memory trajectory_rag zero_shot; do
  for family in $ALL_FOLDS; do cell "$arm" "$family" "$TAG"; done
done
for family in $ALL_FOLDS; do memento_cell "$family" "_${TAG}"; done
say "pass 1 (single-family) finished for $MODEL"

# ---- pass 2: sibling-grouped, only the three folds that have a sibling.
# zero_shot holds no memory, so hiding a sibling cannot change it -- its rows are reused.
export A9_SIBLINGS=1
for arm in gmemory skill_memory trajectory_rag; do
  for family in $SIB_FOLDS; do cell "$arm" "$family" "${TAG}sibgrp"; done
done
for family in $SIB_FOLDS; do memento_cell "$family" "grp_${TAG}"; done
say "pass 2 (sibling-grouped) finished for $MODEL; nothing here was scored"
