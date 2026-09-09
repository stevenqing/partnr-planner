#!/usr/bin/env bash
# The G-Memory retrieval placebo on the folds this table is missing it for.
#
# `A9_GMEM_SHUFFLE=1` randomises which records G-Memory retrieves. It answers "how much of
# this arm's fold score is retrieval at all", and only the 72B single-family column has it.
# It is a CONTROL, never a baseline: it must never be substituted for G-Memory itself.
#
# Run:  MODEL=30B setsid nohup bash scripts/drivers/viki_shuffled_folds.sh \
#         > outputs/shuffled_30b.log 2>&1 < /dev/null &
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
MODEL=${MODEL:?set MODEL}
WORKERS=${WORKERS:-8}
STALL_MIN=${STALL_MIN:-20}
cd "$ROOT" || exit 1
export TOKENIZERS_PARALLELISM=false A9_GMEM_SHUFFLE=1
case "$MODEL" in
  72B) BASE=http://192.168.32.40:8050/v1; SERVED=qwen2.5-vl-72b-amendment3-f2; BACKBONE=qwen2_5_vl_72b; TAG="" ;;
  30B) BASE=http://127.0.0.1:8062/v1;     SERVED=qwen3-vl-30b;  BACKBONE=qwen3_vl_30b;  TAG=m30 ;;
  7B)  BASE=http://127.0.0.1:8061/v1;     SERVED=qwen2.5-vl-7b; BACKBONE=qwen2_5_vl_7b; TAG=m7 ;;
  *) echo "MODEL must be 72B / 30B / 7B"; exit 1 ;;
esac
export VIKI_BACKBONE=$BACKBONE VIKI_SERVED_MODEL=$SERVED
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
if ! curl -sf -m 10 "$BASE/models" | grep -q "$SERVED"; then say "ABORT: $BASE not serving $SERVED"; exit 1; fi

ALL=$($PY -c 'import sys;sys.path.insert(0,"scripts");sys.path.insert(0,".");
import viki_amendment9_folds as F;print(" ".join(F.folds()))' 2>/dev/null)
SIB=$($PY -c 'import json;print(" ".join(json.load(open("results/sibling_folds_preregistration.json"))["affected_eval_folds"]))')
rows_of () { $PY -c "import sys;sys.path.insert(0,'scripts');sys.path.insert(0,'.');
import viki_amendment9_folds as F;print(len(F.rows_of('$1')))" 2>/dev/null; }

stall_watch () { local out=$1 pid=$2 last=-1 same=0 now
  while kill -0 "$pid" 2>/dev/null; do sleep 60
    now=$(stat -c %s "$out" 2>/dev/null || echo 0)
    if [ "$now" = "$last" ]; then same=$((same+1)); else same=0; last=$now; fi
    if [ "$same" -ge "$STALL_MIN" ]; then say "STALL $out -- killing $pid"
      pkill -9 -P "$pid" 2>/dev/null; kill -9 "$pid" 2>/dev/null; return; fi
  done; }

cell () {  # family variant
  local family=$1 variant=$2
  local out="$ROOT/results/viki_memory_experiments/amendment8b/folds/$family/gmemory.${variant}.jsonl"
  local want; want=$(rows_of "$family")
  [ "$(wc -l < "$out" 2>/dev/null || echo 0)" = "$want" ] && { say "skip   $variant $family"; return 0; }
  say "start  $variant $family ($want rows)"
  timeout -k 60 10800 "$PY" scripts/viki_amendment8b.py run-arm --arm gmemory --base-url "$BASE" \
      --workers "$WORKERS" --fold "$family" --variant "$variant" \
      >> "outputs/shuffled_${MODEL}.detail.log" 2>&1 &
  local job=$!; stall_watch "$out" "$job" & local watcher=$!
  wait "$job"; local rc=$?; kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
  [ "$rc" -eq 0 ] && say "done   $variant $family -> $(wc -l < "$out" 2>/dev/null || echo 0)" || say "FAILED $variant $family"
}

single="shuffled"; grouped="shuffled.sibgrp"
[ -n "$TAG" ] && { single="shuffled.$TAG"; grouped="shuffled.${TAG}sibgrp"; }
unset A9_SIBLINGS
if [ -n "$TAG" ]; then for f in $ALL; do cell "$f" "$single"; done; fi   # 72B already has these
export A9_SIBLINGS=1
for f in $SIB; do cell "$f" "$grouped"; done
say "shuffled control finished for $MODEL"
