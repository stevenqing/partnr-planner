#!/usr/bin/env bash
# P0: the four prompt-shaped baselines on Qwen3-VL-30B, three splits, twelve cells.
#
# The main table currently has five methods on the 72B and almost nothing anywhere else:
# of the sixty cells a cross-model table needs, two exist (MEMENTO's 30B and 7B on ID).
# Skill memory v2 already spans three models, so the baselines are the missing half of a
# second complete table, and these twelve are it.
#
# Everything is held to the 72B batch's conditions, because a cell run under different
# conditions is not a comparison:
#   * the partner prefix stays on for the baselines -- it leaks the first step of the
#     reference plan, and the 72B baselines had it. Removing it here would flatter us.
#   * skill memory v1 is the `fullactions_k8` configuration: A8B_SKILL_TOPK=8 with the
#     action cap off, which is what the archived 72B cell was produced under.
#   * seed 20260829, and scoring is never done here -- `viki_report_matrix.py` re-scores
#     every cell from raw responses under the JSON-tolerant reading.
#
# The backbone is chosen once, by name, and the endpoint gate refuses to run if what is
# served does not match -- so a cell cannot be produced against one model and labelled
# with another.
#
# zero_shot on ID is already done (tag m30) and is skipped by the `[ -f ]` guard.
#
# Run:  setsid nohup bash scripts/drivers/viki_p0_30b.sh > outputs/p0_30b.log 2>&1 < /dev/null &

set -u

ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
BASE=${BASE:-http://127.0.0.1:8062/v1}
WORKERS=${WORKERS:-8}
# A hung cell and a merely slow one look identical from the outside: both burn CPU with
# hundreds of threads while the endpoint sits at zero requests running. The one thing that
# separates them is whether the output is still growing -- every row is appended and
# flushed as it lands (viki_amendment8b.py:653) -- so a cell that has written nothing for
# this many minutes is hung, not slow. Three such hangs have cost 3h, 10h and 10h of wall
# clock; the last one ran 11 hours on a zero-byte file because only the recombination
# cells were bounded and the one that hung was an ID cell.
STALL_MIN=${STALL_MIN:-20}
# Repeat index. These arms are not greedy -- the same cell run twice agreed on only 1 of
# 297 rows -- so a single run reports one draw and says nothing about its own spread.
# Three repeats under identical conditions give a mean and a standard deviation, and the
# tag carries the index so a repeat can never overwrite the one before it.
REP=${REP:-1}
# `NOTHINK=1` drops the thinking instruction from the benchmark's own system prompt and
# keeps the answer instruction (see `drop_think_rule` in viki_amendment8b.py). It is a NEW
# CONDITION, not a fix: cells produced under it are not comparable with the 72B batch and
# get their own tag so they can never be merged into the main table by accident.
NOTHINK=${NOTHINK:-0}
BASE_TAG=${BASE_TAG:-m30}
if [ "$NOTHINK" = "1" ]; then
  export VIKI_NO_THINK=1
  BASE_TAG="${BASE_TAG}nt"
fi
TAG="$BASE_TAG"
[ "$REP" != "1" ] && TAG="${TAG}r${REP}"

cd "$ROOT" || exit 1
export TOKENIZERS_PARALLELISM=false
export VIKI_BACKBONE=qwen3_vl_30b VIKI_SERVED_MODEL=qwen3-vl-30b
# The skill-memory arm's configuration, copied from drivers/a9_fullactions.sh so the cell
# matches the archived 72B one. These are read at import time by the arm builder.
export A8B_SKILL_TOPK=8 A9_ROLE_AWARE=0 A9_PATTERN_SLOTS=0 A9_MODE="" A9_ACTION_CAP=0

A8B=results/viki_memory_experiments/amendment8b
A10=results/viki_memory_experiments/amendment10
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

# Waiting is done on the PID and never by matching a command line: `pgrep -f <pattern>`
# also matches whatever shell has that pattern in its own command line, which has wedged a
# supervisor of this batch four separate times. Killing likewise goes to a PID we own.
stall_watch () {     # out pid
  local out=$1 pid=$2 last=-1 same=0 now
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    now=$(stat -c %s "$out" 2>/dev/null || echo 0)
    if [ "$now" = "$last" ]; then same=$((same + 1)); else same=0; last=$now; fi
    if [ "$same" -ge "$STALL_MIN" ]; then
      say "STALL  $out has not grown in $STALL_MIN min -- killing $pid"
      # Children first: orphaning the worker by killing the `timeout` wrapper ahead of it
      # would leave the hung python holding the endpoint. -P matches on parent, not on a
      # command line, so it cannot match this script.
      pkill -9 -P "$pid" 2>/dev/null
      kill -9 "$pid" 2>/dev/null
      return
    fi
  done
}

# Every cell goes through here -- both shapes, not just the shape that hung last time.
guarded () {         # out hard_limit_sec cmd...
  local out=$1 hard=$2 job watcher rc
  shift 2
  # -k: a wedged worker has ignored TERM before, so follow it with KILL a minute later.
  timeout -k 60 "$hard" "$@" &
  job=$!
  stall_watch "$out" "$job" &
  watcher=$!
  wait "$job"; rc=$?
  kill "$watcher" 2>/dev/null
  wait "$watcher" 2>/dev/null
  return "$rc"
}

if ! curl -sf "$BASE/models" | grep -q qwen3-vl-30b; then
  say "ABORT: $BASE is not serving qwen3-vl-30b"
  exit 1
fi

# --------------------------------------------------------------- ID, 924 rows each

id_cell () {         # arm variant
  local arm=$1 variant=$2
  local out="$A8B/${arm}.${variant}.jsonl"
  local have; have=$(wc -l < "$out" 2>/dev/null || echo 0)
  if [ "$have" -eq 924 ]; then say "skip   ID $arm ($variant) -- 924 rows"; return 0; fi
  if [ "$have" -gt 0 ]; then
    say "clear  ID $arm ($variant) -- $have rows is not 924, starting over"
    rm -f "$out" "$out.run.json" "${out%.jsonl}.summary.json"
  fi
  say "start  ID $arm ($variant)"
  # The hard limit is only a backstop, sized off the slowest ID cell ever measured here
  # (trajectory_rag, 85 min) so it can never cut a healthy run short -- the 3600 the
  # recombination cells use would kill that arm every single time. What actually catches a
  # hang is the stall watch, in twenty minutes rather than the ten hours this cost on 09-04.
  guarded "$out" 10800 \
      $PY scripts/viki_amendment8b.py run-arm --arm "$arm" --base-url "$BASE" \
      --workers "$WORKERS" --variant "$variant" \
      && say "done   ID $arm -> $out" \
      || say "FAILED ID $arm"
}

id_cell zero_shot      "$TAG"
id_cell trajectory_rag "$TAG"
id_cell gmemory        "$TAG"
# The v1 arm carries its configuration in the variant name, as the 72B cell does.
id_cell skill_memory   "fullactions_k8_$TAG"

# ------------------------------------------------- recombination, 297 rows per variant
#
# Imaged and text are one paired design over the same 297 rows; running only one of them
# would leave the pair unusable, so both are always run for an arm.

recomb_cell () {     # arm
  local arm=$1
  for split in imaged text; do
    local out="$A10/$split/${arm}.${TAG}.jsonl"
    local have; have=$(wc -l < "$out" 2>/dev/null || echo 0)
    if [ "$have" -eq 297 ]; then say "skip   recomb $split $arm -- 297 rows"; continue; fi
    if [ "$have" -gt 0 ]; then
      say "clear  recomb $split $arm -- $have rows is not 297, starting over"
      rm -f "$out" "$out.run.json" "${out%.jsonl}.summary.json"
    fi
    say "start  recomb $split $arm"
    # Bounded, because this arm has twice hung with the endpoint idle, hundreds of
    # blocked threads and a zero-byte output -- a failure that looks exactly like slow
    # work from the outside and once cost three hours of wall clock. A cell that has not
    # produced anything in an hour is not going to.
    guarded "$out" 3600 \
        $PY scripts/viki_amendment10_run.py --split "$split" --arm "$arm" \
        --base-url "$BASE" --workers "$WORKERS" --tag "$TAG" \
        && say "done   recomb $split $arm -> $out" \
        || say "FAILED recomb $split $arm"
  done
}

for arm in zero_shot trajectory_rag gmemory skill_memory; do
  recomb_cell "$arm"
done

say "P0 generation finished; nothing here was scored -- run viki_report_matrix.py"
