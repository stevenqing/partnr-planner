#!/usr/bin/env bash
# RQ2 no_trace downstream: waits for viki_rq2_no_trace.sh to finish its build stage, then runs the
# no_trace library (possibly empty -> the evaluator's own fallback) through the Ours replay pipeline
# on all three models at once, each against its own endpoint. Rows are this run's own, never the
# zero-shot files.
#
#   setsid nohup bash scripts/drivers/viki_rq2_notrace_downstream.sh \
#     > outputs/paper_viki_0914/rq2_notrace_downstream.log 2>&1 < /dev/null &
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
IND_LOG=${IND_LOG:-outputs/paper_viki_0914/rq2_notrace_induction.log}
LIBS=results/paper_viki_iclr2027/libraries/rq2/no_trace
L=${L:-outputs/paper_viki_0914}
URL_72B=${URL_72B:-http://127.0.0.1:8050/v1}
URL_30B=${URL_30B:-http://127.0.0.1:8062/v1}
URL_7B=${URL_7B:-http://127.0.0.1:8061/v1}
cd "$ROOT" || exit 1
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

# Up to 12 h for the induction to reach "build done".
for _ in $(seq 1 1440); do
  grep -q "FATAL" "$IND_LOG" 2>/dev/null && { say "induction reported FATAL -- not launching:"; grep "FATAL" "$IND_LOG"; exit 1; }
  grep -q "build done:" "$IND_LOG" 2>/dev/null && break
  sleep 30
done
grep -q "build done:" "$IND_LOG" 2>/dev/null || { say "induction never reached build done"; exit 1; }
[ -f "$LIBS/memory_all.json" ] || { say "build done but $LIBS/memory_all.json missing"; exit 1; }
n=$(ls "$LIBS"/memory_heldout_*.json 2>/dev/null | wc -l)
[ "$n" -eq 8 ] || { say "expected 8 fold libraries in $LIBS, found $n"; exit 1; }
say "no_trace libraries ready: $(sha256sum "$LIBS/memory_all.json" | cut -c1-16)…"

pids=()
for spec in "72B $URL_72B" "30B $URL_30B" "7B $URL_7B"; do
  set -- $spec
  EXPERIMENT=rq2 MODEL=$1 CONDITION=no_trace BASE_URL=$2 WORKERS=8 \
    setsid nohup bash scripts/drivers/viki_replay_cells.sh > "$L/rq2_notrace_$1.log" 2>&1 < /dev/null &
  pids+=($!); say "rq2 no_trace $1 -> $2 pid $!"
done
for p in "${pids[@]}"; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
for m in 72B 30B 7B; do say "$m: $(tail -n 1 "$L/rq2_notrace_$m.log")"; done
say "no_trace downstream finished"
