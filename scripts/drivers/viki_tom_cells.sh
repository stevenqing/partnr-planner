#!/usr/bin/env bash
# Figure 2 ToM arm (VIKI-L2): one model, the four canonical splits, run one after another.
#
#   id                 924 rows, one run
#   ood_single_family  the same 924 rows as its own job: 8 single-family fold runs (mirroring
#                      how the zero-shot OOD cells were produced by viki_folds_crossmodel.sh
#                      / a9_folds.sh), then merge-folds into one cell. Never copied from id.
#   cg_image           297 rows (recombination.imaged.parquet)
#   pure_text          297 rows (recombination.text.parquet)
#
# ToM = the zero_shot request + viki_amendment6.TOM_REASONING_TEMPLATE on the system message
# (scripts/viki_tom_arm.py). Think condition: VIKI_NO_THINK is unset here and the runner
# refuses it. Rows land in results/paper_viki_iclr2027/raw/figure2_tom/<slug>/<split>/.
#
# One generation job per endpoint: check nothing else is using BASE before launching.
# Run:  MODEL=7B setsid nohup bash scripts/drivers/viki_tom_cells.sh \
#         > outputs/tom_7b.log 2>&1 < /dev/null &
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
MODEL=${MODEL:?set MODEL to 72B, 30B or 7B}
WORKERS=${WORKERS:-8}
STALL_MIN=${STALL_MIN:-20}
OUT_ROOT=${OUT_ROOT:-$ROOT/results/paper_viki_iclr2027/raw}

case "$MODEL" in
  72B) BASE=${BASE:-http://192.168.32.40:8050/v1}; SERVED=qwen2.5-vl-72b-amendment3-f2
       BACKBONE=qwen2_5_vl_72b; SLUG=qwen2.5-vl-72b-instruct ;;
  30B) BASE=${BASE:-http://127.0.0.1:8062/v1};     SERVED=qwen3-vl-30b
       BACKBONE=qwen3_vl_30b;   SLUG=qwen3-vl-30b-a3b-instruct ;;
  7B)  BASE=${BASE:-http://127.0.0.1:8061/v1};     SERVED=qwen2.5-vl-7b
       BACKBONE=qwen2_5_vl_7b;  SLUG=qwen2.5-vl-7b-instruct ;;
  *)   echo "MODEL must be 72B, 30B or 7B"; exit 1 ;;
esac

cd "$ROOT" || exit 1
unset VIKI_NO_THINK A9_SIBLINGS
export TOKENIZERS_PARALLELISM=false
export VIKI_BACKBONE=$BACKBONE VIKI_SERVED_MODEL=$SERVED
# The environment the zero-shot drivers ran under (viki_p0_7b.sh / viki_p0_30b.sh). These
# only configure memory arms, so they do not touch a memoryless request; set for parity.
export A8B_SKILL_TOPK=8 A9_ROLE_AWARE=0 A9_PATTERN_SLOTS=0 A9_MODE="" A9_ACTION_CAP=0
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

probe () {
  local out code
  out=$(curl -s -m 10 -w '\n%{http_code}' "$BASE/models")
  code=$(printf '%s\n' "$out" | tail -n 1)
  if [ "$code" != "200" ] || ! printf '%s\n' "$out" | grep -q "\"$SERVED\""; then
    say "ABORT: $BASE/models returned $code or does not serve $SERVED"
    return 1
  fi
}

# Wait on the PID, never on a command line (pgrep -f matches the supervising shell).
# The runner has its own in-process stall/hard guard that records the status in run.json;
# this external one exists because a wedged GIL stops the in-process watchdog too.
stall_watch () {     # out pid
  local out=$1 pid=$2 last=-1 same=0 now
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    now=$(stat -c %s "$out" 2>/dev/null || echo 0)
    if [ "$now" = "$last" ]; then same=$((same + 1)); else same=0; last=$now; fi
    if [ "$same" -ge $((STALL_MIN + 1)) ]; then
      say "STALL  $out has not grown in $((STALL_MIN + 1)) min -- killing $pid"
      for child in $(ps -o pid= --ppid "$pid" 2>/dev/null); do kill -9 "$child" 2>/dev/null; done
      kill -9 "$pid" 2>/dev/null
      return
    fi
  done
}

cell () {            # split fold hard_limit_sec
  local split=$1 fold=$2 hard=$3 dir job watcher rc
  dir="$OUT_ROOT/figure2_tom/$SLUG/$split"
  [ -n "$fold" ] && dir="$dir/folds/$fold"
  mkdir -p "$dir"
  probe || return 1
  say "start  $split ${fold:+fold=$fold }-> $dir"
  # The runner's own hard limit fires 5 min before the external one so it can write run.json.
  timeout -k 60 "$hard" "$PY" scripts/viki_tom_arm.py run --split "$split" ${fold:+--fold "$fold"} \
      --base-url "$BASE" --workers "$WORKERS" --out-root "$OUT_ROOT" \
      --stall-min "$STALL_MIN" --hard-timeout-sec $((hard - 300)) --log-path "$dir/run.log" \
      >> "$dir/run.log" 2>&1 &
  job=$!
  stall_watch "$dir/rows.jsonl" "$job" &
  watcher=$!
  wait "$job"; rc=$?
  kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
  if [ "$rc" -eq 0 ]; then
    say "done   $split ${fold:+fold=$fold }rows=$(wc -l < "$dir/rows.jsonl" 2>/dev/null || echo 0)"
  else
    say "FAILED $split ${fold:+fold=$fold }rc=$rc (see $dir/run.log, $dir/run.json; rerun resumes)"
  fi
  return "$rc"
}

probe || exit 1
say "model=$MODEL backbone=$BACKBONE base=$BASE slug=$SLUG out=$OUT_ROOT"

# Hard limits sized off the zero-shot history (72B ID 32 min, 72B recomb 12 min, 7B/30B
# faster) with head-room for the longer ToM reasoning; the stall guard is what catches hangs.
cell id "" 10800

FOLDS=$($PY scripts/viki_amendment9_folds.py --names 2>/dev/null)
COUNT=$(printf '%s\n' "$FOLDS" | grep -c .)
if [ "$COUNT" -ne 8 ]; then
  say "ABORT ood: expected 8 folds, got $COUNT"
else
  for fold in $FOLDS; do cell ood_single_family "$fold" 7200; done
  $PY scripts/viki_tom_arm.py merge-folds --out-root "$OUT_ROOT" \
      >> "$OUT_ROOT/figure2_tom/$SLUG/ood_single_family/merge.log" 2>&1 \
      && say "merged ood_single_family (complete)" \
      || say "merged ood_single_family INCOMPLETE -- see merge.log / run.json"
fi

cell cg_image "" 7200
cell pure_text "" 7200

# Count what landed: the driver never decides completeness from its own exit codes.
for split in id ood_single_family cg_image pure_text; do
  f="$OUT_ROOT/figure2_tom/$SLUG/$split/run.json"
  say "cell   $split: $($PY -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("produced_n"), "/", d.get("expected_n"), "complete=", d.get("complete"))' "$f" 2>/dev/null || echo absent)"
done
say "ToM cells for $MODEL finished; aggregates come from the exporter, not from here"
