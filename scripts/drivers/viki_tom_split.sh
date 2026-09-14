#!/usr/bin/env bash
# Figure 2 ToM arm, a chosen subset of splits against a chosen endpoint -- so the four splits of
# one model can run in parallel on several replicas of the same served model.
# Same runner, guards and output layout as viki_tom_cells.sh (which runs all four in sequence).
#
#   MODEL=7B SPLITS="cg_image" BASE=http://127.0.0.1:8064/v1 \
#     setsid nohup bash scripts/drivers/viki_tom_split.sh > outputs/paper_viki_0914/tom_7b_cg.log 2>&1 < /dev/null &
#
# Never point two launches at the same split: they share the run directory.
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
MODEL=${MODEL:?set MODEL to 72B, 30B or 7B}
SPLITS=${SPLITS:?set SPLITS, e.g. "cg_image pure_text"}
WORKERS=${WORKERS:-8}
STALL_MIN=${STALL_MIN:-20}
OUT_ROOT=${OUT_ROOT:-$ROOT/results/paper_viki_iclr2027/raw}

case "$MODEL" in
  72B) BASE=${BASE:?set BASE for 72B}; SERVED=qwen2.5-vl-72b-amendment3-f2
       BACKBONE=qwen2_5_vl_72b; SLUG=qwen2.5-vl-72b-instruct ;;
  30B) BASE=${BASE:-http://127.0.0.1:8062/v1}; SERVED=qwen3-vl-30b
       BACKBONE=qwen3_vl_30b;   SLUG=qwen3-vl-30b-a3b-instruct ;;
  7B)  BASE=${BASE:-http://127.0.0.1:8061/v1}; SERVED=qwen2.5-vl-7b
       BACKBONE=qwen2_5_vl_7b;  SLUG=qwen2.5-vl-7b-instruct ;;
  *)   echo "MODEL must be 72B, 30B or 7B"; exit 1 ;;
esac

cd "$ROOT" || exit 1
unset VIKI_NO_THINK A9_SIBLINGS
export TOKENIZERS_PARALLELISM=false
export VIKI_BACKBONE=$BACKBONE VIKI_SERVED_MODEL=$SERVED
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

# Wait up to 30 min for a freshly started replica to come up.
for _ in $(seq 1 180); do probe >/dev/null 2>&1 && break; sleep 10; done
probe || exit 1
say "model=$MODEL base=$BASE slug=$SLUG splits=$SPLITS workers=$WORKERS"

for split in $SPLITS; do
  case "$split" in
    id) cell id "" 10800 ;;
    cg_image|pure_text) cell "$split" "" 7200 ;;
    ood_single_family)
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
      fi ;;
    *) say "unknown split $split" ;;
  esac
done

for split in $SPLITS; do
  f="$OUT_ROOT/figure2_tom/$SLUG/$split/run.json"
  say "cell   $split: $($PY -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("produced_n"), "/", d.get("expected_n"), "complete=", d.get("complete"))' "$f" 2>/dev/null || echo absent)"
done
say "ToM $MODEL [$SPLITS] finished; aggregates come from the exporter"
