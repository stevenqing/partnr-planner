#!/usr/bin/env bash
# Repeats of our own cells, because the table has none.
#
# The 30B baselines carry three repeats each; every `ours` cell is a single draw, so the
# table has no variance estimate on the arm it is arguing for. That is the first thing a
# reviewer asks about a 0.6126.
#
# A cell replays archived answers, so it is not free of sampling: 16-23% of rows fail the
# first plan and take a live re-ask (viki_eval_v2_intent_choice.py:291). That path is what
# a repeat measures. Everything else is deterministic, so a repeat costs ~150-210 calls
# rather than 924.
#
# Run:  MODEL=72B REP=2 setsid nohup bash scripts/drivers/viki_ours_repeats.sh \
#         > outputs/ours_rep72b_2.log 2>&1 < /dev/null &
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
MODEL=${MODEL:?set MODEL}
REP=${REP:-2}
WORKERS=${WORKERS:-8}
cd "$ROOT" || exit 1
export TOKENIZERS_PARALLELISM=false
A11=results/viki_memory_experiments/amendment11
EVAL=scripts/viki_eval_v2_intent_choice.py
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

case "$MODEL" in
  72B) BASE=http://192.168.32.40:8050/v1; SERVED=qwen2.5-vl-72b-amendment3-f2; REPLAY=intent_crew_clean ;;
  30B) BASE=http://127.0.0.1:8062/v1;     SERVED=qwen3-vl-30b;                 REPLAY=m30_id ;;
  7B)  BASE=http://127.0.0.1:8061/v1;     SERVED=qwen2.5-vl-7b;                REPLAY=m7_id ;;
  *) echo "MODEL must be 72B / 30B / 7B"; exit 1 ;;
esac
if ! curl -sf -m 10 "$BASE/models" | grep -q "$SERVED"; then
  say "ABORT: $BASE is not serving $SERVED"; exit 1
fi

# Replay sources differ per split for the 72B; the smaller models reuse their ID archive
# for the ID and fold cells, which is what the original cells did.
replay_for () {  # split
  case "$1" in
    id) echo "$A11/$REPLAY.jsonl" ;;
    recombination-text)   [ "$MODEL" = 72B ] && echo "$A11/recomb_text_agentic.jsonl"   || echo "$A11/${MODEL,,}_recomb_text.jsonl" ;;
    recombination-imaged) [ "$MODEL" = 72B ] && echo "$A11/recomb_imaged_agentic.jsonl" || echo "$A11/${MODEL,,}_recomb_imaged.jsonl" ;;
  esac
}

cell () {        # tag memory split
  local tag="$1_r$REP" memory=$2 split=$3
  local out="$A11/$tag.jsonl"
  [ -f "$out" ] && { say "skip   $tag"; return 0; }
  [ -f "$memory" ] || { say "未执行，缺 $memory"; return 1; }
  local src; src=$(replay_for "$split")
  [ -f "$src" ] || { say "未执行，缺 replay $src"; return 1; }
  say "start  $tag"
  timeout -k 60 10800 "$PY" "$EVAL" --memory "$memory" --split "$split" --tag "$tag" \
      --replay "$src" --model "$SERVED" --base-url "$BASE" --workers "$WORKERS" \
      >> "outputs/ours_repeats_${MODEL}.detail.log" 2>&1 \
      && say "done   $tag" || say "FAILED $tag"
}

M=outputs/v2_memories
cell "v2_ours_${MODEL}_id"        $M/memory_all.json               id
cell "v2_oursall_${MODEL}_text"   $M/memory_all.json               recombination-text
cell "v2_oursall_${MODEL}_imaged" $M/memory_all.json               recombination-imaged
for f in $($PY -c 'import sys;sys.path.insert(0,"scripts");sys.path.insert(0,".");
import viki_amendment9_folds as F;print(" ".join(F.folds()))' 2>/dev/null); do
  cell "v2_fold_${MODEL}_$f"    "$M/memory_heldout_$f.json"    id
done
for f in $($PY -c 'import json;print(" ".join(json.load(open("results/sibling_folds_preregistration.json"))["affected_eval_folds"]))'); do
  cell "v2_foldgrp_${MODEL}_$f" "$M/memory_heldoutgrp_$f.json" id
done
say "repeat $REP finished for $MODEL"
