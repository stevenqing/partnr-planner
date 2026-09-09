#!/usr/bin/env bash
# Wave 2: the two gaps that needed a code change first.
#
#   72B  MEMENTO on the sibling-grouped folds. `viki_memento_rag.build` now takes a group
#        (comma-separated or a list); the single-family form is unchanged and every
#        archived graph still builds identically. Verified against known inputs:
#        2698 user_pattern nodes with nothing held out, 2448 for one family, 2258 for the
#        pair.
#   30B  our own arm under no-think, the condition the 30B baselines also have. The
#        transformation is `drop_think_rule` imported from the baseline harness, not a
#        rewrite, so the two columns are the same condition. These cells are generated
#        live (no --replay): the archived answers were produced with thinking on.
#   7B   the same, matching the `m7nt` baselines wave 1 is producing.
#
# Each part waits for its own endpoint to be free -- one generation job per endpoint.
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
PART=${PART:?set PART to 72B / 30B / 7B}
WORKERS=${WORKERS:-8}
cd "$ROOT" || exit 1
export TOKENIZERS_PARALLELISM=false
A11=results/viki_memory_experiments/amendment11
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

# Wait on the wave-1 chain for this endpoint by its log name, matched on the shell's own
# argv is exactly what wedges supervisors here, so match the driver file names instead and
# accept that this shell is not one of them.
wait_free () {
  local pattern=$1
  while ps -eo cmd | grep -qE "$pattern"; do sleep 120; done
}

case "$PART" in
  72B)
    wait_free "[v]iki_ours_repeats.sh|[v]iki_shuffled_folds.sh"
    say "8050 free; MEMENTO on the sibling-grouped folds"
    SERVED=qwen2.5-vl-72b-amendment3-f2; BASE=http://192.168.32.40:8050/v1
    $PY - <<'PYEOF' > /tmp/sibgroups.txt
import json
pre = json.load(open("results/sibling_folds_preregistration.json"))
groups = {f: g for g in pre["groups"] for f in g}
for family in pre["affected_eval_folds"]:
    print("%s\t%s" % (family, ",".join(groups[family])))
PYEOF
    while IFS=$'\t' read -r family group; do
      tag="memento_foldgrp_$family"
      out="$A11/$tag.jsonl"
      [ -f "$out" ] && { say "skip   $tag"; continue; }
      say "start  $tag (hiding $group)"
      timeout -k 60 10800 $PY scripts/viki_eval_memento.py --model "$SERVED" --base-url "$BASE" \
          --split id --workers "$WORKERS" --exclude-family "$group" --only-family "$family" \
          --tag "$tag" >> outputs/wave2_72b.detail.log 2>&1 \
          && say "done   $tag -> $(wc -l < "$out" 2>/dev/null || echo 0) rows" || say "FAILED $tag"
    done < /tmp/sibgroups.txt
    ;;
  30B|7B)
    wait_free "[v]iki_ours_repeats.sh|[v]iki_shuffled_folds.sh|[v]iki_p0_7b.sh"
    if [ "$PART" = 30B ]; then BASE=http://127.0.0.1:8062/v1; SERVED=qwen3-vl-30b
    else BASE=http://127.0.0.1:8061/v1; SERVED=qwen2.5-vl-7b; fi
    if ! curl -sf -m 10 "$BASE/models" | grep -q "$SERVED"; then say "ABORT: $BASE not serving $SERVED"; exit 1; fi
    export VIKI_NO_THINK=1
    say "$BASE free; our arm under no-think"
    for split in id recombination-text recombination-imaged; do
      case $split in
        id) tag="v2_ours_${PART}_id_nt" ;;
        recombination-text) tag="v2_oursall_${PART}_text_nt" ;;
        recombination-imaged) tag="v2_oursall_${PART}_imaged_nt" ;;
      esac
      out="$A11/$tag.jsonl"
      [ -f "$out" ] && { say "skip   $tag"; continue; }
      say "start  $tag (live, no replay)"
      timeout -k 60 21600 $PY scripts/viki_eval_v2_intent_choice.py \
          --memory outputs/v2_memories/memory_all.json --split "$split" --tag "$tag" \
          --model "$SERVED" --base-url "$BASE" --workers "$WORKERS" \
          >> "outputs/wave2_${PART}.detail.log" 2>&1 \
          && say "done   $tag -> $(wc -l < "$out" 2>/dev/null || echo 0) rows" || say "FAILED $tag"
    done
    ;;
esac
say "wave 2 part $PART finished"
