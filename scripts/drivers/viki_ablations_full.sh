#!/usr/bin/env bash
# The two ablations on every model, and on the memory the main table actually reports.
#
# What existed: 72B only, and its text/imaged cells ran on the HALF memory (`comp_cd`),
# so the published ablation never touched the full library the headline numbers come from.
# Two things are produced here:
#
#   *_full   the ablation against `memory_all.json` -- the memory the main table reports.
#   *_half   30B and 7B against `comp_cd`, so the archived 72B half-memory cells finally
#            have the other two models beside them.
#
# Cells replay archived answers, so this costs re-ask calls only.
#
# Run:  MODEL=30B setsid nohup bash scripts/drivers/viki_ablations_full.sh \
#         > outputs/abl_30b.log 2>&1 < /dev/null &
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
MODEL=${MODEL:?set MODEL}
WORKERS=${WORKERS:-8}
cd "$ROOT" || exit 1
export TOKENIZERS_PARALLELISM=false
A11=results/viki_memory_experiments/amendment11
# Which build's memories the ablation runs against, and the cell-name prefix that goes with
# it. Defaults keep every path this driver has ever written.
M=${M:-outputs/v2_memories}
TAG=${TAG:-v2}
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

case "$MODEL" in
  72B) BASE=http://192.168.32.40:8050/v1; SERVED=qwen2.5-vl-72b-amendment3-f2
       R_ID=$A11/intent_crew_clean.jsonl; R_T=$A11/recomb_text_agentic.jsonl; R_I=$A11/recomb_imaged_agentic.jsonl ;;
  30B) BASE=http://127.0.0.1:8062/v1; SERVED=qwen3-vl-30b
       R_ID=$A11/m30_id.jsonl; R_T=$A11/m30_recomb_text.jsonl; R_I=$A11/m30_recomb_imaged.jsonl ;;
  7B)  BASE=http://127.0.0.1:8061/v1; SERVED=qwen2.5-vl-7b
       R_ID=$A11/m7_id.jsonl; R_T=$A11/m7_recomb_text.jsonl; R_I=$A11/m7_recomb_imaged.jsonl ;;
  *) echo "MODEL must be 72B / 30B / 7B"; exit 1 ;;
esac
if ! curl -sf -m 10 "$BASE/models" | grep -q "$SERVED"; then say "ABORT: $BASE not serving $SERVED"; exit 1; fi

cell () {        # tag memory split replay flag
  local tag=$1 memory=$2 split=$3 replay=$4 flag=$5
  local out="$A11/$tag.jsonl"
  [ -f "$out" ] && { say "skip   $tag"; return 0; }
  [ -f "$memory" ] || { say "未执行，缺 $memory"; return 1; }
  [ -f "$replay" ] || { say "未执行，缺 replay $replay"; return 1; }
  say "start  $tag"
  timeout -k 60 10800 "$PY" scripts/viki_eval_v2_intent_choice.py --memory "$memory" \
      --split "$split" --tag "$tag" --replay "$replay" --model "$SERVED" --base-url "$BASE" \
      --workers "$WORKERS" "$flag" >> "outputs/abl_${MODEL}.detail.log" 2>&1 \
      && say "done   $tag" || say "FAILED $tag"
}

for ab in noground noorder; do
  [ "$ab" = noground ] && flag=--no-grounding || flag=--no-order
  # full memory: the one the main table reports, for all three models
  cell "${TAG}_ablfull_${ab}_${MODEL}_id"     $M/memory_all.json id                   "$R_ID" "$flag"
  cell "${TAG}_ablfull_${ab}_${MODEL}_text"   $M/memory_all.json recombination-text   "$R_T"  "$flag"
  cell "${TAG}_ablfull_${ab}_${MODEL}_imaged" $M/memory_all.json recombination-imaged "$R_I"  "$flag"
  # half memory, to sit beside the archived 72B cells (which were produced this way)
  if [ "$MODEL" != 72B ]; then
    cell "${TAG}_abl_${ab}_${MODEL}_id"     $M/memory_all.json               id                   "$R_ID" "$flag"
    cell "${TAG}_abl_${ab}_${MODEL}_text"   $M/memory_comp_cut_delivery.json recombination-text   "$R_T"  "$flag"
    cell "${TAG}_abl_${ab}_${MODEL}_imaged" $M/memory_comp_cut_delivery.json recombination-imaged "$R_I"  "$flag"
  fi
done
say "ablations finished for $MODEL"
