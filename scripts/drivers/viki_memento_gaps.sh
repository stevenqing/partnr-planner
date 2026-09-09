#!/usr/bin/env bash
# The four MEMENTO-style cells the cross-model table is missing: 30B and 7B on both
# recombination splits. `baseline_comparison.json` declares them itself under "missing".
#
# The 30B pair runs first, on 8062, which nothing else is using. The 7B pair waits for
# `viki_p0_7b.sh` to release 8061 -- one generation job per endpoint is a hard rule here.
# Scoring happens nowhere in this script; the report re-scores from raw responses.
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
A11="$ROOT/results/viki_memory_experiments/amendment11"
WORKERS=${WORKERS:-8}
cd "$ROOT" || exit 1
export TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

cell () {            # model base_url split tag
  local model=$1 base=$2 split=$3 tag=$4
  local out="$A11/$tag.jsonl"
  local have; have=$(wc -l < "$out" 2>/dev/null || echo 0)
  if [ "$have" -eq 297 ]; then say "skip   $tag -- 297 rows"; return 0; fi
  [ "$have" -gt 0 ] && { say "clear  $tag -- $have rows is not 297"; rm -f "$out"; }
  if ! curl -sf -m 10 "$base/models" | grep -q "$model"; then
    say "ABORT  $tag -- $base is not serving $model"; return 1
  fi
  say "start  $tag"
  timeout -k 60 7200 "$PY" scripts/viki_eval_memento.py --model "$model" --base-url "$base" \
      --split "$split" --workers "$WORKERS" --tag "$tag" >> "$ROOT/outputs/memento_gaps.detail.log" 2>&1 \
      && say "done   $tag ($(wc -l < "$out") rows)" || say "FAILED $tag"
}

cell qwen3-vl-30b http://127.0.0.1:8062/v1 recombination-text   memento_recomb_text_m30
cell qwen3-vl-30b http://127.0.0.1:8062/v1 recombination-imaged memento_recomb_imaged_m30

# 8061 is held by the 7B baseline batch; wait for it to exit before touching that endpoint.
say "waiting for viki_p0_7b.sh to release 8061"
while ps -eo cmd | grep -q "[v]iki_p0_7b.sh"; do sleep 120; done
say "8061 free"
cell qwen2.5-vl-7b http://127.0.0.1:8061/v1 recombination-text   memento_recomb_text_m7
cell qwen2.5-vl-7b http://127.0.0.1:8061/v1 recombination-imaged memento_recomb_imaged_m7
say "memento gap cells finished"
