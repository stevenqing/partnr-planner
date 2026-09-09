#!/usr/bin/env bash
# Rerun of both model arms after the 22:08 double failure.
#
# What went wrong the first time, and what this changes:
#   1. HARD_TIMEOUT was 14400 (4h). With four cells x 45 procs sharing 180 cores the 7B pair
#      only reached 334/369 and 340/369 and was killed mid-split. -> 28800 here.
#   2. The Qwen3-8B server on 8101 was shut down at 22:08:08 and the 8B pair kept running
#      against nothing, writing 46 zero-scored episodes (sim_step_count == 0, runtime ~13s).
#      -> this script probes the endpoint before each pair, and the pairs run one after the
#         other so each gets the whole box instead of a quarter of it.
# The two contaminated 8B directories are in outputs/stale/*_deadendpoint_0907_2220.
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
LIB=results/partnr_operators_iir1.json
BASE=results/partnr_operators.json
POOL=val_mini
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

# The band cell owns the box until it is done; it is privileged-arm and touches no endpoint.
until [ -f outputs/report/priv_iir1/base_rep/CELL.json ]; do sleep 60; done
say "band cell done; starting the model reruns"

pair () {           # $1 tag  $2 model  $3 url  $4 think-flag-value  $5 gpu
  local tag=$1 model=$2 url=$3 nothink=$4 gpu=$5
  local code
  code=$(curl -s -m 10 -o /dev/null -w "%{http_code}" "$url/models")
  if [ "$code" != "200" ]; then say "SKIP $tag -- $url returned $code"; return 1; fi
  say "$tag base+accepted starting (endpoint $code)"
  ( MODEL=$model URL=$url NO_THINK=$nothink OPERATORS=$BASE PROCS=45 GPU=$gpu HARD_TIMEOUT=28800 \
      bash scripts/drivers/partnr_model_cell.sh $POOL ${tag}_base     outputs/report/model_rerun ) &
  local a=$!
  ( MODEL=$model URL=$url NO_THINK=$nothink OPERATORS=$LIB  PROCS=45 GPU=$gpu HARD_TIMEOUT=28800 \
      bash scripts/drivers/partnr_model_cell.sh $POOL ${tag}_accepted outputs/report/model_rerun ) &
  local b=$!
  wait $a $b
  /root/venvs/partnr/bin/python scripts/partnr_gate_compare.py --pool $POOL \
      --a "outputs/report/model_rerun/${tag}_base" --b "outputs/report/model_rerun/${tag}_accepted" \
      --json "outputs/report/model_rerun/compare_${tag}_${POOL}.json" \
      > "outputs/report/model_rerun/compare_${tag}_${POOL}.txt" 2>&1
  say "$tag done -- check 'complete' in compare_${tag}_${POOL}.json before reading any number"
}

# Both pairs render on GPU 0, the only free card: GPU 1 serves the 8B, 2-5 the 30B, 7 the
# 7B, and GPU 6 carries another project's vLLM on port 8110 (17h old, not ours to move).
# PROCS drops from 80 to 45 accordingly -- a habitat worker holds ~750 MB, so a pair at 80
# would ask for ~120 GB on a 94 GB card. 45 x 2 is ~68 GB and still saturates the 180 cores.
pair m7b qwen2.5-vl-7b http://127.0.0.1:8061/v1 0 0
pair m8b Qwen3-8B      http://127.0.0.1:8101/v1 1 0
say "both model arms rerun on $POOL"
