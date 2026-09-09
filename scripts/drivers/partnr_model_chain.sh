#!/usr/bin/env bash
# The reported arm, on both small models the box serves. Starts when the privileged
# val_mini pair is done so the GPUs are not oversubscribed.
#
# GPU 1 is excluded: the Qwen3-8B server lives there.
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
LIB=results/partnr_operators_iir1.json
BASE=results/partnr_operators.json
POOL=${POOL:-val_mini}
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

until [ -f outputs/report/priv_iir1/base/CELL.json ] && [ -f outputs/report/priv_iir1/accepted/CELL.json ]; do
  sleep 60
done
say "privileged $POOL finished; starting the model arms"

( MODEL=qwen2.5-vl-7b URL=http://127.0.0.1:8061/v1 OPERATORS=$BASE PROCS=45 GPU=0 \
    bash scripts/drivers/partnr_model_cell.sh $POOL m7b_base     outputs/report/model_iir1 ) &
a=$!
( MODEL=qwen2.5-vl-7b URL=http://127.0.0.1:8061/v1 OPERATORS=$LIB  PROCS=45 GPU=0 \
    bash scripts/drivers/partnr_model_cell.sh $POOL m7b_accepted outputs/report/model_iir1 ) &
b=$!
( MODEL=Qwen3-8B URL=http://127.0.0.1:8101/v1 NO_THINK=1 OPERATORS=$BASE PROCS=45 GPU=6 \
    bash scripts/drivers/partnr_model_cell.sh $POOL m8b_base     outputs/report/model_iir1 ) &
c=$!
( MODEL=Qwen3-8B URL=http://127.0.0.1:8101/v1 NO_THINK=1 OPERATORS=$LIB  PROCS=45 GPU=6 \
    bash scripts/drivers/partnr_model_cell.sh $POOL m8b_accepted outputs/report/model_iir1 ) &
d=$!
wait $a $b $c $d

for m in m7b m8b; do
  /root/venvs/partnr/bin/python scripts/partnr_gate_compare.py --pool $POOL \
      --a "outputs/report/model_iir1/${m}_base" --b "outputs/report/model_iir1/${m}_accepted" \
      --json "outputs/report/model_iir1/compare_${m}_${POOL}.json" \
      > "outputs/report/model_iir1/compare_${m}_${POOL}.txt" 2>&1
done
say "model arms on $POOL finished"
