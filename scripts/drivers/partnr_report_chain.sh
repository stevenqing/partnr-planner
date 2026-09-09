#!/usr/bin/env bash
# The reported cells. Run only after acceptance has closed: this touches `val_mini` and
# `val`, and `val_mini` is a subset of `val`, so anything selected here contaminates both.
#
#   base        the R-only memory as it stands, 21 operators, no is_in_room entry
#   accepted    the same plus whatever the execution gate accepted on `gate_iir`
#   conf        the accepted library on `conf_iir`, a train pool opened exactly once
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PY=/root/venvs/partnr/bin/python
TAG=${TAG:-iir1}
LIB="results/partnr_operators_${TAG}.json"
[ -f "$LIB" ] || { echo "no accepted library at $LIB"; exit 1; }

say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

say "confirmation pool (opened once)"
OPERATORS="$LIB" PROCS=60 HARD_TIMEOUT=5400 GPU=${GPU:-1} \
  bash scripts/drivers/partnr_gate_cell.sh conf_iir "accepted" "outputs/confirm/$TAG"
OPERATORS=results/partnr_operators.json PROCS=60 HARD_TIMEOUT=5400 GPU=${GPU:-1} \
  bash scripts/drivers/partnr_gate_cell.sh conf_iir "base" "outputs/confirm/$TAG"
$PY scripts/partnr_gate_compare.py --pool conf_iir \
    --a "outputs/confirm/$TAG/base" --b "outputs/confirm/$TAG/accepted" \
    --json "outputs/confirm/$TAG/compare.json"

for split in val_mini val; do
  say "reported split $split"
  OPERATORS=results/partnr_operators.json PROCS=60 HARD_TIMEOUT=14400 GPU=${GPU:-1} \
    bash scripts/drivers/partnr_gate_cell.sh "$split" "base" "outputs/report/$TAG"
  OPERATORS="$LIB" PROCS=60 HARD_TIMEOUT=14400 GPU=${GPU:-1} \
    bash scripts/drivers/partnr_gate_cell.sh "$split" "accepted" "outputs/report/$TAG"
  $PY scripts/partnr_gate_compare.py --pool "$split" \
      --a "outputs/report/$TAG/base" --b "outputs/report/$TAG/accepted" \
      --json "outputs/report/$TAG/compare_$split.json"
done
say "reported cells finished"
