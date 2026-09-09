#!/usr/bin/env bash
# The paired band on the reported split itself.
#
# Two identical cells on a 40-episode train pool agreed exactly, twice, and that was taken
# to mean the privileged arm is deterministic. On `val_mini` it is not: the same library at
# the same commit moved 38 of 366 episodes between 09-05 and today, and the PARTNR eval
# path has not changed since (the only commit touching `our_method` after it edits VIKI's
# `memory.py`, which `partnr_planner` does not import). So the band has to be measured on
# the split a number is reported on, not on a convenient probe.
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
# Waits on the 7B pair only. This cell is privileged-arm -- it calls no model and does not
# touch an endpoint -- so it needs the CPUs the 7B pair frees, not the 8B pair's.
until [ -f outputs/report/model_iir1/m7b_base/CELL.json ] && [ -f outputs/report/model_iir1/m7b_accepted/CELL.json ]; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] model arms done; measuring the val_mini band"
OPERATORS=results/partnr_operators.json PROCS=60 GPU=0 HARD_TIMEOUT=14400 \
  bash scripts/drivers/partnr_gate_cell.sh val_mini base_rep outputs/report/priv_iir1
/root/venvs/partnr/bin/python scripts/partnr_gate_compare.py --pool val_mini \
    --a outputs/report/priv_iir1/base --b outputs/report/priv_iir1/base_rep \
    --json outputs/report/priv_iir1/band_val_mini.json > outputs/report/priv_iir1/band_val_mini.txt 2>&1
echo "[$(date +%H:%M:%S)] band measured"
