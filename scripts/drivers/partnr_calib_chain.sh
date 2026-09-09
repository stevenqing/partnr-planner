#!/usr/bin/env bash
# Two-way calibration of the execution gate: can it tell a correct operator from one
# broken on purpose? The trace-matching gate could not, at any threshold.
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
for v in no_is_on_top is_on_top_factory is_on_top_reversed is_on_top_keyswap; do
  GPU=1 PROCS=40 HARD_TIMEOUT=3600 OPERATORS="results/partnr_calib/$v.json" \
    bash scripts/drivers/partnr_gate_cell.sh calib_ontop "$v" outputs/calib
done
echo "calibration cells finished"
