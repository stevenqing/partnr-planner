#!/usr/bin/env bash
# Confirmation and the reported splits, on the library acceptance produced.
# Cells are run side by side across the free GPUs; process counts are kept below the level
# at which memory ran out. GPU 1 is NOT one of them: a vLLM server (Qwen3-8B, port 8101)
# holds 0.88 of it, and the sim cells that landed there lost most of their workers to
# `cache.pin_memory(): CUDA error: out of memory` -- twice, before the cause was found.
# Check what is on a GPU before putting 60 habitat processes on it.
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
LIB=results/partnr_operators_iir1.json
BASE=results/partnr_operators.json
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

# 1. the confirmation pool, opened once
say "conf_iir"
( OPERATORS=$BASE PROCS=45 GPU=0 HARD_TIMEOUT=5400 bash scripts/drivers/partnr_gate_cell.sh conf_iir base     outputs/confirm/iir1 ) &
p1=$!
( OPERATORS=$LIB  PROCS=45 GPU=6 HARD_TIMEOUT=5400 bash scripts/drivers/partnr_gate_cell.sh conf_iir accepted outputs/confirm/iir1 ) &
p2=$!
wait $p1 $p2
/root/venvs/partnr/bin/python scripts/partnr_gate_compare.py --pool conf_iir \
    --a outputs/confirm/iir1/base --b outputs/confirm/iir1/accepted \
    --json outputs/confirm/iir1/compare.json > outputs/confirm/iir1/compare.txt 2>&1
say "conf_iir done"

# 2. privileged arm on val_mini
say "val_mini privileged"
( OPERATORS=$BASE PROCS=60 GPU=0 HARD_TIMEOUT=10800 bash scripts/drivers/partnr_gate_cell.sh val_mini base     outputs/report/priv_iir1 ) &
p1=$!
( OPERATORS=$LIB  PROCS=60 GPU=6 HARD_TIMEOUT=10800 bash scripts/drivers/partnr_gate_cell.sh val_mini accepted outputs/report/priv_iir1 ) &
p2=$!
wait $p1 $p2
/root/venvs/partnr/bin/python scripts/partnr_gate_compare.py --pool val_mini \
    --a outputs/report/priv_iir1/base --b outputs/report/priv_iir1/accepted \
    --json outputs/report/priv_iir1/compare_val_mini.json > outputs/report/priv_iir1/compare_val_mini.txt 2>&1
say "val_mini privileged done"
