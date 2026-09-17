#!/bin/bash
# HANDOVER-2026-09-17 §4 priority 1: is no_casting's +11.7 a 72B-only effect? Same zero-GPU replay on 30B.
# Four arms + the reference-library arm, both splits, all in parallel (pure CPU). `full` must reproduce
# the published 0.5736 / 0.3539 or the script writes ALARM into the json.
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
O=outputs/viki_ablation
REF=results/viki_memory_experiments/amendment11/skill_memory_v2.json
pids=()
for s in id fold; do
  $PY scripts/viki_ablation_replay.py --model 30B --split $s \
      --json $O/v3_${s}_30B.json --md $O/ABLATION_v3_${s}_30B.md > $O/v3_${s}_30B.log 2>&1 &
  pids+=($!)
  $PY scripts/viki_ablation_replay.py --model 30B --split $s --arms full --library $REF \
      --json $O/v3_lib_${s}_30B.json --md $O/ABLATION_v3_lib_${s}_30B.md > $O/v3_lib_${s}_30B.log 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
echo "ABLATION_30B_DONE rc=$rc $(date +%T)" >> $O/v3_30B_chain.log
