#!/bin/bash
# Per-layer post-answer ablation (the user ruled no_casting out as a reported arm, 09-17).
# 72B and 30B, both splits, in parallel (pure CPU). `full` must reproduce the published cell.
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
O=outputs/viki_ablation
pids=()
for m in ${MODELS:-72B 30B}; do
  for s in id fold; do
    $PY scripts/viki_ablation_replay.py --model $m --split $s \
        --json $O/v3_layers_${s}_${m}.json --md $O/ABLATION_v3_layers_${s}_${m}.md > $O/v3_layers_${s}_${m}.log 2>&1 &
    pids+=($!)
  done
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
echo "ABLATION_LAYERS_DONE rc=$rc $(date +%T)" >> $O/v3_layers_chain.log
