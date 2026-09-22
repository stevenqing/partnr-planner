#!/bin/bash
# P arm, both CG splits in sequence on the local 72B endpoint (one generation job per endpoint).
# Hard timeout per split + stall guard (output jsonl not growing for 20 min -> kill).
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
export A8B_SKILL_TOPK=8 A9_ROLE_AWARE=0 A9_PATTERN_SLOTS=0
PY=/root/venvs/partnr/bin/python
LOGDIR=results/incontext_library_2026-09-21/work/B
mkdir -p "$LOGDIR"
code=$(curl -s -m 5 -o /dev/null -w %{http_code} http://127.0.0.1:8050/v1/models)
[ "$code" = 200 ] || { echo "endpoint not up ($code)"; exit 1; }
for split in text imaged; do
  out=results/incontext_library_2026-09-21/runs/$split/incontext_library.jsonl
  timeout 5400 $PY scripts/incontext_library/run_p_arm.py --split $split > "$LOGDIR/run_$split.log" 2>&1 &
  pid=$!
  last=-1; since=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    n=$(wc -l < "$out" 2>/dev/null || echo 0)
    if [ "$n" != "$last" ]; then last=$n; since=$(date +%s)
    elif [ $(( $(date +%s) - since )) -ge 1200 ]; then
      echo "STALL $split at $n rows" >> "$LOGDIR/guard.txt"; kill $pid; break
    fi
  done
  wait $pid; echo "$split rc=$? rows=$(wc -l < "$out" 2>/dev/null)" >> "$LOGDIR/exit.txt"
done
echo done >> "$LOGDIR/exit.txt"
