#!/bin/bash
# Wait until all 18 B3 cells are complete (cell_status.py "complete": true, never exit codes), then run b4_metrics.py
# (final, no --allow-incomplete), then stop every B slot worker and every 8B vLLM of Part B by PID / process group.
set -u
P=/mnt/pfs/devs/pn5wp/shishuqing
O=$P/partnr-isambard-C/outputs/iclr_two_exp_B
S=$P/partnr-planner/scripts/iclr_two_exp/B
C=$P/partnr-planner/scripts/iclr_master/C/cell_status.py
PY=/root/venvs/partnr/bin/python
while true; do
  n=0
  for s in 0 1 2; do for c in a0 a1 both; do for f in A B; do
    e=90; [ $f = B ] && e=107
    $PY $C $O/s${s}_${c}_${f} $e 2>/dev/null | grep -q '"complete": true' && n=$((n + 1))
  done; done; done
  echo "$(date '+%F %T') complete cells $n/18"
  [ $n -eq 18 ] && break
  sleep 300
done
$PY $S/b4_metrics.py > $O/b4_final.json 2> $O/b4_final.err
echo "$(date '+%F %T') b4 rc=$?"
for f in $O/slot_*.pid; do p=$(cat $f); kill -0 $p 2>/dev/null && { kill -TERM $p; echo "stopped slot $f $p"; }; done
for f in $P/iclr_two_exp_B_run/vllm8b_*.pid; do
  p=$(cat $f); kill -0 $p 2>/dev/null || continue
  kill -TERM -- -$p 2>/dev/null; sleep 20; kill -0 $p 2>/dev/null && kill -KILL -- -$p
  echo "stopped vllm $f $p"
done
echo "FINISH DONE"
