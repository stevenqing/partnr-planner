#!/bin/bash
# One GPU lane of the C queue: cells one after another, each resumed until complete (max 3 tries).
# Entries: <cond>:<seed> for eval-half cells (98 episodes), smoke:<cond>:<seed> for 5-episode smoke cells.
# A smoke cell must come back with every started episode scored or step-limited and two retrieval dumps per
# episode; otherwise the lane stops before any eval-half cell.
# Usage: lane.sh <gpu> <num_proc> <timeout_s> <entry> [<entry> ...]
set -u
GPU=$1; NP=$2; TMO=$3; shift 3
P=/mnt/pfs/devs/pn5wp/shishuqing
S=$P/partnr-planner/scripts/iclr_master/C
REPO=$P/partnr-isambard-C
O=$REPO/outputs/iclr_master_C
CAP=1700
PY=/root/venvs/partnr/bin/python
for e in "$@"; do
  if [ "${e%%:*}" = smoke ]; then r=${e#smoke:}; c=${r%%:*}; s=${r##*:}; cell=$O/smoke_${c}_s$s; DATA=task_classification_datasets/hr_eval_smoke5.json.gz; N=5; tries=1
  else c=${e%%:*}; s=${e##*:}; cell=$O/${c}_s$s; DATA=task_classification_datasets/hr_eval_half.json.gz; N=98; tries=3; fi
  for try in $(seq 1 $tries); do
    st=$($PY $S/cell_status.py $cell $N 2>/dev/null)
    echo "$st" | grep -q '"complete": true' && break
    used=$(ls $O/*/hydra/results/*.json.gz/stats/*.json 2>/dev/null | grep -v /_stale/ | wc -l)
    if [ $((used + N)) -gt $CAP ]; then echo "$(date '+%F %T') CAP: $used episodes used, not starting $e" >> $O/queue.log; exit 3; fi
    echo "$(date '+%F %T') gpu$GPU start $e try $try" >> $O/queue.log
    bash $S/run_cell.sh $REPO $cell $GPU $NP $DATA $c $s $TMO
    echo "$(date '+%F %T') gpu$GPU end $e try $try $($PY $S/cell_status.py $cell $N)" >> $O/queue.log
  done
  if [ $N = 5 ]; then
    st=$($PY $S/cell_status.py $cell $N)
    d=$(echo "$st" | $PY -c "import json,sys;x=json.load(sys.stdin);print(x['done'], x['dumps'], x['fake'])")
    set -- $d; dn=$1; du=$2; fk=$3
    if [ "$fk" != 0 ] || [ "$du" -lt $((2 * dn)) ] || [ "$dn" -lt 3 ]; then
      echo "$(date '+%F %T') gpu$GPU SMOKE FAILED $e: $st -- lane stops" >> $O/queue.log; exit 4
    fi
    shift $#
  fi
done
echo "$(date '+%F %T') gpu$GPU lane finished" >> $O/queue.log
