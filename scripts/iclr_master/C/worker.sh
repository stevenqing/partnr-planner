#!/bin/bash
# GPU worker for the C queue (replaces lane.sh). Cells are claimed from queue.txt with mkdir, so any number of
# workers can share one queue. Per-cell settings are read from gpu<g>.conf at each cell start:
#   NP=<num_proc>   EXTRA_OVERRIDES="<hydra overrides>"
# Main cells start only after every smoke cell passed its gate. touch STOP_gpu<g> to stop after the current cell.
# Usage: worker.sh <gpu> [<adopt_cell_name> <adopt_pid>]   (adopt: a cell already running on this GPU)
set -u
GPU=$1; ADOPT=${2:-}; APID=${3:-}
P=/mnt/pfs/devs/pn5wp/shishuqing
S=$P/partnr-planner/scripts/iclr_master/C
REPO=$P/partnr-isambard-C
O=$REPO/outputs/iclr_master_C
PY=/root/venvs/partnr/bin/python
CAP=1700; TMO=108000; N=98
mkdir -p $O/claims
log() { echo "$(date '+%F %T') gpu$GPU $*" >> $O/queue.log; }
smoke_ok() {
  for c in C0 C1 C2 C3 C4; do
    st=$($PY $S/cell_status.py $O/smoke_${c}_s0 5 2>/dev/null) || return 1
    echo "$st" | $PY -c "import json,sys;x=json.load(sys.stdin);sys.exit(0 if x['fake']==0 and x['done']>=3 and x['dumps']>=2*x['done'] else 1)" || return 1
  done
}
run_until_complete() {  # $1 = cell name like C1_s0
  local name=$1 c=${1%%_s*} s=${1##*_s}
  for try in 1 2 3; do
    st=$($PY $S/cell_status.py $O/$name $N 2>/dev/null)
    echo "$st" | grep -q '"complete": true' && { log "complete $name $st"; return 0; }
    used=$(ls $O/*/hydra/results/*.json.gz/stats/*.json 2>/dev/null | grep -v /_stale/ | wc -l)
    [ $((used + N)) -gt $CAP ] && { log "CAP $used used, not starting $name"; exit 3; }
    NP=2; EXTRA_OVERRIDES=""; [ -f $O/gpu$GPU.conf ] && . $O/gpu$GPU.conf
    log "start $name try $try NP=$NP EXTRA='$EXTRA_OVERRIDES'"
    EXTRA_OVERRIDES="$EXTRA_OVERRIDES" bash $S/run_cell2.sh $REPO $O/$name $GPU $NP task_classification_datasets/hr_eval_half.json.gz $c $s $TMO
    log "end $name try $try $($PY $S/cell_status.py $O/$name $N)"
  done
}
if [ -n "$ADOPT" ]; then
  mkdir -p $O/claims/$ADOPT 2>/dev/null; echo gpu$GPU > $O/claims/$ADOPT/owner
  log "adopt $ADOPT pid $APID"
  while kill -0 $APID 2>/dev/null; do sleep 60; done
  case $ADOPT in smoke_*) ;; *) run_until_complete $ADOPT ;; esac
fi
until smoke_ok; do
  ps -eo cmd | grep -q "[r]un_cell.sh .*smoke_" || { log "SMOKE GATE FAILED, worker stops"; exit 4; }
  sleep 120
done
while read -r name; do
  [ -z "$name" ] && continue
  [ -f $O/STOP_gpu$GPU ] && { log "STOP file, worker exits"; exit 0; }
  mkdir $O/claims/$name 2>/dev/null || continue
  echo gpu$GPU > $O/claims/$name/owner
  run_until_complete $name
done < $S/queue.txt
log "worker finished"
