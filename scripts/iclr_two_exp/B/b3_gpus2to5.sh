#!/bin/bash
# B3 phase 3: GPUs 2-5 join B3 after Part A's vLLM service stops.
# Waits by PID for A's server (argument), then requires GPUs 2-5 to carry no compute process for 5 consecutive minutes
# (A may start its next model; if so, wait by PID on that one too). Then on each GPU 2..5: one 8B vLLM
# (port 8200+g, util 0.35, 64k context) and two slot workers (habitat on the same GPU). GPUs 6/7 are not touched.
# Usage: b3_gpus2to5.sh <A_server_pid>
set -u
APID=$1
P=/mnt/pfs/devs/pn5wp/shishuqing
S=$P/partnr-planner/scripts/iclr_two_exp/B
log() { echo "$(date '+%F %T') $*"; }
UUIDS=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1>=2 && $1<=5 {print $2}' | tr '\n' '|' | sed 's/|$//')
while true; do
  log "waiting on A server pid $APID"
  while kill -0 $APID 2>/dev/null; do sleep 30; done
  quiet=0
  while [ $quiet -lt 10 ]; do
    others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -E "$UUIDS" | awk -F', ' '{print $2}' | sort -u)
    if [ -z "$others" ]; then quiet=$((quiet + 1)); else break; fi
    sleep 30
  done
  [ $quiet -ge 10 ] && break
  # something is on GPUs 2-5: follow the process group leader of the first one (A's next server) and wait again
  APID=$(ps -o pgid= -p $(echo "$others" | head -1) | tr -d ' ')
  log "GPUs 2-5 busy again (pids: $(echo $others)); now waiting on pgid leader $APID"
done
log "GPUs 2-5 free"
for g in 2 3 4 5; do bash $S/serve_8b.sh $g $((8200 + g)) 0.35 128 65536; done
for g in 2 3 4 5; do
  for i in $(seq 1 60); do curl -s -m 3 -o /dev/null http://127.0.0.1:$((8200 + g))/health && break; sleep 10; done
  curl -s -m 3 -o /dev/null http://127.0.0.1:$((8200 + g))/health || { log "endpoint $g did not come up"; continue; }
  log "endpoint gpu$g up"
  bash $S/launch_slot.sh $g $((8200 + g)) g${g}a; sleep 120
  bash $S/launch_slot.sh $g $((8200 + g)) g${g}b
done
log "GPUS2TO5 DONE"
