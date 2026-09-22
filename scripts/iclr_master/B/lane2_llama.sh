#!/bin/bash
# GPU 2 lane, llama8b cells only (replaces the first unfiltered lane 2, 09-22 02:36).
# Usage: lane2_llama.sh [pid of an orphaned run_cell.sh to wait for] [its cell id]
# If that cell ends not-done, its claim is released so this lane resumes it (resume=True).
PFS=/mnt/pfs/devs/pn5wp/shishuqing
HERE=$PFS/partnr-planner/scripts/iclr_master/B
O=$PFS/partnr-isambard/outputs/iclr_master/B
WAITPID=${1:-} WAITCELL=${2:-}
if [ -n "$WAITPID" ]; then
  while kill -0 $WAITPID 2>/dev/null; do sleep 60; done
  if ! grep -q '^status: done' $O/$WAITCELL/exit.txt 2>/dev/null; then
    echo "$(date -Is) lane 2: $WAITCELL ended not-done, releasing claim for resume" >> $O/lane_2.log
    rm -rf $O/$WAITCELL/.claim
  fi
fi
bash $HERE/queue.sh 2 2000 2 llama8b 70000
