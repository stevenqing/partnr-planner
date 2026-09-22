#!/bin/bash
# Restart a queue lane on a new cells.tsv order without killing the cell it is running.
# Usage: lane_restart.sh <gpu> <orphan run_cell.sh pid> <its cell id>
# Waits for the orphan cell; if it ended not-done, releases its claim so the new lane resumes it.
PFS=/mnt/pfs/devs/pn5wp/shishuqing
HERE=$PFS/partnr-planner/scripts/iclr_master/B
O=$PFS/partnr-isambard/outputs/iclr_master/B
GPU=$1 WAITPID=$2 WAITCELL=$3
while kill -0 $WAITPID 2>/dev/null; do sleep 60; done
grep -q '^status: done' $O/$WAITCELL/exit.txt 2>/dev/null || { echo "$(date -Is) lane $GPU: $WAITCELL ended not-done, claim released" >> $O/lane_$GPU.log; rm -rf $O/$WAITCELL/.claim; }
bash $HERE/queue.sh $GPU 2000 2 "" 70000
