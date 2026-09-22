#!/bin/bash
# Start one B3 slot worker detached. Usage: launch_slot.sh <habitat_gpu> <llm_port> <slot_name>
# Refuses if the endpoint does not answer or the habitat GPU has less than 8 GB + 2.5 GB x NP free.
set -u
GPU=$1; PORT=$2; SLOT=$3
P=/mnt/pfs/devs/pn5wp/shishuqing
S=$P/partnr-planner/scripts/iclr_two_exp/B
O=$P/partnr-isambard-C/outputs/iclr_two_exp_B
NP=$(cat $O/NP)
curl -s -m 5 -o /dev/null http://127.0.0.1:$PORT/health || { echo "refuse: endpoint $PORT down"; exit 1; }
free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits)
need=$(( 8192 + 2560 * NP ))
[ "$free" -ge "$need" ] || { echo "refuse: gpu$GPU free ${free}MiB < ${need}MiB"; exit 1; }
[ -f $O/slot_$SLOT.pid ] && kill -0 $(cat $O/slot_$SLOT.pid) 2>/dev/null && { echo "refuse: slot $SLOT running"; exit 1; }
setsid nohup bash $S/worker_b.sh $GPU $PORT $SLOT > $O/slot_$SLOT.out 2>&1 < /dev/null &
echo $! > $O/slot_$SLOT.pid
echo "slot $SLOT pid $(cat $O/slot_$SLOT.pid) habitat gpu$GPU llm :$PORT NP=$NP"
