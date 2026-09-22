#!/bin/bash
# B3 phase 2: when b2_build.sh has finished (its driver.log says "B2 DONE") and its 70B server is gone, start one
# Llama-3.1-8B vLLM on each of GPUs 0 and 1 (ports 8200, 8201; util 0.35, 64k context) and two slot workers per GPU
# (habitat on the same GPU, NP from $O/NP). Checks free memory before each start (launch_slot.sh refuses otherwise).
set -u
P=/mnt/pfs/devs/pn5wp/shishuqing
S=$P/partnr-planner/scripts/iclr_two_exp/B
W=$P/iclr_two_exp_B_build
until grep -q "B2 DONE" $W/driver.log 2>/dev/null; do sleep 120; done
V=$(cat $W/vllm.pid)
while kill -0 $V 2>/dev/null; do sleep 30; done
sleep 60
for g in 0 1; do
  used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "$(date '+%F %T') gpu$g used ${used}MiB before 8B start"
  bash $S/serve_8b.sh $g $((8200 + g)) 0.35 128 65536
done
for g in 0 1; do
  for i in $(seq 1 60); do curl -s -m 3 -o /dev/null http://127.0.0.1:$((8200 + g))/health && break; sleep 10; done
  curl -s -m 3 -o /dev/null http://127.0.0.1:$((8200 + g))/health || { echo "endpoint $g did not come up"; continue; }
  echo "$(date '+%F %T') endpoint gpu$g up"
  bash $S/launch_slot.sh $g $((8200 + g)) g${g}a; sleep 240
  bash $S/launch_slot.sh $g $((8200 + g)) g${g}b
done
echo "PHASE2 DONE"
