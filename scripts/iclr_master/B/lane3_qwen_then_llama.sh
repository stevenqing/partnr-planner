#!/bin/bash
# GPU 3 lane: all qwen7b cells against the vLLM endpoint (num_proc 4), then stop that endpoint by PID
# and continue with the remaining llama8b cells (in-process HF, num_proc 2) like lane 2.
PFS=/mnt/pfs/devs/pn5wp/shishuqing
HERE=$PFS/partnr-planner/scripts/iclr_master/B
L=$PFS/iclr_master_logs
O=$PFS/partnr-isambard/outputs/iclr_master/B
bash $HERE/queue.sh 3 2000 4 qwen7b 20000
p=$(cat $L/vllm_qwen7b_8090.pid 2>/dev/null)
if [ -n "$p" ]; then
  echo "$(date -Is) lane 3: qwen cells finished, stopping vLLM pid $p" >> $O/lane_3.log
  kill $p; while kill -0 $p 2>/dev/null; do sleep 5; done
fi
[ -e $O/STOP ] || bash $HERE/queue.sh 3 2000 2 llama8b 70000
