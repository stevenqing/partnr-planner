#!/bin/bash
# Shared-LLM switch check on one GPU: for each backbone, the same 2 R_S episodes (Ours R-only, seed 0,
# PYTHONHASHSEED 0, num_proc 1) with +iclr_share_llm=False and =True, run at the same time on <gpu>;
# traces compared byte for byte (compare_switch_off.py / first_divergence.py); peak memory per process logged.
PFS=/mnt/pfs/devs/pn5wp/shishuqing
HERE=$PFS/partnr-planner/scripts/iclr_master/B
B=$PFS/partnr-isambard
V=$B/outputs/iclr_master/B_verify/share_llm
GPU=${1:-4}
mkdir -p $V; cd $B/task_classification_datasets
/root/venvs/partnr/bin/python $HERE/make_subset.py rerange+spatial_matched_subtasks.json.gz iclr_master/verify_share2.json.gz 976,224
( while :; do nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader >> $V/mem_samples.csv; sleep 20; done ) &
MON=$!
for bb in llama8b qwen7b; do
  ICLR_SHARE_LLM=False bash $HERE/run_cell.sh ../B_verify/share_llm/${bb}_off $bb ours_RS_R DS:iclr_master/verify_share2.json.gz 0 $GPU 10800 1 &
  p1=$!
  ICLR_SHARE_LLM=True bash $HERE/run_cell.sh ../B_verify/share_llm/${bb}_on $bb ours_RS_R DS:iclr_master/verify_share2.json.gz 0 $GPU 10800 1 &
  p2=$!
  wait $p1 $p2
  R=$V; for s in off on; do cp $R/${bb}_$s/pid $R/${bb}_$s/pid.txt 2>/dev/null; done
  /root/venvs/partnr/bin/python $HERE/compare_switch_off.py $V/${bb}_off/hydra/results/verify_share2.json.gz $V/${bb}_on/hydra/results/verify_share2.json.gz 976,224 > $V/${bb}_compare.txt 2>&1
  /root/venvs/partnr/bin/python $HERE/first_divergence.py $V/${bb}_off/hydra/results/verify_share2.json.gz $V/${bb}_on/hydra/results/verify_share2.json.gz 976,224 >> $V/${bb}_compare.txt 2>&1
done
kill $MON
echo done > $V/FINISHED
