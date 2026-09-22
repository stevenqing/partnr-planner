#!/bin/bash
# GPU 3: (1) switch-off check, 5 episodes already run in B_llama8b_T1_ours_RS_R_s0 (old seeded demo, no env_over code),
# rerun with the new demo and +iclr_env_over_metrics=False, compared by compare_switch_off.py;
# (2) switch-on check on episode 1461 (env ended it in s0); (3) the queue, all backbones, 2 processes.
PFS=/mnt/pfs/devs/pn5wp/shishuqing
HERE=$PFS/partnr-planner/scripts/iclr_master/B
B=$PFS/partnr-isambard
O=$B/outputs/iclr_master/B
V=$B/outputs/iclr_master/B_verify
mkdir -p $V
cd $B/task_classification_datasets
/root/venvs/partnr/bin/python $HERE/make_subset.py rerange+spatial_matched_subtasks.json.gz iclr_master/verify5.json.gz 1065,217,224,976,1461
/root/venvs/partnr/bin/python $HERE/make_subset.py rerange+spatial_matched_subtasks.json.gz iclr_master/verify_on_1461.json.gz 1461
ICLR_ENV_OVER=False bash $HERE/run_cell.sh ../B_verify/switch_off_5 llama8b ours_RS_R DS:iclr_master/verify5.json.gz 0 3 14400 2
bash $HERE/run_cell.sh ../B_verify/switch_on_1461 llama8b ours_RS_R DS:iclr_master/verify_on_1461.json.gz 0 3 7200 1
/root/venvs/partnr/bin/python $HERE/compare_switch_off.py \
  $O/B_llama8b_T1_ours_RS_R_s0/hydra/results/rerange+spatial_matched_subtasks.json.gz \
  $V/switch_off_5/hydra/results/verify5.json.gz 1065,217,224,976,1461 > $V/switch_off_compare.txt 2>&1
echo "$(date -Is) lane 3: verification finished ($(tail -1 $V/switch_off_compare.txt)); starting queue" >> $O/lane_3.log
[ -e $O/STOP ] || bash $HERE/queue.sh 3 2000 2 "" 70000
