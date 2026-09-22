#!/bin/bash
# Switch-off check (spec 1.3), 5 R_S episodes, Llama-3.1-8B Ours R-only, seed 0, PYTHONHASHSEED 0:
#   old = planner_demo_seeded_v1.py (seed switch only; the file the first queue attempt ran), no iclr_env_over key
#   new = planner_demo_seeded.py with +iclr_env_over_metrics=False
# run at the same time on GPU 2 and GPU 3, then compare_switch_off.py. Then start both queue lanes.
PFS=/mnt/pfs/devs/pn5wp/shishuqing
HERE=$PFS/partnr-planner/scripts/iclr_master/B
B=$PFS/partnr-isambard
O=$B/outputs/iclr_master/B
V=$B/outputs/iclr_master/B_verify
L=$PFS/iclr_master_logs
mkdir -p $V
sed 's/^    seed = 47668090$/    seed = int(config.get("iclr_seed", 47668090)); print(f"ICLR_SEED {seed}", flush=True)/' \
  $B/habitat_llm/examples/planner_demo.py > $B/habitat_llm/examples/planner_demo_seeded_v1.py
sha256sum $B/habitat_llm/examples/planner_demo_seeded_v1.py $B/habitat_llm/examples/planner_demo_seeded.py > $V/demo_sha256.txt
DS=DS:iclr_master/verify5.json.gz
DEMO_MODULE=planner_demo_seeded_v1 ICLR_ENV_OVER=omit bash $HERE/run_cell.sh ../B_verify/old_v1 llama8b ours_RS_R $DS 0 2 14400 2 &
P1=$!
ICLR_ENV_OVER=False bash $HERE/run_cell.sh ../B_verify/new_off llama8b ours_RS_R $DS 0 3 14400 2 &
P2=$!
wait $P1 $P2
/root/venvs/partnr/bin/python $HERE/compare_switch_off.py $V/old_v1/hydra/results/verify5.json.gz \
  $V/new_off/hydra/results/verify5.json.gz 1065,217,224,976,1461 > $V/switch_off_compare.txt 2>&1
R=$(tail -1 $V/switch_off_compare.txt)
echo "$(date -Is) switch-off check: $R" | tee -a $O/lane_2.log >> $O/lane_3.log
if [ "$R" != ALL_IDENTICAL ]; then echo "$(date -Is) HALT: switch-off check not identical, queue not started" | tee -a $O/lane_2.log >> $O/lane_3.log; exit 1; fi
setsid nohup bash $HERE/queue.sh 2 2000 2 "" 70000 > $L/lane_2.out 2>&1 < /dev/null & echo $! > $L/lane_2.pid
sleep 60
setsid nohup bash $HERE/queue.sh 3 2000 2 "" 70000 > $L/lane_3.out 2>&1 < /dev/null & echo $! > $L/lane_3.pid
