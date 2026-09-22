#!/bin/bash
# User instruction 09-22 16:2x: build fold B now, concurrently with fold A, on the same 70B endpoint (port 8090).
# Takes over from b2_build.sh: its driver bash (PID given) is killed with SIGKILL before it reaches its fold-B step
# (the fold-A build it started keeps running, it is a separate process). This script then waits on the fold-A
# builder PID and on its own fold-B build, writes the same driver.log lines b2_build.sh would ("build fold k rc=",
# "B2 DONE"), and stops the 70B server by its process group.
# Usage: b2_foldB_now.sh <b2_build_driver_pid> <foldA_builder_pid> <vllm_pgid>
set -u
DRV=$1; APID=$2; VPID=$3
P=/mnt/pfs/devs/pn5wp/shishuqing
W=$P/iclr_two_exp_B_build
RES=$P/partnr-planner/results/iclr_two_exp_2026-09-22/B
S=$P/partnr-planner/scripts/iclr_two_exp/B
REPO=$P/partnr-isambard-C
PY=/root/venvs/partnr/bin/python
kill -KILL $DRV 2>/dev/null
echo "$(date '+%F %T') b2_build.sh driver $DRV killed; fold A builder $APID continues; fold B starts concurrently" >> $W/driver.log
[ -e $W/lib_B ] && { echo "refuse: lib_B exists" >> $W/driver.log; exit 1; }
mkdir -p $W/in_B/results/heterogeneous+rerange.json.gz
ln -sfn $P/iclr_master_C_build/src/traces $W/in_B/results/heterogeneous+rerange.json.gz/traces
cp $RES/foldB_build_log.csv $W/in_B/results/episode_result_log.csv
echo "$(date '+%F %T') build fold B start ($(($(wc -l < $W/in_B/results/episode_result_log.csv) - 1)) episodes)" >> $W/driver.log
(cd $REPO/our_method && timeout 21600 $PY $S/build_with_provenance.py \
    --results-dir $W/in_B/results --output-dir $W/lib_B \
    --include-failed --use-llm --use-api --vllm-host 127.0.0.1 --vllm-port 8090 --patch-failed \
    > $W/build_B.log 2>&1 < /dev/null) &
BPID=$!
done_a=0; done_b=0
while [ $done_a = 0 ] || [ $done_b = 0 ]; do
  sleep 60
  if [ $done_a = 0 ] && ! kill -0 $APID 2>/dev/null; then
    rc=1; grep -q "Memory saved" $W/build_A.log && [ -f $W/lib_A/provenance.json ] && rc=0
    echo "$(date '+%F %T') build fold A rc=$rc api_failed=$(grep -c 'API call failed' $W/build_A.log)" >> $W/driver.log; done_a=1
  fi
  if [ $done_b = 0 ] && ! kill -0 $BPID 2>/dev/null; then
    wait $BPID; rc=$?
    echo "$(date '+%F %T') build fold B rc=$rc api_failed=$(grep -c 'API call failed' $W/build_B.log)" >> $W/driver.log; done_b=1
  fi
done
echo "$(date '+%F %T') requests_200 both folds: $(grep -c 'POST /v1/chat/completions HTTP/1.1" 200' $W/vllm.log)" >> $W/driver.log
kill -TERM -- -$VPID 2>/dev/null; sleep 20; kill -KILL -- -$VPID 2>/dev/null
echo "$(date '+%F %T') vllm stopped" >> $W/driver.log
echo "B2 DONE" >> $W/driver.log
