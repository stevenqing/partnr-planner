#!/bin/bash
# Watcher: when b2_build.sh reports a fold finished, run the Gate B2 coverage + sources check for that fold, which
# installs the library for B3 only if its sources check passes. Polls driver.log; exits after both folds.
set -u
P=/mnt/pfs/devs/pn5wp/shishuqing
W=$P/iclr_two_exp_B_build
RES=$P/partnr-planner/results/iclr_two_exp_2026-09-22/B
S=$P/partnr-planner/scripts/iclr_two_exp/B
for k in A B; do
  until grep -q "build fold $k rc=" $W/driver.log 2>/dev/null; do sleep 60; done
  if grep -q "build fold $k rc=0" $W/driver.log && [ -f $W/lib_$k/provenance.json ]; then
    echo "$(date '+%F %T') checking fold $k"
    /root/venvs/partnr/bin/python $S/b2_gate_and_sources.py $W $RES $k
  else
    echo "$(date '+%F %T') fold $k build failed or no provenance; not installed"
  fi
done
echo "POST DONE"
