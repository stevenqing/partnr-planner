#!/bin/bash
# Switch-on check: waits for the first "ICLR_ENV_OVER: habitat env ended episode <id>" in any B cell log,
# then waits for that episode's stats file and records whether it carries metrics -> B_verify/switch_on_check.txt
PFS=/mnt/pfs/devs/pn5wp/shishuqing
O=$PFS/partnr-isambard/outputs/iclr_master/B
OUT=$PFS/partnr-isambard/outputs/iclr_master/B_verify/switch_on_check.txt
while :; do
  hit=$(grep -a -H -m1 "ICLR_ENV_OVER: habitat env ended" $O/B_*/run.log 2>/dev/null | head -1)
  [ -n "$hit" ] && break; sleep 120
done
cell=$(dirname ${hit%%:*}); ep=$(echo "$hit" | grep -o "episode [0-9]*" | grep -o "[0-9]*")
if [ -z "$ep" ]; then  # cells started before 05:10 print no episode id: take the process's next "Metrics For Run 0 Episode"
  pre=$(grep -a -n -m1 "ICLR_ENV_OVER: habitat env ended" $cell/run.log | cut -d: -f1)
  ep=$(tail -n +$pre $cell/run.log | grep -a -m1 -o "Episode [0-9]*:" | grep -o "[0-9]*")
fi
f=$(ls -d $cell/hydra/results/*/)stats/$ep.json
for i in $(seq 1 180); do [ -f $f ] && break; sleep 60; done
{ echo "detected: $(date -Is)"; echo "cell: $(basename $cell)"; echo "episode: $ep"; echo "log line: ${hit#*:}"
  echo "stats file: $f"; cat $f 2>/dev/null || echo "MISSING"; echo
  echo "planner-log: $(ls $(dirname $f)/../planner-log/planner-log-episode_${ep}_0.json 2>/dev/null || echo MISSING)"
  grep -q '"stats"' $f 2>/dev/null && echo "VERDICT: metrics written" || echo "VERDICT: NO metrics"; } > $OUT
