#!/bin/bash
# One line per cell of the B.2 queue: state, episodes with stats, success/pc over episodes with metrics,
# noMet = episodes whose stats file has no metrics (env_over / error, see cell_verdict.py), attempts.
# Usage (remote): bash status.sh      states: done | running | failed | incomplete | queued
PFS=/mnt/pfs/devs/pn5wp/shishuqing
HERE=$PFS/partnr-planner/scripts/iclr_master/B
O=$PFS/partnr-isambard/outputs/iclr_master/B
tot_done=0; tot_eps=0
printf "%-34s %-10s %8s %7s %7s %5s %3s\n" cell state eps succ pc noMet att
while IFS=$'\t' read -r CELL BB KEY TGT SEED N; do
  [[ $CELL == \#* ]] && continue
  d=$O/$CELL; st=queued; eps=0; sc=""; pc=""; nerr=""
  if [ -d $d/hydra ]; then
    eps=$(ls $d/hydra/results/*/stats/*.json 2>/dev/null | wc -l)
    if [ -f $d/pid ] && kill -0 $(cat $d/pid) 2>/dev/null; then st=running
    elif grep -q '^status: done' $d/exit.txt 2>/dev/null; then st=done
    elif [ -f $d/FAILED ]; then st=failed
    else st=incomplete; fi
    read sc pc nerr < <(/root/venvs/partnr/bin/python - "$d" <<'PY'
import json, glob, sys
D = [json.loads(open(f).read()) for f in glob.glob(sys.argv[1] + "/hydra/results/*/stats/*.json")]
S = [json.loads(d["stats"]) for d in D if "stats" in d]
print(f"{sum(s['task_state_success'] for s in S)/len(S):.3f} {sum(s['task_percent_complete'] for s in S)/len(S):.3f} {len(D)-len(S)}" if S else f"- - {len(D)-len(S)}")
PY
)
  elif [ -f $d/exit.txt ]; then st="$(head -1 $d/exit.txt)"; fi
  att=$(grep -c start $d/attempts.txt 2>/dev/null || echo 0)
  tot_eps=$((tot_eps + eps)); [ "$st" = done ] && tot_done=$((tot_done + 1))
  printf "%-34s %-10s %4s/%-3s %7s %7s %5s %3s\n" "$CELL" "$st" "$eps" "$N" "$sc" "$pc" "${nerr:-}" "$att"
done < $HERE/cells.tsv
echo "cells done: $tot_done / $(grep -vc '^#' $HERE/cells.tsv); episodes with stats: $tot_eps (EPISODE_CAP 7200)"
for g in 2 3 4; do [ -f $O/lane_$g.log ] && echo "lane $g: $(tail -1 $O/lane_$g.log)"; done
[ -e $O/STOP ] && echo "STOP file present"
