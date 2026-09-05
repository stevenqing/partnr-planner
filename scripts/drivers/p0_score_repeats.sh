#!/usr/bin/env bash
# Score the 48 cells the chain produced on 09-04 and nobody has scored.
#
# The archived outputs/p0_30b_report.json covers tag m30 only -- it was built at
# 09-04 12:38 and the chain did not finish until 20:59. Three think repeats give the
# spread the single-draw table never had; the no-think family is a SEPARATE CONDITION
# and gets its own file so it can never be pooled with the think cells.
#
# Serial on purpose: scoring re-runs every row through the symbolic simulator, and one
# job at a time has been the rule on this box all week.
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PY=/root/venvs/partnr/bin/python
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
for tag in m30r2 m30r3 m30nt m30ntr2 m30ntr3; do
  out=outputs/p0_30b_report.$tag.json
  if [ -s "$out" ]; then say "skip  $tag -- $out exists"; continue; fi
  say "start $tag"
  if $PY scripts/viki_p0_report.py --models 30B --tag "$tag" --json "$out" \
        > "outputs/p0_score_$tag.log" 2>&1; then
    say "done  $tag -> $out"
  else
    say "FAILED $tag -- see outputs/p0_score_$tag.log"
  fi
done
say "all tags finished"
