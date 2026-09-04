#!/usr/bin/env bash
# Repeats two and three of the 30B baseline table.
#
# These arms are not greedy: the same cell generated twice agreed on 1 of 297 rows. A
# single number per cell is therefore one draw from a distribution nobody has measured,
# and every McNemar p-value reported off it inherits that. Three runs under identical
# conditions give each cell a mean and a standard deviation, and make it possible to say
# whether a gap between two arms is larger than the spread of either.
#
# Conditions are byte-identical to repeat one -- same backbone, same worker count, same
# prompt, same seed constant -- because a repeat that changes anything measures the change
# instead of the spread.
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
for rep in 2 3; do
  echo "===== repeat $rep starting $(date +%H:%M:%S) ====="
  REP=$rep bash scripts/drivers/viki_p0_30b.sh
done
echo "===== repeats finished $(date +%H:%M:%S) ====="
