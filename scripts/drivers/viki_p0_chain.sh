#!/usr/bin/env bash
# Everything left on the 30B, run one after another on the one endpoint.
#
# Sequential on purpose. Two generation jobs against the same served model have already
# cost this project two deadlocks today -- hundreds of blocked threads, an idle endpoint
# and a zero-byte output that looked exactly like slow progress. One at a time is slower
# on paper and finishes sooner in practice.
#
#   think, repeats 2 and 3    the spread of the table already reported. These arms are not
#                             greedy -- one cell regenerated agreed with itself on 1 row of
#                             297 -- so the published numbers are single draws until this
#                             lands.
#   no-think, repeats 1..3    the same twelve cells with the thinking instruction removed.
#                             A separate table, never merged with the think one.
#
# Every cell is guarded by its own row count, so a killed run resumes rather than appending
# a second generation onto a half-written file.
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
say () { echo "===== $(date +%m-%d\ %H:%M:%S) $* ====="; }

for rep in 2 3; do
  say "think, repeat $rep"
  REP=$rep bash scripts/drivers/viki_p0_30b.sh
done

for rep in 1 2 3; do
  say "no-think, repeat $rep"
  NOTHINK=1 REP=$rep bash scripts/drivers/viki_p0_30b.sh
done

say "chain finished"
