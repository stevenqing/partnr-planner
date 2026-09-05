#!/usr/bin/env bash
# Wait out the 70-episode verification, report it, then launch the post-fix sweep.
#
# Chained on the box rather than driven from the session, so an ssh drop cannot orphan the
# hand-off. The sweep is held until the verification's runner is really gone: both want
# the same GPU, and starting the second while the first still holds it is an OOM an hour
# in rather than an error at launch.
set -u

ROOT=/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
CELL=$ROOT/outputs/dead_fix/val_mini/v2_memory_R_deadonly
GPU=${GPU:-0}

cd "$ROOT" || exit 1
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

# Wait by looking for the runner itself, matched on the full interpreter path. Never
# `pgrep -f` on a loose pattern here: it matches this script and the ssh that launched it,
# which has cost six sessions already.
say "waiting for the verification runner to finish"
for i in $(seq 1 120); do
    if ! ps -eo cmd | grep -F "$PY -m habitat_llm.examples.planner_demo" | grep -qv grep; then
        say "runner gone after ${i}m"
        break
    fi
    sleep 60
done

N=$(ls "$CELL/results/val_mini.json.gz/stats" 2>/dev/null | wc -l)
say "verification scored $N episodes"

say "=== paired report on the 70 ==="
$PY scripts/partnr_dead_fix_report.py 2>&1
say "=== report done (json at outputs/partnr_dead_fix.json) ==="

# Only hand the GPU on if the verification actually produced a full table. A partial cell
# means something went wrong, and stacking an eight-hour sweep behind it would bury it.
if [ "$N" -lt 70 ]; then
    say "ABORT: verification scored $N/70, not launching the post-fix sweep"
    exit 1
fi

say "launching post-fix six-cell sweep on gpu $GPU"
GPU=$GPU bash scripts/drivers/rerun_priv_postfix.sh
