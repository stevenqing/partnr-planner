#!/usr/bin/env bash
# Wait for round two to finish, then build the v3 libraries and score every column.
#
#   ROUND2_PID=<pid> bash scripts/drivers/viki_v3_chain.sh
#
# Waiting is by PID and only by PID: `pgrep -f` matches this script's own command line and
# has killed a remote shell seven times in this project.
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
ROUND2_PID=${ROUND2_PID:-$(cat "$ROOT/outputs/v3/round2.pid" 2>/dev/null || echo 0)}
HARD_WAIT=${HARD_WAIT:-14400}
STALL=${STALL:-2700}
cd "$ROOT" || exit 1
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

say "waiting on round two (pid=$ROUND2_PID)"
started=$SECONDS
last=-1; changed=$SECONDS
while kill -0 "$ROUND2_PID" 2>/dev/null; do
    sleep 60
    done_now=$(ls outputs/agentic_rung/v3/*/*/verdict.json 2>/dev/null | wc -l)
    if [ "$done_now" -ne "$last" ]; then last=$done_now; changed=$SECONDS; fi
    if [ $((SECONDS - changed)) -ge "$STALL" ]; then
        say "STALL: $done_now verdicts unchanged for ${STALL}s -- going on with what exists"
        break
    fi
    if [ $((SECONDS - started)) -ge "$HARD_WAIT" ]; then
        say "TIMEOUT after ${HARD_WAIT}s -- going on with what exists"
        break
    fi
done
say "round two finished with $(ls outputs/agentic_rung/v3/*/*/verdict.json 2>/dev/null | wc -l)/112 verdicts, $(grep -c 'passed=True' outputs/v3/round2.log 2>/dev/null || echo 0) passes"

# The summary the round-two driver writes at its own exit; if it was killed, write it here.
[ -f outputs/v3/round2_summary.json ] || say "no round2_summary.json (driver did not reach its summary)"

# Cells that left no verdict are re-run once. Three died on 2026-09-09 with Errno 28 when
# the box's root filesystem filled up -- a cell lost to the machine, not to the model, and
# the driver skips only cells that actually produced a verdict, so this fills exactly those.
missing=$(( 112 - $(ls outputs/agentic_rung/v3/*/*/verdict.json 2>/dev/null | wc -l) ))
if [ "$missing" -gt 0 ]; then
    say "=== $missing cells left no verdict; one refill pass ==="
    bash scripts/drivers/viki_v3_round2.sh
    say "refill done: $(ls outputs/agentic_rung/v3/*/*/verdict.json 2>/dev/null | wc -l)/112"
fi

say "=== build and score ==="
bash scripts/drivers/viki_v3_build_and_score.sh
status=$?
say "build_and_score exited $status"

say "=== per-column tally (v3 cells) ==="
"$PY" - <<'PYEOF'
import json, glob
from pathlib import Path
rows = []
for path in sorted(glob.glob("results/viki_memory_experiments/amendment11/v3_*.jsonl")):
    name = Path(path).stem
    if "_fold_" in name:
        continue
    records = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not records:
        continue
    solved = sum(1 for r in records if r.get("reason") == "SOLVED")
    old = Path(path).with_name(name.replace("v3_", "v2_", 1) + ".jsonl")
    before = None
    if old.is_file():
        prev = [json.loads(l) for l in old.read_text().splitlines() if l.strip()]
        before = sum(1 for r in prev if r.get("reason") == "SOLVED")
    rows.append((name, solved, len(records), before))
print("%-34s %8s %8s %8s" % ("cell", "v3", "n", "v2"))
for name, solved, n, before in rows:
    print("%-34s %8d %8d %8s" % (name, solved, n, "-" if before is None else before))
Path("outputs/v3/column_tally.json").write_text(json.dumps(
    [{"cell": n, "solved": s, "n": t, "v2_solved": b} for n, s, t, b in rows], indent=1))
print("\nwrote outputs/v3/column_tally.json")
PYEOF
say "chain done"
