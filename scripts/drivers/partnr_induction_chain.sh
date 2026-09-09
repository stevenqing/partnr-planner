#!/usr/bin/env bash
# The PARTNR induction run, end to end, once the gate is calibrated.
#
# Order matters and is not negotiable: proposal reads rollouts only, acceptance executes on
# `gate_iir`, and `conf_iir` is opened exactly once, at the end, on whatever acceptance
# produced. The reported splits are never touched by anything that selects.
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PY=/root/venvs/partnr/bin/python
TAG=${TAG:-iir1}
ROLLOUTS=${ROLLOUTS:-results/partnr_rollouts/rec_R_iir}
KEY=${KEY:-is_in_room}
CANDIDATES=${CANDIDATES:-6}
MOVES=${MOVES:-30}
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

say "1/4 propose ($CANDIDATES candidates from $ROLLOUTS)"
$PY scripts/partnr_propose_operators.py --rollouts "$ROLLOUTS" --target-key "$KEY" \
    --moves "$MOVES" --candidates "$CANDIDATES" --out "outputs/propose/$TAG" || exit 1

say "1b/4 the no-trace control: same agent, same task, tools that refuse to read"
$PY scripts/partnr_propose_operators.py --rollouts "$ROLLOUTS" --target-key "$KEY" \
    --moves "$MOVES" --candidates "$CANDIDATES" --no-traces \
    --out "outputs/propose/${TAG}_notrace" || true

say "2/4 groundability (free)"
$PY scripts/partnr_groundable.py --candidates "outputs/propose/$TAG/candidates.json" \
    --json "outputs/propose/$TAG/groundable.json" > /dev/null

say "3/4 execution gate on gate_iir"
$PY scripts/partnr_gate_batch.py --candidates "outputs/propose/$TAG/candidates.json" \
    --pool gate_iir --tag "$TAG" --procs 60 || exit 1

say "4/4 accepted library"
$PY - "$TAG" <<'PYEOF'
import json, sys
from pathlib import Path
tag = sys.argv[1]
summary = json.loads(Path(f"outputs/gate/{tag}/SUMMARY.json").read_text())
base = json.loads(Path("results/partnr_operators.json").read_text())["operators"]
accepted = [r["operator"] for r in summary["results"] if r["accepted"]]
out = Path(f"results/partnr_operators_{tag}.json")
out.write_text(json.dumps({"operators": base + accepted, "accepted": len(accepted),
                           "from": f"outputs/gate/{tag}/SUMMARY.json"}, indent=1))
print(json.dumps({"accepted": len(accepted), "library": str(out),
                  "size": len(base) + len(accepted)}, indent=1))
PYEOF
say "chain finished -- confirmation and the reported splits are separate, deliberate steps"
