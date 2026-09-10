#!/usr/bin/env bash
# v3: assemble the libraries the two induction rounds produced, then score every column.
#
#   bash scripts/drivers/viki_v3_build_and_score.sh
#
# Layer 1 for a family is that family's accepted operators from BOTH rounds -- round one
# (`outputs/agentic_rung/v2_<family>`) and round two under the repaired interface
# (`outputs/agentic_rung/v3/<family>`). The assembler is the same one v2 used and it is the
# only thing that admits an operator: dedup by effect and abstracted body, support
# RE-MEASURED by running the operator on induction-half episodes, and a minimum of two.
# Nothing a model claimed about its own operator survives that step.
#
# Then `viki_v2_evaluate.py --tag-prefix v3` unions the family libraries per column, mines
# Layers 2 and 3 on that column's own pool, and scores by replaying archived answers. That
# is nearly free: the prompt never contains the library, so a model's answer does not depend
# on which one is loaded and no generation is repeated. The endpoints are needed only for
# the per-row fallback re-ask.
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
LIBS=${LIBS:-outputs/v3_libraries}
MEMS=${MEMS:-outputs/v3_memories}
MODELS=${MODELS:-"72B 30B 7B"}
PROBE=${PROBE:-60}
cd "$ROOT" || exit 1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LIBS" "$MEMS"
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

families=$("$PY" -c "
import json
print(' '.join(e['family'] for e in json.load(open('outputs/v3/targets.json'))['families']))")

say "=== per-family libraries (round 1 + round 2) ==="
for family in $families; do
    out="$LIBS/library_$family.json"
    if [ -f "$out" ]; then say "skip   $family"; continue; fi
    roots=""
    [ -d "outputs/agentic_rung/v2_$family" ] && roots="$roots outputs/agentic_rung/v2_$family"
    [ -d "outputs/agentic_rung/v3/$family" ] && roots="$roots outputs/agentic_rung/v3/$family"
    if [ -z "$roots" ]; then say "no rung output for $family"; continue; fi
    # A family that admits nothing writes no file; the union treats that as "contributes
    # nothing", which is a result, not a failure.
    "$PY" scripts/viki_assemble_agentic_library.py --rung-root $roots --probe "$PROBE" \
        --out "$out" --report "$LIBS/assembly_$family.json" \
        > "$LIBS/assemble_$family.log" 2>&1 \
        && say "built  $family -> $(grep -c '"effect"' "$out" 2>/dev/null || echo ?) operators" \
        || say "nothing admitted for $family"
done

say "=== what the libraries hold ==="
"$PY" - "$LIBS" <<'PYEOF'
import json, sys, collections
from pathlib import Path
libs = Path(sys.argv[1])
total = collections.Counter()
for path in sorted(libs.glob("library_*.json")):
    ops = json.loads(path.read_text())["operators"]
    for op in ops:
        shape = ("COORD[%d]" % len(op.get("roles", [])) if op.get("coordinated")
                 else " ".join(a[0] for a in (op.get("body") or [])))
        total[(op["effect"]["key"], shape)] += 1
    print("%-46s %d" % (path.stem.replace("library_", ""), len(ops)))
print("\ndistinct operator shapes across families (before the union dedups them):")
for (key, shape), n in total.most_common():
    print("  %-14s %-58s donors=%d" % (key, shape[:58], n))
PYEOF

say "=== columns: union, build memories, score by replay ==="
"$PY" scripts/drivers/viki_v2_evaluate.py --libs-root "$LIBS" --out-root "$MEMS" \
    --tag-prefix v3 --models $MODELS
say "done"
