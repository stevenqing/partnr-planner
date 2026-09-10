#!/usr/bin/env bash
# Round two of the family induction, under the repaired submission interface.
#
#   bash scripts/drivers/viki_v3_round2.sh [family ...]
#
# One cell is one (family, seed episode, sampling seed, effect key). Everything that decides
# acceptance is unchanged -- binds and achieves on >= 2 unseen episodes, plus a marginal
# coverage gain over the family's round-one library. What changed is the submission
# interface (`--interface-v2`): what `?x` and `?y` mean, that spares are `?z1, ?z2, ...`,
# that a coordinated operator can be submitted at all, and a refusal that says what each
# variable actually became. Measured on three set_plate cells before this driver existed:
# 0/3 without it, 2/3 with it, and the accepted body is byte-identical to the reference
# library's own sealed-cupboard operator.
#
# This is a NEW CELL, not a tweak to the frozen one: nothing here may be pooled with the
# v2 pass rate.
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
URL=${URL:-http://192.168.32.40:8050/v1}
MODEL=${MODEL:-qwen2.5-vl-72b-amendment3-f2}
SAMPLES=${SAMPLES:-2}
MOVES=${MOVES:-20}
TEMP=${TEMP:-0.7}
KEYS=${KEYS:-"pos.name is_activated"}
WORKERS=${WORKERS:-4}
CELL_TIMEOUT=${CELL_TIMEOUT:-2400}
TARGETS=${TARGETS:-outputs/v3/targets.json}
OUT=${OUT:-outputs/v3}

cd "$ROOT" || exit 1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/logs"
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

# The endpoint is probed before anything is launched: a cell that runs against a dead
# endpoint writes a transcript of nothing but connection errors and looks like a model that
# cannot induce.
if ! curl -s -m 10 "${URL%/v1}/v1/models" | grep -q '"id"'; then
    say "FATAL endpoint $URL is not answering"; exit 1
fi

[ -f "$TARGETS" ] || { say "FATAL no targets: $TARGETS (run scripts/viki_v3_round2_targets.py)"; exit 1; }

wanted="$*"
mapfile -t JOBS < <("$PY" - "$TARGETS" "$SAMPLES" "$KEYS" "$wanted" <<'PYEOF'
import json, sys
targets = json.load(open(sys.argv[1]))
samples = int(sys.argv[2])
keys = sys.argv[3].split()
wanted = set(sys.argv[4].split()) if len(sys.argv) > 4 and sys.argv[4].strip() else None
for entry in targets["families"]:
    if wanted and entry["family"] not in wanted:
        continue
    for key in keys:
        for seed in entry["seeds"]:
            for sample in range(samples):
                print("\t".join([entry["family"], key, str(seed), str(sample),
                                 entry["library"],
                                 " ".join(str(i) for i in entry["holdout"]),
                                 " ".join(str(i) for i in entry["coverage_pool"])]))
PYEOF
)
say "${#JOBS[@]} cells, $WORKERS at a time, ${CELL_TIMEOUT}s each"

run_cell () {
    IFS=$'\t' read -r family key seed sample library holdout pool <<< "$1"
    local slug=${key//./_}
    local tag="v3/${family}/e${seed}_s${sample}_${slug}"
    local log="$OUT/logs/${family}_e${seed}_s${sample}_${slug}.log"
    if [ -f "outputs/agentic_rung/$tag/verdict.json" ]; then echo "skip $tag"; return 0; fi
    timeout "$CELL_TIMEOUT" "$PY" scripts/viki_agentic_rung_abstraction.py \
        --tag "$tag" --seed-episode "$seed" --sample-seed $((20260829 + sample)) \
        --moves "$MOVES" --temperature "$TEMP" --base-url "$URL" --model "$MODEL" \
        --target-key "$key" --interface-v2 \
        --library "$library" --holdout $holdout --coverage-pool $pool \
        > "$log" 2>&1
    local status=$?
    local passed
    passed=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('passed'))" \
             "outputs/agentic_rung/$tag/verdict.json" 2>/dev/null || echo "no-verdict")
    echo "[$(date +%H:%M:%S)] $tag status=$status passed=$passed"
}

pids=()
for job in "${JOBS[@]}"; do
    run_cell "$job" &
    pids+=($!)
    # Wait by PID, never by pattern: `pgrep -f` matches this script's own command line.
    while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do sleep 5; done
done
for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done

say "all cells finished; summarising"
"$PY" - "$OUT" <<'PYEOF'
import json, glob, sys, collections
from pathlib import Path
out = Path(sys.argv[1])
rows, bodies = [], collections.Counter()
for path in sorted(glob.glob("outputs/agentic_rung/v3/*/*/verdict.json")):
    verdict = json.loads(Path(path).read_text())
    family = Path(path).parts[-3]
    passed = str(verdict.get("passed")) == "True"
    operator = verdict.get("operator") or {}
    shape = ("COORD[%d]" % len(operator.get("roles", [])) if operator.get("coordinated")
             else " ".join(a[0] for a in (operator.get("body") or [])))
    rows.append({"family": family, "cell": Path(path).parts[-2], "passed": passed,
                 "moves": verdict.get("moves_used"), "shape": shape if passed else None,
                 "works_on": verdict.get("works_on")})
    if passed:
        bodies[(family, shape)] += 1
summary = {"cells": len(rows), "passed": sum(r["passed"] for r in rows), "rows": rows,
           "accepted_by_family": {"%s :: %s" % k: v for k, v in bodies.items()}}
(out / "round2_summary.json").write_text(json.dumps(summary, indent=1))
print("cells %d  passed %d" % (summary["cells"], summary["passed"]))
for k, v in sorted(bodies.items()):
    print("  %-46s %s (x%d)" % (k[0][:46], k[1], v))
print("wrote %s" % (out / "round2_summary.json"))
PYEOF
