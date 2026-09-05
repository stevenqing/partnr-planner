#!/usr/bin/env bash
# The agent-built operator library on the recombination axis, against the rule-based one.
#
# This is the axis the paper's claim actually rests on. ID is not: CLAUDE.md records 92.5%
# verbatim train/test instruction overlap there, so neither 98.81% nor 74.24% separates
# remembering from generalising. Recombination is built to.
#
# `viki_eval_v2_intent_choice.py` already takes `--memory` and `--split recombination-*`,
# so this is running an existing path, not building one. (`viki_amendment10_run.py` is a
# different thing -- its `skill_memory` arm is v1, a retrieval provider that puts skill text
# in the prompt, and it has no v2 arm at all. Do not confuse the two.)
#
# Unlike the ID evaluation this GENERATES: 297 rows per split go through the model. Both
# libraries are scored on the same split with the same model and the same prompt, so the
# only thing differing between the two rows of the table is Layer 1. Layers 2 and 3 are the
# reference's in both, as everywhere else.
set -u

ROOT=/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
OUT=$ROOT/outputs/recomb
URL=${URL:-http://192.168.32.40:8050/v1}
MODEL=${MODEL:-qwen2.5-vl-72b-amendment3-f2}
WORKERS=${WORKERS:-16}
# The 09-04 recombination cells ran under a 3600s guard and this one is a lighter prompt,
# but the guard is sized off the slow case, not the average.
HARD=${HARD:-3600}
STALL=${STALL:-1200}

cd "$ROOT" || exit 1
mkdir -p "$OUT"
export TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

if ! curl -sf "$URL/models" | grep -q "$MODEL"; then
    say "ABORT: $URL is not serving $MODEL"; exit 1
fi

cell () {                        # label memory split
    local label=$1 memory=$2 split=$3
    local tag="recomb_${split#recombination-}_${label}"
    local done_file="$OUT/$tag.done"
    [ -f "$done_file" ] && { say "skip   $tag"; return 0; }
    local log="$OUT/$tag.log"
    say "start  $tag  (memory=$memory split=$split)"
    timeout $HARD $PY scripts/viki_eval_v2_intent_choice.py \
        --memory "$memory" --split "$split" --tag "$tag" \
        --model "$MODEL" --base-url "$URL" --workers "$WORKERS" >> "$log" 2>&1 &
    local runner=$! last=0 quiet=0 now
    # Wait by PID; `pgrep -f` matches this script's own command line.
    while kill -0 "$runner" 2>/dev/null; do
        sleep 60
        now=$(wc -c < "$log" 2>/dev/null || echo 0)
        if [ "$now" -gt "$last" ]; then last=$now; quiet=0; else quiet=$((quiet + 60)); fi
        if [ "$quiet" -ge "$STALL" ]; then
            say "STALL  $tag: log flat ${quiet}s -- killing $runner"
            kill -9 "$runner" 2>/dev/null; break
        fi
    done
    wait "$runner"; local status=$?
    [ "$status" -eq 0 ] && touch "$done_file"
    say "$([ "$status" -eq 0 ] && echo done || echo FAILED)  $tag"
    grep -E "^accuracy|accuracy " "$log" | tail -2
}

REF=results/viki_memory_experiments/amendment11/skill_memory_v2.json
AGENT=outputs/agentic_memory_runner.json

# One endpoint, so strictly sequential. Reference first on each split, so a broken cell is
# caught on the arm whose answer is already known.
for split in recombination-text recombination-imaged; do
    cell reference "$REF"   "$split"
    cell agentic   "$AGENT" "$split"
done

say "=== recombination table ==="
$PY - <<'PYEOF' | tee "$OUT/recomb_table.txt"
import json, re
from pathlib import Path
OUT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner/outputs/recomb")
rows = {}
for log in sorted(OUT.glob("recomb_*.log")):
    text = log.read_text()
    hit = re.findall(r"accuracy\s+(\d+)/(\d+)", text)
    if hit:
        got, seen = hit[-1]
        rows[log.stem] = {"solved": int(got), "total": int(seen),
                          "rate": round(int(got) / int(seen), 4)}
(OUT / "recomb_table.json").write_text(json.dumps(rows, indent=1))
print("recombination, 297 rows per split, 72B, generated (not replayed)")
print("only Layer 1 differs between the two rows; Layers 2 and 3 are the reference's\n")
print("%-12s %22s %22s" % ("library", "text", "imaged"))
for label in ("reference", "agentic"):
    cells = []
    for split in ("text", "imaged"):
        row = rows.get("recomb_%s_%s" % (split, label))
        cells.append("%s/%s = %.2f%%" % (row["solved"], row["total"], 100 * row["rate"])
                     if row else "missing")
    print("%-12s %22s %22s" % (label, *cells))
print("\n-> %s" % (OUT / "recomb_table.json"))
PYEOF
say "recombination job finished -> $OUT"
