#!/usr/bin/env bash
# Recombination with error bars, and the 30B cells that were missing.
#
# The single-run table showed the agent-built library at 28.62% / 27.61% on text / imaged
# against G-Memory's 3.37% / 4.71%, which is the largest margin anywhere in this comparison
# and therefore the one that most needs a band. G-Memory's own numbers are means over three
# runs with an sd, so ours have to be the same kind of quantity to sit in the same table.
#
# What that sd measures, stated because it is easy to over-read: generation here is
# `temperature=0`, so three runs re-run a greedy decode and the spread is endpoint and
# batching nondeterminism, not sampling. That is exactly what the G-Memory band is too --
# the P0 repeats re-ran the same configuration -- so the two are comparable. It is NOT a
# confidence interval over episodes.
#
# 30B was never run on recombination for either library; those cells are new here.
#
# One generation job per endpoint, so the two models run in parallel and each is alone on
# its server. Cells already on disk are skipped, so the 72B run-1 cells from the earlier
# job are reused rather than repeated.
set -u

ROOT=/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
OUT=$ROOT/outputs/recomb
RUNS=${RUNS:-3}
HARD=${HARD:-3600}
STALL=${STALL:-1200}

cd "$ROOT" || exit 1
mkdir -p "$OUT"
export TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

REF=results/viki_memory_experiments/amendment11/skill_memory_v2.json
AGENT=outputs/agentic_memory_runner.json

cell () {                        # tag memory split model url
    local tag=$1 memory=$2 split=$3 model=$4 url=$5
    local log="$OUT/$tag.log"
    [ -f "$OUT/$tag.done" ] && { say "skip   $tag"; return 0; }
    say "start  $tag"
    timeout $HARD $PY scripts/viki_eval_v2_intent_choice.py \
        --memory "$memory" --split "$split" --tag "$tag" \
        --model "$model" --base-url "$url" --workers 16 >> "$log" 2>&1 &
    local runner=$! last=0 quiet=0 now
    # Wait by PID; `pgrep -f` matches this script's own command line.
    while kill -0 "$runner" 2>/dev/null; do
        sleep 60
        now=$(wc -c < "$log" 2>/dev/null || echo 0)
        if [ "$now" -gt "$last" ]; then last=$now; quiet=0; else quiet=$((quiet + 60)); fi
        if [ "$quiet" -ge "$STALL" ]; then
            say "STALL  $tag -- killing $runner"; kill -9 "$runner" 2>/dev/null; break
        fi
    done
    wait "$runner"; [ $? -eq 0 ] && touch "$OUT/$tag.done"
    say "done   $tag  $(grep -oE 'accuracy +[0-9]+/[0-9]+ = [0-9.]+%' "$log" | tail -1)"
}

chain () {                       # short model url
    local short=$1 model=$2 url=$3
    for split in recombination-text recombination-imaged; do
        local s=${split#recombination-}
        for run in $(seq 1 $RUNS); do
            for pair in "reference:$REF" "agentic:$AGENT"; do
                local lab=${pair%%:*} mem=${pair#*:}
                # run 1 of the 72B cells already exists under the earlier naming
                local tag="recomb_${short}_${s}_${lab}_r${run}"
                if [ "$short" = "72b" ] && [ "$run" = "1" ] && [ -f "$OUT/recomb_${s}_${lab}.done" ]; then
                    say "reuse  recomb_${s}_${lab} as $tag"
                    continue
                fi
                cell "$tag" "$mem" "$split" "$model" "$url"
            done
        done
    done
}

chain 72b qwen2.5-vl-72b-amendment3-f2 http://192.168.32.40:8050/v1 &
P1=$!
chain 30b qwen3-vl-30b                 http://127.0.0.1:8062/v1 &
P2=$!
wait $P1; wait $P2

say "=== recombination, mean +- sd over $RUNS runs ==="
$PY - <<'PYEOF' | tee "$OUT/recomb_repeats.txt"
import json, re, statistics
from pathlib import Path
OUT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner/outputs/recomb")

def rate(stem):
    log = OUT / (stem + ".log")
    if not log.is_file():
        return None
    hit = re.findall(r"accuracy\s+(\d+)/(\d+)", log.read_text())
    return int(hit[-1][0]) / int(hit[-1][1]) if hit else None

table = {}
for model in ("72b", "30b"):
    for split in ("text", "imaged"):
        for label in ("reference", "agentic"):
            runs = []
            first = rate("recomb_%s_%s" % (split, label)) if model == "72b" else None
            if first is not None:
                runs.append(first)
            for run in (1, 2, 3):
                value = rate("recomb_%s_%s_%s_r%d" % (model, split, label, run))
                if value is not None:
                    runs.append(value)
            if runs:
                table["%s|%s|%s" % (model, split, label)] = {
                    "runs": [round(r, 4) for r in runs],
                    "mean": round(statistics.mean(runs), 4),
                    "sd": round(statistics.stdev(runs), 4) if len(runs) > 1 else 0.0,
                }
(OUT / "recomb_repeats.json").write_text(json.dumps(table, indent=1))

print("recombination, 297 rows per cell, temperature 0 -- the sd is endpoint")
print("nondeterminism across re-runs, the same quantity as the G-Memory band.\n")
print("%-8s %-8s %22s %22s" % ("model", "split", "reference", "agentic"))
for model in ("72b", "30b"):
    for split in ("text", "imaged"):
        cells = []
        for label in ("reference", "agentic"):
            row = table.get("%s|%s|%s" % (model, split, label))
            cells.append("%.2f%% +- %.2f (n=%d)" % (100 * row["mean"], 100 * row["sd"],
                                                    len(row["runs"])) if row else "missing")
        print("%-8s %-8s %22s %22s" % (model, split, *cells))
print("\n-> %s" % (OUT / "recomb_repeats.json"))
PYEOF
say "recombination repeats finished"
