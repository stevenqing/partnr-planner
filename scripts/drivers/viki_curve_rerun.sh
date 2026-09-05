#!/usr/bin/env bash
# Re-measure the abstraction-rung capability curve under the repaired framework.
#
# Required, not optional: `docs/AGENTIC-OPERATOR-INDUCTION.md` says any change to the frozen
# framework "invalidates the comparison and must re-run every model", and four changes were
# made on 2026-09-05 -- `normalise_request`, `run_operator` trying every robot,
# `contrast_actors`, and the informative refusal. The 09-04 curve (72B 40% / 30B 40% /
# 7B 0%) was collected while the harness answered a correctly-formed submission with
# "unknown tool None" in 152 of 524 transcripts, the 7B cell worst of all: 10 of its 14 runs,
# 125 rejections. That curve is a lower bound and its zero is the least trustworthy point on it.
#
# Everything else is the frozen cell exactly: seed episodes 0/1/3/7/9, holdout pool
# 4/6/8/10/12, three samples, 18 moves, temperature 0.7, no `--target-key`, no `--library`.
# Only the harness repairs differ, which is the whole point of re-running.
set -u

ROOT=/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
OUT=$ROOT/outputs/parity
SWEEP_HARD=${SWEEP_HARD:-10800}
STALL=${STALL:-1200}

cd "$ROOT" || exit 1
mkdir -p "$OUT"
export TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

one () {                          # label model url
    local label=$1 model=$2 url=$3
    local tag="${label}_v4"
    local dir="outputs/agentic_rung_sweep/$tag"
    [ -f "$dir/summary.json" ] && { say "skip   $tag"; return 0; }
    if ! curl -sf "${url%/v1}/v1/models" | grep -q "$model"; then
        say "ABORT $tag: $url is not serving $model"; return 1
    fi
    mkdir -p "$dir"
    local log="$OUT/sweep_$tag.log"
    say "start  $tag"
    timeout $SWEEP_HARD $PY scripts/viki_agentic_rung_sweep.py \
        --model "$model" --base-url "$url" --label "$tag" \
        --episodes 0 1 3 7 9 --holdout-pool 4 6 8 10 12 \
        --samples 3 --workers 4 --moves 18 --temperature 0.7 >> "$log" 2>&1 &
    local runner=$! last=0 quiet=0 now
    # Wait by PID. `pgrep -f` matches this script's own command line and the ssh above it.
    while kill -0 "$runner" 2>/dev/null; do
        sleep 60
        now=$(wc -c < "$log" 2>/dev/null || echo 0)
        if [ "$now" -gt "$last" ]; then last=$now; quiet=0; else quiet=$((quiet + 60)); fi
        if [ "$quiet" -ge "$STALL" ]; then
            say "STALL  $tag: log flat ${quiet}s -- killing $runner"
            kill -9 "$runner" 2>/dev/null; break
        fi
    done
    wait "$runner"
    say "done   $tag -> $(tr -d '\n ' < $dir/summary.json 2>/dev/null | head -c 170)"
}

# One generation job per endpoint; the three endpoints are independent, so the three models
# run in parallel and each is alone on its server.
one 72b qwen2.5-vl-72b-amendment3-f2 http://192.168.32.40:8050/v1 &
P1=$!
one 30b qwen3-vl-30b                 http://127.0.0.1:8062/v1 &
P2=$!
one 7b  qwen2.5-vl-7b                http://127.0.0.1:8061/v1 &
P3=$!
wait $P1; wait $P2; wait $P3

say "=== repaired capability curve ==="
$PY - <<'PYEOF' | tee "$OUT/capability_curve_v4.txt"
import json, glob, os
rows = []
for path in sorted(glob.glob("outputs/agentic_rung_sweep/*_v4/summary.json")):
    d = json.load(open(path))
    old = {"72b_v4": "6/15 = 40%", "30b_v4": "6/15 = 40%", "7b_v4": "0/15 = 0%"}.get(d["label"], "?")
    rows.append({"label": d["label"], "model": d["model"], "passed": d["passed"],
                 "runs": d["runs"], "rate": d["rate"], "by_episode": d["by_episode"],
                 "before_repair": old})
    print("%-10s %-32s %2d/%-3d = %-6s   (before the repairs: %s)"
          % (d["label"], d["model"], d["passed"], d["runs"], d["rate"], old))
json.dump(rows, open("outputs/capability_curve_v4.json", "w"), indent=1)
print("\n-> outputs/capability_curve_v4.json")
PYEOF
say "curve re-run finished"
