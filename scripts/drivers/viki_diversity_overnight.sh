#!/usr/bin/env bash
# Drive the ladder at what the current library fails, and see whether the library moves.
#
# The parity run left the agent-built library at 97/200 (0.485) with exactly two operators,
# assembled from twenty-four accepted submissions. Deduplication is what exposed the
# problem: every seed episode was solvable by the plainest body, so the plainest body came
# back every time. The reference reaches 194/200 with ten achievement operators, and the
# extra eight are not better bodies -- they are the cases the plain one does not cover
# (subject sealed in a container, target sealed, object already held), each carrying the
# preconditions that say when it applies.
#
# So seeds come from `viki_rung_residual_targets.py`: episodes the *current* library fails,
# split into coverage holes (no operator for the key) and variant holes (an operator was
# offered and the plan missed). Holdout is drawn from the residual as well, so an operator
# can only pass by covering something the library does not already cover. Nothing here
# names what is missing -- the residual is a fact about the library, not a hint.
#
# Known limitation, recorded rather than worked around: the rung's acceptance test does not
# exercise preconditions at all. An operator passes by working on two episodes; nothing asks
# what it does where it should not apply, which is why both agent-built operators carry
# `preconditions: {}`. The protocol's preconditions rung -- drop a fact, replay, keep it
# dropped if the effect still holds -- is the next one to build, and this job does not
# substitute for it.
set -u

ROOT=/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
OUT=$ROOT/outputs/parity
SHIM=scripts/viki_library_shim.py

URL_72B=${URL_72B:-http://192.168.32.40:8050/v1}
MODEL_72B=${MODEL_72B:-qwen2.5-vl-72b-amendment3-f2}
URL_30B=${URL_30B:-http://127.0.0.1:8062/v1}
MODEL_30B=${MODEL_30B:-qwen3-vl-30b}

# More samples than the parity run: the question here is whether a *different* operator can
# be found at all, and that is a coverage question, so it wants draws.
SAMPLES=${SAMPLES:-5}
WORKERS=${WORKERS:-4}
MOVES=${MOVES:-18}
TEMP=${TEMP:-0.7}
SWEEP_HARD=${SWEEP_HARD:-10800}
STALL=${STALL:-1200}

cd "$ROOT" || exit 1
mkdir -p "$OUT"
export TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

score () {                       # name library.json
    local name=$1 library=$2
    local out="$OUT/bench_$name.json"
    [ -f "$out" ] && { say "skip   score $name (already on disk)"; return 0; }
    say "score  $name"
    VIKI_LIBRARY_JSON="$library" $PY scripts/viki_inducer_bench.py \
        --inducer "$SHIM" --out "$out" >> "$OUT/score_$name.log" 2>&1 \
        || { say "FAILED score $name -- see $OUT/score_$name.log"; return 1; }
    $PY - "$out" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); sc = d.get("self_check") or {}
print("        operators=%s  solved=%s/%s  rate=%s  keys=%s"
      % (d.get("operators"), sc.get("solved"), sc.get("episodes"), sc.get("rate"),
         d.get("effect_keys_offered")))
PYEOF
}

sweep_chain () {                 # label model url
    local label=$1 model=$2 url=$3
    for key in pos.name is_activated; do
        local safe=${key//./_}
        local tag="${label}_resid_${safe}"
        local dir="outputs/agentic_rung_sweep/$tag"
        [ -f "$dir/summary.json" ] && { say "skip   sweep $tag"; continue; }
        local seeds holdout
        seeds=$($PY -c "
import json; t=json.load(open('outputs/rung_residual_targets.json'))['targets'].get('$key',{})
print(' '.join(str(x) for x in t.get('seeds',[])))")
        holdout=$($PY -c "
import json; t=json.load(open('outputs/rung_residual_targets.json'))['targets'].get('$key',{})
print(' '.join(str(x) for x in t.get('holdout',[])))")
        if [ -z "$seeds" ] || [ -z "$holdout" ]; then
            say "SKIP   $tag: no usable residual episodes for $key"; continue
        fi
        mkdir -p "$dir"
        local log="$OUT/sweep_$tag.log"
        say "start  $tag  seeds=[$seeds] holdout=[$holdout]"
        timeout $SWEEP_HARD $PY scripts/viki_agentic_rung_sweep.py \
            --model "$model" --base-url "$url" --label "$tag" \
            --episodes $seeds --holdout-pool $holdout \
            --target-key "$key" --samples $SAMPLES --workers $WORKERS \
            --moves $MOVES --temperature $TEMP >> "$log" 2>&1 &
        local runner=$! last=0 quiet=0 now
        # Wait by PID; never `pgrep -f`, the pattern is in this script's own command line.
        while kill -0 "$runner" 2>/dev/null; do
            sleep 60
            now=$(wc -c < "$log" 2>/dev/null || echo 0)
            if [ "$now" -gt "$last" ]; then last=$now; quiet=0; else quiet=$((quiet + 60)); fi
            if [ "$quiet" -ge "$STALL" ]; then
                say "STALL  $tag: log flat for ${quiet}s -- killing $runner"
                kill -9 "$runner" 2>/dev/null; break
            fi
        done
        wait "$runner"
        say "done   $tag -> $(tr -d '\n ' < $dir/summary.json 2>/dev/null | head -c 180)"
    done
}

say "=== residual targets from the current library ==="
$PY scripts/viki_rung_residual_targets.py >> "$OUT/residual_targets.log" 2>&1 \
    || { say "residual targeting FAILED"; exit 1; }

say "=== residual-seeded sweeps, one job per endpoint, chains in parallel ==="
sweep_chain 72b "$MODEL_72B" "$URL_72B" &
P72=$!
sweep_chain 30b "$MODEL_30B" "$URL_30B" &
P30=$!
wait $P72; wait $P30
say "both chains finished"

say "=== re-assemble everything and score ==="
$PY scripts/viki_assemble_agentic_library.py \
    --out outputs/agentic_library_div.json \
    --report outputs/agentic_library_div_assembly.json >> "$OUT/assemble_div.log" 2>&1 \
    && score agentic_div outputs/agentic_library_div.json \
    || say "assembly FAILED -- see $OUT/assemble_div.log"

say "=== report ==="
$PY scripts/viki_parity_report.py 2>&1 | tee "$OUT/parity_report.txt"
say "diversity job finished -> $OUT"
