#!/usr/bin/env bash
# Can an agent-built operator library reach the rule-based library's effect on VIKI-L2?
#
# The bar is `viki_inducer_bench.py`'s self-check, not the operator count: the reference
# induces 19 operators and solves 200/200 held-out episodes. A library of 25 that also
# solves 200/200 is parity; one of exactly 19 that solves 140 is not. Counting operators
# would reward padding, so nothing here optimises for it.
#
# Where this starts from: the ladder has accepted 12 operators across two models and they
# are all the SAME one -- `pos.name`, body Move/Reach/Grasp/Move/Place. Zero `is_activated`
# (which 71 held-out goals need), zero `unsealed`, zero repair, zero coordination. And no
# agent-built library has ever been through the gate at all, so its score is not a bad
# number, it is a missing one.
#
# Stages, cheapest diagnostic first:
#
#   1 reference ablation   delete operators from the REFERENCE and re-score. Costs no model
#                          call and bounds everything below: if achievement-only operators
#                          cannot reach 1.0 even when the shipped inducer wrote them, then
#                          no amount of sampling tonight can either, and we will know that
#                          before spending the night rather than after.
#   2 assemble + score     the 12 existing passes -> a library -> the gate. The first
#                          end-to-end number this method has ever had.
#   3 coverage sweep       the ladder driven at `pos.name` AND `is_activated`, on both
#                          endpoints. Episodes are selected for the key instead of taken
#                          from the front of the split, which is why every previous pass
#                          came back the same shape.
#   4 re-assemble + score  the headline.
#
# One generation job per endpoint: the 72B chain and the 30B chain run in parallel because
# they are different servers, and each runs its two keys in sequence.
set -u

ROOT=/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
OUT=$ROOT/outputs/parity
SHIM=scripts/viki_library_shim.py

URL_72B=${URL_72B:-http://192.168.32.40:8050/v1}
MODEL_72B=${MODEL_72B:-qwen2.5-vl-72b-amendment3-f2}
URL_30B=${URL_30B:-http://127.0.0.1:8062/v1}
MODEL_30B=${MODEL_30B:-qwen3-vl-30b}

SAMPLES=${SAMPLES:-3}
WORKERS=${WORKERS:-4}
# The frozen rung's own settings, so the only thing that differs from the 40% cell is the
# requested effect key and the episode selection.
MOVES=${MOVES:-18}
TEMP=${TEMP:-0.7}
# Per sweep. The 09-04 rung sweeps took about 20 minutes each; three hours is the slow case,
# not the average, because copying a neighbouring cell's timeout is how the VIKI ID cell got
# killed every round.
SWEEP_HARD=${SWEEP_HARD:-10800}
STALL=${STALL:-1200}

cd "$ROOT" || exit 1
mkdir -p "$OUT"
export TOKENIZERS_PARALLELISM=false

say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

# --- score one library through the gate ------------------------------------------------
score () {                       # name library.json
    local name=$1 library=$2
    local out="$OUT/bench_$name.json"
    [ -f "$out" ] && { say "skip   score $name (already on disk)"; return 0; }
    say "score  $name"
    VIKI_LIBRARY_JSON="$library" $PY scripts/viki_inducer_bench.py \
        --inducer "$SHIM" --out "$out" >> "$OUT/score_$name.log" 2>&1
    local status=$?
    if [ $status -ne 0 ]; then
        say "FAILED score $name (status $status) -- see $OUT/score_$name.log"
        return 1
    fi
    $PY - "$out" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
sc = d.get("self_check") or {}
print("        operators=%s  solved=%s/%s  rate=%s  violations=%s"
      % (d.get("operators"), sc.get("solved"), sc.get("episodes"), sc.get("rate"),
         d.get("contract_violations")))
PYEOF
}

# --- one endpoint's chain of key-targeted sweeps ---------------------------------------
sweep_chain () {                 # label model url
    local label=$1 model=$2 url=$3
    for key in pos.name is_activated; do
        local safe=${key//./_}
        local tag="${label}_${safe}"
        local dir="outputs/agentic_rung_sweep/$tag"
        [ -f "$dir/summary.json" ] && { say "skip   sweep $tag"; continue; }
        local seeds holdout
        seeds=$($PY -c "
import json; t=json.load(open('outputs/rung_targets.json'))['targets'].get('$key',{})
print(' '.join(str(x) for x in t.get('seeds',[])))")
        holdout=$($PY -c "
import json; t=json.load(open('outputs/rung_targets.json'))['targets'].get('$key',{})
print(' '.join(str(x) for x in t.get('holdout',[])))")
        if [ -z "$seeds" ] || [ -z "$holdout" ]; then
            say "SKIP   sweep $tag: no usable episodes for key $key"
            continue
        fi
        mkdir -p "$dir"
        local log="$OUT/sweep_$tag.log"
        say "start  sweep $tag  seeds=[$seeds] holdout=[$holdout]"
        timeout $SWEEP_HARD $PY scripts/viki_agentic_rung_sweep.py \
            --model "$model" --base-url "$url" --label "$tag" \
            --episodes $seeds --holdout-pool $holdout \
            --target-key "$key" --samples $SAMPLES --workers $WORKERS \
            --moves $MOVES --temperature $TEMP >> "$log" 2>&1 &
        local runner=$! last=0 quiet=0 now
        # Wait by PID. Never `pgrep -f` here -- the pattern is in this script's own command
        # line and in the ssh that launched it, which has cost this project six sessions.
        while kill -0 "$runner" 2>/dev/null; do
            sleep 60
            now=$(wc -c < "$log" 2>/dev/null || echo 0)
            if [ "$now" -gt "$last" ]; then last=$now; quiet=0; else quiet=$((quiet + 60)); fi
            if [ "$quiet" -ge "$STALL" ]; then
                say "STALL  $tag: log flat for ${quiet}s -- killing $runner"
                kill -9 "$runner" 2>/dev/null
                break
            fi
        done
        wait "$runner"
        say "done   sweep $tag -> $(cat $dir/summary.json 2>/dev/null | tr -d '\n ' | head -c 200)"
    done
}

# ======================================================================================
say "=== stage 1: reference ablation (no model called) ==="
$PY scripts/viki_reference_ablation.py >> "$OUT/ablation.log" 2>&1 || say "ablation FAILED"
score reference_control  outputs/reference_ablation/ref_all.json
for s in ref_achievement ref_achievement_repair ref_posname_only ref_achievement_posname; do
    score "$s" "outputs/reference_ablation/$s.json"
done

say "=== stage 2: assemble the 12 existing passes and score them ==="
$PY scripts/viki_assemble_agentic_library.py \
    --out outputs/agentic_library_frozen.json \
    --report outputs/agentic_library_frozen_assembly.json >> "$OUT/assemble_frozen.log" 2>&1 \
    && score agentic_frozen outputs/agentic_library_frozen.json \
    || say "assembly of the frozen passes FAILED -- see $OUT/assemble_frozen.log"

say "=== stage 3: coverage targets, then both endpoints in parallel ==="
$PY scripts/viki_rung_targets.py >> "$OUT/targets.log" 2>&1 || { say "targets FAILED"; }
if [ -f outputs/rung_targets.json ]; then
    sweep_chain 72b "$MODEL_72B" "$URL_72B" &
    P72=$!
    sweep_chain 30b "$MODEL_30B" "$URL_30B" &
    P30=$!
    wait $P72; wait $P30
    say "both sweep chains finished"
else
    say "SKIP   stage 3: outputs/rung_targets.json was not written"
fi

say "=== stage 4: re-assemble everything and score ==="
$PY scripts/viki_assemble_agentic_library.py \
    --out outputs/agentic_library_all.json \
    --report outputs/agentic_library_all_assembly.json >> "$OUT/assemble_all.log" 2>&1 \
    && score agentic_all outputs/agentic_library_all.json \
    || say "final assembly FAILED -- see $OUT/assemble_all.log"

say "=== report ==="
$PY scripts/viki_parity_report.py 2>&1 | tee "$OUT/parity_report.txt"
say "overnight parity job finished -> $OUT"
