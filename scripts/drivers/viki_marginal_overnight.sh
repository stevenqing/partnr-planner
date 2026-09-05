#!/usr/bin/env bash
# The same per-family sweeps, but an operator now has to ADD something to be accepted.
#
# The previous cell showed why this is needed. Same-family seeding worked -- 15 of 144 runs
# passed, 72B reaching 10/12 on `set_plate_and_fork_on_table` -- and the library did not move
# at all: 40 submissions, still 3 distinct operators, and that family still 0/23. Every one
# of the 15 re-derived `Move Reach Grasp Move Place`, the operator already in the library.
# They passed because the rung asks "does this achieve its effect on two held-out episodes",
# which an operator already in the library passes trivially, on episodes the planner still
# cannot solve.
#
# So acceptance now asks the question the library is actually short of: adding this operator
# must make at least one holdout episode go from unsolved to solved. A resubmission of what
# the memory already has is refused, and the model is told that rather than being told it
# passed. Whether these models can produce a *variant* is not established either way yet --
# the previous test could not have shown it.
#
# This is a NEW cell: the acceptance rule and the task text both differ. Its numbers are
# never pooled with the frozen rung's 40% or with the `_fam_` cell's 15/144.
#
# This exists because the previous residual sweep's 0/50 measured the seeding, not the model.
# It sorted failing episodes by index, so the seeds landed in `toast_bread_and_set_plate` and
# `wash_fruit_and_serve` while the holdout landed in `clear_table_..._put_in_cabinet` and
# `set_plate_and_fork_on_table`. Deriving an operator from the first and having it work on
# the second is not a variant problem, it is an impossible one: the sealed-target case the
# holdout needs is never demonstrated in the seed family.
#
# `viki_rung_family_targets.py` keeps both sides inside one family and targets only families
# the current library actually fails on the induction half:
#
#   clear_table_with_two_robots_and_put_in_cabinet   0/34
#   set_plate_and_fork_on_table                      0/23
#   sequential_pick_two_and_place                    0/19
#   ensure_all_fruits_on_table                      46/65
#   serve_bread_after_checking_cabinet               0/12
#   dog_push_box_for_two_panda_transport             0/10
#
# A pass here is a checkable claim: an operator derived from one episode of a family the
# library cannot do, which then works on other episodes of that same family. That is the
# first honest test of whether these models can produce a *variant* at all.
set -u

ROOT=/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
OUT=$ROOT/outputs/parity
SHIM=scripts/viki_library_shim.py
TARGETS=outputs/rung_family_targets.json

URL_72B=${URL_72B:-http://192.168.32.40:8050/v1}
MODEL_72B=${MODEL_72B:-qwen2.5-vl-72b-amendment3-f2}
URL_30B=${URL_30B:-http://127.0.0.1:8062/v1}
MODEL_30B=${MODEL_30B:-qwen3-vl-30b}

SAMPLES=${SAMPLES:-3}
WORKERS=${WORKERS:-4}
MOVES=${MOVES:-18}
TEMP=${TEMP:-0.7}
SWEEP_HARD=${SWEEP_HARD:-10800}
STALL=${STALL:-1200}

cd "$ROOT" || exit 1
mkdir -p "$OUT"
export TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

score () {
    local name=$1 library=$2
    local out="$OUT/bench_$name.json"
    [ -f "$out" ] && { say "skip   score $name"; return 0; }
    say "score  $name"
    VIKI_LIBRARY_JSON="$library" $PY scripts/viki_inducer_bench.py \
        --inducer "$SHIM" --out "$out" >> "$OUT/score_$name.log" 2>&1 \
        || { say "FAILED score $name"; return 1; }
    $PY - "$out" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); sc = d.get("self_check") or {}
print("        operators=%s solved=%s/%s rate=%s keys=%s"
      % (d.get("operators"), sc.get("solved"), sc.get("episodes"), sc.get("rate"),
         d.get("effect_keys_offered")))
PYEOF
}

families () { $PY -c "
import json; print(' '.join(json.load(open('$TARGETS'))['targets']))"; }

sweep_chain () {                 # label model url
    local label=$1 model=$2 url=$3
    for family in $(families); do
        local tag="${label}_marg_${family}"
        local dir="outputs/agentic_rung_sweep/$tag"
        [ -f "$dir/summary.json" ] && { say "skip   $tag"; continue; }
        local seeds holdout key
        seeds=$($PY -c "
import json; t=json.load(open('$TARGETS'))['targets']['$family']
print(' '.join(str(x) for x in t['seeds']))")
        holdout=$($PY -c "
import json; t=json.load(open('$TARGETS'))['targets']['$family']
print(' '.join(str(x) for x in t['holdout']))")
        key=$($PY -c "
import json; print(json.load(open('$TARGETS'))['targets']['$family']['target_key'])")
        mkdir -p "$dir"
        local log="$OUT/sweep_$tag.log"
        say "start  $tag  key=$key seeds=[$seeds] holdout=[$holdout]"
        timeout $SWEEP_HARD $PY scripts/viki_agentic_rung_sweep.py \
            --model "$model" --base-url "$url" --label "$tag" \
            --episodes $seeds --holdout-pool $holdout --target-key "$key" \
            --library outputs/agentic_library_fam.json \
            --samples $SAMPLES --workers $WORKERS --moves $MOVES \
            --temperature $TEMP >> "$log" 2>&1 &
        local runner=$! last=0 quiet=0 now
        # Wait by PID; `pgrep -f` would match this script's own command line.
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
        say "done   $tag -> $(tr -d '\n ' < $dir/summary.json 2>/dev/null | head -c 150)"
    done
}

say "=== family-stratified targets from the current library ==="
$PY scripts/viki_rung_family_targets.py >> "$OUT/family_targets.log" 2>&1 \
    || { say "family targeting FAILED"; exit 1; }

say "=== per-family sweeps, one job per endpoint ==="
sweep_chain 72b "$MODEL_72B" "$URL_72B" &
P72=$!
sweep_chain 30b "$MODEL_30B" "$URL_30B" &
P30=$!
wait $P72; wait $P30
say "both chains finished"

say "=== re-assemble, score on the gate, then end to end ==="
$PY scripts/viki_assemble_agentic_library.py \
    --out outputs/agentic_library_marg.json \
    --report outputs/agentic_library_marg_assembly.json >> "$OUT/assemble_marg.log" 2>&1 \
    && score agentic_marg outputs/agentic_library_marg.json \
    || say "assembly FAILED"

# The gate cannot be quoted on its own: 40.5% of its episodes belong to families the
# benchmark never tests, and they are the easy ones, so a pooled gate rate flatters a weak
# library. The end-to-end arm replays model responses from disk, needs no endpoint, and
# takes about a minute. It is a REPORTING instrument -- nothing selects on it.
if [ -f outputs/agentic_library_marg.json ]; then
    $PY scripts/viki_memory_from_library.py \
        --library outputs/agentic_library_marg.json \
        --out outputs/agentic_memory_marg.json >> "$OUT/e2e_marg.log" 2>&1
    say "=== end-to-end VIKI-L2 test ==="
    $PY scripts/viki_eval_skill_memory_v2.py \
        --memory outputs/agentic_memory_marg.json --tag e2e_agentic_marg 2>&1 | tail -16
fi

say "=== per-family coverage after ==="
$PY scripts/viki_rung_family_targets.py --library outputs/agentic_library_marg.json \
    --out outputs/rung_family_targets_after_marg.json 2>&1 | tail -12
say "family job finished -> $OUT"
