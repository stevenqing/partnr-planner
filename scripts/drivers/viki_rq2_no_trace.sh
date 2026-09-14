#!/usr/bin/env bash
# RQ2 condition B, no_trace: the full v3 induction budget re-run with --no-traces.
# PREPARED 2026-09-14, NOT RUN.
#
#   bash scripts/drivers/viki_rq2_no_trace.sh [round1|targets|round2|build|all]
#
# Mirrors the full condition exactly, with only --no-traces added, into
# outputs/agentic_rung/rq2_notrace/:
#
#   round1   scripts/drivers/viki_v2_libraries.py with its defaults: results/frozen_sweep_v2.json
#            (14 families x 56 rungs = 784 cells, 18 moves, T=0.7, build seed, per-rung
#            holdout, --hard 3600, marginal gate against the family's growing working
#            library). Tags rq2_notrace/v2_<family>/e<seed>_<key>. One driver per family (its
#            own out-root and manifest), R1_FAMILY_WORKERS at a time; the default 1 is the
#            sequential order full ran in. Rungs WITHIN a family stay sequential either way,
#            because each one's library depends on the previous ones.
#   targets  scripts/viki_v3_round2_targets.py --round-one-label rq2_notrace/v2_%s: the SAME
#            procedure full used, applied to no_trace's own round-one verdicts. Seeds,
#            holdout and coverage pool come from the episode listing (identical to full's
#            outputs/v3/targets.json, which is checked); only the round-one library each
#            family is handed differs, and it is no_trace's own. No trace is read.
#   round2   the cell command of scripts/drivers/viki_v3_round2.sh (2 samples x 2 seeds x
#            2 keys x 14 families = 112 cells, 20 moves, T=0.7, sample seed 20260829+s,
#            --interface-v2, 4 workers, 2400 s) plus --no-traces. Tags
#            rq2_notrace/v3/<family>/e<seed>_s<sample>_<key>.
#   build    per-family assembly (probe 60, min support 2), 14-family union and the 8
#            single-family folds -- the full pipeline, unchanged -- into
#            results/paper_viki_iclr2027/libraries/rq2/no_trace/. Then the candidates
#            extraction and the ledger refresh.
#
# Guards: endpoint probed before every stage and before every cell (a dead endpoint stops the
# run rather than leaving holes, since later cells depend on earlier ones); resume = rerun the
# same stage, finished cells (verdict.json present) are skipped; per-cell hard timeout; round 2
# also kills a cell whose log has not grown for STALL seconds; cell counts are checked before
# the next stage starts. Waiting is by PID only.
#
# Note on --interface-v2 under --no-traces: the rung adds the INTERFACE prompt text only when
# traces are on (`if args.interface_v2 and not args.no_traces`), so in round 2 the flag only
# adds the "`?x` is never grasped" note to submission feedback. Mirrored as is.
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
URL=${URL:-http://192.168.32.40:8050/v1}
MODEL=${MODEL:-qwen2.5-vl-72b-amendment3-f2}
OUT=${OUT:-outputs/rq2_notrace}
LABEL=${LABEL:-rq2_notrace}
R1_DEFINITION=${R1_DEFINITION:-results/frozen_sweep_v2.json}
R1_HARD=${R1_HARD:-3600}
R1_FAMILY_WORKERS=${R1_FAMILY_WORKERS:-1}
SAMPLES=${SAMPLES:-2}
MOVES=${MOVES:-20}
TEMP=${TEMP:-0.7}
KEYS=${KEYS:-"pos.name is_activated"}
WORKERS=${WORKERS:-4}
CELL_TIMEOUT=${CELL_TIMEOUT:-2400}
STALL=${STALL:-1200}
PROBE=${PROBE:-60}
FULL_TARGETS=${FULL_TARGETS:-outputs/v3/targets.json}
PAPER=${PAPER:-results/paper_viki_iclr2027/libraries/rq2/no_trace}
STAGE=${1:-all}

cd "$ROOT" || exit 1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/round1" "$OUT/round2/logs" "$OUT/libraries"
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
alive () { curl -s -m 10 "${URL%/v1}/v1/models" | grep -q '"id"'; }

count_verdicts () { find "outputs/agentic_rung/$LABEL/$1" -name verdict.json 2>/dev/null | wc -l; }

expected_round1 () {
    "$PY" - "$R1_DEFINITION" <<'PYEOF'
import json, sys
definition = json.load(open(sys.argv[1]))
print(sum(len(entry["rungs"]) for entry in definition["libraries"] if entry["status"] == "ok"))
PYEOF
}

stage_round1 () {
    alive || { say "FATAL endpoint $URL is not answering"; return 1; }
    local families
    families=$("$PY" - "$R1_DEFINITION" <<'PYEOF'
import json, sys
definition = json.load(open(sys.argv[1]))
print(" ".join(entry["family"] for entry in definition["libraries"] if entry["status"] == "ok"))
PYEOF
)
    say "round 1: $(expected_round1) cells, $R1_FAMILY_WORKERS family driver(s) at a time"
    run_family () {
        local family=$1 dir="$OUT/round1/$1"
        mkdir -p "$dir"
        # viki_v2_libraries.py rewrites its manifest on start; keep the previous attempt's.
        [ -f "$dir/run_manifest.json" ] && cp "$dir/run_manifest.json" "$dir/run_manifest.$(date +%Y%m%d_%H%M%S).json"
        "$PY" scripts/drivers/viki_v2_libraries.py --definition "$R1_DEFINITION" \
            --base-url "$URL" --model "$MODEL" --out-root "$dir" --hard "$R1_HARD" \
            --skip-notrace --label-format "$LABEL/v2_%s" --no-traces \
            --families "$family" --probe-endpoint >> "$dir/driver.log" 2>&1
        echo "[$(date +%H:%M:%S)] round1 $family exit=$?"
    }
    local pids=()
    for family in $families; do
        run_family "$family" &
        pids+=($!)
        while [ "$(jobs -rp | wc -l)" -ge "$R1_FAMILY_WORKERS" ]; do sleep 10; done
    done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done
    local have want
    have=$(find "outputs/agentic_rung/$LABEL" -path "*/v2_*/*" -name verdict.json 2>/dev/null | wc -l)
    want=$(expected_round1)
    say "round 1 verdicts $have / $want"
    [ "$have" -eq "$want" ]
}

stage_targets () {
    local have want
    have=$(find "outputs/agentic_rung/$LABEL" -path "*/v2_*/*" -name verdict.json 2>/dev/null | wc -l)
    want=$(expected_round1)
    [ "$have" -eq "$want" ] || { say "FATAL round 1 incomplete ($have / $want); targets not written"; return 1; }
    "$PY" scripts/viki_v3_round2_targets.py --out "$OUT/round2" --round-one-label "$LABEL/v2_%s" \
        > "$OUT/round2/targets.log" 2>&1 || { say "FATAL targets script failed"; return 1; }
    "$PY" - "$OUT/round2/targets.json" "$FULL_TARGETS" <<'PYEOF'
import json, sys
mine, full = (json.load(open(p))["families"] for p in sys.argv[1:3])
shape = lambda entries: [(e["family"], e["total_episodes"], e["seeds"], e["holdout"], e["coverage_pool"]) for e in entries]
if shape(mine) != shape(full):
    print("FATAL no_trace round-2 seeds/holdout/pool differ from full's targets"); sys.exit(1)
print("targets match full's seeds/holdout/pool for %d families; no_trace round-one library sizes %s"
      % (len(mine), {e["family"]: e["library_size"] for e in mine}))
PYEOF
}

stage_round2 () {
    alive || { say "FATAL endpoint $URL is not answering"; return 1; }
    local targets="$OUT/round2/targets.json"
    [ -f "$targets" ] || { say "FATAL no targets: $targets (run the targets stage)"; return 1; }
    mapfile -t JOBS < <("$PY" - "$targets" "$SAMPLES" "$KEYS" <<'PYEOF'
import json, sys
targets = json.load(open(sys.argv[1]))
samples = int(sys.argv[2])
keys = sys.argv[3].split()
for entry in targets["families"]:
    for key in keys:
        for seed in entry["seeds"]:
            for sample in range(samples):
                print("\t".join([entry["family"], key, str(seed), str(sample), entry["library"],
                                 " ".join(str(i) for i in entry["holdout"]),
                                 " ".join(str(i) for i in entry["coverage_pool"])]))
PYEOF
)
    say "round 2: ${#JOBS[@]} cells, $WORKERS at a time, ${CELL_TIMEOUT}s each, stall ${STALL}s"
    run_cell () {
        IFS=$'\t' read -r family key seed sample library holdout pool <<< "$1"
        local slug=${key//./_}
        local tag="$LABEL/v3/${family}/e${seed}_s${sample}_${slug}"
        local log="$OUT/round2/logs/${family}_e${seed}_s${sample}_${slug}.log"
        if [ -f "outputs/agentic_rung/$tag/verdict.json" ]; then echo "skip $tag"; return 0; fi
        [ -f "$log" ] && mv "$log" "$log.$(date +%s).prev"
        timeout "$CELL_TIMEOUT" "$PY" scripts/viki_agentic_rung_abstraction.py \
            --tag "$tag" --seed-episode "$seed" --sample-seed $((20260829 + sample)) \
            --moves "$MOVES" --temperature "$TEMP" --base-url "$URL" --model "$MODEL" \
            --target-key "$key" --interface-v2 --no-traces \
            --library "$library" --holdout $holdout --coverage-pool $pool \
            > "$log" 2>&1 &
        local cell=$!
        while kill -0 "$cell" 2>/dev/null; do
            sleep 30
            if [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt "$STALL" ]; then
                echo "[$(date +%H:%M:%S)] STALL $tag: log silent > ${STALL}s, killing $cell"
                kill "$cell" 2>/dev/null
                break
            fi
        done
        wait "$cell" 2>/dev/null
        local status=$?
        local passed
        passed=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('passed'))" \
                 "outputs/agentic_rung/$tag/verdict.json" 2>/dev/null || echo "no-verdict")
        echo "[$(date +%H:%M:%S)] $tag status=$status passed=$passed"
    }
    local pids=() dead=0
    for job in "${JOBS[@]}"; do
        if ! alive; then say "endpoint $URL stopped answering; no further cells launched"; dead=1; break; fi
        run_cell "$job" &
        pids+=($!)
        while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do sleep 5; done
    done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done
    local have
    have=$(find "outputs/agentic_rung/$LABEL/v3" -name verdict.json 2>/dev/null | wc -l)
    say "round 2 verdicts $have / ${#JOBS[@]}"
    [ "$dead" -eq 0 ] && [ "$have" -eq "${#JOBS[@]}" ]
}

stage_build () {
    local r1 r2 want1
    r1=$(find "outputs/agentic_rung/$LABEL" -path "*/v2_*/*" -name verdict.json 2>/dev/null | wc -l)
    r2=$(find "outputs/agentic_rung/$LABEL/v3" -name verdict.json 2>/dev/null | wc -l)
    want1=$(expected_round1)
    if [ "$r1" -ne "$want1" ] || [ "$r2" -eq 0 ]; then
        [ "${ALLOW_INCOMPLETE:-0}" = 1 ] || { say "FATAL cells incomplete (r1 $r1/$want1, r2 $r2); set ALLOW_INCOMPLETE=1 to build anyway"; return 1; }
    fi
    local libs="$OUT/libraries"
    mkdir -p "$libs" "$PAPER"
    read -r -a FAMILIES <<< "$("$PY" -c "import json;print(' '.join(json.load(open('$R1_DEFINITION'))['build_families']))")"
    read -r -a FOLDS <<< "$("$PY" -c "import json;print(' '.join(json.load(open('$R1_DEFINITION'))['fold_families']))")"
    for family in "${FAMILIES[@]}"; do
        local out="$libs/library_$family.json"
        [ -f "$out" ] && { say "skip   $family"; continue; }
        local roots=""
        [ -d "outputs/agentic_rung/$LABEL/v2_$family" ] && roots="$roots outputs/agentic_rung/$LABEL/v2_$family"
        [ -d "outputs/agentic_rung/$LABEL/v3/$family" ] && roots="$roots outputs/agentic_rung/$LABEL/v3/$family"
        [ -z "$roots" ] && { say "no rung output for $family"; continue; }
        "$PY" scripts/viki_assemble_agentic_library.py --rung-root $roots --probe "$PROBE" \
            --out "$out" --report "$libs/assembly_$family.json" > "$libs/assemble_$family.log" 2>&1 \
            && say "built  $family" || say "nothing admitted for $family"
    done
    # The union needs at least one library file. When no family admits anything, an explicit
    # empty library stands in, so the memory still exists (Layer 1 empty, Layers 2/3 mined)
    # and the downstream evaluation can run on it rather than being replaced by a zero-shot file.
    local empty="$libs/library__no_operator_admitted.json"
    union_of () {   # out excluded family...
        local out=$1 excluded=$2; shift 2
        [ -f "$out" ] && { say "skip union $(basename "$out")"; return 0; }
        local present=()
        for f in "$@"; do [ -f "$libs/library_$f.json" ] && present+=("$libs/library_$f.json"); done
        if [ "${#present[@]}" -eq 0 ]; then
            [ -f "$empty" ] || echo '{"operators": []}' > "$empty"
            present=("$empty")
        fi
        local extra=()
        [ -n "$excluded" ] && extra=(--excluded-family "$excluded")
        "$PY" scripts/viki_union_library.py --libraries "${present[@]}" --families "$@" \
            --out "$out" "${extra[@]}" > "${out%.json}.log" 2>&1
    }
    union_of "$PAPER/memory_all.json" "" "${FAMILIES[@]}"
    for held in "${FOLDS[@]}"; do
        local rest=()
        for f in "${FAMILIES[@]}"; do [ "$f" != "$held" ] && rest+=("$f"); done
        union_of "$PAPER/memory_heldout_$held.json" "$held" "${rest[@]}"
    done
    sha256sum "$PAPER"/memory_*.json "$libs"/library_*.json 2>/dev/null > "$PAPER/SHA256SUMS"
    "$PY" scripts/viki_rq2_no_exec_admission.py extract --rung-prefix "$LABEL/" \
        --family-libs "$libs" --union-memory "$PAPER/memory_all.json" \
        --out results/paper_viki_iclr2027/libraries/rq2/candidates_no_trace.jsonl
    "$PY" scripts/viki_rq2_induction_ledger.py
    say "build done: $PAPER"
}

case "$STAGE" in
    round1)  stage_round1 ;;
    targets) stage_targets ;;
    round2)  stage_round2 ;;
    build)   stage_build ;;
    all)     stage_round1 && stage_targets && stage_round2 && stage_build ;;
    *) echo "usage: $0 [round1|targets|round2|build|all]"; exit 2 ;;
esac
