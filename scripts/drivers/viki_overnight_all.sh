#!/usr/bin/env bash
# Everything remaining after P0's libraries, in one chain.
#
# What costs calls and what does not, stated because it decides the shape of this job:
# the intent prompt is `user content + INSTRUCTION`, a module-level constant, and the
# memory is consulted only AFTER the answer arrives. So an archived answer can be scored
# against any library, and every evaluation cell here runs with `--replay` at zero
# generation cost. The only exceptions are the per-row fallback re-ask (spec §5, measured
# at 0 rows on every recombination cell) and P1, which builds three new libraries.
#
# Guards, per this repo's rules: one generation job on the endpoint at a time, a hard
# timeout AND a stall guard on every cell sized to that cell rather than copied, waiting
# by PID and never by a process-name pattern, cells already on disk skipped so the job is
# resumable, and every report written to disk before it is printed.
set -u

ROOT=/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
A11=$ROOT/results/viki_memory_experiments/amendment11
OUT=$ROOT/outputs/overnight_all
LIBS=$ROOT/outputs/p0_libraries
URL72=${URL72:-http://192.168.32.40:8050/v1}
MODEL72=${MODEL72:-qwen2.5-vl-72b-amendment3-f2}
# 8050 is the external box; 8061 and 8062 are local. Probing 192.168.32.40 for the latter
# two answers nothing, which is not the same as their being down.
URL30=${URL30:-http://127.0.0.1:8062/v1}
MODEL30=${MODEL30:-qwen3-vl-30b}
URL7=${URL7:-http://127.0.0.1:8061/v1}
MODEL7=${MODEL7:-qwen2.5-vl-7b}
SEEDS=${SEEDS:-"20260901 20260902 20260903"}
WAIT_PID=${WAIT_PID:-}
HARD_EVAL=${HARD_EVAL:-5400}
STALL_EVAL=${STALL_EVAL:-1200}
HARD_BUILD=${HARD_BUILD:-36000}
STALL_BUILD=${STALL_BUILD:-1800}

mkdir -p "$OUT"
cd "$ROOT" || exit 1
export TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

# Wait for a still-running job by PID. Never `pgrep -f` -- it matches this script's own
# command line, which has deadlocked a watcher here six times.
if [ -n "$WAIT_PID" ]; then
    say "waiting for PID $WAIT_PID"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    say "PID $WAIT_PID gone"
fi

run_guarded () {                 # label logfile hard stall -- command...
    local label=$1 log=$2 hard=$3 stall=$4; shift 4
    "$@" >> "$log" 2>&1 &
    local pid=$! last=0 quiet=0 now begun
    begun=$(date +%s)
    while kill -0 "$pid" 2>/dev/null; do
        sleep 30
        now=$(wc -c < "$log" 2>/dev/null || echo 0)
        if [ "$now" -gt "$last" ]; then last=$now; quiet=0; else quiet=$((quiet + 30)); fi
        if [ "$quiet" -ge "$stall" ]; then
            say "STALL  $label: log flat ${quiet}s -- killing $pid"; kill -9 "$pid" 2>/dev/null; break
        fi
        if [ $(( $(date +%s) - begun )) -ge "$hard" ]; then
            say "TIMEOUT $label after ${hard}s -- killing $pid"; kill -9 "$pid" 2>/dev/null; break
        fi
    done
    wait "$pid" 2>/dev/null; return $?
}

endpoint_for () {                # model -> "<base-url> <served model>"
    # The fallback re-ask must go to the model whose answers are being replayed. Sending a
    # 30B cell's re-ask to 72B would put a stronger model inside a weaker model's row.
    case "$1" in
        72B) echo "$URL72 $MODEL72" ;;
        30B) echo "$URL30 $MODEL30" ;;
        7B)  echo "$URL7 $MODEL7" ;;
        *)   echo "" ;;
    esac
}

replay_source () {               # model split -> archived answers
    case "$1/$2" in
        72B/id)                   echo "$A11/intent_crew_clean.jsonl" ;;
        72B/recombination-text)   echo "$A11/recomb_text_agentic.jsonl" ;;
        72B/recombination-imaged) echo "$A11/recomb_imaged_agentic.jsonl" ;;
        30B/id)                   echo "$A11/m30_id.jsonl" ;;
        30B/recombination-text)   echo "$A11/m30_recomb_text.jsonl" ;;
        30B/recombination-imaged) echo "$A11/m30_recomb_imaged.jsonl" ;;
        7B/id)                    echo "$A11/m7_id.jsonl" ;;
        7B/recombination-text)    echo "$A11/m7_recomb_text.jsonl" ;;
        7B/recombination-imaged)  echo "$A11/m7_recomb_imaged.jsonl" ;;
        *) echo "" ;;
    esac
}

cell () {                        # tag memory model split [extra flags...]
    local tag=$1 memory=$2 model=$3 split=$4; shift 4
    [ -f "$A11/$tag.jsonl" ] && { say "skip   $tag"; return 0; }
    local src; src=$(replay_source "$model" "$split")
    if [ -z "$src" ] || [ ! -f "$src" ]; then
        say "SKIP   $tag -- 未执行，缺 ${src:-replay source for $model/$split}"; return 0
    fi
    if [ ! -f "$memory" ]; then
        say "SKIP   $tag -- 未执行，缺 $memory"; return 0
    fi
    local endpoint url served
    endpoint=$(endpoint_for "$model")
    url=${endpoint% *}; served=${endpoint#* }
    if [ -z "$endpoint" ] || ! curl -sf --max-time 8 "$url/models" | grep -q "$served"; then
        say "SKIP   $tag -- 未执行，缺 endpoint for $model (${url:-unset} serving ${served:-unset})"
        return 0
    fi
    say "start  $tag  (memory=$(basename "$memory") split=$split model=$model at $url)"
    run_guarded "$tag" "$OUT/$tag.log" "$HARD_EVAL" "$STALL_EVAL" \
        $PY scripts/viki_eval_v2_intent_choice.py --memory "$memory" --split "$split" \
            --tag "$tag" --replay "$src" --model "$served" --base-url "$url" \
            --workers 8 "$@"
    grep -E "^accuracy|re-asked|infeasible" "$OUT/$tag.log" | tail -2
}

# ---------------------------------------------------------------- memory artefacts
for seed in $SEEDS; do
    lib=$LIBS/agentic_library_$seed.json
    mem=$LIBS/agentic_memory_$seed.json
    [ -f "$lib" ] || { say "未执行，缺 $lib"; continue; }
    [ -f "$mem" ] || $PY scripts/viki_memory_from_library.py --library "$lib" --out "$mem"
done

# ---------------------------------------------------------------- P0: 72B, three splits
say "=== P0: ours, 72B ==="
for seed in $SEEDS; do
    mem=$LIBS/agentic_memory_$seed.json
    cell "ours_72B_id_$seed"        "$mem" 72B id
    cell "ours_72B_text_$seed"      "$mem" 72B recombination-text
    cell "ours_72B_imaged_$seed"    "$mem" 72B recombination-imaged
done

# ---------------------------------------------------------------- P3: 30B and 7B
# Each model's re-ask goes to that model's own endpoint (8062 and 8061, both local).
say "=== P3: ours, 30B and 7B ==="
for seed in $SEEDS; do
    mem=$LIBS/agentic_memory_$seed.json
    for model in 30B 7B; do
        cell "ours_${model}_id_$seed"     "$mem" "$model" id
        cell "ours_${model}_text_$seed"   "$mem" "$model" recombination-text
        cell "ours_${model}_imaged_$seed" "$mem" "$model" recombination-imaged
    done
done

# ---------------------------------------------------------------- P4: ablations
say "=== P4: ablations on the agent library, 72B ==="
for seed in $SEEDS; do
    mem=$LIBS/agentic_memory_$seed.json
    cell "abl_noorder_72B_id_$seed"     "$mem" 72B id                    --no-order
    cell "abl_noorder_72B_text_$seed"   "$mem" 72B recombination-text    --no-order
    cell "abl_noorder_72B_imaged_$seed" "$mem" 72B recombination-imaged  --no-order
    cell "abl_noground_72B_id_$seed"     "$mem" 72B id                   --no-grounding
    cell "abl_noground_72B_text_$seed"   "$mem" 72B recombination-text   --no-grounding
    cell "abl_noground_72B_imaged_$seed" "$mem" 72B recombination-imaged --no-grounding
done

# ---------------------------------------------------------------- appendix: reference
say "=== appendix: reference library, 72B ==="
REF=$A11/skill_memory_v2.json
cell "appendix_reference_72B_id"     "$REF" 72B id
cell "appendix_reference_72B_text"   "$REF" 72B recombination-text
cell "appendix_reference_72B_imaged" "$REF" 72B recombination-imaged

# ---------------------------------------------------------------- P1: no-trace control
say "=== P1: no-trace control libraries (arm d) ==="
if curl -sf --max-time 8 "$URL72/models" | grep -q "$MODEL72"; then
    run_guarded "p1_notrace" "$OUT/p1_notrace.log" "$HARD_BUILD" "$STALL_BUILD" \
        $PY scripts/drivers/viki_p0_libraries.py --base-url "$URL72" --model "$MODEL72" \
            --no-traces --label-prefix p1nt --out-root "$ROOT/outputs/p1_notrace"
    for seed in $SEEDS; do
        lib=$ROOT/outputs/p1_notrace/agentic_library_$seed.json
        mem=$ROOT/outputs/p1_notrace/agentic_memory_$seed.json
        [ -f "$lib" ] || { say "未执行，缺 $lib"; continue; }
        [ -f "$mem" ] || $PY scripts/viki_memory_from_library.py --library "$lib" --out "$mem"
        cell "notrace_72B_id_$seed"     "$mem" 72B id
        cell "notrace_72B_text_$seed"   "$mem" 72B recombination-text
        cell "notrace_72B_imaged_$seed" "$mem" 72B recombination-imaged
    done
else
    say "SKIP P1 -- 未执行，缺 $URL72 serving $MODEL72"
fi

# ---------------------------------------------------------------- report
say "=== report ==="
$PY scripts/viki_results_report.py --out "$ROOT/results/agent_library_$(date +%F)" \
    >> "$OUT/report.log" 2>&1
tail -20 "$OUT/report.log"
say "overnight chain done"
