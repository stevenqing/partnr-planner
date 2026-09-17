#!/usr/bin/env bash
# Does is_clean come up once the type check stops throwing furniture subjects away? gate_H only, 30B only.
#
# The 06:11 run read is_clean 0.000 -> 0.000 on both models and both pools while is_powered_on reached
# 0.727/0.778. That column was the bench: every is_clean proposition in these pools names furniture
# ("clean the dining table"), and `project()` opens by dropping a subject that is furniture or a room --
# right for a placement, whose subject must be carryable, wrong for a unary state predicate. The model
# wrote `table_7 | clean | -` and the type check discarded it. State predicates are now typed before the
# placement rules; the placement rule itself is untouched (`partnr_typed_state_selftest.py` checks both).
#
# gate_H is the tuning pool and the right place to confirm a repair. conf_H is NOT rerun here: it has
# already been reported once on this arm, and it stays untouched until this cell says the repair works.
#
# Both arms are rerun rather than pairing against last night's cells, so the comparison is within one
# code version. Last night's state_off cell is then compared to this one as evidence -- not just a
# self-test assertion -- that the switch being off leaves an existing cell exactly where it was.
#
#   Q=outputs/cand_iface_0917/state_fix; mkdir -p $Q; cp scripts/drivers/partnr_state_fix_0917.sh $Q/queue.sh
#   setsid nohup bash $Q/queue.sh > $Q/queue.log 2>&1 < /dev/null &
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PFS=/mnt/pfs/devs/pn5wp/shishuqing
PY=/root/venvs/partnr/bin/python
VLLM=/root/venvs/vllm/bin/python
Q=outputs/cand_iface_0917/state_fix
OLD=outputs/cand_iface_0917/state_iface/cells      # the 06:11 cells, same pool, pre-repair code
mkdir -p "$Q" "$PFS/tmp" "$PFS/vllm_cache"
export HF_HOME=$PFS/hf HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$PFS/vllm_cache TMPDIR=$PFS/tmp VLLM_ENGINE_READY_TIMEOUT_S=3600
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/v1/models" | grep -q "^200$"; }
root_free_gb () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
gpu_used_mb () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$(( $1 + 1 ))p"; }

# The model is a parameter because the card this was written for filled up mid-load: 30B took 947 s to
# load off PFS and someone else's 24.7 GiB landed on GPU 3 in the meantime ("Available KV cache memory:
# -8.88 GiB"). What this cell tests is the bench -- whether the type check still throws away
# `table_7 | clean | -` -- and is_clean read 0.000 on 7B too, so 7B answers it at 16 GiB and a 1-minute load.
MTAG=${MTAG:-7b}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
REV=${REV:-}
SERVED=${SERVED:-qwen2.5-vl-7b}
UTIL=${UTIL:-0.40}
NO_THINK=${NO_THINK:-0}
LIB=results/partnr_operators_llm5.json
PRIOR=results/partnr_inside_prior_train_R_only.json
FROZEN="typed_stages=True typed_beside=True typed_same_object=True"
POOL=gate_H
GPU=${GPU:-1}
PORT=${PORT:-8063}
ORACLE=outputs/gate/h30b

both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }
both_plus () { echo "+evaluation.agents.agent_0.planner.plan_config.$1=$2 +evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

report () { local name=$1; shift; say "report $name"; timeout 3600 "$@" > "$Q/$name.txt" 2>&1 < /dev/null || say "WARN report $name exit $?"; }

say "queue start (commit $(git rev-parse --short HEAD)); root free $(root_free_gb)G"
"$PY" scripts/partnr_typed_state_selftest.py > "$Q/selftest.txt" 2>&1 \
  || { say "REFUSING: self-test failed -- see $Q/selftest.txt"; exit 6; }
say "self-test passed: $(tail -1 "$Q/selftest.txt")"
grep -q "nothing answers to the state subject" our_method/skill_memory_v2/partnr_typed_goals.py \
  || { say "REFUSING: the furniture-subject repair is not in the code"; exit 6; }

# Hydra parses every override before a card is touched.
extra=""; for kv in $FROZEN; do extra="$extra $(both "${kv%%=*}" "${kv#*=}")"; done
for flag in False True; do
  timeout 600 "$PY" -m habitat_llm.examples.planner_demo \
    --config-name baselines/skill_memory_v2_typed_vllm.yaml --cfg job \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$POOL.json.gz" \
    $(both operators "$LIB") $(both inside_prior "$PRIOR") $(both typed_examples RS) \
    $(both_plus typed_state "$flag") $extra > "$Q/dryrun_$flag.txt" 2>&1 \
    || { say "REFUSING: hydra rejects typed_state=$flag"; tail -5 "$Q/dryrun_$flag.txt" | cut -c1-200; exit 6; }
  grep -q "typed_state: $(echo "$flag" | tr 'A-Z' 'a-z')" "$Q/dryrun_$flag.txt" \
    || { say "REFUSING: typed_state=$flag did not reach plan_config"; exit 6; }
done
say "hydra dry run ok"

# Not "is the card empty" -- no card on this box is empty any more, and demanding one is how this cell
# refused to start at all. What matters is whether what is left holds the weights, the KV cache and 24
# habitat processes (~0.75 GiB each). NEED_MB is checked again after the load, because the 10:24 attempt
# passed this test and then lost the card to someone else during a 947-second load.
NEED_MB=${NEED_MB:-56000}
used=$(gpu_used_mb "$GPU"); total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n "$(( GPU + 1 ))p")
free=$(( total - used ))
[ "$free" -ge "$NEED_MB" ] || { say "REFUSING: GPU $GPU has ${free}MiB free, need ${NEED_MB}MiB"; exit 4; }
say "GPU $GPU: ${free}MiB free of ${total}MiB (${used}MiB is someone else's)"
probe "$PORT" && { say "REFUSING: :$PORT already answers"; exit 4; }
log=$Q/vllm-$SERVED-$PORT.log
rev=""; [ -n "$REV" ] && rev="--revision $REV"
(CUDA_VISIBLE_DEVICES=$GPU setsid nohup "$VLLM" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" $rev \
    --served-model-name "$SERVED" --port "$PORT" --tensor-parallel-size 1 --max-model-len 16384 \
    --gpu-memory-utilization "$UTIL" --limit-mm-per-prompt '{"image":1}' > "$log" 2>&1 < /dev/null &) > /dev/null 2>&1
for _ in $(seq 300); do probe "$PORT" && break; sleep 10; done
probe "$PORT" || { say "ALARM :$PORT did not come up in 50 min"; tail -8 "$log" | cut -c1-300; exit 4; }
say ":$PORT up (GPU $GPU)"
reply=$(curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word ok.\"}],\"max_tokens\":8,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
say "smoke: $(echo "$reply" | head -c 140)"
echo "$reply" | grep -q '"content"' && ! echo "$reply" | grep -q '<think>' || { say "ALARM smoke failed"; exit 4; }

for state in off on; do
  flag=False; [ "$state" = "on" ] && flag=True
  CONFIG=baselines/skill_memory_v2_typed_vllm.yaml MODEL=$SERVED URL=http://127.0.0.1:$PORT/v1 \
    NO_THINK=$NO_THINK OPERATORS=$LIB GPU=$GPU PROCS=24 HARD_TIMEOUT=7200 \
    bash scripts/drivers/partnr_model_cell.sh "$POOL" "${POOL}_state_${state}_fix_$MTAG" "$Q/cells" \
      +resume=True $(both inside_prior "$PRIOR") $(both typed_examples RS) \
      $(both_plus typed_state "$flag") $extra >> "$Q/cells.log" 2>&1
  say "cell ${POOL}_state_${state}_fix: $(ls "$Q/cells/${POOL}_state_${state}_fix_$MTAG/results/$POOL.json.gz/stats" 2>/dev/null | wc -l)/60"
done

pid=$(ps -eo pid,args | awk -v p="--port $PORT" '$0 ~ /vllm.entrypoints.openai.api_server/ && index($0, p) {print $1}' | head -1)
[ -n "$pid" ] && { kill "$pid"; for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid"; say ":$PORT stopped (pid $pid)"; }

OFF=$Q/cells/${POOL}_state_off_fix_$MTAG
ON=$Q/cells/${POOL}_state_on_fix_$MTAG
n_off=$(ls "$OFF/results/$POOL.json.gz/stats" 2>/dev/null | wc -l)
n_on=$(ls "$ON/results/$POOL.json.gz/stats" 2>/dev/null | wc -l)
if [ "$n_off" -eq 60 ] && [ "$n_on" -eq 60 ]; then
  oracle=""
  [ -d "$ORACLE/base" ] && oracle="--cell oracle_base=$ORACLE/base --cell oracle_cand=$ORACLE/cand0"
  report keys_fix_$MTAG "$PY" scripts/partnr_key_admission.py --pool "$POOL" \
      --cell state_off="$OFF" --cell state_on="$ON" --cell prerepair_on="$OLD/${POOL}_state_on_$MTAG" $oracle \
      --compare state_on:state_off state_on:prerepair_on --json "$Q/keys_fix_$MTAG.json"
  for key in is_clean is_powered_on; do
    report pc_${key}_$MTAG "$PY" scripts/partnr_gate_compare.py --pool "$POOL" \
        --a "$OFF" --b "$ON" --key "$key" --json "$Q/pc_${key}_$MTAG.json"
  done
  # Evidence, not assertion: with the switch off, the repaired code must land where last night landed.
  report offarm_unchanged_$MTAG "$PY" scripts/partnr_gate_compare.py --pool "$POOL" \
      --a "$OLD/${POOL}_state_off_$MTAG" --b "$OFF" --key is_clean --json "$Q/offarm_unchanged_$MTAG.json"
else
  say "SKIP reports: cells are $n_off and $n_on of 60"
fi
{
  echo "# is_clean after the furniture-subject repair (gate_H, $MTAG, tuning pool only)"
  echo; echo "commit $(git rev-parse --short HEAD); finished $(date '+%m-%d %H:%M'); library $LIB"
  echo "Pre-repair reading on this pool: is_clean 0.000 -> 0.000, is_powered_on 0.000 -> 0.727."
  echo "Oracle ceiling on gate_H (cand0): is_clean 0.122 (its own gate pool reading was 0.980 on the admitted body)."
  echo; echo '```'
  for st in off on; do printf '%-26s %s\n' "state_$st" "$(cat "$Q/cells/${POOL}_state_${st}_fix_$MTAG/CELL.json" 2>/dev/null | head -c 200)"; done
  echo '```'; echo; echo "## Alarms"; echo '```'
  grep -h -E "ALARM|WARN|SKIP|REFUSING|ENDPOINT_DEAD|STALL|TIMEOUT" "$Q/queue.log" "$Q/cells.log" 2>/dev/null || echo none
  echo '```'
  for r in "$Q"/keys_fix.txt "$Q"/pc_*.txt "$Q"/offarm_unchanged.txt; do
    [ -s "$r" ] || continue
    echo; echo "## $(basename "$r" .txt)"; echo '```'
    grep -v -E "Gym|gymnasium|Please upgrade|PluginManager|migration_guide" "$r" | head -70
    echo '```'
  done
} > "$Q/STATE_FIX_SUMMARY_$MTAG.md"
touch "$Q/ALLDONE_$MTAG"
say "summary: $Q/STATE_FIX_SUMMARY_$MTAG.md"
