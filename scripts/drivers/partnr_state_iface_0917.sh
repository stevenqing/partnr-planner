#!/usr/bin/env bash
# Does the requirement interface, not the library, gate the H family? One switch, two H-dense pools.
#
# 2026-09-17 established that the two admitted H operators move nothing on the model arm
# (`is_powered_on` 0/7 in all four val_mini cells, `is_clean` identical to a library that holds no
# such operator) while the same pair is worth +0.2563 on the oracle arm (`conf_H`). Reading the code
# said why: `partnr_typed_goals.RELATIONS` had three placement relations and nothing else, the line
# grammar is `object | relation | place`, and `parse_typed` dropped any other word -- the model was
# never given a word for `clean`. `typed_state` (off by default, self-tested offline) adds the unary
# relations the LIBRARY offers, so a library without H operators still shows no H relation.
#
# This queue holds the library fixed at llm5 and moves only the switch, on two disjoint H-dense pools:
#   gate_H  60 H_R episodes -- the tuning pool, the one to look at while deciding anything
#   conf_H  60 H_R episodes -- disjoint, reported once, never tuned on
# The oracle cells for the same pair already exist (outputs/{gate,confirm}/h30b) and go into the same
# per-predicate report as the ceiling, so the reading is "of the +0.2563 the oracle arm gets, how much
# does the model arm collect once it can say the word".
#
#   Q=outputs/cand_iface_0917/state_iface; mkdir -p $Q; cp scripts/drivers/partnr_state_iface_0917.sh $Q/queue.sh
#   setsid nohup bash $Q/queue.sh > $Q/queue.log 2>&1 < /dev/null &
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PFS=/mnt/pfs/devs/pn5wp/shishuqing
PY=/root/venvs/partnr/bin/python
VLLM=/root/venvs/vllm/bin/python
Q=outputs/cand_iface_0917/state_iface
Q2=outputs/cand_iface_0916b/h30b_val     # the queue holding the cards until it is done
mkdir -p "$Q" "$PFS/tmp" "$PFS/vllm_cache"
export HF_HOME=$PFS/hf HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$PFS/vllm_cache TMPDIR=$PFS/tmp VLLM_ENGINE_READY_TIMEOUT_S=3600
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/v1/models" | grep -q "^200$"; }
root_free_gb () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
gpu_used_mb () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$(( $1 + 1 ))p"; }

LIB=results/partnr_operators_llm5.json
PRIOR=results/partnr_inside_prior_train_R_only.json
FROZEN="typed_stages=True typed_beside=True typed_same_object=True"
ORACLE_GATE=outputs/gate/h30b
ORACLE_CONF=outputs/confirm/h30b

serve () {  # gpu port model revision served util
  local gpu=$1 port=$2 model=$3 revision=$4 name=$5 util=$6 log=$Q/vllm-$5-$2.log
  [ "$(root_free_gb)" -ge 5 ] || { say "ALARM root filesystem $(root_free_gb)G free; not serving $name"; return 1; }
  probe "$port" && { say "REFUSING to serve: :$port already answers"; return 1; }
  local used; used=$(gpu_used_mb "$gpu")
  [ "$used" -lt 2000 ] || { say "REFUSING to serve $name: GPU $gpu already holds ${used}MiB"; return 1; }
  local rev=""; [ -n "$revision" ] && rev="--revision $revision"
  (CUDA_VISIBLE_DEVICES=$gpu setsid nohup "$VLLM" -m vllm.entrypoints.openai.api_server --model "$model" $rev \
      --served-model-name "$name" --port "$port" --tensor-parallel-size 1 --max-model-len 16384 \
      --gpu-memory-utilization "$util" --limit-mm-per-prompt '{"image":1}' > "$log" 2>&1 < /dev/null &) > /dev/null 2>&1
  for _ in $(seq 300); do probe "$port" && break; sleep 10; done
  probe "$port" || { say "ALARM :$port did not come up in 50 min"; tail -8 "$log" | cut -c1-300; return 1; }
  say ":$port up ($name, GPU $gpu); root free $(root_free_gb)G"
}

stop_port () {  # by PID; never `pkill -f`
  local port=$1 pid
  pid=$(ps -eo pid,args | awk -v p="--port $port" '$0 ~ /vllm.entrypoints.openai.api_server/ && index($0, p) {print $1}' | head -1)
  [ -z "$pid" ] && { say ":$port nothing to stop"; return 0; }
  kill "$pid"; for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
  sleep 10; say ":$port stopped (pid $pid)"
}

report () {
  local name=$1; shift
  say "report $name"
  timeout 3600 "$@" > "$Q/$name.txt" 2>&1 < /dev/null || say "WARN report $name exit $?"
}

both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }
# `typed_state` is new and not in the yaml, so hydra needs `+` to append it rather than override
# ("Key 'typed_state' is not in struct" killed all eight cells of the 02:17 run). And the value must be
# True/False, never on/off: the planner reads `bool(self._setting("typed_state", False))`, and
# bool("off") is True -- both arms would have run with the switch ON and the comparison would have
# read "the switch does nothing" for a reason that has nothing to do with the model.
both_plus () { echo "+evaluation.agents.agent_0.planner.plan_config.$1=$2 +evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

# One cell: the frozen typed configuration on an H pool, with typed_state either off or on.
cell () {  # mtag model nothink gpu port pool state
  local mtag=$1 model=$2 nothink=$3 gpu=$4 port=$5 pool=$6 state=$7
  local name=${pool}_state_${state}_$mtag
  local flag=False; [ "$state" = "on" ] && flag=True
  local extra=""
  for kv in $FROZEN; do extra="$extra $(both "${kv%%=*}" "${kv#*=}")"; done
  CONFIG=baselines/skill_memory_v2_typed_vllm.yaml MODEL=$model URL=http://127.0.0.1:$port/v1 \
    NO_THINK=$nothink OPERATORS=$LIB GPU=$gpu PROCS=24 HARD_TIMEOUT=7200 \
    bash scripts/drivers/partnr_model_cell.sh "$pool" "$name" "$Q/cells" \
      +resume=True $(both inside_prior "$PRIOR") $(both typed_examples RS) \
      $(both_plus typed_state "$flag") $extra >> "$Q/cells_$mtag.log" 2>&1
  local n; n=$(ls "$Q/cells/$name/results/$pool.json.gz/stats" 2>/dev/null | wc -l)
  say "cell $name: $n episodes"
}

# The per-predicate reading, with the oracle arm in the same table as the ceiling.
reports_for () {  # mtag
  local mtag=$1
  for pool in gate_H conf_H; do
    local off=$Q/cells/${pool}_state_off_$mtag on=$Q/cells/${pool}_state_on_$mtag
    local have_off have_on
    have_off=$(ls "$off/results/$pool.json.gz/stats" 2>/dev/null | wc -l)
    have_on=$(ls "$on/results/$pool.json.gz/stats" 2>/dev/null | wc -l)
    if [ "$have_off" -ne 60 ] || [ "$have_on" -ne 60 ]; then
      say "SKIP reports $pool/$mtag: cells are $have_off and $have_on of 60"; continue
    fi
    local oracle=""
    if [ "$pool" = "gate_H" ] && [ -d "$ORACLE_GATE/base" ]; then
      oracle="--cell oracle_base=$ORACLE_GATE/base --cell oracle_cand=$ORACLE_GATE/cand0"
    elif [ "$pool" = "conf_H" ] && [ -d "$ORACLE_CONF/base22" ]; then
      oracle="--cell oracle_base=$ORACLE_CONF/base22 --cell oracle_lib24=$ORACLE_CONF/lib24"
    fi
    report keys_${pool}_$mtag "$PY" scripts/partnr_key_admission.py --pool "$pool" \
        --cell state_off="$off" --cell state_on="$on" $oracle \
        --compare state_on:state_off --json "$Q/keys_${pool}_$mtag.json"
    for key in is_clean is_powered_on; do
      report pc_${pool}_${key}_$mtag "$PY" scripts/partnr_gate_compare.py --pool "$pool" \
          --a "$off" --b "$on" --key "$key" --json "$Q/pc_${pool}_${key}_$mtag.json"
    done
  done
}

line () {  # mtag model revision served nothink gpu port util
  local mtag=$1 model=$2 revision=$3 served=$4 nothink=$5 gpu=$6 port=$7 util=$8
  if serve "$gpu" "$port" "$model" "$revision" "$served" "$util"; then
    if [ "$nothink" = "1" ]; then
      reply=$(curl -s -m 120 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$served\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word ok.\"}],\"max_tokens\":8,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
      say "$mtag smoke: $(echo "$reply" | head -c 140)"
      echo "$reply" | grep -q '"content"' && ! echo "$reply" | grep -q '<think>' || {
        say "ALARM $mtag smoke failed; line skipped"; stop_port "$port"; return; }
    fi
    # gate_H first and both arms of it before conf_H: the tuning pool is the one to look at,
    # and if something is wrong with the switch it shows there before conf_H is spent.
    for pool in gate_H conf_H; do
      for state in off on; do cell "$mtag" "$served" "$nothink" "$gpu" "$port" "$pool" "$state"; done
    done
    stop_port "$port"
    reports_for "$mtag"
  fi
  say "line $mtag done"
}

# Wait for the h30b queue to release the cards -- by its own ALLDONE marker and by the endpoints
# going away, never by matching a script name.
wait_for_cards () {
  local deadline=$(( $(date +%s) + 25200 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ -e "$Q2/ALLDONE" ]; then
      probe 8071 || probe 8064 || { say "h30b queue done and its endpoints are gone"; return 0; }
    fi
    sleep 120
  done
  say "ALARM waited 7h for the h30b queue; nothing started"
  return 1
}

say "queue start (commit $(git rev-parse --short HEAD)); root free $(root_free_gb)G"
# The gate that makes this queue safe to leave alone: if the interface does not do what it says
# offline, no cell is worth running. It also proves the switch is OFF by default, which is what
# keeps the cells already on disk comparable.
"$PY" scripts/partnr_typed_state_selftest.py > "$Q/selftest.txt" 2>&1 \
  || { say "REFUSING: typed_state self-test failed -- see $Q/selftest.txt"; exit 6; }
say "self-test passed: $(tail -1 "$Q/selftest.txt")"
[ -s "$LIB" ] || { say "REFUSING: $LIB missing"; exit 6; }
grep -q "typed_state" our_method/skill_memory_v2/partnr_planner.py || { say "REFUSING: switch not in the planner"; exit 6; }
grep -q 'CONFIG=\${CONFIG' scripts/drivers/partnr_model_cell.sh || { say "REFUSING: model_cell has no CONFIG"; exit 6; }
for pool in gate_H conf_H; do
  [ -s "data/datasets/partnr_episodes/v0_0/$pool.json.gz" ] || { say "REFUSING: pool $pool missing"; exit 6; }
done

# Hydra parses every override here, with no simulator and no endpoint. The 02:17 run spent two
# endpoint loads and eight cell launches to discover an override the config would not accept.
dryrun () {
  local extra=""
  for kv in $FROZEN; do extra="$extra $(both "${kv%%=*}" "${kv#*=}")"; done
  for flag in False True; do
    timeout 600 "$PY" -m habitat_llm.examples.planner_demo \
      --config-name baselines/skill_memory_v2_typed_vllm.yaml --cfg job \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/gate_H.json.gz" \
      $(both operators "$LIB") $(both inside_prior "$PRIOR") $(both typed_examples RS) \
      $(both_plus typed_state "$flag") $extra > "$Q/dryrun_$flag.txt" 2>&1 \
      || { say "REFUSING: hydra rejects the overrides with typed_state=$flag"; tail -5 "$Q/dryrun_$flag.txt" | cut -c1-200; return 1; }
    grep -q "typed_state: $(echo "$flag" | tr 'A-Z' 'a-z')" "$Q/dryrun_$flag.txt" \
      || { say "REFUSING: typed_state=$flag did not reach plan_config"; return 1; }
  done
  say "hydra dry run ok: typed_state reaches plan_config as both False and True"
}
dryrun || exit 6

wait_for_cards || exit 5
line 30b Qwen/Qwen3-VL-30B-A3B-Instruct 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c qwen3-vl-30b 1 3 8071 0.74 \
  > "$Q/line_30b.log" 2>&1 &
A=$!
line 7b Qwen/Qwen2.5-VL-7B-Instruct "" qwen2.5-vl-7b 0 6 8064 0.40 > "$Q/line_7b.log" 2>&1 &
B=$!
echo "$A $B" > "$Q/LINES.pid"
wait "$A"; say "line 30B exited"
wait "$B"; say "line 7B exited"
{
  echo "# Can the model use the H operators once the interface has a word for them?"
  echo
  echo "commit $(git rev-parse --short HEAD); finished $(date '+%m-%d %H:%M'); library $LIB (fixed); switches $FROZEN"
  echo "Only \`typed_state\` moves. Oracle-arm ceiling for the same operator pair: conf_H pc +0.2563,"
  echo "is_clean 0.000 -> 0.981, is_powered_on 0.037 -> 0.926."
  echo; echo '```'
  for m in 30b 7b; do for pool in gate_H conf_H; do for st in off on; do
    d=$Q/cells/${pool}_state_${st}_$m
    printf '%-26s %s\n' "${pool}_state_${st}_$m" "$(cat "$d/CELL.json" 2>/dev/null | head -c 200 || echo '(missing)')"
  done; done; done
  echo '```'; echo; echo "## Alarms"; echo '```'
  grep -h -E "ALARM|WARN|SKIP|REFUSING|ENDPOINT_DEAD|STALL|TIMEOUT" "$Q"/line_*.log "$Q"/cells_*.log 2>/dev/null || echo none
  echo '```'
  for r in "$Q"/keys_*.txt "$Q"/pc_*.txt; do
    [ -s "$r" ] || continue
    echo; echo "## $(basename "$r" .txt)"; echo '```'
    grep -v -E "Gym|gymnasium|Please upgrade|PluginManager|migration_guide" "$r" | head -120
    echo '```'
  done
} > "$Q/STATE_IFACE_SUMMARY.md"
touch "$Q/ALLDONE"
say "summary: $Q/STATE_IFACE_SUMMARY.md"
