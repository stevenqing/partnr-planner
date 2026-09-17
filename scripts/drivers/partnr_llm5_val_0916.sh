#!/usr/bin/env bash
# The 5-operator LLM library (llm5) against iir1 on the frozen typed model arm, val_mini, once per model.
#
# What is being asked (HANDOVER-2026-09-16b §3, priority 1): the privileged arm already showed the single
# LLM-induced is_on_top body is exactly the 20 rule-induced specialisations (gate_ontop and conf_ontop both
# read 0.0000 [0,0], 0 episodes moved), so the open question is only the MODEL arm -- a library of 5
# operators instead of 22, and a predicate menu that gains is_clean and is_powered_on, may change which
# predicate the model asks for. Everything else is held at the frozen configuration
# (tag partnr-comp-freeze-0916): typed interface, RS examples, the three composition switches, same runner,
# same agents, same pool. Only `operators` differs, so the new cell pairs per episode with
#   outputs/cand_iface_0914/val_mini/typed_v7b_RS_final_{7b,30b}
# which were run at commits e24f51e / ec603f9; nothing under our_method/ or habitat_llm/ has changed since.
#
#   Q=outputs/cand_iface_0916b/llm5_val; mkdir -p $Q; cp scripts/drivers/partnr_llm5_val_0916.sh $Q/queue.sh
#   setsid nohup bash $Q/queue.sh > $Q/queue.log 2>&1 < /dev/null &
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PFS=/mnt/pfs/devs/pn5wp/shishuqing
PY=/root/venvs/partnr/bin/python
VLLM=/root/venvs/vllm/bin/python
Q=outputs/cand_iface_0916b/llm5_val
mkdir -p "$Q" "$PFS/tmp" "$PFS/vllm_cache"
export HF_HOME=$PFS/hf HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$PFS/vllm_cache TMPDIR=$PFS/tmp VLLM_ENGINE_READY_TIMEOUT_S=3600
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/v1/models" | grep -q "^200$"; }
root_free_gb () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
gpu_used_mb () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$(( $1 + 1 ))p"; }

FROZEN="typed_stages=True typed_beside=True typed_same_object=True"
LIB=results/partnr_operators_llm5.json
VAL=outputs/cand_iface_0914/val_mini
REP=outputs/cand_iface_0914/val_mini_reports
H7=outputs/headtohead_0913/val_mini_fixed
H30=outputs/headtohead/val_mini
mkdir -p "$REP"

serve () {  # gpu port model revision served util
  local gpu=$1 port=$2 model=$3 revision=$4 name=$5 util=$6 log=$Q/vllm-$5-$2.log
  [ "$(root_free_gb)" -ge 5 ] || { say "ALARM root filesystem $(root_free_gb)G free; not serving $name"; return 1; }
  probe "$port" && { say "REFUSING to serve: :$port already answers"; return 1; }
  # Someone else's service on the card is the usual cause of a CUDA OOM half way in (09-16: the box is
  # no longer ours alone), so the card has to be empty before the endpoint and 24-48 sim processes land on it.
  local used; used=$(gpu_used_mb "$gpu")
  [ "$used" -lt 2000 ] || { say "REFUSING to serve $name: GPU $gpu already holds ${used}MiB"; return 1; }
  local rev=""; [ -n "$revision" ] && rev="--revision $revision"
  (CUDA_VISIBLE_DEVICES=$gpu setsid nohup "$VLLM" -m vllm.entrypoints.openai.api_server --model "$model" $rev \
      --served-model-name "$name" --port "$port" --tensor-parallel-size 1 --max-model-len 16384 \
      --gpu-memory-utilization "$util" --limit-mm-per-prompt '{"image":1}' > "$log" 2>&1 < /dev/null &) > /dev/null 2>&1
  # 50 minutes, then hard failure: loading the same 30B took 5 minutes in the morning and 18 at night
  # (PFS read contention), and 09-16 lost three inductions to a waiter that gave up at 6.7 minutes.
  for _ in $(seq 300); do probe "$port" && break; sleep 10; done
  probe "$port" || { say "ALARM :$port did not come up in 50 min"; tail -8 "$log" | cut -c1-300; return 1; }
  say ":$port up ($name, GPU $gpu); root free $(root_free_gb)G"
}

stop_port () {  # by PID of the endpoint itself: `pkill -f` has killed this shell eight times
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

val_cell () {  # mtag model nothink gpu port procs
  local mtag=$1 model=$2 nothink=$3 gpu=$4 port=$5 procs=$6
  EXAMPLES=RS TAG=final MTAG=$mtag MODEL=$model NO_THINK=$nothink GPU=$gpu PORT=$port PROCS=$procs \
    LIB=$LIB LIBTAG=llm5 SWITCHES="$FROZEN" RESUME=${RESUME:-0} \
    bash scripts/drivers/partnr_typed_val_mini.sh > "$Q/val_$mtag.driver.log" 2>&1 < /dev/null
  say "val $mtag: $(tail -1 "$Q/val_$mtag.driver.log")"
  [ -e "$VAL/typed_v7b_RS_final_${mtag}_llm5/DONE" ]
}

"$PY" - <<'PY'
import gzip, json, os
os.makedirs("outputs/cand_iface_0916b/llm5_val", exist_ok=True)
ids = [str(e["episode_id"]) for e in json.load(gzip.open("data/datasets/partnr_episodes/v0_0/val_mini.json.gz"))["episodes"]]
json.dump({"ids": ids}, open("outputs/cand_iface_0916b/llm5_val/val_mini_pool.json", "w"))
PY

# Per-predicate and paired reports for one model. The whole-pool delta mixes two things -- 19 redundant
# is_on_top bodies dropped, and two predicates (is_clean, is_powered_on) that iir1 could not name at all --
# so the per-key admission reading is what separates them: 327 val_mini episodes carry is_on_top, 13 carry
# is_clean, 7 carry is_powered_on.
reports () {  # mtag old_cell baselines...
  local mtag=$1 old=$2; shift 2
  local new=$VAL/typed_v7b_RS_final_${mtag}_llm5
  [ -e "$new/DONE" ] || { say "SKIP reports for $mtag: cell not complete"; return 0; }
  report llm5_${mtag}_simpair "$PY" scripts/partnr_typed_simpair_report.py --pool val_mini \
      --cell llm5="$new" iir1="$old" "$@" \
      --compare llm5:iir1 $(for b in "$@"; do echo "llm5:${b%%=*}"; done) \
      --json "$REP/val_mini_llm5_vs_iir1_$mtag.json"
  for key in is_on_top is_in_room is_inside is_clean is_powered_on; do
    report llm5_${mtag}_gate_$key "$PY" scripts/partnr_gate_compare.py --pool val_mini \
        --a "$old" --b "$new" --key "$key" --json "$Q/gate_${mtag}_$key.json"
  done
  report llm5_${mtag}_keys "$PY" scripts/partnr_key_admission.py --pool val_mini \
      --cell iir1="$old" --cell llm5="$new" --compare llm5:iir1 --json "$Q/keys_$mtag.json"
  report llm5_${mtag}_failures "$PY" scripts/partnr_failure_classes.py --pool "$Q/val_mini_pool.json" \
      --split val_mini --cell iir1="$old" --cell llm5="$new" --json "$Q/failures_$mtag.json"
}

line_7b () {
  if serve 1 8063 Qwen/Qwen2.5-VL-7B-Instruct "" qwen2.5-vl-7b 0.40; then
    val_cell 7b qwen2.5-vl-7b 0 1 8063 48
    stop_port 8063
    reports 7b "$VAL/typed_v7b_RS_final_7b" \
        stage_beside="$VAL/typed_v7b_RS_stage_beside_7b" react_7b="$H7/react_7b" \
        react_rag_R_7b="$H7/react_rag_R_7b" gmemory_7b="$H7/gmemory_7b" memento_7b="$H7/memento_7b" \
        v2_intent_7b="$H7/v2_accepted_7b"
  fi
  say "line 7B done"
}

line_30b () {
  if serve 3 8071 Qwen/Qwen3-VL-30B-A3B-Instruct 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c qwen3-vl-30b 0.74; then
    reply=$(curl -s -m 120 http://127.0.0.1:8071/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"qwen3-vl-30b","messages":[{"role":"user","content":"Reply with the single word ok."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
    say "30B smoke: $(echo "$reply" | head -c 200)"
    if echo "$reply" | grep -q '"content"' && ! echo "$reply" | grep -q '<think>'; then
      # 24 processes beside a 69 GiB endpoint: 48 is what OOMed this cell at 278/369 on 09-16.
      val_cell 30b qwen3-vl-30b 1 3 8071 24
      stop_port 8071
      reports 30b "$VAL/typed_v7b_RS_final_30b" \
          react="$H30/react" react_rag_R="$H30/react_rag_R" gmemory="$H30/gmemory" memento="$H30/memento" \
          v2_intent="$H30/v2_intent" v2_prompt="$H30/v2_prompt"
    else
      say "ALARM 30B smoke failed; val 30B skipped"
      stop_port 8071
    fi
  fi
  say "line 30B done"
}

say "queue start (commit $(git rev-parse --short HEAD)); root free $(root_free_gb)G"
[ -s "$LIB" ] || { say "REFUSING: $LIB missing"; exit 6; }
[ "$("$PY" -c "import json; print(len(json.load(open('$LIB'))['operators']))")" = 5 ] \
  || { say "REFUSING: $LIB is not the 5-operator library"; exit 6; }
grep -q "LIBTAG" scripts/drivers/partnr_typed_val_mini.sh || { say "REFUSING: val driver has no LIB switch"; exit 6; }
for c in "$VAL/typed_v7b_RS_final_7b" "$VAL/typed_v7b_RS_final_30b"; do
  [ -e "$c/DONE" ] || say "WARN baseline cell $c has no DONE marker"
done
line_7b > "$Q/line_7b.log" 2>&1 &
A=$!
line_30b > "$Q/line_30b.log" 2>&1 &
B=$!
echo "$A $B" > "$Q/LINES.pid"
wait "$A"; say "line 7B exited"
wait "$B"; say "line 30B exited"
{
  echo "# llm5 (5 operators) against iir1 (22) on the frozen typed arm, val_mini"
  echo; echo "commit $(git rev-parse --short HEAD); finished $(date '+%m-%d %H:%M'); switches: $FROZEN; library: $LIB"
  echo; echo '```'
  for m in 7b 30b; do
    printf 'llm5_%s : ' $m; cat "$VAL/typed_v7b_RS_final_${m}_llm5/CELL.json" 2>/dev/null || echo "(no CELL.json)"
    printf 'iir1_%s : ' $m; cat "$VAL/typed_v7b_RS_final_$m/CELL.json" 2>/dev/null || echo "(no CELL.json)"
  done
  echo '```'; echo; echo "## Alarms"; echo '```'
  grep -h -E "ALARM|WARN|SKIP|REFUSING" "$Q/line_7b.log" "$Q/line_30b.log" "$Q"/val_*.driver.log 2>/dev/null || echo none
  echo '```'
  for r in "$Q"/llm5_*_simpair.txt "$Q"/llm5_*_keys.txt "$Q"/llm5_*_gate_*.txt "$Q"/llm5_*_failures.txt; do
    [ -s "$r" ] || continue
    echo; echo "## $(basename "$r" .txt)"; echo '```'
    grep -v -E "Gym|gymnasium|Please upgrade|PluginManager|migration_guide" "$r" | head -160
    echo '```'
  done
} > "$Q/LLM5_VAL_SUMMARY.md"
touch "$Q/ALLDONE"
say "summary: $Q/LLM5_VAL_SUMMARY.md"
