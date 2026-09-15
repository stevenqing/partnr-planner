#!/usr/bin/env bash
# The frozen compositional configuration on val_mini, once per model. Do not tune on these.
#
# Frozen 2026-09-16 (code as of 158a313, tag partnr-comp-freeze-0916): typed interface, RS examples, R-only
# library iir1 and inside prior, plus the three composition switches read off the instruction:
#   typed_stages=True  typed_beside=True  typed_same_object=True
#
#   line 7B  (GPU 1): 7B :8063 -> val_mini cell typed_v7b_RS_final_7b -> reports against the stage+beside
#                     val cell, typed RS, and the four 7B baselines -> stop :8063
#   line 30B (GPU 0): waits for overnight2 (the train 30B cells) to finish -> Qwen3-VL-30B :8071 -> val_mini
#                     cell typed_v7b_RS_final_30b -> reports against the 30B baselines of outputs/headtohead
#                     -> stop :8071
#
#   Q=outputs/cand_iface_0916/val_final; mkdir -p $Q; cp scripts/drivers/partnr_val_final_0916.sh $Q/queue.sh
#   setsid nohup bash $Q/queue.sh > $Q/queue.log 2>&1 < /dev/null &
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PFS=/mnt/pfs/devs/pn5wp/shishuqing
PY=/root/venvs/partnr/bin/python
VLLM=/root/venvs/vllm/bin/python
Q=outputs/cand_iface_0916/val_final
mkdir -p "$Q" "$PFS/tmp" "$PFS/vllm_cache"
export HF_HOME=$PFS/hf HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$PFS/vllm_cache TMPDIR=$PFS/tmp VLLM_ENGINE_READY_TIMEOUT_S=3600
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/v1/models" | grep -q "^200$"; }
root_free_gb () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }

FROZEN="typed_stages=True typed_beside=True typed_same_object=True"
VAL=outputs/cand_iface_0914/val_mini
REP=outputs/cand_iface_0914/val_mini_reports
H7=outputs/headtohead_0913/val_mini_fixed
H30=outputs/headtohead/val_mini
OV2=outputs/cand_iface_0916/overnight2

serve () {  # gpu port model revision served util
  local gpu=$1 port=$2 model=$3 revision=$4 name=$5 util=$6 log=$Q/vllm-$5-$2.log
  [ "$(root_free_gb)" -ge 5 ] || { say "ALARM root filesystem $(root_free_gb)G free; not serving $name"; return 1; }
  probe "$port" && { say "REFUSING to serve: :$port already answers"; return 1; }
  local rev=""; [ -n "$revision" ] && rev="--revision $revision"
  (CUDA_VISIBLE_DEVICES=$gpu setsid nohup "$VLLM" -m vllm.entrypoints.openai.api_server --model "$model" $rev \
      --served-model-name "$name" --port "$port" --tensor-parallel-size 1 --max-model-len 16384 \
      --gpu-memory-utilization "$util" --limit-mm-per-prompt '{"image":1}' > "$log" 2>&1 < /dev/null &) > /dev/null 2>&1
  for _ in $(seq 300); do probe "$port" && break; sleep 10; done
  probe "$port" || { say "ALARM :$port did not come up in 50 min"; tail -8 "$log" | cut -c1-300; return 1; }
  say ":$port up ($name, GPU $gpu); root free $(root_free_gb)G"
}

stop_port () {
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

val_cell () {  # mtag model nothink gpu port
  local mtag=$1 model=$2 nothink=$3 gpu=$4 port=$5
  EXAMPLES=RS TAG=final MTAG=$mtag MODEL=$model NO_THINK=$nothink GPU=$gpu PORT=$port PROCS=48 SWITCHES="$FROZEN" \
    bash scripts/drivers/partnr_typed_val_mini.sh > "$Q/val_$mtag.driver.log" 2>&1 < /dev/null
  say "val $mtag: $(tail -1 "$Q/val_$mtag.driver.log")"
  [ -e "$VAL/typed_v7b_RS_final_$mtag/DONE" ]
}

"$PY" - <<'PY'
import gzip, json
ids = [str(e["episode_id"]) for e in json.load(gzip.open("data/datasets/partnr_episodes/v0_0/val_mini.json.gz"))["episodes"]]
json.dump({"ids": ids}, open("outputs/cand_iface_0916/val_final/val_mini_pool.json", "w"))
PY

line_7b () {
  if serve 1 8063 Qwen/Qwen2.5-VL-7B-Instruct "" qwen2.5-vl-7b 0.40; then
    val_cell 7b qwen2.5-vl-7b 0 1 8063
    stop_port 8063
    NEW=$VAL/typed_v7b_RS_final_7b
    report val7b_report $PY scripts/partnr_typed_simpair_report.py --pool val_mini \
        --cell final="$NEW" stage_beside="$VAL/typed_v7b_RS_stage_beside_7b" typed_RS="$VAL/typed_v7b_RS_7b" \
               react_7b="$H7/react_7b" react_rag_R_7b="$H7/react_rag_R_7b" gmemory_7b="$H7/gmemory_7b" \
               memento_7b="$H7/memento_7b" v2_intent_7b="$H7/v2_accepted_7b" \
        --compare final:stage_beside final:typed_RS final:react_7b final:react_rag_R_7b final:gmemory_7b \
                  final:memento_7b final:v2_intent_7b \
        --json "$REP/val_mini_typed_RS_final_7b.json"
    report val7b_pairs $PY scripts/partnr_pair_cells.py --pool "$Q/val_mini_pool.json" --split val_mini \
        --cell final="$NEW" --cell stage_beside="$VAL/typed_v7b_RS_stage_beside_7b" --cell typed_RS="$VAL/typed_v7b_RS_7b" \
        --compare final:stage_beside final:typed_RS --json "$Q/val7b_pairs.json"
    report val7b_failures $PY scripts/partnr_failure_classes.py --pool "$Q/val_mini_pool.json" --split val_mini \
        --cell typed_RS="$VAL/typed_v7b_RS_7b" --cell stage_beside="$VAL/typed_v7b_RS_stage_beside_7b" --cell final="$NEW" \
        --json "$Q/val7b_failures.json"
  fi
  say "line 7B done"
}

line_30b () {
  local qpid; qpid=$(cat "$OV2/QUEUE.pid" 2>/dev/null)
  while [ ! -e "$OV2/ALLDONE" ]; do
    if [ -n "$qpid" ] && ! kill -0 "$qpid" 2>/dev/null; then
      sleep 60; [ -e "$OV2/ALLDONE" ] || { say "ALARM overnight2 queue $qpid gone without ALLDONE"; break; }
    fi
    sleep 120
  done
  probe 8071 && { say "ALARM :8071 still answers after overnight2; not starting a second 30B"; say "line 30B done"; return; }
  if serve 0 8071 Qwen/Qwen3-VL-30B-A3B-Instruct 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c qwen3-vl-30b 0.74; then
    reply=$(curl -s -m 120 http://127.0.0.1:8071/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"qwen3-vl-30b","messages":[{"role":"user","content":"Reply with the single word ok."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
    say "30B smoke: $(echo "$reply" | head -c 200)"
    if echo "$reply" | grep -q '"content"' && ! echo "$reply" | grep -q '<think>'; then
      val_cell 30b qwen3-vl-30b 1 0 8071
      NEW=$VAL/typed_v7b_RS_final_30b
      report val30b_report $PY scripts/partnr_typed_simpair_report.py --pool val_mini \
          --cell final_30b="$NEW" react="$H30/react" react_rag_R="$H30/react_rag_R" gmemory="$H30/gmemory" \
                 memento="$H30/memento" v2_intent="$H30/v2_intent" v2_prompt="$H30/v2_prompt" final_7b="$VAL/typed_v7b_RS_final_7b" \
          --compare final_30b:react final_30b:react_rag_R final_30b:gmemory final_30b:memento final_30b:v2_intent \
                    final_30b:v2_prompt final_30b:final_7b \
          --json "$REP/val_mini_typed_RS_final_30b.json"
      report val30b_failures $PY scripts/partnr_failure_classes.py --pool "$Q/val_mini_pool.json" --split val_mini \
          --cell final_30b="$NEW" --json "$Q/val30b_failures.json"
    else
      say "ALARM 30B smoke failed; val 30B skipped"
    fi
    stop_port 8071
  fi
  say "line 30B done"
}

say "queue start (commit $(git rev-parse --short HEAD)); root free $(root_free_gb)G"
grep -q "strictly earlier stage" our_method/skill_memory_v2/partnr_typed_goals.py \
  && grep -q "typed_same_object" our_method/skill_memory_v2/partnr_planner.py \
  && grep -q "MTAG" scripts/drivers/partnr_typed_val_mini.sh \
  || { say "REFUSING: frozen code or val driver not synced"; exit 6; }
line_7b > "$Q/line_7b.log" 2>&1 &
A=$!
line_30b > "$Q/line_30b.log" 2>&1 &
B=$!
echo "$A $B" > "$Q/LINES.pid"
wait "$A"; say "line 7B exited"
wait "$B"; say "line 30B exited"
{
  echo "# PARTNR frozen compositional config on val_mini (once per model)"
  echo; echo "commit $(git rev-parse --short HEAD); finished $(date '+%m-%d %H:%M'); switches: $FROZEN"
  echo; echo '```'
  for m in 7b 30b; do printf 'final_%s: ' $m; cat "$VAL/typed_v7b_RS_final_$m/CELL.json" 2>/dev/null || echo "(no CELL.json)"; done
  echo '```'; echo; echo "## Alarms"; echo '```'
  grep -h -E "ALARM|WARN|SKIP|REFUSING" "$Q/line_7b.log" "$Q/line_30b.log" "$Q"/val_*.driver.log 2>/dev/null || echo none
  echo '```'
  for r in val7b_report val7b_pairs val7b_failures val30b_report val30b_failures; do
    [ -s "$Q/$r.txt" ] || continue
    echo; echo "## $r"; echo '```'
    grep -v -E "Gym|gymnasium|Please upgrade|PluginManager|migration_guide" "$Q/$r.txt" | head -200
    echo '```'
  done
} > "$Q/VAL_FINAL_SUMMARY.md"
touch "$Q/ALLDONE"
say "summary: $Q/VAL_FINAL_SUMMARY.md"
