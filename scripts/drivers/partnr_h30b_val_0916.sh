#!/usr/bin/env bash
# The third arm: h30b (9 operators after normalisation) on the frozen typed arm, val_mini, once per model.
#
# Why it exists. `partnr_llm5_val_0916.sh` moves two things at once, so a negative reading there cannot be
# attributed. Normalised, the three libraries are:
#
#   iir1   7 operators, menu {is_on_top, is_inside, is_in_room}          5 of them is_on_top (1 plain + 4 shut)
#   h30b   9 operators, menu + {is_clean, is_powered_on}                 the same 5 is_on_top bodies
#   llm5   5 operators, same menu as h30b                                1 is_on_top body, shut_variant() -> 0
#
# so h30b - iir1 prices the menu alone (13 val_mini episodes carry is_clean, 7 carry is_powered_on) and
# llm5 - h30b prices dropping the 4 redundant is_on_top specialisations alone. Everything else is the frozen
# configuration (tag partnr-comp-freeze-0916) and the cells pair per episode with both existing arms.
#
#   Q=outputs/cand_iface_0916b/h30b_val; mkdir -p $Q; cp scripts/drivers/partnr_h30b_val_0916.sh $Q/queue.sh
#   setsid nohup bash $Q/queue.sh > $Q/queue.log 2>&1 < /dev/null &
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PFS=/mnt/pfs/devs/pn5wp/shishuqing
PY=/root/venvs/partnr/bin/python
VLLM=/root/venvs/vllm/bin/python
Q=outputs/cand_iface_0916b/h30b_val
Q1=outputs/cand_iface_0916b/llm5_val     # the queue this one waits behind on the 30B card
mkdir -p "$Q" "$PFS/tmp" "$PFS/vllm_cache"
export HF_HOME=$PFS/hf HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$PFS/vllm_cache TMPDIR=$PFS/tmp VLLM_ENGINE_READY_TIMEOUT_S=3600
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/v1/models" | grep -q "^200$"; }
root_free_gb () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
gpu_used_mb () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$(( $1 + 1 ))p"; }

FROZEN="typed_stages=True typed_beside=True typed_same_object=True"
LIB=results/partnr_operators_h30b.json
VAL=outputs/cand_iface_0914/val_mini
REP=outputs/cand_iface_0914/val_mini_reports
mkdir -p "$REP"

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

stop_port () {  # by PID: `pkill -f` matches this shell's own command line (eight times now)
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
    LIB=$LIB LIBTAG=h30b SWITCHES="$FROZEN" RESUME=${RESUME:-0} \
    bash scripts/drivers/partnr_typed_val_mini.sh > "$Q/val_$mtag.driver.log" 2>&1 < /dev/null
  say "val $mtag: $(tail -1 "$Q/val_$mtag.driver.log")"
  [ -e "$VAL/typed_v7b_RS_final_${mtag}_h30b/DONE" ]
}

# The three-arm reading for one model. h30b:iir1 is the menu, llm5:h30b is the 4 dropped is_on_top bodies,
# llm5:iir1 is the combined one the other queue already reports.
reports_3arm () {  # mtag
  local mtag=$1
  local a=$VAL/typed_v7b_RS_final_$mtag          # iir1
  local b=$VAL/typed_v7b_RS_final_${mtag}_h30b   # h30b
  local c=$VAL/typed_v7b_RS_final_${mtag}_llm5   # llm5
  for d in "$a" "$b" "$c"; do
    [ -e "$d/DONE" ] || { say "SKIP 3-arm reports for $mtag: $d has no DONE"; return 0; }
  done
  report h30b_${mtag}_simpair "$PY" scripts/partnr_typed_simpair_report.py --pool val_mini \
      --cell iir1="$a" h30b="$b" llm5="$c" --compare h30b:iir1 llm5:h30b llm5:iir1 \
      --json "$REP/val_mini_3arm_$mtag.json"
  # Only the three keys the libraries actually differ on; each gate_compare re-reads 369 planner-logs.
  for key in is_on_top is_clean is_powered_on; do
    report h30b_${mtag}_menu_$key "$PY" scripts/partnr_gate_compare.py --pool val_mini \
        --a "$a" --b "$b" --key "$key" --json "$Q/menu_${mtag}_$key.json"
    report h30b_${mtag}_redundant_$key "$PY" scripts/partnr_gate_compare.py --pool val_mini \
        --a "$b" --b "$c" --key "$key" --json "$Q/redundant_${mtag}_$key.json"
  done
  report h30b_${mtag}_keys "$PY" scripts/partnr_key_admission.py --pool val_mini \
      --cell iir1="$a" --cell h30b="$b" --cell llm5="$c" \
      --compare h30b:iir1 llm5:h30b llm5:iir1 --json "$Q/keys_3arm_$mtag.json"
}

line_7b () {
  if serve 6 8064 Qwen/Qwen2.5-VL-7B-Instruct "" qwen2.5-vl-7b 0.40; then
    val_cell 7b qwen2.5-vl-7b 0 6 8064 48
    stop_port 8064
    reports_3arm 7b
  fi
  say "line 7B done"
}

# The 30B card is busy with the llm5 cell. Wait for THAT line to finish -- by its own marker and by the
# endpoint going away, never by matching a script name (a name matcher blocks on this very shell, and
# cross-endpoint waiters have stalled a whole night before). Six hours, then give up without touching the card.
wait_for_card () {
  local deadline=$(( $(date +%s) + 21600 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if grep -q "line 30B done" "$Q1/line_30b.log" 2>/dev/null || [ -e "$Q1/ALLDONE" ]; then
      probe 8071 || { say "llm5 30B line finished and :8071 is gone"; return 0; }
    fi
    sleep 120
  done
  say "ALARM waited 6h for the 30B card; llm5 line never signalled done"
  return 1
}

line_30b () {
  wait_for_card || { say "line 30B done"; return; }
  local used; used=$(gpu_used_mb 3)
  [ "$used" -lt 2000 ] || { say "ALARM GPU 3 holds ${used}MiB (someone else took it); 30B h30b skipped"; say "line 30B done"; return; }
  if serve 3 8071 Qwen/Qwen3-VL-30B-A3B-Instruct 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c qwen3-vl-30b 0.74; then
    reply=$(curl -s -m 120 http://127.0.0.1:8071/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"qwen3-vl-30b","messages":[{"role":"user","content":"Reply with the single word ok."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
    say "30B smoke: $(echo "$reply" | head -c 160)"
    if echo "$reply" | grep -q '"content"' && ! echo "$reply" | grep -q '<think>'; then
      val_cell 30b qwen3-vl-30b 1 3 8071 24
      stop_port 8071
      reports_3arm 30b
    else
      say "ALARM 30B smoke failed; val 30B skipped"
      stop_port 8071
    fi
  fi
  say "line 30B done"
}

say "queue start (commit $(git rev-parse --short HEAD)); root free $(root_free_gb)G"
[ -s "$LIB" ] || { say "REFUSING: $LIB missing"; exit 6; }
[ "$("$PY" -c "import json; print(len(json.load(open('$LIB'))['operators']))")" = 24 ] \
  || { say "REFUSING: $LIB is not the 24-entry h30b library"; exit 6; }
grep -q "LIBTAG" scripts/drivers/partnr_typed_val_mini.sh || { say "REFUSING: val driver has no LIB switch"; exit 6; }
line_7b > "$Q/line_7b.log" 2>&1 &
A=$!
line_30b > "$Q/line_30b.log" 2>&1 &
B=$!
echo "$A $B" > "$Q/LINES.pid"
wait "$A"; say "line 7B exited"
wait "$B"; say "line 30B exited"
{
  echo "# Three arms on the frozen typed arm, val_mini: iir1 (7 ops) / h30b (9) / llm5 (5)"
  echo
  echo "commit $(git rev-parse --short HEAD); finished $(date '+%m-%d %H:%M'); switches: $FROZEN"
  echo "h30b - iir1 = the two H predicates on the menu; llm5 - h30b = the 4 redundant is_on_top bodies dropped."
  echo; echo '```'
  for m in 7b 30b; do
    for v in "" _h30b _llm5; do
      printf '%-8s %-6s : ' "$m" "${v:-iir1}"; cat "$VAL/typed_v7b_RS_final_$m$v/CELL.json" 2>/dev/null || echo "(no CELL.json)"
    done
  done
  echo '```'; echo; echo "## Alarms"; echo '```'
  grep -h -E "ALARM|WARN|SKIP|REFUSING" "$Q/line_7b.log" "$Q/line_30b.log" "$Q"/val_*.driver.log 2>/dev/null || echo none
  echo '```'
  for r in "$Q"/h30b_*_simpair.txt "$Q"/h30b_*_keys.txt "$Q"/h30b_*_menu_*.txt "$Q"/h30b_*_redundant_*.txt; do
    [ -s "$r" ] || continue
    echo; echo "## $(basename "$r" .txt)"; echo '```'
    grep -v -E "Gym|gymnasium|Please upgrade|PluginManager|migration_guide" "$r" | head -160
    echo '```'
  done
} > "$Q/H30B_VAL_SUMMARY.md"
touch "$Q/ALLDONE"
say "summary: $Q/H30B_VAL_SUMMARY.md"
