#!/usr/bin/env bash
# PARTNR overnight queue, 2026-09-16. Two GPU lines, each one endpoint at a time, everything waits
# by PID / CELL.json and alarms when a watched runner is gone without its CELL.json.
#
#   line A (GPU 1): wait priv_partial + priv_full (running) -> privileged diagnosis reports
#                   -> 7B endpoint :8063 -> train_mini cell stage_beside_same (typed_same_object fix)
#                   -> fix reports -> stop :8063
#   line B (GPU 0): wait the val_mini stage+beside cell (running) -> val_mini report, once
#                   -> stop 7B :8061 -> Qwen3-VL-30B :8071 -> train_mini 30B cells, one after another:
#                      nostage_30b, stage_beside_30b, stage_beside_same_30b -> 30B reports -> stop :8071
#   end: OVERNIGHT_SUMMARY.md from the report texts and every CELL.json.
#
# Launched from a copy (bash re-reads a running script by byte offset):
#   cp scripts/drivers/partnr_overnight_0916.sh outputs/cand_iface_0916/overnight/queue.sh
#   setsid nohup bash outputs/cand_iface_0916/overnight/queue.sh > outputs/cand_iface_0916/overnight/queue.log 2>&1 < /dev/null &
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PY=/root/venvs/partnr/bin/python
VLLM=/root/venvs/vllm/bin/python
Q=outputs/cand_iface_0916/overnight
mkdir -p "$Q"
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/v1/models" | grep -q "^200$"; }

POOL=outputs/cand_iface_0915/stage_simpair_train_mini/episodes.json
ABL=outputs/cand_iface_0915/ablate_train_mini
SIMP=outputs/cand_iface_0915/stage_simpair_train_mini
DIAG=outputs/cand_iface_0916/priv_diag_train_mini
CEIL=outputs/cand_iface_0914/train_mini/ceiling
VAL=outputs/cand_iface_0914/val_mini
VALCELL=$VAL/typed_v7b_RS_stage_beside_7b
H2H=outputs/headtohead_0913/val_mini_fixed
M30=outputs/cand_iface_0916/model30b_train_mini

# Wait for a cell started elsewhere: its runner PID file, then its CELL.json (the driver writes it
# after the runner exits). Returns 1 and says so when the runner is gone and CELL.json never comes.
wait_cell () {  # dir label max_hours
  local dir=$1 label=$2 hours=$3 start; start=$(date +%s)
  local pid; pid=$(cat "$dir/PID" 2>/dev/null)
  [ -z "$pid" ] && { say "ALARM $label: no PID file in $dir"; return 1; }
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    [ $(( $(date +%s) - start )) -ge $((hours * 3600)) ] && { say "ALARM $label: runner $pid still alive after ${hours}h"; return 1; }
  done
  for _ in $(seq 20); do [ -s "$dir/CELL.json" ] && break; sleep 30; done
  [ -s "$dir/CELL.json" ] || { say "ALARM $label: runner $pid exited, no CELL.json after 10 min"; return 1; }
  say "$label finished: $(cat "$dir/CELL.json")"
  [ -e "$dir/DONE" ] || say "WARN $label: no DONE file (incomplete or killed)"
  return 0
}

serve () {  # gpu port model served util logname -> echoes pid, returns 1 if not up in 30 min
  local gpu=$1 port=$2 model=$3 name=$4 util=$5 log=$6
  probe "$port" && { say "REFUSING to serve: :$port already answers"; return 1; }
  (CUDA_VISIBLE_DEVICES=$gpu setsid nohup "$VLLM" -m vllm.entrypoints.openai.api_server --model "$model" \
      --served-model-name "$name" --port "$port" --tensor-parallel-size 1 --max-model-len 16384 \
      --gpu-memory-utilization "$util" --limit-mm-per-prompt '{"image":1}' > "$log" 2>&1 < /dev/null & echo $! > "$Q/serve_$port.pid") > /dev/null 2>&1
  sleep 5
  for _ in $(seq 180); do probe "$port" && break; sleep 10; done
  probe "$port" || { say "ALARM :$port did not come up; tail:"; tail -5 "$log"; return 1; }
  say ":$port up ($name on GPU $gpu, util $util, pid $(cat "$Q/serve_$port.pid"))"
}

stop_port () {  # port -- by the PID of the api_server listening there, never by pattern kill
  local port=$1 pid
  pid=$(ps -eo pid,args | awk -v p="--port $port" '$0 ~ /vllm.entrypoints.openai.api_server/ && index($0, p) {print $1}' | head -1)
  [ -z "$pid" ] && { say ":$port nothing to stop"; return 0; }
  kill "$pid"; for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
  sleep 10; say ":$port stopped (pid $pid)"
}

cell () {  # blocking ablate-driver cell: ROOT_OUT CELL GPU PORT MODEL NO_THINK SWITCHES
  local root=$1 name=$2 gpu=$3 port=$4 model=$5 nothink=$6 switches=$7
  probe "$port" || { say "SKIP $name: :$port not up"; return 1; }
  mkdir -p "$root"
  ROOT_OUT=$root CELL=$name GPU=$gpu PORT=$port MODEL=$model NO_THINK=$nothink SWITCHES="$switches" \
    bash scripts/drivers/partnr_typed_ablate_simpair_train_mini.sh > "$root/$name.driver.log" 2>&1 < /dev/null
  say "cell $name: $(tail -1 "$root/$name.driver.log")"
  [ -e "$root/$name/DONE" ]
}

report () {  # name command... -> text in $Q/<name>.txt, JSON wherever the command writes it
  local name=$1; shift
  say "report $name"
  timeout 3600 "$@" > "$Q/$name.txt" 2>&1 < /dev/null || say "WARN report $name exit $?"
}

# ------------------------------------------------------------------ line A (GPU 1)
line_a () {
  wait_cell "$DIAG/priv_partial" priv_partial 7
  wait_cell "$DIAG/priv_full" priv_full 7
  report priv_pairs $PY scripts/partnr_pair_cells.py --pool "$POOL" \
      --cell ceiling="$CEIL" --cell priv_full="$DIAG/priv_full" --cell priv_partial="$DIAG/priv_partial" \
      --cell typed_stage_beside="$ABL/stage_beside" --cell typed_nostage="$SIMP/typed_RS_nostage" \
      --compare ceiling:priv_full priv_full:priv_partial priv_partial:typed_stage_beside \
                ceiling:typed_stage_beside typed_stage_beside:typed_nostage \
      --json "$Q/priv_pairs.json"
  report priv_failures $PY scripts/partnr_failure_classes.py --pool "$POOL" \
      --cell priv_full="$DIAG/priv_full" --cell priv_partial="$DIAG/priv_partial" \
      --cell typed_stage_beside="$ABL/stage_beside" --json "$Q/priv_failures.json"
  if serve 1 8063 Qwen/Qwen2.5-VL-7B-Instruct qwen2.5-vl-7b 0.40 /root/vllm-qwen2.5-vl-7b-8063.log; then
    cell "$ABL" stage_beside_same 1 8063 qwen2.5-vl-7b 0 "typed_stages=True typed_beside=True typed_same_object=True"
    report fix_pairs $PY scripts/partnr_pair_cells.py --pool "$POOL" \
        --cell same="$ABL/stage_beside_same" --cell beside="$ABL/stage_beside" --cell nostage="$SIMP/typed_RS_nostage" \
        --cell rep="$ABL/stage_rep" --cell stage="$SIMP/typed_RS_stage" \
        --compare same:beside same:nostage rep:stage --json "$Q/fix_pairs.json"
    report fix_failures $PY scripts/partnr_failure_classes.py --pool "$POOL" \
        --cell beside="$ABL/stage_beside" --cell same="$ABL/stage_beside_same" --json "$Q/fix_failures.json"
    stop_port 8063
  fi
  say "line A done"
}

# ------------------------------------------------------------------ line B (GPU 0)
line_b () {
  wait_cell "$VALCELL" val_stage_beside 8
  "$PY" - <<'PY'
import gzip, json
ids = [str(e["episode_id"]) for e in json.load(gzip.open("data/datasets/partnr_episodes/v0_0/val_mini.json.gz"))["episodes"]]
json.dump({"ids": ids}, open("outputs/cand_iface_0916/overnight/val_mini_pool.json", "w"))
PY
  report val_report $PY scripts/partnr_typed_simpair_report.py --pool val_mini \
      --cell typed_RS_stage_beside="$VALCELL" typed_RS="$VAL/typed_v7b_RS_7b" typed_R="$VAL/typed_v7b_R_7b" \
             react_7b="$H2H/react_7b" react_rag_R_7b="$H2H/react_rag_R_7b" gmemory_7b="$H2H/gmemory_7b" \
             memento_7b="$H2H/memento_7b" v2_intent_7b="$H2H/v2_accepted_7b" \
      --compare typed_RS_stage_beside:typed_RS typed_RS_stage_beside:react_7b typed_RS_stage_beside:react_rag_R_7b \
                typed_RS_stage_beside:gmemory_7b typed_RS_stage_beside:memento_7b typed_RS_stage_beside:v2_intent_7b \
      --json outputs/cand_iface_0914/val_mini_reports/val_mini_typed_RS_stage_beside.json
  report val_pairs $PY scripts/partnr_pair_cells.py --pool "$Q/val_mini_pool.json" --split val_mini \
      --cell new="$VALCELL" --cell typed_RS="$VAL/typed_v7b_RS_7b" --cell react_7b="$H2H/react_7b" \
      --compare new:typed_RS new:react_7b --json "$Q/val_pairs.json"
  report val_failures $PY scripts/partnr_failure_classes.py --pool "$Q/val_mini_pool.json" --split val_mini \
      --cell typed_RS="$VAL/typed_v7b_RS_7b" --cell stage_beside="$VALCELL" --json "$Q/val_failures.json"
  stop_port 8061
  if serve 0 8071 Qwen/Qwen3-VL-30B-A3B-Instruct qwen3-vl-30b 0.74 /root/vllm-qwen3-vl-30b-8071.log; then
    # A reasoning template that slipped through would empty every answer; check one before 214 episodes.
    reply=$(curl -s -m 120 http://127.0.0.1:8071/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"qwen3-vl-30b","messages":[{"role":"user","content":"Reply with the single word ok."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
    say "30B smoke: $(echo "$reply" | head -c 300)"
    if echo "$reply" | grep -q '"content"' && ! echo "$reply" | grep -q '<think>'; then
      cell "$M30" nostage_30b 0 8071 qwen3-vl-30b 1 "typed_stages=False"
      cell "$M30" stage_beside_30b 0 8071 qwen3-vl-30b 1 "typed_stages=True typed_beside=True"
      cell "$M30" stage_beside_same_30b 0 8071 qwen3-vl-30b 1 "typed_stages=True typed_beside=True typed_same_object=True"
      report m30_pairs $PY scripts/partnr_pair_cells.py --pool "$POOL" \
          --cell nostage_30b="$M30/nostage_30b" --cell stage_beside_30b="$M30/stage_beside_30b" \
          --cell same_30b="$M30/stage_beside_same_30b" --cell nostage_7b="$SIMP/typed_RS_nostage" \
          --cell stage_beside_7b="$ABL/stage_beside" --cell ceiling="$CEIL" \
          --compare stage_beside_30b:nostage_30b same_30b:stage_beside_30b stage_beside_30b:stage_beside_7b \
                    nostage_30b:nostage_7b ceiling:stage_beside_30b --json "$Q/m30_pairs.json"
      report m30_failures $PY scripts/partnr_failure_classes.py --pool "$POOL" \
          --cell nostage_30b="$M30/nostage_30b" --cell stage_beside_30b="$M30/stage_beside_30b" \
          --cell same_30b="$M30/stage_beside_same_30b" --json "$Q/m30_failures.json"
    else
      say "ALARM 30B smoke failed; 30B cells skipped"
    fi
    stop_port 8071
  fi
  say "line B done"
}

say "queue start (commit $(git rev-parse --short HEAD))"
line_a > "$Q/line_a.log" 2>&1 &
A=$!
line_b > "$Q/line_b.log" 2>&1 &
B=$!
echo "$A $B" > "$Q/LINES.pid"
wait "$A"; say "line A exited: $(tail -1 "$Q/line_a.log")"
wait "$B"; say "line B exited: $(tail -1 "$Q/line_b.log")"

# ------------------------------------------------------------------ summary
{
  echo "# PARTNR overnight 2026-09-16"
  echo
  echo "commit $(git rev-parse --short HEAD); finished $(date '+%m-%d %H:%M')"
  echo
  echo "## Cells"
  echo '```'
  for f in "$VALCELL" "$DIAG/priv_partial" "$DIAG/priv_full" "$ABL/stage_beside_same" "$M30/nostage_30b" "$M30/stage_beside_30b" "$M30/stage_beside_same_30b"; do
    printf '%s: ' "$f"; cat "$f/CELL.json" 2>/dev/null || echo "(no CELL.json)"
    [ -e "$f/DONE" ] || echo "   ^ no DONE"
  done
  echo '```'
  echo
  echo "## Alarms and warnings"
  echo '```'
  grep -h -E "ALARM|WARN|SKIP|REFUSING" "$Q/line_a.log" "$Q/line_b.log" || echo "none"
  echo '```'
  for r in val_pairs val_report val_failures priv_pairs priv_failures fix_pairs fix_failures m30_pairs m30_failures; do
    [ -s "$Q/$r.txt" ] || continue
    echo; echo "## $r"; echo '```'
    grep -v -E "Gym|gymnasium|Please upgrade|PluginManager|migration_guide" "$Q/$r.txt" | head -220
    echo '```'
  done
} > "$Q/OVERNIGHT_SUMMARY.md"
say "summary written: $Q/OVERNIGHT_SUMMARY.md"
touch "$Q/ALLDONE"
