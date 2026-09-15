#!/usr/bin/env bash
# PARTNR overnight queue, second pass, 2026-09-16 early morning.
#
# The first queue's 30B endpoint was launched without HF_HOME, started re-downloading the model into
# /root/.cache on the 197G container root, filled it at 02:09 and died -- which also disconnected
# mutagen and made /tmp unwritable. This pass therefore keeps every cache, temp file and log on
# /mnt/pfs and refuses to start a model while the root filesystem is short of space.
#
#   line A (GPU 1): 7B :8063 -> train_mini cell stage_beside_same2 (same_as only across stages, 158a313)
#                   -> reports against stage_beside_same (first version), stage_beside, nostage -> stop :8063
#   line B (GPU 0): Qwen3-VL-30B :8071 (revision pinned) -> smoke -> nostage_30b, stage_beside_30b,
#                   stage_beside_same2_30b -> reports -> stop :8071
#
#   Q=outputs/cand_iface_0916/overnight2; cp scripts/drivers/partnr_overnight_0916b.sh $Q/queue.sh
#   setsid nohup bash $Q/queue.sh > $Q/queue.log 2>&1 < /dev/null &
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PFS=/mnt/pfs/devs/pn5wp/shishuqing
PY=/root/venvs/partnr/bin/python
VLLM=/root/venvs/vllm/bin/python
Q=outputs/cand_iface_0916/overnight2
mkdir -p "$Q" "$PFS/tmp" "$PFS/vllm_cache"
export HF_HOME=$PFS/hf HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$PFS/vllm_cache TMPDIR=$PFS/tmp
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/v1/models" | grep -q "^200$"; }
root_free_gb () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }

POOL=outputs/cand_iface_0915/stage_simpair_train_mini/episodes.json
ABL=outputs/cand_iface_0915/ablate_train_mini
SIMP=outputs/cand_iface_0915/stage_simpair_train_mini
CEIL=outputs/cand_iface_0914/train_mini/ceiling
DIAG=outputs/cand_iface_0916/priv_diag_train_mini
M30=outputs/cand_iface_0916/model30b_train_mini

serve () {  # gpu port model revision served util -> 0 when up
  local gpu=$1 port=$2 model=$3 revision=$4 name=$5 util=$6
  local log=$Q/vllm-$name-$port.log
  [ "$(root_free_gb)" -ge 5 ] || { say "ALARM root filesystem has $(root_free_gb)G free; not serving $name"; return 1; }
  probe "$port" && { say "REFUSING to serve: :$port already answers"; return 1; }
  local rev=""; [ -n "$revision" ] && rev="--revision $revision"
  (CUDA_VISIBLE_DEVICES=$gpu setsid nohup "$VLLM" -m vllm.entrypoints.openai.api_server --model "$model" $rev \
      --served-model-name "$name" --port "$port" --tensor-parallel-size 1 --max-model-len 16384 \
      --gpu-memory-utilization "$util" --limit-mm-per-prompt '{"image":1}' > "$log" 2>&1 < /dev/null &) > /dev/null 2>&1
  for _ in $(seq 300); do probe "$port" && break; sleep 10; done
  probe "$port" || { say "ALARM :$port did not come up in 50 min; tail:"; tail -8 "$log" | cut -c1-300; return 1; }
  say ":$port up ($name, GPU $gpu, util $util); root free $(root_free_gb)G"
}

stop_port () {
  local port=$1 pid
  pid=$(ps -eo pid,args | awk -v p="--port $port" '$0 ~ /vllm.entrypoints.openai.api_server/ && index($0, p) {print $1}' | head -1)
  [ -z "$pid" ] && { say ":$port nothing to stop"; return 0; }
  kill "$pid"; for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
  sleep 10; say ":$port stopped (pid $pid)"
}

cell () {  # root name gpu port model nothink switches
  local root=$1 name=$2 gpu=$3 port=$4 model=$5 nothink=$6 switches=$7
  probe "$port" || { say "SKIP $name: :$port not up"; return 1; }
  [ "$(root_free_gb)" -ge 2 ] || { say "ALARM root filesystem $(root_free_gb)G free; SKIP $name"; return 1; }
  mkdir -p "$root"
  ROOT_OUT=$root CELL=$name GPU=$gpu PORT=$port MODEL=$model NO_THINK=$nothink SWITCHES="$switches" \
    bash scripts/drivers/partnr_typed_ablate_simpair_train_mini.sh > "$root/$name.driver.log" 2>&1 < /dev/null
  say "cell $name: $(tail -1 "$root/$name.driver.log")"
  [ -e "$root/$name/DONE" ]
}

report () {
  local name=$1; shift
  say "report $name"
  timeout 3600 "$@" > "$Q/$name.txt" 2>&1 < /dev/null || say "WARN report $name exit $?"
}

line_a () {
  if serve 1 8063 Qwen/Qwen2.5-VL-7B-Instruct "" qwen2.5-vl-7b 0.40; then
    cell "$ABL" stage_beside_same2 1 8063 qwen2.5-vl-7b 0 "typed_stages=True typed_beside=True typed_same_object=True"
    report fix2_pairs $PY scripts/partnr_pair_cells.py --pool "$POOL" \
        --cell same2="$ABL/stage_beside_same2" --cell same1="$ABL/stage_beside_same" \
        --cell beside="$ABL/stage_beside" --cell nostage="$SIMP/typed_RS_nostage" \
        --cell priv_partial="$DIAG/priv_partial" --cell ceiling="$CEIL" \
        --compare same2:same1 same2:beside same2:nostage priv_partial:same2 ceiling:same2 --json "$Q/fix2_pairs.json"
    report fix2_failures $PY scripts/partnr_failure_classes.py --pool "$POOL" \
        --cell same1="$ABL/stage_beside_same" --cell same2="$ABL/stage_beside_same2" --json "$Q/fix2_failures.json"
    stop_port 8063
  fi
  say "line A done"
}

line_b () {
  if serve 0 8071 Qwen/Qwen3-VL-30B-A3B-Instruct 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c qwen3-vl-30b 0.74; then
    reply=$(curl -s -m 120 http://127.0.0.1:8071/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"qwen3-vl-30b","messages":[{"role":"user","content":"Reply with the single word ok."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
    say "30B smoke: $(echo "$reply" | head -c 300)"
    if echo "$reply" | grep -q '"content"' && ! echo "$reply" | grep -q '<think>'; then
      cell "$M30" nostage_30b 0 8071 qwen3-vl-30b 1 "typed_stages=False"
      cell "$M30" stage_beside_30b 0 8071 qwen3-vl-30b 1 "typed_stages=True typed_beside=True"
      cell "$M30" stage_beside_same2_30b 0 8071 qwen3-vl-30b 1 "typed_stages=True typed_beside=True typed_same_object=True"
      report m30_pairs $PY scripts/partnr_pair_cells.py --pool "$POOL" \
          --cell nostage_30b="$M30/nostage_30b" --cell stage_beside_30b="$M30/stage_beside_30b" \
          --cell same2_30b="$M30/stage_beside_same2_30b" --cell nostage_7b="$SIMP/typed_RS_nostage" \
          --cell stage_beside_7b="$ABL/stage_beside" --cell same2_7b="$ABL/stage_beside_same2" --cell ceiling="$CEIL" \
          --compare stage_beside_30b:nostage_30b same2_30b:stage_beside_30b same2_30b:nostage_30b \
                    nostage_30b:nostage_7b same2_30b:same2_7b ceiling:same2_30b --json "$Q/m30_pairs.json"
      report m30_failures $PY scripts/partnr_failure_classes.py --pool "$POOL" \
          --cell nostage_30b="$M30/nostage_30b" --cell stage_beside_30b="$M30/stage_beside_30b" \
          --cell same2_30b="$M30/stage_beside_same2_30b" --json "$Q/m30_failures.json"
    else
      say "ALARM 30B smoke failed; 30B cells skipped"
    fi
    stop_port 8071
  fi
  say "line B done"
}

say "queue start (commit $(git rev-parse --short HEAD)); root free $(root_free_gb)G"
grep -q "strictly earlier stage" our_method/skill_memory_v2/partnr_typed_goals.py \
  || { say "REFUSING: refined same_as rule (158a313) not in the synced code"; exit 6; }
line_a > "$Q/line_a.log" 2>&1 &
A=$!
line_b > "$Q/line_b.log" 2>&1 &
B=$!
echo "$A $B" > "$Q/LINES.pid"
wait "$A"; say "line A exited: $(tail -1 "$Q/line_a.log")"
wait "$B"; say "line B exited: $(tail -1 "$Q/line_b.log")"
{
  echo "# PARTNR overnight 2026-09-16, second pass"
  echo; echo "commit $(git rev-parse --short HEAD); finished $(date '+%m-%d %H:%M'); root free $(root_free_gb)G"
  echo; echo "## Cells"; echo '```'
  for f in "$ABL/stage_beside_same2" "$M30/nostage_30b" "$M30/stage_beside_30b" "$M30/stage_beside_same2_30b"; do
    printf '%s: ' "$f"; cat "$f/CELL.json" 2>/dev/null || echo "(no CELL.json)"
    [ -e "$f/DONE" ] || echo "   ^ no DONE"
  done
  echo '```'; echo; echo "## Alarms and warnings"; echo '```'
  grep -h -E "ALARM|WARN|SKIP|REFUSING" "$Q/line_a.log" "$Q/line_b.log" || echo "none"
  echo '```'
  for r in fix2_pairs fix2_failures m30_pairs m30_failures; do
    [ -s "$Q/$r.txt" ] || continue
    echo; echo "## $r"; echo '```'
    grep -v -E "Gym|gymnasium|Please upgrade|PluginManager|migration_guide" "$Q/$r.txt" | head -220
    echo '```'
  done
} > "$Q/OVERNIGHT_SUMMARY.md"
say "summary written: $Q/OVERNIGHT_SUMMARY.md"
touch "$Q/ALLDONE"
