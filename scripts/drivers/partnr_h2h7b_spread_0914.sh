#!/usr/bin/env bash
# The 7B head-to-head spread over the GPUs that freed up at 00:15 on 09-14 (1, 2, 4).
#
# Replaces partnr_h2h7b_night_0913.sh's stage 1/2. At 00:06 everything was on GPU 0: the
# retrieval cell had 66/369 after 90 min at 12 workers and would have been cut by its 8 h
# timeout around 300, and G-Memory / MEMENTO were queued behind the ReAct and ours cells
# (~05:00). Now each arm gets its own endpoint -- same model and archived settings as :8061
# (16k, 0.30), one generating job per endpoint -- and its own GPU for rendering:
#
#   GPU 1  :8063  gmemory_7b       24 workers (memory on CPU, ~0.72 G each)
#   GPU 2  :8064  memento_7b       24 workers, the 7B extraction table from stage 0
#   GPU 4  :8065  react_rag_R_7b   20 workers (all-mpnet-base-v2 on GPU, ~1.75 G each),
#                 resumed in its own directory after the GPU-0 runner was killed
#
# The retrieval cell resumes on the same code (+resume=True, band = 0), a different
# endpoint instance of the same model and settings.
# Guards are partnr_h2h7b_0913.sh's cell(), with the probe on the cell's own endpoint.
# The final report waits for the GPU-0 driver (react_7b, v2_accepted_7b) BY PID.
#
#   bash scripts/drivers/partnr_h2h7b_spread_0914.sh <h2h driver pid> <old rag runner pid>
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
H2H_PID=${1:?h2h driver pid}; OLD_RAG_PID=${2:?old rag runner pid}
PY=/root/venvs/partnr/bin/python
OUT=outputs/headtohead_0913/val_mini
POOL=val_mini
MODEL=qwen2.5-vl-7b
EXTRACT7=results/partnr_memento_extractions_val_mini_7b.json
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "$1/models" | grep -q "^200$"; }

kill -0 "$OLD_RAG_PID" 2>/dev/null && { say "REFUSING: old rag runner $OLD_RAG_PID still alive"; exit 3; }
[ -s "$EXTRACT7" ] || { say "REFUSING: $EXTRACT7 missing"; exit 5; }
for port in 8063 8064 8065; do
  probe "http://127.0.0.1:$port/v1" || { say "REFUSING: :$port not up"; exit 4; }
done

both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

cell () {  # name config gpu port procs [overrides...]
  local name=$1 config=$2 gpu=$3 port=$4 procs=$5; shift 5
  local url=http://127.0.0.1:$port/v1
  local out=$OUT/$name stats=$OUT/$name/results/$POOL.json.gz/stats
  mkdir -p "$out"
  [ -f "$out/DONE" ] && { say "skip $name"; return 0; }
  local free; free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  say "$name: GPU $gpu free ${free} MiB before start"
  local started; started=$(date +%s)
  VLLM_BASE_URL=$url CUDA_VISIBLE_DEVICES=$gpu "$PY" -m habitat_llm.examples.planner_demo \
      --config-name "$config" \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$POOL.json.gz" \
      num_proc="$procs" evaluation.save_video=False +resume=True hydra.run.dir="$out" \
      $(both llm.generation_params.model $MODEL) "$@" >> "$out/run.log" 2>&1 &
  local runner=$!
  echo "$runner" > "$out/PID"
  say "$name pid=$runner config=$config gpu=$gpu url=$url procs=$procs"
  local last=-1 changed=$started misses=0 why=""
  while kill -0 "$runner" 2>/dev/null; do
    sleep 30
    local now count; now=$(date +%s); count=$(ls "$stats" 2>/dev/null | wc -l)
    [ "$count" -ne "$last" ] && { last=$count; changed=$now; }
    if probe "$url"; then misses=0
    else
      misses=$((misses + 1)); say "endpoint miss $misses/2 for $name"
      [ "$misses" -ge 2 ] && { why=endpoint_dead; kill -9 "$runner"; break; }
    fi
    [ $((now - changed)) -ge 1800 ] && { why=stall; kill -9 "$runner"; break; }
    [ $((now - started)) -ge 28800 ] && { why=timeout; kill -9 "$runner"; break; }
  done
  wait "$runner" 2>/dev/null; local status=$?
  # sim_step_count == 0 alone is not a dead endpoint at 7B: the ReAct arms spend 200-1100 s
  # issuing node names that do not exist (living_room_0, kitchen_0) and never step. The
  # dead-endpoint signature is a zero-step episode that also ended within a minute.
  local count fake
  count=$(ls "$stats" 2>/dev/null | wc -l)
  fake=$("$PY" - "$stats" <<'PYEOF'
import glob, json, sys
n = 0
for f in glob.glob(sys.argv[1] + "/*.json"):
    st = json.load(open(f)).get("stats")
    st = json.loads(st) if isinstance(st, str) else st
    n += bool(st and st.get("sim_step_count") == 0 and (st.get("runtime") or 0) < 60)
print(n)
PYEOF
)
  printf '{"cell":"%s","config":"%s","model":"%s","url":"%s","status":%d,"episodes":%d,"fake_episodes":%d,"killed":"%s","seconds":%d}\n' \
      "$name" "$config" "$MODEL" "$url" "$status" "$count" "$fake" "$why" "$(( $(date +%s) - started ))" > "$out/CELL.json"
  [ "$status" -eq 0 ] && [ -z "$why" ] && touch "$out/DONE"
  say "end $name: $(cat "$out/CELL.json")"
}

rm -f "$OUT/react_rag_R_7b/DONE"
cell gmemory_7b     baselines/react_gmemory_vllm.yaml 1 8063 24 &
cell memento_7b     baselines/react_memento_vllm.yaml 2 8064 24 $(both memory_extractions $EXTRACT7) &
cell react_rag_R_7b baselines/react_rag_R_vllm.yaml   4 8065 20 &
wait

while kill -0 "$H2H_PID" 2>/dev/null; do sleep 60; done
"$PY" scripts/partnr_v2_report.py --sweep "$OUT" --dataset "$POOL.json.gz" --split "$POOL" \
    --baseline react_7b --out "$OUT/report_all.json" > "$OUT/report_all.txt" 2>&1
say "report over all five arms -> $OUT/report_all.txt"
say "ALL DONE"
