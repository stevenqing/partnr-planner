#!/usr/bin/env bash
# train_mini sim pairing: does the typed requirement interface's offline gain survive the simulator?
#
# Offline on train_mini the typed interface (typed3 prompt + memory projection) reaches consistent
# recall 0.62 against 0.09 for the planner's free-form prompt, but an 8-episode sim smoke scored
# 0.27 against a 0.98 ceiling. This runs the two arms on the same train episodes, same library
# (iir1), same 7B model, differing only in goal_source, and pairs them per episode. It is train,
# so it may steer the interface; val_mini is still reported once, later.
#
#   intent_7b  baselines/skill_memory_v2_vllm.yaml        (goal_source: llm)   :8061 GPU 0
#   typed_7b   baselines/skill_memory_v2_typed_vllm.yaml  (goal_source: typed) :8063 GPU 1
#
#   setsid nohup bash scripts/drivers/partnr_typed_simpair_train_mini.sh > /tmp/simpair.log 2>&1 &
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PY=/root/venvs/partnr/bin/python
OUT=outputs/cand_iface_0914/simpair_train_mini
POOL=train_mini
MODEL=qwen2.5-vl-7b
N=${N:-120}
OPS=results/partnr_operators_iir1.json
R1=outputs/cand_iface_0914/train_mini/r1
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "$1/models" | grep -q "^200$"; }

[ -e "$OUT" ] && { say "REFUSING: $OUT exists, not running on residue"; exit 3; }
grep -q "_requirements_from_typed" our_method/skill_memory_v2/partnr_planner.py \
  && grep -q "moved: to that kind in the room the task names" our_method/skill_memory_v2/partnr_typed_goals.py \
  || { say "REFUSING: typed interface code missing"; exit 6; }
[ "$("$PY" -c "import json; print(len(json.load(open('results/partnr_object_kinds_train.json'))['kinds']))")" = 106 ] \
  || { say "REFUSING: object kinds file is not the 106-kind union"; exit 6; }
[ -s results/partnr_inside_prior_train.json ] || { say "REFUSING: inside prior missing"; exit 6; }

# The offline pass-2 job shares these endpoints; one generation job per endpoint.
for i in $(seq 240); do [ -e "$R1/ALLDONE_P2" ] && break; sleep 10; done
[ -e "$R1/ALLDONE_P2" ] || { say "REFUSING: offline pass 2 still holding the endpoints"; exit 7; }
for port in 8061 8063; do
  probe "http://127.0.0.1:$port/v1" || { say "REFUSING: :$port not up"; exit 4; }
done

mkdir -p "$OUT"
"$PY" - "$OUT/episodes.json" "$N" <<'PY'
import collections, gzip, json, random, sys
sys.path.insert(0, "scripts")
from partnr_task_types import classify
out, n = sys.argv[1], int(sys.argv[2])
episodes = json.load(gzip.open("data/datasets/partnr_episodes/v0_0/train_mini.json.gz"))["episodes"]
smoke = {"0", "1", "25", "56", "158", "160", "145", "431"}  # the 8-episode sim smoke, kept out
by_type = collections.defaultdict(list)
for e in episodes:
    if str(e["episode_id"]) not in smoke:
        by_type[classify(e)].append(str(e["episode_id"]))
rng = random.Random(20260914)
total = sum(len(v) for v in by_type.values())
ids, counts = [], {}
for kind in sorted(by_type):
    k = min(len(by_type[kind]), round(n * len(by_type[kind]) / total))
    chosen = rng.sample(sorted(by_type[kind], key=int), k)
    ids += chosen
    counts[kind] = k
json.dump({"seed": 20260914, "excluded": sorted(smoke, key=int), "by_type": counts,
           "ids": sorted(ids, key=int)}, open(out, "w"), indent=1)
print(len(ids), counts)
PY
IDS=$("$PY" -c "import json; print(','.join(json.load(open('$OUT/episodes.json'))['ids']))")
WANT=$(echo "$IDS" | tr ',' '\n' | grep -c .)
say "pool: $WANT episodes"

both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

cell () {  # arm config gpu port procs [overrides...]
  local arm=$1 config=$2 gpu=$3 port=$4 procs=$5; shift 5
  local url=http://127.0.0.1:$port/v1 out=$OUT/$arm
  local stats=$out/results/$POOL.json.gz/stats
  mkdir -p "$out"
  local started; started=$(date +%s)
  VLLM_BASE_URL=$url CUDA_VISIBLE_DEVICES=$gpu "$PY" -m habitat_llm.examples.planner_demo \
      --config-name "$config" \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$POOL.json.gz" \
      num_proc="$procs" evaluation.save_video=False +resume=True hydra.run.dir="$PWD/$out" \
      "+episode_id_filter=[$IDS]" \
      $(both llm.generation_params.model $MODEL) "$@" >> "$out/run.log" 2>&1 &
  local runner=$!
  echo "$runner" > "$out/PID"
  say "$arm pid=$runner gpu=$gpu url=$url procs=$procs episodes=$WANT"
  local last=-1 changed=$started misses=0 why=""
  while kill -0 "$runner" 2>/dev/null; do
    sleep 30
    local now count; now=$(date +%s); count=$(ls "$stats" 2>/dev/null | wc -l)
    [ "$count" -ne "$last" ] && { last=$count; changed=$now; }
    if probe "$url"; then misses=0
    else
      misses=$((misses + 1)); say "endpoint miss $misses/2 for $arm"
      [ "$misses" -ge 2 ] && { why=endpoint_dead; kill -9 "$runner"; break; }
    fi
    # v2 arms call the model only at episode start, so running=0 is normal; judge by stats.
    [ $((now - changed)) -ge 1800 ] && { why=stall; kill -9 "$runner"; break; }
    [ $((now - started)) -ge 14400 ] && { why=timeout; kill -9 "$runner"; break; }
  done
  wait "$runner" 2>/dev/null; local status=$?
  local count; count=$(ls "$stats" 2>/dev/null | wc -l)
  local fake; fake=$("$PY" - "$stats" <<'PY'
import glob, json, sys
n = 0
for f in glob.glob(sys.argv[1] + "/*.json"):
    s = json.load(open(f)).get("stats")
    s = json.loads(s) if isinstance(s, str) else (s or {})
    n += int(s.get("sim_step_count", 1) == 0 and (s.get("runtime") or 0) < 60)
print(n)
PY
)
  printf '{"cell":"%s","config":"%s","url":"%s","status":%d,"episodes":%d,"wanted":%d,"fake_episodes":%s,"killed":"%s","seconds":%d}\n' \
      "$arm" "$config" "$url" "$status" "$count" "$WANT" "${fake:-null}" "$why" "$(( $(date +%s) - started ))" > "$out/CELL.json"
  [ "$status" -eq 0 ] && [ -z "$why" ] && [ "$count" -eq "$WANT" ] && touch "$out/DONE"
  say "end $arm: $(cat "$out/CELL.json")"
}

cell intent_7b baselines/skill_memory_v2_vllm.yaml       0 8061 24 $(both operators $OPS) &
cell typed_7b  baselines/skill_memory_v2_typed_vllm.yaml 1 8063 24 $(both operators $OPS) &
wait
touch "$OUT/ALLDONE"
say "ALL DONE"
