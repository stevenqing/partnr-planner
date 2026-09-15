#!/usr/bin/env bash
# train_mini sim pairing: do stages read off the instruction fix the typed arm's temporal tasks?
#
# The typed arm orders only a second place for the same object, but most PARTNR temporal edges
# join different objects; val_mini state_success was R_T 0.17, R_S_T 0.04. Offline on train_mini
# the stage rule (partnr_typed_goals.stage_lines, frozen v2) recovers ordered pairs at 0.971 /
# precision 0.975 on the stored 7B typedRS answers, against 0.278 today. This pairs the two
# orderings in the simulator, everything else identical (RS examples and the R-only inside
# prior, as in the val_mini cells; iir1 library; 7B), on
#   every temporal train_mini episode (133)   -- where the rule can matter
#   the 81 non-temporal episodes of the earlier 120-episode pool -- a control that should not move
# It is train, so it may steer the method; val_mini is reported once, later.
#
#   typed_RS_nostage  typed_stages=False  :8061 GPU 0
#   typed_RS_stage    typed_stages=True   :8063 GPU 1
#
#   setsid nohup bash scripts/drivers/partnr_typed_stage_simpair_train_mini.sh > /tmp/stage_simpair.log 2>&1 < /dev/null &
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PY=/root/venvs/partnr/bin/python
OUT=outputs/cand_iface_0915/stage_simpair_train_mini
OLD_POOL=outputs/cand_iface_0914/simpair_train_mini/episodes.json
POOL=train_mini
MODEL=qwen2.5-vl-7b
OPS=results/partnr_operators_iir1.json
PRIOR=results/partnr_inside_prior_train_R_only.json
PROCS=${PROCS:-24}
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "$1/models" | grep -q "^200$"; }

[ -e "$OUT" ] && { say "REFUSING: $OUT exists, not running on residue"; exit 3; }
grep -q "def stage_lines" our_method/skill_memory_v2/partnr_typed_goals.py \
  && grep -q "typed_stages" our_method/skill_memory_v2/partnr_planner.py \
  && grep -q "return sorted(in_named)\[0\]" our_method/skill_memory_v2/partnr_typed_goals.py \
  || { say "REFUSING: stage rule or v7b typed interface not in the code"; exit 6; }
[ "$("$PY" -c "import json; print(len(json.load(open('results/partnr_object_kinds_train.json'))['kinds']))")" = 106 ] \
  || { say "REFUSING: object kinds file is not the 106-kind union"; exit 6; }
[ -s "$PRIOR" ] && [ -s "$OLD_POOL" ] || { say "REFUSING: $PRIOR or $OLD_POOL missing"; exit 6; }
for port in 8061 8063; do
  probe "http://127.0.0.1:$port/v1" || { say "REFUSING: :$port not up"; exit 4; }
done

mkdir -p "$OUT"
md5sum our_method/skill_memory_v2/partnr_typed_goals.py our_method/skill_memory_v2/partnr_planner.py \
  habitat_llm/conf/planner/skill_memory_v2_typed_planner.yaml > "$OUT/CODE_MD5"
git rev-parse HEAD > "$OUT/COMMIT" 2>/dev/null
"$PY" - "$OUT/episodes.json" "$OLD_POOL" <<'PY'
import collections, gzip, json, sys
sys.path.insert(0, "scripts")
from partnr_task_types import classify
out, old = sys.argv[1], sys.argv[2]
episodes = json.load(gzip.open("data/datasets/partnr_episodes/v0_0/train_mini.json.gz"))["episodes"]
types = {str(e["episode_id"]): classify(e) for e in episodes}
temporal = sorted((i for i, t in types.items() if "T" in t.split("_")), key=int)
control = sorted((i for i in json.load(open(old))["ids"] if "T" not in types[i].split("_")), key=int)
ids = sorted(set(temporal) | set(control), key=int)
by_type = collections.Counter(types[i] for i in ids)
json.dump({"temporal": temporal, "control": control, "by_type": dict(sorted(by_type.items())),
           "ids": ids}, open(out, "w"), indent=1)
print(len(temporal), "temporal +", len(control), "control =", len(ids), dict(by_type))
PY
IDS=$("$PY" -c "import json; print(','.join(json.load(open('$OUT/episodes.json'))['ids']))")
WANT=$(echo "$IDS" | tr ',' '\n' | grep -c .)
say "pool: $WANT episodes"

both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

cell () {  # arm gpu port [overrides...]
  local arm=$1 gpu=$2 port=$3; shift 3
  local url=http://127.0.0.1:$port/v1 out=$OUT/$arm
  local stats=$out/results/$POOL.json.gz/stats
  mkdir -p "$out"
  local started; started=$(date +%s)
  VLLM_BASE_URL=$url CUDA_VISIBLE_DEVICES=$gpu "$PY" -m habitat_llm.examples.planner_demo \
      --config-name baselines/skill_memory_v2_typed_vllm.yaml \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$POOL.json.gz" \
      num_proc="$PROCS" evaluation.save_video=False +resume=True hydra.run.dir="$PWD/$out" \
      "+episode_id_filter=[$IDS]" \
      $(both llm.generation_params.model $MODEL) $(both operators $OPS) \
      $(both inside_prior $PRIOR) $(both typed_examples RS) "$@" >> "$out/run.log" 2>&1 &
  local runner=$!
  echo "$runner" > "$out/PID"
  say "$arm pid=$runner gpu=$gpu url=$url procs=$PROCS episodes=$WANT"
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
    # The model is called only at episode start, so running=0 is normal; judge by stats.
    [ $((now - changed)) -ge 1800 ] && { why=stall; kill -9 "$runner"; break; }
    [ $((now - started)) -ge 25200 ] && { why=timeout; kill -9 "$runner"; break; }
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
  printf '{"cell":"%s","url":"%s","status":%d,"episodes":%d,"wanted":%d,"fake_episodes":%s,"killed":"%s","seconds":%d}\n' \
      "$arm" "$url" "$status" "$count" "$WANT" "${fake:-null}" "$why" "$(( $(date +%s) - started ))" > "$out/CELL.json"
  [ "$status" -eq 0 ] && [ -z "$why" ] && [ "$count" -eq "$WANT" ] && touch "$out/DONE"
  say "end $arm: $(cat "$out/CELL.json")"
}

cell typed_RS_nostage 0 8061 $(both typed_stages False) &
cell typed_RS_stage   1 8063 $(both typed_stages True) &
wait
touch "$OUT/ALLDONE"
say "ALL DONE"
