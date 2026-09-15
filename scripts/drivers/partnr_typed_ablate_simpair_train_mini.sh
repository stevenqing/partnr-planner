#!/usr/bin/env bash
# train_mini sim cells on top of typed_RS_stage: intermediate stops in the prompt, "next to" folding,
# both, and a repeat of the stage cell itself (the typed model arm's run-to-run band is not 0: 4 of
# 81 control episodes moved with identical requirements in the stage pairing).
#
# Same 214-episode pool as outputs/cand_iface_0915/stage_simpair_train_mini (133 temporal + 81
# non-temporal control), same model, library, prior and RS examples; the base for every cell is
# typed_RS_stage there. Cells are launched one line each so they can go as endpoints come up:
#
#   CELL=stage_rep          GPU=0 PORT=8061 SWITCHES="typed_stages=True"
#   CELL=stage_beside       GPU=1 PORT=8063 SWITCHES="typed_stages=True typed_beside=True"
#   CELL=stage_stops        GPU=5 PORT=8065 SWITCHES="typed_stages=True typed_stops=True"
#   CELL=stage_stops_beside GPU=2 PORT=8067 SWITCHES="typed_stages=True typed_stops=True typed_beside=True"
#
#   CELL=... GPU=... PORT=... SWITCHES="..." setsid nohup bash scripts/drivers/partnr_typed_ablate_simpair_train_mini.sh \
#       > outputs/cand_iface_0915/ablate_train_mini/$CELL.driver.log 2>&1 < /dev/null &
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PY=/root/venvs/partnr/bin/python
ROOT_OUT=${ROOT_OUT:-outputs/cand_iface_0915/ablate_train_mini}
BASE_POOL=outputs/cand_iface_0915/stage_simpair_train_mini/episodes.json
POOL=train_mini
MODEL=${MODEL:-qwen2.5-vl-7b}
# NO_THINK=1 for Qwen3 models: their chat template otherwise opens a <think> block.
NO_THINK=${NO_THINK:-0}
OPS=results/partnr_operators_iir1.json
PRIOR=results/partnr_inside_prior_train_R_only.json
CELL=${CELL:?} GPU=${GPU:?} PORT=${PORT:?} SWITCHES=${SWITCHES:?}
PROCS=${PROCS:-24}
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "$1/models" | grep -q "^200$"; }

OUT=$ROOT_OUT/$CELL
[ -e "$OUT" ] && { say "REFUSING: $OUT exists, not running on residue"; exit 3; }
grep -q "def beside_requirements" our_method/skill_memory_v2/partnr_typed_goals.py \
  && grep -q "typed_beside" our_method/skill_memory_v2/partnr_planner.py \
  && grep -q "typed_stops" our_method/skill_memory_v2/partnr_planner.py \
  && grep -q "typed_same_object" our_method/skill_memory_v2/partnr_planner.py \
  && grep -q "def stage_lines" our_method/skill_memory_v2/partnr_typed_goals.py \
  || { say "REFUSING: switches not in the code"; exit 6; }
[ -s "$PRIOR" ] && [ -s "$BASE_POOL" ] || { say "REFUSING: $PRIOR or $BASE_POOL missing"; exit 6; }
probe "http://127.0.0.1:$PORT/v1" || { say "REFUSING: :$PORT not up"; exit 4; }

mkdir -p "$OUT"
cp "$BASE_POOL" "$ROOT_OUT/episodes.json" 2>/dev/null
md5sum our_method/skill_memory_v2/partnr_typed_goals.py our_method/skill_memory_v2/partnr_planner.py \
  habitat_llm/conf/planner/skill_memory_v2_typed_planner.yaml > "$OUT/CODE_MD5"
echo "$SWITCHES model=$MODEL no_think=$NO_THINK" > "$OUT/SWITCHES"
IDS=$("$PY" -c "import json; print(','.join(json.load(open('$BASE_POOL'))['ids']))")
WANT=$(echo "$IDS" | tr ',' '\n' | grep -c .)

both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }
EXTRA=""
for kv in $SWITCHES; do EXTRA="$EXTRA $(both "${kv%%=*}" "${kv#*=}")"; done
if [ "$NO_THINK" = "1" ]; then
  EXTRA="$EXTRA +evaluation.agents.agent_0.planner.plan_config.llm.extra_body.chat_template_kwargs.enable_thinking=False"
  EXTRA="$EXTRA +evaluation.agents.agent_1.planner.plan_config.llm.extra_body.chat_template_kwargs.enable_thinking=False"
fi

url=http://127.0.0.1:$PORT/v1
stats=$OUT/results/$POOL.json.gz/stats
started=$(date +%s)
VLLM_BASE_URL=$url CUDA_VISIBLE_DEVICES=$GPU "$PY" -m habitat_llm.examples.planner_demo \
    --config-name baselines/skill_memory_v2_typed_vllm.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$POOL.json.gz" \
    num_proc="$PROCS" evaluation.save_video=False +resume=True hydra.run.dir="$PWD/$OUT" \
    "+episode_id_filter=[$IDS]" \
    $(both llm.generation_params.model $MODEL) $(both operators $OPS) \
    $(both inside_prior $PRIOR) $(both typed_examples RS) $EXTRA >> "$OUT/run.log" 2>&1 &
runner=$!
echo "$runner" > "$OUT/PID"
say "$CELL pid=$runner gpu=$GPU url=$url model=$MODEL no_think=$NO_THINK procs=$PROCS episodes=$WANT switches=[$SWITCHES]"
last=-1; changed=$started; misses=0; why=""
while kill -0 "$runner" 2>/dev/null; do
  sleep 30
  now=$(date +%s); count=$(ls "$stats" 2>/dev/null | wc -l)
  [ "$count" -ne "$last" ] && { last=$count; changed=$now; }
  if probe "$url"; then misses=0
  else
    misses=$((misses + 1)); say "endpoint miss $misses/2"
    [ "$misses" -ge 2 ] && { why=endpoint_dead; kill -9 "$runner"; break; }
  fi
  # The model is called only at episode start, so running=0 is normal; judge by stats.
  [ $((now - changed)) -ge 1800 ] && { why=stall; kill -9 "$runner"; break; }
  [ $((now - started)) -ge 25200 ] && { why=timeout; kill -9 "$runner"; break; }
done
wait "$runner" 2>/dev/null; status=$?
count=$(ls "$stats" 2>/dev/null | wc -l)
fake=$("$PY" - "$stats" <<'PY'
import glob, json, sys
n = 0
for f in glob.glob(sys.argv[1] + "/*.json"):
    s = json.load(open(f)).get("stats")
    s = json.loads(s) if isinstance(s, str) else (s or {})
    n += int(s.get("sim_step_count", 1) == 0 and (s.get("runtime") or 0) < 60)
print(n)
PY
)
printf '{"cell":"%s","switches":"%s","gpu":"%s","url":"%s","status":%d,"episodes":%d,"wanted":%d,"fake_episodes":%s,"killed":"%s","seconds":%d}\n' \
    "$CELL" "$SWITCHES" "$GPU" "$url" "$status" "$count" "$WANT" "${fake:-null}" "$why" "$(( $(date +%s) - started ))" > "$OUT/CELL.json"
[ "$status" -eq 0 ] && [ -z "$why" ] && [ "$count" -eq "$WANT" ] && touch "$OUT/DONE"
say "end $CELL: $(cat "$OUT/CELL.json")"
