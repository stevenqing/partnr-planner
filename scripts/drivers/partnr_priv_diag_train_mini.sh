#!/usr/bin/env bash
# Where do temporal episodes fail when the requirements and the DAG are given? train_mini, no model.
#
# The typed arm with stages + beside reaches state_success 0.168 on R_S_T and never moves R_T,
# while the privileged arm with the true DAG was 0.23 on val_mini R_S_T and the centralized
# full-observation ceiling 0.77. This runs the privileged arm (episode propositions + temporal
# DAG, iir1 library, decentralized) on the same 214-episode pool as the typed pairings
#   priv_partial   partial observation (as every typed cell)
#   priv_full      full observation     -> priv_full - priv_partial is what exploring costs;
#                                          ceiling - priv_full is decentralization and execution
# The ceiling is outputs/cand_iface_0914/train_mini/ceiling (heuristic_full_obs, all 403 episodes).
# Failure classes come afterwards from planner-log task_explanation and the planner's notes.
#
#   CELL=priv_partial GPU=1 PROCS=48 EXTRA="" setsid nohup bash scripts/drivers/partnr_priv_diag_train_mini.sh > ... &
#   CELL=priv_full    GPU=1 PROCS=48 EXTRA="world_model.partial_obs=False" setsid nohup bash ... &
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PY=/root/venvs/partnr/bin/python
ROOT_OUT=outputs/cand_iface_0916/priv_diag_train_mini
BASE_POOL=outputs/cand_iface_0915/stage_simpair_train_mini/episodes.json
POOL=train_mini
OPS=results/partnr_operators_iir1.json
CELL=${CELL:?} GPU=${GPU:?} PROCS=${PROCS:-48} EXTRA=${EXTRA:-}
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

OUT=$ROOT_OUT/$CELL
[ -e "$OUT" ] && { say "REFUSING: $OUT exists, not running on residue"; exit 3; }
[ -s "$BASE_POOL" ] && [ -s "$OPS" ] || { say "REFUSING: $BASE_POOL or $OPS missing"; exit 6; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/COMMIT" 2>/dev/null
echo "$EXTRA" > "$OUT/EXTRA"
cp "$BASE_POOL" "$ROOT_OUT/episodes.json" 2>/dev/null
IDS=$("$PY" -c "import json; print(','.join(json.load(open('$BASE_POOL'))['ids']))")
WANT=$(echo "$IDS" | tr ',' '\n' | grep -c .)
both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

stats=$OUT/results/$POOL.json.gz/stats
started=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU "$PY" -m habitat_llm.examples.planner_demo \
    --config-name baselines/skill_memory_v2_oracle_goals.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$POOL.json.gz" \
    num_proc="$PROCS" evaluation.save_video=False +resume=True hydra.run.dir="$PWD/$OUT" \
    "+episode_id_filter=[$IDS]" $(both operators $OPS) $EXTRA >> "$OUT/run.log" 2>&1 &
runner=$!
echo "$runner" > "$OUT/PID"
say "$CELL pid=$runner gpu=$GPU procs=$PROCS episodes=$WANT extra=[$EXTRA]"
last=-1; changed=$started; why=""
while kill -0 "$runner" 2>/dev/null; do
  sleep 30
  now=$(date +%s); count=$(ls "$stats" 2>/dev/null | wc -l)
  [ "$count" -ne "$last" ] && { last=$count; changed=$now; }
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
printf '{"cell":"%s","extra":"%s","gpu":"%s","status":%d,"episodes":%d,"wanted":%d,"fake_episodes":%s,"killed":"%s","seconds":%d}\n' \
    "$CELL" "$EXTRA" "$GPU" "$status" "$count" "$WANT" "${fake:-null}" "$why" "$(( $(date +%s) - started ))" > "$OUT/CELL.json"
[ "$status" -eq 0 ] && [ -z "$why" ] && [ "$count" -eq "$WANT" ] && touch "$OUT/DONE"
say "end $CELL: $(cat "$OUT/CELL.json")"
