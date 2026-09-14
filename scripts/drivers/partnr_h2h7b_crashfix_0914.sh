#!/usr/bin/env bash
# Re-run the crashed episodes of the four 7B baseline arms with the crash fixed, splice them
# into a copy of the sweep, and report again.
#
# The crash: a model response with no `Agent_<id>_Action:` line parses to nothing, the agent is
# still marked as replanned, and `update_agent_action_history` asserts (evaluation_runner.py:517).
# It ended 47/35/56/145 episodes of react / rag / gmemory / memento and 0 of ours, so the
# 09-14 table compared the arms on different episodes. Fixed in llm_planner.py (the agent gets
# the "No actions were assigned" observation the empty parse already produced, and replans) and
# utils.py (split on the first "[", one episode per arm). Neither line is reached by an episode
# that did not crash -- the assert or the raise ended every episode that reached them -- so the
# non-crashed episodes stand as run and only the crashed ones are re-run, as with 25857.
#
# Each arm on the endpoint and GPU it ran on. The originals are left untouched; the spliced
# sweep is outputs/headtohead_0913/val_mini_fixed.
#
#   bash scripts/drivers/partnr_h2h7b_crashfix_0914.sh
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PY=/root/venvs/partnr/bin/python
SRC=outputs/headtohead_0913/val_mini
RERUN=outputs/headtohead_0913/val_mini_crashfix
FIXED=outputs/headtohead_0913/val_mini_fixed
IDS=$SRC/crashed_ids_0914.json
POOL=val_mini
MODEL=qwen2.5-vl-7b
EXTRACT7=results/partnr_memento_extractions_val_mini_7b.json
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "$1/models" | grep -q "^200$"; }

[ -s "$IDS" ] || { say "REFUSING: $IDS missing"; exit 5; }
[ -e "$RERUN" ] && { say "REFUSING: $RERUN exists, not re-running on residue"; exit 3; }
[ -e "$FIXED" ] && { say "REFUSING: $FIXED exists"; exit 3; }
grep -q 'No actions were assigned. Please assign action to this agent."' habitat_llm/planner/llm_planner.py \
  || { say "REFUSING: fix not in llm_planner.py"; exit 6; }
for port in 8061 8063 8064 8065; do
  probe "http://127.0.0.1:$port/v1" || { say "REFUSING: :$port not up"; exit 4; }
done

both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }

cell () {  # arm config gpu port procs [overrides...]
  local arm=$1 config=$2 gpu=$3 port=$4 procs=$5; shift 5
  local url=http://127.0.0.1:$port/v1 out=$RERUN/$arm
  local stats=$out/results/$POOL.json.gz/stats
  local ids; ids=$("$PY" -c "import json; print(','.join(str(i) for i in json.load(open('$IDS'))['$arm']))")
  local want; want=$(echo "$ids" | tr ',' '\n' | grep -c .)
  mkdir -p "$out"
  local started; started=$(date +%s)
  VLLM_BASE_URL=$url CUDA_VISIBLE_DEVICES=$gpu "$PY" -m habitat_llm.examples.planner_demo \
      --config-name "$config" \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$POOL.json.gz" \
      num_proc="$procs" evaluation.save_video=False hydra.run.dir="$out" \
      "+episode_id_filter=[$ids]" \
      $(both llm.generation_params.model $MODEL) "$@" >> "$out/run.log" 2>&1 &
  local runner=$!
  echo "$runner" > "$out/PID"
  say "$arm pid=$runner gpu=$gpu url=$url procs=$procs episodes=$want"
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
    [ $((now - changed)) -ge 1800 ] && { why=stall; kill -9 "$runner"; break; }
    [ $((now - started)) -ge 21600 ] && { why=timeout; kill -9 "$runner"; break; }
  done
  wait "$runner" 2>/dev/null; local status=$?
  local count; count=$(ls "$stats" 2>/dev/null | wc -l)
  printf '{"cell":"%s","config":"%s","url":"%s","status":%d,"episodes":%d,"wanted":%d,"killed":"%s","seconds":%d}\n' \
      "$arm" "$config" "$url" "$status" "$count" "$want" "$why" "$(( $(date +%s) - started ))" > "$out/CELL.json"
  [ "$status" -eq 0 ] && [ -z "$why" ] && [ "$count" -eq "$want" ] && touch "$out/DONE"
  say "end $arm: $(cat "$out/CELL.json")"
}

cell react_7b       baselines/react_vllm.yaml          0 8061 24 &
cell gmemory_7b     baselines/react_gmemory_vllm.yaml  1 8063 24 &
cell memento_7b     baselines/react_memento_vllm.yaml  2 8064 24 $(both memory_extractions $EXTRACT7) &
cell react_rag_R_7b baselines/react_rag_R_vllm.yaml    4 8065 20 &
wait

# Splice: every original stats file, then the re-run ones over the crashed ids. Only arms whose
# re-run is complete are spliced; an incomplete arm is left out of the fixed sweep and named.
mkdir -p "$FIXED"
"$PY" - "$SRC" "$RERUN" "$FIXED" "$IDS" <<'PY'
import json, shutil, sys
from pathlib import Path
src, rerun, fixed, ids = map(Path, sys.argv[1:])
crashed = json.load(open(ids))
summary = {}
for arm in ["react_7b", "react_rag_R_7b", "gmemory_7b", "memento_7b", "v2_accepted_7b"]:
    want = [str(i) for i in crashed[arm]]
    new = rerun / arm / "results/val_mini.json.gz/stats"
    if want and not (rerun / arm / "DONE").exists():
        summary[arm] = {"spliced": False, "reason": "re-run incomplete"}
        continue
    dst = fixed / arm / "results/val_mini.json.gz/stats"
    shutil.copytree(src / arm / "results/val_mini.json.gz/stats", dst)
    still = []
    for ep in want:
        shutil.copy(new / f"{ep}.json", dst / f"{ep}.json")
        record = json.load(open(dst / f"{ep}.json"))
        if record.get("stats") is None:
            still.append(ep)
    summary[arm] = {"spliced": True, "rerun": len(want), "still_crashed": still}
json.dump(summary, open(fixed / "SPLICE.json", "w"), indent=1)
print(json.dumps(summary))
PY

"$PY" scripts/partnr_v2_report.py --sweep "$FIXED" --dataset "$POOL.json.gz" --split "$POOL" \
    --baseline react_7b --out "$FIXED/report_fixed.json" > "$FIXED/report_fixed.txt" 2>&1
say "spliced sweep -> $FIXED, report -> $FIXED/report_fixed.txt"
say "ALL DONE"
