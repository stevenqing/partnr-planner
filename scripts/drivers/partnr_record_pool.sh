#!/usr/bin/env bash
# Generate induction material: run PARTNR's own privileged planner over a named pool and
# write one trace per episode.
#
# The planner is `baselines/record_rollouts.yaml`, which plans identically to the
# `heuristic_full_obs` ceiling and only adds the trace. Nothing about the memory is in the
# loop here, so the material is the benchmark's own behaviour and a pool can be enlarged
# at will -- which is the point: train holds 1731 pure-rearrangement episodes carrying
# `is_in_room` against the 15 that train_mini happened to contain.
#
#   bash scripts/drivers/partnr_record_pool.sh <pool> [more pools...]
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
PROCS=${PROCS:-40}
HARD_TIMEOUT=${HARD_TIMEOUT:-7200}
STALL_SECONDS=${STALL_SECONDS:-1200}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
cd "$ROOT" || exit 1

for pool in "$@"; do
  out="outputs/record/$pool"
  traces="results/partnr_rollouts/$pool"
  if [ -f "$out/DONE" ]; then echo "skip $pool"; continue; fi
  mkdir -p "$out" "$traces"
  export PARTNR_RECORD_DIR="$ROOT/$traces"
  started=$(date +%s)
  "$PY" -m habitat_llm.examples.planner_demo \
      --config-name baselines/record_rollouts.yaml \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$pool.json.gz" \
      num_proc="$PROCS" evaluation.save_video=False \
      hydra.run.dir="$out" >> "$out/run.log" 2>&1 &
  runner=$!
  echo "$runner" > "$out/PID"
  echo "[$(date +%H:%M:%S)] record $pool pid=$runner -> $traces"
  last=-1; changed=$started
  while kill -0 "$runner" 2>/dev/null; do
    sleep 30
    now=$(date +%s); count=$(ls "$traces" 2>/dev/null | wc -l)
    if [ "$count" -ne "$last" ]; then last=$count; changed=$now; fi
    if [ $((now - changed)) -ge "$STALL_SECONDS" ]; then
      echo "[$(date +%H:%M:%S)] STALL $pool at $count traces -- killing $runner"
      kill -9 "$runner" 2>/dev/null; break
    fi
    if [ $((now - started)) -ge "$HARD_TIMEOUT" ]; then
      echo "[$(date +%H:%M:%S)] TIMEOUT $pool at $count traces -- killing $runner"
      kill -9 "$runner" 2>/dev/null; break
    fi
  done
  wait "$runner" 2>/dev/null; status=$?
  count=$(ls "$traces" 2>/dev/null | wc -l)
  printf '{"pool":"%s","traces":%d,"status":%d,"seconds":%d}\n' \
      "$pool" "$count" "$status" "$(( $(date +%s) - started ))" > "$out/RECORD.json"
  [ "$status" -eq 0 ] && touch "$out/DONE"
  cat "$out/RECORD.json"
done
