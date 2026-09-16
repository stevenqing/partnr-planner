#!/usr/bin/env bash
# Finish the 30B val_mini cell of the frozen compositional config, then report it.
#
# It died of CUDA OOM at 278/369 on 09-16 09:08: 48 habitat processes (~0.75 GiB each) beside a 69 GiB
# endpoint on a 95 GiB card. Only the process count was wrong: a first resume try at util 0.62 refused to
# start ("Available KV cache memory: -7.67 GiB") because 61 GiB does not even hold the 58.2 GiB of weights.
# So the endpoint keeps util 0.74 (~69 GiB) and the cell gets 24 processes (~18 GiB): ~87 GiB of 95. Resume is safe because the code has not changed since the cell started
# (commit e24f51e; the only later commit adds RESUME to the driver). The report is the one 30B val report:
# do not tune on it, and quote it only if the cell reaches 369/369.
#
#   Q=outputs/cand_iface_0916/val_final; cp scripts/drivers/partnr_val30b_resume_0916.sh $Q/resume30b.sh
#   setsid nohup bash $Q/resume30b.sh > $Q/resume30b.log 2>&1 < /dev/null &
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PFS=/mnt/pfs/devs/pn5wp/shishuqing
PY=/root/venvs/partnr/bin/python
VLLM=/root/venvs/vllm/bin/python
Q=outputs/cand_iface_0916/val_final
VAL=outputs/cand_iface_0914/val_mini
REP=outputs/cand_iface_0914/val_mini_reports
H30=outputs/headtohead/val_mini
CELL=$VAL/typed_v7b_RS_final_30b
mkdir -p "$Q" "$PFS/tmp" "$PFS/vllm_cache"
export HF_HOME=$PFS/hf HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$PFS/vllm_cache TMPDIR=$PFS/tmp VLLM_ENGINE_READY_TIMEOUT_S=3600
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
# Always leave the marker, including on an early refusal: the 09:16 try exited before it and left a
# waiter polling for a file that could never appear.
trap 'touch "$Q/RESUME30B_DONE"' EXIT
probe () { curl -s -m 10 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/v1/models" | grep -q "^200$"; }
root_free_gb () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
have () { ls "$CELL/results/val_mini.json.gz/stats" 2>/dev/null | wc -l; }

[ -d "$CELL" ] || { say "REFUSING: $CELL does not exist -- nothing to resume"; exit 3; }
grep -q 'RESUME' scripts/drivers/partnr_typed_val_mini.sh || { say "REFUSING: val driver has no RESUME support"; exit 6; }
[ "$(root_free_gb)" -ge 5 ] || { say "ALARM root filesystem $(root_free_gb)G free"; exit 5; }
probe 8071 && { say "REFUSING: :8071 already answers"; exit 4; }
free_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n 1p)
[ "$free_mb" -lt 20000 ] || { say "REFUSING: GPU 0 already holds ${free_mb}MiB"; exit 4; }
say "resuming at $(have)/369; root free $(root_free_gb)G"

log=$Q/vllm-qwen3-vl-30b-8071-resume.log
(CUDA_VISIBLE_DEVICES=0 setsid nohup "$VLLM" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct --revision 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c \
    --served-model-name qwen3-vl-30b --port 8071 --tensor-parallel-size 1 --max-model-len 16384 \
    --gpu-memory-utilization 0.74 --limit-mm-per-prompt '{"image":1}' > "$log" 2>&1 < /dev/null &) > /dev/null 2>&1
for _ in $(seq 300); do probe 8071 && break; sleep 10; done
probe 8071 || { say "ALARM :8071 did not come up"; tail -8 "$log" | cut -c1-300; exit 4; }
say ":8071 up (util 0.74)"

EXAMPLES=RS TAG=final MTAG=30b MODEL=qwen3-vl-30b NO_THINK=1 GPU=0 PORT=8071 PROCS=24 RESUME=1 \
  SWITCHES="typed_stages=True typed_beside=True typed_same_object=True" \
  bash scripts/drivers/partnr_typed_val_mini.sh > "$Q/val_30b_resume.driver.log" 2>&1 < /dev/null
say "cell: $(tail -1 "$Q/val_30b_resume.driver.log")"
say "episodes now $(have)/369; OOM lines this run: $(grep -c 'out of memory' "$CELL/run.log")"

pid=$(ps -eo pid,args | awk '$0 ~ /vllm.entrypoints.openai.api_server/ && index($0, "--port 8071") {print $1}' | head -1)
[ -n "$pid" ] && { kill "$pid"; for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done; kill -0 "$pid" 2>/dev/null && kill -9 "$pid"; say ":8071 stopped"; }

if [ "$(have)" -eq 369 ]; then
  say "report val30b_report (cell complete)"
  timeout 3600 "$PY" scripts/partnr_typed_simpair_report.py --pool val_mini \
      --cell final_30b="$CELL" react="$H30/react" react_rag_R="$H30/react_rag_R" gmemory="$H30/gmemory" \
             memento="$H30/memento" v2_intent="$H30/v2_intent" v2_prompt="$H30/v2_prompt" final_7b="$VAL/typed_v7b_RS_final_7b" \
      --compare final_30b:react final_30b:react_rag_R final_30b:gmemory final_30b:memento final_30b:v2_intent \
                final_30b:v2_prompt final_30b:final_7b \
      --json "$REP/val_mini_typed_RS_final_30b.json" > "$Q/val30b_report.txt" 2>&1 < /dev/null \
      || say "WARN val30b_report exit $?"
  timeout 1800 "$PY" scripts/partnr_failure_classes.py --pool "$Q/val_mini_pool.json" --split val_mini \
      --cell final_30b="$CELL" --cell final_7b="$VAL/typed_v7b_RS_final_7b" --json "$Q/val30b_failures.json" \
      > "$Q/val30b_failures.txt" 2>&1 < /dev/null || say "WARN val30b_failures exit $?"
else
  say "ALARM cell still incomplete ($(have)/369); reports not rerun"
fi
touch "$Q/RESUME30B_DONE"
say "done"
