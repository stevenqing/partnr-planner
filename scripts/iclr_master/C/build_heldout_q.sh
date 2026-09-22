#!/bin/bash
# C.1.1 route Q: rebuild the H_R library on the build half only, with the same builder and the same
# proposing model as the archived library (our_method/rebuild_all_memories.sh, H_R block):
#   build_hierarchical_skill_memory.py --include-failed --use-llm --use-api --patch-failed,
#   meta-llama/Llama-3.3-70B-Instruct behind vLLM, bf16, max-model-len 8192.
# Resource-only differences: TP 2 on GPUs 0,1 instead of TP 4; gpu-memory-utilization 0.80 and --enforce-eager instead of 0.9 with CUDA graphs
# (other users hold ~10-15 GB on these cards); port 8090. Weights from ModelScope, sha256-checked.
# Input: the heuristic H_R run of 2025-12-30 (traces + episode_result_log.csv), filtered to hr_build_ids.txt.
set -u
P=/mnt/pfs/devs/pn5wp/shishuqing
B=$P/iclr_master_C_build
REPO=$P/partnr-isambard-C
IDS=$P/partnr-planner/results/iclr_master_2026-09-22/C/hr_build_ids.txt
MODEL=$P/models/Llama-3.3-70B-Instruct
PORT=8090
OUTLIB=$B/hierarchical_heterogeneous_rerange_build
[ -e "$OUTLIB" ] && { echo "refuse: $OUTLIB exists"; exit 1; }
grep -q "DONE fail=0" $P/models/llama33_70b_download.log || { echo "model download not verified"; exit 1; }

# build-half results dir: filtered log (source order kept) + the original traces
mkdir -p $B/results/heterogeneous+rerange.json.gz
ln -sfn $B/src/traces $B/results/heterogeneous+rerange.json.gz/traces
/root/venvs/partnr/bin/python - "$B/src/episode_result_log.csv" "$IDS" "$B/results/episode_result_log.csv" <<'PY'
import csv, sys
src, ids, out = sys.argv[1:4]
keep = set(open(ids).read().split())
rows = list(csv.DictReader(open(src)))
sel = [r for r in rows if r["episode_id"] in keep]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(sel)
print("build episodes", len(sel), "failed", sum(float(r["task_state_success"]) < 1 for r in sel))
PY

export CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1
setsid $HOME/venvs/vllm/bin/vllm serve $MODEL \
  --served-model-name meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.80 --enforce-eager --host 127.0.0.1 --port $PORT > $B/vllm.log 2>&1 < /dev/null &
VPID=$!
echo $VPID > $B/vllm.pid
for i in $(seq 1 180); do
  curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200 && break
  kill -0 $VPID 2>/dev/null || { echo "vllm died"; exit 1; }
  sleep 10
done
curl -s http://127.0.0.1:$PORT/v1/models | grep -q Llama-3.3-70B || { echo "endpoint not up"; kill -TERM -- -$VPID; exit 1; }
echo "$(date '+%F %T') vllm up"

cd $REPO/our_method
timeout 21600 /root/venvs/partnr/bin/python build_hierarchical_skill_memory.py \
  --results-dir $B/results --output-dir $OUTLIB \
  --include-failed --use-llm --use-api --vllm-host 127.0.0.1 --vllm-port $PORT --patch-failed \
  > $B/build.log 2>&1 < /dev/null
rc=$?
echo "$(date '+%F %T') build rc=$rc"
echo "api_failed: $(grep -c 'API call failed' $B/build.log)"
echo "requests_200: $(grep -c 'POST /v1/chat/completions HTTP/1.1\" 200' $B/vllm.log)"
kill -TERM -- -$VPID 2>/dev/null; sleep 20; kill -KILL -- -$VPID 2>/dev/null
echo "$(date '+%F %T') vllm stopped"
