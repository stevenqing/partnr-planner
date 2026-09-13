#!/usr/bin/env bash
# The model-arm columns of val_mini that have never come back complete: 7B and 8B, each
# base (21 operators) against accepted (22, with is_in_room).
#
# Why the earlier two attempts are not resumed: 09-07 was cut by a 4h timeout, 09-08 lost its
# endpoint mid-run, and the harness has changed since (world_graph no longer raises on a
# detached object). Resume across code versions is not safe, so this is a fresh directory.
#
# Layout. Only GPUs 0 and 1 are ours. GPU 1 keeps the 7B endpoint (8061, 0.30); the 8B
# endpoint comes back on GPU 0 at 0.30 with the archived args otherwise. Each GPU renders
# its own model's pair. Two cells x 45 workers on one card measured ~0.6 G per worker, so
# endpoint + pair is ~85 G of 98 G, and on gate_S3 two cells on one card did not slow each
# other: the card is not the limit, the 180 cores are, and 4 x 45 is what they hold.
#
#   bash scripts/drivers/partnr_model_valmini_0913.sh <pid to wait for first>
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
WAIT_PID=${1:?pid of the job that holds GPU 0 now}
OUT=outputs/report/model_0913
LIB=results/partnr_operators_iir1.json
BASE=results/partnr_operators.json
POOL=val_mini
PY=/root/venvs/partnr/bin/python
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
mkdir -p "$OUT"

while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
say "pid $WAIT_PID gone"

wait_gpu () {  # gpu floor_mib
  until [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt "$2" ]; do sleep 30; done
}

# ---- 8B endpoint on GPU 0
wait_gpu 0 5000
if ! curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:8101/v1/models | grep -q 200; then
  say "starting Qwen3-8B on GPU 0 :8101"
  CUDA_VISIBLE_DEVICES=0 setsid nohup /root/venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model /mnt/pfs/toolathlon_audit/models/Qwen3-8B --served-model-name Qwen3-8B --port 8101 \
    --seed 0 --max-model-len 32768 --gpu-memory-utilization 0.30 \
    > /root/vllm-qwen3-8b-0913.log 2>&1 < /dev/null &
  for i in $(seq 1 60); do
    curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:8101/v1/models | grep -q 200 && break
    sleep 10
  done
fi
for url in http://127.0.0.1:8061/v1 http://127.0.0.1:8101/v1; do
  code=$(curl -s -m 10 -o /dev/null -w "%{http_code}" "$url/models")
  [ "$code" = "200" ] || { say "REFUSING: $url returned $code"; exit 4; }
done

# No-think smoke: with thinking on, 8B parses zero requirements and the column scores zero.
reply=$(curl -s -m 60 http://127.0.0.1:8101/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen3-8B","max_tokens":64,"temperature":0,
  "chat_template_kwargs":{"enable_thinking":false},
  "messages":[{"role":"user","content":"State what must be true: put the cup on the table. One predicate per line, e.g. is_on_top(cup, table). Requirements:"}]}')
case "$reply" in
  *"<think>"*|"") say "REFUSING: 8B no-think smoke failed: ${reply:0:300}"; exit 5 ;;
esac
say "8B no-think smoke ok: $(echo "$reply" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"][:120].replace(chr(10)," | "))')"

cell () {  # tag model url nothink gpu operators
  MODEL=$2 URL=$3 NO_THINK=$4 GPU=$5 OPERATORS=$6 PROCS=45 HARD_TIMEOUT=28800 STALL_SECONDS=1800 \
    bash scripts/drivers/partnr_model_cell.sh $POOL "$1" "$OUT" +resume=True
}
cell m7b_base     qwen2.5-vl-7b http://127.0.0.1:8061/v1 0 1 $BASE &
cell m7b_accepted qwen2.5-vl-7b http://127.0.0.1:8061/v1 0 1 $LIB  &
wait_gpu 0 40000
cell m8b_base     Qwen3-8B      http://127.0.0.1:8101/v1 1 0 $BASE &
cell m8b_accepted Qwen3-8B      http://127.0.0.1:8101/v1 1 0 $LIB  &
say "four cells started"
wait

for tag in m7b m8b; do
  $PY scripts/partnr_gate_compare.py --pool $POOL --a "$OUT/${tag}_base" --b "$OUT/${tag}_accepted" \
    --json "$OUT/compare_${tag}_${POOL}.json" > "$OUT/compare_${tag}_${POOL}.txt" 2>&1
  say "$tag compared -- read 'complete' first"
done
say "ALL DONE"
