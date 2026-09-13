#!/usr/bin/env bash
# Smoke test for ReAct + Achieve (the skill memory as a tool) at 7B, three val_mini episodes.
#
# Not a result: three episodes, one worker, sharing GPU 0 with tonight's head-to-head. It
# answers only whether the arm starts, whether the model ever calls Achieve, and whether a
# call runs a grounded body to "Successful execution!" in the simulator.
#
#   bash scripts/drivers/partnr_memskill_smoke.sh [episode ids, default 0 101 102]
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
IDS=${*:-0 101 102}
OUT=${OUT:-outputs/memskill_smoke_0913}
URL=http://127.0.0.1:8061/v1
MODEL=qwen2.5-vl-7b
code=$(curl -s -m 10 -o /dev/null -w "%{http_code}" "$URL/models")
[ "$code" = "200" ] || { echo "REFUSING: $URL returned $code"; exit 4; }
used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$used" -lt 92000 ] || { echo "REFUSING: GPU 0 at ${used} MiB"; exit 5; }
mkdir -p "$OUT"
filter="[$(echo $IDS | tr ' ' ',')]"
export VLLM_BASE_URL=$URL MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }
echo "[$(date +%H:%M:%S)] smoke episodes $filter -> $OUT"
CUDA_VISIBLE_DEVICES=0 timeout 3600 /root/venvs/partnr/bin/python -m habitat_llm.examples.planner_demo \
    --config-name baselines/react_memskill_vllm.yaml \
    habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
    num_proc=3 evaluation.save_video=False hydra.run.dir="$OUT" \
    "+episode_id_filter=$filter" $(both llm.generation_params.model $MODEL) \
    >> "$OUT/run.log" 2>&1
echo "[$(date +%H:%M:%S)] exit $?"
