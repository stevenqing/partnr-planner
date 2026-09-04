#!/usr/bin/env bash
# Does closing the loop recover the compositional slope the open-loop composer loses?
#
# Two cells, both with privileged goals so no model is called and the only thing that
# varies is what a refusal costs: the requirement (open loop) or the body (closed loop).
#   the comparison   priv:v2_memory_R = 0.812 already on disk; upper bound is v2_prompt's
#                    0.917, which is this memory read by a loop that was closed all along.
set -u
ROOT=/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
cd "$ROOT" || exit 1
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
export VLLM_BASE_URL=http://127.0.0.1:8062/v1
both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }
out=outputs/sweep/val_mini/v2_open_ctl
mkdir -p "$out"
CUDA_VISIBLE_DEVICES=1 timeout 28800 $PY -m habitat_llm.examples.planner_demo \
  --config-name baselines/skill_memory_v2_oracle_goals.yaml \
  habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
  num_proc=16 evaluation.save_video=False +resume=True hydra.run.dir="$out" \
  $(both operators results/partnr_operators.json) \
  >> "$out/run.log" 2>&1
status=$?
n=$(ls "$out/results/val_mini.json.gz/stats" 2>/dev/null | wc -l)
[ "$status" -eq 0 ] && touch "$out/DONE"
echo "v2_open_ctl status=$status episodes=$n"
