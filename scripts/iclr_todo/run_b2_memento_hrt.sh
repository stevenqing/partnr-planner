#!/bin/bash
# B2 (ICLR TODO spec 2026-09-21): MEMENTO on H_R_T, source memory H_R+H_T (Table 2 row),
# Llama-3.1-8B-Instruct, HF in-process greedy -- the old Isambard pipeline (B repo) unchanged.
# Differs from the archived run_memento_rht_mem_rht.sh only in: memory list without rerange_only
# (Table 2 states H_R+H_T), model path, GPUs, output dir. Leakage checked 09-21: 0/11.
# Usage: run_b2_memento_hrt.sh <GPU list e.g. 2,3>
set -u
GPUS=${1:?gpus}
PFS=/mnt/pfs/devs/pn5wp/shishuqing
cd $PFS/partnr-isambard || exit 1
TAG=b2_memento_hrt_mem_hr_ht
OUT=$PFS/partnr-isambard/outputs/iclr_b2/$TAG
[ -e "$OUT" ] && { echo "refuse: $OUT exists"; exit 1; }
mkdir -p "$OUT"
LOG=$OUT/run.log
MODEL=$(ls -d $PFS/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ | head -1)
[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL"; exit 1; }
ls $MODEL/*.safetensors >/dev/null || { echo "no weights"; exit 1; }

export CUDA_VISIBLE_DEVICES=$GPUS
export HF_HOME=$PFS/hf HF_HUB_OFFLINE=1 TMPDIR=$PFS/tmp
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
PY=/root/venvs/partnr/bin/python
# the B repo's patched transformers-CFG (accepts Llama-3.x PreTrainedTokenizerFast); A's copy does not
export PYTHONPATH=$PFS/partnr-isambard/third_party/transformers-CFG${PYTHONPATH:+:$PYTHONPATH}
MEM="[data/memory_memento_dataset/heterogeneous_rerange/,data/memory_memento_dataset/heterogeneous_temporal/]"

set -x
timeout 14400 $PY -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    hydra.run.dir=$OUT/hydra \
    evaluation.sequential_execution=True \
    num_proc=4 \
    habitat.dataset.data_path="task_classification_datasets/heterogeneous+rerange+temporal_unique.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=$MODEL \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=$MODEL \
    evaluation.agents.agent_0.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_1.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir=$MEM \
    evaluation.agents.agent_1.planner.plan_config.rag_dataset_dir=$MEM \
    evaluation.agents.agent_0.planner.plan_config.rag_data_source_name=[habitat_trajectories] \
    evaluation.agents.agent_1.planner.plan_config.rag_data_source_name=[habitat_trajectories] \
    evaluation.agents.agent_0.planner.plan_config.example_type=memento \
    evaluation.agents.agent_1.planner.plan_config.example_type=memento \
    +evaluation.agents.agent_0.planner.plan_config.memory_path=data/memory_memento_dataset/heterogeneous_rerange \
    +evaluation.agents.agent_1.planner.plan_config.memory_path=data/memory_memento_dataset/heterogeneous_rerange \
    +evaluation.agents.agent_0.planner.plan_config.ensure_same_scene=False \
    +evaluation.agents.agent_1.planner.plan_config.ensure_same_scene=False \
    +evaluation.agents.agent_0.planner.plan_config.corresponding_memory=False \
    +evaluation.agents.agent_1.planner.plan_config.corresponding_memory=False \
    +evaluation.agents.agent_0.planner.plan_config.rag_top_k=3 \
    +evaluation.agents.agent_1.planner.plan_config.rag_top_k=3 \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=rag_prompt_enhanced \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=rag_prompt_enhanced \
    > "$LOG" 2>&1 < /dev/null &
RUN=$!
set +x
echo "$RUN" > "$OUT/pid"
# stall guard: log not growing for 20 min -> kill the whole process group
last=0; since=$(date +%s)
while kill -0 $RUN 2>/dev/null; do
  sleep 60
  sz=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  if [ "$sz" != "$last" ]; then last=$sz; since=$(date +%s)
  elif [ $(( $(date +%s) - since )) -ge 1200 ]; then
    echo "STALL: log static 20 min, killing" >> "$OUT/guard.txt"; kill $RUN; sleep 30; pkill -KILL -g $$ -f habitat_llm.examples.planner_demo; break
  fi
done
wait $RUN; rc=$?
echo "rc=$rc" > "$OUT/exit.txt"
{
  echo "memento_loaded: $(grep -c 'Loading 2 MEMENTO memory file' "$LOG")"
  echo "memento_missing: $(grep -ci 'MEMENTO.*not available\|HAS_MEMENTO_RETRIEVAL' "$LOG")"
  echo "progress: $(grep -o 'Progress: \[[0-9]*/[0-9]*\]' "$LOG" | tail -1)"
  echo "skipped_episodes: $(grep -c 'Skipping evaluating episode' "$LOG")"
} >> "$OUT/exit.txt"
