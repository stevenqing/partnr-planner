#!/bin/bash
# run_cell.sh plus EXTRA_OVERRIDES (space-separated hydra overrides, e.g. the shared-model switch from part B).
# One evaluation cell of spec part C (H_R held-out, role-typed retrieval), Llama-3.1-8B, HF in-process greedy.
# Base config = Table 1 Llama-3.1-8B ours run (all_scripts/bash_files/run_ours_rs_mem_r.sh): v4 template,
# max_tokens 2000, rag_top_k 5, hierarchical retrieval, all ablation switches at their defaults.
# Only the dataset and the memory differ; per-condition overrides come from cond_overrides().
#
# Usage: run_cell.sh <repo_dir> <out_dir> <gpu> <num_proc> <dataset_rel> <cond C0..C4|C1plain> <seed|none> <timeout_s>
set -u
REPO=$1; OUT=$2; GPU=$3; NPROC=$4; DATA=$5; COND=$6; SEED=$7; TMO=$8
PFS=/mnt/pfs/devs/pn5wp/shishuqing
# held-out library from route Q (build_heldout_q.sh), copied here after the build
MEM=data/hierarchical_skill_memory_heldout_hr_q/hierarchical_heterogeneous_rerange/
PY=/root/venvs/partnr/bin/python
MODEL=$(ls -d $PFS/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ | head -1)
[ -f "$MODEL/config.json" ] || { echo "no model"; exit 2; }
cd "$REPO" || exit 2
mkdir -p "$OUT"
DUMP=$OUT/retrieval_dump
A0=evaluation.agents.agent_0.planner.plan_config
A1=evaluation.agents.agent_1.planner.plan_config
H0="You are Agent 0. Start with the first subtask in the instruction."
H1="You are Agent 1. The other agent starts with the first subtask. Choose a different object."

common=(
  --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml
  hydra.run.dir=$OUT/hydra
  evaluation.sequential_execution=True
  num_proc=$NPROC
  habitat.dataset.data_path=$DATA
  +resume=True
)
for A in $A0 $A1; do
  common+=(
    $A.llm.generation_params.max_tokens=2000
    $A.llm.inference_mode=hf
    $A.llm.generation_params.engine=$MODEL
    +$A.rag_top_k=5
    "$A.rag_dataset_dir=[$MEM]"
    "$A.rag_data_source_name=[hierarchical_memory]"
    $A.example_type=hierarchical
    +$A.ablation_config.include_coop_skills=True
    +$A.ablation_config.include_ind_skills=True
    +$A.ablation_config.use_hierarchical_structure=True
    +$A.ablation_config.random_retrieval=False
  )
done
common+=(
  instruct@$A0.instruct=rag_prompt_sequential_cooperation_skills_v4
  instruct@$A1.instruct=rag_prompt_sequential_cooperation_skills_v4
)
case $COND in
  C1plain) extra=($A0.enable_rag=True $A1.enable_rag=True) ;;   # switches off: identity check only
  C0) extra=($A0.enable_rag=False $A1.enable_rag=False) ;;
  C1) extra=($A0.enable_rag=True $A1.enable_rag=True) ;;
  C2) extra=($A0.enable_rag=True $A1.enable_rag=False) ;;
  C3) extra=($A0.enable_rag=True $A1.enable_rag=True) ;;   # agent 1 include_ind_skills=False, set below
  C4) extra=($A0.enable_rag=True $A1.enable_rag=True "+$A0.role_hint='$H0'" "+$A1.role_hint='$H1'") ;;
  *) echo "bad cond $COND"; exit 2 ;;
esac
# C3: the follower retrieves from the cooperation branch only (existing ablation switch of the B repo).
if [ "$COND" = C3 ]; then
  for i in "${!common[@]}"; do
    [ "${common[$i]}" = "+$A1.ablation_config.include_ind_skills=True" ] && common[$i]="+$A1.ablation_config.include_ind_skills=False"
  done
fi
if [ "$COND" != C1plain ]; then
  extra+=(+seed_override=$SEED "+$A0.retrieval_dump_dir=$DUMP" "+$A1.retrieval_dump_dir=$DUMP" +iclr_env_over_metrics=True)
fi

export CUDA_VISIBLE_DEVICES=$GPU
export HF_HOME=$PFS/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=$PFS/tmp
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO/third_party/transformers-CFG:$REPO
# The pipeline iterates over Python sets of room/object names when it writes the prompt, so with a random
# hash seed two runs of the pristine code already differ in the first prompt (seen 2026-09-22 on episode 439).
# Fix the hash seed: to the run seed for C cells, to 0 for the switch-off identity check.
if [ "$SEED" = none ]; then export PYTHONHASHSEED=0; else export PYTHONHASHSEED=$SEED; fi

# shellcheck disable=SC2206
[ -n "${EXTRA_OVERRIDES:-}" ] && extra+=(${EXTRA_OVERRIDES})
LOG=$OUT/run.log
{ echo "=== $(date '+%F %T') start cond=$COND seed=$SEED gpu=$GPU nproc=$NPROC repo=$REPO"; printf '%q ' "${common[@]}" "${extra[@]}"; echo; } >> $OUT/launch.txt
setsid $PY -m habitat_llm.examples.planner_demo "${common[@]}" "${extra[@]}" >> "$LOG" 2>&1 < /dev/null &
RUN=$!
echo $RUN > $OUT/pid
start=$(date +%s); last=-1; since=$start; why=done
while kill -0 $RUN 2>/dev/null; do
  sleep 60
  sz=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  now=$(date +%s)
  if [ "$sz" != "$last" ]; then last=$sz; since=$now; fi
  if [ $((now - since)) -ge 1200 ]; then why=stall; fi
  if [ $((now - start)) -ge $TMO ]; then why=timeout; fi
  if [ $why != done ]; then
    echo "$(date '+%F %T') $why: killing pgid $RUN" >> $OUT/guard.txt
    kill -TERM -- -$RUN 2>/dev/null; sleep 30; kill -KILL -- -$RUN 2>/dev/null
    break
  fi
done
wait $RUN 2>/dev/null; rc=$?
echo "$(date '+%F %T') end rc=$rc why=$why elapsed=$(( $(date +%s) - start ))s" >> $OUT/launch.txt
exit 0
