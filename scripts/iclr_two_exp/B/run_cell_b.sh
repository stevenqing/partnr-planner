#!/bin/bash
# One evaluation cell of SPEC Part B (held-out H_R, one-sided memory), Llama-3.1-8B.
# Base config: repo B all_scripts/bash_files/run_ours_hr_mem_hr.sh (the only H_R "ours" script on disk: H_R data,
# H_R memory slot, template rag_prompt_sequential_cooperation_skills, max_tokens 1500, rag_top_k 5, hierarchical
# retrieval, all ablation switches at their defaults). Changes: the dataset (one fold), the memory (the other
# fold's library), per-agent enable_rag for the condition, and the 09-22 switches (seed_override, PYTHONHASHSEED,
# env_over metrics, retrieval_dump). Backend: BACKEND=hf (in-process, original) or BACKEND=vllm (opt-in
# inference_mode vllm_openai, env ICLR_VLLM_URL).
# Validation mode (VALIDATE=C1|C2): the 09-22 C smoke config instead (v4 template, max_tokens 2000, route-Q library),
# to compare the vLLM backend against the HF smoke cells on disk.
#
# Usage: run_cell_b.sh <repo> <out_dir> <gpu> <num_proc> <dataset_rel> <cond both|a0|a1> <seed> <mem_rel> <timeout_s>
set -u
REPO=$1; OUT=$2; GPU=$3; NPROC=$4; DATA=$5; COND=$6; SEED=$7; MEM=$8; TMO=$9
PFS=/mnt/pfs/devs/pn5wp/shishuqing
PY=/root/venvs/partnr/bin/python
MODEL=$(ls -d $PFS/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/)
[ -f "$MODEL/config.json" ] || { echo "no model"; exit 2; }
BACKEND=${BACKEND:-hf}
cd "$REPO" || exit 2
mkdir -p "$OUT"
DUMP=$OUT/retrieval_dump
A0=evaluation.agents.agent_0.planner.plan_config
A1=evaluation.agents.agent_1.planner.plan_config
TEMPLATE=rag_prompt_sequential_cooperation_skills; MAXTOK=1500
if [ -n "${VALIDATE:-}" ]; then TEMPLATE=rag_prompt_sequential_cooperation_skills_v4; MAXTOK=2000; fi
MODE=hf; [ "$BACKEND" = vllm ] && MODE=vllm_openai

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
    $A.llm.generation_params.max_tokens=$MAXTOK
    $A.llm.inference_mode=$MODE
    $A.llm.generation_params.engine=$MODEL
    +$A.rag_top_k=5
    "$A.rag_dataset_dir=[$MEM]"
    "$A.rag_data_source_name=[hierarchical_memory]"
    $A.example_type=hierarchical
    +$A.ablation_config.include_coop_skills=True
    +$A.ablation_config.include_ind_skills=True
    +$A.ablation_config.use_hierarchical_structure=True
    +$A.ablation_config.random_retrieval=False
    instruct@$A.instruct=$TEMPLATE
    "+$A.retrieval_dump_dir=$DUMP"
  )
done
case ${VALIDATE:-$COND} in
  both|C1) extra=($A0.enable_rag=True $A1.enable_rag=True) ;;
  a0|C2)   extra=($A0.enable_rag=True $A1.enable_rag=False) ;;
  a1)      extra=($A0.enable_rag=False $A1.enable_rag=True) ;;
  *) echo "bad cond $COND"; exit 2 ;;
esac
extra+=(+seed_override=$SEED +iclr_env_over_metrics=True)
[ "$BACKEND" = hf ] && extra+=(+iclr_share_llm=True)

export CUDA_VISIBLE_DEVICES=$GPU
export HF_HOME=$PFS/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=$PFS/tmp
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO/third_party/transformers-CFG:$REPO
export PYTHONHASHSEED=$SEED
export ICLR_VLLM_LOG_DIR=$OUT/llm_calls
LOG=$OUT/run.log
{ echo "=== $(date '+%F %T') start cond=$COND seed=$SEED gpu=$GPU nproc=$NPROC backend=$BACKEND url=${ICLR_VLLM_URL:-} mem=$MEM repo=$REPO"; printf '%q ' "${common[@]}" "${extra[@]}"; echo; } >> $OUT/launch.txt
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
wait $RUN 2>/dev/null
echo "$(date '+%F %T') end why=$why elapsed=$(( $(date +%s) - start ))s" >> $OUT/launch.txt
exit 0
