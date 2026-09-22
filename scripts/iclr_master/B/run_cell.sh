#!/bin/bash
# Run one PARTNR evaluation cell on the old Isambard pipeline (B repo = partnr-isambard).
# Usage: run_cell.sh <cell_id> <backbone> <argkey> <target> <seed> <gpus> <timeout_s> [num_proc=4]
#   backbone: llama8b | qwen7b ; argkey: file stem in args/ ; target: RS|HRT|HRST|SMOKE_RS|SMOKE_HR
# Output: $B/outputs/iclr_master/B/<cell_id>/{hydra/,run.log,pid,exit.txt,guard.txt}
# Always +resume=True (episodes whose planner-log exists are skipped); seed via planner_demo_seeded.py (+iclr_seed).
set -u
CELL=$1 BACKBONE=$2 ARGKEY=$3 TARGET=$4 SEED=$5 GPUS=$6 TIMEOUT=$7 NPROC=${8:-4}
PFS=/mnt/pfs/devs/pn5wp/shishuqing
B=$PFS/partnr-isambard
HERE=$PFS/partnr-planner/scripts/iclr_master/B
OUT=$B/outputs/iclr_master/B/$CELL
mkdir -p "$OUT"; cd $B || exit 1
LOG=$OUT/run.log
case $TARGET in
  RS) DS=rerange+spatial_matched_subtasks.json.gz ;;
  HRT) DS=heterogeneous+rerange+temporal_unique.json.gz ;;
  HRST) DS=heterogeneous+rerange+spatial+temporal.json.gz ;;
  SMOKE_RS) DS=iclr_master/smoke_rs2.json.gz ;;
  SMOKE_HR) DS=iclr_master/smoke_hr2.json.gz ;;
  DS:*) DS=${TARGET#DS:} ;;
  *) echo "bad target $TARGET"; exit 2 ;;
esac
case $BACKBONE in
  llama8b) MODEL=$(ls -d $PFS/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ | head -1)
           [ -n "$MODEL" ] && ls $MODEL/*.safetensors >/dev/null 2>&1 || { echo "no weights for $BACKBONE" | tee "$OUT/exit.txt"; exit 3; }
           LLM=("evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf"
                "evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf"
                "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=$MODEL"
                "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=$MODEL") ;;
  # Qwen2.5-7B: the archived Isambard Qwen runs (old/run_planner_demo_*qwen*.sh) use llm=qwen, inference_mode=hf,
  # constrained_generation=False and the Qwen-specific instruct yaml of each prompt family (*_qwen.yaml, whose
  # actions_parser is zero_shot_action_parser_qwen). The Llama templates end with the literal "Action[parameter]",
  # which Qwen copies as "Action[Explore[x]]" and the parser rejects (09-22: 76/76 format errors); the *_qwen
  # templates spell out the bracket format instead. Mapping below; MEMENTO's rag_prompt_enhanced has no Qwen variant.
  qwen7b)  MODEL=$(ls -d $PFS/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ 2>/dev/null | head -1)
           [ -n "$MODEL" ] && ls $MODEL/*.safetensors >/dev/null 2>&1 || { echo "no weights for $BACKBONE" | tee "$OUT/exit.txt"; exit 3; }
           LLM=("llm@evaluation.agents.agent_0.planner.plan_config.llm=qwen" "llm@evaluation.agents.agent_1.planner.plan_config.llm=qwen"
                "evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf"
                "evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf"
                "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=$MODEL"
                "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=$MODEL"
                "evaluation.agents.agent_0.planner.plan_config.constrained_generation=False"
                "evaluation.agents.agent_1.planner.plan_config.constrained_generation=False")
           QWEN_INSTRUCT='s/instruct=rag_prompt_sequential_cooperation_skills_v4$/instruct=rag_prompt_sequential_cooperation_skills_qwen/; s/instruct=rag_prompt_sequential_state_reflection$/instruct=rag_prompt_sequential_state_reflection_qwen/; s/instruct=zero_shot_prompt_org$/instruct=zero_shot_prompt_qwen/; s/instruct=zero_shot_prompt_tom$/instruct=zero_shot_prompt_qwen_tom/' ;;
  *) echo "bad backbone"; exit 2 ;;
esac
[ -f $B/habitat_llm/examples/${DEMO_MODULE:-planner_demo_seeded}.py ] || { echo "no demo module"; exit 4; }
mapfile -t ARGS < <(grep -v 'habitat.dataset.data_path=' $HERE/args/$ARGKEY.args | sed 's/^--config-name /--config-name=/' | sed "${QWEN_INSTRUCT:-}")
# the backbone block may set constrained_generation itself; drop the archived line then (no duplicate hydra keys)
if printf '%s\n' "${LLM[@]}" | grep -q constrained_generation; then
  mapfile -t ARGS < <(printf '%s\n' "${ARGS[@]}" | grep -v 'constrained_generation=')
fi

export CUDA_VISIBLE_DEVICES=$GPUS
export HF_HOME=$PFS/hf HF_HUB_OFFLINE=1 TMPDIR=$PFS/tmp
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$B/third_party/transformers-CFG${PYTHONPATH:+:$PYTHONPATH}
# Without a fixed hash seed the pipeline is not reproducible across processes: set iteration order changes
# the "Known Rooms" / "Seen objects" lists, hence the template retrieval query, the retrieved skills and the
# prompt (seen 09-22 03:20: same episode, same config, different first-turn prompt). The seed fixes it too.
export PYTHONHASHSEED=$SEED
PY=/root/venvs/partnr/bin/python
# DEMO_MODULE / ICLR_ENV_OVER=omit are only for the switch-off check (lane_verify.sh)
SWITCH=("+iclr_env_over_metrics=${ICLR_ENV_OVER:-True}"); [ "${ICLR_ENV_OVER:-}" = omit ] && SWITCH=()
# ICLR_SHARE_LLM=True|False (default False until verified; queue.sh sets it)
SWITCH+=("+iclr_share_llm=${ICLR_SHARE_LLM:-False}")
CMD=($PY -m habitat_llm.examples.${DEMO_MODULE:-planner_demo_seeded} "${ARGS[@]}"
  hydra.run.dir=$OUT/hydra +resume=True +iclr_seed=$SEED num_proc=$NPROC
  "${SWITCH[@]}"
  habitat.dataset.data_path=task_classification_datasets/$DS
  "${LLM[@]}")
printf '%s\n' "${CMD[@]}" > "$OUT/cmd.txt"
echo "=== start $(date -Is) attempt" >> "$OUT/attempts.txt"
setsid "${CMD[@]}" >> "$LOG" 2>&1 < /dev/null &
RUN=$!
echo $RUN > "$OUT/pid"
start=$(date +%s); last=-1; since=$start; why=""
while kill -0 $RUN 2>/dev/null; do
  sleep 60
  now=$(date +%s); sz=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  if [ "$sz" != "$last" ]; then last=$sz; since=$now; fi
  if [ $((now - since)) -ge 1200 ]; then why="STALL log static 20 min"; fi
  if [ $((now - start)) -ge $TIMEOUT ]; then why="TIMEOUT ${TIMEOUT}s"; fi
  if [ -n "$why" ]; then
    echo "$(date -Is) $why, killing pgid $RUN" >> "$OUT/guard.txt"
    kill -TERM -- -$RUN 2>/dev/null; sleep 30; kill -KILL -- -$RUN 2>/dev/null; break
  fi
done
wait $RUN; rc=$?
$PY $HERE/cell_verdict.py "$OUT" "$DS" "$rc" "$why" > "$OUT/exit.txt"
cat "$OUT/exit.txt"
