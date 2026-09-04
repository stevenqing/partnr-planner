#!/usr/bin/env bash
# Re-measure the privileged sweep, because its archived numbers do not reproduce.
#
# `v2_memory_R` is archived at 0.7400 (compositional slope 0.812) from the 09-02 overnight
# sweep. Four reruns -- my edited planner, the committed planner, with and without
# `+resume`, and finally the entire tree restored to HEAD -- all return 0.6752 / 0.736,
# bit-identical to each other. The composed hydra config is byte-identical to the archived
# run's, the operator library predates it and has not been touched, and the reruns are
# deterministic, so this is not run-to-run noise: whatever produced 0.7400 is not
# recoverable from the repository as committed.
#
# Every cell in that sweep ran in the same batch, so none of them can be quoted until it
# has been re-measured. These four are the rest of it. Privileged goals mean no model is
# called at all, so this costs GPU time and nothing else.
#
# Results go to a separate directory rather than over the archive: the archived numbers
# are evidence about what happened on 09-02 and deleting them would destroy the only
# record of the discrepancy.
set -u

ROOT=/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
SPLIT=val_mini
PROCS=${PROCS:-20}
GPU=${GPU:-1}
OUT=outputs/sweep_remeasured/$SPLIT

cd "$ROOT" || exit 1
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false
export VLLM_BASE_URL=http://127.0.0.1:8062/v1

both () { echo "evaluation.agents.agent_0.planner.plan_config.$1=$2 evaluation.agents.agent_1.planner.plan_config.$1=$2"; }
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

run () {             # name config extra...
  local name=$1 config=$2; shift 2
  local out="$OUT/$name"
  [ -f "$out/DONE" ] && { say "skip   $name"; return 0; }
  mkdir -p "$out"
  say "start  $name"
  CUDA_VISIBLE_DEVICES=$GPU timeout 28800 $PY -m habitat_llm.examples.planner_demo \
      --config-name "$config" \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/$SPLIT.json.gz" \
      num_proc="$PROCS" evaluation.save_video=False +resume=True \
      hydra.run.dir="$out" "$@" >> "$out/run.log" 2>&1
  local status=$? n
  n=$(ls "$out/results/$SPLIT.json.gz/stats" 2>/dev/null | wc -l)
  [ "$status" -eq 0 ] && touch "$out/DONE"
  say "$([ "$status" -eq 0 ] && echo done || echo FAILED)  $name ($n episodes)"
}

R=results/partnr_operators.json
ALL=results/partnr_operators_train_mini_all.json

run ceiling        baselines/heuristic_full_obs.yaml
run v2_memory_R    baselines/skill_memory_v2_oracle_goals.yaml $(both operators $R)
run v2_memory_all  baselines/skill_memory_v2_oracle_goals.yaml $(both operators $ALL)
run v2_R_nofold    baselines/skill_memory_v2_oracle_goals.yaml $(both operators $R) $(both allow_spatial_composition False)
run v2_R_noorder   baselines/skill_memory_v2_oracle_goals.yaml $(both operators $R) $(both use_episode_order False)

say "re-measurement finished -> $OUT"
