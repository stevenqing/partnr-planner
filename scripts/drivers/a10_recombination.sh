#!/bin/bash
# The recombination split, four arms over both of its forms.
#
# The two forms carry identical instances and identical ground truth and differ only
# in whether the model is shown the picture or the same scene written out. That pair
# is the control for the question the metadata could not answer -- whether an asset
# restored to init_pos appears in the row's image. If the forms agree, it did not
# matter; if the imaged form is worse, the missing object shows up as a measurement.
#
# skill_memory runs the fullactions configuration: the only renderer change that did
# not make things worse on either earlier split, and the one with the cleanest margin
# over trajectory_rag and zero_shot.
cd "$(dirname "$0")/../.." || exit 1
PY=/root/venvs/partnr/bin/python
BASE=http://192.168.32.40:8050/v1
D=results/viki_memory_experiments/amendment10
export TOKENIZERS_PARALLELISM=false
say(){ echo "===== $(date '+%m-%d %H:%M:%S') $* ====="; }

say "freezing the split before any inference"
$PY - <<'PYEOF'
import hashlib, json, pathlib
root = pathlib.Path(
    "results/viki_memory_experiments/amendment10"
)
def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
record = {
    "split_imaged_sha256": digest(root / "recombination.imaged.parquet"),
    "split_text_sha256": digest(root / "recombination.text.parquet"),
    "generator_sha256": digest("scripts/viki_amendment10_build_split.py"),
    "runner_sha256": digest("scripts/viki_amendment10_run.py"),
    "plan_format_sha256": digest("scripts/viki_plan_format.py"),
}
(root / "freeze.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
PYEOF

for SPLIT in imaged text; do
  for ARM in zero_shot trajectory_rag gmemory skill_memory; do
    TAG=""
    if [ "$ARM" = "skill_memory" ]; then
      export A8B_SKILL_TOPK=8 A9_ACTION_CAP=0 A9_ROLE_AWARE=0 A9_PATTERN_SLOTS=0 A9_MODE=""
      TAG="fullactions_k8"
    else
      unset A9_ACTION_CAP A9_STEP_ALIGNED
      export A8B_SKILL_TOPK=8 A9_ROLE_AWARE=0 A9_PATTERN_SLOTS=0 A9_MODE=""
    fi
    for r in 1 2 3; do
      say "split=$SPLIT arm=$ARM tag=$TAG round $r"
      $PY scripts/viki_amendment10_run.py --split "$SPLIT" --arm "$ARM" \
          --base-url $BASE --workers 8 ${TAG:+--tag $TAG} && break
      sleep 15
    done
  done
done

say "A10 RECOMBINATION FINISHED"
