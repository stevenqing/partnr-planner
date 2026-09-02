#!/bin/bash
# Held-out-family evaluation. Lives in the repo rather than being written over ssh:
# the previous version was assembled inside a single-quoted ssh argument, and the
# quotes in sys.path.insert(0,'scripts') terminated that argument, so the family
# list came back empty and phase 3 silently ran nothing.
cd "$(dirname "$0")/../.." || exit 1
PY=/root/venvs/partnr/bin/python
BASE=http://192.168.32.40:8050/v1
D=results/viki_memory_experiments/amendment8b
export TOKENIZERS_PARALLELISM=false
say(){ echo "===== $(date "+%m-%d %H:%M:%S") $* ====="; }

# The fold arm must run the same configuration as the in-distribution variant it
# is named after. Without these exports skill_memory silently falls back to the
# defaults -- TOPK=4 and the renderer that scored 7.58% -- and the fold numbers
# would describe a configuration nobody chose.
VARIANT="${1:-queryfix_k8}"
case "$VARIANT" in
  queryfix_k8)    export A8B_SKILL_TOPK=8 A9_ROLE_AWARE=0 A9_PATTERN_SLOTS=0 A9_MODE="" ;;
  patternslot_k8) export A8B_SKILL_TOPK=8 A9_ROLE_AWARE=1 A9_PATTERN_SLOTS=2 A9_MODE="" ;;
  grounded_k8)    export A8B_SKILL_TOPK=8 A9_ROLE_AWARE=1 A9_PATTERN_SLOTS=2 A9_MODE=grounded A9_GROUND_TOPK=3 ;;
  rescore_k8)     export A8B_SKILL_TOPK=8 A9_ROLE_AWARE=1 A9_PATTERN_SLOTS=2 A9_MODE=rescore A9_GROUND_TOPK=3 ;;
  fullactions_k8) export A8B_SKILL_TOPK=8 A9_ROLE_AWARE=0 A9_PATTERN_SLOTS=0 A9_MODE="" A9_ACTION_CAP=0 ;;
  stepaligned_k8) export A8B_SKILL_TOPK=8 A9_ROLE_AWARE=0 A9_PATTERN_SLOTS=0 A9_MODE="" A9_ACTION_CAP=0 A9_STEP_ALIGNED=1 ;;
  *) echo "unknown variant $VARIANT"; exit 1 ;;
esac
say(){ echo "===== $(date "+%m-%d %H:%M:%S") $* ====="; }
say "fold config: VARIANT=$VARIANT TOPK=$A8B_SKILL_TOPK ROLE=$A9_ROLE_AWARE SLOTS=$A9_PATTERN_SLOTS MODE=$A9_MODE"
FAMILIES=$($PY scripts/viki_amendment9_folds.py --names 2>/dev/null)
COUNT=$(printf '%s\n' "$FAMILIES" | grep -c .)
say "families resolved: $COUNT"
if [ "$COUNT" -ne 8 ]; then
  say "expected 8 families, got $COUNT -- refusing to run a partial split"
  printf '%s\n' "$FAMILIES" | head
  exit 1
fi

printf '%s\n' "$FAMILIES" | while read -r fam; do
  [ -z "$fam" ] && continue
  for arm in zero_shot trajectory_rag gmemory skill_memory; do
    extra=""
    if [ "$arm" = "skill_memory" ] && [ -n "$VARIANT" ]; then extra="--variant $VARIANT"; fi
    for r in 1 2 3; do
      say "fold=$fam arm=$arm round $r"
      $PY scripts/viki_amendment8b.py run-arm --arm "$arm" --base-url $BASE --workers 8 \
          --fold "$fam" $extra && break
      sleep 15
    done
  done
  say "fold $fam done"
done

say "FOLD REPORT"
$PY scripts/viki_amendment9_fold_report.py "$VARIANT" 2>/dev/null
say "A9 FOLDS FINISHED"
