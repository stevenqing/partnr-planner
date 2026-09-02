#!/bin/bash
# Eight held-out-family folds: rebuild the memory as if the family had never existed.
#
# The families are the eight the test manifest carries, which is the split amendment9
# already used, so the numbers sit beside the archived plan-writing arms rather than
# beside a split of their own. For each fold both memory layers are rebuilt without the
# family: the operator library re-induced from the remaining training episodes, and the
# vocabulary re-harvested from them. Nothing about the goal parser changes, because it
# is zero-shot and never saw a family in the first place -- which is also why this whole
# experiment costs no tokens.
cd "$(dirname "$0")/../.." || exit 1
PY=/root/venvs/partnr/bin/python
export TOKENIZERS_PARALLELISM=false
say(){ echo "===== $(date '+%m-%d %H:%M:%S') $* ====="; }

FAMILIES="clear_table_with_two_robots_and_put_in_cabinet cut_fruit_on_board \
toast_bread_and_set_plate cut_two_fruits_on_board parallel_human_dual_asset_to_plate_or_bowl \
set_plate_and_fork_on_table ensure_all_fruits_on_table dog_push_box_for_two_panda_transport"

for family in $FAMILIES; do
  say "fold: holding out $family"
  $PY scripts/viki_amendment11_vocabulary.py --exclude-family "$family" \
      --out "vocabulary.fold_${family}.json" 2>/dev/null | tail -4
  $PY scripts/viki_amendment11_induce.py --per-family 250 --exclude-family "$family" \
      --out "operators.fold_${family}.json" 2>/dev/null | head -5
done
say "all folds built"
