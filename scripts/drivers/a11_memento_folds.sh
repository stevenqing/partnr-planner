#!/bin/bash
# The ported MEMENTO memory on eight held-out families.
#
# One run per family: the graph is rebuilt without that family's training episodes and
# only that family's test rows are scored, so the arm faces the same question ours does.
# Run sequentially -- three of these at once starve the shared endpoint.
cd "$(dirname "$0")/../.." || exit 1
P=/root/venvs/partnr/bin/python
export TOKENIZERS_PARALLELISM=false
for family in clear_table_with_two_robots_and_put_in_cabinet cut_fruit_on_board \
  toast_bread_and_set_plate cut_two_fruits_on_board parallel_human_dual_asset_to_plate_or_bowl \
  set_plate_and_fork_on_table ensure_all_fruits_on_table dog_push_box_for_two_panda_transport; do
  echo "===== $(date '+%H:%M:%S') fold $family ====="
  $P scripts/viki_eval_memento.py --top-k 1 --workers 14 \
     --exclude-family "$family" --only-family "$family" \
     --tag "memento_fold_${family}" 2>/dev/null | grep -E "^accuracy|^graph"
done
echo "===== all memento folds done ====="
