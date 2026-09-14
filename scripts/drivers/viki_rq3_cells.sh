#!/usr/bin/env bash
# RQ3 representation ablation cells on the current v3 library, one model x one condition.
#
#   MODEL=72B|30B|7B  CONDITION=full|no_grounding|no_order
#
# Four canonical splits, each an independent run directory with its own run.json:
#   id                  memory_all.json on the 924-row id manifest
#   cg_image            memory_all.json on recombination-imaged (297)
#   pure_text           memory_all.json on recombination-text (297)
#   ood_single_family   8 fold runs, each fold library (transformed per condition) on its own
#                       family's id rows, then assembled into 924 unique rows
#
# Libraries come from results/paper_viki_iclr2027/libraries/rq3/<condition>/ (written by
# scripts/viki_rq3_transform_library.py; `full` is a byte copy of outputs/v3_memories).
# The ablation mechanism is the evaluator's runtime flag; see transform_report.json for
# why that, and not the library bytes, is authoritative for no_grounding.
#
# Cells replay the per-model archived answers (viki_v2_evaluate.REPLAY); only re-asks are
# live. Runs are strictly sequential: one job on the endpoint at a time. Re-running the
# driver resumes: complete runs are skipped, incomplete ones fill only missing indices.
#
# Run (remote):
#   MODEL=7B CONDITION=no_order setsid nohup bash scripts/drivers/viki_rq3_cells.sh \
#       > results/paper_viki_iclr2027/raw/rq3/logs/7B_no_order.log 2>&1 < /dev/null &
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
MODEL=${MODEL:?set MODEL=72B|30B|7B}
CONDITION=${CONDITION:?set CONDITION=full|no_grounding|no_order}
SPLITS=${SPLITS:-"id cg_image pure_text ood_single_family"}
cd "$ROOT" || exit 1
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

case "$MODEL" in 72B|30B|7B) ;; *) echo "MODEL must be 72B / 30B / 7B"; exit 1 ;; esac
case "$CONDITION" in
  full) FLAG=() ;;
  no_grounding) FLAG=(--flag=--no-grounding) ;;
  no_order) FLAG=(--flag=--no-order) ;;
  *) echo "CONDITION must be full / no_grounding / no_order"; exit 1 ;;
esac
LIBS=results/paper_viki_iclr2027/libraries/rq3/$CONDITION
BASE=outputs/v3_memories
RAW=results/paper_viki_iclr2027/raw/rq3/$MODEL/$CONDITION
FOLDS="clear_table_with_two_robots_and_put_in_cabinet cut_fruit_on_board cut_two_fruits_on_board
dog_push_box_for_two_panda_transport ensure_all_fruits_on_table parallel_human_dual_asset_to_plate_or_bowl
set_plate_and_fork_on_table toast_bread_and_set_plate"
[ -f "$LIBS/memory_all.json" ] || { say "未执行，缺 $LIBS/memory_all.json (run viki_rq3_transform_library.py)"; exit 1; }

ran=0; ok=0
for split in $SPLITS; do
  if [ "$split" != ood_single_family ]; then
    ran=$((ran+1))
    say "start  $MODEL $CONDITION $split"
    "$PY" scripts/viki_rq3_run_cell.py run --model "$MODEL" --condition "$CONDITION" --split "$split" \
        --library "$LIBS/memory_all.json" --base-library "$BASE/memory_all.json" "${FLAG[@]}" \
        --tag "rq3_${CONDITION}_${MODEL}_${split}" --out-dir "$RAW/$split" \
      && { ok=$((ok+1)); say "done   $split"; } || say "INCOMPLETE $split"
  else
    dirs=()
    for fam in $FOLDS; do
      ran=$((ran+1))
      lib="$LIBS/memory_heldout_${fam}.json"
      d="$RAW/ood_single_family/fold_${fam}"; dirs+=("$d")
      say "start  $MODEL $CONDITION ood fold $fam"
      "$PY" scripts/viki_rq3_run_cell.py run --model "$MODEL" --condition "$CONDITION" \
          --split ood_single_family --family "$fam" \
          --library "$lib" --base-library "$BASE/memory_heldout_${fam}.json" "${FLAG[@]}" \
          --tag "rq3_${CONDITION}_${MODEL}_fold_${fam}" --out-dir "$d" \
        && { ok=$((ok+1)); say "done   fold $fam"; } || say "INCOMPLETE fold $fam"
    done
    say "assemble ood_single_family"
    "$PY" scripts/viki_rq3_run_cell.py assemble --model "$MODEL" --condition "$CONDITION" \
        --split ood_single_family --tag "rq3_${CONDITION}_${MODEL}_ood_single_family" \
        --out-dir "$RAW/ood_single_family/assembled" --fold-dirs "${dirs[@]}" \
      && say "done   ood assembly" || say "INCOMPLETE ood assembly"
  fi
done
say "finished $MODEL $CONDITION: $ok/$ran runs complete"
