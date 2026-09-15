#!/usr/bin/env bash
# Ours-pipeline replay cells for RQ2 / RQ3 conditions, one model x one condition x chosen splits,
# against a chosen endpoint. Generalises viki_rq3_cells.sh (which is RQ3-only and uses the
# evaluator's built-in endpoint) so conditions and models can run in parallel on replicas.
#
#   EXPERIMENT=rq2|rq3  MODEL=72B|30B|7B  CONDITION=<name>  [SPLITS="id cg_image pure_text ood_single_family"]
#   [BASE_URL=http://127.0.0.1:PORT/v1]  [WORKERS=8]
#
# Libraries: results/paper_viki_iclr2027/libraries/<experiment>/<condition>/memory_all.json and
# memory_heldout_<family>.json (OOD always runs the fold library, never the 14-family one).
#   rq3 no_grounding / no_order add the evaluator's runtime flag (authoritative, see transform_report.json).
#   rq2 conditions use the library as is (no flag).
# Rows land in results/paper_viki_iclr2027/raw/<experiment>/<MODEL>/<CONDITION>/<split>/.
# Re-running resumes: complete runs are skipped, incomplete ones fill only missing indices.
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
EXPERIMENT=${EXPERIMENT:?set EXPERIMENT=rq2|rq3}
MODEL=${MODEL:?set MODEL=72B|30B|7B}
CONDITION=${CONDITION:?set CONDITION}
SPLITS=${SPLITS:-"id cg_image pure_text ood_single_family"}
WORKERS=${WORKERS:-8}
BASE_URL=${BASE_URL:-}
cd "$ROOT" || exit 1
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

case "$MODEL" in 72B|30B|7B) ;; *) echo "MODEL must be 72B / 30B / 7B"; exit 1 ;; esac
FLAG=()
case "$EXPERIMENT:$CONDITION" in
  rq3:no_grounding) FLAG=(--flag=--no-grounding) ;;
  rq3:no_order)     FLAG=(--flag=--no-order) ;;
  rq3:full|rq2:full|rq2:no_trace|rq2:no_execution_admission) ;;
  *) echo "unsupported $EXPERIMENT condition $CONDITION"; exit 1 ;;
esac
LIBS=results/paper_viki_iclr2027/libraries/$EXPERIMENT/$CONDITION
BASE=outputs/v3_memories
RAW=results/paper_viki_iclr2027/raw/$EXPERIMENT/$MODEL/$CONDITION
URL=(); [ -n "$BASE_URL" ] && URL=(--base-url "$BASE_URL")
FOLDS="clear_table_with_two_robots_and_put_in_cabinet cut_fruit_on_board cut_two_fruits_on_board
dog_push_box_for_two_panda_transport ensure_all_fruits_on_table parallel_human_dual_asset_to_plate_or_bowl
set_plate_and_fork_on_table toast_bread_and_set_plate"
[ -f "$LIBS/memory_all.json" ] || { say "未执行，缺 $LIBS/memory_all.json"; exit 1; }

# TIMEOUT (seconds) overrides the runner's per-split default (1800 s), which a replay with a near-empty
# library -- most rows re-asked live -- can exceed on the 924-row id split.
TIMEOUT=${TIMEOUT:-}
TO=(); [ -n "$TIMEOUT" ] && TO=(--timeout "$TIMEOUT")
common=(--experiment "$EXPERIMENT" --model "$MODEL" --condition "$CONDITION" --workers "$WORKERS" "${URL[@]}" "${TO[@]}")
ran=0; ok=0
for split in $SPLITS; do
  if [ "$split" != ood_single_family ]; then
    ran=$((ran+1))
    say "start  $EXPERIMENT $MODEL $CONDITION $split"
    "$PY" scripts/viki_rq3_run_cell.py run "${common[@]}" --split "$split" \
        --library "$LIBS/memory_all.json" --base-library "$BASE/memory_all.json" "${FLAG[@]}" \
        --tag "${EXPERIMENT}_${CONDITION}_${MODEL}_${split}" --out-dir "$RAW/$split" \
      && { ok=$((ok+1)); say "done   $split"; } || say "INCOMPLETE $split"
  else
    dirs=()
    for fam in $FOLDS; do
      ran=$((ran+1))
      lib="$LIBS/memory_heldout_${fam}.json"
      d="$RAW/ood_single_family/fold_${fam}"; dirs+=("$d")
      [ -f "$lib" ] || { say "未执行 fold $fam，缺 $lib"; continue; }
      say "start  $EXPERIMENT $MODEL $CONDITION ood fold $fam"
      "$PY" scripts/viki_rq3_run_cell.py run "${common[@]}" --split ood_single_family --family "$fam" \
          --library "$lib" --base-library "$BASE/memory_heldout_${fam}.json" "${FLAG[@]}" \
          --tag "${EXPERIMENT}_${CONDITION}_${MODEL}_fold_${fam}" --out-dir "$d" \
        && { ok=$((ok+1)); say "done   fold $fam"; } || say "INCOMPLETE fold $fam"
    done
    say "assemble ood_single_family"
    "$PY" scripts/viki_rq3_run_cell.py assemble --experiment "$EXPERIMENT" --model "$MODEL" --condition "$CONDITION" \
        --split ood_single_family --tag "${EXPERIMENT}_${CONDITION}_${MODEL}_ood_single_family" \
        --out-dir "$RAW/ood_single_family/assembled" --fold-dirs "${dirs[@]}" \
      && say "done   ood assembly" || say "INCOMPLETE ood assembly"
  fi
done
say "finished $EXPERIMENT $MODEL $CONDITION: $ok/$ran runs complete"
