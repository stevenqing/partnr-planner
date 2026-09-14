#!/usr/bin/env bash
# RQ3 on 7B (:8061), strictly sequential: equivalence evidence on id, then the
# no_grounding and no_order cells (4 splits, OOD as 8 folds + assembly).
#
# Run (remote):
#   setsid nohup bash scripts/drivers/viki_rq3_7b_chain.sh \
#       > results/paper_viki_iclr2027/raw/rq3/logs/7B_chain.log 2>&1 < /dev/null &
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
PY=${PY:-/root/venvs/partnr/bin/python}
cd "$ROOT" || exit 1
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
LIBS=results/paper_viki_iclr2027/libraries/rq3
EQ=results/paper_viki_iclr2027/raw/rq3/equivalence/7B

eq () {   # name library flag-or-empty
  local name=$1 lib=$2 flag=$3
  say "equivalence $name"
  "$PY" scripts/viki_rq3_run_cell.py run --model 7B --condition "equiv_$name" --split id \
      --library "$lib" --base-library outputs/v3_memories/memory_all.json ${flag:+--flag=$flag} \
      --tag "rq3eq_${name}_7B_id" --out-dir "$EQ/$name" \
      --note "RQ3 equivalence evidence (not a paper cell)" || say "INCOMPLETE $name"
}
eq flag_no_grounding__full_lib     $LIBS/full/memory_all.json         --no-grounding
eq flag_no_order__full_lib         $LIBS/full/memory_all.json         --no-order
eq noflag__no_grounding_lib        $LIBS/no_grounding/memory_all.json ""
eq noflag__no_order_lib            $LIBS/no_order/memory_all.json     ""

MODEL=7B CONDITION=no_grounding bash scripts/drivers/viki_rq3_cells.sh
MODEL=7B CONDITION=no_order     bash scripts/drivers/viki_rq3_cells.sh
say "CHAIN DONE"
