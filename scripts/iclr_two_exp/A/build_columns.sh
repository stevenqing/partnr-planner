#!/usr/bin/env bash
# Column memories for the replay-mined arm, built by the UNCHANGED scripts/viki_union_library.py
# exactly as scripts/drivers/viki_v2_evaluate.py builds the admitted ones:
#   memory_all.json            union of all 14 family libraries (ID, CG w/ and w/o Image)
#   memory_heldout_<f>.json    union of the other 13 (single-family OOD), one per fold family
# Layer 2/3 of each output are then replaced by the admitted memory's own (SPEC A2: same ordering
# rules and vocabulary) in a separate step (swap_layers.py); the union's re-mined copies are kept
# beside them for the equality check.
set -u
ROOT=/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
PY=/root/venvs/partnr/bin/python
cd "$ROOT" || exit 1
LIBS=results/iclr_two_exp_2026-09-22/A/cert/family_libraries
OUT=results/iclr_two_exp_2026-09-22/A/memories_union
mkdir -p "$OUT"
read -r -a BUILD < <($PY -c "import json;print(' '.join(json.load(open('results/frozen_sweep_v2.json'))['build_families']))")
read -r -a FOLD < <($PY -c "import json;print(' '.join(json.load(open('results/frozen_sweep_v2.json'))['fold_families']))")
libs_for () { for f in "$@"; do [ -f "$LIBS/library_$f.json" ] && echo "$LIBS/library_$f.json"; done; }
pids=()
$PY scripts/viki_union_library.py --libraries $(libs_for "${BUILD[@]}") --families "${BUILD[@]}" \
    --out "$OUT/memory_all.json" > "$OUT/union_all.log" 2>&1 &
pids+=($!)
for held in "${FOLD[@]}"; do
  rest=(); for f in "${BUILD[@]}"; do [ "$f" != "$held" ] && rest+=("$f"); done
  $PY scripts/viki_union_library.py --libraries $(libs_for "${rest[@]}") --families "${rest[@]}" \
      --excluded-family "$held" --out "$OUT/memory_heldout_$held.json" > "$OUT/union_$held.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do while kill -0 "$p" 2>/dev/null; do sleep 5; done; done
echo "done $(ls $OUT/*.json | wc -l) memories"
