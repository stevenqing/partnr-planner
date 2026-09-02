#!/bin/bash
# G-Memory with its retrieval disabled: same arm, same prompt shape, same insights,
# but the trajectory is drawn uniformly from the permitted records instead of by
# similarity. Run over the same 8 held-out-family folds so it is directly paired
# with the real arm, row by row.
cd "$(dirname "$0")/../.." || exit 1
PY=/root/venvs/partnr/bin/python
BASE=http://192.168.32.40:8050/v1
export TOKENIZERS_PARALLELISM=false
export A9_GMEM_SHUFFLE=1
say(){ echo "===== $(date "+%m-%d %H:%M:%S") $* ====="; }

FAMILIES=$($PY scripts/viki_amendment9_folds.py --names 2>/dev/null)
COUNT=$(printf '%s\n' "$FAMILIES" | grep -c .)
say "families resolved: $COUNT"
if [ "$COUNT" -ne 8 ]; then
  say "expected 8 families, got $COUNT -- refusing to run a partial split"
  exit 1
fi

printf '%s\n' "$FAMILIES" | while read -r fam; do
  [ -z "$fam" ] && continue
  for r in 1 2 3; do
    say "fold=$fam arm=gmemory variant=shuffled round $r"
    $PY scripts/viki_amendment8b.py run-arm --arm gmemory --base-url $BASE \
        --workers 8 --fold "$fam" --variant shuffled && break
    sleep 15
  done
done
say "A9 SHUFFLE CONTROL FINISHED"
