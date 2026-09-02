#!/bin/bash
# The renderer cut every skill's action sequence at five. For a two-agent skill the
# stored sequence is 11-12 actions, so the cut always landed before the tail --
# which for the clear_table family is where Open[cabinet] lives, the one action
# those 220 rows cannot be solved without. This runs the same K=8 configuration as
# queryfix_k8 with the cap removed and nothing else changed.
cd "$(dirname "$0")/../.." || exit 1
PY=/root/venvs/partnr/bin/python
BASE=http://192.168.32.40:8050/v1
export TOKENIZERS_PARALLELISM=false
export A8B_SKILL_TOPK=8 A9_ROLE_AWARE=0 A9_PATTERN_SLOTS=0 A9_MODE=""
export A9_ACTION_CAP=0
say(){ echo "===== $(date "+%m-%d %H:%M:%S") $* ====="; }
say "config: TOPK=$A8B_SKILL_TOPK ACTION_CAP=$A9_ACTION_CAP (0 = no truncation)"
for r in 1 2 3; do
  say "in-distribution run, round $r"
  $PY scripts/viki_amendment8b.py run-arm --arm skill_memory --base-url $BASE \
      --workers 8 --variant fullactions_k8 && break
  sleep 15
done
say "A9 FULLACTIONS FINISHED"
