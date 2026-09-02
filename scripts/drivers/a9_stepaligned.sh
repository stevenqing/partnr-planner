#!/bin/bash
# The renderer showed five entries of a flattened action chain. This shows the real
# per-step, per-robot structure recovered from the training parquet, for the skills
# whose source plans agree on it, and the uncapped flat list for the rest.
cd "$(dirname "$0")/../.." || exit 1
PY=/root/venvs/partnr/bin/python
BASE=http://192.168.32.40:8050/v1
export TOKENIZERS_PARALLELISM=false
export A8B_SKILL_TOPK=8 A9_ROLE_AWARE=0 A9_PATTERN_SLOTS=0 A9_MODE=""
export A9_ACTION_CAP=0 A9_STEP_ALIGNED=1
say(){ echo "===== $(date "+%m-%d %H:%M:%S") $* ====="; }
say "gate first: the run is refused if the structure does not reach the prompt"
$PY scripts/viki_amendment9_stepalign_gate.py || { say "GATE FAILED -- not running"; exit 1; }
for r in 1 2 3; do
  say "in-distribution run, round $r"
  $PY scripts/viki_amendment8b.py run-arm --arm skill_memory --base-url $BASE \
      --workers 8 --variant stepaligned_k8 && break
  sleep 15
done
say "A9 STEPALIGNED FINISHED"
