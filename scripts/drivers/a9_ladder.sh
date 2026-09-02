#!/bin/bash
# Amendment 9 ladder, second run. Lives in the repo: the first version was written
# through a single-quoted ssh argument and its embedded quotes corrupted it.
cd "$(dirname "$0")/../.." || exit 1
PY=/root/venvs/partnr/bin/python
BASE=http://192.168.32.40:8050/v1
D=results/viki_memory_experiments/amendment8b
export TOKENIZERS_PARALLELISM=false
say(){ echo "===== $(date "+%m-%d %H:%M:%S") $* ====="; }

run_rung(){   # $1 rung  $2 variant tag
  local rung=$1 tag=$2 prog="$D/skill_memory.$tag.jsonl"
  say "RUNG $rung (variant=$tag) MODE=$A9_MODE ROLE=$A9_ROLE_AWARE SLOTS=$A9_PATTERN_SLOTS TOPK=$A8B_SKILL_TOPK"
  say "  gate check"
  if ! $PY scripts/viki_amendment9_gate.py "$rung" >/dev/null 2>"/root/gate-$rung.err"; then
    say "  GATE FAILED for $rung -- skipping"
    $PY -c "
import json,sys
d=json.load(open('$D/gate_$rung.json'))
for k in ('partner_in_query','rows_with_a_pattern_skill','rows_with_role_text_in_prompt','grounded_rows','memory_chars_median'):
    print('   ',k,'=',d.get(k))
print('    gates',d['gates'])" 2>/dev/null
    tail -3 "/root/gate-$rung.err" 2>/dev/null
    return 1
  fi
  $PY -c "
import json
d=json.load(open('$D/gate_$rung.json'))
for k in ('partner_in_query','rows_with_a_pattern_skill','rows_with_role_text_in_prompt','grounded_rows','memory_chars_median'):
    print('   ',k,'=',d.get(k))" 2>/dev/null
  say "  gate passed"

  for r in 1 2 3; do
    local before=$([ -f "$prog" ] && wc -l < "$prog" || echo 0)
    say "  run round $r ($before/924)"
    $PY scripts/viki_amendment8b.py run-arm --arm skill_memory --base-url $BASE \
        --workers 8 --variant "$tag" && { say "  RUNG $rung DONE"; return 0; }
    local after=$([ -f "$prog" ] && wc -l < "$prog" || echo 0)
    [ "$after" = "$before" ] && [ "$r" -gt 1 ] && { say "  no progress, stopping"; return 1; }
    sleep 15
  done
  return 1
}

export A8B_SKILL_TOPK=8

say "RUNG patternslot: reserve prompt slots for coordination patterns (+ C1)"
export A9_MODE=""; export A9_ROLE_AWARE=1; export A9_PATTERN_SLOTS=2
run_rung patternslot patternslot_k8

say "RUNG grounded: attach one concrete episode (C2)"
export A9_MODE=grounded; export A9_GROUND_TOPK=3
run_rung grounded grounded_k8

say "RUNG rescore: + LLM relevance rescoring (C3)"
export A9_MODE=rescore
run_rung rescore rescore_k8

say "LADDER REPORT"
$PY scripts/viki_amendment9_report.py 2>/dev/null
say "A9 LADDER2 FINISHED"
