#!/bin/bash
# Unattended continuation of Amendment 9.
#
# Two renderer variants exist only in distribution: fullactions_k8 (the five-action
# cap removed) and stepaligned_k8 (the flat action list replaced by the real
# per-step, per-robot structure recovered from the training parquet). Neither has
# been run on the held-out-family folds, and the folds are where a coordination
# representation is supposed to matter, so both are taken there tonight.
#
# Every stage is resumable: run-arm appends only the rows a results file is missing
# and refuses to continue if the recorded metadata differs, so re-running this
# script costs nothing for work already done. A stage that fails is logged and the
# next one still runs -- one bad stage should not waste the remaining hours.
cd "$(dirname "$0")/../.." || exit 1
PY=/root/venvs/partnr/bin/python
export TOKENIZERS_PARALLELISM=false
LOG=results/viki_memory_experiments/amendment8b/overnight
mkdir -p "$LOG"

say(){ echo "===== $(date '+%m-%d %H:%M:%S') $* ====="; }
stage(){
  local name="$1"; shift
  say "BEGIN $name"
  if "$@" > "$LOG/$name.log" 2>&1; then
    say "OK $name"
  else
    say "FAILED $name (exit $?) -- continuing; see $LOG/$name.log"
  fi
}

say "overnight job starting"

# 1. Finish the in-distribution step-aligned run if it is not already complete.
#    A concurrent copy may still be going; wait it out rather than racing it.
while pgrep -f a9_stepaligned.sh > /dev/null; do sleep 60; done
stage 01-stepaligned-id bash scripts/drivers/a9_stepaligned.sh

# 2. The mechanism gate again, on the record, before spending fold inference.
stage 02-stepaligned-gate env A8B_SKILL_TOPK=8 A9_ACTION_CAP=0 A9_STEP_ALIGNED=1 \
     $PY scripts/viki_amendment9_stepalign_gate.py

# 3. Both renderer variants over the eight held-out-family folds.
stage 03-folds-stepaligned bash scripts/drivers/a9_folds.sh stepaligned_k8
stage 04-folds-fullactions bash scripts/drivers/a9_folds.sh fullactions_k8

# 5. Reports. Scored on both metrics; the tolerant one is the one to read.
stage 05-report-id $PY scripts/viki_amendment9_report.py
stage 06-report-id-mcnemar $PY scripts/viki_amendment9_id_tolerant_mcnemar.py
stage 07-report-folds $PY scripts/viki_amendment9_fold_tolerant.py \
     queryfix_k8 grounded_k8 fullactions_k8 stepaligned_k8
stage 08-report-fullactions $PY scripts/viki_amendment9_fullactions_report.py
stage 09-family-split $PY scripts/viki_amendment9_family_split.py
stage 10-ood-robustness $PY scripts/viki_amendment9_ood_robustness.py

say "A9 OVERNIGHT FINISHED"
