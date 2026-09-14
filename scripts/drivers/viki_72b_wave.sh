#!/usr/bin/env bash
# Every 72B job of the VIKI ICLR 2027 supplement, launched in parallel against one local 72B
# endpoint once it serves qwen2.5-vl-72b-amendment3-f2:
#   Figure 2 ToM         id / ood_single_family / cg_image / pure_text  (4 launchers)
#   RQ3 no_grounding     ood_single_family (id / cg_image / pure_text reuse v3_ablfull rows)
#   RQ3 no_order         ood_single_family
#   RQ2 no_execution_admission   4 splits
#   RQ2 no_trace induction       round1 (14 family drivers at once) -> targets -> round2 -> build
# The no_trace downstream cells need the build's libraries and are launched separately.
#
#   BASE=http://127.0.0.1:8050/v1 setsid nohup bash scripts/drivers/viki_72b_wave.sh \
#     > outputs/paper_viki_0914/wave_72b.log 2>&1 < /dev/null &
set -u
ROOT=${ROOT:-/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner}
BASE=${BASE:-http://127.0.0.1:8050/v1}
L=${L:-outputs/paper_viki_0914}
cd "$ROOT" || exit 1
mkdir -p "$L"
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

for _ in $(seq 1 360); do
  curl -s -m 5 "$BASE/models" | grep -q '"qwen2.5-vl-72b-amendment3-f2"' && break
  sleep 10
done
curl -s -m 5 "$BASE/models" | grep -q '"qwen2.5-vl-72b-amendment3-f2"' || { say "72B never came up at $BASE"; exit 1; }
say "72B up at $BASE"

for s in id ood_single_family cg_image pure_text; do
  MODEL=72B SPLITS=$s BASE=$BASE WORKERS=8 setsid nohup bash scripts/drivers/viki_tom_split.sh \
    > "$L/tom_72b_$s.log" 2>&1 < /dev/null &
  say "tom 72B $s pid $!"
done
for c in no_grounding no_order; do
  EXPERIMENT=rq3 MODEL=72B CONDITION=$c SPLITS=ood_single_family BASE_URL=$BASE WORKERS=8 \
    setsid nohup bash scripts/drivers/viki_replay_cells.sh > "$L/rq3_${c}_72b.log" 2>&1 < /dev/null &
  say "rq3 72B $c pid $!"
done
EXPERIMENT=rq2 MODEL=72B CONDITION=no_execution_admission BASE_URL=$BASE WORKERS=8 \
  setsid nohup bash scripts/drivers/viki_replay_cells.sh > "$L/rq2_noexec_72b.log" 2>&1 < /dev/null &
say "rq2 72B no_execution_admission pid $!"
URL=$BASE R1_FAMILY_WORKERS=14 WORKERS=16 setsid nohup bash scripts/drivers/viki_rq2_no_trace.sh all \
  > "$L/rq2_notrace_induction.log" 2>&1 < /dev/null &
say "rq2 no_trace induction pid $!"
say "launched-72B"
