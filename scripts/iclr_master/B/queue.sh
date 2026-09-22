#!/bin/bash
# One lane of the B.2 queue. Usage: queue.sh <gpu> <per_episode_s> [max num_proc=4] [backbone filter, e.g. llama8b] [min free MiB=auto]
# Walks cells.tsv in order; a cell is claimed atomically (mkdir .claim) so two lanes never take the same cell.
# Per cell: skip if exit.txt says done; else run_cell.sh (always resume); retries (resume) when not done; env_over episodes do not make a cell not-done; then move on.
# Processes per cell = min(<max num_proc>, the backbone's np in np_per_backbone.tsv); that table is re-read at every
# cell start, so lowering a backbone's np there takes effect from the lane's next cell.
# Waits (does not start) while the GPU has less free memory than np * peak_mib + 8192 (or the explicit 5th arg).
# Hard timeout = ceil(n/np) * per_episode_s * 2 + 1800.
# 09-22: ICLR_SHARE_LLM=True (one HF model per process instead of one per agent; verified byte-identical traces,
# B_verify/share_llm/VERDICT.txt) is exported to run_cell.sh unless the caller sets it.
set -u
GPU=$1 PER_EP=$2 NPMAX=${3:-4} ONLY=${4:-} MINFREE_ARG=${5:-}
export ICLR_SHARE_LLM=${ICLR_SHARE_LLM:-True}
PFS=/mnt/pfs/devs/pn5wp/shishuqing
HERE=$PFS/partnr-planner/scripts/iclr_master/B
O=$PFS/partnr-isambard/outputs/iclr_master/B
mkdir -p $O
[ -e $O/STOP ] && { echo "STOP file present"; exit 0; }
echo "$(date -Is) lane $GPU up: max np $NPMAX, ICLR_SHARE_LLM=$ICLR_SHARE_LLM" >> $O/lane_$GPU.log
grep -v '^#' $HERE/cells.tsv | while IFS=$'\t' read -r CELL BB KEY TGT SEED N; do
  [ -e $O/STOP ] && { echo "$(date -Is) STOP file, lane $GPU exits"; break; }
  [ -n "$ONLY" ] && [ "$BB" != "$ONLY" ] && continue
  mkdir -p $O/$CELL
  grep -q '^status: done' $O/$CELL/exit.txt 2>/dev/null && continue
  mkdir $O/$CELL/.claim 2>/dev/null || continue
  echo "lane $GPU" > $O/$CELL/.claim/owner
  read -r BNP PEAK < <(awk -v b="$BB" '$1 == b {print $2, $3}' $HERE/np_per_backbone.tsv)
  NP=${BNP:-1}; [ "$NPMAX" -lt "$NP" ] && NP=$NPMAX
  MINFREE=${MINFREE_ARG:-$(( NP * ${PEAK:-50000} + 8192 ))}
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
    [ "$free" -ge "$MINFREE" ] && break
    echo "$(date -Is) lane $GPU waits: GPU $GPU free ${free} MiB < $MINFREE for $CELL np $NP" >> $O/lane_$GPU.log; sleep 300
  done
  TO=$(( (N + NP - 1) / NP * PER_EP * 2 + 1800 ))
  for attempt in 1 2 3; do
    echo "$(date -Is) lane $GPU start $CELL attempt $attempt np $NP timeout $TO" >> $O/lane_$GPU.log
    bash $HERE/run_cell.sh $CELL $BB $KEY $TGT $SEED $GPU $TO $NP > /dev/null 2>&1 < /dev/null
    st=$(grep '^status:' $O/$CELL/exit.txt 2>/dev/null)
    echo "$(date -Is) lane $GPU end $CELL attempt $attempt $st $(grep -E '^(episodes_with_stats|env_over|error):' $O/$CELL/exit.txt | tr '\n' ' ')" >> $O/lane_$GPU.log
    [ "$st" = "status: done" ] && break
    grep -q '^no weights' $O/$CELL/exit.txt && break
  done
  grep -q '^status: done' $O/$CELL/exit.txt || echo failed > $O/$CELL/FAILED
done
echo "$(date -Is) lane $GPU finished queue" >> $O/lane_$GPU.log
