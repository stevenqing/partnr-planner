#!/bin/bash
# B3 slot worker. Claims cells from queue_b.txt (mkdir claim), runs each with run_cell_b.sh on the vLLM backend.
# A slot = habitat processes on <habitat_gpu>, LLM calls to the vLLM endpoint at <llm_port>, one cell at a time.
# Several slots may share a GPU or an endpoint. Every cell uses the same NP (fixed at queue creation, $O/NP) so episode-to-process
# assignment is identical across conditions. Cell name: s<seed>_<cond>_<fold>; fold X is evaluated with the library
# built from the other fold. touch $O/STOP_<slot> (or $O/STOP_ALL) to stop after the current cell.
# Usage: worker_b.sh <habitat_gpu> <llm_port> <slot_name>
set -u
GPU=$1; PORT=$2; SLOT=$3
P=/mnt/pfs/devs/pn5wp/shishuqing
S=$P/partnr-planner/scripts/iclr_two_exp/B
REPO=$P/partnr-isambard-C
O=$REPO/outputs/iclr_two_exp_B
PY=/root/venvs/partnr/bin/python
NP=$(cat $O/NP)
CAP=2000; TMO=43200
mkdir -p $O/claims
log() { echo "$(date '+%F %T') $SLOT gpu$GPU $*" >> $O/queue.log; }
used_eps() { ls $O/s?_*/hydra/results/*.json.gz/stats/*.json 2>/dev/null | wc -l; }
run_until_complete() {  # $1 = cell name s<seed>_<cond>_<fold>
  local name=$1 seed cond fold other n
  IFS=_ read -r seed cond fold <<< "$name"; seed=${seed#s}
  other=B; [ $fold = B ] && other=A
  n=$(grep -c . $P/partnr-planner/results/iclr_two_exp_2026-09-22/B/fold${fold}_ids.txt)
  for try in 1 2 3; do
    st=$($PY $S/../../iclr_master/C/cell_status.py $O/$name $n 2>/dev/null)
    echo "$st" | grep -q '"complete": true' && { log "complete $name $st"; return 0; }
    [ $(( $(used_eps) + n )) -gt $((CAP + 400)) ] && { log "CAP reached, not starting $name"; exit 3; }
    curl -s -m 5 -o /dev/null http://127.0.0.1:$PORT/health || { log "endpoint $PORT down, not starting $name"; sleep 300; continue; }
    log "start $name try $try NP=$NP port=$PORT"
    BACKEND=vllm ICLR_VLLM_URL=http://127.0.0.1:$PORT/v1 ICLR_VLLM_MODEL=llama31-8b \
      bash $S/run_cell_b.sh $REPO $O/$name $GPU $NP task_classification_datasets/hr_fold${fold}.json.gz $cond $seed \
      data/hierarchical_skill_memory_iclr2exp/lib_${other}/hierarchical_heterogeneous_rerange/ $TMO
    log "end $name try $try $($PY $S/../../iclr_master/C/cell_status.py $O/$name $n)"
  done
}
lib_ready() {  # $1 = cell name; the library that evaluates this fold must be installed (after its sources check)
  local f=${1##*_} o=B; [ $f = B ] && o=A
  [ -f $REPO/data/hierarchical_skill_memory_iclr2exp/lib_$o/hierarchical_heterogeneous_rerange/L_coop_skills.json.gz ]
}
# Scan the queue in order; claim the first unclaimed cell whose library is installed; rescan until nothing is left.
while true; do
  { [ -f $O/STOP_$SLOT ] || [ -f $O/STOP_ALL ]; } && { log "STOP file, worker exits"; exit 0; }
  pending=0; picked=""
  while read -r name; do
    [ -z "$name" ] && continue
    [ -d $O/claims/$name ] && continue
    pending=$((pending + 1))
    lib_ready $name || continue
    mkdir $O/claims/$name 2>/dev/null || continue
    picked=$name; break
  done < $S/queue_b.txt
  [ -z "$picked" ] && [ $pending -eq 0 ] && break
  if [ -z "$picked" ]; then sleep 120; continue; fi
  echo "$SLOT gpu$GPU" > $O/claims/$picked/owner
  run_until_complete $picked
done
log "worker finished"
