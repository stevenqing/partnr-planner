#!/usr/bin/env bash
# Re-run gate_S3 episode 25857 in the cells where it crashed, on the patched harness.
#
# The crash was `Object handbag_0 has no parent` (world_graph.py:211): both agents picked
# and placed the handbag and one graph lost the edge. That line now describes the object as
# unknown instead of raising. Because the raise ended the episode, no episode that finished
# ever reached it, so the other 59 episodes of every cell are unchanged by the patch and only
# the crashed episode needs a re-run. This is the one sanctioned cross-version splice; the
# crashed stats are kept under outputs/gate/fix25857/crashed/.
#
#   bash scripts/drivers/partnr_fix25857.sh
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
PY=/root/venvs/partnr/bin/python
OUT=outputs/gate/fix25857
say () { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$OUT/crashed"

one () {  # label target_cell_dir operators config [llm]
  local label=$1 target=$2 ops=$3 config=$4 llm=${5:-}
  if [ -n "$llm" ]; then
    LLM_MODEL=qwen2.5-vl-7b VLLM_BASE_URL=http://127.0.0.1:8061/v1 \
    CONFIG=$config OPERATORS=$ops GPU=1 PROCS=1 HARD_TIMEOUT=2400 STALL_SECONDS=1800 \
    bash scripts/drivers/partnr_gate_cell.sh gate_S3 "$label" "$OUT" "+episode_id_filter=[25857]"
  else
    CONFIG=$config OPERATORS=$ops GPU=1 PROCS=1 HARD_TIMEOUT=2400 STALL_SECONDS=1800 \
    bash scripts/drivers/partnr_gate_cell.sh gate_S3 "$label" "$OUT" "+episode_id_filter=[25857]"
  fi
  local new=$OUT/$label/results/gate_S3.json.gz/stats/25857.json
  local old=$target/results/gate_S3.json.gz/stats/25857.json
  if python3 -c "import json,sys; sys.exit(0 if 'stats' in json.load(open('$new')) else 1)" 2>/dev/null; then
    cp -n "$old" "$OUT/crashed/$label.25857.json"
    cp "$new" "$old"
    say "$label: spliced $(python3 -c "import json; print(json.loads(json.load(open('$new'))['stats'])['task_percent_complete'])")"
  else
    say "$label: still no score -- $(python3 -c "import json; print(json.load(open('$new')).get('info','missing').strip().splitlines()[-1])" 2>&1)"
  fi
}

ORACLE=baselines/skill_memory_v2_oracle_goals.yaml
VLLM=baselines/skill_memory_v2_vllm.yaml
one oracle_cand0 outputs/gate/nxt2/cand0 results/partnr_gate/nxt2/cand0.json $ORACLE &
one oracle_cand2 outputs/gate/nxt2/cand2 results/partnr_gate/nxt2/cand2.json $ORACLE &
one oracle_cand3 outputs/gate/nxt2/cand3 results/partnr_gate/nxt2/cand3.json $ORACLE &
one 7b_cand3 outputs/gate/nxt2_7b/cand3 results/partnr_gate/nxt2/cand3.json $VLLM llm &
wait

# The oracle arm's adjudication is re-read on the spliced cells; the first one is kept.
cp -n outputs/gate/nxt2/ADJUDICATION.json "$OUT/crashed/nxt2.ADJUDICATION.before.json"
for c in cand0 cand2 cand3; do
  $PY scripts/partnr_gate_compare.py --pool gate_S3 --a outputs/gate/nxt2/base \
    --b outputs/gate/nxt2/$c --key is_next_to --json outputs/gate/nxt2/$c/compare.json > /dev/null
done
$PY scripts/partnr_gate_adjudicate.py --tag nxt2 --pool gate_S3 \
  --candidates results/partnr_candidates_next_to_handwritten.json \
  --library results/partnr_operators_iir1.json > "$OUT/nxt2.adjudicate.stdout"
$PY scripts/partnr_gate_compare.py --pool gate_S3 --a outputs/gate/nxt2_7b/base \
  --b outputs/gate/nxt2_7b/cand3 --key is_next_to --json outputs/gate/nxt2_7b/cand3/compare.json > /dev/null
say "ALL DONE"
