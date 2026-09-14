#!/usr/bin/env bash
# Scripted verification of the Achieve tool on val_mini, no model (baselines/achieve_scripted.yaml).
#
# The reference is outputs/report/priv_iir1/accepted: the same privileged requirements, split
# and ordering, and the same 22-operator library, with the grounded chain executed by the
# planner. Here every requirement is one Achieve call, so an episode that scores lower is the
# tool's execution path (resolve / explore / ground / sub-skills / shut retry / graph report).
#
#   bash scripts/drivers/partnr_achieve_scripted.sh pilot   # 16 episodes, every 23rd id
#   bash scripts/drivers/partnr_achieve_scripted.sh full    # all 369
#
# GPU 5, which is shared with another user's process (~16 G): refuses above 60 G.
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
MODE=${1:?pilot or full}
PY=/root/venvs/partnr/bin/python
OUT=outputs/achieve_scripted_0914
REF=outputs/report/priv_iir1/accepted
LIB=results/partnr_operators_iir1.json
GPU=5
say () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$used" -lt 60000 ] || { say "REFUSING: GPU $GPU at ${used} MiB"; exit 5; }
mkdir -p "$OUT"

case $MODE in
  pilot)
    ids=$("$PY" - "$REF/results/val_mini.json.gz/stats" <<'PY'
import glob, os, sys
ids = sorted(int(os.path.basename(f)[:-5]) for f in glob.glob(sys.argv[1] + "/*.json"))
print(",".join(str(i) for i in ids[::23]))
PY
)
    say "pilot episodes [$ids]"
    CONFIG=baselines/achieve_scripted.yaml OPERATORS=$LIB GPU=$GPU PROCS=16 \
      HARD_TIMEOUT=3600 STALL_SECONDS=1200 \
      bash scripts/drivers/partnr_gate_cell.sh val_mini pilot "$OUT" "+episode_id_filter=[$ids]"
    cell=pilot ;;
  full)
    CONFIG=baselines/achieve_scripted.yaml OPERATORS=$LIB GPU=$GPU PROCS=24 \
      HARD_TIMEOUT=21600 STALL_SECONDS=1800 \
      bash scripts/drivers/partnr_gate_cell.sh val_mini full "$OUT"
    cell=full ;;
  *) say "unknown mode $MODE"; exit 1 ;;
esac

# Written to disk before anything is printed.
"$PY" scripts/partnr_gate_compare.py --pool val_mini --a "$REF" --b "$OUT/$cell" \
    --json "$OUT/compare_$cell.json" > "$OUT/compare_$cell.txt" 2>&1
"$PY" - "$OUT/$cell" "$REF" > "$OUT/tool_$cell.txt" 2>&1 <<'PY'
import collections, glob, json, os, re, sys
cell, ref = sys.argv[1], sys.argv[2]
def score(root, ep):
    try:
        d = json.load(open(f"{root}/results/val_mini.json.gz/stats/{ep}.json"))
        st = d.get("stats"); st = json.loads(st) if isinstance(st, str) else st
        return None if st is None else st.get("task_percent_complete")
    except FileNotFoundError:
        return "missing"
calls = collections.Counter(); outcomes = collections.Counter(); refusals = collections.Counter()
rows = []
for f in sorted(glob.glob(f"{cell}/results/val_mini.json.gz/planner-log/*.json")):
    ep = re.search(r"episode_(\d+)_", f).group(1)
    log = json.load(open(f))
    n_calls = 0
    for step in log.get("steps", []):
        for uid, action in (step.get("high_level_actions") or {}).items():
            if action and action[0] == "Achieve" and step.get("replanned", {}).get(uid):
                n_calls += 1
        for uid, text in (step.get("responses") or {}).items():
            text = str(text or "")
            if text.startswith("Successful execution!") and " via " in text:
                outcomes["achieve success"] += 1
            elif text.startswith("Achieve") or text.startswith("No stored skill") or "is not a known" in text or "was not found" in text:
                outcomes["achieve refused"] += 1
                refusals[re.sub(r"[a-z_]+_\d+", "<x>", text)[:110]] += 1
    calls[ep] = n_calls
    rows.append((int(ep), score(ref, ep), score(cell, ep), n_calls))
print("episodes with a planner log:", len(rows), " Achieve issued:", sum(calls.values()),
      " episodes with zero Achieve:", sum(1 for r in rows if r[3] == 0))
print("responses:", dict(outcomes))
print("top refusals:"); [print(f"  {n:4d}  {t}") for t, n in refusals.most_common(12)]
print("\nep  ref(priv chain)  achieve  calls")
for ep, a, b, n in sorted(rows):
    flag = "  <-- lower" if isinstance(a, float) and isinstance(b, float) and b < a - 1e-9 else ""
    print(f"{ep:4d}  {a!s:>8}  {b!s:>8}  {n:4d}{flag}")
PY
say "compare -> $OUT/compare_$cell.txt, tool counts -> $OUT/tool_$cell.txt"
