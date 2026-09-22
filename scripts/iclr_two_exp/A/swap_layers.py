#!/usr/bin/env python3
"""SPEC A2: same ordering rules and vocabulary. Take the replay-mined column memory built by the
unchanged union script, keep its Layer 1, and put in the admitted column memory's Layer 2 and 3.
Records whether the union's re-mined layers were already identical (they are mined from the
same pool by the same deterministic code)."""
import hashlib, json
from pathlib import Path
ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
SRC = ROOT / "results/iclr_two_exp_2026-09-22/A/memories_union"
DST = ROOT / "results/iclr_two_exp_2026-09-22/A/memories"
OURS = ROOT / "outputs/v3_memories"
DST.mkdir(parents=True, exist_ok=True)
canon = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()
report = []
for path in sorted(SRC.glob("memory_*.json")):
    mine = json.loads(path.read_text())
    ours = json.loads((OURS / path.name).read_text())
    row = {"memory": path.name, "layer1_ops": len(mine["layer1"]["operators"]),
           "layer1_ids": [o.get("provenance", {}).get("mined_id") for o in mine["layer1"]["operators"]],
           "admitted_layer1_ops": len(ours["layer1"]["operators"]),
           "layer2_identical_before_swap": canon(mine["layer2"]) == canon(ours["layer2"]),
           "layer3_identical_before_swap": canon(mine["layer3"]) == canon(ours["layer3"])}
    mine["layer2_union_remined"] = mine["layer2"]
    mine["layer3_union_remined"] = mine["layer3"]
    mine["layer2"], mine["layer3"] = ours["layer2"], ours["layer3"]
    mine["layers_2_3_from"] = str((OURS / path.name).relative_to(ROOT))
    (DST / path.name).write_text(json.dumps(mine, indent=1))
    row["sha256"] = hashlib.sha256((DST / path.name).read_bytes()).hexdigest()
    report.append(row)
(DST / "swap_report.json").write_text(json.dumps(report, indent=1))
for r in report:
    print(r)
