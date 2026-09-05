"""How many episodes does each cell abandon at step 0, and are they the same episodes?

If the all-types operator library abandons fewer, the cause is operator coverage. If every
cell abandons the same set, the cause is upstream of the library.
"""
import sys, json
sys.path.insert(0, "scripts")
from pathlib import Path
from partnr_compositional_report import read_cell, episode_types, METRIC

BASE = Path("outputs/sweep_remeasured/val_mini")
types = episode_types("val_mini")
sets = {}
for name in ["ceiling", "v2_memory_R", "v2_memory_all", "v2_R_nofold", "v2_R_noorder", "v2_retry"]:
    cell = BASE / name
    logs = cell / "results/val_mini.json.gz/planner-log"
    if not logs.is_dir():
        print("%-16s no planner-log" % name)
        continue
    scores = {k: v[METRIC] for k, v in read_cell(cell, "val_mini.json.gz").items()}
    dead = set()
    for p in logs.glob("planner-log-episode_*_0.json"):
        eid = p.name.split("_")[1]
        if eid not in scores:
            continue
        d = json.loads(p.read_text())
        steps = d.get("steps", [])
        if len(steps) == 1 and all(a[0] == "Done" for a in (steps[0].get("high_level_actions") or {}).values()):
            dead.add(eid)
    sets[name] = dead
    bt = {t: sum(1 for e in dead if types.get(e) == t) for t in ["R", "R_S", "R_T", "R_S_T", "H_R"]}
    print("%-16s dead=%3d/%3d  mean=%.4f  %s" % (name, len(dead), len(scores),
          sum(scores.values()) / len(scores), bt))

print()
ours = sets.get("v2_memory_R", set())
for name, s in sets.items():
    if name == "v2_memory_R":
        continue
    print("%-16s 与 v2_memory_R 的死集: 交 %3d  仅它 %3d  仅我们 %3d" %
          (name, len(s & ours), len(s - ours), len(ours - s)))

json.dump({k: sorted(v, key=int) for k, v in sets.items()},
          open("outputs/partnr_dead_sets.json", "w"), indent=1)
print("\n-> outputs/partnr_dead_sets.json")
