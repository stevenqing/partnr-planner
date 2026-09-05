"""ceiling vs our composer, paired per episode. Both cells have oracle goals and call no
model, so any difference between them is the executor and nothing else."""
import sys, json
sys.path.insert(0, "scripts")
from pathlib import Path
from collections import defaultdict
from partnr_compositional_report import read_cell, episode_types, METRIC

BASE = Path("outputs/sweep_remeasured/val_mini")
types = episode_types("val_mini")
cells = {}
for name in ["ceiling", "v2_memory_R", "v2_retry"]:
    v = read_cell(BASE / name, "val_mini.json.gz")
    cells[name] = {k: x[METRIC] for k, x in v.items()}
    print(name, len(cells[name]), "episodes")

C, V = cells["ceiling"], cells["v2_memory_R"]
both = sorted(set(C) & set(V), key=lambda e: int(e) if e.isdigit() else 0)
print()
print("paired episodes:", len(both))

buck = defaultdict(list)
for e in both:
    buck[types.get(e, "?")].append((C[e], V[e]))

print()
print("type      n   ceiling     ours   paired-gap   total-loss")
tot = 0.0
for t in ["R", "R_S", "R_T", "R_S_T", "H_R"]:
    p = buck.get(t, [])
    if not p:
        continue
    c = sum(a for a, _ in p) / len(p)
    v = sum(b for _, b in p) / len(p)
    loss = sum(a - b for a, b in p)
    tot += loss
    print("%-8s %3d   %7.3f  %7.3f   %10.3f   %10.1f" % (t, len(p), c, v, c - v, loss))
print("%-8s %3d   %7s  %7s   %10s   %10.1f" % ("TOTAL", len(both), "", "", "", tot))

tgt = [(e, C[e], V[e], types.get(e, "?")) for e in both if C[e] >= 0.99 and V[e] < 0.99]
tgt.sort(key=lambda r: r[2])
zero = [r for r in tgt if r[2] == 0.0]
print()
print("ceiling perfect but ours not:", len(tgt), " of which ours scored exactly 0:", len(zero))
bt, bz = defaultdict(int), defaultdict(int)
for e, c, v, t in tgt:
    bt[t] += 1
for e, c, v, t in zero:
    bz[t] += 1
print()
print("type      ceiling-perfect-ours-not    ours-zero")
for t in ["R", "R_S", "R_T", "R_S_T", "H_R"]:
    if bt[t]:
        print("%-8s %22d %12d" % (t, bt[t], bz[t]))

json.dump([{"episode": e, "ceiling": c, "ours": v, "type": t} for e, c, v, t in tgt],
          open("outputs/ceiling_vs_ours_targets.json", "w"), indent=1)
print()
print("-> outputs/ceiling_vs_ours_targets.json")
