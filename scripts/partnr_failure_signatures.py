"""Why does the open-loop composer score zero where the closed-loop arm does not?

Reads our own planner logs and buckets each episode by the error the agents kept getting.
`ceiling` is deliberately not the reference here: it is centralized and fully observed, so
the gap to it is the cost of decentralization and partial observation, not of the executor.
The reference is v2_prompt, which is the same memory under the same conditions.
"""
import sys, json, re
sys.path.insert(0, "scripts")
from pathlib import Path
from collections import defaultdict, Counter
from partnr_compositional_report import read_cell, episode_types, METRIC

BASE = Path("outputs/sweep_remeasured/val_mini")
CELL = BASE / "v2_memory_R"
LOGS = CELL / "results/val_mini.json.gz/planner-log"
types = episode_types("val_mini")
ours = {k: v[METRIC] for k, v in read_cell(CELL, "val_mini.json.gz").items()}

SIGS = [
    ("unknown-entity", re.compile(r"not present in the graph")),
    ("wrong-room", re.compile(r"not in the same room|different room")),
    ("occupied-hand", re.compile(r"already holding|hand is full|not empty")),
    ("unreachable", re.compile(r"could not (?:find|reach)|failed to navigate|no path")),
    ("closed-container", re.compile(r"is closed|need to open")),
    ("grasp-fail", re.compile(r"failed to pick|could not pick|not close enough")),
]

rows = []
for path in sorted(LOGS.glob("planner-log-episode_*_0.json")):
    eid = path.name.split("_")[1]
    if eid not in ours:
        continue
    d = json.loads(path.read_text())
    steps = d.get("steps", [])
    counts = Counter()
    for s in steps:
        for r in (s.get("responses") or {}).values():
            if not r:
                continue
            for name, rx in SIGS:
                if rx.search(r):
                    counts[name] += 1
    last = steps[-1] if steps else {}
    rows.append({
        "episode": eid,
        "type": types.get(eid, "?"),
        "score": ours[eid],
        "steps": last.get("total_step_count", len(steps)),
        "sim_steps": last.get("sim_step_count", 0),
        "errs": dict(counts),
        "top": counts.most_common(1)[0][0] if counts else "none",
    })

zero = [r for r in rows if r["score"] == 0.0]
part = [r for r in rows if 0.0 < r["score"] < 0.99]
full = [r for r in rows if r["score"] >= 0.99]
print("episodes: %d   zero: %d   partial: %d   full: %d" % (len(rows), len(zero), len(part), len(full)))

def table(label, group):
    print()
    print("== %s (n=%d): dominant error signature ==" % (label, len(group)))
    c = Counter(r["top"] for r in group)
    for name, n in c.most_common():
        print("  %-18s %4d  (%4.1f%%)" % (name, n, 100.0 * n / max(len(group), 1)))

table("score 0", zero)
table("score 1", full)

print()
print("== unknown-entity 的出现率（按分数分层）==")
for label, group in [("zero", zero), ("partial", part), ("full", full)]:
    hit = sum(1 for r in group if r["errs"].get("unknown-entity"))
    tot = sum(r["errs"].get("unknown-entity", 0) for r in group)
    print("  %-8s %3d/%3d episodes 命中 (%4.1f%%)   累计报错 %6d 次" %
          (label, hit, len(group), 100.0 * hit / max(len(group), 1), tot))

print()
print("== 步数预算 ==")
for label, group in [("zero", zero), ("partial", part), ("full", full)]:
    if not group:
        continue
    st = sorted(r["steps"] for r in group)
    mx = sum(1 for r in group if r["steps"] >= 1500)
    print("  %-8s 中位步数 %5d   >=1500 步(疑似耗尽预算) %3d/%3d" %
          (label, st[len(st) // 2], mx, len(group)))

print()
print("== zero 组按任务类型 x 主导错误 ==")
grid = defaultdict(Counter)
for r in zero:
    grid[r["type"]][r["top"]] += 1
for t in ["R", "R_S", "R_T", "R_S_T", "H_R"]:
    if t in grid:
        print("  %-6s %s" % (t, dict(grid[t])))

json.dump(rows, open("outputs/partnr_failure_signatures.json", "w"), indent=1)
print()
print("-> outputs/partnr_failure_signatures.json")
