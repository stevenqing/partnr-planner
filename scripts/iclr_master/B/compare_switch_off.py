"""Switch-off check (spec 1.3): same episodes, same config, old planner_demo_seeded.py (no env_over code) vs
new one with +iclr_env_over_metrics=False. Compares, per episode, every prompt file, the planner-log action
sequence and the stats. Usage: compare_switch_off.py <old results dir> <new results dir> <ids comma-separated>"""
import json, sys, pathlib, hashlib
old, new, ids = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3].split(",")
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None
allsame = True
for i in ids:
    row = {"episode": i}
    for sub in ("stats/%s.json" % i, "planner-log/planner-log-episode_%s_0.json" % i):
        a, b = old / sub, new / sub
        if sub.startswith("stats"):
            da = json.loads(a.read_text()) if a.exists() else None
            db = json.loads(b.read_text()) if b.exists() else None
            strip = lambda d: None if d is None else ({k: v for k, v in json.loads(d["stats"]).items() if k != "runtime"} if "stats" in d else d["info"].strip().splitlines()[-1])
            row["stats_same"] = strip(da) == strip(db)
        else:
            if a.exists() and b.exists():
                la, lb = json.loads(a.read_text()), json.loads(b.read_text())
                acts = lambda L: [json.dumps(x.get("high_level_actions"), sort_keys=True) for x in (L if isinstance(L, list) else L.get("planner_infos", L.get("steps", [])))]
                row["actions_same"] = acts(la) == acts(lb)
                row["n_steps"] = (len(acts(la)), len(acts(lb)))
            else:
                row["actions_same"] = (a.exists() == b.exists())
    pa = sorted((old / "traces").rglob("*episode_%s_*" % i)); pb = sorted((new / "traces").rglob("*episode_%s_*" % i))
    row["traces_same"] = [h(x) for x in pa] == [h(x) for x in pb] and len(pa) > 0
    allsame &= all(v for k, v in row.items() if k.endswith("_same"))
    print(row)
print("ALL_IDENTICAL" if allsame else "DIFFERENT")
