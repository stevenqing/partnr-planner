import json
from pathlib import Path
rows = json.load(open("outputs/partnr_failure_signatures.json"))
dead = [r for r in rows if r["score"] == 0.0 and r["steps"] <= 3]
print("zero-step episodes:", len(dead))
print("by type:", {t: sum(1 for r in dead if r["type"] == t) for t in ["R", "R_S", "R_T", "R_S_T", "H_R"]})
print("step counts seen:", sorted({r["steps"] for r in dead}))
L = Path("outputs/sweep_remeasured/val_mini/v2_memory_R/results/val_mini.json.gz/planner-log")
for r in dead[:3]:
    name = "planner-log-episode_" + r["episode"] + "_0.json"
    d = json.loads((L / name).read_text())
    print("\n" + "=" * 70)
    print("episode", r["episode"], r["type"], "steps=", r["steps"])
    print("task:", d["task"][:200])
    for i, s in enumerate(d["steps"]):
        print("-- step", i, "--")
        print("  high_level_actions:", s.get("high_level_actions"))
        print("  is_done:", s.get("is_done"), "replan_required:", s.get("replan_required"))
        resp = {k: (v[:250] if v else v) for k, v in (s.get("responses") or {}).items()}
        print("  responses:", resp)
        print("  stats:", s.get("stats"))
