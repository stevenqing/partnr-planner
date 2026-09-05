"""Calibrate the workbench: a known-good operator must pass, a known-bad one must fail
with an explanation that names the actual defect."""
import json, sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from pathlib import Path
from viki_induction_tools import Workbench
from our_method.skill_memory_v2.simulator import SEED

REF = json.loads(Path("results/viki_memory_experiments/amendment11/skill_memory_v2.json").read_text())
bench = Workbench("/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet",
                  "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R", SEED,
                  {"layer2": REF["layer2"], "layer3": REF["layer3"]})

print("== list_episodes ==")
episodes = bench.list_episodes(limit=5)
print(json.dumps(episodes, indent=1))

print()
print("== show_trace(0) completions ==")
shown = bench.show_trace(0, max_steps=3)
print(json.dumps({k: shown[k] for k in ("task_name", "history_len", "completions", "agents")},
                 indent=1, default=str)[:1200])

print()
print("== check_actor on the first completion ==")
first = shown["completions"][0]
print(json.dumps(bench.check_actor(0, 0, first["actor_guess"]), indent=1, default=str)[:700])

ops = REF["layer1"]["operators"]
good = next(o for o in ops if not o.get("coordinated") and o["effect"]["key"] == "pos.name")
print()
print("== a reference operator, on a DIFFERENT episode ==")
print("effect:", json.dumps(good["effect"]), " body:", json.dumps(good["body"]))
for j in (2, 4, 6):
    out = bench.run_operator(good, j)
    print("  episode %d -> bound=%s effect_holds=%s failure=%s"
          % (j, out.get("bound"), out.get("effect_holds"), out.get("failure")))

bad = json.loads(json.dumps(good))
bad["body"] = [[a[0]] + ["?apple_0" if t == "?x" else t for t in a[1:]] for a in good["body"]]
print()
print("== the same operator with a pseudo-variable body (the failure we observed) ==")
print("body:", json.dumps(bad["body"]))
report = bench.try_bind(bad, 2)
print(json.dumps({"bound": report["bound"],
                  "first_requirement": report["requirements"][0] if report["requirements"] else None},
                 indent=1, default=str)[:1100])
