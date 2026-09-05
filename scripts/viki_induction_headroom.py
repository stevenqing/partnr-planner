"""Where does VIKI-L2's replay induction give up, and how often?

Mirrors `induce()` branch for branch, counting instead of building, using the module's own
`replay`, `_bind`, `_segment_start` and `_runs_alone` so the numbers describe the shipped
inducer. `induction.py` is not modified.
"""
import argparse, json, time, sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from collections import Counter
from pathlib import Path

from our_method.skill_memory_v2 import induction
from our_method.skill_memory_v2.build import load_episodes
from our_method.skill_memory_v2.simulator import SEED, Simulator

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=Path,
                    default=Path("/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet"))
parser.add_argument("--benchmark-root", type=Path,
                    default=Path("/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R"))
parser.add_argument("--limit", type=int, default=0, help="0 = the whole induction half")
parser.add_argument("--per-family", type=int, default=250)
parser.add_argument("--out", type=Path, default=Path("outputs/viki_induction_headroom.json"))
args = parser.parse_args()

sim = Simulator(args.benchmark_root)
episodes = load_episodes(args.train)
induction_set = episodes[::2]          # exactly what build.py induces from
if args.limit:
    induction_set = induction_set[: args.limit]
print("induction episodes:", len(induction_set))

outcomes = Counter()
seen_family = Counter()
completions_total = 0
give_up = Counter()
kinds = Counter()
effects_seen = Counter()
effects_lost = Counter()
repair_candidates = 0
start = time.time()

for position, truth in enumerate(induction_set):
    if not isinstance(truth, dict) or not truth.get("time_steps"):
        outcomes["NO_PLAN"] += 1
        continue
    family = truth.get("task_name", "?")
    if seen_family[family] >= args.per_family:
        continue
    seen_family[family] += 1
    trace, status = induction.replay(truth, sim, SEED)
    outcomes[status] += 1
    if trace is None:
        continue

    for index, step in enumerate(trace["history"]):
        for robot, action in step["actions"].items():
            if action[0] == "Open" and len(action) >= 2:
                repair_candidates += 1

    for index, actor, predicate in trace["completions"]:
        completions_total += 1
        status_map = induction.predicate_status(predicate) or {}
        shape = ("pos.name" if "pos.name" in status_map
                 else "is_activated" if status_map.get("is_activated") is True
                 else "|".join(sorted(status_map.keys())) or "empty")
        effects_seen[shape] += 1
        if actor is None:
            give_up["completion: no actor attributed"] += 1
            effects_lost[shape] += 1
            continue
        mapping, effect = induction._bind(predicate)
        if effect is None:
            give_up["completion: effect shape not in _bind"] += 1
            effects_lost[shape] += 1
            continue
        seg = induction._segment_start(trace["completions"], index, actor)
        state = trace["states"][seg]
        alone = induction._runs_alone(state, trace["history"], seg, index, actor, predicate, sim)
        body = [list(trace["history"][s]["actions"][actor])
                for s in range(seg, index + 1) if actor in trace["history"][s]["actions"]]
        if alone and not body:
            give_up["completion: empty body"] += 1
            effects_lost[shape] += 1
            continue
        kinds["achievement" if alone else "coordination"] += 1

    if (position + 1) % 50 == 0:
        rate = (time.time() - start) / (position + 1)
        print("  %d episodes, %.2f s/episode, eta %.1f min for the full half"
              % (position + 1, rate, rate * len(episodes[::2]) / 60.0), flush=True)

elapsed = time.time() - start
kept = sum(kinds.values())
print()
print("elapsed %.1f s  (%.2f s/episode)" % (elapsed, elapsed / max(len(induction_set), 1)))
print("replay outcomes:", dict(outcomes))
print()
print("completions found       %6d" % completions_total)
print("  induced               %6d  (%.1f%%)" % (kept, 100.0 * kept / max(completions_total, 1)))
print("  dropped               %6d  (%.1f%%)" % (completions_total - kept,
      100.0 * (completions_total - kept) / max(completions_total, 1)))
print("  kinds:", dict(kinds))
print("  repair candidates (Open actions seen): %d" % repair_candidates)
print()
print("== where it gives up ==")
for k, n in give_up.most_common():
    print("  %-42s %6d" % (k, n))
print()
print("== completions by predicate shape: seen vs lost ==")
print("  %-34s %8s %8s %7s" % ("status shape", "seen", "lost", "lost%"))
for shape, n in effects_seen.most_common(12):
    l = effects_lost.get(shape, 0)
    print("  %-34s %8d %8d %6.1f%%" % (shape[:34], n, l, 100.0 * l / max(n, 1)))

args.out.parent.mkdir(parents=True, exist_ok=True)
json.dump({"episodes": len(induction_set), "elapsed_sec": elapsed,
           "replay_outcomes": dict(outcomes), "completions": completions_total,
           "induced": kept, "kinds": dict(kinds), "give_up": dict(give_up),
           "effects_seen": dict(effects_seen), "effects_lost": dict(effects_lost),
           "repair_candidates": repair_candidates}, open(args.out, "w"), indent=1)
print("\n->", args.out)
