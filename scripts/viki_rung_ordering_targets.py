"""Seed the ordering rung where the memory fails to recover an ordering it should.

The family targeter picks families by *coverage* residual, and the families whose orderings
are missing are not among them -- `cut_fruit_on_board` is fully solved on the induction half,
because there the episodes' own goals carry their temporal constraints and `order_for` is
never consulted. Selecting by coverage would therefore point this rung at the wrong episodes
entirely.

Selection here is by the thing the rung is about: episodes whose `temporal_constraints` say
an ordering exists and whose memory emits none. That is training-set ground truth against
the library's own behaviour -- no model, no reference library, no test split.

Seeds and holdout come from the same family and never overlap, for the reason the
family-stratified targeter exists: an operator derived from one family's demonstration and
graded on another's is being asked an impossible question, and a zero from that measures the
seeding.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import viki_fork_guard  # noqa: E402

viki_fork_guard.install()

from viki_eval_skill_memory_v2 import visits_of  # noqa: E402
from viki_induction_tools import Workbench  # noqa: E402
from our_method.skill_memory_v2 import planner as planner_module  # noqa: E402
from our_method.skill_memory_v2.memory import FORMAT, SkillMemoryV2  # noqa: E402
from our_method.skill_memory_v2.simulator import SEED  # noqa: E402

TRAIN = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet"
BENCHMARK = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R"
REFERENCE = Path("results/viki_memory_experiments/amendment11/skill_memory_v2.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("outputs/agentic_library_runner.json"))
    parser.add_argument("--scan", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--holdout", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("outputs/rung_ordering_targets.json"))
    arguments = parser.parse_args()

    reference = json.loads(REFERENCE.read_text())
    bench = Workbench(TRAIN, BENCHMARK, SEED,
                      {"layer2": reference["layer2"], "layer3": reference["layer3"]})
    operators = json.loads(arguments.library.read_text())["operators"]
    memory = SkillMemoryV2({
        "format": FORMAT, "built_from": "ordering_targets", "excluded_family": None,
        "seed": SEED, "per_family": 0, "layer1": {"operators": operators},
        "layer2": reference["layer2"], "layer3": reference["layer3"]})

    missing, recovered = defaultdict(list), defaultdict(list)
    for index in range(min(arguments.scan, len(bench.episodes))):
        episode = bench.episodes[index]
        if not isinstance(episode, dict) or not episode.get("time_steps"):
            continue
        if not (episode.get("temporal_constraints") or []):
            continue
        blind = {k: v for k, v in episode.items() if k != "time_steps"}
        metadata = bench.sim.metadata(blind, SEED)
        env = bench.sim.world(metadata)
        requirements = [r["predicate"] for r in planner_module.collect_requirements(metadata)]
        try:
            order = memory.order_for(requirements, visits_of(env, requirements, memory))
        except Exception:
            continue
        family = str(episode.get("task_name"))
        (recovered if (order and any(len(g) > 1 for g in order)) else missing)[family].append(index)

    targets = {}
    for family, episodes in sorted(missing.items(), key=lambda kv: -len(kv[1])):
        holdout = episodes[-arguments.holdout:]
        seeds = [e for e in episodes if e not in holdout][: arguments.seeds]
        targets[family] = {
            "orderings_missing": len(episodes),
            "orderings_recovered": len(recovered.get(family, [])),
            "seeds": seeds, "holdout": holdout,
            "target_key": "is_activated",
            "usable": len(seeds) >= 1 and len(holdout) >= 2 and not (set(seeds) & set(holdout)),
        }

    payload = {"library": str(arguments.library),
               "scanned": min(arguments.scan, len(bench.episodes)),
               "ordering_score": bench.ordering_score(operators),
               "targets": targets}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=1))

    print("library %s" % arguments.library)
    print("ordering score: %s\n" % payload["ordering_score"])
    print("%-40s %9s %10s %-20s %-20s %s"
          % ("family", "missing", "recovered", "seeds", "holdout", "usable"))
    for family, row in targets.items():
        print("%-40s %9d %10d %-20s %-20s %s"
              % (family[:40], row["orderings_missing"], row["orderings_recovered"],
                 row["seeds"], row["holdout"], row["usable"]))
    print("\n-> %s" % arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
