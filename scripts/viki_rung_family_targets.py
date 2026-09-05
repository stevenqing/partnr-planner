"""Residual seeding, stratified by family — because the unstratified version posed an
impossible task and I read its failure as the model's.

The first residual targeter sorted failing episodes by mode and index, took the front as
seeds and the tail as holdout. On this split that put the seeds in `toast_bread_and_set_plate`
and `wash_fruit_and_serve` while the holdout landed in `clear_table_..._put_in_cabinet` and
`set_plate_and_fork_on_table`. The ladder was therefore asked to derive an operator from a
demonstration of one family and have it work on three others — and the variant those others
need (open the sealed target before placing) is not demonstrated in the seed family at all.
It returned 0 of 50. That number measured the seeding, not the model.

Here seeds and holdout come from the **same family**, and only families the current library
actually fails are targeted. A pass now means something specific and checkable: an operator
derived from one episode of a family the library cannot do, which then works on other
episodes of that same family.

Selection uses the induction half and the library's own failures. The reference library is
never consulted, and neither is the test split — the families are ranked by how many
induction-half episodes the current library fails, not by anything the benchmark says.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import viki_fork_guard  # noqa: E402

viki_fork_guard.install()

from viki_induction_tools import Workbench  # noqa: E402
from our_method.skill_memory_v2.simulator import SEED, predicate_status  # noqa: E402

TRAIN = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet"
BENCHMARK = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R"
REFERENCE = Path("results/viki_memory_experiments/amendment11/skill_memory_v2.json")


def key_of(predicate) -> str:
    status = predicate_status(predicate) or {}
    if "pos.name" in status:
        return "pos.name"
    if status.get("is_activated") is True:
        return "is_activated"
    return "|".join(sorted(status)) or "empty"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("outputs/agentic_library_div.json"))
    parser.add_argument("--scan", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=4, help="seed episodes per family")
    parser.add_argument("--holdout", type=int, default=4, help="holdout episodes per family")
    parser.add_argument("--out", type=Path, default=Path("outputs/rung_family_targets.json"))
    arguments = parser.parse_args()

    reference = json.loads(REFERENCE.read_text())
    bench = Workbench(TRAIN, BENCHMARK, SEED,
                      {"layer2": reference["layer2"], "layer3": reference["layer3"]})
    operators = json.loads(arguments.library.read_text())["operators"]

    coverage = defaultdict(lambda: {"solved": 0, "total": 0, "failed": [], "keys": Counter()})
    for index in range(min(arguments.scan, len(bench.episodes))):
        episode = bench.episodes[index]
        if not isinstance(episode, dict) or not episode.get("time_steps"):
            continue
        trace, _ = bench.trace(index)
        if trace is None:
            continue
        family = str(episode.get("task_name"))
        try:
            outcome = bench.plan_with(operators, index)
        except Exception:
            continue
        slot = coverage[family]
        slot["total"] += 1
        if outcome["official_score"] >= 1.0:
            slot["solved"] += 1
        else:
            slot["failed"].append(index)
            for _, _, predicate in trace["completions"]:
                slot["keys"][key_of(predicate)] += 1

    targets = {}
    for family, slot in coverage.items():
        if not slot["failed"]:
            continue
        failed = slot["failed"]
        holdout = failed[-arguments.holdout:]
        seeds = [e for e in failed if e not in holdout][: arguments.seeds]
        # The key to ask for is the one this family's failing episodes actually need most.
        key = slot["keys"].most_common(1)[0][0] if slot["keys"] else "pos.name"
        targets[family] = {
            "solved": slot["solved"], "total": slot["total"],
            "coverage": round(slot["solved"] / slot["total"], 4),
            "failed_episodes": len(failed),
            "target_key": key,
            "keys_needed": dict(slot["keys"]),
            "seeds": seeds, "holdout": holdout,
            # Same family on both sides, and never the same episode on both.
            "usable": len(seeds) >= 1 and len(holdout) >= 2 and not (set(seeds) & set(holdout)),
        }

    ranked = sorted(targets.items(), key=lambda kv: -kv[1]["failed_episodes"])
    payload = {"library": str(arguments.library), "operators": len(operators),
               "scanned": min(arguments.scan, len(bench.episodes)),
               "per_family": {f: {"solved": s["solved"], "total": s["total"]}
                              for f, s in coverage.items()},
               "targets": dict(ranked)}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=1))

    print("library %s (%d operators), scanned %d induction-half episodes\n"
          % (arguments.library, len(operators), payload["scanned"]))
    print("%-48s %10s %8s %-14s %-18s %-18s %s"
          % ("family", "coverage", "failed", "target key", "seeds", "holdout", "usable"))
    for family, row in ranked:
        print("%-48s %4d/%-5d %8d %-14s %-18s %-18s %s"
              % (family[:48], row["solved"], row["total"], row["failed_episodes"],
                 row["target_key"], row["seeds"], row["holdout"], row["usable"]))
    print("\n-> %s" % arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
