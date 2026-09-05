"""Seed the ladder where the current library actually fails, instead of at the front of the split.

Twenty-four accepted submissions deduplicated to two operators. That is not a sampling
accident: every seed episode was solvable by the plainest body there is, so the plainest
body is what came back, twice. The reference holds six `pos.name` operators and four
`is_activated` ones, and the extra ones are not better bodies -- they are the *cases the
plain body does not cover*: subject sealed in a container (`Move Reach Open Move Reach
Grasp Move Place`), target sealed (`... Reach Open Place`), already in hand, and so on,
each carrying the preconditions that say when it applies. The agent's two operators carry
`preconditions: {}` and `types: {}`, so the planner offers them in states where they cannot
work.

Residual seeding names none of that. It runs the current library over the induction half
and takes the episodes it fails, which is a mechanical fact about the library, not a hint
about what is missing. Two failure modes are kept apart because they call for different
seeds:

  unsupported   the planner had no operator for the effect key at all -- a coverage hole.
  planned_but_failed
                an operator was offered and the plan did not achieve the goal -- a *variant*
                hole, and the one that is currently invisible to the rung, because the rung
                passes an operator that works on two episodes without ever asking what it
                does on a third where it should not apply.

Holdout is drawn from the residual too, so an operator can only pass here by covering
something the current library does not. Seeds and holdout never overlap.

Reads `episodes[::2]` and the library's own outcomes. The reference library is never
consulted -- it is the thing being caught up to.
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
    parser.add_argument("--library", type=Path, default=Path("outputs/agentic_library_all.json"))
    parser.add_argument("--scan", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--holdout", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("outputs/rung_residual_targets.json"))
    arguments = parser.parse_args()

    reference = json.loads(REFERENCE.read_text())
    bench = Workbench(TRAIN, BENCHMARK, SEED,
                      {"layer2": reference["layer2"], "layer3": reference["layer3"]})
    operators = json.loads(arguments.library.read_text())["operators"]

    solved, residual = [], []
    modes = Counter()
    keys_of_episode = {}
    for index in range(min(arguments.scan, len(bench.episodes))):
        trace, _ = bench.trace(index)
        if trace is None:
            continue
        try:
            outcome = bench.plan_with(operators, index)
        except Exception:
            modes["ERROR"] += 1
            continue
        keys = sorted({key_of(p) for _, _, p in trace["completions"]})
        keys_of_episode[index] = keys
        if outcome["official_score"] >= 1.0:
            solved.append(index)
            modes["solved"] += 1
            continue
        mode = "unsupported" if outcome["reason"] == "UNSUPPORTED_PREDICATE" else "planned_but_failed"
        modes[mode] += 1
        residual.append({"episode": index, "mode": mode, "reason": outcome["reason"],
                         "score": outcome["official_score"], "keys": keys})

    # Group by key, preferring the variant holes as seeds: those are the ones the rung is
    # currently blind to, and the coverage holes were already the last sweep's target.
    by_key = defaultdict(list)
    for row in residual:
        for key in row["keys"]:
            by_key[key].append(row)

    targets = {}
    for key, rows in sorted(by_key.items()):
        rows = sorted(rows, key=lambda r: (r["mode"] != "planned_but_failed", r["episode"]))
        episodes = [r["episode"] for r in rows]
        holdout = episodes[-arguments.holdout:]
        seeds = [e for e in episodes if e not in holdout][: arguments.seeds]
        targets[key] = {
            "residual_episodes": len(episodes),
            "modes": dict(Counter(r["mode"] for r in rows)),
            "seeds": seeds,
            "holdout": holdout,
            "usable": len(seeds) >= 1 and len(holdout) >= 2,
        }

    payload = {
        "library": str(arguments.library),
        "operators_in_library": len(operators),
        "scanned": min(arguments.scan, len(bench.episodes)),
        "solved": len(solved),
        "residual": len(residual),
        "modes": dict(modes),
        "targets": targets,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=1))

    print("library %s (%d operators)" % (arguments.library, len(operators)))
    print("scanned %d   solved %d   residual %d   %s"
          % (payload["scanned"], len(solved), len(residual), dict(modes)))
    print()
    print("%-16s %10s %-30s %-22s %-16s %s"
          % ("effect key", "residual", "modes", "seeds", "holdout", "usable"))
    for key, row in targets.items():
        print("%-16s %10d %-30s %-22s %-16s %s"
              % (key, row["residual_episodes"], row["modes"], row["seeds"],
                 row["holdout"], row["usable"]))
    print("\n-> %s" % arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
