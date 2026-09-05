"""Could ANY single operator have passed the marginal test on each family's holdout?

A zero pass rate is only about the model if a pass was available. This asks the question by
trying every operator in the *reference* library against the agent-built library's holdout:
if adding one reference operator turns unsolved holdout episodes into solved ones, then an
operator that passes exists and the model failing to find it is a finding. If none does, the
family is unpassable by construction and its runs say nothing about capability.

**This is a control and must never gate selection.** It reads the reference library, so
using it to choose which families to sweep would leak reference knowledge into the seeding
and the whole "induced from our traces, not from pretraining" claim with it. Seeds come from
the residual only. This script is run afterwards, to say which of the numbers already
collected are interpretable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import viki_fork_guard  # noqa: E402

viki_fork_guard.install()

from viki_induction_tools import Workbench  # noqa: E402
from our_method.skill_memory_v2.simulator import SEED  # noqa: E402

TRAIN = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet"
BENCHMARK = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R"
REFERENCE = Path("results/viki_memory_experiments/amendment11/skill_memory_v2.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("outputs/agentic_library_marg.json"))
    parser.add_argument("--targets", type=Path, default=Path("outputs/rung_family_targets.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/viki_family_passability.json"))
    arguments = parser.parse_args()

    reference = json.loads(REFERENCE.read_text())
    bench = Workbench(TRAIN, BENCHMARK, SEED,
                      {"layer2": reference["layer2"], "layer3": reference["layer3"]})
    library = json.loads(arguments.library.read_text())["operators"]
    targets = json.loads(arguments.targets.read_text())["targets"]

    rows = {}
    for family, target in targets.items():
        holdout = target["holdout"]
        unsolved = [j for j in holdout
                    if bench.plan_with(library, j)["official_score"] < 1.0]
        gains = []
        for position, operator in enumerate(reference["layer1"]["operators"]):
            got = [j for j in unsolved
                   if bench.plan_with(library + [operator], j)["official_score"] >= 1.0]
            if got:
                gains.append({"operator": position,
                              "kind": operator.get("kind", "achievement"),
                              "effect": operator["effect"]["key"],
                              "episodes_gained": len(got)})
        rows[family] = {"holdout": holdout, "unsolved": unsolved,
                        "single_operator_suffices": bool(gains),
                        "which": gains,
                        "kinds_that_work": sorted({g["kind"] for g in gains})}

    payload = {"library": str(arguments.library), "rows": rows,
               "passable": sorted(f for f, r in rows.items() if r["single_operator_suffices"]),
               "unpassable": sorted(f for f, r in rows.items() if not r["single_operator_suffices"])}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=1))

    print("%-48s %-10s %-24s %s" % ("family", "unsolved", "kinds that would pass", "passable"))
    for family, row in rows.items():
        print("%-48s %-10d %-24s %s"
              % (family[:48], len(row["unsolved"]), ",".join(row["kinds_that_work"]) or "-",
                 row["single_operator_suffices"]))
    print("\npassable families:   %s" % payload["passable"])
    print("unpassable families: %s   <- runs on these say nothing about capability"
          % payload["unpassable"])
    print("\n-> %s" % arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
