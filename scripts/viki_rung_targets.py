"""Which induction-half episodes actually demonstrate which effect key.

The first rung sweep took episodes 0, 1, 3, 7, 9 and a holdout pool of 4, 6, 8, 10, 12 --
reasonable as a first draw, and the reason every one of the twelve passes came back as the
same `pos.name` pick-and-place: nothing ever asked the model for another shape, and the
holdout it was graded on did not require one.

Coverage has to be selected, not hoped for. This walks the induction half, records which
effect keys each episode's completions carry, and emits seed and holdout sets per key with
**no episode in both** -- an operator verified on an episode it was derived from would be
measuring memorisation.

Deterministic and model-free; it reads the same half `viki_induction_tools.Workbench`
exposes and nothing else.
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
    parser.add_argument("--scan", type=int, default=120,
                        help="how many induction-half episodes to walk")
    parser.add_argument("--seeds", type=int, default=5, help="seed episodes per key")
    parser.add_argument("--holdout", type=int, default=4, help="holdout episodes per key")
    parser.add_argument("--out", type=Path, default=Path("outputs/rung_targets.json"))
    arguments = parser.parse_args()

    reference = json.loads(REFERENCE.read_text())
    bench = Workbench(TRAIN, BENCHMARK, SEED,
                      {"layer2": reference["layer2"], "layer3": reference["layer3"]})

    carries = defaultdict(list)
    for index in range(min(arguments.scan, len(bench.episodes))):
        trace, _ = bench.trace(index)
        if trace is None:
            continue
        for _, _, predicate in trace["completions"]:
            key = key_of(predicate)
            if index not in carries[key]:
                carries[key].append(index)

    targets = {}
    for key, episodes in sorted(carries.items()):
        # Holdout taken from the far end so that seeds and holdout are disjoint even when a
        # key is thin, and the split does not move if `--scan` grows.
        holdout = episodes[-arguments.holdout:]
        seeds = [e for e in episodes if e not in holdout][: arguments.seeds]
        targets[key] = {"episodes_carrying_it": len(episodes), "seeds": seeds,
                        "holdout": holdout,
                        "usable": len(seeds) >= 1 and len(holdout) >= 2}

    payload = {"scanned": arguments.scan, "induction_half": len(bench.episodes),
               "targets": targets}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=1))

    print("scanned %d of %d induction-half episodes\n" % (arguments.scan, len(bench.episodes)))
    print("%-16s %10s %-22s %-16s %s" % ("effect key", "episodes", "seeds", "holdout", "usable"))
    for key, row in targets.items():
        print("%-16s %10d %-22s %-16s %s"
              % (key, row["episodes_carrying_it"], row["seeds"], row["holdout"], row["usable"]))
    print("\n-> %s" % arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
