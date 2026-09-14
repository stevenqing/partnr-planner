#!/usr/bin/env python3
"""Round two of the family induction: what each family is asked to add, and to what.

Round one gave every family an empty library, so the first operator that worked was
accepted and the cell ended. A family that needs two variants -- the plain delivery and the
one that opens a shut cupboard first -- could therefore only ever contribute one, and
`set_plate_and_fork_on_table` contributed the plain one. That single missing variant is
worth 155 rows of the ID column (`results/viki_coord_gap_72B_full.json`).

Round two changes exactly one thing: the family's own round-one library is handed to the
rung as `--library`, so an operator is admitted only if it makes an episode the family
already covers... not covered. A repeat of round one's own body adds nothing and is refused
with what the library already holds, which is the marginal rule that already exists, applied
within the family instead of across the union.

Nothing else moves: same tools, same acceptance (binds and achieves on >= 2 unseen
episodes), same per-family isolation, same union afterwards. This writes the targets;
`scripts/drivers/viki_v3_round2.sh` runs them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import viki_fork_guard  # noqa: E402

viki_fork_guard.install()

from viki_induction_tools import Workbench  # noqa: E402
from our_method.skill_memory_v2.simulator import SEED  # noqa: E402

TRAIN = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet"
BENCHMARK = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R"
RUNG = ROOT / "outputs/agentic_rung"
REFERENCE = ROOT / "results/viki_memory_experiments/amendment11/skill_memory_v2.json"


def round_one_library(family: str, label_format: str = "v2_%s") -> List[Dict[str, Any]]:
    """The operators this family contributed in round one, from its own verdicts."""
    out, seen = [], set()
    for path in sorted(RUNG.glob((label_format % family) + "/*/verdict.json")):
        verdict = json.loads(path.read_text())
        if str(verdict.get("passed")) != "True":
            continue
        operator = verdict.get("operator") or {}
        key = json.dumps([operator.get("effect"), operator.get("body"),
                          operator.get("roles")], sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(operator)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2, help="seed episodes per family")
    parser.add_argument("--holdout", type=int, default=4)
    parser.add_argument("--pool", type=int, default=8, help="coverage pool size")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/v3")
    # RQ2 no_trace (2026-09-14): read round one from another rung directory, e.g.
    # "rq2_notrace/v2_%s". Seeds, holdout and pool come from the episode listing and do not
    # depend on it. Default keeps the v3 targets byte-identical.
    parser.add_argument("--round-one-label", default="v2_%s")
    arguments = parser.parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)

    reference = json.loads(REFERENCE.read_text())
    bench = Workbench(TRAIN, BENCHMARK, SEED,
                      {"layer2": reference["layer2"], "layer3": reference["layer3"]})
    families = sorted({truth.get("task_name") for truth in bench.episodes
                       if isinstance(truth, dict) and truth.get("task_name")})

    targets = {"families": []}
    for family in families:
        listing = bench.list_episodes(family, arguments.seeds + arguments.holdout + arguments.pool, 0)
        indices = [item["index"] for item in listing["episodes"]]
        if len(indices) < arguments.seeds + arguments.holdout:
            continue
        library = round_one_library(family, arguments.round_one_label)
        library_path = arguments.out / ("lib_%s.json" % family)
        library_path.write_text(json.dumps({"operators": library}, indent=1))
        seeds = indices[:arguments.seeds]
        entry = {
            "family": family,
            "total_episodes": listing["total"],
            "library": str(library_path.relative_to(ROOT)),
            "library_size": len(library),
            "seeds": seeds,
            # The holdout never contains a seed, so an operator is always judged on episodes
            # it was not derived from.
            "holdout": [i for i in indices if i not in seeds][:arguments.holdout],
            "coverage_pool": [i for i in indices if i not in seeds][:arguments.pool],
        }
        targets["families"].append(entry)

    path = arguments.out / "targets.json"
    path.write_text(json.dumps(targets, indent=1))
    for entry in targets["families"]:
        print("%-46s n=%-4d lib=%d seeds=%s holdout=%s"
              % (entry["family"], entry["total_episodes"], entry["library_size"],
                 entry["seeds"], entry["holdout"]))
    print("\nwrote %s (%d families)" % (path, len(targets["families"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
