#!/usr/bin/env python3
"""Put known-correct operators through the workbench's own acceptance test.

`run_operator` is what decides whether an agent's operator is real: the rung accepts a
submission when it binds and its effect then holds on at least two held-out episodes. So
before asking why no agent ever produced a coordination operator or an unsealing one, the
reference library's own -- induced mechanically, in the library the whole method is
measured against -- are put through the same test. An operator that is correct by
construction and fails here is a statement about the test, not about the operator.

This is the check [[viki-harness-was-the-bottleneck]] asks for, and it is meant to be run
twice: once before the verifier is changed and once after, so the repair is a difference
between two measured columns rather than an assertion.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
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
REFERENCE = ROOT / "results/viki_memory_experiments/amendment11/skill_memory_v2.json"


def describe(operator: Dict[str, Any]) -> str:
    if operator.get("coordinated"):
        return "COORD[%d] %s" % (
            len(operator["roles"]),
            " | ".join(" ".join(item["action"][0] for item in role["actions"])
                       for role in operator["roles"]))
    return "%s %s :: %s" % (operator.get("kind", "?"),
                            (operator.get("effect") or {}).get("key"),
                            " ".join(action[0] for action in (operator.get("body") or [])))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", type=int, nargs="*", default=[0, 1, 3, 8, 10, 13],
                        help="reference operator indices to put through the test")
    parser.add_argument("--episodes", type=int, default=6,
                        help="episodes per operator, taken from its own donor families")
    parser.add_argument("--label", default="before",
                        help="which column this run is: before or after the repair")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)
    out = arguments.json or ROOT / ("results/viki_workbench_known_answer_%s_%s.json"
                                    % (arguments.label, date.today().isoformat()))

    reference = json.loads(REFERENCE.read_text())
    ref_ops = reference["layer1"]["operators"]
    bench = Workbench(TRAIN, BENCHMARK, SEED,
                      {"layer2": reference["layer2"], "layer3": reference["layer3"]})

    report = {"generated": date.today().isoformat(), "label": arguments.label,
              "episodes_per_operator": arguments.episodes, "operators": []}

    for i in arguments.operators:
        operator = ref_ops[i]
        families = operator.get("families") or []
        indices: List[int] = []
        for family in families:
            listing = bench.list_episodes(family, arguments.episodes, 0)
            for item in (listing if isinstance(listing, list) else listing.get("episodes", [])):
                index = item.get("index") if isinstance(item, dict) else item
                if isinstance(index, int) and index not in indices:
                    indices.append(index)
            if len(indices) >= arguments.episodes:
                break
        indices = indices[:arguments.episodes]

        outcomes = []
        for index in indices:
            try:
                result = bench.run_operator(operator, index)
            except Exception as error:                                   # noqa: BLE001
                result = {"bound": False, "error": "%s: %s" % (type(error).__name__, error)}
            outcomes.append({
                "episode": index,
                "bound": bool(result.get("bound")),
                "effect_holds": bool(result.get("effect_holds")),
                "failure": result.get("failure") or result.get("error")
                           or (result.get("binding") or {}).get("why"),
            })
        entry = {
            "index": i, "operator": describe(operator), "families": families,
            "coordinated": bool(operator.get("coordinated")),
            "effect_key": (operator.get("effect") or {}).get("key"),
            "episodes": indices,
            "bound": sum(o["bound"] for o in outcomes),
            "passes": sum(o["bound"] and o["effect_holds"] for o in outcomes),
            "n": len(outcomes),
            "outcomes": outcomes,
            "why": dict(Counter(str(o["failure"])[:90] for o in outcomes if not
                                (o["bound"] and o["effect_holds"])).most_common(2)),
        }
        report["operators"].append(entry)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
        print("ref%-2d %-62s bound %d/%d  passes %d/%d  %s"
              % (i, entry["operator"][:62], entry["bound"], entry["n"],
                 entry["passes"], entry["n"], list(entry["why"])[:1]), flush=True)

    accepted = [e["index"] for e in report["operators"] if e["passes"] >= 2]
    report["would_be_accepted"] = accepted
    out.write_text(json.dumps(report, indent=1))
    print("\nwould pass the rung's >=2 rule: %s" % accepted)
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
