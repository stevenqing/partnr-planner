#!/usr/bin/env python3
"""Combine family libraries into the memory a column is scored with. Read-only, zero calls.

Libraries are built per family and combined here, never rebuilt per split. The rules are
fixed by the spec and are not the same for the three layers:

  Layer 1  union of the named families' libraries, deduplicated by effect schema and body
           signature, provenance merged, and support RECOMPUTED on the union's own episode
           pool -- the per-family counts are not carried over and not summed.
  Layer 2  NOT unioned. Re-mined on the union's episode pool, MIN_SUPPORT and MIN_PRECISION
           unchanged. Mining is Layer-1-independent by construction (`dependencies.mine`
           takes no library), so this is a property of the pool, not of the operators.
  Layer 3  likewise re-harvested on that pool.

A held-out-family column therefore never sees the held-out family in any layer, which is
what the fold artefacts do for the rule-based library and what a per-column rebuild would
otherwise have to do by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def body_signature(operator: Dict[str, Any]) -> str:
    effect = operator.get("effect") or {}
    if operator.get("coordinated"):
        body = [[item["action"] for item in role["actions"]] for role in operator.get("roles", [])]
    else:
        body = operator.get("body", [])
    return json.dumps([effect.get("key"), effect.get("subject"), effect.get("value"),
                       bool(operator.get("coordinated")), body], sort_keys=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--libraries", type=Path, nargs="+", required=True)
    parser.add_argument("--families", nargs="+", required=True,
                        help="the episode pool Layers 2 and 3 are mined on, and support "
                             "recomputed over")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT.parent / "VIKI-R")
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--probe", type=int, default=60,
                        help="episodes from the union pool that support is measured over")
    parser.add_argument("--min-support", type=int, default=2)
    arguments = parser.parse_args(argv)

    from our_method.skill_memory_v2 import dependencies, vocabulary
    from our_method.skill_memory_v2.build import load_episodes
    from our_method.skill_memory_v2.memory import FORMAT
    from our_method.skill_memory_v2.simulator import SEED, Simulator
    from viki_induction_tools import Workbench

    train = arguments.train or (arguments.benchmark_root / "data/VIKI-R/viki/VIKI-L2/train.parquet")
    families = set(arguments.families)
    episodes = load_episodes(train)
    induction = [e for i, e in enumerate(episodes) if i % 2 == 0]
    pool = [e for e in induction
            if isinstance(e, dict) and e.get("task_name") in families]
    pool_indices = [i for i, e in enumerate(induction)
                    if isinstance(e, dict) and e.get("task_name") in families]

    # ---- Layer 1: union, dedup, merged provenance
    merged: Dict[str, Dict[str, Any]] = {}
    contributions = Counter()
    for path in arguments.libraries:
        if not path.is_file():
            print("未执行，缺 %s" % path)
            return 1
        record = json.loads(path.read_text())
        operators = record["operators"] if isinstance(record, dict) else record
        for operator in operators:
            signature = body_signature(operator)
            contributions[path.name] += 1
            if signature in merged:
                slot = merged[signature]
                for field in ("families", "runner_types"):
                    for name in operator.get(field) or []:
                        if name not in slot.setdefault(field, []):
                            slot[field].append(name)
                slot.setdefault("from_libraries", []).append(path.name)
            else:
                entry = dict(operator)
                entry["from_libraries"] = [path.name]
                merged[signature] = entry

    # ---- support recomputed on the union pool, never summed across families
    sim = Simulator(arguments.benchmark_root)
    bench = Workbench(train, arguments.benchmark_root, SEED)
    probe = pool_indices[:arguments.probe]
    operators, rows = [], []
    for signature, operator in merged.items():
        works = []
        for index in probe:
            try:
                outcome = bench.run_operator(operator, index)
            except Exception:                                        # noqa: BLE001
                continue
            if outcome.get("bound") and outcome.get("effect_holds"):
                works.append(index)
        operator = dict(operator)
        operator["support"] = len(works)
        rows.append({"effect_key": (operator.get("effect") or {}).get("key"),
                     "body": [a[0] for a in operator.get("body", [])],
                     "from_libraries": operator.get("from_libraries"),
                     "measured_support": len(works),
                     "admitted": len(works) >= arguments.min_support})
        if len(works) >= arguments.min_support:
            operators.append(operator)
    operators.sort(key=lambda item: -item.get("support", 0))

    # ---- Layers 2 and 3 re-mined on the same pool
    layer2 = dependencies.mine(pool, sim, SEED, 250, None)
    layer3 = vocabulary.harvest(pool, None)

    record = {"format": FORMAT,
              "built_from": "union of %s" % ", ".join(p.name for p in arguments.libraries),
              "union_families": sorted(families), "excluded_family": None,
              "seed": SEED, "per_family": 250,
              "layer1": {"operators": operators},
              "layer2": layer2, "layer3": layer3,
              "union_report": {"pool_episodes": len(pool),
                               "support_probe_episodes": len(probe),
                               "operators_in": sum(contributions.values()),
                               "distinct_after_dedup": len(merged),
                               "admitted": len(operators),
                               "rows": rows}}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    digest = hashlib.sha256(arguments.out.read_bytes()).hexdigest()

    print("families        %s" % ", ".join(sorted(families)))
    print("pool episodes   %d   support probe %d" % (len(pool), len(probe)))
    print("operators in    %d -> distinct %d -> admitted %d"
          % (sum(contributions.values()), len(merged), len(operators)))
    for row in sorted(rows, key=lambda r: -r["measured_support"]):
        print("   %-13s %-52s support %-4d %s"
              % (row["effect_key"], " ".join(row["body"]), row["measured_support"],
                 "" if row["admitted"] else "REJECTED"))
    print("layer2 kept_patterns %d   layer3 places %d"
          % (len(layer2.get("kept_patterns", [])), len(layer3.get("places", []))))
    print("wrote %s  sha256 %s" % (arguments.out, digest[:16]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
