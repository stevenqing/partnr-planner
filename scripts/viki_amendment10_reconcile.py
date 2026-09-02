#!/usr/bin/env python3
"""Settle the disagreement between the two support measurements, against the scorer.

One script reported no family pair jointly satisfiable in any scene; the other
reported 623 scenes supporting two or more families. Both cannot be right, and a
third of scenes came out as not supporting their own task, which is impossible for
rows the benchmark ships as solvable. So the template model is checked against the
only authority available: the official scorer, run on each donor row's own
ground-truth plan.

The suspected error is that required() treats a goal's pos.name as an asset that
must exist. eval resolves an unknown move/place target as a bare Position, so
"table" or "kitchen work area" need not be live, while a toaster that must be
activated does. Rather than guess the rule, each family's own rows are scored and
the asset model is reported next to what the scorer actually accepts.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import SOURCE_PARQUET, load_manifest, native
from viki_amendment10_compose import judge
from viki_amendment10_pairs import live


def named(truth: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Split the constraint names by the role they play, since the scorer treats
    them differently: an activated asset must exist, a placement target need not."""
    manipulated: Set[str] = set()
    positions: Set[str] = set()
    activated: Set[str] = set()
    groups = list(truth.get("goal_constraints") or [])
    for constraint in truth.get("temporal_constraints") or []:
        groups.extend(constraint)
    for group in groups:
        for item in group:
            status = item.get("status") or {}
            name = str(item.get("name"))
            if status.get("is_activated"):
                activated.add(name)
            else:
                manipulated.add(name)
            where = status.get("pos.name")
            if where:
                positions.add(str(where))
    return {
        "manipulated": manipulated,
        "positions": positions,
        "activated": activated,
    }


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()

    by_family: Dict[str, list] = defaultdict(list)
    for index in sorted(manifest):
        truth = native(test.iloc[index].to_dict())["reward_model"]["ground_truth"]
        by_family[truth.get("task_name")].append((index, truth))

    print("Does the scorer accept each family's own rows, and what is live in them?")
    print()
    for family, rows in sorted(by_family.items()):
        accepted = 0
        codes: Counter = Counter()
        live_hits: Counter = Counter()
        for index, truth in rows[:40]:
            ok, code = judge(scorer, truth["time_steps"], truth)
            if ok:
                accepted += 1
            else:
                codes[code or "False"] += 1
            assets = live(truth)
            roles = named(truth)
            for role, names in roles.items():
                for name in names:
                    live_hits[(role, name in assets)] += 1
        sampled = len(rows[:40])
        print(f"{family}  ({len(rows)} rows, {sampled} sampled)")
        print(f"  own ground-truth plan accepted: {accepted}/{sampled}")
        if codes:
            print(f"  refusals: {dict(codes)}")
        summary = {
            role: {
                "live": live_hits[(role, True)],
                "not live": live_hits[(role, False)],
            }
            for role in ("manipulated", "positions", "activated")
        }
        print(f"  constraint names by role, live vs not: {json.dumps(summary)}")
        example = rows[0][1]
        roles = named(example)
        print(f"  example goals: {json.dumps(example['goal_constraints'])[:220]}")
        print(f"  example roles: { {k: sorted(v) for k, v in roles.items()} }")
        print()


if __name__ == "__main__":
    main()
