#!/usr/bin/env python3
"""Which family pairs can actually share a scene?

The first pairing failed on the data, not on the design: in all 297 cutting rows the
cabinet's init_pos is null, and eval_single drops null-position assets before the
environment is built, so cabinet and knife never co-exist as live assets.

The earlier support count was measured wrongly. It asked whether a family's
manipulated objects appear in a scene, which ignores that filter. The question that
decides the construction is narrower: a family needs the assets its goal and
temporal constraints name, and a scene offers only the assets whose init_pos is not
null. This reports, for every pair, how many evaluated scenes hold both requirement
sets, so the pairing is chosen from the data rather than from what would be tidy.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import MEMORY_PARQUET, SOURCE_PARQUET, load_manifest, native
from viki_amendment9_folds import train_family_by_index


def live(truth: Dict[str, Any]) -> Set[str]:
    return {
        name.rsplit("_", 1)[0]
        for name, positions in (truth.get("init_pos") or {}).items()
        if positions is not None
        and not (name.startswith("R") and name[1:].isdigit())
    }


def required(truth: Dict[str, Any]) -> Set[str]:
    """Assets the checker will look up: those named in the goals, the temporal
    stages, and the positions those goals demand."""
    names: Set[str] = set()
    groups = list(truth.get("goal_constraints") or [])
    for constraint in truth.get("temporal_constraints") or []:
        groups.extend(constraint)
    for group in groups:
        for item in group:
            names.add(str(item.get("name")))
            where = (item.get("status") or {}).get("pos.name")
            if where:
                names.add(str(where))
    return names


def main() -> None:
    train = pd.read_parquet(MEMORY_PARQUET)
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()
    families = train_family_by_index()

    needs: Dict[str, Set[str]] = defaultdict(set)
    robot_types: Dict[str, Counter] = defaultdict(Counter)
    for i in range(len(train)):
        truth = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"]
        family = families.get(i)
        needs[family] |= required(truth)
        robot_types[family][
            json.dumps(
                sorted(v for v in (truth.get("robots") or {}).values() if v)
            )
        ] += 1

    print("what each family's constraints name:")
    for family, assets in sorted(needs.items()):
        print(f"  {family:46s} {sorted(assets)}")

    scenes = {}
    for index in sorted(manifest):
        truth = native(test.iloc[index].to_dict())["reward_model"]["ground_truth"]
        scenes[index] = (live(truth), truth.get("task_name"))

    print()
    print("family pairs by the number of evaluated scenes holding BOTH requirement "
          "sets as live assets:")
    support: Counter = Counter()
    for a, b in combinations(sorted(needs), 2):
        want = needs[a] | needs[b]
        count = sum(1 for assets, _ in scenes.values() if want <= assets)
        if count:
            support[(a, b)] = count
    if not support:
        print("  none. no two families' requirements are ever live in one scene.")
    for (a, b), count in support.most_common(15):
        print(f"  {count:4d}  {a}  +  {b}")

    print()
    print("for reference, how many scenes hold each single family's requirements:")
    for family in sorted(needs):
        count = sum(1 for assets, _ in scenes.values() if needs[family] <= assets)
        print(f"  {count:4d}  {family}")

    print()
    print("robot rosters each family is authored for:")
    for family in sorted(robot_types):
        top = robot_types[family].most_common(2)
        print(f"  {family:46s} {top}")


if __name__ == "__main__":
    main()
