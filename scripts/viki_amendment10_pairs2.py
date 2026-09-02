#!/usr/bin/env python3
"""Family pair support, with the asset requirement taken from the scorer's own rules.

Two corrections carried into this pass. transform_actions discards a step whose
actions still carry the dataset's null for an idle robot, which made every family
fail its own ground truth; the nulls are dropped before scoring, as the tolerant
metric already did. And a goal's pos.name is not automatically an asset: the role
tally showed cabinet, plate, bowl, sink and toaster live in every row that names
them, while table and kitchen work area are live in none -- eval resolves an unknown
move or place target as a bare Position, so only container targets have to exist.

A family's requirement is therefore the assets it manipulates, the assets it
activates, and the container targets it places into. Each family is checked against
its own rows first: a model that cannot satisfy a family in its own scenes is wrong,
and this one is only used once that check passes.
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

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import MEMORY_PARQUET, SOURCE_PARQUET, load_manifest, native
from viki_amendment9_folds import train_family_by_index
from viki_amendment10_compose import judge
from viki_amendment10_pairs import live

FIXED_SHARE = 0.9


def requirement(truth: Dict[str, Any], containers: Set[str]) -> Set[str]:
    names: Set[str] = set()
    groups = list(truth.get("goal_constraints") or [])
    for constraint in truth.get("temporal_constraints") or []:
        groups.extend(constraint)
    for group in groups:
        for item in group:
            names.add(str(item.get("name")))
            where = (item.get("status") or {}).get("pos.name")
            if where and str(where) in containers:
                names.add(str(where))
    return names


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    containers = set(scorer.eval_single.__globals__["CONTAINER_ASSETS"])
    print(f"container assets: {sorted(containers)}")

    train = pd.read_parquet(MEMORY_PARQUET)
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()
    families = train_family_by_index()

    rows_per_family: Counter = Counter()
    names_per_family: Dict[str, Counter] = defaultdict(Counter)
    size_per_family: Dict[str, Counter] = defaultdict(Counter)
    for i in range(len(train)):
        truth = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"]
        family = families.get(i)
        rows_per_family[family] += 1
        want = requirement(truth, containers)
        names_per_family[family].update(want)
        size_per_family[family][len(want)] += 1

    fixed: Dict[str, Set[str]] = {}
    variable: Dict[str, Set[str]] = {}
    slots: Dict[str, int] = {}
    for family, counts in names_per_family.items():
        total = rows_per_family[family]
        fixed[family] = {n for n, c in counts.items() if c >= FIXED_SHARE * total}
        variable[family] = set(counts) - fixed[family]
        slots[family] = max(
            0, size_per_family[family].most_common(1)[0][0] - len(fixed[family])
        )

    scenes: Dict[int, Any] = {}
    for index in sorted(manifest):
        truth = native(test.iloc[index].to_dict())["reward_model"]["ground_truth"]
        scenes[index] = (live(truth), truth.get("task_name"), truth)

    def supports(family: str, assets: Set[str]):
        if not fixed[family] <= assets:
            return False, []
        options = sorted(variable[family] & assets)
        return (len(options) >= slots[family]), options

    print()
    print("self-check: a scene must support the family whose task it carries")
    own = Counter()
    for assets, mine, _ in scenes.values():
        if mine in fixed:
            own["supported" if supports(mine, assets)[0] else "NOT supported"] += 1
    print(f"  {dict(own)}")
    accepted = sum(
        1
        for _, _, truth in list(scenes.values())[:60]
        if judge(scorer, truth["time_steps"], truth)[0]
    )
    print(f"  scorer accepts own ground truth on 60 sampled rows: {accepted}/60")

    print()
    print(f"{'family':46s} fixed + slots")
    for family in sorted(fixed):
        print(f"  {family:44s} {sorted(fixed[family])} + {slots[family]}")

    print()
    print("scenes supporting each family:")
    for family in sorted(fixed):
        n = sum(1 for assets, _, _ in scenes.values() if supports(family, assets)[0])
        print(f"  {n:4d}/924  {family}")

    print()
    print("scenes supporting BOTH families, with enough distinct slot fillers:")
    found: Counter = Counter()
    for a, b in combinations(sorted(fixed), 2):
        count = 0
        for assets, _, _ in scenes.values():
            ok_a, options_a = supports(a, assets)
            ok_b, options_b = supports(b, assets)
            if ok_a and ok_b and len(set(options_a) | set(options_b)) >= slots[a] + slots[b]:
                count += 1
        if count:
            found[(a, b)] = count
    if not found:
        print("  none")
    for (a, b), count in found.most_common(20):
        print(f"  {count:4d}  {a}  +  {b}")


if __name__ == "__main__":
    main()
