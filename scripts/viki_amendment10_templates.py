#!/usr/bin/env python3
"""Which family pairs can share a scene, measured per template rather than per union.

The previous count took a family's requirement to be the union of every asset any of
its rows names, which no single scene holds -- every family scored zero against its
own scenes, so the metric was wrong rather than the data being empty. A family's real
requirement is a template: a few fixed assets that every row of it names, plus slots
filled by whichever item that scene happens to offer. cut_fruit always needs the
knife and the board and then some fruit; clear_table always needs the cabinet and
then two things to put in it.

A scene supports a family when its fixed assets are live and its slots can be filled
from what is live. A scene supports a pair when it supports both and the slots can be
filled without the two goals fighting over the same item. Only the fixed part is
inferred from the data here; nothing is hand-declared.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import MEMORY_PARQUET, SOURCE_PARQUET, load_manifest, native
from viki_amendment9_folds import train_family_by_index
from viki_amendment10_pairs import live, required

FIXED_SHARE = 0.9  # an asset named by ~every row of a family is part of its fixture


def main() -> None:
    train = pd.read_parquet(MEMORY_PARQUET)
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()
    families = train_family_by_index()

    rows_per_family: Counter = Counter()
    names_per_family: Dict[str, Counter] = defaultdict(Counter)
    slots_per_family: Dict[str, Counter] = defaultdict(Counter)
    for i in range(len(train)):
        truth = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"]
        family = families.get(i)
        rows_per_family[family] += 1
        wanted = required(truth)
        for name in wanted:
            names_per_family[family][name] += 1
        slots_per_family[family][len(wanted)] += 1

    fixed: Dict[str, Set[str]] = {}
    variable: Dict[str, Set[str]] = {}
    slot_count: Dict[str, int] = {}
    for family, counts in names_per_family.items():
        total = rows_per_family[family]
        fixed[family] = {n for n, c in counts.items() if c >= FIXED_SHARE * total}
        variable[family] = set(counts) - fixed[family]
        typical = slots_per_family[family].most_common(1)[0][0]
        slot_count[family] = max(0, typical - len(fixed[family]))

    print(f"{'family':46s} {'rows':>5s}  fixed assets / slots")
    for family in sorted(fixed):
        print(
            f"  {family:44s} {rows_per_family[family]:5d}  "
            f"{sorted(fixed[family])} + {slot_count[family]} slot(s) from "
            f"{len(variable[family])} options"
        )

    scenes: Dict[int, Set[str]] = {}
    for index in sorted(manifest):
        truth = native(test.iloc[index].to_dict())["reward_model"]["ground_truth"]
        scenes[index] = live(truth)

    def supports(family: str, assets: Set[str]) -> Tuple[bool, List[str]]:
        if not fixed[family] <= assets:
            return False, []
        options = sorted(variable[family] & assets)
        if len(options) < slot_count[family]:
            return False, []
        return True, options

    print()
    print("scenes supporting each family on its own:")
    single: Dict[str, int] = {}
    for family in sorted(fixed):
        single[family] = sum(1 for a in scenes.values() if supports(family, a)[0])
        print(f"  {single[family]:4d}/{len(scenes)}  {family}")

    print()
    print("scenes supporting BOTH families of a pair, with disjoint slot fillers:")
    support: Counter = Counter()
    for a, b in combinations(sorted(fixed), 2):
        count = 0
        for assets in scenes.values():
            ok_a, options_a = supports(a, assets)
            ok_b, options_b = supports(b, assets)
            if not (ok_a and ok_b):
                continue
            # the two goals must not need the same item, or the recombination is
            # a single goal wearing two labels
            pool = set(options_a) | set(options_b)
            if len(pool) >= slot_count[a] + slot_count[b]:
                count += 1
        if count:
            support[(a, b)] = count
    if not support:
        print("  none")
    for (a, b), count in support.most_common(20):
        print(f"  {count:4d}  {a}  +  {b}")


if __name__ == "__main__":
    main()
