#!/usr/bin/env python3
"""Is each VIKI-L2 scene tailored to exactly one task family?

No pair of families is jointly satisfiable in any of the 924 evaluated scenes, and
the single-family counts add up suspiciously close to the total without ever
overlapping. The explanation worth testing is that a scene is built for its own
task: the assets that task needs are given positions and everything else is left
null, which eval_single then filters away.

If that holds, a cross-task recombination split cannot be cut from this data at all
-- the obstruction is the dataset's design, not the construction -- and the only
route left is adding assets to init_pos, which desynchronises the scene from its
image. The cost of that route is quantified here too, as the number of assets a
scene would need before it could carry a second family's goal.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import MEMORY_PARQUET, SOURCE_PARQUET, load_manifest, native
from viki_amendment9_folds import train_family_by_index
from viki_amendment10_pairs import live, required

FIXED_SHARE = 0.9


def main() -> None:
    train = pd.read_parquet(MEMORY_PARQUET)
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()
    families = train_family_by_index()

    rows_per_family: Counter = Counter()
    names_per_family: Dict[str, Counter] = {}
    slots_per_family: Dict[str, Counter] = {}
    for i in range(len(train)):
        truth = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"]
        family = families.get(i)
        rows_per_family[family] += 1
        names_per_family.setdefault(family, Counter()).update(required(truth))
        slots_per_family.setdefault(family, Counter())[len(required(truth))] += 1

    fixed: Dict[str, Set[str]] = {}
    variable: Dict[str, Set[str]] = {}
    slots: Dict[str, int] = {}
    for family, counts in names_per_family.items():
        total = rows_per_family[family]
        fixed[family] = {n for n, c in counts.items() if c >= FIXED_SHARE * total}
        variable[family] = set(counts) - fixed[family]
        slots[family] = max(
            0, slots_per_family[family].most_common(1)[0][0] - len(fixed[family])
        )

    def supports(family: str, assets: Set[str]) -> bool:
        return fixed[family] <= assets and len(
            variable[family] & assets
        ) >= slots[family]

    own_only = Counter()
    matrix: Counter = Counter()
    missing_for_cabinet: Counter = Counter()
    for index in sorted(manifest):
        truth = native(test.iloc[index].to_dict())["reward_model"]["ground_truth"]
        assets = live(truth)
        mine = truth.get("task_name")
        supported = {f for f in fixed if supports(f, assets)}
        matrix[len(supported)] += 1
        if supported == {mine}:
            own_only["exactly its own family"] += 1
        elif mine in supported:
            own_only["its own family and others"] += 1
        elif supported:
            own_only["others but not its own"] += 1
        else:
            own_only["no family at all"] += 1
        if mine in ("cut_fruit_on_board", "cut_two_fruits_on_board"):
            need = fixed["clear_table_with_two_robots_and_put_in_cabinet"] - assets
            spare = len(
                variable["clear_table_with_two_robots_and_put_in_cabinet"] & assets
            )
            shortfall = max(
                0, slots["clear_table_with_two_robots_and_put_in_cabinet"] - spare
            )
            missing_for_cabinet[len(need) + shortfall] += 1

    print("how many family templates each evaluated scene supports:")
    for count, n in sorted(matrix.items()):
        print(f"  supports {count} famil{'y' if count == 1 else 'ies'}: {n:4d} scenes")
    print()
    print("relative to the scene's own task:")
    for label, n in own_only.most_common():
        print(f"  {label:32s} {n:4d}  {100*n/len(manifest):5.1f}%")
    print()
    print(
        "for the 297 cutting scenes, assets that would have to be added to "
        "init_pos before they could also carry the cabinet goal:"
    )
    for count, n in sorted(missing_for_cabinet.items()):
        print(f"  {count} asset(s) to add: {n:4d} scenes")


if __name__ == "__main__":
    main()
