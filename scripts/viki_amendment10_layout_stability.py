#!/usr/bin/env python3
"""Is an asset's placement a property of the layout, or resampled for every row?

The transplant route needs the first. If layout 5 always puts the wine in the same
place, then giving a cutting row of layout 5 that wine is restoring a fact the
dataset itself records, and the row's image should already show it. If instead every
row resamples where objects sit, a null asset is genuinely absent from that scene and
adding it would contradict the picture the model is shown.

The previous count could not tell these apart, because init_pos maps an asset to a
LIST of candidate positions -- eval_single picks one with random.choice -- so
"several positions" conflated within-row candidates with across-row variation. Here
each row contributes one frozen set per asset, and the question is whether those sets
agree across the rows of a layout.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import MEMORY_PARQUET, SOURCE_PARQUET, native


def main() -> None:
    train = pd.read_parquet(MEMORY_PARQUET)
    test = pd.read_parquet(SOURCE_PARQUET)

    per_asset: Dict[Any, Dict[str, Set[FrozenSet[str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    rows_placing: Dict[Any, Counter] = defaultdict(Counter)
    for frame in (train, test):
        for i in range(len(frame)):
            truth = native(frame.iloc[i].to_dict())["reward_model"]["ground_truth"]
            layout = truth.get("layout_id")
            for name, positions in (truth.get("init_pos") or {}).items():
                if positions is None:
                    continue
                if name.startswith("R") and name[1:].isdigit():
                    continue
                values = positions if isinstance(positions, list) else [positions]
                per_asset[layout][name].add(frozenset(str(v) for v in values))
                rows_placing[layout][name] += 1

    verdict: Counter = Counter()
    candidates: Counter = Counter()
    examples = []
    for layout, assets in sorted(per_asset.items(), key=lambda kv: str(kv[0])):
        for name, sets in assets.items():
            verdict[
                "same set in every row that places it"
                if len(sets) == 1
                else f"{min(len(sets), 5)} different sets"
            ] += 1
            if len(sets) == 1:
                candidates[len(next(iter(sets)))] += 1
            elif len(examples) < 5:
                examples.append(
                    f"  layout {layout} / {name}: seen as "
                    + " | ".join(sorted(str(sorted(s)) for s in sets)[:3])
                )

    total = sum(verdict.values())
    print(f"(layout, asset) pairs examined: {total}")
    for label, count in verdict.most_common():
        print(f"  {label:38s} {count:5d}  {100*count/total:5.1f}%")

    print()
    print("for the stable ones, how many candidate positions the row offers:")
    for size, count in sorted(candidates.items()):
        print(f"  {size} position(s): {count:5d}")

    if examples:
        print()
        print("examples of assets whose placement varies between rows:")
        for line in examples:
            print(line)

    print()
    print("rows that place each asset, by layout (a sample):")
    for layout in list(sorted(per_asset, key=str))[:3]:
        common = rows_placing[layout].most_common(6)
        print(f"  layout {layout}: {common}")


if __name__ == "__main__":
    main()
