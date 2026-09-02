#!/usr/bin/env python3
"""How many cutting scenes could take a second goal using only stable placements?

Placement is resampled per row: only 204 of 440 (layout, asset) pairs keep the same
position set across the rows that place them, and the rest move between the cabinet
and the counters. Borrowing a position in general is therefore unsound.

The narrow version restricts the transplant to assets the dataset never moves within
a layout. That is stricter than the earlier attempt, not looser, and it bounds what
the route could yield. It does not settle the other half of the problem -- whether an
asset left null in a row is rendered in that row's image at all -- which metadata
cannot answer, so this number is an upper bound on a route that still needs that
question resolved.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import MEMORY_PARQUET, SOURCE_PARQUET, load_manifest, native
from viki_amendment9_folds import train_family_by_index
from viki_amendment10_pairs import live
from viki_amendment10_recombine import CUTTING, delivery_goals


def main() -> None:
    train = pd.read_parquet(MEMORY_PARQUET)
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()
    families = train_family_by_index()
    catalogue = delivery_goals(train, families)

    seen: Dict[Any, Dict[str, Set[FrozenSet[str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for frame in (train, test):
        for i in range(len(frame)):
            truth = native(frame.iloc[i].to_dict())["reward_model"]["ground_truth"]
            layout = truth.get("layout_id")
            for name, positions in (truth.get("init_pos") or {}).items():
                if positions is None or (
                    name.startswith("R") and name[1:].isdigit()
                ):
                    continue
                values = positions if isinstance(positions, list) else [positions]
                seen[layout][name].add(frozenset(str(v) for v in values))

    stable = {
        layout: {
            name: next(iter(sets))
            for name, sets in assets.items()
            if len(sets) == 1
        }
        for layout, assets in seen.items()
    }
    print(
        "stable (layout, asset) pairs: "
        f"{sum(len(v) for v in stable.values())} of "
        f"{sum(len(v) for v in seen.values())}"
    )

    yield_counts: Counter = Counter()
    examples = []
    for index in sorted(manifest):
        truth = native(test.iloc[index].to_dict())["reward_model"]["ground_truth"]
        if truth.get("task_name") not in CUTTING:
            continue
        layout = truth.get("layout_id")
        assets = live(truth)
        used = {
            str(a[1])
            for step in truth["time_steps"]
            for a in step["actions"].values()
            if a is not None and len(a) > 1
        }
        options = {}
        for name, positions in stable.get(layout, {}).items():
            kind = name.rsplit("_", 1)[0]
            if kind in catalogue and kind not in assets and kind not in used:
                options[kind] = sorted(positions)
        yield_counts[min(len(options), 4)] += 1
        if options and len(examples) < 6:
            examples.append(
                f"  row {index} (layout {layout}): could add "
                f"{ {k: v for k, v in sorted(options.items())[:3]} }"
            )

    total = sum(yield_counts.values())
    print()
    print(f"of the {total} cutting scenes:")
    for count, n in sorted(yield_counts.items()):
        label = f"{count} stable spare asset(s)" if count < 4 else "4 or more"
        print(f"  {label:28s} {n:4d}  {100*n/total:5.1f}%")
    usable = total - yield_counts[0]
    print()
    print(f"upper bound on this route: {usable} of {total} cutting scenes "
          f"({100*usable/total:.1f}%)")
    for line in examples:
        print(line)


if __name__ == "__main__":
    main()
