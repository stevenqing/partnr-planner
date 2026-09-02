#!/usr/bin/env python3
"""Do the null entries in init_pos name assets the layout actually contains?

Every cutting scene holds three or four live assets and its own task consumes all of
them, so no spare asset exists to carry a second goal, and a recombination split
cannot be cut from the scenes as they stand. The remaining route is to give one of
the fifty null assets a position -- but only if that asset really is in the layout,
or the scene would stop matching its own image.

The dataset can answer that itself. Rows carry a layout_id, and an asset left null in
one row may be given a real position in another row of the same layout. Where that
happens the asset is part of the layout and its position is the dataset's own, not an
invention. This measures how far that gets: how many assets each layout supplies
beyond a cutting task's own, and whether the positions agree across the rows that do
place them.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import MEMORY_PARQUET, SOURCE_PARQUET, load_manifest, native
from viki_amendment10_pairs import live
from viki_amendment10_recombine import CUTTING, delivery_goals
from viki_amendment9_folds import train_family_by_index


def main() -> None:
    train = pd.read_parquet(MEMORY_PARQUET)
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()
    families = train_family_by_index()
    catalogue = delivery_goals(train, families)

    # Every row of either split contributes what it knows about its layout.
    pool: Dict[Any, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    rows_per_layout: Counter = Counter()
    for frame in (train, test):
        for i in range(len(frame)):
            truth = native(frame.iloc[i].to_dict())["reward_model"]["ground_truth"]
            layout = truth.get("layout_id")
            rows_per_layout[layout] += 1
            for name, positions in (truth.get("init_pos") or {}).items():
                if positions is None:
                    continue
                if name.startswith("R") and name[1:].isdigit():
                    continue
                pool[layout][name].update(
                    positions if isinstance(positions, list) else [positions]
                )

    print(f"layouts seen: {len(pool)}   rows contributing: {sum(rows_per_layout.values())}")
    sizes = Counter(len(assets) for assets in pool.values())
    print(f"assets a layout is known to place: {sorted(sizes.items())}")

    agree = Counter()
    for layout, assets in pool.items():
        for name, positions in assets.items():
            agree["one position" if len(positions) == 1 else "several positions"] += 1
    print(f"position agreement across rows of a layout: {dict(agree)}")

    print()
    print("for the 297 cutting scenes, what the layout could add:")
    added: Counter = Counter()
    examples: List[str] = []
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
        known = pool.get(layout, {})
        spare = {
            name.rsplit("_", 1)[0]: sorted(positions)
            for name, positions in known.items()
            if name.rsplit("_", 1)[0] in catalogue
            and name.rsplit("_", 1)[0] not in assets
            and name.rsplit("_", 1)[0] not in used
        }
        added[min(len(spare), 5)] += 1
        if spare and len(examples) < 5:
            examples.append(
                f"  row {index} (layout {layout}): live={sorted(assets)} "
                f"-> layout also places {sorted(spare)[:6]}"
            )
    for count, n in sorted(added.items()):
        label = f"{count} spare deliverable(s)" if count < 5 else "5 or more"
        print(f"  {label:26s} {n:4d} scenes")
    print()
    for line in examples:
        print(line)


if __name__ == "__main__":
    main()
