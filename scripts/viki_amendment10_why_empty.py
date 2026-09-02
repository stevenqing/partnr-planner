#!/usr/bin/env python3
"""Why does no cutting scene yield a spare delivery asset?

The pair analysis says 108 scenes support cut_fruit and single_move together, and
the composer then found none. The composer adds two conditions the pair analysis did
not: the delivered asset must not be one the cutting plan already touches, and the
delivery target must not be either. single_move's catalogue is largely fruit, which
is exactly what the cutting task consumes, so the spare requirement may be removing
everything. This prints the intermediate sets for a handful of cutting scenes rather
than reasoning about them.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import MEMORY_PARQUET, SOURCE_PARQUET, load_manifest, native
from viki_amendment9_folds import train_family_by_index
from viki_amendment10_pairs import live
from viki_amendment10_recombine import CUTTING, DELIVERY, delivery_goals


def main() -> None:
    train = pd.read_parquet(MEMORY_PARQUET)
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()
    families = train_family_by_index()
    catalogue = delivery_goals(train, families)
    print(f"single_move catalogue ({len(catalogue)}): {sorted(catalogue)}")
    print()

    tally: Counter = Counter()
    shown = 0
    for index in sorted(manifest):
        truth = native(test.iloc[index].to_dict())["reward_model"]["ground_truth"]
        if truth.get("task_name") not in CUTTING:
            continue
        assets = live(truth)
        used = {
            str(a[1])
            for step in truth["time_steps"]
            for a in step["actions"].values()
            if a is not None and len(a) > 1
        }
        deliverable = {i for i in catalogue if i in assets}
        spare = deliverable - used
        targets = {
            t
            for i in spare
            for t, _ in [catalogue[i].most_common(1)[0]]
        }
        tally[
            "has a spare deliverable" if spare else "no spare deliverable"
        ] += 1
        if spare:
            tally["spare and target unused"] += bool(targets - used)
        if shown < 6:
            shown += 1
            print(f"row {index}  {truth['task_name']}")
            print(f"  live assets ({len(assets)}): {sorted(assets)}")
            print(f"  used by the donor plan: {sorted(used)}")
            print(f"  deliverable and live: {sorted(deliverable)}")
            print(f"  spare after removing used: {sorted(spare)}")
            print(f"  their usual targets: {sorted(targets)}")
            print()
    print(dict(tally))


if __name__ == "__main__":
    main()
