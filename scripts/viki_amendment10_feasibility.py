#!/usr/bin/env python3
"""Can VIKI-L2 be extended into a cross-task recombination split?

Every structural measurement says the benchmark has no recombination: 98 action
primitives, 381 distinct plans, 24 coordination templates, and 860 of the 924
evaluated rows use a primitive set contained in one single family's repertoire. So
the compositional split has to be built rather than selected.

The construction worth building is a scene that already contains the objects of two
families, given a task that needs both families' coordination patterns over disjoint
objects -- put the pumpkin in the cabinet AND cut the pear on the board. Neither
family demonstrates that pair, so a memory has to combine two entries rather than
retrieve one.

Three things have to hold first, and all three are checkable offline:
  G1  what the official scorer validates besides the step list, since a synthetic
      row that cannot be scored is worse than no row at all
  G2  whether scenes exist holding two families' object sets at once
  G3  whether the two patterns can be laid out without both demanding the same robot
"""

from __future__ import annotations

import inspect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import MEMORY_PARQUET, SOURCE_PARQUET, load_manifest, native
from viki_amendment9_folds import train_family_by_index


def objects_of(steps) -> Set[str]:
    out = set()
    for step in steps:
        for action in step["actions"].values():
            if action is not None and len(action) > 1:
                out.add(str(action[1]))
    return out


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    train = pd.read_parquet(MEMORY_PARQUET)
    test = pd.read_parquet(SOURCE_PARQUET)
    families = train_family_by_index()

    print("=" * 74)
    print("G1  what a row carries, and what the scorer reads from it")
    print("=" * 74)
    row = native(test.iloc[0].to_dict())
    print(f"top-level keys: {sorted(row)}")
    gt = row["reward_model"]["ground_truth"]
    print(f"ground_truth keys: {sorted(gt)}")
    for key, value in sorted(gt.items()):
        if key == "time_steps":
            continue
        text = json.dumps(value)
        print(f"  {key}: {text[:300]}")
    print()
    try:
        print("eval_single signature:", inspect.signature(scorer.eval_single))
        source = inspect.getsource(scorer.eval_single)
        print("eval_single reads these ground_truth fields:")
        for field in sorted(gt):
            if f'"{field}"' in source or f"'{field}'" in source:
                print(f"    {field}")
        print(f"(eval_single is {len(source.splitlines())} lines)")
    except Exception as error:  # the scorer is loaded from source, not imported
        print(f"could not introspect eval_single: {error}")

    print()
    print("=" * 74)
    print("G2  do scenes already hold two families' object sets?")
    print("=" * 74)
    repertoire: Dict[str, Set[str]] = defaultdict(set)
    for i in range(len(train)):
        steps = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"][
            "time_steps"
        ]
        repertoire[families.get(i)] |= objects_of(steps)
    print("objects each family manipulates:")
    for family, objects in sorted(repertoire.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(objects):3d}  {family}")

    manifest = load_manifest()
    scene_objects: Dict[int, Set[str]] = {}
    for index in sorted(manifest):
        gt_row = native(test.iloc[index].to_dict())["reward_model"]["ground_truth"]
        scene_objects[index] = objects_of(gt_row["time_steps"])

    pair_support: Counter = Counter()
    for index, present in scene_objects.items():
        for a, objects_a in repertoire.items():
            for b, objects_b in repertoire.items():
                if a >= b:
                    continue
                if objects_a & present and objects_b & present:
                    pair_support[(a, b)] += 1
    print()
    print("family pairs whose objects both appear in an evaluated scene (top 12):")
    for (a, b), count in pair_support.most_common(12):
        overlap = len(repertoire[a] & repertoire[b])
        print(f"  {count:4d} rows   {a} + {b}   (shared objects: {overlap})")

    print()
    print("=" * 74)
    print("G3  do two families' patterns compete for the same robot?")
    print("=" * 74)
    load: Dict[str, Counter] = defaultdict(Counter)
    for i in range(len(train)):
        steps = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"][
            "time_steps"
        ]
        for step in steps:
            for robot, action in step["actions"].items():
                if action is not None:
                    load[families.get(i)][robot] += 1
    print("share of actions each robot performs, per family:")
    for family, counts in sorted(load.items()):
        total = sum(counts.values()) or 1
        share = "  ".join(
            f"{robot} {100*counts[robot]/total:4.0f}%" for robot in sorted(counts)
        )
        print(f"  {family:46s} {share}")


if __name__ == "__main__":
    main()
