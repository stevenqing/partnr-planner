#!/usr/bin/env python3
"""How strict does the abstraction have to be before coordination crosses families?

The first pass abstracted object names only and found 24 templates, 23 of them
confined to a single family. That could be the abstraction being too tight rather
than the benchmark being that separated -- keeping the exact step count and the
object-variable identity means two plans with the same coordination shape but a
different number of items look different. So the same question is asked at four
increasingly loose levels, ending at one that keeps almost nothing but who acts
when. If coordination still does not cross families at the loosest level, no
abstraction of this trajectory format can transfer, and the held-out-family split
is unwinnable by any memory rather than merely hard for ours.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import (
    MEMORY_PARQUET,
    SOURCE_PARQUET,
    load_manifest,
    native,
)
from viki_amendment9_folds import folds, rows_of, train_family_by_index


def levels(steps: Any) -> Dict[str, Tuple]:
    robots = sorted({r for step in steps for r in step["actions"]})
    variables: Dict[str, str] = {}
    typed, skeleton = [], []
    per_robot: Dict[str, list] = {r: [] for r in robots}
    for step in steps:
        typed_row, skel_row = [], []
        for robot in robots:
            action = step["actions"].get(robot)
            if action is None:
                typed_row.append((robot, "idle", ""))
                skel_row.append((robot, "idle"))
                continue
            verb = str(action[0])
            target = str(action[1]) if len(action) > 1 else ""
            if target and target not in variables:
                variables[target] = f"O{len(variables) + 1}"
            typed_row.append((robot, verb, variables.get(target, "")))
            skel_row.append((robot, verb))
            per_robot[robot].append(verb)
        typed.append(tuple(typed_row))
        skeleton.append(tuple(skel_row))
    return {
        # object identity + step count + who acts when
        "typed": tuple(typed),
        # step count + who acts when, objects dropped
        "skeleton": tuple(skeleton),
        # each robot's own verb order, step alignment dropped
        "per_robot": tuple(sorted((r, tuple(v)) for r, v in per_robot.items())),
        # only the shape of parallelism: how many robots act at each step
        "parallelism": tuple(
            sum(1 for a in step["actions"].values() if a is not None) for step in steps
        ),
    }


def main() -> None:
    manifest = load_manifest()
    test = pd.read_parquet(SOURCE_PARQUET)
    train = pd.read_parquet(MEMORY_PARQUET)
    families = train_family_by_index()
    per_row_family = {i: f for f in folds() for i in rows_of(f)}

    names = ("typed", "skeleton", "per_robot", "parallelism")
    banks: Dict[str, Dict[Tuple, Set[str]]] = {n: defaultdict(set) for n in names}
    for i in range(len(train)):
        steps = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"][
            "time_steps"
        ]
        family = families.get(i)
        for name, key in levels(steps).items():
            banks[name][key].add(family)

    print(f"{'level':14s} {'distinct':>9s} {'in >1 family':>13s} {'max families':>13s}")
    for name in names:
        bank = banks[name]
        spread = Counter(len(f) for f in bank.values())
        print(
            f"{name:14s} {len(bank):9d} "
            f"{sum(n for k, n in spread.items() if k > 1):13d} "
            f"{max(spread):13d}"
        )

    print()
    print("Of the 924 evaluated rows, how many keep a match after their family is held out:")
    for name in names:
        bank = banks[name]
        survives = 0
        absent = 0
        for index in sorted(manifest):
            steps = native(test.iloc[index].to_dict())["reward_model"]["ground_truth"][
                "time_steps"
            ]
            homes = bank.get(levels(steps)[name], set())
            if not homes:
                absent += 1
            elif homes - {per_row_family.get(index)}:
                survives += 1
        print(
            f"  {name:14s} {survives:4d}/924 = {100*survives/924:5.1f}%"
            f"   (absent from the bank entirely: {absent})"
        )


if __name__ == "__main__":
    main()
