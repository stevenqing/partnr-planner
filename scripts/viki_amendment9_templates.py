#!/usr/bin/env python3
"""Is there a coordination-template layer that transfers where plans do not?

The compositional check found 7196 training rows carrying only 381 distinct plans,
and the held-out-family folds found that none of those plans crosses a family
boundary. That combination is only interesting if the regularity lives one level up:
if the plans are instances of a much smaller set of agent-indexed, step-indexed
templates, and if those templates DO cross families, then a memory that stored
templates rather than plans would have something to say about a held-out family,
where a memory that stores plans has nothing.

That is a precondition, not a result, and it is measurable offline. A template is a
plan with object names replaced by O1, O2, ... in order of first appearance. Robot
ids are kept: VIKI fixes R1 as the evaluated robot and the robot types differ, so
the id carries real role information rather than an arbitrary label.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import (
    MEMORY_PARQUET,
    OUTPUT_DIR,
    SOURCE_PARQUET,
    load_manifest,
    native,
)
from viki_amendment9_folds import folds, rows_of, train_family_by_index


def templatise(steps: Any) -> Tuple:
    """The plan with object names abstracted to variables, structure untouched."""
    variables: Dict[str, str] = {}
    out: List[Tuple] = []
    for step in steps:
        row = []
        for robot in sorted(step["actions"]):
            action = step["actions"][robot]
            if action is None:
                row.append((robot, "idle", ""))
                continue
            verb = str(action[0])
            target = str(action[1]) if len(action) > 1 else ""
            if target and target not in variables:
                variables[target] = f"O{len(variables) + 1}"
            row.append((robot, verb, variables.get(target, "")))
        out.append(tuple(row))
    return tuple(out)


def plan_key(steps: Any) -> Tuple:
    return tuple(
        tuple(
            (robot, json.dumps(step["actions"][robot], sort_keys=True))
            for robot in sorted(step["actions"])
        )
        for step in steps
    )


def main() -> None:
    manifest = load_manifest()
    test = pd.read_parquet(SOURCE_PARQUET)
    train = pd.read_parquet(MEMORY_PARQUET)
    families = train_family_by_index()

    train_plans: Dict[Tuple, Set[str]] = defaultdict(set)
    train_templates: Dict[Tuple, Set[str]] = defaultdict(set)
    for i in range(len(train)):
        steps = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"][
            "time_steps"
        ]
        family = families.get(i)
        train_plans[plan_key(steps)].add(family)
        train_templates[templatise(steps)].add(family)

    print(f"training rows                     {len(train)}")
    print(f"distinct plans                    {len(train_plans)}")
    print(f"distinct coordination templates   {len(train_templates)}")
    spread = Counter(len(f) for f in train_templates.values())
    print(f"templates by number of families:  {sorted(spread.items())}")
    print(
        f"templates seen in >1 family:      "
        f"{sum(n for k, n in spread.items() if k > 1)}"
    )

    per_row_family = {i: f for f in folds() for i in rows_of(f)}
    inside = Counter()
    for index in sorted(manifest):
        steps = native(test.iloc[index].to_dict())["reward_model"]["ground_truth"][
            "time_steps"
        ]
        held = per_row_family.get(index)
        plan_homes = train_plans.get(plan_key(steps), set())
        tmpl_homes = train_templates.get(templatise(steps), set())
        if not tmpl_homes:
            inside["template absent from the bank entirely"] += 1
        elif tmpl_homes - {held}:
            inside["template also demonstrated by another family"] += 1
        else:
            inside["template only in the row's own family"] += 1
        if plan_homes - {held}:
            inside["(plan itself survives the fold)"] += 1

    print()
    print("For each of the 924 evaluated rows, after its own family is held out:")
    for key, count in inside.most_common():
        print(f"  {key:46s} {count:4d}  {100*count/len(manifest):5.1f}%")

    out = OUTPUT_DIR / "coordination_templates.json"
    out.write_text(
        json.dumps(
            {
                "train_rows": len(train),
                "distinct_plans": len(train_plans),
                "distinct_templates": len(train_templates),
                "template_family_spread": dict(sorted(spread.items())),
            },
            indent=2,
        )
    )
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
