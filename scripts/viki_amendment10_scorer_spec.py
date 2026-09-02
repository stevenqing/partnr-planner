#!/usr/bin/env python3
"""Read the official scorer's goal semantics before authoring any goal.

eval_single reads goal_constraints, init_pos, robots and temporal_constraints, and
does not read time_steps at all: scoring is goal satisfaction, not plan matching.
That is what makes a recombination split constructible -- two families' goals can be
conjoined and the benchmark's own checker will grade the result. But conjoining is
only sound if the nesting semantics are known, so the checker is read rather than
guessed, together with the shapes the dataset actually uses for each field.
"""

from __future__ import annotations

import inspect
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import MEMORY_PARQUET, native
from viki_amendment9_folds import train_family_by_index


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    print("=" * 74)
    print("eval_single")
    print("=" * 74)
    print(inspect.getsource(scorer.eval_single))
    for name in ("check_goal", "check_constraint", "transform_actions", "simulate"):
        function = getattr(scorer, name, None)
        if function is not None and name != "transform_actions":
            try:
                print("=" * 74)
                print(name)
                print("=" * 74)
                print(inspect.getsource(function))
            except Exception:
                pass

    train = pd.read_parquet(MEMORY_PARQUET)
    families = train_family_by_index()
    print("=" * 74)
    print("shapes the dataset uses")
    print("=" * 74)
    outer, inner, types, statuses = Counter(), Counter(), Counter(), Counter()
    temporal = Counter()
    examples = {}
    for i in range(len(train)):
        gt = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"]
        goals = gt.get("goal_constraints") or []
        outer[len(goals)] += 1
        for group in goals:
            inner[len(group)] += 1
            for item in group:
                types[item.get("type")] += 1
                statuses[tuple(sorted((item.get("status") or {}).keys()))] += 1
        temporal[len(gt.get("temporal_constraints") or [])] += 1
        family = families.get(i)
        if family not in examples and len(goals) and len(goals[0]) > 1:
            examples[family] = (goals, gt.get("temporal_constraints"))
    print(f"goal_constraints outer length: {sorted(outer.items())}")
    print(f"goal_constraints inner length: {sorted(inner.items())}")
    print(f"constraint types: {types.most_common()}")
    print(f"status key sets: {statuses.most_common(6)}")
    print(f"temporal_constraints length: {sorted(temporal.items())}")

    print()
    for family in (
        "clear_table_with_two_robots_and_put_in_cabinet",
        "cut_two_fruits_on_board",
    ):
        if family in examples:
            goals, temporal_constraints = examples[family]
            print("-" * 74)
            print(family)
            print("  goal_constraints:", json.dumps(goals)[:700])
            print("  temporal_constraints:", json.dumps(temporal_constraints)[:300])


if __name__ == "__main__":
    main()
