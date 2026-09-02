#!/usr/bin/env python3
"""The concrete goal shapes of the two families a recombination split would join.

The previous pass established the semantics: eval_single grades goal satisfaction,
assets are keyed by type with the trailing index stripped, init_pos values are
position lists sampled at random, and cabinet is special-cased as an isolated
container -- which is why clear_table cannot be solved without Open[cabinet]. It did
not print the per-family examples, because the selector required an inner group of
more than one and every inner group holds exactly one constraint.

This prints them, together with the temporal constraint shape, the container list,
and the judge's own evaluation code, so a generated goal can be written against what
the checker does rather than against an assumption about it.
"""

from __future__ import annotations

import inspect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import MEMORY_PARQUET, native
from viki_amendment9_folds import train_family_by_index

FAMILIES = (
    "clear_table_with_two_robots_and_put_in_cabinet",
    "cut_two_fruits_on_board",
    "cut_fruit_on_board",
    "toast_bread_and_set_plate",
)


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    globals_ = scorer.eval_single.__globals__
    containers = globals_.get("CONTAINER_ASSETS")
    print(f"CONTAINER_ASSETS = {containers}")
    judge = globals_.get("Eval")
    if judge is not None:
        for name in ("eval", "check_goal_constraints", "check_temporal_constraints"):
            method = getattr(judge, name, None)
            if method is None:
                continue
            try:
                print("=" * 74)
                print(f"Eval.{name}")
                print("=" * 74)
                print(inspect.getsource(method))
            except Exception as error:
                print(f"  (source unavailable: {error})")

    train = pd.read_parquet(MEMORY_PARQUET)
    families = train_family_by_index()
    seen: dict = defaultdict(list)
    temporal_shapes: Counter = Counter()
    for i in range(len(train)):
        gt = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"]
        family = families.get(i)
        for constraint in gt.get("temporal_constraints") or []:
            temporal_shapes[json.dumps(constraint, sort_keys=True)[:160]] += 1
        if family in FAMILIES and len(seen[family]) < 2:
            seen[family].append(gt)

    print("=" * 74)
    print("temporal_constraints, the distinct shapes that occur")
    print("=" * 74)
    for shape, count in temporal_shapes.most_common(8):
        print(f"  {count:5d}  {shape}")

    for family in FAMILIES:
        for gt in seen.get(family, []):
            print("=" * 74)
            print(f"{family}   ({len(gt['time_steps'])} steps)")
            print("=" * 74)
            print(f"  description: {gt['description']}")
            print(f"  robots: {json.dumps(gt['robots'])}")
            print(f"  idle_robots: {json.dumps(gt.get('idle_robots'))}")
            print(f"  goal_constraints: {json.dumps(gt['goal_constraints'])}")
            print(f"  temporal_constraints: {json.dumps(gt['temporal_constraints'])}")
            inventory = [k for k in gt["init_pos"] if not k.startswith("R")]
            print(f"  init_pos holds {len(inventory)} assets, e.g. {inventory[:12]}")
            sample_key = next(
                (k for k in gt["init_pos"] if k.startswith("cabinet")), inventory[0]
            )
            print(
                f"  init_pos[{sample_key!r}] = "
                f"{json.dumps(gt['init_pos'][sample_key])[:200]}"
            )
            break


if __name__ == "__main__":
    main()
