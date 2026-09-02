#!/usr/bin/env python3
"""One verified cross-task recombination instance: a cutting task plus a delivery.

The pair analysis, once its two errors were corrected and it passed its own
self-checks, found that no two cooperative families ever share a scene: every
workable pair joins a cooperative family to single_move_asset_to_target, which needs
one slot from twenty-two options and so is live in all 924 scenes. cut_fruit_on_board
plus single_move is the largest such pair at 108 scenes.

The recombination gives two robots three jobs -- carry the fruit to the board, work
the knife, and deliver a third asset elsewhere -- where the bank holds the two-robot
cutting pattern and the one-robot delivery pattern only apart. Solving it needs a
reallocation that no stored trajectory demonstrates.

Nothing is asserted. The donor's own plan is scored first, every extension is scored,
and the reference is compressed only as far as the checker still accepts, because the
reference length is the bound a prediction is held to.
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

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import MEMORY_PARQUET, SOURCE_PARQUET, load_manifest, native
from viki_amendment9_folds import train_family_by_index
from viki_amendment10_pairs import live
from viki_plan_format import evaluate, steps_of

CUTTING = ("cut_fruit_on_board", "cut_two_fruits_on_board")
DELIVERY = "single_move_asset_to_target"


def delivery_goals(train, families) -> Dict[str, Counter]:
    """The (asset, target) pairs single_move actually asks for, so a generated goal
    is one the benchmark already poses rather than one invented for it."""
    pairs: Dict[str, Counter] = defaultdict(Counter)
    for i in range(len(train)):
        truth = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"]
        if families.get(i) != DELIVERY:
            continue
        for group in truth.get("goal_constraints") or []:
            for item in group:
                where = (item.get("status") or {}).get("pos.name")
                if where:
                    pairs[str(item["name"])][str(where)] += 1
    return pairs


def delivery_branch(item: str, target: str) -> List[List[str]]:
    return [
        ["Move", item],
        ["Reach", item],
        ["Grasp", item],
        ["Move", target],
        ["Place", target],
    ]


def append_branch(steps, robot: str, actions) -> List[Dict[str, Any]]:
    robots = sorted({r for step in steps for r in step["actions"]})
    out = json.loads(json.dumps(steps))
    for action in actions:
        out.append({"actions": {r: (action if r == robot else None) for r in robots}})
    return steps_of(out)


def compress(scorer, steps, truth, floor: int):
    best = steps
    improved = True
    while improved:
        improved = False
        for position in range(len(best) - 1, floor, -1):
            merged = json.loads(json.dumps(best))
            tail = merged.pop(position)
            target = merged[position - 1]
            if any(
                target["actions"].get(r) is not None and a is not None
                for r, a in tail["actions"].items()
            ):
                continue
            for robot, action in tail["actions"].items():
                if action is not None:
                    target["actions"][robot] = action
            ok, _ = evaluate(scorer, steps_of(merged), truth)
            if ok:
                best = merged
                improved = True
                break
    return best


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    train = pd.read_parquet(MEMORY_PARQUET)
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()
    families = train_family_by_index()
    catalogue = delivery_goals(train, families)
    print(f"single_move poses {len(catalogue)} distinct assets to deliver")

    skipped: Counter = Counter()
    for index in sorted(manifest):
        row = native(test.iloc[index].to_dict())
        truth = row["reward_model"]["ground_truth"]
        if truth.get("task_name") not in CUTTING:
            skipped["not a cutting donor"] += 1
            continue
        assets = live(truth)
        used = {
            str(a[1])
            for step in truth["time_steps"]
            for a in step["actions"].values()
            if a is not None and len(a) > 1
        }
        options = [
            (item, target)
            for item, targets in catalogue.items()
            if item in assets and item not in used
            for target, _ in [targets.most_common(1)[0]]
            if target not in used
        ]
        if not options:
            skipped["no spare delivery asset in this scene"] += 1
            continue
        base_ok, base_code = evaluate(scorer, truth["time_steps"], truth)
        if not base_ok:
            skipped[f"donor plan refused ({base_code})"] += 1
            continue

        item, target = sorted(options)[0]
        built = json.loads(json.dumps(truth))
        built["goal_constraints"] = list(truth["goal_constraints"]) + [
            [
                {
                    "is_satisfied": True,
                    "name": item,
                    "status": {"is_activated": None, "pos.name": target},
                    "type": "asset",
                }
            ]
        ]
        built["task_name"] = "recombine_cut_and_deliver"
        built["description"] = (
            truth["description"].rstrip(". ")
            + f", and move the {item} to the {target}."
        )

        actors = list(
            dict.fromkeys(
                r
                for step in truth["time_steps"]
                for r, a in step["actions"].items()
                if a is not None
            )
        )
        for robot in actors:
            extended = append_branch(
                truth["time_steps"], robot, delivery_branch(item, target)
            )
            ok, code = evaluate(scorer, extended, built)
            if not ok:
                skipped[f"delivery on {robot} refused ({code})"] += 1
                continue
            tight = compress(scorer, extended, built, len(truth["time_steps"]) - 1)
            final_ok, _ = evaluate(scorer, tight, built)
            print("=" * 74)
            print(f"donor row {index}   {truth['task_name']}")
            print(f"  robots: {json.dumps(truth['robots'])}")
            print(f"  donor plan accepted: {base_ok} ({len(truth['time_steps'])} steps)")
            print(f"  added goal: {item} -> {target}   carried by {robot}")
            print(f"  extended {len(extended)} steps -> compressed {len(tight)} steps"
                  f"   (accepted: {final_ok})")
            print()
            print(f"  description: {built['description']}")
            print(f"  goal_constraints: {json.dumps(built['goal_constraints'])}")
            print(f"  temporal_constraints: "
                  f"{json.dumps(built['temporal_constraints'])[:300]}")
            print()
            print("  reference plan:")
            for step in tight:
                cells = ", ".join(
                    f"{r} {json.dumps(a) if a else 'idle'}"
                    for r, a in sorted(step["actions"].items())
                )
                print(f"    step {step['step']:2d}: {cells}")
            return
    print("no instance composed. donors were skipped for:")
    for reason, count in skipped.most_common():
        print(f"  {count:5d}  {reason}")


if __name__ == "__main__":
    main()
