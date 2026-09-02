#!/usr/bin/env python3
"""Compose a recombination instance from a verified donor row.

The first attempt authored a plan for an arbitrary scene and the scorer refused it
twice over: the scene's robots were a fixed arm and a quadruped, which cannot Move
and Grasp the way the plan assumed, and eval_single calls filter_none_values first,
so the many assets whose init_pos is null never enter the environment at all -- the
chosen scene had no knife to activate.

Both constraints come free if the donor is a row of one of the families being
combined. A cut_fruit row has robots that can cut, a knife and a board that really
exist, and a ground-truth plan the scorer already accepts. The recombination then
adds the other family's goal -- an asset into the cabinet, which the scorer treats
as an isolated container -- and extends that verified plan rather than inventing one.

Nothing here is asserted: the donor's own plan is re-checked, each extension is
checked, and only what eval_single accepts is kept.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import SOURCE_PARQUET, load_manifest, native

SEED = 20240617
BOARD = "wooden cutting board"
CUTTING = ("cut_fruit_on_board", "cut_two_fruits_on_board")
STORABLE = ("spoon", "fork", "bowl", "cup", "bottle", "banana", "apple", "peach", "pear")


def live_assets(truth: Dict[str, Any]) -> Dict[str, Any]:
    """Assets that survive filter_none_values, i.e. the ones the env will hold."""
    out = {}
    for name, positions in (truth.get("init_pos") or {}).items():
        if name.startswith("R") and name[1:].isdigit():
            continue
        if positions is None:
            continue
        out[name.rsplit("_", 1)[0]] = positions
    return out


def judge(scorer, plan, truth) -> Tuple[Optional[bool], Optional[str]]:
    """Kept as a thin alias: the format rules now live in viki_plan_format, so a
    fix there reaches every caller instead of one."""
    from viki_plan_format import evaluate

    return evaluate(scorer, plan, truth)


def cabinet_branch(item: str) -> List[List[str]]:
    return [
        ["Move", item],
        ["Reach", item],
        ["Grasp", item],
        ["Move", "cabinet"],
        ["Reach", "cabinet"],
        ["Open", "cabinet"],
        ["Place", "cabinet"],
    ]


def append_branch(
    steps: List[Dict[str, Any]], robot: str, actions: List[List[str]]
) -> List[Dict[str, Any]]:
    """Run the branch after the donor plan, on one robot, others idle."""
    robots = sorted({r for step in steps for r in step["actions"]})
    out = json.loads(json.dumps(steps))
    for offset, action in enumerate(actions):
        out.append(
            {
                "actions": {r: (action if r == robot else None) for r in robots},
                "step": len(steps) + offset + 1,
            }
        )
    return out


def compress(scorer, steps, truth, keep_prefix: int):
    """Pull the appended tail earlier, step by step, while the scorer still accepts.
    The reference plan bounds how long a prediction may be, so a padded reference
    would hand the split away; this keeps it near the shortest the checker allows."""
    best = steps
    for start in range(len(steps) - 1, keep_prefix, -1):
        merged = json.loads(json.dumps(best))
        if start >= len(merged):
            continue
        tail = merged.pop(start)
        target = merged[start - 1]
        clash = any(
            target["actions"].get(r) is not None and action is not None
            for r, action in tail["actions"].items()
        )
        if clash:
            continue
        for robot, action in tail["actions"].items():
            if action is not None:
                target["actions"][robot] = action
        for number, step in enumerate(merged, 1):
            step["step"] = number
        ok, _ = judge(scorer, merged, truth)
        if ok:
            best = merged
    return best


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()

    reasons: Counter = Counter()
    for index in sorted(manifest):
        row = native(test.iloc[index].to_dict())
        truth = row["reward_model"]["ground_truth"]
        if truth.get("task_name") not in CUTTING:
            reasons["donor is not a cutting row"] += 1
            continue
        assets = live_assets(truth)
        if "cabinet" not in assets:
            reasons["scene has no cabinet with a real position"] += 1
            continue
        used = {
            str(a[1])
            for step in truth["time_steps"]
            for a in step["actions"].values()
            if a is not None and len(a) > 1
        }
        spare = [item for item in STORABLE if item in assets and item not in used]
        if not spare:
            reasons["no spare storable asset"] += 1
            continue

        base_ok, base_code = judge(scorer, truth["time_steps"], truth)
        if not base_ok:
            reasons[f"donor plan itself rejected ({base_code})"] += 1
            continue

        item = spare[0]
        built = json.loads(json.dumps(truth))
        built["goal_constraints"] = list(truth["goal_constraints"]) + [
            [
                {
                    "is_satisfied": True,
                    "name": item,
                    "status": {"is_activated": None, "pos.name": "cabinet"},
                    "type": "asset",
                }
            ]
        ]
        built["task_name"] = "recombine_cut_and_cabinet"
        built["description"] = (
            truth["description"].rstrip(". ")
            + f", and put the {item} away in the cabinet."
        )

        robots = sorted(
            r
            for step in truth["time_steps"]
            for r, a in step["actions"].items()
            if a is not None
        )
        for robot in dict.fromkeys(robots):
            extended = append_branch(truth["time_steps"], robot, cabinet_branch(item))
            ok, code = judge(scorer, extended, built)
            if not ok:
                reasons[f"cabinet branch on {robot} rejected ({code})"] += 1
                continue
            tight = compress(scorer, extended, built, len(truth["time_steps"]) - 1)
            print("=" * 74)
            print(f"donor row {index}  task_name={truth['task_name']}")
            print(f"  robots: {json.dumps(truth['robots'])}")
            print(f"  donor plan accepted: {base_ok}  ({len(truth['time_steps'])} steps)")
            print(f"  spare asset for the cabinet goal: {item}")
            print(f"  extended plan accepted on {robot}: {len(extended)} steps")
            print(f"  after compression: {len(tight)} steps")
            print()
            print(f"  description: {built['description']}")
            print(f"  goal_constraints: {json.dumps(built['goal_constraints'])}")
            print(f"  temporal_constraints: {json.dumps(built['temporal_constraints'])}")
            print()
            print("  reference plan:")
            for step in tight:
                cells = ", ".join(
                    f"{r} {json.dumps(a) if a else 'idle'}"
                    for r, a in sorted(step["actions"].items())
                )
                print(f"    step {step['step']:2d}: {cells}")
            return
    print("no instance could be composed. why donors were skipped:")
    for reason, count in reasons.most_common():
        print(f"  {count:5d}  {reason}")


if __name__ == "__main__":
    main()
