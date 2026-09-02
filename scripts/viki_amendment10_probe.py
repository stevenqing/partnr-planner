#!/usr/bin/env python3
"""Build one recombination instance and make the official scorer accept it.

Before generating a split, one instance has to survive the benchmark's own checker.
The construction takes a donor scene and gives it two goals drawn from different
families -- one asset into the cabinet, which the scorer treats as an isolated
container and so cannot be filled without Open[cabinet], and the knife activated
after a fruit reaches the board, which is the cutting families' ordering. Neither
pairing occurs in any of the 7196 training plans.

The reference plan is not asserted, it is searched: candidate step lists are handed
to eval_single and only what it accepts is kept, shortest first. That matters for
fairness as much as for validity, because scoring bounds the prediction by the
reference length, so a padded reference would quietly make the split easy.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import SOURCE_PARQUET, load_manifest, native

SEED = 20240617
BOARD = "wooden cutting board"


def act(verb: str, target: str) -> List[str]:
    return [verb, target]


def fetch_chain(robot: str, item: str) -> List[List[str]]:
    return [act("Move", item), act("Reach", item), act("Grasp", item)]


def interleave(
    left: List[List[str]], right: List[List[str]], robots=("R1", "R2")
) -> List[Dict[str, Any]]:
    """Lay two per-robot action lists side by side, one step per position."""
    steps = []
    for position in range(max(len(left), len(right))):
        actions = {
            robots[0]: left[position] if position < len(left) else None,
            robots[1]: right[position] if position < len(right) else None,
        }
        steps.append({"actions": actions, "step": position + 1})
    return steps


def candidate_plans(store: str, cut: str) -> List[List[Dict[str, Any]]]:
    """R1 takes the cabinet errand, R2 the cutting errand, and the mirror image.
    Both orders are offered because which robot suits which errand is exactly the
    allocation a solver is supposed to work out, and the scorer decides whether a
    given assignment is feasible."""
    plans = []
    for first, second in (("R1", "R2"), ("R2", "R1")):
        cabinet_branch = fetch_chain(first, store) + [
            act("Move", "cabinet"),
            act("Reach", "cabinet"),
            act("Open", "cabinet"),
            act("Place", "cabinet"),
        ]
        cutting_branch = fetch_chain(second, cut) + [
            act("Move", BOARD),
            act("Place", BOARD),
            act("Move", "knife"),
            act("Reach", "knife"),
            act("Grasp", "knife"),
            act("Interact", "knife"),
        ]
        order = (first, second)
        plans.append(interleave(cabinet_branch, cutting_branch, order))
    return plans


def build(row: Dict[str, Any], store: str, cut: str) -> Dict[str, Any]:
    truth = json.loads(json.dumps(row["reward_model"]["ground_truth"]))
    truth["goal_constraints"] = [
        [
            {
                "is_satisfied": True,
                "name": store,
                "status": {"is_activated": None, "pos.name": "cabinet"},
                "type": "asset",
            }
        ],
        [
            {
                "is_satisfied": True,
                "name": "knife",
                "status": {"is_activated": True, "pos.name": None},
                "type": "asset",
            }
        ],
    ]
    truth["temporal_constraints"] = [
        [
            [
                {
                    "is_satisfied": True,
                    "name": cut,
                    "status": {"is_activated": None, "pos.name": BOARD},
                    "type": "asset",
                }
            ],
            [
                {
                    "is_satisfied": True,
                    "name": "knife",
                    "status": {"is_activated": True, "pos.name": None},
                    "type": "asset",
                }
            ],
        ]
    ]
    truth["description"] = (
        f"Put the {store} away in the cabinet, and set the {cut} on the "
        f"{BOARD} and cut it with the knife."
    )
    truth["task_name"] = "recombine_cabinet_and_cut"
    return truth


def accepts(scorer, plan: List[Dict[str, Any]], truth: Dict[str, Any]) -> bool:
    transformed = scorer.transform_actions(json.loads(json.dumps(plan)))
    if not transformed:
        return False
    globals_ = scorer.eval_single.__globals__
    original = globals_["random"]
    try:
        globals_["random"] = random.Random(SEED)
        return bool(scorer.eval_single(transformed, truth))
    except Exception as error:
        print(f"    eval_single raised: {error}")
        return False
    finally:
        globals_["random"] = original


def assets_of(truth: Dict[str, Any]) -> set:
    return {
        key.rsplit("_", 1)[0]
        for key in truth["init_pos"]
        if not (key.startswith("R") and key[1:].isdigit())
    }


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()

    movable = ["apple", "pear", "peach", "banana", "spoon", "fork", "bread"]
    tried = 0
    for index in sorted(manifest):
        row = native(test.iloc[index].to_dict())
        truth = row["reward_model"]["ground_truth"]
        present = assets_of(truth)
        if not {"cabinet", "knife", BOARD} <= present:
            continue
        options = [item for item in movable if item in present]
        if len(options) < 2:
            continue
        store, cut = options[0], options[1]
        tried += 1
        built = build(row, store, cut)
        print(f"donor row {index}  scene has cabinet, knife, board")
        print(f"  store={store!r} into the cabinet   cut={cut!r} on the board")
        for number, plan in enumerate(candidate_plans(store, cut), 1):
            ok = accepts(scorer, plan, built)
            print(f"  candidate {number}: {len(plan)} steps -> "
                  f"{'ACCEPTED' if ok else 'rejected'}")
            if ok:
                print()
                print("  reference plan:")
                for step in plan:
                    cells = ", ".join(
                        f"{robot} {json.dumps(action) if action else 'idle'}"
                        for robot, action in step["actions"].items()
                    )
                    print(f"    step {step['step']}: {cells}")
                print()
                print("  goal_constraints:")
                print("   ", json.dumps(built["goal_constraints"]))
                print("  temporal_constraints:")
                print("   ", json.dumps(built["temporal_constraints"]))
                print()
                print(f"  description: {built['description']}")
                return
        if tried >= 4:
            print("\nno candidate accepted in the first four donor scenes")
            return
    print(f"\nscenes examined: {tried}")


if __name__ == "__main__":
    main()
