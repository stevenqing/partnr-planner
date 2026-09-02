#!/usr/bin/env python3
"""Which check rejects a hand-authored recombination plan?

eval_single builds its own judge and drops it, so the error code it records is not
visible from outside. The body is short and its source is available, so it is
re-run here with the judge kept, and the branches are tried separately as well as
together. Knowing whether the refusal is INVALID_COMMAND, NOT_FOUND_ENTITY,
ACTION_NOT_FEASIBLE, ACTION_NOT_COMPATIBLE or a failed constraint decides whether the
construction needs a different plan or a different design.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import SOURCE_PARQUET, load_manifest, native
from viki_amendment10_probe import (
    BOARD,
    SEED,
    act,
    assets_of,
    build,
    fetch_chain,
    interleave,
)


def judge(scorer, plan: List[Dict[str, Any]], truth: Dict[str, Any]):
    """eval_single's body, with the judge kept so its error code can be read."""
    globals_ = scorer.eval_single.__globals__
    Eval = globals_["Eval"]
    filter_none_values = globals_["filter_none_values"]
    containers = globals_["CONTAINER_ASSETS"]
    Position = globals_.get("Position")

    transformed = scorer.transform_actions(json.loads(json.dumps(plan)))
    if not transformed:
        return None, "TRANSFORM_EMPTY"

    truth = filter_none_values(json.loads(json.dumps(truth)))
    metadata = {"agents": {}, "assets": {}}
    for robot_id, robot_type in truth["robots"].items():
        metadata["agents"][robot_id] = {"type": robot_type, "pos": {"name": robot_id}}
    generator = random.Random(SEED)
    for asset_name, positions in truth["init_pos"].items():
        if asset_name.startswith("R") and asset_name[1:].isdigit():
            continue
        asset_type = asset_name.rsplit("_", 1)[0]
        metadata["assets"][asset_type] = {
            "pos": {"name": generator.choice(positions)}
        }
        if asset_type in containers:
            metadata["assets"][asset_type]["params"] = {
                "is_container": True,
                "position_kwargs": {
                    "name": asset_type,
                    "isolated": asset_type == "cabinet",
                },
            }
    metadata["goal_constraints"] = truth["goal_constraints"]
    metadata["temporal_constraints"] = truth["temporal_constraints"]

    judger = Eval()
    judger.set_env(metadata)
    ok = judger.eval(transformed)
    return ok, getattr(judger, "error_desc_code", None)


def solo(actions: List[List[str]], robot: str) -> List[Dict[str, Any]]:
    return [
        {"actions": {robot: action}, "step": n + 1} for n, action in enumerate(actions)
    ]


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()

    for index in sorted(manifest):
        row = native(test.iloc[index].to_dict())
        truth = row["reward_model"]["ground_truth"]
        present = assets_of(truth)
        if not {"cabinet", "knife", BOARD, "apple", "pear"} <= present:
            continue
        store, cut = "apple", "pear"
        built = build(row, store, cut)
        print(f"donor row {index}: store={store} cut={cut}")
        print(f"  robots: {json.dumps(truth['robots'])}")

        cabinet_branch = fetch_chain("R1", store) + [
            act("Move", "cabinet"),
            act("Reach", "cabinet"),
            act("Open", "cabinet"),
            act("Place", "cabinet"),
        ]
        cutting_solo = fetch_chain("R2", cut) + [
            act("Move", BOARD),
            act("Place", BOARD),
            act("Move", "knife"),
            act("Reach", "knife"),
            act("Grasp", "knife"),
            act("Interact", "knife"),
        ]
        # The real cutting families split this errand over two robots; that split is
        # tried too, since one robot doing all of it may simply not be feasible.
        cut_fetch = fetch_chain("R1", cut) + [act("Move", BOARD), act("Place", BOARD)]
        knife_branch = fetch_chain("R2", "knife") + [
            act("Move", BOARD),
            act("Reach", BOARD),
            act("Interact", "knife"),
        ]

        trials = {
            "cabinet errand alone, one robot": solo(cabinet_branch, "R1"),
            "cutting errand alone, one robot": solo(cutting_solo, "R2"),
            "cutting errand alone, two robots": interleave(cut_fetch, knife_branch),
            "both errands, parallel branches": interleave(
                cabinet_branch, cutting_solo
            ),
            "both errands, cabinet then cutting": solo(cabinet_branch, "R1")
            + [
                {"actions": {"R1": a, "R2": b}, "step": len(cabinet_branch) + n + 1}
                for n, (a, b) in enumerate(
                    zip(
                        cut_fetch + [None] * len(knife_branch),
                        knife_branch + [None] * len(cut_fetch),
                    )
                )
                if a or b
            ],
        }
        for label, plan in trials.items():
            ok, code = judge(scorer, plan, built)
            print(f"  {label:38s} {len(plan):2d} steps -> {ok}  {code or ''}")
        return


if __name__ == "__main__":
    main()
