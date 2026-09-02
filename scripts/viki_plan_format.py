#!/usr/bin/env python3
"""One place for the plan-format rules every VIKI-L2 tool needs.

The dataset writes `null` for a robot that stands still, and the official
transform_actions discards any step still carrying it, so a plan handed over raw
transforms to nothing and scores zero. That has now been rediscovered three times in
three separate tools -- the tolerant metric, the grounding module, and the
recombination judge -- each time as a silent zero rather than an error. It lives here
now so the next tool inherits the fix instead of the bug.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

SEED = 20240617


def strip_idle(plan: Any) -> Any:
    """A deep copy of the plan with idle robots removed from every step."""
    plan = json.loads(json.dumps(plan))
    if not isinstance(plan, list):
        return plan
    for step in plan:
        if isinstance(step, dict) and isinstance(step.get("actions"), dict):
            step["actions"] = {
                robot: action
                for robot, action in step["actions"].items()
                if action is not None
            }
    return plan


def transform(scorer, plan: Any):
    """transform_actions, with idle robots dropped first."""
    return scorer.transform_actions(strip_idle(plan))


def evaluate(
    scorer, plan: Any, truth: Dict[str, Any]
) -> Tuple[Optional[bool], Optional[str]]:
    """Run the official checker and return its verdict with its error code.

    eval_single builds its own judge and drops it, so its reason for refusing is
    invisible from outside; the body is reproduced here only to keep that reason.
    The environment is built exactly as eval_single builds it, including the seeded
    random.choice over each asset's candidate positions.
    """
    globals_ = scorer.eval_single.__globals__
    Eval = globals_["Eval"]
    filter_none_values = globals_["filter_none_values"]
    containers = globals_["CONTAINER_ASSETS"]

    transformed = transform(scorer, plan)
    if not transformed:
        return None, "TRANSFORM_EMPTY"

    data = filter_none_values(json.loads(json.dumps(truth)))
    metadata: Dict[str, Any] = {"agents": {}, "assets": {}}
    for robot_id, robot_type in data["robots"].items():
        metadata["agents"][robot_id] = {"type": robot_type, "pos": {"name": robot_id}}
    generator = random.Random(SEED)
    for name, positions in data["init_pos"].items():
        if name.startswith("R") and name[1:].isdigit():
            continue
        kind = name.rsplit("_", 1)[0]
        metadata["assets"][kind] = {"pos": {"name": generator.choice(positions)}}
        if kind in containers:
            metadata["assets"][kind]["params"] = {
                "is_container": True,
                "position_kwargs": {"name": kind, "isolated": kind == "cabinet"},
            }
    metadata["goal_constraints"] = data["goal_constraints"]
    metadata["temporal_constraints"] = data["temporal_constraints"]

    judger = Eval()
    judger.set_env(metadata)
    try:
        ok = judger.eval(transformed)
    except Exception as error:
        return None, f"RAISED {type(error).__name__}: {error}"
    return ok, getattr(judger, "error_desc_code", None)


def steps_of(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Renumber a step list so the step field matches its position."""
    for number, step in enumerate(plan, 1):
        step["step"] = number
    return plan
