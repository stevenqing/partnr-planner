#!/usr/bin/env python3
"""Restore the agent-step structure the memory pipeline dropped.

The training parquet holds each plan step-aligned across robots, with idle robots
marked. The text traces the bank was built from lost that: step_number never
advanced, and the extractor concatenated agent_0_actions + agent_1_actions, so a
cooperation skill's stored action list presents parallel work as one sequential
chain. The renderer then showed the first five entries of it.

The link back is recoverable without re-extracting anything. Episode keys are
parquet row numbers and every episode carries exactly its plan's actions, so an
instance's action list identifies its source plan. The link is not unique -- most
instances match ten or more plans -- but the candidates agree on the rendered
structure for 59.4% of cooperation instances, and only those are rewritten. The
rest keep the flat rendering rather than being given a structure invented for them.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

ACTIONS_LINE = re.compile(r"^Actions: .*$", re.MULTILINE)


def token(action) -> str:
    target = action[1] if len(action) > 1 else ""
    return f"{action[0]}[{target}]"


def orderings(steps) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    robots = sorted({r for step in steps for r in step["actions"]})
    by_agent = [
        token(step["actions"][r])
        for r in robots
        for step in steps
        if step["actions"].get(r) is not None
    ]
    by_step = [
        token(step["actions"][r])
        for step in steps
        for r in robots
        if step["actions"].get(r) is not None
    ]
    return tuple(by_agent), tuple(by_step)


def structure(steps) -> Tuple:
    robots = sorted({r for step in steps for r in step["actions"]})
    return tuple(
        tuple(
            (r, token(step["actions"][r]) if step["actions"].get(r) else "idle")
            for r in robots
        )
        for step in steps
    )


def render(shape: Tuple) -> str:
    lines = ["Coordination (step: what each robot does):"]
    for number, row in enumerate(shape, 1):
        cells = ", ".join(f"{robot} {what}" for robot, what in row)
        lines.append(f"  step {number}: {cells}")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def index() -> Dict[Tuple[str, ...], str]:
    """action list -> rendered structure, only where the candidates agree."""
    import pandas as pd

    from viki_amendment8b import MEMORY_PARQUET, native

    frame = pd.read_parquet(MEMORY_PARQUET)
    exact: Dict[Tuple[str, ...], List[Tuple]] = defaultdict(list)
    multiset: Dict[Tuple[str, ...], List[Tuple]] = defaultdict(list)
    for i in range(len(frame)):
        steps = native(frame.iloc[i].to_dict())["reward_model"]["ground_truth"][
            "time_steps"
        ]
        shape = structure(steps)
        by_agent, by_step = orderings(steps)
        exact[by_agent].append(shape)
        if by_step != by_agent:
            exact[by_step].append(shape)
        multiset[tuple(sorted(by_agent))].append(shape)

    resolved: Dict[Tuple[str, ...], str] = {}
    for table in (exact, multiset):
        for key, shapes in table.items():
            unique = set(shapes)
            if len(unique) == 1 and key not in resolved:
                resolved[key] = render(next(iter(unique)))
    return resolved


def structure_for(action_sequence: List[str]) -> Optional[str]:
    table = index()
    key = tuple(action_sequence)
    if key in table:
        return table[key]
    return table.get(tuple(sorted(action_sequence)))


def rewrite(text: str, action_sequence: Optional[List[str]]) -> Tuple[str, bool]:
    """Replace the flat Actions: line with the step-aligned structure, if resolved."""
    if not action_sequence:
        return text, False
    shape = structure_for(action_sequence)
    if not shape:
        return text, False
    if not ACTIONS_LINE.search(text):
        return text, False
    return ACTIONS_LINE.sub(lambda _: shape, text, count=1), True
