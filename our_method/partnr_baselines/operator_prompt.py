#!/usr/bin/env python3
"""Skill memory v2's own operators, rendered as prompt text for the ReAct planner.

This is the cell the four-way comparison is missing. `react_rag_R`, `gmemory` and
`memento` all hand their memory to the model as text and let it plan; skill memory v2
does not -- it executes the memory directly and the model only states the goal. So a win
for us confounds two claims: that operators are a better *representation* than
trajectories or knowledge graphs, and that executing a memory beats reading one.

This arm separates them. The same operator library, from the same 161 rearrange-only
rollouts, written into the same `{rag_examples}` slot of the same planner as the three
baselines. If it lands with them, the representation was never the point and executability
is the whole contribution. If it lands above them, the representation carries weight on
its own. Either answer is worth having, and on VIKI-L2 the analogous arms -- the body menu
and the prose rendering -- both came in *below* the trajectory baseline, so the prior is
that this one does too.

The whole library goes in, not a retrieved subset. Six operators is small enough to fit,
and giving this arm everything makes it the generous version of itself: whatever it loses
by, it does not lose for want of the right operator being retrieved.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skill_memory_v2.partnr_memory import PartnrSkillMemory  # noqa: E402

HEADER = """These are the skills remembered from past rearrangement work in this house.
Each one says what becomes true and the sequence of actions that brought it about.
`?x` is the object being moved, `?y` where it must end up, and `?z` a container that had
to be opened on the way. Substitute the names of real objects and furniture."""

FOOTER = """Use these as reference for how such work is carried out here. Act one step at
a time in the required format."""


def render_operator(operator: Dict[str, Any], index: int) -> str:
    effect = operator["effect"]
    body = "; ".join(
        f"{action[0]}[{action[1]}]" if len(action) > 1 and action[1] else action[0]
        for action in operator["body"]
    )
    conditions = operator.get("preconditions") or {}
    notes = []
    if conditions.get("target_starts_shut"):
        notes.append("used when the container involved starts shut")
    if conditions.get("needs_exploration"):
        notes.append("used when the object has not been seen yet")
    suffix = f"  ({'; '.join(notes)})" if notes else ""
    return (f"{index}. To make {effect['key']}({effect['subject']}, {effect['value']}) true"
            f"{suffix}:\n     {body}\n     [seen in {operator.get('support', 0)} past "
            f"placements, {len(operator['body'])} actions]")


def render(memory: PartnrSkillMemory) -> str:
    lines = [HEADER, ""]
    for index, operator in enumerate(memory.operators, 1):
        lines.append(render_operator(operator, index))
    lines += ["", FOOTER]
    return "\n".join(lines)


class PartnrOperatorPrompt:
    """The library as text. Fixed per run -- there is nothing to retrieve against."""

    def __init__(self, path: str):
        self.memory = PartnrSkillMemory.load(path)
        self.text = render(self.memory)

    def __call__(self, instruction: str) -> str:
        return self.text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", default="results/partnr_operators.json")
    arguments = parser.parse_args()
    rendered = PartnrOperatorPrompt(arguments.operators)
    print(f"{len(rendered.memory.operators)} operators, {len(rendered.text)} chars\n")
    print(rendered.text)
