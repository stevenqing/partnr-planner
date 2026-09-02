#!/usr/bin/env python3
"""Do an instance's candidate source plans agree on the coordination structure?

Relinking by action list is not unique: 95.6% of instances match ten or more plans.
That is only safe if the candidates agree on what would be rendered -- which robot
acts at which step and which robot idles. Plan length already agrees for all of
them, but length is weaker than structure, and rendering a structure the skill does
not have would be worse than the truncation it replaces. This counts the distinct
structures behind each instance and reports the share that resolve to exactly one.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import MEMORY_PARQUET, OUTPUT_DIR, native
from viki_amendment9_relink import orderings, token

BANK = OUTPUT_DIR / "skill_memory_bank"


def structure(steps) -> Tuple:
    """What a rendering would show: per step, each robot's action or idle."""
    robots = sorted({r for step in steps for r in step["actions"]})
    return tuple(
        tuple(
            (robot, token(step["actions"][robot]) if step["actions"].get(robot) else "idle")
            for robot in robots
        )
        for step in steps
    )


def main() -> None:
    train = pd.read_parquet(MEMORY_PARQUET)
    plans = [
        native(train.iloc[i].to_dict())["reward_model"]["ground_truth"]["time_steps"]
        for i in range(len(train))
    ]
    exact: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    multiset: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for i, steps in enumerate(plans):
        by_agent, by_step = orderings(steps)
        exact[by_agent].append(i)
        if by_step != by_agent:
            exact[by_step].append(i)
        multiset[tuple(sorted(by_agent))].append(i)

    agree = Counter()
    example = None
    with gzip.open(BANK / "L_coop_skills.json.gz", "rt") as handle:
        skills = json.load(handle)
    for skill in skills.values():
        for instance in skill.get("instances", []):
            sequence = instance.get("context", {}).get("action_sequence")
            if not sequence:
                continue
            hits = exact.get(tuple(sequence)) or multiset.get(
                tuple(sorted(sequence)), []
            )
            if not hits:
                agree["unlinkable"] += 1
                continue
            shapes = {structure(plans[j]) for j in hits}
            agree[f"{min(len(shapes), 5)} distinct structure(s)"] += 1
            if len(shapes) == 1 and example is None:
                example = (skill["name"], sequence, plans[hits[0]])

    total = sum(agree.values())
    print(f"cooperation instances carrying an action list: {total}")
    for key, count in sorted(agree.items()):
        print(f"  {key:28s} {count:6d}  {100*count/total:5.1f}%")

    if example:
        name, sequence, steps = example
        print()
        print("=" * 74)
        print(f"what the renderer shows today, for skill {name!r}:")
        print("  Actions: " + " -> ".join(sequence[:5]) + "   (capped at five)")
        print()
        print("what the linked source plan actually is:")
        robots = sorted({r for step in steps for r in step["actions"]})
        for n, step in enumerate(steps, 1):
            cells = []
            for robot in robots:
                action = step["actions"].get(robot)
                cells.append(f"{robot} {token(action) if action else 'idle':22s}")
            print(f"  step {n}: " + " ".join(cells))
        print("=" * 74)


if __name__ == "__main__":
    main()
