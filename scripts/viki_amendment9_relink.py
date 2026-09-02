#!/usr/bin/env python3
"""Can a skill instance be linked back to its source plan without re-extracting?

The episodic check showed every episode carries exactly the parquet plan's actions
and nothing else, and that episode keys are parquet row numbers. A cooperation
instance stores that same flattened action list. So the instance can be matched to
its source plan by the action list alone, and the source plan still has the timing
the bank lost -- which robot acts at which step, and which robot idles.

The match is only usable if it is close to unique. This counts, for every instance,
how many episodes carry exactly its action list, in order and as a multiset. If most
instances resolve to one plan, the agent-step structure can be restored offline and
the re-extraction is unnecessary; if they resolve to many, rendering any one of them
would invent a structure the skill does not have.
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

BANK = OUTPUT_DIR / "skill_memory_bank"


def token(action) -> str:
    target = action[1] if len(action) > 1 else ""
    return f"{action[0]}[{target}]"


def orderings(steps) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Two ways the trace parser could have serialised a plan: agent-major, and
    step-major. Both are tried because the episodic check found the bank split
    almost evenly between them."""
    robots = sorted({r for step in steps for r in step["actions"]})
    by_agent, by_step = [], []
    for robot in robots:
        for step in steps:
            action = step["actions"].get(robot)
            if action is not None:
                by_agent.append(token(action))
    for step in steps:
        for robot in robots:
            action = step["actions"].get(robot)
            if action is not None:
                by_step.append(token(action))
    return tuple(by_agent), tuple(by_step)


def main() -> None:
    train = pd.read_parquet(MEMORY_PARQUET)
    exact: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    multiset: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for i in range(len(train)):
        steps = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"][
            "time_steps"
        ]
        by_agent, by_step = orderings(steps)
        exact[by_agent].append(i)
        if by_step != by_agent:
            exact[by_step].append(i)
        multiset[tuple(sorted(by_agent))].append(i)
    print(f"parquet plans indexed: {len(train)}")

    tally = Counter()
    candidates = Counter()
    distinct_shapes = Counter()
    for name in ("L_coop_skills", "L_ind_skills"):
        with gzip.open(BANK / f"{name}.json.gz", "rt") as handle:
            skills = json.load(handle)
        for skill in skills.values():
            for instance in skill.get("instances", []):
                sequence = instance.get("context", {}).get("action_sequence")
                if not sequence:
                    continue
                key = tuple(sequence)
                hits = exact.get(key)
                if hits:
                    tally[f"{name}: exact order match"] += 1
                else:
                    hits = multiset.get(tuple(sorted(sequence)), [])
                    tally[
                        f"{name}: multiset match" if hits else f"{name}: no match"
                    ] += 1
                if hits:
                    candidates[min(len(hits), 10)] += 1
                    shapes = {
                        len(
                            native(train.iloc[j].to_dict())["reward_model"][
                                "ground_truth"
                            ]["time_steps"]
                        )
                        for j in hits
                    }
                    distinct_shapes[len(shapes)] += 1

    print()
    for key, count in sorted(tally.items()):
        print(f"  {key:38s} {count:6d}")
    total = sum(candidates.values()) or 1
    print()
    print("number of source plans an instance matches (10 = ten or more):")
    for size, count in sorted(candidates.items()):
        print(f"  {size:3d} plans  {count:6d}  {100*count/total:5.1f}%")
    print()
    print("distinct plan LENGTHS among an instance's matches:")
    for size, count in sorted(distinct_shapes.items()):
        print(
            f"  {size:3d} distinct length(s)  {count:6d}  "
            f"{100*count/total:5.1f}%"
        )


if __name__ == "__main__":
    main()
