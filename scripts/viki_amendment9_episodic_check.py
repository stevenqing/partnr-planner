#!/usr/bin/env python3
"""Can the episodic layer be rebuilt from the parquet instead of the text traces?

The step structure is not lost by the LLM extractor. It is lost before it:
build_hierarchical_skill_memory parses text traces where step_number only advances
on an observation line, those traces carry no observation lines, and so all 52081
actions are stamped step 0. Re-running extraction over that input would produce the
same flat sequences, so a re-extraction is only worth its cost if the episodic layer
is rebuilt first from a source that still has the timing.

The training parquet has it: every row carries the step-aligned multi-robot plan the
scorer grades against. This checks the two things a rebuild depends on -- that the
episodic keys line up with parquet rows, and that the actions agree once the parquet
plan is flattened the way the trace parser flattened it.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import MEMORY_PARQUET, OUTPUT_DIR, native

BANK = OUTPUT_DIR / "skill_memory_bank"


def flatten(steps):
    """Per agent, in agent order, the actions in step order -- the shape the trace
    parser produced, so the two can be compared."""
    robots = sorted({r for step in steps for r in step["actions"]})
    out = []
    for robot in robots:
        for step in steps:
            action = step["actions"].get(robot)
            if action is not None:
                target = action[1] if len(action) > 1 else ""
                out.append((robot, str(action[0]), str(target)))
    return out


def main() -> None:
    with gzip.open(BANK / "episodic_memory.json.gz", "rt") as handle:
        episodic = json.load(handle)
    train = pd.read_parquet(MEMORY_PARQUET)
    print(f"episodic entries {len(episodic)}   parquet rows {len(train)}")

    keys = list(episodic)
    print(f"key format sample: {keys[:3]}")
    bad = [k for k in keys if not k.startswith("train:")]
    print(f"keys not of the form train:N -> {len(bad)}")

    tally = Counter()
    for key, entry in episodic.items():
        if not key.startswith("train:"):
            continue
        index = int(key.split(":", 1)[1])
        if index >= len(train):
            tally["index beyond the parquet"] += 1
            continue
        row = native(train.iloc[index].to_dict())
        steps = row["reward_model"]["ground_truth"]["time_steps"]
        want = flatten(steps)
        got = [
            (
                f"R{a['agent_id'] + 1}",
                str(a["action_type"]),
                str(a["action_args"][0]) if a.get("action_args") else "",
            )
            for a in entry["actions"]
            if a.get("action_type") != "observation"
        ]
        if got == want:
            tally["actions match the parquet plan exactly"] += 1
        elif sorted(got) == sorted(want):
            tally["same actions, different order"] += 1
        elif [g[1:] for g in got] == [w[1:] for w in want]:
            tally["same verbs and targets, agents differ"] += 1
        else:
            tally["mismatch"] += 1

    print()
    for key, count in tally.most_common():
        print(f"  {key:42s} {count:5d}  {100*count/len(episodic):5.1f}%")

    print()
    steps_seen = Counter(
        a.get("step_number") for e in episodic.values() for a in e["actions"]
    )
    print(f"step_number values currently stored: {dict(steps_seen)}")
    lengths = Counter(
        len(native(train.iloc[i].to_dict())["reward_model"]["ground_truth"]["time_steps"])
        for i in range(len(train))
    )
    print(f"parquet plan lengths: {sorted(lengths.items())}")


if __name__ == "__main__":
    main()
