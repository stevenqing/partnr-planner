#!/usr/bin/env python3
"""Layer 3: the names this world uses, learned from the training episodes.

Thirty-eight of the forty-nine surviving failures are one thing: the model writes
`kitchen island` and the world calls it `kitchen island area`, so the judge compares
two strings that denote the same counter and refuses. Nothing in the instruction says
which of those is the official spelling. It is a fact about the domain, it is stated
plainly in several thousand training episodes, and it is exactly what a memory is for.

Two vocabularies come out: the asset types that exist, and the places a goal can name.
They are kept apart because a prediction snaps onto them differently -- an object that
matches nothing is a wrong object and is dropped, while a place that matches nothing is
still a place and is kept in its cleaned spelling.

Only `train.parquet` is read. The test split is opened once, at the end, purely to
report how much of it this vocabulary happens to cover; that number is a diagnostic
and never feeds the build. Keeping the two apart is the whole point of the exercise.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment11_composer import OUT

DATA = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2"


def harvest(frame: pd.DataFrame, exclude: str = None) -> tuple[Counter, Counter, Counter]:
    assets, places, goal_targets = Counter(), Counter(), Counter()
    for position in range(len(frame)):
        truth = bench.get_ground_truth(bench.to_native(frame.iloc[position].to_dict()))
        if not isinstance(truth, dict):
            continue
        if exclude and truth.get("task_name") == exclude:
            continue
        for name, positions in (truth.get("init_pos") or {}).items():
            if positions is None or (name.startswith("R") and name[1:].isdigit()):
                continue
            assets[name.rsplit("_", 1)[0]] += 1
            for item in positions:
                if isinstance(item, str):
                    places[item] += 1
        for constraint in truth.get("goal_constraints") or []:
            stack = [constraint]
            while stack:
                node = stack.pop()
                if isinstance(node, list):
                    stack.extend(node)
                elif isinstance(node, dict):
                    target = (node.get("status") or {}).get("pos.name")
                    if isinstance(target, str):
                        goal_targets[target] += 1
    return assets, places, goal_targets


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="vocabulary.json")
    parser.add_argument("--exclude-family", default=None,
                        help="build as if this family had never been seen (held-out fold)")
    arguments = parser.parse_args()

    train = pd.read_parquet(DATA / "train.parquet")
    assets, positions, goal_targets = harvest(train, arguments.exclude_family)

    # A goal may name a place that is never an initial position, and an asset may serve
    # as one (a bowl is somewhere to put an apple), so the place vocabulary is the union
    # rather than either alone.
    places = Counter(positions)
    places.update(goal_targets)
    for asset, count in assets.items():
        places[asset] += count

    record = {
        "built_from": "VIKI-L2 train.parquet only",
        "excluded_family": arguments.exclude_family,
        "train_episodes": int(len(train)),
        "assets": [name for name, _ in assets.most_common()],
        "places": [name for name, _ in places.most_common()],
        "goal_targets": {name: count for name, count in goal_targets.most_common()},
        "asset_counts": {name: count for name, count in assets.most_common()},
    }

    missing = [name for name in goal_targets if name not in places]
    print(f"train episodes      {len(train)}")
    print(f"asset types         {len(assets)}")
    print(f"place names         {len(places)}")
    print(f"distinct goal targets in train: {len(goal_targets)}")
    print(f"  {list(goal_targets)[:20]}")
    print(f"self-check: every train goal target is in the vocabulary -> {not missing} {missing}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / arguments.out).write_text(json.dumps(record, indent=2) + "\n")

    if arguments.exclude_family is None:
        # Diagnostic only, computed after the artefact is written and never fed back.
        test = pd.read_parquet(DATA / "test.parquet")
        _, _, test_targets = harvest(test)
        covered = sum(count for name, count in test_targets.items() if name in places)
        total = sum(test_targets.values())
        uncovered = sorted(name for name in test_targets if name not in places)
        print(f"\ndiagnostic (not used in the build): test goal targets covered "
              f"{covered}/{total} = {covered / total * 100:.2f}%")
        print(f"  distinct test targets: {len(test_targets)}; not in vocabulary: {uncovered}")
    print(f"\nwrote {OUT / arguments.out}")


if __name__ == "__main__":
    main()
