#!/usr/bin/env python3
"""What the 924 interactive rows actually ask for, in the simulator's own terms.

Probe 1 asks whether a symbolic composer, handed the ground-truth goal predicates,
can build a plan the official judge accepts. Before writing the composer this counts
the shapes it will have to cover: how many goal predicates a row carries, which
status keys they set, how many robots are live, how long the reference plan is (the
composer's own plan may not exceed it), and how many rows carry a temporal
constraint. Nothing here scores anything; it is the composer's requirements list.
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

MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "results/viki_memory_experiments/amendment8b/interactive_manifest.jsonl"
)


def main() -> None:
    indices = [json.loads(line)["index"] for line in MANIFEST.read_text().splitlines() if line.strip()]
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")

    predicate_counts = Counter()
    status_keys = Counter()
    predicate_types = Counter()
    robot_counts = Counter()
    robot_types = Counter()
    plan_lengths = Counter()
    temporal_counts = Counter()
    asset_counts = Counter()
    families = Counter()
    goal_targets = Counter()
    rows = []

    for index in indices:
        sample = bench.to_native(frame.iloc[index].to_dict())
        truth = bench.get_ground_truth(sample)
        robots = {r: t for r, t in truth["robots"].items() if t is not None}
        assets = {a: p for a, p in truth["init_pos"].items() if p is not None and not (a.startswith("R") and a[1:].isdigit())}
        goals = truth["goal_constraints"]
        flat = []
        for outer in goals:
            for inner in outer if isinstance(outer, list) else [outer]:
                flat.append(inner)
        predicate_counts[len(flat)] += 1
        for predicate in flat:
            predicate_types[predicate.get("type")] += 1
            for key, value in predicate.get("status", {}).items():
                if value is None:
                    continue
                status_keys[key] += 1
                if key == "pos.name":
                    goal_targets[value] += 1
        robot_counts[len(robots)] += 1
        robot_types[tuple(sorted(robots.values()))] += 1
        plan_lengths[len(truth["time_steps"])] += 1
        temporal_counts[len(truth.get("temporal_constraints") or [])] += 1
        asset_counts[len(assets)] += 1
        families[truth["task_name"]] += 1
        rows.append(
            {
                "index": index,
                "task_name": truth["task_name"],
                "robots": len(robots),
                "predicates": len(flat),
                "plan_len": len(truth["time_steps"]),
                "temporal": len(truth.get("temporal_constraints") or []),
                "assets": len(assets),
            }
        )

    def show(title, counter, limit=None):
        print(f"\n=== {title} ===")
        items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
        for key, count in items[:limit]:
            print(f"  {key!r:<44} {count:>5}  {count / len(indices) * 100:5.1f}%")
        if limit and len(items) > limit:
            print(f"  ... {len(items) - limit} more")

    print(f"rows: {len(indices)}")
    show("goal predicates per row", predicate_counts)
    show("predicate type", predicate_types)
    show("status key set by goals", status_keys)
    show("goal pos.name target", goal_targets, 15)
    show("live robots per row", robot_counts)
    show("robot roster", robot_types, 10)
    show("reference plan length (composer's budget)", plan_lengths)
    show("temporal constraints per row", temporal_counts)
    show("live assets per row", asset_counts)
    show("task family", families, 20)

    out = Path(__file__).resolve().parents[1] / "results/viki_memory_experiments/amendment11"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "shape.csv", index=False)
    print(f"\nwrote {out / 'shape.csv'}")


if __name__ == "__main__":
    main()
