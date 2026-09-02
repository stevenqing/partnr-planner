#!/usr/bin/env python3
"""Does our arm emit serialised plans because its memory stores serialised plans?

Every action in the episodic layer carries agent_id but step_number == 0, and the
cooperation skills store agent_0_actions + agent_1_actions concatenated. So no
layer of our hierarchy records which robots act at the same step. If that is what
the model is imitating, its plans should be about as long as the flattened action
count of the ground truth rather than its step count, and the gap should be largest
exactly where the ground truth is most parallel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import OUTPUT_DIR, SOURCE_PARQUET, load_manifest, native
from viki_amendment9_diag102 import canon, load_rows, parse_plan
from viki_amendment9_folds import folds, rows_of

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "queryfix_k8"


def lengths(truth: Dict[str, Any]) -> Dict[str, int]:
    steps = truth["time_steps"]
    flat = sum(
        1
        for step in steps
        for action in step["actions"].values()
        if action is not None
    )
    return {"parallel": len(steps), "serial": flat}


def main() -> None:
    manifest = load_manifest()
    test = pd.read_parquet(SOURCE_PARQUET)
    root = OUTPUT_DIR / "folds"

    rows: Dict[int, Dict[str, Any]] = {}
    theirs: Dict[int, Dict[str, Any]] = {}
    for family in folds():
        rows.update(load_rows(root / family / f"skill_memory.{VARIANT}.jsonl"))
        theirs.update(load_rows(root / family / "gmemory.jsonl"))

    records: List[Dict[str, Any]] = []
    for index in sorted(set(rows) & set(theirs) & set(manifest)):
        truth = native(test.iloc[index].to_dict())["reward_model"]["ground_truth"]
        want = lengths(truth)
        if want["parallel"] == want["serial"]:
            continue  # a fully sequential plan cannot separate the two hypotheses
        for arm, source in (("ours", rows), ("gmemory", theirs)):
            plan = canon(parse_plan(source[index]["response"]))
            if not plan:
                continue
            records.append(
                {
                    "arm": arm,
                    "index": index,
                    "emitted": len(plan),
                    "parallel": want["parallel"],
                    "serial": want["serial"],
                }
            )

    print(
        f"rows whose ground truth is genuinely parallel: "
        f"{len({r['index'] for r in records})}"
    )
    print()
    print(
        f"{'arm':10s} {'emitted':>9s} {'gt steps':>9s} {'gt actions':>11s} "
        f"{'closer to':>12s}"
    )
    for arm in ("ours", "gmemory"):
        subset = [r for r in records if r["arm"] == arm]
        if not subset:
            continue
        near_serial = sum(
            1
            for r in subset
            if abs(r["emitted"] - r["serial"]) < abs(r["emitted"] - r["parallel"])
        )
        exact_parallel = sum(1 for r in subset if r["emitted"] == r["parallel"])
        exact_serial = sum(1 for r in subset if r["emitted"] == r["serial"])
        print(
            f"{arm:10s} {median(r['emitted'] for r in subset):9.1f} "
            f"{median(r['parallel'] for r in subset):9.1f} "
            f"{median(r['serial'] for r in subset):11.1f} "
            f"{100*near_serial/len(subset):11.1f}%"
        )
        print(
            f"           exact match to the step count {exact_parallel:4d}/"
            f"{len(subset)} = {100*exact_parallel/len(subset):.1f}%   "
            f"to the flattened action count {exact_serial:4d}/{len(subset)} = "
            f"{100*exact_serial/len(subset):.1f}%"
        )

    out = OUTPUT_DIR / "folds" / f"diag_timing.{VARIANT}.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"\nper-row lengths written to {out}")


if __name__ == "__main__":
    main()
