#!/usr/bin/env python3
"""Is G-Memory's out-of-distribution success exactly the transferable subset?

The abstraction levels give a ceiling: 167 of the 924 rows (18.1%) have their verb
skeleton -- who performs which verb at which step, objects dropped -- demonstrated
by a family other than their own. That is the most any memory can supply on the
held-out folds at that level of structure, and G-Memory scores 20.78% there. If its
correct rows sit inside that subset, the two numbers are the same fact and a
skeleton-based memory has no headroom over it. If they sit outside, it is winning
by a route the skeleton does not explain and the headroom is real.
"""

from __future__ import annotations

import sys
from collections import Counter
from math import comb
from pathlib import Path
from typing import Any, Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import (
    MEMORY_PARQUET,
    OUTPUT_DIR,
    SOURCE_PARQUET,
    load_manifest,
    native,
)
from viki_amendment9_diag102 import load_rows, tolerant
from viki_amendment9_folds import folds, rows_of, train_family_by_index
from viki_amendment9_template_levels import levels

ARMS = ("zero_shot", "trajectory_rag", "gmemory", "skill_memory.queryfix_k8")
LEVEL = "skeleton"


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    manifest = load_manifest()
    test = pd.read_parquet(SOURCE_PARQUET)
    train = pd.read_parquet(MEMORY_PARQUET)
    families = train_family_by_index()
    per_row_family = {i: f for f in folds() for i in rows_of(f)}

    bank: Dict[Any, Set[str]] = {}
    for i in range(len(train)):
        steps = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"][
            "time_steps"
        ]
        bank.setdefault(levels(steps)[LEVEL], set()).add(families.get(i))

    truth = {}
    transferable = set()
    for index in sorted(manifest):
        gt = native(test.iloc[index].to_dict())["reward_model"]["ground_truth"]
        truth[index] = gt
        homes = bank.get(levels(gt["time_steps"])[LEVEL], set())
        if homes - {per_row_family.get(index)}:
            transferable.add(index)
    print(
        f"rows whose verb skeleton is demonstrated by another family: "
        f"{len(transferable)}/924 = {100*len(transferable)/924:.1f}%"
    )

    root = OUTPUT_DIR / "folds"
    scores: Dict[str, Dict[int, int]] = {}
    for arm in ARMS:
        rows: Dict[int, Any] = {}
        for family in folds():
            rows.update(load_rows(root / family / f"{arm}.jsonl"))
        scores[arm] = {
            i: tolerant(scorer, r["response"], truth[i]) for i, r in rows.items()
        }

    print()
    print(f"{'arm':28s} {'on transferable':>22s} {'on the rest':>22s}")
    for arm, values in scores.items():
        inside = [v for i, v in values.items() if i in transferable]
        outside = [v for i, v in values.items() if i not in transferable]
        print(
            f"{arm:28s} "
            f"{sum(inside):4d}/{len(inside):<4d} ({100*sum(inside)/max(1,len(inside)):5.1f}%)"
            f"{'':6s}"
            f"{sum(outside):4d}/{len(outside):<4d} ({100*sum(outside)/max(1,len(outside)):5.1f}%)"
        )

    print()
    print("Where G-Memory's out-of-distribution correct rows actually sit:")
    hits = {i for i, v in scores["gmemory"].items() if v}
    print(f"  correct rows                       {len(hits)}")
    print(f"  of which inside the transferable subset  {len(hits & transferable)}")
    print(f"  of which outside it                      {len(hits - transferable)}")
    if transferable:
        print(
            f"  enrichment: {100*len(hits & transferable)/len(transferable):.1f}% of "
            f"transferable rows correct vs "
            f"{100*len(hits - transferable)/max(1, 924-len(transferable)):.1f}% of the rest"
        )

    print()
    print("Same for our arm:")
    ours = {i for i, v in scores["skill_memory.queryfix_k8"].items() if v}
    print(f"  correct rows                       {len(ours)}")
    print(f"  of which inside the transferable subset  {len(ours & transferable)}")
    print(f"  of which outside it                      {len(ours - transferable)}")

    print()
    print("Per fold, transferable rows and who gets them:")
    for family in folds():
        keep = set(rows_of(family)) & transferable
        if not keep:
            print(f"  {family:46s} 0 transferable")
            continue
        cells = " ".join(
            f"{arm.split('.')[0][:9]:>9s}={sum(scores[arm].get(i, 0) for i in keep):3d}"
            for arm in ARMS
        )
        print(f"  {family:46s} n={len(keep):4d}  {cells}")


if __name__ == "__main__":
    main()
