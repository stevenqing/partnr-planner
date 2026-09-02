#!/usr/bin/env python3
"""Is our advantage over the baselines spread across the folds or carried by one?

On the held-out folds our arm beats trajectory_rag 3.5x and zero_shot 3.3x, both at
p < 1e-14. That is the comparison our method is actually about -- a skill hierarchy
against flat trajectory retrieval -- so it matters whether it survives dropping the
single easiest fold. parallel_human_dual_asset is the fold where even the memoryless
arm scores 37%, and it holds 86 of the 924 rows, so it is the one that could carry
the result on its own.

Every fold is dropped in turn, not just that one, so the check cannot be accused of
removing exactly the inconvenient fold.
"""

from __future__ import annotations

import sys
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import OUTPUT_DIR, SOURCE_PARQUET, load_manifest, native
from viki_amendment9_diag102 import load_rows, tolerant
from viki_amendment9_folds import folds, rows_of

ARMS = (
    "zero_shot",
    "trajectory_rag",
    "gmemory.shuffled",
    "skill_memory.queryfix_k8",
    "gmemory",
)


def mcnemar(a: Dict[int, int], b: Dict[int, int], keep) -> Dict[str, Any]:
    shared = sorted((set(a) & set(b)) & keep)
    n01 = sum(1 for i in shared if not a[i] and b[i])
    n10 = sum(1 for i in shared if a[i] and not b[i])
    total = n01 + n10
    if total == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    smaller = min(n01, n10)
    return {
        "a_only": n10,
        "b_only": n01,
        "p": min(1.0, 2 * sum(comb(total, j) for j in range(smaller + 1)) / 2**total),
    }


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    manifest = load_manifest()
    test = pd.read_parquet(SOURCE_PARQUET)
    truth = {
        i: native(test.iloc[i].to_dict())["reward_model"]["ground_truth"]
        for i in sorted(manifest)
    }

    scores: Dict[str, Dict[int, int]] = {}
    for arm in ARMS:
        rows: Dict[int, Any] = {}
        for family in folds():
            rows.update(load_rows(OUTPUT_DIR / "folds" / family / f"{arm}.jsonl"))
        scores[arm] = {
            i: tolerant(scorer, r["response"], truth[i]) for i, r in rows.items()
        }

    ours = "skill_memory.queryfix_k8"
    all_rows = set(manifest)
    print("Leave-one-fold-out. Each line drops that fold and rescores the rest.")
    print(
        f"{'dropped fold':46s} {'n':>5s} "
        + " ".join(f"{a.replace('skill_memory.','sm.')[:12]:>12s}" for a in ARMS)
    )
    for dropped in [None] + list(folds()):
        keep = all_rows - (set(rows_of(dropped)) if dropped else set())
        cells = []
        for arm in ARMS:
            hits = sum(v for i, v in scores[arm].items() if i in keep)
            cells.append(f"{100*hits/len(keep):11.1f}%")
        label = dropped or "(nothing dropped)"
        print(f"{label:46s} {len(keep):5d} " + " ".join(f"{c:>12s}" for c in cells))

    print()
    print("Our arm against each baseline, with the easiest fold removed:")
    easiest = "parallel_human_dual_asset_to_plate_or_bowl"
    for label, keep in (
        ("all 8 folds", all_rows),
        (f"without {easiest}", all_rows - set(rows_of(easiest))),
    ):
        print(f"  {label} (n={len(keep)})")
        for other in ARMS:
            if other == ours:
                continue
            result = mcnemar(scores[ours], scores[other], keep)
            star = (
                "***" if result["p"] < 0.001
                else "**" if result["p"] < 0.01
                else "*" if result["p"] < 0.05 else ""
            )
            print(
                f"      vs {other:28s} {result['a_only']:4d}/{result['b_only']:<4d} "
                f"p={result['p']:.3g} {star}"
            )


if __name__ == "__main__":
    main()
