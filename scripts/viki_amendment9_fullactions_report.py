#!/usr/bin/env python3
"""Did removing the five-action render cap change anything, and where?

The cap cut every skill's action list at five entries. A two-agent skill stores
eleven or twelve, so the tail was always lost, and for clear_table the tail holds
Open[cabinet] -- the action those 220 rows cannot be solved without. This scores the
uncapped variant against the one it was branched from, family by family, because a
change aimed at one family should show up there and not everywhere.
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path
from statistics import median
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
    "gmemory",
    "skill_memory.queryfix_k8",
    "skill_memory.fullactions_k8",
)


def mcnemar(a: Dict[int, int], b: Dict[int, int]):
    shared = sorted(set(a) & set(b))
    n01 = sum(1 for i in shared if not a[i] and b[i])
    n10 = sum(1 for i in shared if a[i] and not b[i])
    total = n01 + n10
    if total == 0:
        return 0, 0, 1.0
    smaller = min(n01, n10)
    return (
        n10,
        n01,
        min(1.0, 2 * sum(comb(total, j) for j in range(smaller + 1)) / 2**total),
    )


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    manifest = load_manifest()
    test = pd.read_parquet(SOURCE_PARQUET)
    truth = {
        i: native(test.iloc[i].to_dict())["reward_model"]["ground_truth"]
        for i in sorted(manifest)
    }

    raw = {a: load_rows(OUTPUT_DIR / f"{a}.jsonl") for a in ARMS}
    scores = {
        a: {i: tolerant(scorer, r["response"], truth[i]) for i, r in rows.items()}
        for a, rows in raw.items()
    }

    print(f"{'arm':30s} {'official':>16s} {'json-tolerant':>16s} {'mem chars':>11s}")
    for arm in ARMS:
        rows = raw[arm]
        off = sum(r["task_score"] for r in rows.values())
        tol = sum(scores[arm].values())
        chars = median(r["memory_prompt_chars"] for r in rows.values())
        print(
            f"{arm:30s} {off:5d} ({100*off/len(rows):5.2f}%) "
            f"{tol:5d} ({100*tol/len(rows):5.2f}%) {chars:11.0f}"
        )

    print()
    header = " ".join(
        f"{a.replace('skill_memory.', 'sm.')[:14]:>14s}" for a in ARMS
    )
    print(f"{'family':46s} {'n':>4s} {header}")
    for family in folds():
        keep = set(rows_of(family))
        cells = []
        for arm in ARMS:
            hits = sum(v for i, v in scores[arm].items() if i in keep)
            cells.append(f"{hits:4d} ({100*hits/len(keep):5.1f}%)")
        print(f"{family:46s} {len(keep):4d} " + " ".join(f"{c:>14s}" for c in cells))

    print()
    print("McNemar exact, JSON-tolerant:")
    for a, b in (
        ("skill_memory.queryfix_k8", "skill_memory.fullactions_k8"),
        ("gmemory", "skill_memory.fullactions_k8"),
    ):
        n10, n01, p = mcnemar(scores[a], scores[b])
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {a:28s} vs {b:28s} {n10:4d}/{n01:<4d} p={p:.3g} {star}")

    family = "clear_table_with_two_robots_and_put_in_cabinet"
    keep = set(rows_of(family))
    print()
    print(f"On {family} alone (n={len(keep)}):")
    for arm in ARMS:
        hits = sum(v for i, v in scores[arm].items() if i in keep)
        print(f"  {arm:30s} {hits:4d}/{len(keep)} = {100*hits/len(keep):5.1f}%")
    sub_a = {i: v for i, v in scores["skill_memory.queryfix_k8"].items() if i in keep}
    sub_b = {i: v for i, v in scores["skill_memory.fullactions_k8"].items() if i in keep}
    n10, n01, p = mcnemar(sub_a, sub_b)
    print(f"  capped vs uncapped on this family: {n10}/{n01} p={p:.3g}")


if __name__ == "__main__":
    main()
