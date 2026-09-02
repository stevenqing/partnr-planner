#!/usr/bin/env python3
"""Held-out-family results under both scorers, for every arm and skill variant.

The official parser rejects the `null` the dataset itself uses for idle robots, so
arms that emit real trajectories are penalised for a parsing artefact rather than a
planning error. Both numbers are reported; the McNemar tests use the tolerant one.
"""

from __future__ import annotations

import json
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

BASE_ARMS = ("zero_shot", "trajectory_rag", "gmemory")
VARIANTS = sys.argv[1:] or ["queryfix_k8", "grounded_k8"]


def mcnemar(a: Dict[int, int], b: Dict[int, int]) -> Dict[str, Any]:
    shared = sorted(set(a) & set(b))
    n01 = sum(1 for i in shared if not a[i] and b[i])
    n10 = sum(1 for i in shared if a[i] and not b[i])
    total = n01 + n10
    if total == 0:
        return {"n": len(shared), "a_only": 0, "b_only": 0, "p": 1.0}
    smaller = min(n01, n10)
    return {
        "n": len(shared),
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

    # Control runs are filed as arm variants (gmemory.shuffled), so any that exist
    # are picked up automatically rather than needing their own report.
    controls = sorted(
        {
            path.name[: -len(".jsonl")]
            for family in folds()
            for path in (OUTPUT_DIR / "folds" / family).glob("gmemory.*.jsonl")
            if not path.name.endswith(".run.json")
        }
    )
    names = list(BASE_ARMS) + controls + [f"skill_memory.{v}" for v in VARIANTS]
    official: Dict[str, Dict[int, int]] = {n: {} for n in names}
    lenient: Dict[str, Dict[int, int]] = {n: {} for n in names}
    root = OUTPUT_DIR / "folds"
    for family in folds():
        for name in names:
            for index, row in load_rows(root / family / f"{name}.jsonl").items():
                official[name][index] = int(row["task_score"])
                lenient[name][index] = tolerant(scorer, row["response"], truth[index])

    print(f"{'arm':30s} {'n':>5s} {'official':>16s} {'json-tolerant':>16s}")
    for name in names:
        rows = official[name]
        n = len(rows) or 1
        off = sum(rows.values())
        tol = sum(lenient[name].values())
        print(
            f"{name:30s} {len(rows):5d} {off:5d} ({100*off/n:5.2f}%) "
            f"{tol:5d} ({100*tol/n:5.2f}%)"
        )

    print()
    print("McNemar exact on the JSON-tolerant score:")
    for a, b in combinations(names, 2):
        result = mcnemar(lenient[a], lenient[b])
        star = (
            "***" if result["p"] < 0.001
            else "**" if result["p"] < 0.01
            else "*" if result["p"] < 0.05 else ""
        )
        print(
            f"  {a:28s} vs {b:28s} n={result['n']:4d} "
            f"{result['a_only']:4d}/{result['b_only']:<4d} p={result['p']:.3g} {star}"
        )

    print()
    print("Per fold, JSON-tolerant:")
    header = " ".join(f"{n.replace('skill_memory.','sm.')[:14]:>14s}" for n in names)
    print(f"{'fold':46s} {'n':>4s} {header}")
    for family in folds():
        keep = set(rows_of(family))
        cells = []
        for name in names:
            hits = sum(v for i, v in lenient[name].items() if i in keep)
            cells.append(f"{hits:4d} ({100*hits/len(keep):5.1f}%)")
        print(f"{family:46s} {len(keep):4d} " + " ".join(f"{c:>14s}" for c in cells))


if __name__ == "__main__":
    main()
