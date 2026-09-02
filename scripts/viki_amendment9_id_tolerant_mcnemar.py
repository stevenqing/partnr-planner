#!/usr/bin/env python3
"""In-distribution paired tests on the JSON-tolerant score.

viki_amendment9_report.py prints its McNemar table on the official score. For the
variants whose format rate is around 50% -- grounded and rescore both emit the
`null` the official parser rejects -- that table compares a parsing artefact rather
than a planning difference, so the same tests are recomputed here on the tolerant
score. The two tables disagree, and the tolerant one is the one to read.
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

    scores: Dict[str, Dict[int, int]] = {}
    for path in sorted(OUTPUT_DIR.glob("*.jsonl")):
        if path.name.endswith(".aligned.jsonl") or "manifest" in path.name:
            continue
        rows = load_rows(path)
        if len(rows) != len(manifest):
            continue
        scores[path.name[: -len(".jsonl")]] = {
            i: tolerant(scorer, r["response"], truth[i]) for i, r in rows.items()
        }

    print("McNemar exact, in-distribution, JSON-tolerant score:")
    for a, b in combinations(sorted(scores), 2):
        result = mcnemar(scores[a], scores[b])
        star = (
            "***" if result["p"] < 0.001
            else "**" if result["p"] < 0.01
            else "*" if result["p"] < 0.05 else ""
        )
        print(
            f"  {a:28s} vs {b:28s} n={result['n']:4d} "
            f"{result['a_only']:4d}/{result['b_only']:<4d} p={result['p']:.3g} {star}"
        )


if __name__ == "__main__":
    main()
