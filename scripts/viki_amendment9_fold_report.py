#!/usr/bin/env python3
"""Per-fold and pooled results for the held-out-family split.

Every fold is reported, not a selected subset. The pooled number is the sum over
folds, which scores each of the 924 rows exactly once, so it is directly
comparable with the in-distribution table.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from viki_amendment8b import OUTPUT_DIR
from viki_amendment9_folds import folds, rows_of

ARMS = ("zero_shot", "trajectory_rag", "gmemory", "skill_memory")


def load(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows[record["index"]] = record
    return rows


def mcnemar(a: Dict[int, Any], b: Dict[int, Any]) -> Dict[str, Any]:
    shared = sorted(set(a) & set(b))
    n01 = sum(1 for i in shared if not a[i]["task_score"] and b[i]["task_score"])
    n10 = sum(1 for i in shared if a[i]["task_score"] and not b[i]["task_score"])
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
    variant = sys.argv[1] if len(sys.argv) > 1 else ""
    root = OUTPUT_DIR / "folds"
    pooled: Dict[str, Dict[int, Any]] = {arm: {} for arm in ARMS}

    print(f"{'fold':46s} {'rows':>5s} " + " ".join(f"{a:>16s}" for a in ARMS))
    for family in folds():
        cells = []
        for arm in ARMS:
            tag = f"{arm}.{variant}" if variant and arm == "skill_memory" else arm
            rows = load(root / family / f"{tag}.jsonl")
            pooled[arm].update(rows)
            expected = len(rows_of(family))
            if len(rows) != expected:
                cells.append(f"{len(rows)}/{expected} …")
            else:
                hits = sum(r["task_score"] for r in rows.values())
                cells.append(f"{hits:3d}/{expected:<3d} {100*hits/expected:5.1f}%")
        print(f"{family:46s} {len(rows_of(family)):5d} " + " ".join(f"{c:>16s}" for c in cells))

    print()
    complete = {a: r for a, r in pooled.items() if len(r) == 924}
    if not complete:
        print("no arm has all 924 fold rows yet")
        return
    print("pooled over folds (each of the 924 rows scored once):")
    for arm, rows in complete.items():
        hits = sum(r["task_score"] for r in rows.values())
        fmt = sum(r["format_score"] for r in rows.values())
        print(
            f"  {arm:16s} {hits:4d}/924 = {100*hits/924:5.2f}%   format {100*fmt/924:5.1f}%"
        )
    print()
    print("McNemar exact, pooled:")
    names = sorted(complete)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            result = mcnemar(complete[a], complete[b])
            star = (
                "***" if result["p"] < 0.001 else "**" if result["p"] < 0.01
                else "*" if result["p"] < 0.05 else ""
            )
            print(
                f"  {a:16s} vs {b:16s} {result['a_only']:4d}/{result['b_only']:<4d} "
                f"p={result['p']:.3g} {star}"
            )


if __name__ == "__main__":
    main()
