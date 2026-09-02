#!/usr/bin/env python3
"""Results on the recombination split, both forms, with the goals scored apart.

Three things are asked of the same runs.

What the arms do on a task whose answer is not in the bank in one piece: the bank
holds the two-robot cutting pattern and the one-robot delivery pattern, and no
trajectory that carries both, so a memory has to combine two entries.

Whether the image mattered: the two forms carry identical instances and identical
ground truth, so the paired difference is the image's contribution, and it settles
the one question the metadata could not -- whether an asset restored to init_pos is
actually in the picture.

And where the failures fall. Each instance has a cutting goal and a delivery goal;
scoring a prediction against each alone says which half it lost. If failures cluster
on the delivery half specifically in the imaged form, that is the missing object
again, from a second and independent direction.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import native
from viki_amendment9_diag102 import parse_plan, tolerant
from viki_amendment10_run import SPLIT_DIR, split_path
from viki_plan_format import evaluate

SPLITS = ("imaged", "text")


def load(path: Path) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows[record["index"]] = record
    return rows


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


def star(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def halves(truth: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The instance split back into the two goals it was built from."""
    goals = truth["goal_constraints"]
    cutting = json.loads(json.dumps(truth))
    cutting["goal_constraints"] = goals[:-1]
    delivery = json.loads(json.dumps(truth))
    delivery["goal_constraints"] = goals[-1:]
    delivery["temporal_constraints"] = []
    return {"cutting": cutting, "delivery": delivery}


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    arms: Dict[str, Dict[str, Dict[int, int]]] = {}
    truths: Dict[str, Dict[int, Any]] = {}

    for split in SPLITS:
        frame = pd.read_parquet(split_path(split))
        truths[split] = {
            i: native(frame.iloc[i].to_dict())["reward_model"]["ground_truth"]
            for i in range(len(frame))
        }
        arms[split] = {}
        folder = SPLIT_DIR / split
        for path in sorted(folder.glob("*.jsonl")):
            rows = load(path)
            if not rows:
                continue
            name = path.name[: -len(".jsonl")]
            arms[split][name] = {
                i: tolerant(scorer, r["response"], truths[split][i])
                for i, r in rows.items()
            }
            arms[split][name + "::official"] = {
                i: int(r["task_score"]) for i, r in rows.items()
            }

    print(f"{'arm':30s} " + " ".join(f"{s + ' official':>18s} {s + ' tolerant':>18s}"
                                     for s in SPLITS))
    names = sorted(
        {n for split in SPLITS for n in arms[split] if not n.endswith("::official")}
    )
    for name in names:
        cells = []
        for split in SPLITS:
            tol = arms[split].get(name, {})
            off = arms[split].get(name + "::official", {})
            n = len(tol) or 1
            cells.append(f"{sum(off.values()):4d} ({100*sum(off.values())/n:5.1f}%)")
            cells.append(f"{sum(tol.values()):4d} ({100*sum(tol.values())/n:5.1f}%)")
        print(f"{name:30s} " + " ".join(f"{c:>18s}" for c in cells))

    for split in SPLITS:
        present = [n for n in names if n in arms[split]]
        if len(present) < 2:
            continue
        print()
        print(f"McNemar exact within the {split} split, JSON-tolerant:")
        for a, b in combinations(present, 2):
            result = mcnemar(arms[split][a], arms[split][b])
            print(
                f"  {a:28s} vs {b:28s} n={result['n']:4d} "
                f"{result['a_only']:4d}/{result['b_only']:<4d} "
                f"p={result['p']:.3g} {star(result['p'])}"
            )

    print()
    print("image contribution: the same arm, imaged against text")
    for name in names:
        if name in arms["imaged"] and name in arms["text"]:
            result = mcnemar(arms["imaged"][name], arms["text"][name])
            print(
                f"  {name:30s} imaged-only {result['a_only']:4d}  "
                f"text-only {result['b_only']:4d}  p={result['p']:.3g} "
                f"{star(result['p'])}"
            )

    print()
    print("which half of the task each arm reaches (JSON-tolerant machinery):")
    print(f"{'arm':30s} {'split':>8s} {'cutting':>12s} {'delivery':>12s} {'both':>12s}")
    for name in names:
        for split in SPLITS:
            rows = load(SPLIT_DIR / split / f"{name}.jsonl")
            if not rows:
                continue
            counts = {"cutting": 0, "delivery": 0, "both": 0}
            for index, record in rows.items():
                parsed = parse_plan(record["response"])
                if parsed is None:
                    continue
                parts = halves(truths[split][index])
                reached = {}
                for half, truth in parts.items():
                    ok, _ = evaluate(scorer, parsed, truth)
                    reached[half] = bool(ok)
                    counts[half] += int(bool(ok))
                counts["both"] += int(all(reached.values()))
            total = len(rows) or 1
            print(
                f"{name:30s} {split:>8s} "
                f"{counts['cutting']:5d} ({100*counts['cutting']/total:4.1f}%) "
                f"{counts['delivery']:5d} ({100*counts['delivery']/total:4.1f}%) "
                f"{counts['both']:5d} ({100*counts['both']/total:4.1f}%)"
            )


if __name__ == "__main__":
    main()
