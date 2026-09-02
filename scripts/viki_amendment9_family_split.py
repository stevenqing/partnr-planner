#!/usr/bin/env python3
"""Split the fold results by whether the fold actually removed the task.

The retrieval audit shows the family partition is finer than the task: holding out
cut_fruit_on_board leaves cut_two_fruits_on_board in the bank, and 90% of that
fold's neighbours come from it. Four folds retrieve from one concentrated sibling,
four retrieve scattered across unrelated families. Scoring the two groups apart
says whether the held-out comparison measures transfer or sibling retrieval.

The in-distribution score for the same families is printed alongside, because a
fold that reads 0% is only evidence about transfer if the arm can do those rows at
all when their own family is present.
"""

from __future__ import annotations

import sys
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

# From sibling_retrieval.json: the share of a fold's rows whose top permitted
# neighbour comes from a single other family.
SIBLING = {
    "cut_fruit_on_board": "cut_two_fruits_on_board 90%",
    "cut_two_fruits_on_board": "cut_fruit_on_board 95%",
    "toast_bread_and_set_plate": "set_plate_and_fork_on_table 100%",
    "parallel_human_dual_asset_to_plate_or_bowl": "sequential_pick_two_and_place 74%",
}
ARMS = (
    "zero_shot",
    "trajectory_rag",
    "gmemory",
    "gmemory.shuffled",
    "skill_memory.queryfix_k8",
)
ID_NAME = {"gmemory.shuffled": None, "skill_memory.queryfix_k8": "skill_memory.queryfix_k8"}


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    manifest = load_manifest()
    test = pd.read_parquet(SOURCE_PARQUET)
    truth = {
        i: native(test.iloc[i].to_dict())["reward_model"]["ground_truth"]
        for i in sorted(manifest)
    }
    per_row_family = {i: f for f in folds() for i in rows_of(f)}

    fold_scores: Dict[str, Dict[int, int]] = {}
    for arm in ARMS:
        rows: Dict[int, Any] = {}
        for family in folds():
            rows.update(load_rows(OUTPUT_DIR / "folds" / family / f"{arm}.jsonl"))
        fold_scores[arm] = {
            i: tolerant(scorer, r["response"], truth[i]) for i, r in rows.items()
        }

    id_scores: Dict[str, Dict[int, int]] = {}
    for arm in ARMS:
        name = ID_NAME.get(arm, arm)
        if name is None:
            continue
        rows = load_rows(OUTPUT_DIR / f"{name}.jsonl")
        if len(rows) == len(manifest):
            id_scores[arm] = {
                i: tolerant(scorer, r["response"], truth[i]) for i, r in rows.items()
            }

    leaked = {i for i, f in per_row_family.items() if f in SIBLING}
    clean = {i for i in per_row_family if i not in leaked}
    print(
        f"folds whose neighbours concentrate in one sibling family: "
        f"{len(SIBLING)}/8, {len(leaked)} rows"
    )
    print(f"folds whose neighbours scatter across unrelated families: "
          f"{8 - len(SIBLING)}/8, {len(clean)} rows")
    print()
    print(
        f"{'arm':28s} {'sibling folds (fold)':>22s} {'clean folds (fold)':>22s} "
        f"{'clean folds (in-dist)':>23s}"
    )
    for arm in ARMS:
        values = fold_scores[arm]
        a = [values[i] for i in leaked if i in values]
        b = [values[i] for i in clean if i in values]
        cell_id = "        --"
        if arm in id_scores:
            c = [id_scores[arm][i] for i in clean if i in id_scores[arm]]
            cell_id = f"{sum(c):4d}/{len(c):<4d} ({100*sum(c)/max(1,len(c)):5.1f}%)"
        print(
            f"{arm:28s} "
            f"{sum(a):4d}/{len(a):<4d} ({100*sum(a)/max(1,len(a)):5.1f}%)"
            f"{'':4s}"
            f"{sum(b):4d}/{len(b):<4d} ({100*sum(b)/max(1,len(b)):5.1f}%)"
            f"{'':5s}{cell_id}"
        )

    print()
    print("Per family, in-distribution, JSON-tolerant (the bank still has its own):")
    header = " ".join(f"{a.replace('skill_memory.','sm.')[:13]:>13s}" for a in id_scores)
    print(f"{'family':46s} {'n':>4s} {header}")
    for family in folds():
        keep = set(rows_of(family))
        cells = []
        for arm in id_scores:
            hits = sum(v for i, v in id_scores[arm].items() if i in keep)
            cells.append(f"{hits:4d} ({100*hits/len(keep):5.1f}%)")
        mark = " *sibling" if family in SIBLING else ""
        print(f"{family:46s} {len(keep):4d} " + " ".join(f"{c:>13s}" for c in cells) + mark)


if __name__ == "__main__":
    main()
