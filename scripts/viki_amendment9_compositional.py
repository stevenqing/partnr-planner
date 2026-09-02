#!/usr/bin/env python3
"""The compositional split, as filed in the Amendment 9 preregistration.

    "rows whose ground-truth plan uses an action pair present in the bank only in
     separate families"

An action is a `Verb[target]` token, the same unit the skill bank stores. For each
unordered pair of actions the bank is asked which families contain both of them in
one plan. A pair is *split* when each action occurs somewhere in the bank but no
single family ever puts them in the same plan; a test row is compositional when its
ground-truth plan contains at least one split pair. Answering such a row requires
joining material the bank only ever shows apart.

The split selects rows; it does not change the bank, so it is scored from the
existing in-distribution runs and costs no inference. Two pairings are reported
because the filed wording does not fix one: pairs of actions co-occurring anywhere
in the plan, and pairs issued by different robots at the same step, which is the
coordination-specific reading.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any, Dict, Set, Tuple

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
from viki_amendment9_folds import train_family_by_index

ARMS = (
    "zero_shot",
    "trajectory_rag",
    "gmemory",
    "skill_memory.queryfix_k8",
    "skill_memory.patternslot_k8",
    "skill_memory.grounded_k8",
)


def token(action: Any) -> str:
    if isinstance(action, list) and action:
        target = action[1] if len(action) > 1 else ""
        return f"{action[0]}[{target}]"
    return str(action)


def plan_actions(steps: Any) -> Set[str]:
    return {
        token(a) for step in steps for a in step["actions"].values() if a is not None
    }


def plan_pairs(steps: Any, mode: str) -> Set[Tuple[str, str]]:
    """Unordered action pairs of a plan under one of the two readings."""
    if mode == "coactive":
        pairs = set()
        for step in steps:
            live = sorted(
                {token(a) for a in step["actions"].values() if a is not None}
            )
            pairs.update(combinations(live, 2))
        return pairs
    actions = sorted(
        {
            token(a)
            for step in steps
            for a in step["actions"].values()
            if a is not None
        }
    )
    return set(combinations(actions, 2))


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
    train = pd.read_parquet(MEMORY_PARQUET)
    families = train_family_by_index()

    truth = {
        i: native(test.iloc[i].to_dict())["reward_model"]["ground_truth"]
        for i in sorted(manifest)
    }

    train_steps = [
        native(train.iloc[i].to_dict())["reward_model"]["ground_truth"]["time_steps"]
        for i in range(len(train))
    ]

    scores: Dict[str, Dict[int, int]] = {}
    for arm in ARMS:
        rows = load_rows(OUTPUT_DIR / f"{arm}.jsonl")
        if len(rows) != len(manifest):
            print(f"  (skipping {arm}: {len(rows)}/{len(manifest)} rows)")
            continue
        scores[arm] = {
            i: tolerant(scorer, r["response"], truth[i]) for i, r in rows.items()
        }

    for mode, label in (
        ("cooccur", "pairs co-occurring anywhere in the plan"),
        ("coactive", "pairs issued by different robots at the same step"),
        ("novel_set", "whole action set not covered by any single family"),
    ):
        print("=" * 78)
        print(f"Compositional split — {label}")
        print("=" * 78)

        # The filed wording is about families, not plans: a pair counts when each
        # action is present in the bank and the families that contain them are
        # disjoint, so no single family ever demonstrates the two together.
        action_families: Dict[str, Set[str]] = defaultdict(set)
        pair_families: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for i, steps in enumerate(train_steps):
            family = families.get(i)
            for step in steps:
                for action in step["actions"].values():
                    if action is not None:
                        action_families[token(action)].add(family)
            for pair in plan_pairs(steps, "coactive" if mode == "coactive" else "cooccur"):
                pair_families[pair].add(family)

        def is_split(pair: Tuple[str, str]) -> bool:
            left, right = action_families.get(pair[0]), action_families.get(pair[1])
            return bool(left) and bool(right) and not (left & right)

        print(
            f"bank: {len(action_families)} distinct actions across "
            f"{len(set().union(*action_families.values()))} families, "
            f"{len(pair_families)} pairs demonstrated together"
        )

        if mode == "novel_set":
            # The pair criteria come out empty, so a strictly harder one is also
            # reported: the row's whole set of primitives is not contained in any
            # one family's repertoire, which is the weakest sense in which a plan
            # can require joining material from more than one family. This goes
            # beyond the filed wording and is labelled as an addition, not a
            # substitution for it.
            by_family: Dict[str, Set[str]] = defaultdict(set)
            for action, fams in action_families.items():
                for fam in fams:
                    by_family[fam].add(action)
            selected = sorted(
                i
                for i in truth
                if not any(
                    plan_actions(truth[i]["time_steps"]) <= repertoire
                    for repertoire in by_family.values()
                )
            )
        else:
            selected = sorted(
                i
                for i in truth
                if any(is_split(p) for p in plan_pairs(truth[i]["time_steps"], mode))
            )
        print(f"compositional rows: {len(selected)}/{len(truth)}")
        if not selected:
            print("no row requires a pair the bank never shows together.\n")
            continue

        print()
        print(f"{'arm':30s} {'compositional':>18s} {'rest':>18s}")
        for arm, values in scores.items():
            inside = [values[i] for i in selected if i in values]
            outside = [values[i] for i in values if i not in set(selected)]
            print(
                f"{arm:30s} "
                f"{sum(inside):5d}/{len(inside):<5d} ({100*sum(inside)/max(1,len(inside)):5.1f}%) "
                f"{sum(outside):5d}/{len(outside):<5d} ({100*sum(outside)/max(1,len(outside)):5.1f}%)"
            )

        print()
        print("McNemar exact on the compositional rows, JSON-tolerant:")
        keep = set(selected)
        for a, b in combinations(scores, 2):
            sub_a = {i: v for i, v in scores[a].items() if i in keep}
            sub_b = {i: v for i, v in scores[b].items() if i in keep}
            result = mcnemar(sub_a, sub_b)
            star = (
                "***" if result["p"] < 0.001
                else "**" if result["p"] < 0.01
                else "*" if result["p"] < 0.05 else ""
            )
            print(
                f"  {a:28s} vs {b:28s} n={result['n']:4d} "
                f"{result['a_only']:4d}/{result['b_only']:<4d} "
                f"p={result['p']:.3g} {star}"
            )
        print()

        out = OUTPUT_DIR / f"compositional.{mode}.json"
        out.write_text(json.dumps({"mode": mode, "rows": selected}, indent=2))
        print(f"row ids written to {out}\n")


if __name__ == "__main__":
    main()
