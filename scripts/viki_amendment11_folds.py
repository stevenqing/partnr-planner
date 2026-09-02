#!/usr/bin/env python3
"""Eight held-out-family folds: does a memory built without the family still plan it?

This is the only claim on this line that has not been tested. The written macros cover
exactly the eight families of this split because they were written after reading them,
so a number they produce says nothing about a family nobody prepared for. An induced
library is supposed to be different in kind: the delivery body it learned from cutting
fruit is the same body clearing a table needs, so holding out a family should cost
little. Holding out the relay should cost everything, because the relay is stated in one
family alone. Both halves of that prediction are checked here.

Four arms per fold, chosen so a loss can be attributed rather than merely observed:

  fold memory      operators and vocabulary both rebuilt without the family. The claim.
  full memory      both built from everything. The ceiling this fold could have reached.
  written macros   the hand-written bodies with the fold's vocabulary; they know the
                   family, so this is the upper bound an operator library is chasing --
                   and beating it is not possible, only matching it.
  fold ops, full vocabulary
                   separates a loss in the operator library from a loss in the names.

The goal parser is untouched: it is zero-shot, it never saw any family, and its answers
are the ones already on disk. So no fold here spends a token, and no fold's score can be
moved by resampling the model.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment11_composer import (
    OUT,
    SEED,
    build_metadata,
    load_operators,
    load_sim,
    plan_for,
    score,
)
from viki_amendment11_goalparse import extract_json, place_vocabulary, to_predicates

SOURCE = "probe2_zeroshot_v2"


def solve(truth, parsed, metadata, vocabulary, operators, viki2, SimEnv, Checker, entities):
    goals, temporal, _ = to_predicates(
        parsed, sorted(metadata["assets"]), place_vocabulary(truth, vocabulary)
    )
    if not goals:
        return 0.0, "NO_USABLE_GOAL", 0
    blind = {k: v for k, v in truth.items() if k != "time_steps"}
    plan, reason = None, "NO_SCHEDULE"
    for drop in range(0, min(3, len(goals))):
        for subset in combinations(range(len(goals)), len(goals) - drop):
            for keep_order in (True, False):
                blind["goal_constraints"] = [goals[i] for i in subset]
                blind["temporal_constraints"] = temporal if keep_order else []
                plan, reason = plan_for(
                    blind, viki2, SimEnv, Checker, entities, SEED, operators=operators
                )
                if plan:
                    break
            if plan:
                break
        if plan:
            break
    return plan, reason, goals


def main() -> None:
    scorer, SimEnv, Checker, entities, Eval, viki2 = load_sim()
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    saved = [
        json.loads(line)
        for line in (OUT / f"{SOURCE}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in saved:
        by_family[record["task_name"]].append(record)

    full_operators = load_operators()
    full_vocabulary = json.loads((OUT / "vocabulary.json").read_text())

    arms = ["fold memory", "full memory", "written macros", "fold ops, full vocabulary"]
    results: Dict[str, Dict[str, List[int]]] = {arm: {} for arm in arms}
    library_sizes: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    for family in sorted(by_family):
        operators_path = OUT / f"operators.fold_{family}.json"
        vocabulary_path = OUT / f"vocabulary.fold_{family}.json"
        if not operators_path.is_file() or not vocabulary_path.is_file():
            print(f"skipping {family}: fold memory not built")
            continue
        fold_operators = load_operators(operators_path)
        fold_vocabulary = json.loads(vocabulary_path.read_text())
        library_sizes[family] = {
            "fold_operators": len(fold_operators or []),
            "full_operators": len(full_operators or []),
            "fold_coordinated": sum(1 for item in (fold_operators or []) if item.get("coordinated")),
            "full_coordinated": sum(1 for item in (full_operators or []) if item.get("coordinated")),
            "fold_places": len(fold_vocabulary.get("places", [])),
            "full_places": len(full_vocabulary.get("places", [])),
        }
        settings = {
            "fold memory": (fold_operators, fold_vocabulary),
            "full memory": (full_operators, full_vocabulary),
            "written macros": (None, fold_vocabulary),
            "fold ops, full vocabulary": (fold_operators, full_vocabulary),
        }
        for arm in arms:
            results[arm][family] = [0, 0]

        for record in by_family[family]:
            truth = bench.get_ground_truth(bench.to_native(frame.iloc[record["index"]].to_dict()))
            metadata = build_metadata(
                {k: v for k, v in truth.items() if k != "time_steps"}, viki2, SEED
            )
            parsed = extract_json(record.get("raw") or "")
            row = {"index": record["index"], "task_name": family}
            for arm in arms:
                operators, vocabulary = settings[arm]
                accuracy = 0.0
                if parsed is not None:
                    plan, reason, _ = solve(
                        truth, parsed, metadata, vocabulary, operators,
                        viki2, SimEnv, Checker, entities,
                    )
                    if plan:
                        accuracy = score(scorer, plan, truth, SEED)[0]
                results[arm][family][0] += int(accuracy == 1.0)
                results[arm][family][1] += 1
                row[arm] = accuracy
            rows.append(row)
        done = {arm: results[arm][family] for arm in arms}
        print(f"{family:<48} " + "  ".join(
            f"{arm.split(',')[0][:9]}={done[arm][0]}/{done[arm][1]}" for arm in arms
        ), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "folds_layer1.csv", index=False)

    print("\n=== held-out family, four arms ===")
    header = f"{'family':<48}" + "".join(f"{arm:>28}" for arm in arms)
    print(header)
    for family in sorted(results[arms[0]]):
        line = f"{family:<48}"
        for arm in arms:
            hit, total = results[arm][family]
            line += f"{hit:>13}/{total:<4}{hit / total * 100:>8.1f}%"
        print(line)
    line = f"{'ALL':<48}"
    for arm in arms:
        hit = sum(v[0] for v in results[arm].values())
        total = sum(v[1] for v in results[arm].values())
        line += f"{hit:>13}/{total:<4}{hit / total * 100:>8.1f}%"
    print(line)

    print("\n=== what each fold's memory lost ===")
    for family, sizes in library_sizes.items():
        print(f"  {family:<48} operators {sizes['fold_operators']}/{sizes['full_operators']}"
              f"  (coordinated {sizes['fold_coordinated']}/{sizes['full_coordinated']})"
              f"  places {sizes['fold_places']}/{sizes['full_places']}")
    print(f"\nwrote {OUT / 'folds_layer1.csv'}")


if __name__ == "__main__":
    main()
