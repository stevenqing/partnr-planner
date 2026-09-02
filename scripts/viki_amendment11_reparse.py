#!/usr/bin/env python3
"""Re-score a saved probe 2 run through the current parser, spending no tokens.

Every response is kept verbatim in the run's jsonl, so a fix to the parser or the
composer can be priced against the identical model output rather than a fresh sample.
That keeps a parser change from being confounded with generation variance, and it
makes the cost of a parsing bug legible: same answers, different score.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

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
from viki_amendment11_goalparse import (
    extract_json,
    load_vocabulary,
    place_vocabulary,
    to_predicates,
)

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "probe2_zeroshot_v2"
TAG = sys.argv[2] if len(sys.argv) > 2 else SOURCE + "_reparsed"


def main() -> None:
    scorer, SimEnv, Checker, entities, Eval, viki2 = load_sim()
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    records = [json.loads(line) for line in (OUT / f"{SOURCE}.jsonl").read_text().splitlines() if line.strip()]

    operators = load_operators() if "induced" in sys.argv[3:] else None
    print(f"operators: induced, {len(operators)} usable" if operators else "operators: hand-written")
    vocabulary = load_vocabulary("novocab" not in sys.argv[3:])
    print(f"layer 3 vocabulary: {'loaded' if vocabulary else 'not used'}")

    out, reasons, families = [], Counter(), defaultdict(lambda: [0, 0])
    for record in records:
        truth = bench.get_ground_truth(bench.to_native(frame.iloc[record["index"]].to_dict()))
        family = record["task_name"]
        families[family][1] += 1
        row = {"index": record["index"], "task_name": family, "accuracy": 0.0,
               "budget": len(truth["time_steps"]), "plan_len": 0, "reason": "NO_PREDICTION"}
        parsed = extract_json(record.get("raw") or "")
        if parsed is not None:
            metadata = build_metadata({k: v for k, v in truth.items() if k != "time_steps"}, viki2, SEED)
            goals, temporal, snapped = to_predicates(
                parsed, sorted(metadata["assets"]), place_vocabulary(truth, vocabulary)
            )
            row["snapped"] = snapped
            if goals:
                blind = {k: v for k, v in truth.items() if k != "time_steps"}
                plan, reason = None, "NO_SCHEDULE"
                for drop in range(0, min(3, len(goals))):
                    for subset in combinations(range(len(goals)), len(goals) - drop):
                        for keep_order in (True, False):
                            blind["goal_constraints"] = [goals[i] for i in subset]
                            blind["temporal_constraints"] = temporal if keep_order else []
                            plan, reason = plan_for(blind, viki2, SimEnv, Checker, entities, SEED, operators=operators)
                            if plan:
                                break
                        if plan:
                            break
                    if plan:
                        break
                row["reason"] = reason
                if plan:
                    row["plan_len"] = len(plan)
                    row["accuracy"] = score(scorer, plan, truth, SEED)[0]
                    if row["accuracy"] == 0.0:
                        row["reason"] = "OVER_BUDGET" if len(plan) > row["budget"] else "GOAL_UNMET"
                if row["accuracy"] == 1.0:
                    row["reason"] = "SOLVED"
            else:
                row["reason"] = "NO_USABLE_GOAL"
        else:
            row["reason"] = "UNPARSEABLE"
        reasons[row["reason"]] += 1
        families[family][0] += int(row["accuracy"] == 1.0)
        out.append(row)

    pd.DataFrame(out).to_csv(OUT / f"{TAG}.csv", index=False)
    solved = sum(r["accuracy"] for r in out)
    print(f"\n=== {TAG}: same model answers, current parser, {len(out)} rows ===")
    print(f"accuracy  {solved:.0f}/{len(out)} = {solved / len(out) * 100:.2f}%")
    for reason, count in reasons.most_common():
        print(f"  {reason:<24} {count:>5}  {count / len(out) * 100:5.1f}%")
    print("\nby family:")
    for family, (hit, seen) in sorted(families.items(), key=lambda kv: -kv[1][1]):
        print(f"  {family:<48} {hit:>4}/{seen:<4} {hit / seen * 100:5.1f}%")
    print(f"\nwrote {OUT / (TAG + '.csv')}")


if __name__ == "__main__":
    main()
