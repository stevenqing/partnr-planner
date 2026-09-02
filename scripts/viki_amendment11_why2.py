#!/usr/bin/env python3
"""Where probe 2's predicted goals cost the plan, using only what is already saved.

The predictions are on disk, so the two suspicions can be priced without spending a
single further token. First: the model was asked what must happen in what order and
answered generously, and an order the task does not actually impose serialises work
that could have run in parallel, which costs steps against a budget measured in steps.
Re-composing the same predicted goals with the predicted order discarded separates
that cost from every other. Second: the rows the composer refused outright are dumped
in full, since a refusal means the stated goals were unreachable rather than wrong.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment11_composer import OUT, SEED, compose, load_sim, score

TAG = sys.argv[1] if len(sys.argv) > 1 else "probe2_zeroshot"


def main() -> None:
    scorer, SimEnv, Checker, entities, Eval, viki2 = load_sim()
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    records = [json.loads(line) for line in (OUT / f"{TAG}.jsonl").read_text().splitlines() if line.strip()]

    variants = {"as predicted": True, "order discarded": False}
    totals = {name: 0 for name in variants}
    families = {name: defaultdict(lambda: [0, 0]) for name in variants}
    reasons = {name: Counter() for name in variants}
    refusals = []

    for record in records:
        predicted = record.get("predicted")
        truth = bench.get_ground_truth(bench.to_native(frame.iloc[record["index"]].to_dict()))
        family = record["task_name"]
        if not predicted:
            for name in variants:
                families[name][family][1] += 1
                reasons[name]["NO_PREDICTION"] += 1
            continue
        for name, keep_order in variants.items():
            blind = {k: v for k, v in truth.items() if k != "time_steps"}
            blind["goal_constraints"] = predicted["goal_constraints"]
            blind["temporal_constraints"] = predicted["temporal_constraints"] if keep_order else []
            plan, reason = compose(blind, viki2, SimEnv, Checker, entities, SEED)
            accuracy = score(scorer, plan, truth, SEED)[0] if plan else 0.0
            if plan and accuracy == 0.0:
                reason = "OVER_BUDGET" if len(plan) > len(truth["time_steps"]) else "GOAL_UNMET"
            reasons[name]["SOLVED" if accuracy else reason] += 1
            totals[name] += int(accuracy == 1.0)
            families[name][family][0] += int(accuracy == 1.0)
            families[name][family][1] += 1
            if name == "as predicted" and reason in ("NO_SCHEDULE", "UNSUPPORTED_PREDICATE") and len(refusals) < 4:
                refusals.append(
                    {
                        "family": family,
                        "index": record["index"],
                        "description": truth.get("description", "")[:120],
                        "true_goals": truth["goal_constraints"],
                        "predicted_goals": predicted["goal_constraints"],
                        "reason": reason,
                    }
                )

    total = len(records)
    for name in variants:
        print(f"\n=== {TAG} :: {name} ===")
        print(f"accuracy  {totals[name]}/{total} = {totals[name] / total * 100:.2f}%")
        for reason, count in reasons[name].most_common():
            print(f"  {reason:<24} {count:>5}  {count / total * 100:5.1f}%")
        print("  by family:")
        for fam, (hit, seen) in sorted(families[name].items(), key=lambda kv: -kv[1][1]):
            print(f"    {fam:<48} {hit:>4}/{seen:<4} {hit / seen * 100:5.1f}%")

    print("\n=== rows the composer refused ===")
    for item in refusals:
        print(json.dumps(item, indent=2)[:1200])


if __name__ == "__main__":
    main()
