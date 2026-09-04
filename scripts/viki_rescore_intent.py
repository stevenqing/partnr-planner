#!/usr/bin/env python3
"""Re-score an archived intent run offline, under either delegation arm, on any split.

The intent prompt is one fixed constant and carries a `robots` field, so a run collected
once can be scored as memory-dispatch or as model-dispatch without asking the model
anything. Everything this prints therefore costs no tokens and cannot move because the
model was sampled again -- which is the only way a 24-point difference can be attributed
to delegation rather than to a resample.

It also records what the delegation costs concretely: the plan length, whether the plan
fits the benchmark's `len(gt)/len(pred) >= 0.99` budget, and how often the model's casting
had to be abandoned for a free search.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment11_goalparse import extract_json
from our_method.skill_memory_v2 import SEED, SkillMemoryV2, Simulator
from viki_eval_skill_memory_v2 import visits_of
from viki_eval_v2_intent_choice import SPLITS, to_requirement
from viki_intent_crew import CREW_CHOICES, collect, solve

OUT = ROOT / "results/viki_memory_experiments/amendment11"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True, help="tag of an archived intent run")
    parser.add_argument("--crew", choices=CREW_CHOICES, default="memory")
    parser.add_argument("--split", choices=sorted(SPLITS), default="id")
    parser.add_argument("--memory", type=Path, default=OUT / "skill_memory_v2.json")
    parser.add_argument("--tag", default=None)
    arguments = parser.parse_args()

    sim = Simulator(BENCHMARK_ROOT)
    memory = SkillMemoryV2.load(arguments.memory)
    if arguments.split == "id":
        frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    else:
        frame = pd.read_parquet(
            ROOT / "results/viki_memory_experiments/amendment10" / SPLITS[arguments.split]
        )
    records = [json.loads(l) for l in
               (OUT / f"{arguments.responses}.jsonl").read_text().splitlines() if l.strip()]
    use_crew = arguments.crew == "model"

    rows, reasons, recasts = [], Counter(), 0
    for record in records:
        truth = bench.get_ground_truth(bench.to_native(frame.iloc[record["index"]].to_dict()))
        blind = {k: v for k, v in truth.items() if k != "time_steps"}
        metadata = sim.metadata(blind, SEED)
        row = {"index": record["index"], "task_name": record.get("task_name", "?"),
               "accuracy": 0.0, "plan_len": 0, "budget": len(truth["time_steps"]),
               "recast": False}
        parsed = extract_json(record.get("raw") or "")
        work = (parsed or {}).get("work") if isinstance(parsed, dict) else None
        if not isinstance(work, list):
            row["reason"] = "UNPARSEABLE"
            reasons["UNPARSEABLE"] += 1
            rows.append(row)
            continue
        scene = sorted(metadata["assets"])
        requirements, crew = collect(work, to_requirement, memory, scene, metadata)
        if not requirements:
            row["reason"] = "NO_USABLE_WORK"
            reasons["NO_USABLE_WORK"] += 1
            rows.append(row)
            continue
        env = sim.world(metadata)
        temporal = memory.order_for(requirements, visits_of(env, requirements, memory))
        plan, reason, recast = solve(
            blind, memory, sim, SEED, requirements, crew, temporal, use_crew
        )
        recasts += int(recast)
        accuracy = sim.score(plan, truth, SEED) if plan else 0.0
        if plan and accuracy == 0.0:
            reason = "OVER_BUDGET" if len(plan) > len(truth["time_steps"]) else "GOAL_UNMET"
        row.update({"accuracy": accuracy, "plan_len": len(plan) if plan else 0,
                    "recast": recast,
                    "reason": "SOLVED" if accuracy == 1.0 else reason})
        reasons[row["reason"]] += 1
        rows.append(row)

    table = pd.DataFrame(rows)
    tag = arguments.tag or f"{arguments.responses}_crew-{arguments.crew}"
    table.to_csv(OUT / f"{tag}.csv", index=False)
    hit = int(table["accuracy"].sum())
    n = len(table)
    planned = table[table["plan_len"] > 0]
    print(f"\n=== {tag} ===")
    print(f"responses  {arguments.responses}.jsonl   split {arguments.split}   crew {arguments.crew}")
    print(f"accuracy   {hit}/{n} = {hit / n * 100:.2f}%")
    if len(planned):
        within = int((planned["plan_len"] <= planned["budget"]).sum())
        print(f"plan length  mean {planned['plan_len'].mean():.2f}  "
              f"vs reference {planned['budget'].mean():.2f}   "
              f"within budget {within}/{len(planned)} = {within / len(planned) * 100:.1f}%")
    if use_crew:
        print(f"recast (model casting unusable)  {recasts}/{n} = {recasts / n * 100:.1f}%")
    for reason, count in reasons.most_common():
        print(f"  {reason:<22} {count:>5}  {count / n * 100:5.1f}%")
    print(f"\nwrote {OUT / (tag + '.csv')}")


if __name__ == "__main__":
    main()
