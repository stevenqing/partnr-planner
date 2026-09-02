#!/usr/bin/env python3
"""Held-out family, for the intent interface, on answers already collected.

Each family is planned by a memory rebuilt without it -- operators re-induced, orderings
re-mined, vocabulary re-harvested from the other families' training episodes. The model's
answers are replayed from the run file, so a fold costs nothing and cannot move because
the model was sampled again. Run it once per model over that model's own answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment11_goalparse import extract_json
from our_method.skill_memory_v2 import SEED, SkillMemoryV2, Simulator, planner
from our_method.skill_memory_v2.build import build, load_episodes
from viki_eval_skill_memory_v2 import visits_of
from viki_eval_v2_intent_choice import to_requirement

OUT = ROOT / "results/viki_memory_experiments/amendment11"


def solve(truth, record, memory, sim):
    blind = {k: v for k, v in truth.items() if k != "time_steps"}
    metadata = sim.metadata(blind, SEED)
    env = sim.world(metadata)
    parsed = extract_json(record.get("raw") or "")
    work = (parsed or {}).get("work") if isinstance(parsed, dict) else None
    if not isinstance(work, list):
        return 0.0, "UNPARSEABLE"
    scene = sorted(metadata["assets"])
    requirements = [r for r in (to_requirement(memory, item, scene)
                                for item in work if isinstance(item, dict)) if r]
    if not requirements:
        return 0.0, "NO_USABLE_WORK"
    blind["goal_constraints"] = [[requirement] for requirement in requirements]
    blind["temporal_constraints"] = memory.order_for(requirements, visits_of(env, requirements, memory))
    plan, reason = planner.plan(blind, memory, sim, SEED)
    if plan is None:
        blind["temporal_constraints"] = []
        plan, reason = planner.plan(blind, memory, sim, SEED)
    accuracy = sim.score(plan, truth, SEED) if plan else 0.0
    if plan and accuracy == 0.0:
        reason = "OVER_BUDGET" if len(plan) > len(truth["time_steps"]) else "GOAL_UNMET"
    return accuracy, ("SOLVED" if accuracy == 1.0 else reason)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True, help="tag of an intent run's jsonl")
    parser.add_argument("--tag", default=None)
    arguments = parser.parse_args()

    sim = Simulator(BENCHMARK_ROOT)
    episodes = load_episodes(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/train.parquet")
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    records = [json.loads(l) for l in (OUT / f"{arguments.responses}.jsonl").read_text().splitlines() if l.strip()]
    by_family = defaultdict(list)
    for record in records:
        by_family[record["task_name"]].append(record)

    full = SkillMemoryV2.load(OUT / "skill_memory_v2.json")
    rows, totals = [], defaultdict(lambda: [0, 0])
    for family in sorted(by_family):
        path = OUT / f"skill_memory_v2.fold_{family}.json"
        fold = SkillMemoryV2.load(path) if path.is_file() else build(episodes, sim, SEED, 250, family)
        if not path.is_file():
            fold.save(path)
        for record in by_family[family]:
            truth = bench.get_ground_truth(bench.to_native(frame.iloc[record["index"]].to_dict()))
            for arm, memory in (("fold", fold), ("full", full)):
                accuracy, reason = solve(truth, record, memory, sim)
                totals[(arm, family)][0] += int(accuracy == 1.0)
                totals[(arm, family)][1] += 1
                rows.append({"index": record["index"], "task_name": family, "arm": arm,
                             "accuracy": accuracy, "reason": reason})
        print(f"{family:<48} fold={totals[('fold', family)][0]}/{totals[('fold', family)][1]}"
              f"  full={totals[('full', family)][0]}/{totals[('full', family)][1]}", flush=True)

    tag = arguments.tag or f"{arguments.responses}_folds"
    pd.DataFrame(rows).to_csv(OUT / f"{tag}.csv", index=False)
    print(f"\n=== {tag}: held-out family, intent interface ===")
    print(f"{'family':<48}{'fold memory':>22}{'full memory':>22}")
    for family in sorted(by_family):
        line = f"{family:<48}"
        for arm in ("fold", "full"):
            hit, total = totals[(arm, family)]
            line += f"{hit:>12}/{total:<4}{hit / total * 100:>5.1f}%"
        print(line)
    line = f"{'ALL':<48}"
    for arm in ("fold", "full"):
        hit = sum(v[0] for (a, _), v in totals.items() if a == arm)
        total = sum(v[1] for (a, _), v in totals.items() if a == arm)
        line += f"{hit:>12}/{total:<4}{hit / total * 100:>5.1f}%"
    print(line)
    print(f"\nwrote {OUT / (tag + '.csv')}")


if __name__ == "__main__":
    main()
