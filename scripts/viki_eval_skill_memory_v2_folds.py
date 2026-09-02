#!/usr/bin/env python3
"""Skill memory v2 on eight held-out families, each planned by a memory built without it.

Every layer is rebuilt per fold -- operators re-induced, orderings re-mined, vocabulary
re-harvested -- from the training episodes of the other families only. The goal parser
is untouched and its answers are replayed from disk, so a fold's score cannot move
because the model was sampled again, and no fold spends a token.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
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
from viki_eval_skill_memory_v2 import to_predicates, visits_of

OUT = ROOT / "results/viki_memory_experiments/amendment11"


def solve(truth, record, memory, sim):
    blind = {k: v for k, v in truth.items() if k != "time_steps"}
    metadata = sim.metadata(blind, SEED)
    env = sim.world(metadata)
    parsed = extract_json(record.get("raw") or "")
    if parsed is None:
        return 0.0, "UNPARSEABLE"
    goals, _ = to_predicates(parsed, memory, sorted(metadata["assets"]), True)
    if not goals:
        return 0.0, "NO_USABLE_GOAL"
    flat = [group[0] for group in goals]
    temporal = memory.order_for(flat, visits_of(env, flat, memory))
    plan, reason = None, "NO_SCHEDULE"
    for drop in range(0, min(3, len(goals))):
        for subset in combinations(range(len(goals)), len(goals) - drop):
            for keep_order in (True, False):
                blind["goal_constraints"] = [goals[i] for i in subset]
                blind["temporal_constraints"] = temporal if keep_order else []
                plan, reason = planner.plan(blind, memory, sim, SEED)
                if plan:
                    break
            if plan:
                break
        if plan:
            break
    accuracy = sim.score(plan, truth, SEED) if plan else 0.0
    if plan and accuracy == 0.0:
        reason = "OVER_BUDGET" if len(plan) > len(truth["time_steps"]) else "GOAL_UNMET"
    return accuracy, ("SOLVED" if accuracy == 1.0 else reason)


def main() -> None:
    sim = Simulator(BENCHMARK_ROOT)
    episodes = load_episodes(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/train.parquet")
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    records = [json.loads(line) for line in (OUT / "probe2_zeroshot_v2.jsonl").read_text().splitlines() if line.strip()]
    by_family = defaultdict(list)
    for record in records:
        by_family[record["task_name"]].append(record)

    full = SkillMemoryV2.load(OUT / "skill_memory_v2.json")
    rows, totals = [], defaultdict(lambda: [0, 0])
    lost = {}
    for family in sorted(by_family):
        path = OUT / f"skill_memory_v2.fold_{family}.json"
        if path.is_file():
            fold = SkillMemoryV2.load(path)
        else:
            fold = build(episodes, sim, SEED, 250, family)
            fold.save(path)
        lost[family] = {
            "operators": f"{len(fold.operators)}/{len(full.operators)}",
            "coordinated": f"{sum(1 for o in fold.operators if o.get('coordinated'))}/"
                           f"{sum(1 for o in full.operators if o.get('coordinated'))}",
            "order_rules": f"{len(fold.rules)}/{len(full.rules)}",
            "places": f"{len(fold.vocab.get('places', []))}/{len(full.vocab.get('places', []))}",
        }
        for record in by_family[family]:
            truth = bench.get_ground_truth(bench.to_native(frame.iloc[record["index"]].to_dict()))
            for arm, memory in (("fold", fold), ("full", full)):
                accuracy, reason = solve(truth, record, memory, sim)
                totals[(arm, family)][0] += int(accuracy == 1.0)
                totals[(arm, family)][1] += 1
                rows.append({"index": record["index"], "task_name": family,
                             "arm": arm, "accuracy": accuracy, "reason": reason})
        print(f"{family:<48} fold={totals[('fold', family)][0]}/{totals[('fold', family)][1]}"
              f"  full={totals[('full', family)][0]}/{totals[('full', family)][1]}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "v2_folds.csv", index=False)
    print("\n=== skill memory v2, held-out family ===")
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
    print("\n=== what each fold's memory lost ===")
    for family, sizes in lost.items():
        print(f"  {family:<48} {sizes}")
    print(f"\nwrote {OUT / 'v2_folds.csv'}")


if __name__ == "__main__":
    main()
