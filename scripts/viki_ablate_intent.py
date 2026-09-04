#!/usr/bin/env python3
"""Layer ablations for the intent interface, re-scored offline on saved answers.

Every configuration below sees the identical model output, so a difference between them
is the memory layer and nothing else -- not a resample, not a prompt change. Run once per
model over that model's own run file; it costs no tokens.

  full            all three layers
  no order        Layer 2 removed: the requirements are planned in any order the
                  scheduler finds feasible
  no grounding    Layer 3 removed: names are taken as the model spelled them
  no order+ground both removed, which is the memory reduced to its operator library
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment11_goalparse import extract_json
from our_method.skill_memory_v2 import SEED, SkillMemoryV2, Simulator, planner
from viki_eval_skill_memory_v2 import visits_of
from viki_eval_v2_intent_choice import to_requirement
from viki_intent_crew import CREW_CHOICES, casting_of

OUT = ROOT / "results/viki_memory_experiments/amendment11"
ARMS = {"full": (True, True), "no order": (False, True),
        "no grounding": (True, False), "no order+grounding": (False, False)}


def score_one(truth, record, memory, sim, use_order: bool, ground: bool,
              use_crew: bool = False) -> float:
    blind = {k: v for k, v in truth.items() if k != "time_steps"}
    metadata = sim.metadata(blind, SEED)
    env = sim.world(metadata)
    parsed = extract_json(record.get("raw") or "")
    work = (parsed or {}).get("work") if isinstance(parsed, dict) else None
    if not isinstance(work, list):
        return 0.0
    scene = sorted(metadata["assets"])
    requirements, crew = [], []
    for item in work:
        if not isinstance(item, dict):
            continue
        if ground:
            requirement = to_requirement(memory, item, scene)
        else:
            raw = item.get("X")
            kind = str(item.get("do", "")).strip().lower()
            where = item.get("Y")
            if raw not in scene or kind not in ("put", "use", "open"):
                requirement = None
            elif kind == "put" and isinstance(where, str):
                requirement = {"type": "asset", "name": raw, "is_satisfied": True,
                               "status": {"pos.name": where.strip()}}
            elif kind == "use":
                requirement = {"type": "asset", "name": raw, "is_satisfied": True,
                               "status": {"is_activated": True}}
            else:
                requirement = None
        if requirement:
            requirements.append(requirement)
            crew.append([n for n in (item.get("robots") or []) if n in metadata["agents"]])
    if not requirements:
        return 0.0
    casting = casting_of(requirements, crew, use_crew)
    blind["goal_constraints"] = [[requirement] for requirement in requirements]
    blind["temporal_constraints"] = (
        memory.order_for(requirements, visits_of(env, requirements, memory)) if use_order else []
    )
    plan, _ = planner.plan(blind, memory, sim, SEED, crew=casting)
    if plan is None and use_order:
        blind["temporal_constraints"] = []
        plan, _ = planner.plan(blind, memory, sim, SEED, crew=casting)
    if plan is None and casting:
        # A casting the world cannot honour falls back to a free search, as in the run
        # script; without this the ablation would punish the arm for the scheduler.
        blind["temporal_constraints"] = (
            memory.order_for(requirements, visits_of(env, requirements, memory)) if use_order else []
        )
        plan, _ = planner.plan(blind, memory, sim, SEED)
    return sim.score(plan, truth, SEED) if plan else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True)
    parser.add_argument("--memory", type=Path, default=OUT / "skill_memory_v2.json")
    parser.add_argument("--crew", choices=CREW_CHOICES, default="memory",
                        help="who assigns the robots: the memory's search, or the model")
    parser.add_argument("--tag", default=None)
    arguments = parser.parse_args()

    sim = Simulator(BENCHMARK_ROOT)
    memory = SkillMemoryV2.load(arguments.memory)
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    records = [json.loads(l) for l in (OUT / f"{arguments.responses}.jsonl").read_text().splitlines() if l.strip()]

    rows = []
    for record in records:
        truth = bench.get_ground_truth(bench.to_native(frame.iloc[record["index"]].to_dict()))
        row = {"index": record["index"], "task_name": record["task_name"]}
        for arm, (use_order, ground) in ARMS.items():
            row[arm] = score_one(truth, record, memory, sim, use_order, ground,
                                 arguments.crew == "model")
        rows.append(row)

    table = pd.DataFrame(rows)
    tag = arguments.tag or f"{arguments.responses}_ablation_crew-{arguments.crew}"
    table.to_csv(OUT / f"{tag}.csv", index=False)
    print(f"=== {tag}: layer ablation on {len(table)} rows, same answers throughout ===")
    print(f"  crew: {arguments.crew}")
    for arm in ARMS:
        hit = int(table[arm].sum())
        print(f"  {arm:<20} {hit:>4}/{len(table)} = {hit / len(table) * 100:6.2f}%")
    print(f"\nwrote {OUT / (tag + '.csv')}")


if __name__ == "__main__":
    main()
