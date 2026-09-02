#!/usr/bin/env python3
"""Which of the model's three decisions is costing the operator-choice arm its score?

Choosing operators asks the model for three things at once: what work the task needs,
which body suits the scene, and who should run it. A wrong answer to any one of them
scores zero, so the arm's number says nothing about which of the three it is bad at.
Each decision is handed back to the memory in turn, on the answers already collected, so
the split costs no tokens and cannot move by resampling.

The variant column is the one to watch. `use when the destination is a container that
starts shut` asks the model to know something the picture may not show and the metadata
certainly does not -- whereas the planner reads it off the world. If that column is where
the score is, the arm is not measuring the model's planning at all; it is measuring
whether the model can guess a hidden state.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import permutations, product
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
from our_method.skill_memory_v2.simulator import state_facts

OUT = ROOT / "results/viki_memory_experiments/amendment11"
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "opchoice_smoke"


def build(memory, catalogue, work, env, robots, pick_variant: bool):
    """The model's work list as chains, optionally letting the memory pick each body."""
    chains, groups = [], 0
    for item in work:
        if not isinstance(item, dict):
            continue
        chosen = catalogue.get(str(item.get("op", "")).strip().upper())
        if chosen is None:
            continue
        subject = memory.canonical_asset(item.get("X"), sorted(env.assets))
        if subject is None:
            continue
        key = chosen["effect"]["key"]
        predicate = {"type": "asset", "name": subject, "is_satisfied": True, "status": {}}
        if key == "pos.name":
            where = item.get("Y")
            if not isinstance(where, str):
                continue
            predicate["status"]["pos.name"] = memory.canonical_place(where)
        elif key == "is_activated":
            predicate["status"]["is_activated"] = True
        else:
            predicate["status"]["pos.name"] = subject  # a repair names only its container

        operator = chosen
        if pick_variant and key in ("pos.name", "is_activated"):
            ranked = memory.operators_for(key, state_facts(env, predicate), coordinated=False)
            if ranked:
                operator = ranked[0]

        binding = {"?x": subject}
        if operator["effect"]["key"] == "pos.name":
            binding["?y"] = predicate["status"].get("pos.name", subject)
        for number, spare in enumerate(memory.spare_variables(operator), 1):
            value = item.get(f"Z{number}")
            bound = memory.canonical_asset(value, sorted(env.assets)) if value else None
            if bound is None:
                fits = [name for name, asset in env.assets.items()
                        if name not in binding.values()
                        and memory.suits(asset, operator.get("types", {}).get(spare, {}))]
                bound = fits[0] if fits else None
            if bound is None:
                continue
            binding[spare] = bound
        crew = [name for name in (item.get("robots") or []) if name in robots]
        if operator.get("coordinated"):
            groups += 1
            group = f"d{groups}"
            crew = crew + [name for name in robots if name not in crew]
            for slot, role in enumerate(operator["roles"]):
                actions = [[e["action"][0]] +
                           [t if t.startswith("?r") else binding.get(t, t) for t in e["action"][1:]]
                           for e in role["actions"]]
                if any(t.startswith("?") and not t.startswith("?r") for a in actions for t in a[1:]):
                    break
                chains.append({"actions": actions, "after": [e["after"] for e in role["actions"]],
                               "guard": [], "group": group, "role": slot,
                               "robot": crew[slot] if slot < len(crew) else robots[0]})
        else:
            body = [[a[0]] + [binding.get(t, t) for t in a[1:]] for a in operator["body"]]
            if any(t.startswith("?") for a in body for t in a[1:]):
                continue
            chains.append({"actions": body, "guard": [], "robot": crew[0] if crew else robots[0]})
    return chains


def schedule_as_given(metadata, chains, sim):
    plans = defaultdict(list)
    for chain in chains:
        plans[chain["robot"]].append({k: v for k, v in chain.items() if k != "robot"})
    return planner.schedule(metadata, dict(plans), sim)


def schedule_by_search(metadata, chains, sim, robots):
    best = None
    stripped = [{k: v for k, v in chain.items() if k != "robot"} for chain in chains]
    for order in permutations(range(len(stripped))):
        ordered = [stripped[i] for i in order]
        for assignment in product(robots, repeat=len(ordered)):
            taken, clash = {}, False
            for chain, robot in zip(ordered, assignment):
                if chain.get("group") is None:
                    continue
                if (chain["group"], robot) in taken:
                    clash = True
                    break
                taken[(chain["group"], robot)] = chain["role"]
            if clash:
                continue
            plans = defaultdict(list)
            for chain, robot in zip(ordered, assignment):
                plans[robot].append(chain)
            steps = planner.schedule(metadata, dict(plans), sim)
            if steps is not None and (best is None or len(steps) < len(best)):
                best = steps
    return best


def main() -> None:
    sim = Simulator(BENCHMARK_ROOT)
    memory = SkillMemoryV2.load(OUT / "skill_memory_v2.json")
    _, catalogue = memory.menu()
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    records = [json.loads(l) for l in (OUT / f"{SOURCE}.jsonl").read_text().splitlines() if l.strip()]

    arms = {
        "as the model said": (False, False),
        "memory picks the body": (True, False),
        "memory casts the crew": (False, True),
        "memory does both": (True, True),
    }
    totals = {name: 0 for name in arms}
    reasons = {name: Counter() for name in arms}
    families = {name: defaultdict(lambda: [0, 0]) for name in arms}

    for record in records:
        truth = bench.get_ground_truth(bench.to_native(frame.iloc[record["index"]].to_dict()))
        blind = {k: v for k, v in truth.items() if k != "time_steps"}
        metadata = sim.metadata(blind, SEED)
        robots = list(metadata["agents"])
        parsed = extract_json(record.get("raw") or "")
        work = (parsed or {}).get("work") if isinstance(parsed, dict) else None
        family = record["task_name"]
        for name, (pick_variant, search) in arms.items():
            families[name][family][1] += 1
            if not isinstance(work, list):
                reasons[name]["UNPARSEABLE"] += 1
                continue
            env = sim.world(metadata)
            chains = build(memory, catalogue, work, env, robots, pick_variant)
            if not chains:
                reasons[name]["NO_USABLE_CHOICE"] += 1
                continue
            steps = (schedule_by_search(metadata, chains, sim, robots) if search
                     else schedule_as_given(metadata, chains, sim))
            if steps is None:
                reasons[name]["NO_SCHEDULE"] += 1
                continue
            accuracy = sim.score(steps, truth, SEED)
            if accuracy == 1.0:
                totals[name] += 1
                families[name][family][0] += 1
                reasons[name]["SOLVED"] += 1
            else:
                reasons[name]["OVER_BUDGET" if len(steps) > len(truth["time_steps"]) else "GOAL_UNMET"] += 1

    print(f"=== where the operator-choice arm loses its score ({len(records)} rows) ===\n")
    for name in arms:
        print(f"{name:<26} {totals[name]:>4}/{len(records)}  {totals[name] / len(records) * 100:5.1f}%   "
              + "  ".join(f"{k}={v}" for k, v in reasons[name].most_common()))
    print("\nby family:")
    header = f"{'family':<48}" + "".join(f"{name[:18]:>20}" for name in arms)
    print(header)
    for family in sorted(families["as the model said"]):
        line = f"{family:<48}"
        for name in arms:
            hit, total = families[name][family]
            line += f"{hit:>13}/{total:<6}"
        print(line)


if __name__ == "__main__":
    main()
