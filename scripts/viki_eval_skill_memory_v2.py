#!/usr/bin/env python3
"""Evaluate skill memory v2 end to end on the VIKI-L2 interactive split.

The model is asked only what must become true; the memory supplies everything else. The
responses were collected once and are replayed from disk, so every configuration below
is scored against identical model output and no difference between them can be an
artefact of resampling.

`--order memory` is the point of the exercise. Until now the ordering came from the
model, prompted by a rule written after seeing which families failed -- honest enough to
report but not honest enough to publish. With Layer 2 the ordering is derived from
patterns mined off the training episodes' own temporal constraints, so the model's
answer about order is discarded and the rule is no longer anybody's hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment11_goalparse import extract_json
from our_method.skill_memory_v2 import SEED, SkillMemoryV2, Simulator
from our_method.skill_memory_v2 import planner
from our_method.skill_memory_v2.simulator import predicate_status

OUT = ROOT / "results/viki_memory_experiments/amendment11"


def to_predicates(parsed: Dict[str, Any], memory: SkillMemoryV2, scene_assets: List[str],
                  ground: bool) -> Tuple[List[Any], List[Any]]:
    """The model's answer in the judge's schema, with names grounded by Layer 3."""
    def predicate(item: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        raw = item.get("asset") or item.get("name")
        asset = memory.canonical_asset(raw, scene_assets) if ground else (
            raw if raw in scene_assets else None
        )
        if asset is None:
            return None
        status: Dict[str, Any] = {}
        if item.get("activated") is True or item.get("is_activated") is True:
            status["is_activated"] = True
        where = item.get("at") or item.get("pos") or (item.get("status") or {}).get("pos.name")
        if isinstance(where, str) and where.strip():
            status["pos.name"] = memory.canonical_place(where) if ground else where.strip()
        return {"type": "asset", "name": asset, "is_satisfied": True, "status": status} if status else None

    goals = [p for p in (predicate(item) for item in parsed.get("goals") or []) if p]

    def stages(constraint):
        out = []
        for stage in constraint if isinstance(constraint, list) else []:
            items = stage if isinstance(stage, list) else [stage]
            built = [p for p in (predicate(item) for item in items) if p]
            if built:
                out.append(built)
        return out

    order = parsed.get("order") or []
    nested = isinstance(order, list) and order and all(
        isinstance(e, list) and e and all(isinstance(i, list) for i in e) for e in order
    )
    temporal = []
    for constraint in (order if nested else [order]):
        built = stages(constraint)
        if len(built) >= 2:
            temporal.append(built)
    return [[goal] for goal in goals], temporal


def visits_of(env, requirements: List[Dict[str, Any]], memory: SkillMemoryV2) -> Dict[int, set]:
    """Which objects the body chosen for each requirement would address.

    Layer 2's sharpest pattern needs this: the fruit must be on the board because the
    knife is used at the board, and those two predicates share no argument at all. Only
    the operator body knows the board is involved.
    """
    from our_method.skill_memory_v2.simulator import state_facts

    out: Dict[int, set] = {}
    for index, predicate in enumerate(requirements):
        status = predicate_status(predicate)
        key = "pos.name" if "pos.name" in status else "is_activated"
        if predicate["name"] not in env.assets:
            out[index] = set()
            continue
        # Every body the memory would consider, not just the cheapest. Mining read the
        # targets off the body the reference plan actually ran, and at planning time
        # which body runs is not settled until the schedule is laid out -- so the
        # question the pattern really asks is whether any body for this effect goes
        # there.
        candidates = memory.operators_for(key, state_facts(env, predicate), coordinated=False)
        targets = set()
        for operator in candidates[:6]:
            for action in operator.get("body", []):
                for item in action[1:]:
                    if item == "?x":
                        targets.add(predicate["name"])
                    elif item == "?y" and isinstance(status.get("pos.name"), str):
                        targets.add(status["pos.name"])
                    elif item.startswith("?z"):
                        wants = operator.get("types", {}).get(item, {})
                        targets.update(
                            name for name, asset in env.assets.items() if memory.suits(asset, wants)
                        )
        out[index] = targets
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path,
                        default=OUT / "skill_memory_v2.json")
    parser.add_argument("--responses", default="probe2_zeroshot_v2")
    parser.add_argument("--order", choices=["model", "memory"], default="memory")
    parser.add_argument("--no-grounding", action="store_true",
                        help="skip Layer 3 and take the model's spelling as written")
    parser.add_argument("--oracle-goals", action="store_true",
                        help="hand the true predicates over instead of the model's")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()

    sim = Simulator(BENCHMARK_ROOT)
    memory = SkillMemoryV2.load(arguments.memory)
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/train.parquet".replace("train", "test"))
    records = [json.loads(line) for line in (OUT / f"{arguments.responses}.jsonl").read_text().splitlines() if line.strip()]
    if arguments.limit:
        records = records[: arguments.limit]

    rows, reasons, families = [], Counter(), defaultdict(lambda: [0, 0])
    for record in records:
        truth = bench.get_ground_truth(bench.to_native(frame.iloc[record["index"]].to_dict()))
        blind = {k: v for k, v in truth.items() if k != "time_steps"}
        metadata = sim.metadata(blind, SEED)
        env = sim.world(metadata)
        family = record["task_name"]

        if arguments.oracle_goals:
            goals, temporal = truth["goal_constraints"], truth["temporal_constraints"]
        else:
            parsed = extract_json(record.get("raw") or "")
            if parsed is None:
                reasons["UNPARSEABLE"] += 1
                families[family][1] += 1
                rows.append({"index": record["index"], "task_name": family, "accuracy": 0.0,
                             "reason": "UNPARSEABLE"})
                continue
            goals, temporal = to_predicates(
                parsed, memory, sorted(metadata["assets"]), not arguments.no_grounding
            )
            if arguments.order == "memory":
                flat = [group[0] for group in goals]
                temporal = memory.order_for(flat, visits_of(env, flat, memory))

        plan, reason = None, "NO_SCHEDULE"
        for drop in range(0, min(3, max(1, len(goals)))):
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
        outcome = "SOLVED" if accuracy == 1.0 else reason
        reasons[outcome] += 1
        families[family][0] += int(accuracy == 1.0)
        families[family][1] += 1
        rows.append({"index": record["index"], "task_name": family, "accuracy": accuracy,
                     "plan_len": len(plan) if plan else 0,
                     "budget": len(truth["time_steps"]), "reason": outcome})

    tag = arguments.tag or f"v2_{arguments.order}order" + ("_nogrounding" if arguments.no_grounding else "")
    pd.DataFrame(rows).to_csv(OUT / f"{tag}.csv", index=False)
    solved = sum(row["accuracy"] for row in rows)
    print(f"\n=== {tag}: skill memory v2, {len(rows)} rows ===")
    print(f"memory     {arguments.memory.name}")
    print(f"goals      {'oracle' if arguments.oracle_goals else 'model'}   "
          f"order {arguments.order}   grounding {'off' if arguments.no_grounding else 'Layer 3'}")
    print(f"accuracy   {solved:.0f}/{len(rows)} = {solved / len(rows) * 100:.2f}%")
    for reason, count in reasons.most_common():
        print(f"  {reason:<22} {count:>5}  {count / len(rows) * 100:5.1f}%")
    print("\nby family:")
    for family, (hit, total) in sorted(families.items(), key=lambda kv: -kv[1][1]):
        print(f"  {family:<48} {hit:>4}/{total:<4} {hit / total * 100:5.1f}%")
    print(f"\nwrote {OUT / (tag + '.csv')}")


if __name__ == "__main__":
    main()
