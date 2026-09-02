#!/usr/bin/env python3
"""What the composer is missing, one failing family at a time.

Prints the goal predicates, the temporal constraint, the drawn asset positions, the
reference plan and the composed plan side by side, with the judge's own refusal code
for the composed one. The reference plan is the answer key for the macro library:
whatever it does that the library cannot express is the next macro.
"""

from __future__ import annotations

import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment11_composer import (
    MANIFEST,
    SEED,
    build_metadata,
    compose,
    flatten_goals,
    load_sim,
    requirement_holds,
)

PER_FAMILY = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ONLY = sys.argv[2] if len(sys.argv) > 2 else None


def main() -> None:
    scorer, SimEnv, Checker, entities, Eval, viki2 = load_sim()
    indices = [json.loads(line)["index"] for line in MANIFEST.read_text().splitlines() if line.strip()]
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")

    shown = defaultdict(int)
    for index in indices:
        truth = bench.get_ground_truth(bench.to_native(frame.iloc[index].to_dict()))
        family = truth["task_name"]
        if ONLY and ONLY not in family:
            continue
        if shown[family] >= PER_FAMILY:
            continue
        shown[family] += 1

        blind = {k: v for k, v in truth.items() if k != "time_steps"}
        metadata = build_metadata(blind, viki2, SEED)
        env = SimEnv(metadata=copy.deepcopy(metadata))

        print("=" * 78)
        print(f"[{family}] index={index}")
        print(f"  description : {truth.get('description', '')[:110]}")
        print(f"  robots      : { {r: t for r, t in truth['robots'].items() if t} }")
        print(f"  drawn assets: { {n: (a.pos.name, 'container' if getattr(a, 'is_container', False) else '', getattr(a, 'container_position', None) and a.container_position.isolated) for n, a in env.assets.items()} }")
        print("  goals:")
        for predicate in flatten_goals(metadata["goal_constraints"]):
            state = "HOLDS" if requirement_holds(env, predicate) else "unmet"
            status = {k: v for k, v in predicate.get("status", {}).items() if v is not None}
            print(f"    [{state}] {predicate['name']:<14} sat={predicate.get('is_satisfied')} {status}")
        print(f"  temporal    : {json.dumps(metadata['temporal_constraints'])[:400]}")
        print("  reference plan:")
        for step in truth["time_steps"]:
            actions = {r: a for r, a in step["actions"].items() if a is not None}
            print(f"    {step['step']:>2}. {actions}")

        plan, reason = compose(blind, viki2, SimEnv, Checker, entities, SEED)
        print(f"  composed ({reason}), len={len(plan) if plan else 0} vs budget {len(truth['time_steps'])}:")
        if plan:
            for step in plan:
                print(f"    {step['step']:>2}. {step['actions']}")
            judge = Eval()
            judge.set_env(build_metadata(blind, viki2, SEED))
            transformed = scorer.transform_actions(plan)
            ok = judge.eval(transformed)
            print(f"  judge: {ok}  code={judge.error_desc_code}")


if __name__ == "__main__":
    main()
