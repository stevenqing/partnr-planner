#!/usr/bin/env python3
"""What the 49 surviving failures are asking the memory to supply.

The design question is what a skill memory should store, and the honest way to answer
it is to read what is still missing rather than to reason about it. Each failing row
is shown as the difference between the predicates the task really has and the ones the
model stated: what it invented, what it missed, and whether it got the dependency
right. Those three columns are the specification for the memory's goal layer.
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
from viki_amendment11_composer import OUT, SEED, build_metadata, flatten_predicates, load_sim
from viki_amendment11_goalparse import extract_json, to_predicates


def brief(constraints) -> set:
    out = set()
    for predicate in flatten_predicates(constraints):
        status = {k: v for k, v in (predicate.get("status") or {}).items() if v is not None}
        for key, value in status.items():
            out.add(f"{predicate['name']} {'@' if key == 'pos.name' else '!'} {value}")
    return out


def edges(constraints) -> set:
    out = set()
    for constraint in constraints or []:
        stages = [brief(stage) for stage in constraint]
        for earlier, later in zip(stages, stages[1:]):
            for a in sorted(earlier):
                for b in sorted(later):
                    out.add(f"{a}  ->  {b}")
    return out


def main() -> None:
    scorer, SimEnv, Checker, entities, Eval, viki2 = load_sim()
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    scored = pd.read_csv(OUT / "probe2_zeroshot_v4.csv").set_index("index")
    saved = {
        json.loads(line)["index"]: json.loads(line)
        for line in (OUT / "probe2_zeroshot_v2.jsonl").read_text().splitlines()
        if line.strip()
    }

    def places(truth):
        names = set()
        for asset, positions in (truth.get("init_pos") or {}).items():
            if positions is None or (asset.startswith("R") and asset[1:].isdigit()):
                continue
            names.add(asset.rsplit("_", 1)[0])
            names.update(p for p in positions if isinstance(p, str))
        return sorted(names)

    buckets = Counter()
    shown = defaultdict(int)
    for index, row in scored.iterrows():
        if row["accuracy"] == 1.0:
            continue
        truth = bench.get_ground_truth(bench.to_native(frame.iloc[index].to_dict()))
        parsed = extract_json(saved[index].get("raw") or "")
        metadata = build_metadata({k: v for k, v in truth.items() if k != "time_steps"}, viki2, SEED)
        goals, temporal, _ = to_predicates(parsed or {}, sorted(metadata["assets"]), places(truth))
        true_goals, pred_goals = brief(truth["goal_constraints"]), brief(goals or [])
        true_edges, pred_edges = edges(truth["temporal_constraints"]), edges(temporal or [])
        missing, invented = true_goals - pred_goals, pred_goals - true_goals
        label = (
            "missed a goal" if missing
            else "wrong dependency" if true_edges - pred_edges or pred_edges - true_edges
            else "invented a goal" if invented
            else "goals right, plan failed"
        )
        buckets[(row["task_name"], label, row["reason"])] += 1
        if shown[(row["task_name"], label)] < 1:
            shown[(row["task_name"], label)] += 1
            print("=" * 76)
            print(f"[{row['task_name']}] index={index}  {label}  ({row['reason']}, "
                  f"len {row['plan_len']} vs budget {row['budget']})")
            print(f"  desc     : {truth.get('description', '')[:130]}")
            print(f"  true     : {sorted(true_goals)}")
            print(f"  predicted: {sorted(pred_goals)}")
            if missing:
                print(f"  MISSED   : {sorted(missing)}")
            if invented:
                print(f"  INVENTED : {sorted(invented)}")
            print(f"  true dep : {sorted(true_edges)}")
            print(f"  pred dep : {sorted(pred_edges)}")

    print("\n=== residual failures ===")
    for (family, label, reason), count in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3}  {family:<48} {label:<26} {reason}")


if __name__ == "__main__":
    main()
