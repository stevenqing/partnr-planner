#!/usr/bin/env python3
"""Why is our arm 0/220 on clear_table when the answer is in the bank?

In distribution, with the family's own episodes present, G-Memory scores 39.1% on
clear_table_with_two_robots_and_put_in_cabinet and we score zero. Nothing about
transfer or abstraction explains that: the material is there and one arm uses it.
Whatever is wrong is inside our pipeline, so this reads the rendered memory and the
emitted plan for those rows and asks where they diverge from the ground truth.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import (
    EXPOSED_STEPS,
    OUTPUT_DIR,
    SELF_ROBOT,
    SOURCE_PARQUET,
    load_manifest,
    native,
)
from viki_amendment9_diag102 import canon, load_rows, parse_plan, verbs
from viki_amendment9_folds import rows_of

FAMILY = sys.argv[1] if len(sys.argv) > 1 else (
    "clear_table_with_two_robots_and_put_in_cabinet"
)
VARIANT = "queryfix_k8"


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    manifest = load_manifest()
    test = pd.read_parquet(SOURCE_PARQUET)
    rows = sorted(set(rows_of(FAMILY)) & set(manifest))
    print(f"{FAMILY}: {len(rows)} rows")

    ours = load_rows(OUTPUT_DIR / f"skill_memory.{VARIANT}.jsonl")
    theirs = load_rows(OUTPUT_DIR / "gmemory.jsonl")

    stage: Counter = Counter()
    shape: Counter = Counter()
    gt_lengths, our_lengths = [], []
    import random

    for index in rows:
        truth = native(test.iloc[index].to_dict())["reward_model"]["ground_truth"]
        gt = canon(truth["time_steps"])
        gt_lengths.append(len(gt))
        parsed = parse_plan(ours[index]["response"])
        if parsed is None:
            stage["unparseable"] += 1
            continue
        if isinstance(parsed, list):
            for step in parsed:
                if isinstance(step, dict) and isinstance(step.get("actions"), dict):
                    step["actions"] = {
                        k: v for k, v in step["actions"].items() if v is not None
                    }
        plan = canon(parsed)
        our_lengths.append(len(plan))
        transformed = scorer.transform_actions(parsed)
        if not transformed:
            stage["no valid actions"] += 1
        else:
            globals_ = scorer.eval_single.__globals__
            original = globals_["random"]
            try:
                globals_["random"] = random.Random(20240617)
                ok = scorer.eval_single(transformed, truth)
            except Exception:
                ok = False
            finally:
                globals_["random"] = original
            if not ok:
                stage["goal not reached"] += 1
            elif len(truth["time_steps"]) / len(transformed) >= 0.99:
                stage["correct"] += 1
            else:
                stage["goal reached, too many steps"] += 1
        if len(plan) != len(gt):
            shape["wrong length"] += 1
        elif verbs(plan) != verbs(gt):
            shape["right length, wrong verbs"] += 1
        else:
            shape["right length and verbs"] += 1

    print(f"\nwhere our arm stops (n={len(rows)}):")
    for key, count in stage.most_common():
        print(f"    {key:34s} {count:4d}  {100*count/len(rows):5.1f}%")
    print(f"\nplan shape vs ground truth:")
    for key, count in shape.most_common():
        print(f"    {key:34s} {count:4d}  {100*count/len(rows):5.1f}%")
    print(
        f"\nplan length: ground truth median {median(gt_lengths):.0f}, "
        f"ours median {median(our_lengths):.0f}"
    )

    print("\nmemory size on these rows (median chars):")
    print(f"    ours     {median(ours[i]['memory_prompt_chars'] for i in rows):8.0f}")
    print(f"    gmemory  {median(theirs[i]['memory_prompt_chars'] for i in rows):8.0f}")

    import viki_amendment8_memory as memories

    provider = memories.SkillMemory(
        OUTPUT_DIR / "skill_memory_bank", EXPOSED_STEPS, SELF_ROBOT
    )
    sample = native(test.iloc[rows[0]].to_dict())
    truth = sample["reward_model"]["ground_truth"]
    print("\n" + "=" * 78)
    print(f"row {rows[0]} ground truth")
    print("=" * 78)
    print(json.dumps(truth["time_steps"], indent=1)[:900])
    print("=" * 78)
    print("what our memory shows for it")
    print("=" * 78)
    print(provider.prompt(rows[0], sample))
    print("=" * 78)
    print("what we answered")
    print("=" * 78)
    print(ours[rows[0]]["response"][:1200])


if __name__ == "__main__":
    main()
