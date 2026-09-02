#!/usr/bin/env python3
"""Why does G-Memory still win on the held-out-family folds?

The fold split removes every training episode of the evaluated row's own family,
so neither arm can retrieve a same-family neighbour. G-Memory nevertheless scores
20.78% against our 12.07%, and the paired test says it is right on 102 rows where
we are wrong. This script asks what those 102 rows look like.

The central question is whether the fold is out-of-distribution at the level that
matters. A family is held out by name; the plan is what gets scored. If the same
action sequence also occurs under a different family name, the answer is still
inside the permitted bank and nearest-trajectory retrieval can copy it. That would
make the fold a rename rather than a distribution shift, and it is checkable
offline: no retrieval and no model calls, just the plan strings.
"""

from __future__ import annotations

import ast
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import (
    MEMORY_PARQUET,
    OUTPUT_DIR,
    SOURCE_PARQUET,
    load_manifest,
    native,
)
from viki_amendment9_folds import folds, rows_of, train_family_by_index

from habitat_llm.evaluation import viki_bench as bench  # noqa: E402

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
SEED = 20240617
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "queryfix_k8"


def parse_plan(response: str) -> Any:
    """The tolerant parser: whichever of the two readers accepts the body."""
    match = ANSWER.search(response)
    if not match:
        return None
    body = match.group(1).strip()
    if re.search(r"\{\{.*\}\}", body, re.DOTALL):
        body = body.replace("{{", "{").replace("}}", "}")
    for parser in (ast.literal_eval, json.loads):
        try:
            return parser(body)
        except Exception:
            continue
    return None


def canon(steps: Any) -> Tuple:
    """Plan identity: per step, the set of (robot, action) with idle robots dropped.

    Idle robots are written `null` by the dataset and omitted by some models; that
    difference is a serialisation detail, not a different plan, so it is removed
    before comparing.
    """
    if not isinstance(steps, list):
        return ()
    out = []
    for step in steps:
        if not isinstance(step, dict):
            return ()
        actions = step.get("actions")
        if not isinstance(actions, dict):
            return ()
        live = tuple(
            sorted(
                (robot, json.dumps(action, sort_keys=True))
                for robot, action in actions.items()
                if action is not None
            )
        )
        out.append(live)
    return tuple(out)


def verbs(plan: Tuple) -> List[List[str]]:
    """Per step, the sorted verbs. Actions are usually [verb, target] but some
    model outputs are bare strings or empty lists, so the head is taken defensively."""
    out = []
    for step in plan:
        names = []
        for _, encoded in step:
            action = json.loads(encoded)
            if isinstance(action, list) and action:
                names.append(str(action[0]))
            else:
                names.append(str(action))
        out.append(sorted(names))
    return out


def tolerant(scorer, response: str, truth: Dict[str, Any]) -> int:
    parsed = parse_plan(response)
    if parsed is None:
        return 0
    if isinstance(parsed, list):
        for step in parsed:
            if isinstance(step, dict) and isinstance(step.get("actions"), dict):
                step["actions"] = {
                    k: v for k, v in step["actions"].items() if v is not None
                }
    transformed = scorer.transform_actions(parsed)
    if not transformed:
        return 0
    globals_ = scorer.eval_single.__globals__
    original = globals_["random"]
    try:
        globals_["random"] = random.Random(SEED)
        ok = scorer.eval_single(transformed, truth)
    except Exception:
        return 0
    finally:
        globals_["random"] = original
    if not ok:
        return 0
    return int(len(truth["time_steps"]) / len(transformed) >= 0.99)


def load_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows[record["index"]] = record
    return rows


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    manifest = load_manifest()
    test = pd.read_parquet(SOURCE_PARQUET)
    truth = {
        i: native(test.iloc[i].to_dict())["reward_model"]["ground_truth"]
        for i in sorted(manifest)
    }

    # Every training plan, indexed by its canonical form, with the family it came
    # from. This is the pool both arms may retrieve from once the row's own family
    # is masked.
    train = pd.read_parquet(MEMORY_PARQUET)
    families = train_family_by_index()
    plan_index: Dict[Tuple, set] = defaultdict(set)
    for i in range(len(train)):
        gt = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"]
        plan_index[canon(gt["time_steps"])].add(families.get(i))
    print(f"training plans: {len(train)} rows, {len(plan_index)} distinct plans")

    root = OUTPUT_DIR / "folds"
    per_row_family: Dict[int, str] = {}
    gm: Dict[int, Dict[str, Any]] = {}
    ours: Dict[int, Dict[str, Any]] = {}
    for family in folds():
        for index in rows_of(family):
            per_row_family[index] = family
        gm.update(load_rows(root / family / "gmemory.jsonl"))
        ours.update(load_rows(root / family / f"skill_memory.{VARIANT}.jsonl"))

    shared = sorted(set(gm) & set(ours))
    gm_hit = {i: tolerant(scorer, gm[i]["response"], truth[i]) for i in shared}
    our_hit = {i: tolerant(scorer, ours[i]["response"], truth[i]) for i in shared}
    disagree = [i for i in shared if gm_hit[i] and not our_hit[i]]
    reverse = [i for i in shared if our_hit[i] and not gm_hit[i]]
    print(
        f"shared rows {len(shared)}   gmemory-only {len(disagree)}   "
        f"ours-only {len(reverse)}"
    )

    # Is the answer still in the bank after the family is held out?
    def availability(indices: List[int]) -> Counter:
        tally: Counter = Counter()
        for index in indices:
            held = per_row_family[index]
            homes = plan_index.get(canon(truth[index]["time_steps"]), set())
            permitted = homes - {held}
            if not homes:
                tally["gt plan absent from train"] += 1
            elif permitted:
                tally["gt plan survives in another family"] += 1
            else:
                tally["gt plan only in the held-out family"] += 1
        return tally

    print()
    print("Is the ground-truth plan still retrievable after the fold masks the family?")
    for label, indices in (
        ("all shared rows", shared),
        ("gmemory-only rows", disagree),
        ("ours-only rows", reverse),
    ):
        tally = availability(indices)
        total = sum(tally.values()) or 1
        print(f"  {label} (n={len(indices)})")
        for key, count in tally.most_common():
            print(f"      {key:42s} {count:4d}  {100*count/total:5.1f}%")

    # What each arm actually emitted on the 102.
    print()
    print("On the rows only G-Memory gets right:")
    copied = Counter()
    for index in disagree:
        held = per_row_family[index]
        answer = canon(parse_plan(gm[index]["response"]))
        homes = plan_index.get(answer, set())
        if not answer:
            copied["unparseable"] += 1
        elif not homes:
            copied["answer matches no training plan (composed)"] += 1
        elif homes - {held}:
            copied["answer copies a plan from a permitted family"] += 1
        else:
            copied["answer matches only held-out-family plans"] += 1
    for key, count in copied.most_common():
        print(f"      {key:46s} {count:4d}  {100*count/len(disagree):5.1f}%")

    print()
    print("What our arm emitted on those same rows:")
    modes = Counter()
    for index in disagree:
        parsed = parse_plan(ours[index]["response"])
        if parsed is None:
            modes["unparseable"] += 1
            continue
        ours_canon = canon(parsed)
        gt_canon = canon(truth[index]["time_steps"])
        if not ours_canon:
            modes["parsed but not a plan"] += 1
        elif len(ours_canon) != len(gt_canon):
            modes[
                "wrong length ("
                + ("too short" if len(ours_canon) < len(gt_canon) else "too long")
                + ")"
            ] += 1
        else:
            same_verbs = verbs(ours_canon) == verbs(gt_canon)
            modes[
                "right length, right verbs, wrong targets"
                if same_verbs
                else "right length, wrong verbs"
            ] += 1
    for key, count in modes.most_common():
        print(f"      {key:46s} {count:4d}  {100*count/len(disagree):5.1f}%")

    # The tolerant score folds two very different failures into one zero: a plan
    # the simulator rejects, and a plan it accepts that simply uses more steps than
    # the ground truth. The second is a length problem, not a reasoning problem, and
    # is worth separating before concluding anything about abstraction.
    def stage(response: str, truth: Dict[str, Any]) -> str:
        parsed = parse_plan(response)
        if parsed is None:
            return "unparseable"
        if isinstance(parsed, list):
            for step in parsed:
                if isinstance(step, dict) and isinstance(step.get("actions"), dict):
                    step["actions"] = {
                        k: v for k, v in step["actions"].items() if v is not None
                    }
        transformed = scorer.transform_actions(parsed)
        if not transformed:
            return "no valid actions"
        globals_ = scorer.eval_single.__globals__
        original = globals_["random"]
        try:
            globals_["random"] = random.Random(SEED)
            ok = scorer.eval_single(transformed, truth)
        except Exception:
            return "scorer raised"
        finally:
            globals_["random"] = original
        if not ok:
            return "goal not reached"
        ratio = len(truth["time_steps"]) / len(transformed)
        return "correct" if ratio >= 0.99 else "goal reached, too many steps"

    print()
    print("Where each arm stops, over all 924 fold rows:")
    stages = {"gmemory": gm, "ours": ours}
    for name, rows in stages.items():
        tally = Counter(stage(rows[i]["response"], truth[i]) for i in shared)
        print(f"  {name}")
        for key, count in tally.most_common():
            print(f"      {key:34s} {count:4d}  {100*count/len(shared):5.1f}%")

    print()
    print("Fold breakdown of the disagreement:")
    by_family = Counter(per_row_family[i] for i in disagree)
    rev_family = Counter(per_row_family[i] for i in reverse)
    for family in folds():
        n = len([i for i in shared if per_row_family[i] == family])
        if n:
            print(
                f"  {family:46s} n={n:4d}  gmemory-only {by_family[family]:3d}  "
                f"ours-only {rev_family[family]:3d}"
            )

    out = OUTPUT_DIR / "folds" / f"diag102.{VARIANT}.json"
    out.write_text(
        json.dumps(
            {
                "variant": VARIANT,
                "shared": len(shared),
                "gmemory_only": disagree,
                "ours_only": reverse,
            },
            indent=2,
        )
    )
    print(f"\nrow ids written to {out}")


if __name__ == "__main__":
    main()
