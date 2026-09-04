#!/usr/bin/env python3
"""Probe 2: ask the model for the goal, not the plan, and let the composer plan.

Probe 1 showed the planning layer is exact once the goal predicates are known
(924/924). So the whole benchmark now rests on one question: can the model read the
picture and the instruction and say what has to become true? This asks it that, in
the judge's own vocabulary, and hands the answer to the composer. The score is the
official one against the real ground truth, so nothing here is an estimate of an
estimate -- it is the method, end to end.

The model is given exactly what the plan-writing arms are given: the benchmark's own
system prompt, with its robot roster and action primitives, and the image. It is not
given the partner prefix those arms received, which leaks a step of the reference
plan; a goal parser has no use for it and is not helped by it.

What it is deliberately not given is the vocabulary. Goal targets in this split draw
from a small closed set and predicates from two keys, but that is a fact about the
benchmark, and reading it off the test set would be reading off the answers. A model
that has to guess whether the counter is called `kitchen work area` will sometimes
guess wrong, and the size of that loss is the first thing memory should be asked to
pay for -- so it is measured here, alone, before any memory is added.

`--assets` is the one concession, and it is reported separately: it lists the asset
names in the scene without their positions. It exists to separate two failures that
otherwise look identical -- naming a thing wrongly, and reasoning about the goal
wrongly.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import threading
from collections import Counter, defaultdict
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd
from openai import OpenAI

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment11_composer import (
    MANIFEST,
    flatten_predicates,
    OUT,
    SEED,
    build_metadata,
    load_operators,
    plan_for,
    load_sim,
    score,
)

# Which backbone this process talks to. One switch drives both the gate key and the
# served name, because the two disagreeing is exactly how a cell gets written with the
# wrong model in its metadata: the run would pass the gate against one model and label
# itself with another. Default is the 72B every archived cell was produced on.
from viki_amendment5 import BACKBONE, SERVED_MODEL_FOR_BACKBONE  # noqa: E402

SERVED_MODEL = SERVED_MODEL_FOR_BACKBONE
BASE_URL = "http://192.168.32.40:8050/v1"
MAX_TOKENS = 900
TEMPERATURE = 0

INSTRUCTION = """\
Ignore the output format described above. Do NOT write a plan and do NOT use \
<think> or <answer> tags.

Instead, state only what must be TRUE when the task is finished, and in what order.

Reply with one JSON object and nothing else:

{
  "goals": [
    {"asset": "<object>", "at": "<where it must end up>"},
    {"asset": "<object>", "activated": true}
  ],
  "order": [
    [{"asset": "<object>", "at": "<place>"}],
    [{"asset": "<object>", "activated": true}]
  ]
}

Rules:
- "goals" lists every end condition the task requires. Use "at" for something that \
must end up somewhere, and "activated": true for something that must be switched on, \
used, or operated (a toaster run, a knife used to cut).
- "order" is ONLY for a condition that physically cannot hold until another one \
already does: something must be inside the toaster before the toaster can be run, \
something must be on the board before it can be cut, a bowl must be on the counter \
before anything can be put into that bowl. The test is whether the later condition \
acts on, or places something into, the very object the earlier condition puts in \
place.
- Do NOT give an order just because the instruction mentions one thing before \
another, and do NOT give an order for two things that merely end up in the same \
place -- putting two objects into the same cabinet is two independent conditions and \
they can happen at once. When in doubt, give [].
- A condition may appear in "order" even if it is not itself a goal.
- Name each object and place exactly as it would be addressed by a robot command.
"""


def build_messages(sample: Dict[str, Any], assets: Optional[List[str]]) -> List[Dict[str, Any]]:
    messages = bench.get_messages(sample)
    user_message = next(message for message in reversed(messages) if message["role"] == "user")
    extra = INSTRUCTION
    if assets is not None:
        extra += f"\nObjects present in this scene: {', '.join(sorted(assets))}.\n"
    content = user_message["content"]
    if isinstance(content, list):
        text_item = next(item for item in content if item["type"] == "text")
        text_item["text"] = f"{text_item['text']}\n\n{extra}"
    else:
        user_message["content"] = f"{content}\n\n{extra}"
    return messages


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for position in range(start, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : position + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


VOCABULARY_PATH = OUT / "vocabulary.json"


def load_vocabulary(enabled: bool = True) -> Optional[Dict[str, Any]]:
    """Layer 3 if it has been built: the names this world uses, learned from training."""
    if not enabled or not VOCABULARY_PATH.is_file():
        return None
    return json.loads(VOCABULARY_PATH.read_text())


def place_vocabulary(truth: Dict[str, Any], vocabulary: Optional[Dict[str, Any]] = None) -> List[str]:
    """The names a predicted place is allowed to snap onto.

    Without Layer 3 this is only what the row itself happens to mention, which is why a
    counter the row never mentions cannot be spelled the world's way however plainly the
    instruction names it -- thirty-eight failures were exactly that. With Layer 3 it is
    the domain's vocabulary, taken from the training split, and the row's own scene stops
    being consulted at all: the pipeline leans on one less thing it was handed.
    """
    if vocabulary is not None:
        return sorted(set(vocabulary.get("places", [])) | set(vocabulary.get("assets", [])))
    names = set()
    for asset, positions in (truth.get("init_pos") or {}).items():
        if positions is None or (asset.startswith("R") and asset[1:].isdigit()):
            continue
        names.add(asset.rsplit("_", 1)[0])
        names.update(position for position in positions if isinstance(position, str))
    return sorted(names)


def clean(name: str) -> str:
    """The identifier spelling a model reaches for, written the way the world spells it."""
    return " ".join(name.replace("_", " ").replace("-", " ").split())


def normalise(name: Any, known: List[str]) -> Optional[str]:
    """Snap a predicted name onto one the environment knows, when it plainly means it.

    A miss here is a naming failure, not a reasoning failure, and the two are worth
    telling apart: only exact, case, and whole-word containment are accepted, so a
    wrong object stays wrong.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    # A model writing an identifier reaches for `kitchen_work_area`; the environment
    # calls it `kitchen work area`. That is a spelling convention, not a different
    # place, and it is fixed blindly rather than by consulting any vocabulary -- the
    # cleaned form is also what gets returned when nothing matches, so a place the
    # scene never mentions can still be named correctly.
    name = name.strip()
    for candidate in (name, name.replace("_", " ").replace("-", " ").strip()):
        if candidate in known:
            return candidate
    lowered = {item.lower(): item for item in known}
    for candidate in (name, name.replace("_", " ").replace("-", " ").strip()):
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    name = name.replace("_", " ").replace("-", " ").strip()
    words = set(re.findall(r"[a-z0-9]+", name.lower()))
    best, best_score = None, 0
    for item in known:
        item_words = set(re.findall(r"[a-z0-9]+", item.lower()))
        if not item_words:
            continue
        overlap = len(words & item_words)
        if overlap and (words <= item_words or item_words <= words) and overlap > best_score:
            best, best_score = item, overlap
    return best


def to_predicates(
    parsed: Dict[str, Any], asset_names: List[str], place_names: List[str]
) -> Tuple[Optional[List[Any]], Optional[List[Any]], int]:
    """The model's answer in the judge's schema, with names snapped to the scene."""
    snapped = 0

    def predicate(item: Any) -> Optional[Dict[str, Any]]:
        nonlocal snapped
        if not isinstance(item, dict):
            return None
        raw_asset = item.get("asset") or item.get("name")
        asset = normalise(raw_asset, asset_names)
        if asset is None:
            return None
        if asset != raw_asset:
            snapped += 1
        status: Dict[str, Any] = {}
        if item.get("activated") is True or item.get("is_activated") is True:
            status["is_activated"] = True
        where = item.get("at") or item.get("pos") or (item.get("status") or {}).get("pos.name")
        if isinstance(where, str) and where.strip():
            # A place need not be an asset -- the judge compares the name it is given
            # against the name the goal asks for, so an unrecognised place is kept
            # rather than dropped, but kept in the world's spelling.
            target = normalise(where, place_names) or clean(where)
            if target != where.strip():
                snapped += 1
            status["pos.name"] = target
        if not status:
            return None
        return {"type": "asset", "name": asset, "is_satisfied": True, "status": status}

    goals = [p for p in (predicate(item) for item in parsed.get("goals") or []) if p]
    if not goals:
        return None, None, snapped
    def build_stages(constraint: Any) -> List[List[Dict[str, Any]]]:
        stages = []
        for stage in constraint if isinstance(constraint, list) else []:
            items = stage if isinstance(stage, list) else [stage]
            built = [p for p in (predicate(item) for item in items) if p]
            if built:
                stages.append(built)
        return stages

    order = parsed.get("order") or []
    temporal: List[Any] = []
    nested = isinstance(order, list) and order and all(
        isinstance(element, list)
        and element
        and all(isinstance(inner, list) for inner in element)
        for element in order
    )
    for constraint in (order if nested else [order]):
        stages = build_stages(constraint)
        if len(stages) >= 2:
            temporal.append(stages)
    return [[goal] for goal in goals], temporal, snapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--assets", action="store_true", help="list the scene's object names")
    parser.add_argument("--tag", default="probe2_zeroshot")
    parser.add_argument("--induced", action="store_true",
                        help="plan with the induced operator library (Layer 1)")
    parser.add_argument("--no-vocabulary", action="store_true",
                        help="snap places onto the row's own scene instead of Layer 3")
    arguments = parser.parse_args()

    scorer, SimEnv, Checker, entities, Eval, viki2 = load_sim()
    indices = [json.loads(line)["index"] for line in MANIFEST.read_text().splitlines() if line.strip()]
    if arguments.limit:
        indices = indices[: arguments.limit]
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    client = OpenAI(api_key="EMPTY", base_url=BASE_URL, max_retries=5, timeout=3600)
    operators = load_operators() if arguments.induced else None
    vocabulary = load_vocabulary(not arguments.no_vocabulary)
    print(f"layer 3 vocabulary: {'loaded' if vocabulary else 'not used'}")
    lock = threading.Lock()
    done = [0]

    def work(index: int) -> Dict[str, Any]:
        sample = bench.to_native(frame.iloc[index].to_dict())
        truth = bench.get_ground_truth(sample)
        metadata = build_metadata(
            {k: v for k, v in truth.items() if k != "time_steps"}, viki2, SEED
        )
        asset_names = sorted(metadata["assets"])
        messages = build_messages(sample, asset_names if arguments.assets else None)
        record: Dict[str, Any] = {
            "index": index,
            "task_name": truth["task_name"],
            "accuracy": 0.0,
            "reason": "",
            "snapped": 0,
            "plan_len": 0,
            "budget": len(truth["time_steps"]),
        }
        try:
            completion = client.chat.completions.create(
                model=SERVED_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            text = completion.choices[0].message.content or ""
        except Exception as error:
            record["reason"] = f"REQUEST_FAILED:{type(error).__name__}"
            return record
        record["raw"] = text[-4000:]
        parsed = extract_json(text)
        if parsed is None:
            record["reason"] = "UNPARSEABLE"
            return record
        goals, temporal, snapped = to_predicates(
            parsed, asset_names, place_vocabulary(truth, vocabulary)
        )
        record["snapped"] = snapped
        if goals is None:
            record["reason"] = "NO_USABLE_GOAL"
            return record
        record["predicted"] = {"goal_constraints": goals, "temporal_constraints": temporal}
        # Ground truth writes its unused status keys as null and nests one level
        # deeper in places, so equality has to be taken on the content rather than
        # the spelling: the set of (asset, is_satisfied, non-null status) triples.
        def canonical(constraints: Any) -> set:
            out = set()
            for predicate_ in flatten_predicates(constraints):
                status = {k: v for k, v in (predicate_.get("status") or {}).items() if v is not None}
                out.add(
                    json.dumps(
                        [predicate_.get("name"), predicate_.get("is_satisfied", True), status],
                        sort_keys=True,
                    )
                )
            return out

        record["goals_exact"] = canonical(goals) == canonical(truth["goal_constraints"])
        record["goals_superset"] = canonical(goals) >= canonical(truth["goal_constraints"])
        blind = {k: v for k, v in truth.items() if k != "time_steps"}
        blind["temporal_constraints"] = temporal
        plan, reason, dropped, relaxed = None, "NO_SCHEDULE", 0, False
        # A model that narrates the route ("the pear goes into the box, the box goes
        # to the sink") states way-stations as requirements, and no robot can carry a
        # box that only a legged robot can push. Rather than return nothing, the
        # composer keeps the most it can actually schedule: everything first, then
        # the same goals without the stated order, then progressively fewer goals. A
        # real requirement is only ever dropped when it was unreachable anyway.
        for drop in range(0, min(3, len(goals))):
            for subset in combinations(range(len(goals)), len(goals) - drop):
                for keep_order in (True, False):
                    blind["goal_constraints"] = [goals[i] for i in subset]
                    blind["temporal_constraints"] = temporal if keep_order else []
                    plan, reason = plan_for(blind, viki2, SimEnv, Checker, entities, SEED, operators=operators)
                    if plan:
                        dropped, relaxed = drop, not keep_order
                        break
                if plan:
                    break
            if plan:
                break
        record["dropped_goals"] = dropped
        record["relaxed_order"] = relaxed
        record["reason"] = reason
        if plan:
            record["plan_len"] = len(plan)
            record["accuracy"] = score(scorer, plan, truth, SEED)[0]
            if record["accuracy"] == 0.0:
                record["reason"] = "OVER_BUDGET" if len(plan) > record["budget"] else "GOAL_UNMET"
        if record["accuracy"] == 1.0:
            record["reason"] = "SOLVED"
        return record

    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        futures = {pool.submit(work, index): index for index in indices}
        for future in as_completed(futures):
            record = future.result()
            with lock:
                records.append(record)
                done[0] += 1
                if done[0] % 100 == 0:
                    solved = sum(r["accuracy"] for r in records)
                    print(f"  {done[0]}/{len(indices)}  solved={solved:.0f} "
                          f"({solved / len(records) * 100:.1f}%)", flush=True)

    records.sort(key=lambda r: r["index"])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{arguments.tag}.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("raw", "predicted")} for r in records])
    solved = int(table["accuracy"].sum())
    print(f"\n=== {arguments.tag}: composer driven by predicted goals, {len(records)} rows ===")
    print(f"accuracy        {solved}/{len(records)} = {solved / len(records) * 100:.2f}%")
    print(f"goals exact     {table.get('goals_exact', pd.Series(dtype=bool)).fillna(False).mean() * 100:.2f}%")
    print(f"goals superset  {table.get('goals_superset', pd.Series(dtype=bool)).fillna(False).mean() * 100:.2f}%")
    print(f"goals dropped   {(table.get('dropped_goals', pd.Series(dtype=float)).fillna(0) > 0).mean() * 100:.2f}% of rows")
    print(f"order relaxed   {table.get('relaxed_order', pd.Series(dtype=bool)).fillna(False).mean() * 100:.2f}% of rows")
    print(f"names snapped   {(table['snapped'] > 0).mean() * 100:.2f}% of rows")
    print("\noutcome:")
    for reason, count in Counter(table["reason"]).most_common():
        print(f"  {reason:<24} {count:>5}  {count / len(records) * 100:5.1f}%")
    print("\nby family:")
    grouped = defaultdict(lambda: [0, 0])
    for record in records:
        grouped[record["task_name"]][0] += int(record["accuracy"] == 1.0)
        grouped[record["task_name"]][1] += 1
    for family, (hit, total) in sorted(grouped.items(), key=lambda kv: -kv[1][1]):
        print(f"  {family:<48} {hit:>4}/{total:<4} {hit / total * 100:5.1f}%")
    print(f"\nwrote {OUT / (arguments.tag + '.jsonl')}")


if __name__ == "__main__":
    main()
