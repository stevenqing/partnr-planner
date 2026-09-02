#!/usr/bin/env python3
"""Generate the cross-task recombination split, in an imaged and a text-only form.

A cutting scene holds three or four live assets and its own plan uses all of them, so
a second goal needs an asset the scene does not currently place. The layout supplies
one: 204 of 440 (layout, asset) pairs keep the same position set in every row that
places them, and every cutting scene has at least four such assets that its own task
never touches. Adding one back is restoring a placement the dataset itself records
for that layout rather than inventing a scene.

What that cannot settle is whether the asset appears in the row's rendered image, and
the images are too small to read. So the split is emitted twice from the same
instances: once keeping the image, once replacing it with the scene inventory in
text. If the two agree, the perceptual worry did not matter; if the imaged form is
worse, that is the missing object showing up as evidence rather than as a caveat.

Every instance is verified by the official checker: the donor's own plan first, then
the extension, then the compressed reference, which is compressed only as far as the
checker still accepts because that length is the bound predictions are held to.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment8b import MEMORY_PARQUET, OUTPUT_DIR, SOURCE_PARQUET, load_manifest, native
from viki_amendment9_folds import train_family_by_index
from viki_amendment10_pairs import live
from viki_amendment10_recombine import CUTTING, delivery_goals
from viki_plan_format import evaluate, steps_of

OUT = OUTPUT_DIR.parent / "amendment10"
IMAGE_SENTENCES = (
    "I will provide you with an image of robots in a scene, available robots and "
    "their action primitives, and a task description."
)
TEXT_SENTENCES = (
    "I will provide you with a list of the objects in the scene and where they are, "
    "available robots and their action primitives, and a task description."
)
LOOK_SENTENCE = "You must first analyze the image to fully understand the scene depicted."
READ_SENTENCE = "You must first read the object list to fully understand the scene."
VISUAL_SENTENCE = (
    "Your reasoning must strictly adhere to the visual content of the image and the "
    "task description"
)
LISTED_SENTENCE = (
    "Your reasoning must strictly adhere to the listed scene contents and the "
    "task description"
)


def stable_placements(train, test) -> Dict[Any, Dict[str, List[str]]]:
    seen: Dict[Any, Dict[str, Set[FrozenSet[str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for frame in (train, test):
        for i in range(len(frame)):
            truth = native(frame.iloc[i].to_dict())["reward_model"]["ground_truth"]
            layout = truth.get("layout_id")
            for name, positions in (truth.get("init_pos") or {}).items():
                if positions is None or (
                    name.startswith("R") and name[1:].isdigit()
                ):
                    continue
                values = positions if isinstance(positions, list) else [positions]
                seen[layout][name].add(frozenset(str(v) for v in values))
    return {
        layout: {
            name: sorted(next(iter(sets)))
            for name, sets in assets.items()
            if len(sets) == 1
        }
        for layout, assets in seen.items()
    }


def inventory_text(truth: Dict[str, Any]) -> str:
    lines = []
    for name, positions in sorted((truth.get("init_pos") or {}).items()):
        if positions is None or (name.startswith("R") and name[1:].isdigit()):
            continue
        values = positions if isinstance(positions, list) else [positions]
        lines.append(f"{name.rsplit('_', 1)[0]} at {' or '.join(sorted(values))}")
    robots = ", ".join(
        f"{robot} is a {kind}"
        for robot, kind in sorted((truth.get("robots") or {}).items())
        if kind
    )
    return f"Scene contents: {'; '.join(lines)}. Robots: {robots}."


def delivery_branch(item: str, target: str) -> List[List[str]]:
    return [
        ["Move", item],
        ["Reach", item],
        ["Grasp", item],
        ["Move", target],
        ["Place", target],
    ]


def append_branch(steps, robot: str, actions) -> List[Dict[str, Any]]:
    robots = sorted({r for step in steps for r in step["actions"]})
    out = json.loads(json.dumps(steps))
    for action in actions:
        out.append({"actions": {r: (action if r == robot else None) for r in robots}})
    return steps_of(out)


def compress(scorer, steps, truth, floor: int):
    best = steps
    improved = True
    while improved:
        improved = False
        for position in range(len(best) - 1, floor, -1):
            merged = json.loads(json.dumps(best))
            tail = merged.pop(position)
            target = merged[position - 1]
            if any(
                target["actions"].get(r) is not None and a is not None
                for r, a in tail["actions"].items()
            ):
                continue
            for robot, action in tail["actions"].items():
                if action is not None:
                    target["actions"][robot] = action
            ok, _ = evaluate(scorer, steps_of(merged), truth)
            if ok:
                best = merged
                improved = True
                break
    return best


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    train = pd.read_parquet(MEMORY_PARQUET)
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()
    catalogue = delivery_goals(train, train_family_by_index())
    placements = stable_placements(train, test)

    imaged: List[Dict[str, Any]] = []
    textual: List[Dict[str, Any]] = []
    reasons: Counter = Counter()

    for index in sorted(manifest):
        raw = test.iloc[index].to_dict()
        row = native(raw)
        truth = row["reward_model"]["ground_truth"]
        if truth.get("task_name") not in CUTTING:
            continue
        layout = truth.get("layout_id")
        assets = live(truth)
        used = {
            str(a[1])
            for step in truth["time_steps"]
            for a in step["actions"].values()
            if a is not None and len(a) > 1
        }
        options = sorted(
            (name, positions)
            for name, positions in placements.get(layout, {}).items()
            if (kind := name.rsplit("_", 1)[0]) in catalogue
            and kind not in assets
            and kind not in used
        )
        if not options:
            reasons["no stable spare asset"] += 1
            continue
        base_ok, base_code = evaluate(scorer, truth["time_steps"], truth)
        if not base_ok:
            reasons[f"donor plan refused ({base_code})"] += 1
            continue

        name, positions = options[index % len(options)]
        kind = name.rsplit("_", 1)[0]
        target = catalogue[kind].most_common(1)[0][0]
        if target in used:
            reasons["delivery target already used"] += 1
            continue

        built = json.loads(json.dumps(truth))
        built["init_pos"][name] = list(positions)
        built["goal_constraints"] = list(truth["goal_constraints"]) + [
            [
                {
                    "is_satisfied": True,
                    "name": kind,
                    "status": {"is_activated": None, "pos.name": target},
                    "type": "asset",
                }
            ]
        ]
        built["task_name"] = "recombine_cut_and_deliver"
        built["description"] = (
            truth["description"].rstrip(". ")
            + f", and move the {kind} to the {target}."
        )
        built["source_row"] = index
        built["added_asset"] = {"name": name, "positions": list(positions),
                                "target": target}

        actors = list(
            dict.fromkeys(
                r
                for step in truth["time_steps"]
                for r, a in step["actions"].items()
                if a is not None
            )
        )
        reference = None
        for robot in actors:
            extended = append_branch(
                truth["time_steps"], robot, delivery_branch(kind, target)
            )
            ok, code = evaluate(scorer, extended, built)
            if ok:
                reference = compress(
                    scorer, extended, built, len(truth["time_steps"]) - 1
                )
                break
            reasons[f"delivery refused on {robot} ({code})"] += 1
        if reference is None:
            continue
        final_ok, _ = evaluate(scorer, reference, built)
        if not final_ok:
            reasons["compressed reference refused"] += 1
            continue
        built["time_steps"] = reference

        prompt = json.loads(json.dumps(row["prompt"]))
        user = next(m for m in prompt if m.get("role") == "user")
        if "<image>" not in user["content"]:
            reasons["prompt has no image marker"] += 1
            continue
        imaged_row = copy.deepcopy(raw)
        imaged_row["reward_model"] = {
            **native(raw["reward_model"]),
            "ground_truth": built,
        }
        imaged_prompt = json.loads(json.dumps(prompt))
        next(m for m in imaged_prompt if m.get("role") == "user")["content"] = (
            "<image>" + built["description"]
        )
        imaged_row["prompt"] = imaged_prompt
        imaged.append(imaged_row)

        text_row = copy.deepcopy(raw)
        text_row["reward_model"] = {
            **native(raw["reward_model"]),
            "ground_truth": built,
        }
        text_row.pop("images", None)
        text_prompt = json.loads(json.dumps(prompt))
        system = next(m for m in text_prompt if m.get("role") == "system")
        content = system["content"]
        for old, new in (
            (IMAGE_SENTENCES, TEXT_SENTENCES),
            (LOOK_SENTENCE, READ_SENTENCE),
            (VISUAL_SENTENCE, LISTED_SENTENCE),
        ):
            if old not in content:
                reasons[f"system prompt lacks: {old[:32]}"] += 1
            content = content.replace(old, new)
        system["content"] = content
        next(m for m in text_prompt if m.get("role") == "user")["content"] = (
            inventory_text(built) + "\n" + built["description"]
        )
        text_row["prompt"] = text_prompt
        textual.append(text_row)

    OUT.mkdir(parents=True, exist_ok=True)
    for label, rows in (("imaged", imaged), ("text", textual)):
        if not rows:
            print(f"{label:8s}    0 rows -- nothing written")
            continue
        path = OUT / f"recombination.{label}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{label:8s} {len(rows):4d} rows -> {path}")
        print(f"         sha256 {digest}")

    print()
    print(f"donors examined: {sum(1 for i in sorted(manifest))}")
    print("skips:")
    for reason, count in reasons.most_common():
        print(f"  {count:5d}  {reason}")

    if imaged:
        sample = imaged[0]["reward_model"]["ground_truth"]
        print()
        print(f"example (source row {sample['source_row']}):")
        print(f"  added: {json.dumps(sample['added_asset'])}")
        print(f"  description: {sample['description']}")
        print(f"  goals: {json.dumps(sample['goal_constraints'])}")
        print(f"  reference plan, {len(sample['time_steps'])} steps:")
        for step in sample["time_steps"]:
            cells = ", ".join(
                f"{r} {json.dumps(a) if a else 'idle'}"
                for r, a in sorted(step["actions"].items())
            )
            print(f"    step {step['step']:2d}: {cells}")


if __name__ == "__main__":
    main()
