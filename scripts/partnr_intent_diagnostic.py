#!/usr/bin/env python3
"""Where the intent arm loses its four tenths: does the model say the wrong thing, or
say the right thing in a name the world will not answer to?

The head-to-head put `v2_intent` at 0.331 against the same composer fed ground-truth
propositions at 0.740. That gap is one prompt: the model reads the instruction and the
(partially observed) world and writes the goal predicates. This grades that step alone,
offline, against the episode's own propositions -- no simulator, no LLM, no rerun. The
responses are already on disk, appended to the archived prompts.

Failures that look identical in the score are counted apart, because they cost
different things to fix:

  unstatable   the predicate the episode wants is not in the menu the memory offers.
               A perfect model cannot state it. This is the transfer boundary showing
               up as a recall cap, and it is reported as a ceiling, not as an error.
  vocabulary   the model used the instruction's word and the world uses another --
               `phone` for `cellphone_1`. The reasoning was right and no entity
               answers to the name.
  grounding    right object, right predicate, wrong instance of the target -- one of
               the four living-room tables.
  reasoning    a predicate that is not required, or a required one never mentioned.

A predicted name counts as correct when `GraphView.resolve` -- the planner's own
by-instance-of-category rule, replicated here -- maps it onto the entity the
proposition names. Grading against the fully explored graph is deliberate: it asks
whether the name would ever bind, so exploration failures cannot masquerade as naming
failures.

The `lexical` tier is an upper bound and is labelled as one. It forgives a subject
whose name is lexically related to the ground truth's (`plant` for `plant_container`,
`phone` for `cellphone`) and so measures the most a better vocabulary could recover --
not what any particular fix would actually deliver.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "our_method"))

DATASET = ROOT / "data/datasets/partnr_episodes/v0_0"
INTENT = ROOT / "outputs/headtohead/val_mini/v2_intent/results/val_mini.json.gz"
GRAPHS = ROOT / "outputs/sweep/val_mini/ceiling/results/val_mini.json.gz"

INDEXED = re.compile(r"^(.*)_\d+$")


def category(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    match = INDEXED.match(str(name))
    return match.group(1) if match else str(name)


def related(asked: Optional[str], truth: Optional[str]) -> bool:
    """Would a better vocabulary have joined these two names?

    Deliberately generous in one direction only -- containment or a token subset --
    so the tier it feeds reads as a ceiling on what naming help can recover. It does
    not join `toy_vehicle` to `toy_animal`, which share a token but neither contains
    the other.
    """
    if asked is None or truth is None:
        return False
    a = category(str(asked).strip().lower().replace(" ", "_"))
    b = category(str(truth).strip().lower())
    if a == b or a in b or b in a:
        return True
    return set(a.split("_")) <= set(b.split("_")) or set(b.split("_")) <= set(a.split("_"))


RELAXED = False


def resolve(asked: Optional[str], names: List[str], exclude=()) -> Optional[str]:
    """`GraphView.resolve`, verbatim in behaviour, over a bare list of names.

    With `RELAXED`, two rules are appended -- never reordered, so nothing that binds
    today binds differently. They exist to price one fix and nothing else: the model
    answers in the instruction's word because at step 0 it has seen no objects, and
    `phone` is not a prefix of `cellphone_1`, so a correct requirement never binds.
    Suffix first, then containment, shortest candidate wins as the least assuming.
    """
    if asked is None:
        return None
    if asked in names and asked not in exclude:
        return asked
    want = str(asked).strip().lower().replace(" ", "_")
    pool = [name for name in names if name not in exclude]
    rules = [
        [n for n in pool if n.lower() == want],
        [n for n in pool if n.lower().rsplit("_", 1)[0] == want],
        [n for n in pool if n.lower().startswith(want + "_")],
    ]
    if RELAXED:
        rules += [
            [n for n in pool if category(n.lower()).endswith("_" + want)
             or category(n.lower()).endswith(want)],
            [n for n in pool if want in category(n.lower())],
        ]
    for candidates in rules:
        if candidates:
            return sorted(candidates, key=lambda n: (len(n), n))[0]
    return None


# ------------------------------------------------------------------ the world's names

def graph_names(episode_key: str) -> Tuple[Dict[str, str], List[str], List[str]]:
    """handle -> name, every name, and the object names, from the ceiling run's graph.

    The ceiling arm is centralized and fully observed, so its graph holds every entity;
    the union over agents and steps guards against a late discovery.
    """
    path = GRAPHS / "detailed_traces" / f"detailed_trace-{episode_key}.pkl"
    if not path.exists():
        return {}, [], []
    with open(path, "rb") as handle:
        trace = pickle.load(handle)
    handle_to_name: Dict[str, str] = {}
    names: set = set()
    objects: set = set()
    for agent_history in (trace.get("action_history") or {}).values():
        for element in agent_history:
            for graph in (getattr(element, "world_graph", None) or {}).values():
                for entity in graph.graph.keys():
                    name = getattr(entity, "name", None)
                    if not name:
                        continue
                    names.add(name)
                    sim_handle = getattr(entity, "sim_handle", None)
                    if sim_handle:
                        handle_to_name[str(sim_handle)] = name
                    if type(entity).__name__ == "Object":
                        objects.add(name)
    return handle_to_name, sorted(names), sorted(objects)


# ------------------------------------------------------------------ the two goal sets

def ground_truth(episode: Dict[str, Any], handle_to_name: Dict[str, str], fold: bool):
    """The episode's propositions as requirements, through the planner's own functions.

    Folded by default, because the arm being graded folds: a `next to` is not separate
    work, it is a slot on a placement that is already in the list. Grading against the
    unfolded propositions would charge the model for a requirement its own composer
    never plans.
    """
    from skill_memory_v2.partnr_planner import fold_spatial, requirements_from_propositions

    propositions = [
        SimpleNamespace(function_name=p.get("function_name"), args=p.get("args") or {})
        for p in episode.get("evaluation_propositions") or []
    ]
    requirements = requirements_from_propositions(propositions, handle_to_name, {}, {})
    for requirement in requirements:
        requirement.setdefault("after_propositions", [])
    if fold:
        requirements, _ = fold_spatial(requirements)
    return requirements


def predicted(path: Path):
    """The model's answer, parsed the way the planner parses it.

    The archived file is the prompt with the response appended, so the menu of allowed
    predicates and the answer come out of the same file -- the keys accepted here are
    exactly the keys that run accepted.
    """
    text = path.read_text(errors="replace")
    head, marker, answer = text.rpartition("Requirements:\n")
    if not marker:
        return [], set(), 0
    menu = set(re.findall(r"^\s{2}(\w+)\(", head, flags=re.M))
    out = []
    lines = 0
    for line in answer.splitlines():
        line = line.strip().strip("-*. ")
        if not line:
            continue
        lines += 1
        if "(" not in line or not line.endswith(")"):
            continue
        key, _, rest = line.partition("(")
        key = key.strip()
        if key not in menu:
            continue
        arguments = [piece.strip() for piece in rest[:-1].split(",") if piece.strip()]
        if not arguments:
            continue
        out.append({"key": key, "subject": arguments[0],
                    "target": arguments[1] if len(arguments) > 1 else None})
    return out, menu, lines


# ------------------------------------------------------------------ grading

def grade(gt, pred, names, menu):
    """Match predictions to requirements and name the reason for every miss.

    Subjects are resolved with the same `exclude` discipline `_bind` uses, so two
    predictions over one category take two instances rather than one twice, and
    matching is greedy in the order the model wrote them -- the order the planner
    would have bound them in.
    """
    taken: set = set()
    bound = []
    for requirement in pred:
        subject = resolve(requirement["subject"], names, taken)
        if subject is not None:
            taken.add(subject)
        bound.append({
            **requirement,
            "subject_bound": subject,
            "target_bound": resolve(requirement["target"], names) if requirement["target"] else None,
        })

    open_gt = list(range(len(gt)))
    exact: Dict[int, int] = {}
    target_miss: Dict[int, int] = {}
    lexical: Dict[int, int] = {}

    def claim(store, test):
        for position, requirement in enumerate(bound):
            if position in exact or position in target_miss or position in lexical:
                continue
            for index in list(open_gt):
                if test(requirement, gt[index]):
                    store[position] = index
                    open_gt.remove(index)
                    break

    # Tightest first, so a prediction is credited at the best tier it earns.
    claim(exact, lambda p, g: p["key"] == g["key"]
          and p["subject_bound"] == g["subject"] and p["target_bound"] == g["target"])
    claim(target_miss, lambda p, g: p["key"] == g["key"] and p["subject_bound"] == g["subject"])
    claim(lexical, lambda p, g: p["key"] == g["key"] and related(p["subject"], g["subject"]))

    reasons: Counter = Counter()
    for position, requirement in enumerate(bound):
        if position in exact:
            continue
        if position in target_miss:
            want = gt[target_miss[position]]
            same = category(requirement["target_bound"]) == category(want["target"])
            reasons["grounding: wrong instance, right kind of target" if same
                    else "grounding: wrong kind of target"] += 1
        elif position in lexical:
            want = gt[lexical[position]]
            reasons["vocabulary: the world calls it something else" if requirement["subject_bound"] is None
                    else "vocabulary: the name bound to the wrong entity"] += 1
        elif requirement["subject_bound"] is None:
            reasons["reasoning: named a thing that is not in this episode"] += 1
        else:
            reasons["reasoning: predicate not required"] += 1

    unstatable = 0
    for index in open_gt:
        if gt[index]["key"] not in menu:
            unstatable += 1
            reasons["unstatable: predicate not in the memory's menu"] += 1
        else:
            reasons["reasoning: requirement never stated"] += 1

    return {
        "gt": len(gt),
        "statable": sum(1 for r in gt if r["key"] in menu),
        "pred": len(bound),
        "exact": len(exact),
        "target_miss": len(target_miss),
        "lexical": len(lexical),
        "unstatable_missed": unstatable,
        "reasons": reasons,
        "bound": bound,
        "missed": [gt[i] for i in open_gt],
    }


# ------------------------------------------------------------------ report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val_mini")
    parser.add_argument("--no-fold", action="store_true",
                        help="grade against the unfolded propositions instead")
    parser.add_argument("--relaxed", action="store_true",
                        help="price the binding fix: allow suffix and containment matches")
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    global RELAXED
    RELAXED = args.relaxed

    from partnr_task_types import classify

    with gzip.open(DATASET / f"{args.split}.json.gz") as handle:
        episodes = {str(e["episode_id"]): e for e in json.load(handle)["episodes"]}

    prompts = sorted((INTENT / "prompts" / "0").glob("prompt-episode_*-0.txt"))
    print(f"{len(prompts)} archived responses, {len(episodes)} episodes in {args.split}, "
          f"ground truth {'unfolded' if args.no_fold else 'folded'}"
          f"{', RELAXED binding' if args.relaxed else ''}\n")

    totals: Counter = Counter()
    reasons: Counter = Counter()
    by_type: Dict[str, Counter] = defaultdict(Counter)
    rows = []
    skipped: Counter = Counter()

    for path in prompts:
        episode_key = path.name[len("prompt-"):-len("-0.txt")]
        episode = episodes.get(episode_key.split("_")[1])
        if episode is None:
            skipped["no such episode"] += 1
            continue
        handle_to_name, names, objects = graph_names(episode_key)
        if not names:
            skipped["no ceiling graph"] += 1
            continue
        gt = ground_truth(episode, handle_to_name, fold=not args.no_fold)
        if not gt:
            skipped["no resolvable propositions"] += 1
            continue
        pred, menu, _ = predicted(path)
        result = grade(gt, pred, names, menu)
        kind = classify(episode)

        for bucket in (totals, by_type[kind]):
            for field in ("gt", "statable", "pred", "exact", "target_miss", "lexical"):
                bucket[field] += result[field]
            bucket["episodes"] += 1
            bucket["perfect"] += int(result["exact"] == result["gt"] == result["pred"])
        reasons.update(result["reasons"])
        rows.append((episode_key, kind, result, episode["instruction"]))

    gt_n, pred_n = max(totals["gt"], 1), max(totals["pred"], 1)
    statable = max(totals["statable"], 1)
    tiers = [
        ("exact  (key + object + target instance)", totals["exact"]),
        ("+ any target of the right kind", totals["exact"] + totals["target_miss"]),
        ("+ the world's name for the object (upper bound)",
         totals["exact"] + totals["target_miss"] + totals["lexical"]),
    ]

    print("== overall ==")
    print(f"  episodes graded {totals['episodes']}   requirements {totals['gt']}"
          f"   predicates predicted {totals['pred']}")
    print(f"  of those requirements, {totals['statable']} = {totals['statable']/gt_n:.1%} "
          f"use a predicate the menu offers -- a perfect model tops out there\n")
    print(f"  {'tier':48s} {'n':>6s} {'recall':>8s} {'/statable':>10s} {'precision':>10s}")
    for label, count in tiers:
        print(f"  {label:48s} {count:6d} {count/gt_n:8.4f} {count/statable:10.4f} "
              f"{count/pred_n:10.4f}")
    print(f"\n  episodes fully correct with nothing spurious: {totals['perfect']}/"
          f"{totals['episodes']} = {totals['perfect']/max(totals['episodes'],1):.1%}")
    if skipped:
        print(f"  skipped: {dict(skipped)}")

    print("\n== the ledger (every predicted item and every missed requirement) ==")
    for reason, count in reasons.most_common():
        print(f"  {count:6d}  {reason}")

    print("\n== by task type ==")
    print(f"  {'type':8s} {'eps':>4s} {'gt':>6s} {'statable':>9s} {'pred':>6s} "
          f"{'exact':>7s} {'+target':>8s} {'+vocab':>8s}")
    for kind in sorted(by_type):
        b = by_type[kind]
        n = max(b["gt"], 1)
        print(f"  {kind:8s} {b['episodes']:4d} {b['gt']:6d} "
              f"{b['statable']/n:8.1%} {b['pred']:6d} "
              f"{b['exact']/n:7.3f} {(b['exact']+b['target_miss'])/n:8.3f} "
              f"{(b['exact']+b['target_miss']+b['lexical'])/n:8.3f}")

    for index in range(args.examples):
        if index >= len(rows):
            break
        episode_key, kind, result, instruction = rows[index]
        print(f"\n  {episode_key} [{kind}]  {instruction}")
        for requirement in result["bound"]:
            print(f"    said  {requirement['key']}({requirement['subject']}, "
                  f"{requirement['target']})  ->  {requirement['subject_bound']}, "
                  f"{requirement['target_bound']}")
        for want in result["missed"]:
            print(f"    want  {want['key']}({want['subject']}, {want['target']})")

    if args.json:
        args.json.write_text(json.dumps({
            "totals": dict(totals),
            "reasons": dict(reasons),
            "by_type": {k: dict(v) for k, v in by_type.items()},
            "skipped": dict(skipped),
        }, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
