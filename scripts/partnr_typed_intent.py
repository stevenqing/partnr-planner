#!/usr/bin/env python3
"""The requirement interface with the memory proposing and the model only choosing.

At 7B the one-shot "write the end state" prompt recovers 0.061 of the requirements, and
asking again with the whole world in view does not move it (0.056) -- the model writes
`is_inside(laptop, workout_1)`, a furniture name as the subject, one table of four. Those
are type errors, and the memory already knows the types: a placement's target is a piece
of furniture, a container is something that opens, `is_in_room` ends in a room. This
measures, offline and on train, how much of the gap closes when the memory enumerates what
is type-valid and the model is left to pick.

Arms, all graded by the same function on the same episodes:

  free         the planner's own prompt, byte for byte (`_requirements_from_llm`).
  free+proj    the same answers passed through the type projection below. No extra call:
               it prices what typing buys when the model is not told about it.
  typed        the candidate prompt: kinds of object shortlisted from the memory's
               vocabulary, relations offered with the targets their type allows.
  typed+proj   its answers through the same projection.

The projection is the memory's type check. A subject must be a kind of object the memory
has seen; a target that is a room makes the requirement `is_in_room`, a piece of furniture
makes it a placement (`is_inside` only if the furniture opens); `is_in_room` is dropped when
a placement of the same object already lands in that room. Anything it cannot type is
dropped, since every target is known at step 0 and an unknown one never binds.

Grading. `exact` is `partnr_intent_diagnostic.grade` unchanged, so the column is
comparable with the 0.061 on record. `typed` is the tier this work is judged on and fixes
two things the old tier gets wrong once `is_in_room` is on the menu: the ground truth's
room is a region name (`living room`) that no prediction can equal, and a proposition over
several acceptable receptacles or objects credits only the first. Here a prediction
matches when its key agrees, its subject is an instance of a kind the proposition accepts,
and its target is one of the proposition's alternatives (rooms by category).

Graphs come from a fully observed no-LLM run on the same split (the ceiling config), cut
to the first recorded step so every object is where it started; the prompt's world block
is that graph with the objects removed, which is what the agent sees at step 0.

  python scripts/partnr_typed_intent.py --split train_mini \\
      --graphs outputs/cand_iface_0914/train_mini/ceiling/results/train_mini.json.gz/detailed_traces \\
      --arms free typed --model qwen2.5-vl-7b --base-url http://127.0.0.1:8063/v1 \\
      --out outputs/cand_iface_0914/train_mini/r1
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "our_method"))

import partnr_intent_diagnostic as diag  # noqa: E402

DATASET = ROOT / "data/datasets/partnr_episodes/v0_0"
RELATIONS = {"on": "is_on_top", "inside": "is_inside", "in_room": "is_in_room"}
PLACEMENTS = ("is_on_top", "is_inside")


def category(name: Optional[str]) -> str:
    return diag.category(str(name).strip().lower().replace(" ", "_")) if name else ""


# ------------------------------------------------------------------ the scene at step 0

def first_graph(trace_path: Path):
    """Agent 0's graph at the first recorded step of a fully observed run."""
    with open(trace_path, "rb") as handle:
        trace = pickle.load(handle)
    history = trace.get("action_history") or {}
    events = sorted(
        (element.timestamp, agent, index, element)
        for agent, elements in history.items()
        for index, element in enumerate(elements)
    )
    for _, _, _, element in events:
        graphs = getattr(element, "world_graph", None) or {}
        if graphs:
            return graphs.get(0) or graphs[sorted(graphs)[0]]
    return None


class Scene:
    def __init__(self, graph):
        from habitat_llm.llm.instruct.utils import get_world_descr
        from habitat_llm.world_model import Floor, Object

        self.handle_to_name = {
            str(n.sim_handle): n.name for n in graph.graph if getattr(n, "sim_handle", None)
        }
        self.objects = sorted(n.name for n in graph.get_all_objects())
        self.rooms = sorted(n.name for n in graph.get_all_rooms())
        furniture = graph.get_all_furnitures()
        self.furniture = sorted(n.name for n in furniture)
        self.openable = sorted(
            n.name for n in furniture
            if n.properties.get("is_articulated") and not isinstance(n, Floor)
        )
        self.room_of_furniture = {
            furn.name: room for room, items in graph.group_furniture_by_room().items()
            for furn in items
        }
        self.names = sorted(set(self.objects) | set(self.rooms) | set(self.furniture))
        blind = copy.deepcopy(graph)
        blind.remove_all_nodes_of_type(Object)
        self.world = get_world_descr(blind, agent_uid=0, include_room_name=True, add_state_info=True)

    def resolve(self, asked: Optional[str], pool: List[str], exclude=()) -> Optional[str]:
        return diag.resolve(asked, pool, exclude) if asked else None


# ------------------------------------------------------------------ ground truth

def ground_truth(episode: Dict[str, Any], scene: Scene) -> List[Dict[str, Any]]:
    """Folded requirements with every subject and target the proposition would accept."""
    requirements = diag.ground_truth(episode, scene.handle_to_name, fold=True)
    propositions = episode.get("evaluation_propositions") or []
    for requirement in requirements:
        args = (propositions[requirement["proposition"]].get("args") or {}) \
            if requirement.get("proposition") is not None else {}
        handles = args.get("object_handles") or args.get("entity_handles_a") or []
        if isinstance(handles, str):
            handles = [handles]
        pool = {scene.handle_to_name.get(str(h)) for h in handles} - {None}
        requirement["subject_kinds"] = {category(n) for n in pool} or {category(requirement["subject"])}
        # A kind match is only as good as the kind is unambiguous: if the scene holds an
        # instance of this kind the proposition does not accept, naming the kind may bind
        # the wrong one. Counted so the lenient tier can be read against a strict one.
        requirement["ambiguous"] = any(
            category(o) in requirement["subject_kinds"] and o not in pool for o in scene.objects
        )
        alternatives = set()
        for target in requirement.get("alternatives") or ([requirement["target"]] if requirement["target"] else []):
            if target in scene.names:
                alternatives.add(target)
            else:  # a region name: any room of that kind satisfies the proposition
                alternatives |= {r for r in scene.rooms if category(r) == category(target)}
        requirement["target_ok"] = alternatives
    return requirements


def vocabulary(graph_dir: Path, limit: Optional[int] = None) -> List[str]:
    kinds: Counter = Counter()
    for index, path in enumerate(sorted(graph_dir.glob("detailed_trace-*.pkl"))):
        if limit and index >= limit:
            break
        graph = first_graph(path)
        if graph is not None:
            kinds.update(category(n.name) for n in graph.get_all_objects())
    return sorted(kinds)


def singular(word: str) -> str:
    for suffix, repl in (("ies", "y"), ("ches", "ch"), ("shes", "sh"), ("sses", "ss"), ("es", "e"), ("s", "")):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)] + repl
    return word


def shortlist(instruction: str, kinds: List[str]) -> List[str]:
    """Kinds of object the instruction could be talking about, generously."""
    words = {singular(w) for w in re.findall(r"[a-z]+", instruction.lower())}
    joined = instruction.lower().replace(" ", "")
    out = []
    for kind in kinds:
        tokens = [singular(t) for t in kind.split("_")]
        if (any(t in words for t in tokens) or kind.replace("_", "") in joined
                or any(len(w) >= 4 and (w in t or t in w) for w in words for t in tokens if len(t) >= 4)):
            out.append(kind)
    return out or list(kinds)


# ------------------------------------------------------------------ prompts and parsing

def free_prompt(scene: Scene, instruction: str, menu: List[Dict[str, Any]]) -> str:
    lines = "\n".join(
        f"  {item['effect']}({'object, target' if item['arity'] == 2 else 'object'})" for item in menu
    )
    return (
        f"{scene.world}\n\n"
        f"Task: {instruction}\n\n"
        "State what must be true when the task is done, one predicate per line, using "
        "only these predicates and only names from the description above:\n"
        f"{lines}\n\n"
        "Write nothing else: no actions, no order, no agent assignment.\n"
        "Requirements:\n"
    )


# Format examples for `typed2`. Written by hand over a house that does not exist, so no
# episode of any split leaks into the prompt; they teach the shape of an answer, not a scene.
EXAMPLES = """Examples (from other houses):
Task: Move the mug and the book from the kitchen counter to the bedroom table.
mug | on | table_7
book | on | table_7
Task: Put the towel in the bathroom cabinet, then take the soap to the laundry room.
towel | inside | cabinet_3
soap | in_room | laundryroom_1
Task: First bring the vase to the living room couch. Then place it on the dining table.
vase | on | couch_12
vase | on | table_20
"""


def typed_prompt(scene: Scene, instruction: str, kinds: List[str], effects: List[str],
                 examples: bool = False) -> str:
    if examples == "typed4":
        return typed4_prompt(scene, instruction, kinds, effects)
    if examples:
        return typed2_prompt(scene, instruction, kinds, effects, rules3=(examples == "typed3"))
    offered = []
    if "is_on_top" in effects:
        offered.append("  on | <a furniture name from the Furniture list above>   (the object rests on it)")
    if "is_inside" in effects and scene.openable:
        offered.append(f"  inside | one of: {', '.join(scene.openable)}   (the object is put inside it)")
    if "is_in_room" in effects:
        offered.append(f"  in_room | one of: {', '.join(scene.rooms)}   "
                       "(only when the task names a room but no furniture in it)")
    return (
        f"{scene.world}\n\n"
        f"Task: {instruction}\n\n"
        "List what must be true when the task is done: one line per object and place, "
        "in the form\n  object | relation | place\n"
        "using only the words offered below.\n\n"
        f"object: one of these kinds -- {', '.join(kinds)}\n"
        "relation | place:\n" + "\n".join(offered) + "\n\n"
        "If the task moves the same object more than once, write one line for every place, "
        "in the order the task gives them. Write nothing else.\n"
        "Requirements:\n"
    )


EXAMPLES4 = """Examples (from other houses):
Task: Move the mug and the book from the kitchen counter to the bedroom table.
Objects: mug, book
mug | on | table_7
book | on | table_7
Task: Put the towel in the bathroom cabinet, then take the soap to the laundry room.
Objects: towel, soap
towel | inside | cabinet_3
soap | in_room | laundryroom_1
Task: First bring the vase and the plate to the living room couch. Then place them on the dining table.
Objects: vase, plate
vase | on | couch_12
plate | on | couch_12
vase | on | table_20
plate | on | table_20
"""


def typed4_prompt(scene: Scene, instruction: str, kinds: List[str], effects: List[str]) -> str:
    """`typed3` with the objects named first, so none is dropped.

    181 of typed3's 506 misses on train_mini were an object offered in the list and never
    written -- mostly the second object of "them" and the objects of a later stage. Naming
    them on one line first gives every one of them a line to be written for.
    """
    text = typed2_prompt(scene, instruction, kinds, effects, rules3=True)
    head, _, _ = text.partition("Examples (from other houses):")
    head = head.replace(
        "Nothing else.\n",
        "Nothing else.\nFirst write one line 'Objects: ...' naming every object the task mentions "
        "(including each one meant by 'them' or 'all'), then at least one line for each of them.\n",
        1,
    )
    return f"{head}{EXAMPLES4}\nTask: {instruction}\nRequirements:\n"


def typed2_prompt(scene: Scene, instruction: str, kinds: List[str], effects: List[str],
                  rules3: bool = False) -> str:
    """`typed`, told that the lists are what may be chosen from, not what to write.

    `rules3` adds the two rules the typed2 residuals asked for: the objects are not in the
    description yet (the model refused an episode for it), and where an object starts is
    not a requirement (it wrote the source cabinet as a destination).
    """
    offered = []
    if "is_on_top" in effects:
        offered.append("  on       -- place: one furniture name from the Furniture list above")
    if "is_inside" in effects and scene.openable:
        offered.append(f"  inside   -- place: one of {', '.join(scene.openable)}")
    if "is_in_room" in effects:
        offered.append(f"  in_room  -- place: one of {', '.join(scene.rooms)}; "
                       "use it only when the task names a room and no furniture in it")
    return (
        f"{scene.world}\n\n"
        "Say where each object must be when the task is done, one line each, as\n"
        "  object | relation | place\n\n"
        f"object: a kind the task mentions, from -- {', '.join(kinds)}\n"
        "relation:\n" + "\n".join(offered) + "\n\n"
        "Rules: choose the single piece of furniture the task means -- in the room the task "
        "names -- and never list alternatives. Write a second line for an object only if the "
        "task moves it again, in the task's order. Nothing else.\n"
        + ("Objects are not in the description yet: name them by kind from the list above. "
           "Write only where objects must end up (and any stop the task asks for on the way), "
           "never where they start.\n" if rules3 else "")
        + "\n"
        f"{EXAMPLES}\n"
        f"Task: {instruction}\n"
        "Requirements:\n"
    )


def parse_free(text: str, effects: List[str]) -> List[Dict[str, Any]]:
    out = []
    for line in str(text).splitlines():
        line = line.strip().strip("-*. ")
        if "(" not in line or not line.endswith(")"):
            continue
        key, _, rest = line.partition("(")
        key = key.strip()
        arguments = [p.strip() for p in rest[:-1].split(",") if p.strip()]
        if key in effects and arguments:
            out.append({"key": key, "subject": arguments[0],
                        "target": arguments[1] if len(arguments) > 1 else None})
    return out


def parse_typed(text: str, effects: List[str]) -> List[Dict[str, Any]]:
    out = []
    for line in str(text).splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line.strip())
        pieces = [p.strip() for p in line.split("|")]
        if len(pieces) != 3 or not all(pieces):
            continue
        key = RELATIONS.get(pieces[1].lower().replace(" ", "_"))
        if key in effects:
            out.append({"key": key, "subject": pieces[0], "target": pieces[2]})
    return out


INSIDE_PRIOR: Optional[Dict[str, Dict[str, int]]] = None


def inside_allowed(furniture: str) -> bool:
    """Whether the benchmark calls a placement on this kind of furniture `is_inside`.

    Read from train, not asked of the model: on 468 train episodes PARTNR labels 677
    placements `is_on_top` and 3 `is_inside` -- "in the cabinet" is `is_on_top` there. With
    no prior loaded the model's word stands, which is what the arms before this did.
    """
    if INSIDE_PRIOR is None:
        return True
    counts = INSIDE_PRIOR.get(category(furniture), {})
    return counts.get("is_inside", 0) > counts.get("is_on_top", 0)


KEEP_DUPLICATES = False
SNAP_FURNITURE = False
ROOM_SNAP = False
COLLAPSE_PLACES = False
SNAP_FIRST = False
NUMBERS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "both": 2, "pair": 2}


def copies_allowed(instruction: str, kind: str, written: int) -> int:
    """How many identical lines for this kind the instruction licenses.

    "2 spoons" is two requirements over two spoons, and the planner binds them to different
    instances; a bare "spoon" is one. A plural without a number keeps what the model wrote,
    up to four, since "all the books" has no count in the text.
    """
    raw = re.findall(r"[a-z0-9]+", instruction.lower())
    words = [singular(w) for w in raw]
    head = singular(kind.split("_")[-1])
    for index, word in enumerate(words):
        if word != head:
            continue
        for back in (1, 2, 3):
            if index - back >= 0:
                previous = raw[index - back]
                if previous.isdigit():
                    return int(previous)
                if previous in NUMBERS:
                    return NUMBERS[previous]
        if raw[index] != word:
            return min(written, 4)
    return 1


def snap(target: Optional[str], scene: Scene, instruction: str) -> Optional[str]:
    """A furniture name the house does not have, mapped to that kind where the task points.

    `table_30` in a house whose tables are 12, 13 and 52 is the model naming the kind and
    inventing the index. The instance is taken from the room the instruction names when
    that room has exactly one of the kind, or from the house when it does; otherwise none,
    because picking among several is the instance choice this does not claim to make.
    """
    kind = category(target)
    same = [f for f in scene.furniture if category(f) == kind]
    if not same:
        return None
    text = instruction.lower()
    named = {r for r in scene.rooms if category(r).replace("_", " ") in text}
    in_named = [f for f in same if scene.room_of_furniture.get(f) in named]
    if len(in_named) == 1:
        return in_named[0]
    # The typed arm's sim cell lost 4 whole episodes to this refusal (`table_30` with four
    # living-room tables): nothing kept, no step taken. Any table in the named room is often
    # one the proposition accepts, so with SNAP_FIRST the first is taken.
    if SNAP_FIRST and in_named:
        return sorted(in_named)[0]
    return same[0] if len(same) == 1 else None


def project(pred: List[Dict[str, Any]], scene: Scene, kinds: List[str], effects: List[str],
            instruction: str = ""):
    """The memory's type check: keep what can be typed, retype what is mistyped."""
    counts: Counter = Counter()
    kept = []
    for item in pred:
        subject = category(item["subject"])
        if subject in {category(n) for n in scene.furniture} | {category(n) for n in scene.rooms}:
            counts["dropped: subject is furniture or a room"] += 1
            continue
        if subject not in kinds:
            near = sorted((k for k in kinds if diag.related(subject, k)), key=len)
            if not near:
                counts["dropped: no kind of object answers to the subject"] += 1
                continue
            subject = near[0]
            counts["retyped: subject to a known kind"] += 1
        target = item.get("target")
        room = scene.resolve(target, scene.rooms)
        furn = None if room else scene.resolve(target, scene.furniture)
        if not room and not furn and SNAP_FURNITURE:
            furn = snap(target, scene, instruction)
            if furn:
                counts["snapped: made-up furniture index"] += 1
        if room:
            key = "is_in_room"
            target = room
        elif furn:
            key = "is_inside" if (item["key"] == "is_inside" and furn in scene.openable
                                  and inside_allowed(furn)) else "is_on_top"
            target = furn
        else:
            counts["dropped: target is neither a room nor furniture"] += 1
            continue
        if key not in effects:
            counts["dropped: relation not in the memory"] += 1
            continue
        if key != item["key"]:
            counts[f"retyped: {item['key']} -> {key}"] += 1
        kept.append({"key": key, "subject": subject, "target": target})
    unique = []
    for item in kept:
        if item not in unique:
            unique.append(item)
        elif KEEP_DUPLICATES:
            written = sum(1 for other in kept if other == item)
            if sum(1 for other in unique if other == item) < copies_allowed(instruction, item["subject"], written):
                unique.append(item)
                counts["kept: repeated line the instruction counts"] += 1
    # An object sent to more than two places is a list of options, not an answer: no
    # instruction here moves one thing through three stages, and the composer would try
    # to satisfy every line.
    if COLLAPSE_PLACES:
        # Dropping every line of an over-listed object emptied 7 of the typed sim cell's 120
        # episodes. Keep what the listing still says: the room it points at, and the order.
        named = {r for r in scene.rooms if category(r).replace("_", " ") in instruction.lower()}

        def room_of(target):
            return target if target in scene.rooms else scene.room_of_furniture.get(target)

        chosen: Dict[Any, str] = {}
        collapsed = []
        for item in unique:
            # Same room *and* same kind of place: "the table, then the couch" in one room is two
            # stages (23 reference requirements lost when this keyed on the room alone), while
            # four living-room tables for one object is a list of options.
            slot = (item["subject"], room_of(item["target"]), item["key"] in PLACEMENTS,
                    category(item["target"]))
            if slot in chosen and chosen[slot] != item["target"]:
                counts["collapsed: another place of the same kind in the same room"] += 1
                continue
            chosen.setdefault(slot, item["target"])
            collapsed.append(item)
        order = defaultdict(list)
        for item in collapsed:
            if item["target"] not in order[item["subject"]]:
                order[item["subject"]].append(item["target"])
        keep = {}
        for subject, targets in order.items():
            if len(targets) <= 2:
                keep[subject] = set(targets)
            else:
                in_named = [t for t in targets if room_of(t) in named][:3]
                keep[subject] = set(in_named or targets[:2])
        trimmed = [item for item in collapsed if item["target"] in keep[item["subject"]]]
        if len(trimmed) < len(collapsed):
            counts["trimmed: places beyond the named rooms or the first two"] += len(collapsed) - len(trimmed)
        unique = trimmed
    else:
        places = defaultdict(set)
        for item in unique:
            places[item["subject"]].add(item["target"])
        listed = {subject for subject, targets in places.items() if len(targets) > 2}
        if listed:
            counts["dropped: object listed over more than two places"] += sum(
                1 for item in unique if item["subject"] in listed)
            unique = [item for item in unique if item["subject"] not in listed]
    # The model names the right kind of furniture in the wrong room far more often than it
    # names the wrong kind (176 against 21 on train_mini). The room is in the instruction's
    # words, so the memory moves the placement there when exactly one named room has that
    # kind; with two candidate rooms (a source and a destination) it leaves the answer alone.
    if ROOM_SNAP and instruction:
        text = instruction.lower()
        named = {r for r in scene.rooms if category(r).replace("_", " ") in text}
        for item in unique:
            if item["key"] not in PLACEMENTS or not named:
                continue
            if scene.room_of_furniture.get(item["target"]) in named:
                continue
            kind = category(item["target"])
            options = sorted(f for f in scene.furniture
                             if category(f) == kind and scene.room_of_furniture.get(f) in named)
            if options and len({scene.room_of_furniture.get(f) for f in options}) == 1:
                item["target"] = options[0]
                counts["moved: to that kind in the room the task names"] += 1
    placed_rooms = defaultdict(set)
    for item in unique:
        if item["key"] in PLACEMENTS:
            placed_rooms[item["subject"]].add(scene.room_of_furniture.get(item["target"]))
    out = []
    for item in unique:
        if item["key"] == "is_in_room" and item["target"] in placed_rooms[item["subject"]]:
            counts["dropped: is_in_room implied by a placement"] += 1
            continue
        out.append(item)
    return out, counts


# ------------------------------------------------------------------ grading

def grade_typed(gt: List[Dict[str, Any]], pred: List[Dict[str, Any]], scene: Scene):
    """Recall at the kind level, with the two ways it can flatter kept in view.

    `consistent` credits a match only when the model wrote no more lines for that kind of
    object than the ground truth has for it: listing a cup on every table in the house
    finds the right table and is still not an answer, and the composer would plan all
    seven. `ambiguous` counts matches on requirements whose kind has an instance in the
    scene the proposition does not accept, where naming the kind may bind the wrong one.
    """
    taken: set = set()
    open_gt = list(range(len(gt)))
    hits = []
    left = []
    kinds_said: Counter = Counter()
    wrong_target = 0
    for item in pred:
        subject = scene.resolve(item["subject"], scene.objects, taken)
        kind = category(subject) if subject else category(item["subject"])
        kinds_said[kind] += 1
        target = scene.resolve(item.get("target"), scene.names) if item.get("target") else None
        hit = next((i for i in open_gt if gt[i]["key"] == item["key"] and kind in gt[i]["subject_kinds"]
                    and (target in gt[i]["target_ok"] or (not gt[i]["target_ok"] and target is None))), None)
        if hit is not None:
            open_gt.remove(hit)
            hits.append((kind, hit))
            if subject:
                taken.add(subject)
        else:
            left.append((item, kind, target))
            if any(gt[i]["key"] == item["key"] and kind in gt[i]["subject_kinds"] for i in open_gt):
                wrong_target += 1
    # A placement on furniture in room R makes `is_in_room(x, R)` true when it is executed, so
    # it answers that requirement even though the key differs. Counted apart as `implied`.
    implied = 0
    for item, kind, target in left:
        if item["key"] not in PLACEMENTS:
            continue
        room = scene.room_of_furniture.get(target)
        hit = next((i for i in open_gt if gt[i]["key"] == "is_in_room" and kind in gt[i]["subject_kinds"]
                    and room in gt[i]["target_ok"]), None)
        if hit is not None:
            open_gt.remove(hit)
            hits.append((kind, hit))
            implied += 1
    kinds_wanted = Counter(kind for r in gt for kind in r["subject_kinds"])
    return {
        "matched": len(hits),
        "implied": implied,
        "consistent": sum(1 for kind, _ in hits if kinds_said[kind] <= kinds_wanted[kind]),
        "ambiguous": sum(1 for _, i in hits if gt[i]["ambiguous"]),
        "wrong_target": wrong_target,
    }


def reference_lines(gt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"key": r["key"], "subject": category(r["subject"]), "target": r["target"]} for r in gt]


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--graphs", type=Path, required=True, help="detailed_traces of a full-obs run on --split")
    ap.add_argument("--vocab-graphs", type=Path, nargs="+", default=None,
                    help="detailed_traces dirs the object vocabulary is read from (default: --graphs, "
                         "which leaks the episodes' own kinds into the shortlist; pass train runs "
                         "disjoint from --split for a number that means anything)")
    ap.add_argument("--operators", default="results/partnr_operators_iir1.json")
    ap.add_argument("--inside-prior", type=Path, default=None,
                    help="per furniture kind is_on_top/is_inside counts from train episodes disjoint "
                         "from --split (outputs/cand_iface_0914/inside_prior_train.json)")
    ap.add_argument("--arms", nargs="+", default=["free", "typed"], choices=["free", "typed", "typed2", "typed3", "typed4"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:8063/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None, help="k/n: only every n-th episode starting at k, "
                    "so n endpoints can each take one share")
    ap.add_argument("--skip-rows", type=Path, nargs="+", default=[],
                    help="rows.jsonl files whose episodes are already done")
    ap.add_argument("--merge", type=Path, nargs="+", default=None,
                    help="call nothing: summarize these rows.jsonl files together into --out")
    ap.add_argument("--reproject-graphs", type=Path, nargs="+", default=None,
                    help="with --merge: re-grade the stored answers under the current projection, "
                         "kinds and prior, finding each episode's graph in these dirs")
    ap.add_argument("--vocab-csv", type=Path, default=None,
                    help="also take kinds from the benchmark's object category table "
                         "(data/hssd-hab/metadata/object_categories_filtered.csv)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="projection keeps repeated lines up to the count the instruction gives")
    ap.add_argument("--snap-furniture", action="store_true",
                    help="projection maps a made-up furniture index to that kind in the named room")
    ap.add_argument("--room-snap", action="store_true",
                    help="projection moves a placement into the one room the task names that has that kind")
    ap.add_argument("--collapse-places", action="store_true",
                    help="instead of dropping an object listed over >2 places: one place per room, "
                         "then the named rooms' places (up to 3) or the first two written")
    ap.add_argument("--snap-first", action="store_true",
                    help="a made-up furniture index with several candidates in the named rooms "
                         "snaps to the first instead of being dropped")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import os
    from partnr_task_types import classify
    from skill_memory_v2.partnr_memory import PartnrSkillMemory

    global INSIDE_PRIOR
    if args.inside_prior:
        INSIDE_PRIOR = json.loads((ROOT / args.inside_prior).read_text())
    memory = PartnrSkillMemory.load(str(ROOT / args.operators))
    effects = memory.effects()
    menu = memory.intents()
    kinds = {kind for directory in (args.vocab_graphs or [args.graphs]) for kind in vocabulary(directory)}
    if args.vocab_csv:
        import csv
        with open(ROOT / args.vocab_csv) as handle:
            kinds |= {category(row["clean_category"]) for row in csv.DictReader(handle)
                      if row.get("clean_category", "").strip()}
    kinds = sorted(kinds)
    global KEEP_DUPLICATES, SNAP_FURNITURE, ROOM_SNAP, COLLAPSE_PLACES, SNAP_FIRST
    KEEP_DUPLICATES, SNAP_FURNITURE, ROOM_SNAP = args.keep_duplicates, args.snap_furniture, args.room_snap
    COLLAPSE_PLACES, SNAP_FIRST = args.collapse_places, args.snap_first
    with gzip.open(DATASET / f"{args.split}.json.gz") as handle:
        episodes = {str(e["episode_id"]): e for e in json.load(handle)["episodes"]}
    # Only episodes whose stats have landed: the run that writes these graphs may still be
    # going, and a trace without its stats file may be half written.
    stats = args.graphs.parent / "stats"
    traces = sorted(
        path for path in args.graphs.glob("detailed_trace-*.pkl")
        if not stats.is_dir()
        or (stats / f"{path.name[len('detailed_trace-'):-len('.pkl')].split('_')[1]}.json").exists()
    )
    done = set()
    for rows_file in args.skip_rows:
        done |= {json.loads(line)["episode"] for line in open(rows_file)}
    traces = [path for path in traces if path.name[len("detailed_trace-"):-len(".pkl")] not in done]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        traces = traces[k::n]
    if args.limit:
        traces = traces[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    llm = None
    if args.model:
        from omegaconf import OmegaConf
        from habitat_llm.llm.vllm_chat import VLLMChat
        os.environ["VLLM_BASE_URL"] = args.base_url
        conf = OmegaConf.load(ROOT / "habitat_llm/conf/llm/vllm.yaml")
        conf.generation_params.model = args.model
        llm = VLLMChat(conf)

    def one(path: Path, answers: Optional[Dict[str, str]] = None, prompts: Optional[Dict[str, str]] = None):
        """Grade one episode. With `answers`, the stored responses are re-graded and no model
        is called -- the projection can then be changed without paying for generation."""
        key = path.name[len("detailed_trace-"):-len(".pkl")]
        episode = episodes.get(key.split("_")[1])
        graph = first_graph(path)
        if episode is None or graph is None:
            return {"episode": key, "status": "no episode or graph"}
        scene = Scene(graph)
        gt = ground_truth(episode, scene)
        if not gt:
            return {"episode": key, "status": "no resolvable propositions"}
        short = shortlist(episode["instruction"], kinds)
        row = {"episode": key, "status": "ok", "type": classify(episode), "gt": len(gt),
               "statable": sum(1 for r in gt if r["key"] in effects),
               "shortlist": len(short),
               "subject_in_shortlist": sum(1 for r in gt if r["subject_kinds"] & set(short)),
               "subject_in_vocab": sum(1 for r in gt if r["subject_kinds"] & set(kinds)),
               "target_typed": sum(1 for r in gt if r["key"] != "is_inside" or r["target_ok"] & set(scene.openable)),
               "arms": {}}
        ref, _ = project(reference_lines(gt), scene, kinds, effects, episode["instruction"])
        row["reference"] = grade_typed(gt, ref, scene)["matched"]
        for arm in args.arms:
            if answers is not None:
                prompt, answer = (prompts or {}).get(arm, ""), answers.get(arm, "")
            else:
                prompt = free_prompt(scene, episode["instruction"], menu) if arm == "free" \
                    else typed_prompt(scene, episode["instruction"], short, effects,
                                      examples=arm if arm in ("typed2", "typed3", "typed4") else False)
                # typed4 opens with an `Objects:` line that 7B follows with a blank line, and the
                # planner's "\n\n" stop then ends the answer before any requirement (221/342 on
                # train_mini). It stops at the next task instead; only `|` lines are parsed.
                stop = "\nTask:" if arm == "typed4" else "\n\n"
                answer = "" if llm is None else (llm.generate(prompt, stop=stop, max_length=384) or "")
            raw = parse_free(answer, effects) if arm == "free" else parse_typed(answer, effects)
            projected, counts = project(raw, scene, kinds, effects, episode["instruction"])
            exact = diag.grade(diag.ground_truth(episode, scene.handle_to_name, fold=True), raw,
                               scene.names, set(effects))["exact"] if arm == "free" else None
            row["arms"][arm] = {"prompt": prompt, "answer": answer, "pred": len(raw),
                                "pred_proj": len(projected), "exact": exact,
                                "typed": grade_typed(gt, raw, scene),
                                "typed_proj": grade_typed(gt, projected, scene),
                                "projection": dict(counts)}
        return row

    if args.merge:
        rows = [json.loads(line) for rows_file in args.merge for line in open(rows_file)]
        seen = Counter(r["episode"] for r in rows)
        if any(count > 1 for count in seen.values()):
            raise SystemExit(f"episodes in more than one rows file: {[e for e, c in seen.items() if c > 1][:5]}")
        if args.reproject_graphs:
            where = {path.name[len("detailed_trace-"):-len(".pkl")]: path
                     for directory in args.reproject_graphs for path in directory.glob("detailed_trace-*.pkl")}
            stored = [r for r in rows if r["status"] == "ok"]
            missing = [r["episode"] for r in stored if r["episode"] not in where]
            if missing:
                raise SystemExit(f"no graph for {len(missing)} stored episodes, e.g. {missing[:3]}")
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                rows = list(pool.map(
                    lambda r: one(where[r["episode"]],
                                  answers={arm: r["arms"][arm]["answer"] for arm in args.arms},
                                  prompts={arm: r["arms"][arm]["prompt"] for arm in args.arms}),
                    stored))
    else:
        with ThreadPoolExecutor(max_workers=args.workers if llm else 1) as pool:
            rows = list(pool.map(one, traces))

    ok = [r for r in rows if r["status"] == "ok"]
    total = Counter()
    for r in ok:
        for field in ("gt", "statable", "subject_in_shortlist", "subject_in_vocab", "target_typed", "reference"):
            total[field] += r[field]
    summary = {"split": args.split, "model": args.model, "operators": args.operators, "effects": effects,
               "vocabulary": len(kinds), "episodes": len(ok),
               "status": dict(Counter(r["status"] for r in rows)), "totals": dict(total), "arms": {}}
    for arm in args.arms:
        a = Counter()
        proj = Counter()
        for r in ok:
            x = r["arms"][arm]
            a["pred"] += x["pred"]; a["pred_proj"] += x["pred_proj"]
            a["exact"] += x["exact"] or 0
            a["typed"] += x["typed"]["matched"]; a["typed_proj"] += x["typed_proj"]["matched"]
            a["wrong_target"] += x["typed"]["wrong_target"]; a["wrong_target_proj"] += x["typed_proj"]["wrong_target"]
            a["consistent"] += x["typed"]["consistent"]; a["consistent_proj"] += x["typed_proj"]["consistent"]
            a["ambiguous_proj"] += x["typed_proj"]["ambiguous"]
            a["implied_proj"] += x["typed_proj"].get("implied", 0)
            # The sim cell's zero-step episodes: an answer that the projection empties.
            a["emptied"] += int(x["pred"] > 0 and x["pred_proj"] == 0)
            a["episodes"] += 1
            proj.update(x["projection"])
        by_type = defaultdict(Counter)
        for r in ok:
            b = by_type[r["type"]]
            b["gt"] += r["gt"]; b["typed"] += r["arms"][arm]["typed"]["matched"]
            b["typed_proj"] += r["arms"][arm]["typed_proj"]["matched"]; b["episodes"] += 1
        summary["arms"][arm] = {"counts": dict(a), "projection": dict(proj),
                                "by_type": {k: dict(v) for k, v in sorted(by_type.items())}}

    with open(args.out / "rows.jsonl", "w") as handle:  # disk before print
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))

    gt_n = max(total["gt"], 1)
    print(f"{args.split}: {len(ok)} episodes, {total['gt']} requirements, vocabulary {len(kinds)} kinds, "
          f"effects {effects}")
    print(f"  statable {total['statable']/gt_n:.3f}   subject kind in vocabulary {total['subject_in_vocab']/gt_n:.3f}"
          f"   in shortlist {total['subject_in_shortlist']/gt_n:.3f}   is_inside target openable-typed "
          f"{total['target_typed']/gt_n:.3f}")
    print(f"  reference answers through projection + typed grader: {total['reference']/gt_n:.3f}")
    for arm, s in summary["arms"].items():
        c = Counter(s["counts"])
        print(f"\n  [{arm}]  exact {c['exact']/gt_n:.3f}   typed recall {c['typed']/gt_n:.3f} "
              f"(precision {c['typed']/max(c['pred'],1):.3f})   +proj recall {c['typed_proj']/gt_n:.3f} "
              f"(precision {c['typed_proj']/max(c['pred_proj'],1):.3f})")
        print(f"      consistent recall {c['consistent']/gt_n:.3f}   +proj {c['consistent_proj']/gt_n:.3f}"
              f"   (+proj matches on ambiguous kinds: {c['ambiguous_proj']}; "
              f"is_in_room answered by a placement: {c['implied_proj']})")
        print(f"      answers the projection empties: {c['emptied']}/{c['episodes']}")
        for reason, n in Counter(s["projection"]).most_common():
            print(f"      {n:6d}  {reason}")
        for kind, b in s["by_type"].items():
            print(f"      {kind:8s} eps {b['episodes']:4d}  gt {b['gt']:5d}  typed {b['typed']/max(b['gt'],1):.3f}"
                  f"  +proj {b['typed_proj']/max(b['gt'],1):.3f}")
    print(f"\nwrote {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
