"""Requirements chosen from what the memory says is type-valid, not written freely.

The intent arm asks the model to write the end state in predicates, and at 7B that form
fails on types rather than on reading: `is_inside(laptop, workout_1)`, a table as the
subject, every table in the house listed as the destination. The memory already knows the
types -- a placement ends on a piece of furniture, `is_in_room` ends in a room, the kinds of
object that exist -- so here it offers them and checks the answer, and the model is left to
choose which object, which relation and which place.

Three pieces, shared by the planner (`goal_source: typed`) and the offline inner loop
(`scripts/partnr_typed_intent.py`), so what is tuned on train is what runs:

  shortlist     kinds of object the instruction could mean, from the train vocabulary.
  typed_prompt  the offered relations with the places their type allows.
  project       the type check on the answer: retype, drop what cannot be typed, drop
                an object listed over more than two places, drop `is_in_room` implied by a
                placement, and say `is_inside` only where train says the benchmark does.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

RELATIONS = {"on": "is_on_top", "inside": "is_inside", "in_room": "is_in_room"}
PLACEMENTS = ("is_on_top", "is_inside")
INDEXED = re.compile(r"^(.*)_\d+$")

# Written by hand over a house that does not exist, so no episode of any split leaks into
# the prompt; they teach the shape of an answer, not a scene.
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


def category(name: Optional[str]) -> str:
    if not name:
        return ""
    name = str(name).strip().lower().replace(" ", "_")
    match = INDEXED.match(name)
    return match.group(1) if match else name


def related(asked: Optional[str], truth: Optional[str]) -> bool:
    """Containment or a token subset -- `phone` joins `cellphone`, `toy_vehicle` does not
    join `toy_animal`."""
    if not asked or not truth:
        return False
    a, b = category(asked), category(truth)
    if a == b or a in b or b in a:
        return True
    return set(a.split("_")) <= set(b.split("_")) or set(b.split("_")) <= set(a.split("_"))


def resolve(asked: Optional[str], names: List[str], exclude=()) -> Optional[str]:
    """Exact name, then instance-of-category, then prefix; the least assuming wins."""
    if not asked:
        return None
    if asked in names and asked not in exclude:
        return asked
    want = str(asked).strip().lower().replace(" ", "_")
    pool = [name for name in names if name not in exclude]
    for candidates in (
        [n for n in pool if n.lower() == want],
        [n for n in pool if n.lower().rsplit("_", 1)[0] == want],
        [n for n in pool if n.lower().startswith(want + "_")],
    ):
        if candidates:
            return sorted(candidates, key=lambda n: (len(n), n))[0]
    return None


class StepZero:
    """What the typing needs from a world graph, all of it known before anything is seen."""

    def __init__(self, rooms, furniture, openable, room_of_furniture):
        self.rooms = sorted(rooms)
        self.furniture = sorted(furniture)
        self.openable = sorted(openable)
        self.room_of_furniture = dict(room_of_furniture)

    @classmethod
    def from_graph(cls, graph) -> "StepZero":
        from habitat_llm.world_model import Floor

        furniture = graph.get_all_furnitures()
        return cls(
            rooms=[n.name for n in graph.get_all_rooms()],
            furniture=[n.name for n in furniture],
            openable=[n.name for n in furniture
                      if n.properties.get("is_articulated") and not isinstance(n, Floor)],
            room_of_furniture={furn.name: room
                               for room, items in graph.group_furniture_by_room().items()
                               for furn in items},
        )


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


# The compositional protocol builds everything from rearrangement alone and tests on its
# compositions, so the examples are a variable: `R` shows single-stage rearrangement only,
# `RS` adds spatial ("next to") but no ordering, `RST` is the original set, whose third example
# teaches a two-stage move. `R` and `RS` also drop the rules that describe stages.
EXAMPLES_R = """Examples (from other houses):
Task: Move the mug and the book from the kitchen counter to the bedroom table.
mug | on | table_7
book | on | table_7
Task: Take the soap to the laundry room.
soap | in_room | laundryroom_1
Task: Bring the vase to the living room couch.
vase | on | couch_12
Task: Put the phone on the office shelves.
phone | on | shelves_4
"""

EXAMPLES_RS = """Examples (from other houses):
Task: Move the mug and the book from the kitchen counter to the bedroom table.
mug | on | table_7
book | on | table_7
Task: Take the soap to the laundry room.
soap | in_room | laundryroom_1
Task: Move the cup and the plate to the dining table and place them next to each other.
cup | on | table_20
plate | on | table_20
Task: Put the lamp on the bedroom table next to the clock.
lamp | on | table_9
"""

EXAMPLE_SETS = {"R": EXAMPLES_R, "RS": EXAMPLES_RS, "RST": EXAMPLES}


def typed_prompt(world: str, instruction: str, kinds: List[str], scene: StepZero,
                 effects: List[str], examples: str = "RST") -> str:
    """The `typed3` prompt of the inner loop (`examples="RST"`), or its R / RS variants."""
    if examples not in EXAMPLE_SETS:
        raise ValueError(f"typed examples must be one of {sorted(EXAMPLE_SETS)}, not {examples!r}")
    offered = []
    if "is_on_top" in effects:
        offered.append("  on       -- place: one furniture name from the Furniture list above")
    if "is_inside" in effects and scene.openable:
        offered.append(f"  inside   -- place: one of {', '.join(scene.openable)}")
    if "is_in_room" in effects:
        offered.append(f"  in_room  -- place: one of {', '.join(scene.rooms)}; "
                       "use it only when the task names a room and no furniture in it")
    return (
        f"{world}\n\n"
        "Say where each object must be when the task is done, one line each, as\n"
        "  object | relation | place\n\n"
        f"object: a kind the task mentions, from -- {', '.join(kinds)}\n"
        "relation:\n" + "\n".join(offered) + "\n\n"
        + (
            "Rules: choose the single piece of furniture the task means -- in the room the task "
            "names -- and never list alternatives. Write a second line for an object only if the "
            "task moves it again, in the task's order. Nothing else.\n"
            "Objects are not in the description yet: name them by kind from the list above. "
            "Write only where objects must end up (and any stop the task asks for on the way), "
            "never where they start.\n\n"
            if examples == "RST" else
            "Rules: choose the single piece of furniture the task means -- in the room the task "
            "names -- and never list alternatives. Nothing else.\n"
            "Objects are not in the description yet: name them by kind from the list above. "
            "Write only where objects must end up, never where they start.\n\n"
        )
        + f"{EXAMPLE_SETS[examples]}\n"
        f"Task: {instruction}\n"
        "Requirements:\n"
    )


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


def inside_allowed(furniture: str, prior: Optional[Dict[str, Dict[str, int]]]) -> bool:
    """Whether the benchmark calls a placement on this kind of furniture `is_inside`.

    Read from train, not asked of the model: on 468 train episodes PARTNR labels 677
    placements `is_on_top` and 3 `is_inside` -- "in the cabinet" is `is_on_top` there.
    Without a prior the model's word stands.
    """
    if prior is None:
        return True
    counts = prior.get(category(furniture), {})
    return counts.get("is_inside", 0) > counts.get("is_on_top", 0)


NUMBERS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "both": 2, "pair": 2}


def copies_allowed(instruction: str, kind: str, written: int) -> int:
    """How many identical lines for this kind the instruction licenses: "2 spoons" is two,
    a bare "spoon" one, a plural without a number what the model wrote up to four."""
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


def snap(target: Optional[str], scene: StepZero, instruction: str) -> Optional[str]:
    """A furniture index the house does not have, mapped to that kind in the room the task
    names when that room has exactly one, or in the house when it does; otherwise none."""
    kind = category(target)
    same = [f for f in scene.furniture if category(f) == kind]
    if not same:
        return None
    text = instruction.lower()
    named = {r for r in scene.rooms if category(r).replace("_", " ") in text}
    in_named = [f for f in same if scene.room_of_furniture.get(f) in named]
    if len(in_named) == 1:
        return in_named[0]
    # Refusing here emptied 4 of the typed sim cell's 120 episodes (`table_30` with four
    # living-room tables); any table in the named room is often one the proposition accepts.
    if in_named:
        return sorted(in_named)[0]
    return same[0] if len(same) == 1 else None


def project(pred: List[Dict[str, Any]], scene: StepZero, kinds: List[str], effects: List[str],
            inside_prior: Optional[Dict[str, Dict[str, int]]] = None, instruction: str = ""
            ) -> Tuple[List[Dict[str, Any]], Counter]:
    """The memory's type check: keep what can be typed, retype what is mistyped.

    Matches the inner loop's `--keep-duplicates --snap-furniture --room-snap --collapse-places
    --snap-first` projection (variant v7b on all of train_mini: typed3 consistent recall 0.651,
    answers emptied 3/395, reference 0.904).
    """
    counts: Counter = Counter()
    fixed = {category(n) for n in scene.furniture} | {category(n) for n in scene.rooms}
    kept = []
    for item in pred:
        subject = category(item["subject"])
        if subject in fixed:
            counts["dropped: subject is furniture or a room"] += 1
            continue
        if subject not in kinds:
            near = sorted((k for k in kinds if related(subject, k)), key=len)
            if not near:
                counts["dropped: no kind of object answers to the subject"] += 1
                continue
            subject = near[0]
            counts["retyped: subject to a known kind"] += 1
        target = item.get("target")
        room = resolve(target, scene.rooms)
        furn = None if room else resolve(target, scene.furniture)
        if not room and not furn:
            furn = snap(target, scene, instruction)
            if furn:
                counts["snapped: made-up furniture index"] += 1
        if room:
            key, target = "is_in_room", room
        elif furn:
            key = "is_inside" if (item["key"] == "is_inside" and furn in scene.openable
                                  and inside_allowed(furn, inside_prior)) else "is_on_top"
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
        else:
            written = sum(1 for other in kept if other == item)
            if sum(1 for other in unique if other == item) < copies_allowed(instruction, item["subject"], written):
                unique.append(item)
                counts["kept: repeated line the instruction counts"] += 1
    # An object listed over many places is mostly a list of options, but dropping all of its
    # lines emptied 7 of the typed sim cell's 120 episodes. Keep what the listing still says:
    # one place per kind of furniture per room ("the table, then the couch" in one room is two
    # stages; four living-room tables is one choice), then the named rooms' places (up to 3)
    # or the first two written.
    named = {r for r in scene.rooms if category(r).replace("_", " ") in instruction.lower()}

    def room_of(target):
        return target if target in scene.rooms else scene.room_of_furniture.get(target)

    chosen: Dict[Any, str] = {}
    collapsed = []
    for item in unique:
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
            keep[subject] = set([t for t in targets if room_of(t) in named][:3] or targets[:2])
    trimmed = [item for item in collapsed if item["target"] in keep[item["subject"]]]
    if len(trimmed) < len(collapsed):
        counts["trimmed: places beyond the named rooms or the first two"] += len(collapsed) - len(trimmed)
    unique = trimmed
    # The right kind of furniture in the wrong room is the commonest target error (176 against
    # 21 wrong kinds on train_mini). The room is in the instruction's words, so the placement
    # moves there when exactly one named room has that kind; with two it is left alone.
    if instruction:
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


def as_requirements(chosen: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Planner requirements, with a second place for the same object ordered after the first.

    The answer's line order is the only ordering this arm has, and the prompt asks for the
    task's order exactly so that "first the couch, then the table" becomes a stage rather
    than two contradictory end states.
    """
    requirements: List[Dict[str, Any]] = []
    last: Dict[str, int] = {}
    for item in chosen:
        index = len(requirements)
        earlier = last.get(item["subject"])
        requirements.append({
            "key": item["key"],
            "subject": item["subject"],
            "target": item["target"],
            "alternatives": [item["target"]],
            "next_to": None,
            "proposition": index,
            "after_propositions": [earlier] if earlier is not None else [],
        })
        last[item["subject"]] = index
    return requirements
