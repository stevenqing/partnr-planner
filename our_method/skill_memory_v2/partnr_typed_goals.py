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
# PARTNR's H family is unary -- "the mug must be clean" names no place -- and the line grammar
# below is `object | relation | place`, three columns always. So a state line writes `-` where a
# placement names furniture, and everything here is gated on `state`, which is off by default:
# with it off, prompt, parser and projection behave exactly as they did before it existed.
STATE_RELATIONS = {"clean": "is_clean", "powered_on": "is_powered_on"}
STATE_KEYS = set(STATE_RELATIONS.values())

EXAMPLES_STATE = """Task: Clean the mug and switch on the lamp.
mug | clean | -
lamp | powered_on | -
"""
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
                 effects: List[str], examples: str = "RST", stops: bool = False,
                 state: bool = False) -> str:
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
    for word, key in STATE_RELATIONS.items():
        if state and key in effects:
            offered.append(f"  {word:8s} -- no place: write -  (the object itself must end up "
                           f"{word.replace('_', ' ')}; it does not move)")
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
            # `stops` keeps the stage rules without the two-stage example: 37% of intermediate
            # stops on train_mini temporal episodes were never written under the R/RS rules.
            if examples == "RST" or stops else
            "Rules: choose the single piece of furniture the task means -- in the room the task "
            "names -- and never list alternatives. Nothing else.\n"
            "Objects are not in the description yet: name them by kind from the list above. "
            "Write only where objects must end up, never where they start.\n\n"
        )
        + f"{EXAMPLE_SETS[examples]}"
        + (EXAMPLES_STATE if state and any(k in effects for k in STATE_KEYS) else "")
        + "\n"
        f"Task: {instruction}\n"
        "Requirements:\n"
    )


def parse_typed(text: str, effects: List[str], state: bool = False) -> List[Dict[str, Any]]:
    out = []
    for line in str(text).splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line.strip())
        pieces = [p.strip() for p in line.split("|")]
        if len(pieces) != 3 or not all(pieces):
            continue
        word = pieces[1].lower().replace(" ", "_")
        key = RELATIONS.get(word)
        if key is None and state:
            key = STATE_RELATIONS.get(word)
            if key in effects:
                # The third column is the placeholder `-`; a state requirement has no target.
                out.append({"key": key, "subject": pieces[0], "target": None})
                continue
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
            inside_prior: Optional[Dict[str, Dict[str, int]]] = None, instruction: str = "",
            state: bool = False) -> Tuple[List[Dict[str, Any]], Counter]:
    """The memory's type check: keep what can be typed, retype what is mistyped.

    Matches the inner loop's `--keep-duplicates --snap-furniture --room-snap --collapse-places
    --snap-first` projection (variant v7b on all of train_mini: typed3 consistent recall 0.651,
    answers emptied 3/395, reference 0.904).
    """
    counts: Counter = Counter()
    fixed = {category(n) for n in scene.furniture} | {category(n) for n in scene.rooms}
    kept = []
    for item in pred:
        # A state predicate is typed before the placement rules, because its subject is whatever the
        # task names: PARTNR asks for a *table* to be clean (every is_clean proposition in gate_H and
        # conf_H names furniture) and for a *lamp* to be powered on. The rule below -- drop a subject
        # that is furniture or a room -- exists because a placement's subject has to be carryable, and
        # applying it here is what made is_clean 0.000 on both models while is_powered_on reached 0.78:
        # `table_7 | clean | -` parsed correctly and was then thrown away as "subject is furniture".
        if state and item["key"] in STATE_KEYS:
            if item["key"] not in effects:
                counts["dropped: relation not in the memory"] += 1
                continue
            named = resolve(item["subject"], scene.furniture)
            if named:
                kept.append({"key": item["key"], "subject": named, "target": None})
                continue
            state_subject = category(item["subject"])
            if state_subject not in kinds:
                near = sorted((k for k in kinds if related(state_subject, k)), key=len)
                if not near:
                    counts["dropped: nothing answers to the state subject"] += 1
                    continue
                state_subject = near[0]
                counts["retyped: state subject to a known kind"] += 1
            kept.append({"key": item["key"], "subject": state_subject, "target": None})
            continue
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


# ------------------------------------------------------------------ stages from the instruction
#
# Every temporal episode in train_mini and val_mini has a DAG whose closure is "stage(u) <
# stage(v)" over stages contiguous in proposition order, and an order word in the instruction.
# So the DAG is one stage number per line, read here from the instruction's words alone -- no
# temporal example, no model call. Most edges join different objects, which the line-order
# rule below (a second place for the same object) cannot see. Frozen at rule v2, tuned on
# train_mini failures only (scripts/partnr_stage_recovery.py): ordered-pair recall on the
# stored 7B typedRS answers 0.971, precision 0.975, against 0.278 for the line-order rule.

STEP = r"then|next(?!\s+to\b)|finally|lastly|after\s+that|afterwards?|subsequently|secondly|second|third"
LEAD = rf"(?:and\s+)?(?:{STEP}|first(?:ly)?|before|after|once)\b"
# A spatial anchor names an object without moving it; the anchor phrase ends at a comma or "and".
ANCHOR = re.compile(r"\b(?:next\s+to|beside|besides|near|close\s+to)\b[^,;.?!]*?(?=,|;|\.|\?|!|\band\b|$)")
PRONOUN = re.compile(r"\b(?:it|them|they|these|those|both)\b")


def _words(text: str) -> List[str]:
    return [singular(w) for w in re.findall(r"[a-z]+", text.lower())]


def order_clauses(instruction: str) -> List[Dict[str, Any]]:
    """Clauses in text order, each with the order word that opens it (or None)."""
    out = []
    for sentence_index, sentence in enumerate(re.split(r"(?<=[.!?;])\s+", instruction.strip())):
        # Cut before an order word that follows a comma or "and", or that opens the sentence;
        # also before a bare mid-clause "before"/"after" ("place it before moving the cup").
        pieces = re.split(rf"(?:,\s*|\s+and\s+|\s+)(?=(?:{LEAD}))", sentence, flags=re.I)
        # "Before / After / Once A, B": the comma, not an order word, separates the two halves.
        if pieces and re.match(r"\s*(?:before|after(?!\s+that)|once)\b", pieces[0], flags=re.I) and "," in pieces[0]:
            head, _, rest = pieces[0].partition(",")
            pieces = [head, rest] + pieces[1:]
        for position, piece in enumerate(p for p in pieces if p and p.strip(" ,")):
            text = piece.strip(" ,").lower()
            match = re.match(rf"(?:and\s+)?({STEP}|first(?:ly)?|before|after|once)\b", text)
            lead = re.sub(r"\s+", " ", match.group(1)) if match else None
            out.append({"sentence": sentence_index, "position": position, "text": text, "lead": lead})
    return out


def _stage_clauses(parts: List[Dict[str, Any]]) -> None:
    """Give every clause a stage number, in place."""
    stage = 0
    seen_work = False
    by_sentence = defaultdict(list)
    for part in parts:
        by_sentence[part["sentence"]].append(part)
    for sentence in sorted(by_sentence):
        group = by_sentence[sentence]
        head = group[0]["lead"]
        if head in ("before",) and len(group) > 1:
            # "Before A, B": B happens first.
            for part in group[1:]:
                part["stage"] = stage
            group[0]["stage"] = stage + 1
            stage += 1
            seen_work = True
            continue
        for index, part in enumerate(group):
            lead = part["lead"]
            if lead and re.fullmatch(STEP, lead) and seen_work:
                stage += 1
            elif lead in ("before",) and index > 0:
                stage += 1  # "A before B"
            elif lead in ("after",) and index > 0:
                # "A after B": B first. Push every earlier clause of this sentence one stage later.
                for earlier in group[:index]:
                    earlier["stage"] = stage + 1
                part["stage"] = stage
                stage += 1
                seen_work = True
                continue
            elif lead in ("after", "once") and index == 0 and len(group) > 1:
                part["stage"] = stage  # "After A, B": A then B
                stage += 1
                seen_work = True
                for later in group[1:]:
                    later["stage"] = stage
                break
            part["stage"] = stage
            seen_work = True


def _unanchored(text: str) -> str:
    return ANCHOR.sub(" ", text)


def _full_match(kind: str, head: str) -> bool:
    tokens = [singular(t) for t in kind.split("_") if t]
    return bool(tokens) and (all(t in set(_words(head)) for t in tokens)
                             or kind.replace("_", "") in head.replace(" ", ""))


def _named_kinds(part: Dict[str, Any], kinds) -> List[str]:
    """Kinds the clause names other than as a spatial anchor.

    A kind whose full name is not there ("plant_container" for "the plants") still counts on
    one long token of its name, unless another kind in play already claims that token here.
    """
    head = _unanchored(part["text"])
    said = set(_words(head))
    full = [k for k in set(kinds) if _full_match(k, head)]
    claimed = {singular(t) for k in full for t in k.split("_")}
    partial = [k for k in set(kinds) if k not in full
               and any(len(t) >= 4 and singular(t) in said and singular(t) not in claimed
                       for t in k.split("_"))]
    return sorted(full + partial)


def stage_lines(instruction: str, kinds, targets=None) -> Tuple[List[int], List[Dict[str, Any]]]:
    """A stage for each line, given each line's kind of object (and target), in line order."""
    targets = list(targets) if targets is not None else [None] * len(kinds)
    parts = order_clauses(instruction)
    _stage_clauses(parts)
    first_marked = next((i for i, p in enumerate(parts) if p["lead"]), None)
    # Places per kind, in line order: copies sent to the same place are one stage, a second
    # place for the kind is a later one.
    places: Dict[str, List[Any]] = defaultdict(list)
    for kind, target in zip(kinds, targets):
        if target not in places[kind]:
            places[kind].append(target)
    moved_twice = {k for k, p in places.items() if len(p) > 1}
    # A pronoun refers to what the previous sentence (and this sentence so far) named:
    # "Then, place them next to each other", "all these items, including the basket".
    where: Dict[str, List[int]] = defaultdict(list)
    by_sentence: Dict[int, set] = defaultdict(set)
    seen: set = set()
    bump = 0
    for index, part in enumerate(parts):
        named = set(_named_kinds(part, kinds))
        referred = set(named)
        if PRONOUN.search(_unanchored(part["text"]).replace("each other", "")):
            earlier = [s for s in by_sentence if s < part["sentence"]]
            referred |= by_sentence[max(earlier)] if earlier else set()
            referred |= by_sentence[part["sentence"]]
        # Sending an already-named kind to its second place is a new stage even with no
        # order word ("...on the coffee table. Move these toys to the bed.").
        if not part["lead"] and index > 0 and (referred & seen & moved_twice):
            bump += 1
        part["stage"] = part["stage"] + bump
        part["referred"] = sorted(referred)
        part["named"] = sorted(named)
        for kind in referred:
            where[kind].append(index)
        seen |= referred
        by_sentence[part["sentence"]] |= named
    stages: List[Optional[int]] = []
    for kind, target in zip(kinds, targets):
        spots = where.get(kind) or []
        if not spots:
            stages.append(None)
            continue
        k = places[kind].index(target)
        if len(places[kind]) == 1:
            # Overview clauses before the first order word only count when the kind is not
            # mentioned again after it.
            marked = [s for s in spots if first_marked is not None and s >= first_marked]
            chosen = (marked or spots)[0]
        else:
            # The k-th place of the kind goes with the k-th distinct stage it is mentioned in.
            distinct = []
            for s in spots:
                if not distinct or parts[s]["stage"] != parts[distinct[-1]]["stage"]:
                    distinct.append(s)
            chosen = distinct[min(k, len(distinct) - 1)]
        stages.append(parts[chosen]["stage"])
    # A line whose kind the instruction never names keeps the stage of the line before it.
    last = 0
    out = []
    for stage in stages:
        last = stage if stage is not None else last
        out.append(last)
    return out, parts


def as_requirements(chosen: List[Dict[str, Any]], instruction: str = "",
                    stages: bool = False, same_object: bool = False) -> List[Dict[str, Any]]:
    """Planner requirements with their ordering.

    Without `stages`, the answer's line order is the only ordering: a second place for the
    same object is ordered after the first, so "first the couch, then the table" becomes a
    stage rather than two contradictory end states. With `stages`, every line waits for the
    lines of the nearest earlier stage read off the instruction (`stage_lines`), which also
    orders different objects -- most temporal edges in PARTNR do.
    """
    requirements: List[Dict[str, Any]] = []
    if stages and chosen:
        numbers, _ = stage_lines(instruction, [item["subject"] for item in chosen],
                                 [item["target"] for item in chosen])
        levels = sorted(set(numbers))
        previous = {level: levels[i - 1] for i, level in enumerate(levels) if i > 0}
        # The same object moved twice: the k-th copy of a kind at its second place is the k-th
        # copy at its first. Without this link `_bind` excludes every instance another line has
        # bound, so the second stage can never take the object the first stage moved -- on the
        # stage cell 138 of 229 "never saw" notes on temporal episodes were kinds with two places.
        places: Dict[str, List[Any]] = defaultdict(list)
        copies: Dict[Tuple[str, Any], List[int]] = defaultdict(list)
        for index, item in enumerate(chosen):
            if item["target"] not in places[item["subject"]]:
                places[item["subject"]].append(item["target"])
            copies[(item["subject"], item["target"])].append(index)
        same_as: Dict[int, int] = {}
        # Only across stages: places of one kind inside one stage are different objects or a
        # list of options ("the plants on table_18 and table_19", "the cushions on bed_20, 23,
        # 26"), and chaining them made every line bind one instance -- 4 of the 8 episodes the
        # first version made worse on the train_mini pool. A place links to the nearest place of
        # that kind in a strictly earlier stage.
        place_stage = {key: min(numbers[i] for i in indices) for key, indices in copies.items()}
        for kind, targets in places.items():
            for position, later in enumerate(targets):
                stage = place_stage[(kind, later)]
                prior = [t for t in targets[:position] if place_stage[(kind, t)] < stage]
                if not prior:
                    continue
                earlier = max(prior, key=lambda t: place_stage[(kind, t)])
                for k, index in enumerate(copies[(kind, later)]):
                    if k < len(copies[(kind, earlier)]):
                        same_as[index] = copies[(kind, earlier)][k]
        for index, item in enumerate(chosen):
            before = previous.get(numbers[index])
            requirement = {
                "key": item["key"],
                "subject": item["subject"],
                "target": item["target"],
                "alternatives": [item["target"]],
                "next_to": None,
                "proposition": index,
                "stage": numbers[index],
                "after_propositions": [j for j, n in enumerate(numbers) if before is not None and n == before],
            }
            if same_object:
                requirement["same_as"] = same_as.get(index)
            requirements.append(requirement)
        return requirements
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


EACH_OTHER = re.compile(r"^(?:next\s+to|close\s+to|beside|near)\s+(?:each\s+other|one\s+another)\b")
ANCHOR_WORD = re.compile(r"^(?:next\s+to|beside|besides|near|close\s+to)\b")


def beside_requirements(instruction: str, chosen: List[Dict[str, Any]], start: int = 0) -> List[Dict[str, Any]]:
    """`is_next_to` requirements read off the instruction, over objects the answer places.

    The typed answer has no place for "next to", so the relation is read from the words, the
    way stages are: "next to each other" chains the clause's objects in line order ("the phone,
    watch and tape ... next to each other" is phone-watch, watch-tape, as PARTNR writes it);
    "X next to Y" pairs the object named last before the anchor (or the pronoun's objects) with
    the object in the anchor phrase. Only pairs whose two objects the answer places are kept:
    those fold into the placement (`fold_spatial`); a fixed anchor ("next to the sofa") has no
    operator in an R-only library and would only burn retries. The later-staged object carries
    the relation, so the fold's own ordering (after the anchor lands) never contradicts a stage.
    """
    if not chosen:
        return []
    kinds = [item["subject"] for item in chosen]
    numbers, parts = stage_lines(instruction, kinds, [item["target"] for item in chosen])
    placed = [i for i, item in enumerate(chosen) if item["key"] in PLACEMENTS]

    def line_of(kind: str, stage: int) -> Optional[int]:
        same = [i for i in placed if kinds[i] == kind and numbers[i] == stage]
        anywhere = [i for i in placed if kinds[i] == kind]
        return (same or anywhere or [None])[-1]

    pairs: List[Tuple[int, int]] = []
    for part in parts:
        text = part["text"]
        for match in ANCHOR.finditer(text):
            phrase = match.group(0)
            if EACH_OTHER.match(phrase):
                # The clause's own objects when it names two or more; a pronoun's otherwise
                # ("Then, take the backpack and ball ... next to each other" is not the phone).
                named = part.get("named", [])
                group = [line_of(k, part["stage"]) for k in (named if len(named) >= 2 else part.get("referred", []))]
                group = sorted({i for i in group if i is not None})
                pairs += list(zip(group, group[1:]))
                continue
            inside = ANCHOR_WORD.sub("", phrase)
            anchors = [k for k in set(kinds) if _full_match(k, inside)]
            head = _unanchored(text[: match.start()])
            before = [k for k in set(kinds) if _full_match(k, head) and k not in anchors]
            if before:
                # the kind whose name ends latest before the anchor
                def last_position(kind: str) -> int:
                    token = singular(kind.split("_")[-1])
                    spots = [m.start() for m in re.finditer(r"[a-z]+", head) if singular(m.group(0)) == token]
                    return max(spots) if spots else -1
                movers = [max(before, key=last_position)]
            else:
                movers = [k for k in part.get("referred", []) if k not in anchors]
            for mover in movers:
                for anchor in anchors:
                    i, j = line_of(mover, part["stage"]), line_of(anchor, part["stage"])
                    if i is not None and j is not None and i != j:
                        pairs.append((i, j))
    out: List[Dict[str, Any]] = []
    seen = set()
    for i, j in pairs:
        if frozenset((i, j)) in seen:
            continue
        seen.add(frozenset((i, j)))
        holder, other = (i, j) if (numbers[i], i) >= (numbers[j], j) else (j, i)
        out.append({
            "key": "is_next_to",
            "subject": kinds[holder],
            "target": kinds[other],
            "alternatives": [kinds[other]],
            "next_to": None,
            "proposition": start + len(out),
            "after_propositions": [],
        })
    return out
