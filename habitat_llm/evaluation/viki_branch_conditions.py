# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

TRANSPORT_VERBS = (
    "arrange",
    "bring",
    "carry",
    "collect",
    "deposit",
    "deliver",
    "drop",
    "fetch",
    "gather",
    "get",
    "insert",
    "lay",
    "load",
    "move",
    "pick up",
    "place",
    "position",
    "put",
    "relocate",
    "retrieve",
    "serve",
    "set",
    "shift",
    "store",
    "transfer",
    "transport",
)
CUT_VERBS = ("chop", "cut", "slice", "trim")
TRANSPORT_VERB_FORMS = (
    "arrange",
    "arranged",
    "arranging",
    "bring",
    "bringing",
    "brought",
    "carried",
    "carry",
    "carrying",
    "collect",
    "collected",
    "collecting",
    "deposit",
    "deposited",
    "depositing",
    "deliver",
    "delivered",
    "delivering",
    "drop",
    "dropped",
    "dropping",
    "end up",
    "fetch",
    "fetched",
    "fetching",
    "gather",
    "gathered",
    "gathering",
    "get",
    "gets",
    "getting",
    "got",
    "insert",
    "inserted",
    "inserting",
    "lay",
    "laying",
    "leave",
    "leaving",
    "left",
    "line up",
    "lined up",
    "lining up",
    "load",
    "loaded",
    "loading",
    "move",
    "moved",
    "moving",
    "pick up",
    "picked up",
    "picking up",
    "place",
    "placed",
    "placing",
    "position",
    "positioned",
    "positioning",
    "pop",
    "popped",
    "popping",
    "put",
    "putting",
    "relocate",
    "relocated",
    "relocating",
    "retrieve",
    "retrieved",
    "retrieving",
    "rest",
    "rested",
    "resting",
    "serve",
    "served",
    "serving",
    "set",
    "setting",
    "shift",
    "shifted",
    "shifting",
    "store",
    "stored",
    "storing",
    "stack",
    "stacked",
    "stacking",
    "transfer",
    "transferred",
    "transferring",
    "transport",
    "transported",
    "transporting",
)
CUT_VERB_FORMS = (
    "carve",
    "carved",
    "carving",
    "chop",
    "chopped",
    "chopping",
    "cut",
    "cutting",
    "slice",
    "sliced",
    "slicing",
    "trim",
    "trimmed",
    "trimming",
)
RELATION_VERB_FORMS = (
    TRANSPORT_VERB_FORMS
    + CUT_VERB_FORMS
    + (
        "belong",
        "belongs",
        "belonging",
    )
)
TARGET_PREPOSITIONS = (
    "over to",
    "toward",
    "onto",
    "into",
    "inside",
    "to",
    "on",
    "in",
)
CONDITIONAL_TERMS = (
    "absent",
    "can't see",
    "cannot see",
    "don't detect",
    "isn't present",
    "missing",
    "not already",
    "not present",
    "only one",
    "supply the other",
    "whichever",
)
SOURCE_MARKERS = ("from", "including", "such as", "wherever")
INSTRUMENT_MARKERS = ("using", "with")
INSPECTION_CUES = (
    "check",
    "confirm",
    "examine",
    "glance",
    "inspect",
    "look at",
    "look over",
    "observe",
    "scan",
    "sensors",
    "take a look",
    "verify",
)
PRONOUNS = {"it", "them", "one", "ones", "those", "items"}
INVALID_DISCOVERED_REGIONS = {
    "a",
    "an",
    "check",
    "current",
    "it",
    "one",
    "the",
    "them",
}


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if hasattr(value, "tolist"):
        return _native(value.tolist())
    return value


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = normalized.replace("’", "'").replace("—", " ").replace("–", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def get_instruction(sample: Mapping[str, Any]) -> str:
    messages = _native(sample["prompt"])
    return next(
        message["content"].replace("<image>", "").strip()
        for message in messages
        if message["role"] == "user"
    )


def asset_type(entity_id: str) -> str:
    match = re.fullmatch(r"(.+)_([0-9]+)", entity_id)
    if match is None:
        raise ValueError(f"Unexpected VIKI asset id {entity_id!r}")
    return match.group(1)


def initial_state(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    reward_model = _native(sample["reward_model"])
    return reward_model["ground_truth"]["init_pos"]


def derive_train_vocabularies(
    samples: Iterable[Mapping[str, Any]],
) -> Tuple[Set[str], Set[str]]:
    assets: Set[str] = set()
    locations: Set[str] = set()
    for sample in samples:
        for entity_id, positions in initial_state(sample).items():
            if entity_id.startswith("R") and entity_id[1:].isdigit():
                continue
            assets.add(asset_type(entity_id))
            if positions is not None:
                locations.update(str(value) for value in _native(positions))
    return assets, locations


def derive_train_portable_assets(
    samples: Iterable[Mapping[str, Any]],
) -> Set[str]:
    portable = set()
    for sample in samples:
        reward_model = _native(sample["reward_model"])
        for step in reward_model["ground_truth"]["time_steps"]:
            for action in step["actions"].values():
                if action is not None and action[0] in {"Reach", "Grasp"}:
                    portable.add(str(action[1]))
    return portable


def _phrase_pattern(phrases: Iterable[str]) -> str:
    escaped = [
        re.escape(value) for value in sorted(set(phrases), key=len, reverse=True)
    ]
    return "(?:" + "|".join(escaped) + ")"


def _mentions(text: str, vocabulary: Iterable[str]) -> List[Tuple[int, int, str]]:
    results = []
    occupied: List[Tuple[int, int]] = []
    for value in sorted(set(vocabulary), key=len, reverse=True):
        pattern = rf"(?<![a-z0-9]){re.escape(normalize_text(value))}(?![a-z0-9])"
        for match in re.finditer(pattern, text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
            results.append((span[0], span[1], value))
    return sorted(results)


def discover_instruction_regions(
    instructions: Iterable[str],
    asset_vocabulary: Set[str],
    initial_locations: Set[str],
) -> Set[str]:
    known = asset_vocabulary | initial_locations
    verbs = _phrase_pattern(TRANSPORT_VERBS)
    prepositions = _phrase_pattern(TARGET_PREPOSITIONS)
    counts: Dict[str, int] = {}
    for raw_text in instructions:
        text = normalize_text(raw_text)
        for match in re.finditer(
            rf"\b{verbs}\b[^.;!?]{{0,100}}?\b{prepositions}\b\s+(?:the\s+)?"
            rf"([a-z][a-z ]{{0,35}})",
            text,
        ):
            tail = re.split(
                r"\b(?:and|as|at|before|for|if|so|then|to|using|with)\b|[,.;!?]",
                match.group(1),
                maxsplit=1,
            )[0].strip()
            words = tail.split()
            if not words:
                continue
            candidates = [
                " ".join(words[:size]) for size in range(min(4, len(words)), 0, -1)
            ]
            candidate = next(
                (value for value in candidates if value in known), words[0]
            )
            if candidate not in INVALID_DISCOVERED_REGIONS and len(candidate) > 1:
                counts[candidate] = counts.get(candidate, 0) + 1
    return {value for value, count in counts.items() if count >= 2}


@dataclass(frozen=True)
class AssetCondition:
    asset: str
    target: str
    status: str

    def to_dict(self) -> Dict[str, str]:
        return {"asset": self.asset, "target": self.target, "status": self.status}


@dataclass(frozen=True)
class BranchPredicate:
    instruction: str
    branch: str
    conditions: Sequence[AssetCondition]
    unresolved_assets: Sequence[str]

    @property
    def absent_assets(self) -> List[str]:
        return [
            condition.asset
            for condition in self.conditions
            if condition.status != "present_at_target"
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "branch": self.branch,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "absent_assets": self.absent_assets,
            "unresolved_assets": list(self.unresolved_assets),
        }


class AvailabilityPredicate:
    def __init__(
        self,
        asset_vocabulary: Iterable[str],
        location_vocabulary: Iterable[str],
        portable_asset_vocabulary: Optional[Iterable[str]] = None,
    ) -> None:
        self.assets = set(asset_vocabulary)
        self.locations = set(location_vocabulary)
        self.targets = self.assets | self.locations
        self.portable_assets = set(portable_asset_vocabulary or self.assets)
        alias_candidates: Dict[str, List[str]] = {}
        for asset in self.assets:
            words = asset.split()
            if len(words) > 1:
                alias_candidates.setdefault(words[-1], []).append(asset)
        self.asset_aliases = {
            alias: values[0]
            for alias, values in alias_candidates.items()
            if len(values) == 1 and alias not in self.assets and len(alias) > 3
        }

    def _canonicalize_aliases(self, text: str) -> str:
        for alias, asset in sorted(
            self.asset_aliases.items(), key=lambda item: len(item[0]), reverse=True
        ):
            text = re.sub(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                asset,
                text,
            )
        return text

    def _condition_target(self, text: str) -> Optional[str]:
        if not any(term in text for term in CONDITIONAL_TERMS):
            return None
        first_conditional = min(
            (text.index(term) for term in CONDITIONAL_TERMS if term in text),
            default=len(text),
        )
        before_condition = text[:first_conditional]
        if not any(cue in before_condition for cue in INSPECTION_CUES):
            return None
        regions = [
            mention
            for mention in _mentions(before_condition, self.targets)
            if mention[2] in self.locations
        ]
        if not regions:
            return None
        return regions[0][2]

    def _explicit_pairs(self, text: str) -> List[Tuple[str, str]]:
        verbs = _phrase_pattern(RELATION_VERB_FORMS)
        prepositions = _phrase_pattern(TARGET_PREPOSITIONS)
        relation_pattern = re.compile(
            rf"\b(?P<verb>{verbs})\b(?P<body>[^.;!?]{{0,140}}?)"
            rf"\b(?P<prep>{prepositions})\b\s+(?:the\s+)?"
        )
        asset_mentions = _mentions(text, self.assets)
        target_mentions = _mentions(text, self.targets)
        positioned_pairs: List[Tuple[int, str, str]] = []
        focus_assets: List[str] = []
        for match in relation_pattern.finditer(text):
            target = next(
                (
                    value
                    for start, _, value in target_mentions
                    if start >= match.end() and start <= match.end() + 45
                ),
                None,
            )
            if target is None:
                continue
            body_start, body_end = match.span("body")
            subjects = [
                value
                for start, end, value in asset_mentions
                if start >= body_start
                and end <= body_end
                and value in self.portable_assets
            ]
            body = match.group("body")
            if "from" in body:
                absolute_marker = body_start + body.index("from")
                subjects = [
                    value
                    for start, _, value in asset_mentions
                    if body_start <= start < absolute_marker
                    and value in self.portable_assets
                ]
            if any(pronoun in body.split() for pronoun in PRONOUNS) and not subjects:
                subjects = list(focus_assets)
            if not subjects:
                previous = [
                    value
                    for start, _, value in asset_mentions
                    if max(0, match.start() - 100) <= start < match.start()
                    and value in self.portable_assets
                ]
                subjects = previous[-2:]
            subjects = [value for value in subjects if value != target]
            if subjects:
                focus_assets = list(dict.fromkeys(subjects))
            for subject in dict.fromkeys(subjects):
                positioned_pairs.append((match.start(), subject, target))
        conditional_match = re.search(r"\b(?:if|in case|should)\b", text)
        if conditional_match is not None:
            primary = [
                pair for pair in positioned_pairs if pair[0] < conditional_match.start()
            ]
            if primary:
                positioned_pairs = primary
        return [(asset, target) for _, asset, target in positioned_pairs]

    def extract_pairs(self, instruction: str) -> List[Tuple[str, str]]:
        text = self._canonicalize_aliases(normalize_text(instruction))
        asset_mentions = [value for _, _, value in _mentions(text, self.assets)]
        conditional_pairs: List[Tuple[str, str]] = []
        conditional_target = self._condition_target(text)
        if conditional_target is not None:
            source_cutoff = min(
                (text.index(marker) for marker in SOURCE_MARKERS if marker in text),
                default=len(text),
            )
            required_assets = [
                value
                for start, _, value in _mentions(text[:source_cutoff], self.assets)
                if value != conditional_target
            ]
            conditional_pairs.extend(
                (asset, conditional_target) for asset in required_assets
            )
        explicit_pairs = self._explicit_pairs(text)

        fallback_pairs: List[Tuple[str, str]] = []
        if any(verb in text for verb in CUT_VERB_FORMS) and ":" in text:
            prefix, suffix = text.split(":", 1)
            prefix_targets = [
                value
                for _, _, value in _mentions(prefix, self.targets)
                if value in self.portable_assets
            ]
            if prefix_targets:
                target = prefix_targets[-1]
                instrument_cutoff = min(
                    (
                        suffix.index(marker)
                        for marker in INSTRUMENT_MARKERS
                        if marker in suffix
                    ),
                    default=len(suffix),
                )
                subjects = [
                    value
                    for _, _, value in _mentions(
                        suffix[:instrument_cutoff], self.portable_assets
                    )
                    if value != target
                ]
                fallback_pairs.extend((subject, target) for subject in subjects)

        deduplicated = list(dict.fromkeys(conditional_pairs))
        seen = {asset for asset, _ in deduplicated}
        final_explicit: Dict[str, str] = {}
        for asset, target in explicit_pairs + fallback_pairs:
            if asset not in seen:
                final_explicit[asset] = target
        deduplicated.extend(final_explicit.items())
        if not deduplicated and len(asset_mentions) == 1 and conditional_target:
            deduplicated.append((asset_mentions[0], conditional_target))
        return deduplicated

    def evaluate(self, sample: Mapping[str, Any]) -> BranchPredicate:
        instruction = get_instruction(sample)
        pairs = self.extract_pairs(instruction)
        state: Dict[str, List[str]] = {}
        known_assets = set()
        for entity_id, positions in initial_state(sample).items():
            if entity_id.startswith("R") and entity_id[1:].isdigit():
                continue
            entity_type = asset_type(entity_id)
            known_assets.add(entity_type)
            if positions is not None:
                state.setdefault(entity_type, []).extend(
                    str(value) for value in _native(positions)
                )
        conditions = []
        for asset, target in pairs:
            positions = state.get(asset, [])
            if target in positions:
                status = "present_at_target"
            elif positions:
                status = "present_elsewhere"
            else:
                status = "absent_from_scene"
            conditions.append(AssetCondition(asset, target, status))
        referenced_assets = {
            value
            for _, _, value in _mentions(
                self._canonicalize_aliases(normalize_text(instruction)), self.assets
            )
        }
        resolved_assets = {condition.asset for condition in conditions}
        unresolved = sorted(
            asset
            for asset in referenced_assets - resolved_assets
            if asset in known_assets
        )
        if not conditions:
            branch = "not_applicable"
        elif all(condition.status == "present_at_target" for condition in conditions):
            branch = "all_present"
        else:
            branch = "some_absent"
        return BranchPredicate(instruction, branch, conditions, unresolved)
