# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from habitat_llm.evaluation.viki_branch_conditions import (
    AvailabilityPredicate,
    asset_type,
)
from habitat_llm.evaluation.viki_composition import parse_composition
from habitat_llm.evaluation.viki_memory_skill import get_prompt_context

EXTRACTION_SYSTEM_PROMPT = """You build an individual-skill memory from one successful robot plan.
Segment the complete plan at subgoal-completion boundaries. A segment must be a non-empty contiguous range of whole timesteps. Together, the segments must partition every timestep exactly once and in original order.

Use fine-grained reusable skills. End a segment when a concrete subgoal becomes true, for example after finding or grasping an object, after placing or relocating it, after opening or closing a container, after inspecting or changing state, or after completing another distinct manipulation. Do not merge independent object subgoals merely because they belong to one task. Do not split an atomic parallel timestep.

For every segment generate:
- name: a short reusable snake_case behavioral skill name, abstracting away object instance ids;
- description: one concise sentence describing the reusable procedure, without claiming cooperation.

This dataset is single-robot at evaluation scope. Output only individual skills. Never fabricate cooperation skills, partner conditions, actions, or timesteps. Do not copy the plan into the description.

Return JSON only with this schema:
{"segments":[{"start_step":1,"end_step":3,"name":"fetch_object","description":"Navigate to, reach, and grasp a required object."}]}

start_step and end_step are inclusive 1-based positions in the supplied TIMESTEPS array, not the plan's printed step labels."""


class Stage1ExtractionError(ValueError):
    pass


def to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(item) for item in value]
    if hasattr(value, "tolist"):
        return to_native(value.tolist())
    return value


def normalized_skill_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not name or len(name) > 80:
        raise Stage1ExtractionError(f"Invalid skill name {value!r}")
    return name


@dataclass(frozen=True)
class ExtractedSegment:
    start_step: int
    end_step: int
    name: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "name": self.name,
            "description": self.description,
        }


@dataclass(frozen=True)
class Stage1Instance:
    instance_id: str
    source_train_index: int
    segment_index: int
    start_step: int
    end_step: int
    raw_skill_name: str
    skill_name: str
    description: str
    context: str
    demo: Sequence[Mapping[str, Any]]
    self_cond: Mapping[str, Any]
    ordered_units: Sequence[str]
    unit_kinds: Sequence[str]
    source_plan_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "source_train_index": self.source_train_index,
            "segment_index": self.segment_index,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "raw_skill_name": self.raw_skill_name,
            "skill_name": self.skill_name,
            "description": self.description,
            "context": self.context,
            "demo": to_native(self.demo),
            "self_cond": to_native(self.self_cond),
            "ordered_units": list(self.ordered_units),
            "unit_kinds": list(self.unit_kinds),
            "unit_count": len(self.ordered_units),
            "source_plan_sha256": self.source_plan_sha256,
        }


def _segment_units(
    ground_truth: Mapping[str, Any], segments: Sequence[ExtractedSegment]
) -> List[List[Any]]:
    plan = to_native(ground_truth)["time_steps"]
    signature = parse_composition(ground_truth)
    cumulative_events = [0]
    for step in plan:
        cumulative_events.append(
            cumulative_events[-1]
            + sum(action is not None for action in step["actions"].values())
        )
    grouped = []
    for segment_index, segment in enumerate(segments):
        first_event = cumulative_events[segment.start_step - 1] + 1
        last_event = cumulative_events[segment.end_step]
        units = [
            unit
            for unit in signature.units
            if first_event <= unit.event_index <= last_event
            or (
                segment_index == len(segments) - 1
                and unit.event_index > cumulative_events[-1]
            )
        ]
        grouped.append(units)
    return grouped


def _segment_self_condition(
    row_condition: Mapping[str, Any], demo: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    touched_assets = set()
    for step in demo:
        for action in step["actions"].values():
            if action is None:
                continue
            for argument in action[1:]:
                try:
                    touched_assets.add(asset_type(str(argument)))
                except ValueError:
                    touched_assets.add(str(argument))
    conditions = [
        condition
        for condition in row_condition["conditions"]
        if condition["asset"] in touched_assets
    ]
    unresolved_assets = [
        asset for asset in row_condition["unresolved_assets"] if asset in touched_assets
    ]
    if not conditions:
        branch = "not_applicable"
    elif all(condition["status"] == "present_at_target" for condition in conditions):
        branch = "all_present"
    else:
        branch = "some_absent"
    return {
        "instruction": row_condition["instruction"],
        "branch": branch,
        "conditions": conditions,
        "absent_assets": [
            condition["asset"]
            for condition in conditions
            if condition["status"] != "present_at_target"
        ],
        "unresolved_assets": unresolved_assets,
        "segment_assets": sorted(touched_assets),
    }


def build_extraction_messages(sample: Mapping[str, Any]) -> List[Dict[str, str]]:
    instruction, _, _ = get_prompt_context(to_native(sample))
    ground_truth = to_native(sample["reward_model"])["ground_truth"]
    initial_state = ground_truth["init_pos"]
    plan = ground_truth["time_steps"]
    user_prompt = "\n\n".join(
        (
            f"INSTRUCTION:\n{instruction}",
            "INITIAL_STATE:\n"
            + json.dumps(initial_state, sort_keys=True, separators=(",", ":")),
            "TIMESTEPS:\n" + json.dumps(plan, sort_keys=True, separators=(",", ":")),
        )
    )
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _json_value(text: str) -> Any:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced is not None:
        stripped = fenced.group(1)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise Stage1ExtractionError(
            f"Response is not one JSON object: {error}"
        ) from error
    return value


def parse_extraction_response(text: str, plan_length: int) -> List[ExtractedSegment]:
    value = _json_value(text)
    if isinstance(value, list):
        raw_segments = value
    elif (
        isinstance(value, dict)
        and set(value) == {"segments"}
        and isinstance(value["segments"], list)
    ):
        raw_segments = value["segments"]
    else:
        raise Stage1ExtractionError(
            "Response must be a segment array or contain only a segments list"
        )
    if not raw_segments:
        raise Stage1ExtractionError("Response contains no segments")
    segments = []
    expected_start = 1
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise Stage1ExtractionError(f"Segment {index} is not an object")
        required = {"start_step", "end_step", "name", "description"}
        if set(raw_segment) != required:
            raise Stage1ExtractionError(
                f"Segment {index} fields are {sorted(raw_segment)}, expected {sorted(required)}"
            )
        start_step = raw_segment["start_step"]
        end_step = raw_segment["end_step"]
        if type(start_step) is not int or type(end_step) is not int:
            raise Stage1ExtractionError(f"Segment {index} boundaries must be integers")
        if (
            start_step != expected_start
            or end_step < start_step
            or end_step > plan_length
        ):
            raise Stage1ExtractionError(
                f"Segment {index} range {start_step}:{end_step} breaks partition at {expected_start}"
            )
        description = str(raw_segment["description"]).strip()
        if not description or len(description) > 400:
            raise Stage1ExtractionError(f"Segment {index} has invalid description")
        segments.append(
            ExtractedSegment(
                start_step=start_step,
                end_step=end_step,
                name=normalized_skill_name(str(raw_segment["name"])),
                description=description,
            )
        )
        expected_start = end_step + 1
    if expected_start != plan_length + 1:
        raise Stage1ExtractionError(
            f"Segments stop at {expected_start - 1}, plan has {plan_length} timesteps"
        )
    return segments


def build_instances(
    train_index: int,
    sample: Mapping[str, Any],
    segments: Sequence[ExtractedSegment],
    predicate: AvailabilityPredicate,
) -> List[Stage1Instance]:
    native_sample = to_native(sample)
    instruction, _, _ = get_prompt_context(native_sample)
    ground_truth = native_sample["reward_model"]["ground_truth"]
    initial_state = ground_truth["init_pos"]
    plan = ground_truth["time_steps"]
    source_plan_json = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    source_plan_sha256 = hashlib.sha256(source_plan_json.encode("utf-8")).hexdigest()
    context = "\n".join(
        (
            f"Instruction: {instruction}",
            "Initial state: "
            + json.dumps(initial_state, sort_keys=True, separators=(",", ":")),
        )
    )
    row_condition = predicate.evaluate(native_sample).to_dict()
    instances = []
    units_by_segment = _segment_units(ground_truth, segments)
    for segment_index, (segment, units) in enumerate(zip(segments, units_by_segment)):
        demo = plan[segment.start_step - 1 : segment.end_step]
        if not demo:
            raise Stage1ExtractionError("A validated segment produced an empty demo")
        instances.append(
            Stage1Instance(
                instance_id=f"{train_index}:{segment_index}",
                source_train_index=train_index,
                segment_index=segment_index,
                start_step=segment.start_step,
                end_step=segment.end_step,
                raw_skill_name=segment.name,
                skill_name=segment.name,
                description=segment.description,
                context=context,
                demo=demo,
                self_cond=_segment_self_condition(row_condition, demo),
                ordered_units=[unit.canonical() for unit in units],
                unit_kinds=[unit.kind for unit in units],
                source_plan_sha256=source_plan_sha256,
            )
        )
    reconstructed = [step for instance in instances for step in instance.demo]
    if reconstructed != plan:
        raise Stage1ExtractionError(
            "Grounded instance demos do not reconstruct source plan"
        )
    return instances
