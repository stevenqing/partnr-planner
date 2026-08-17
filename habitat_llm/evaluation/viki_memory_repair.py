# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import ast
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Set

from habitat_llm.evaluation.viki_memory_skill import (
    SkillInstance,
    _to_native,
    get_prompt_context,
)

ENTITY_ACTIONS = {"Reach", "Grasp", "Open", "Close", "Interact"}


def _asset_type(entity_id: str) -> str:
    match = re.fullmatch(r"(.+)_([0-9]+)", entity_id)
    if match is None:
        raise ValueError(f"Unexpected VIKI asset instance id {entity_id!r}")
    return match.group(1)


def current_asset_types(sample: Mapping[str, Any]) -> Set[str]:
    ground_truth = _to_native(sample["reward_model"])["ground_truth"]
    return {
        _asset_type(entity_id)
        for entity_id, positions in ground_truth["init_pos"].items()
        if positions is not None
        and not (entity_id.startswith("R") and entity_id[1:].isdigit())
    }


def parse_answer_plan(response: str) -> List[Mapping[str, Any]]:
    match = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    if match is None:
        return []
    try:
        value = ast.literal_eval(match.group(1).strip())
    except (SyntaxError, TypeError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return value


def skill_length_caps(
    instances: Sequence[SkillInstance], multiplier: float = 1.5
) -> Dict[str, int]:
    maxima: Dict[str, int] = {}
    for instance in instances:
        plan = ast.literal_eval(instance.demonstration)
        maxima[instance.skill_name] = max(maxima.get(instance.skill_name, 0), len(plan))
    return {
        skill_name: int(math.ceil(multiplier * maximum))
        for skill_name, maximum in maxima.items()
    }


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reasons: Sequence[str]
    plan_length: int
    length_cap: int
    illegal_actions: Sequence[str]
    ungrounded_entities: Sequence[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "reasons": list(self.reasons),
            "plan_length": self.plan_length,
            "length_cap": self.length_cap,
            "illegal_actions": list(self.illegal_actions),
            "ungrounded_entities": list(self.ungrounded_entities),
        }


class TrainOnlyPlanValidator:
    def __init__(
        self,
        instances: Sequence[SkillInstance],
        length_multiplier: float = 1.5,
    ) -> None:
        self.length_multiplier = length_multiplier
        self.length_caps = skill_length_caps(instances, length_multiplier)

    def validate(
        self,
        sample: Mapping[str, Any],
        response: str,
        routed_skill: str,
    ) -> ValidationResult:
        plan = parse_answer_plan(response)
        if not plan:
            cap = self.length_caps[routed_skill]
            return ValidationResult(
                False,
                ["invalid_answer"],
                0,
                cap,
                [],
                [],
            )

        _, robots, available_actions = get_prompt_context(dict(sample))
        assets = current_asset_types(sample)
        illegal_actions = []
        ungrounded_entities = []
        reasons = []
        for step in plan:
            if not isinstance(step, dict) or not isinstance(step.get("actions"), dict):
                reasons.append("invalid_answer")
                continue
            for robot, action in step["actions"].items():
                if (
                    robot not in robots
                    or not isinstance(action, (list, tuple))
                    or not action
                ):
                    illegal_actions.append(f"{robot}:<malformed>")
                    continue
                action_name = str(action[0])
                if action_name not in available_actions[robot]:
                    illegal_actions.append(f"{robot}:{action_name}")
                if action_name in ENTITY_ACTIONS:
                    for entity in action[1:]:
                        if str(entity) not in assets:
                            ungrounded_entities.append(str(entity))

        cap = self.length_caps[routed_skill]
        if illegal_actions:
            reasons.append("illegal_action")
        if ungrounded_entities:
            reasons.append("ungrounded_entity")
        if len(plan) > cap:
            reasons.append("train_skill_length_cap")
        return ValidationResult(
            not reasons,
            sorted(set(reasons)),
            len(plan),
            cap,
            sorted(set(illegal_actions)),
            sorted(set(ungrounded_entities)),
        )
