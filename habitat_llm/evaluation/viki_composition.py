# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if hasattr(value, "tolist"):
        return _native(value.tolist())
    return value


@dataclass(frozen=True)
class PrimitiveUnit:
    kind: str
    arguments: Tuple[Any, ...]
    event_index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "arguments": list(self.arguments),
            "event_index": self.event_index,
        }

    def canonical(self) -> str:
        arguments = ",".join(str(value) for value in self.arguments)
        return f"{self.kind}({arguments})"


@dataclass(frozen=True)
class CompositionSignature:
    units: Sequence[PrimitiveUnit]
    plan_length: int
    object_count: int
    action_skeleton: Sequence[Sequence[str]]

    def ordered_units(self) -> List[str]:
        return [unit.canonical() for unit in self.units]

    def full_signature(self) -> str:
        value = {
            "ordered_units": self.ordered_units(),
            "plan_length": self.plan_length,
            "object_count": self.object_count,
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "units": [unit.to_dict() for unit in self.units],
            "ordered_units": self.ordered_units(),
            "plan_length": self.plan_length,
            "object_count": self.object_count,
            "action_skeleton": [list(step) for step in self.action_skeleton],
            "full_signature": self.full_signature(),
        }


def _append_unit(
    units: List[PrimitiveUnit],
    kind: str,
    arguments: Tuple[Any, ...],
    event_index: int,
) -> None:
    unit = PrimitiveUnit(kind, arguments, event_index)
    if not units or units[-1].canonical() != unit.canonical():
        units.append(unit)


def parse_composition(ground_truth: Mapping[str, Any]) -> CompositionSignature:
    plan = _native(ground_truth)["time_steps"]
    units: List[PrimitiveUnit] = []
    carried: Dict[str, List[str]] = {}
    fetched_assets: List[str] = []
    relocated_assets: List[str] = []
    pending_open: Dict[str, str] = {}
    pending_check: Dict[str, str] = {}
    event_index = 0
    skeleton = []

    for step in plan:
        actions = [
            (robot, action)
            for robot, action in sorted(step["actions"].items())
            if action is not None
        ]
        skeleton.append(tuple(str(action[0]) for _, action in actions))
        for robot, action in actions:
            event_index += 1
            action_name = str(action[0])
            arguments = tuple(str(value) for value in action[1:])
            target = arguments[0] if arguments else ""
            if action_name == "Grasp":
                _append_unit(units, "fetch", (target,), event_index)
                fetched_assets.append(target)
                carried.setdefault(robot, []).append(target)
                if robot in pending_open:
                    _append_unit(
                        units,
                        "open_container_then_retrieve",
                        (pending_open[robot], target),
                        event_index,
                    )
                    del pending_open[robot]
                if robot in pending_check:
                    _append_unit(
                        units,
                        "check_then_act",
                        (pending_check[robot], target),
                        event_index,
                    )
                    del pending_check[robot]
            elif action_name == "Place":
                for asset in carried.get(robot, []):
                    _append_unit(units, "relocate", (asset, target), event_index)
                    relocated_assets.append(asset)
                carried[robot] = []
            elif action_name == "Open":
                _append_unit(units, "state_change", ("open", target), event_index)
                pending_open[robot] = target
            elif action_name == "Close":
                _append_unit(units, "state_change", ("close", target), event_index)
            elif action_name == "Interact":
                _append_unit(units, "state_change", ("interact", target), event_index)
                pending_check[robot] = target
            elif action_name == "Push":
                _append_unit(
                    units,
                    "state_change",
                    ("push",) + arguments,
                    event_index,
                )
            elif action_name == "Handover":
                _append_unit(
                    units,
                    "state_change",
                    ("handover",) + arguments,
                    event_index,
                )

    unique_relocated = list(dict.fromkeys(relocated_assets))
    if len(unique_relocated) > 1:
        _append_unit(
            units,
            "multi_object_sequence",
            (len(unique_relocated),),
            event_index + 1,
        )
    return CompositionSignature(
        units=units,
        plan_length=len(plan),
        object_count=len(unique_relocated),
        action_skeleton=skeleton,
    )
