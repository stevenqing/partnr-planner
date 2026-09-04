#!/usr/bin/env python3
"""Adapter between VIKI-L2 data and the hierarchical skill memory.

The method itself is untouched. This module only translates, and every
translation choice it makes is listed in ADAPTER_CHOICES so the choices can be
reviewed separately from the results.

Two directions are covered:

  build side     VIKI training plans -> PARTNR-style trace text, which is what
                 HierarchicalSkillMemory.parse_trace_file already consumes.
  retrieval side one VIKI-L2-Interactive row -> the four arguments that
                 HierarchicalRetriever.retrieve expects.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Tuple

# Robot ids in VIKI are R1, R2, R3; the memory speaks Agent_0, Agent_1, ...
ROBOT_PATTERN = re.compile(r"^R(\d+)$")

ADAPTER_CHOICES = {
    "robot_id_mapping": (
        "VIKI R<n> maps to Agent_<n-1>, so R1 becomes Agent_0. The self robot "
        "R1 therefore stays agent 0, matching the method's own convention that "
        "agent 0 is the acting agent."
    ),
    "agent_enumeration": (
        "The upstream builder hardcodes `for agent_id in [0, 1]`. VIKI rows "
        "carry three robots, so the adapter reports every robot present in the "
        "row and the caller iterates over that list instead. Without this, R3 "
        "skills would never be extracted."
    ),
    "objects_section": (
        "The parser's object line carries a distance in metres and a room name, "
        "neither of which VIKI measures. Rather than invent them, the section is "
        "emitted empty. Object identity still reaches the memory through the "
        "action targets on every action line."
    ),
    "furniture_section": (
        "VIKI has no furniture graph. The section is emitted empty rather than "
        "invented, so the memory simply has no furniture features for VIKI."
    ),
    "robot_entities_excluded": (
        "VIKI's init_pos lists the robots themselves alongside the objects. "
        "Robot ids are filtered out of environment_state so the retriever does "
        "not treat a teammate as a manipulable object."
    ),
    "agent_state": (
        "VIKI-L2 is single-shot, so there is no execution-time agent state. "
        "The adapter supplies the initial state from ground_truth.init_pos. "
        "This is a limitation of the benchmark, not a modelling choice."
    ),
    "environment_state": (
        "Likewise built from init_pos plus layout_id. No simulated rollout is "
        "available to update it."
    ),
    "partner_effects": (
        "Derived from the exposed step-1 partner actions of "
        "VIKI-L2-Interactive. This is the one input the interactive extension "
        "genuinely provides, and it is the reason the method can be evaluated "
        "here at all. Supplied both per agent and in the flat action/"
        "moved_objects form generate_query actually reads; the per-agent-only "
        "form used through Amendment 8b silently reached the query as nothing."
    ),
    "query_key_names": (
        "generate_query reads environment_state['seen_objects'] and the flat "
        "partner_effects keys, while _build_context_query reads "
        "environment_state['objects']. Amendment 8b supplied only the latter, so "
        "abstract skill matching ran on the goal string alone. Both spellings are "
        "now emitted from the same source data; no new information is invented."
    ),
    "observation_lines": (
        "Each action line is followed by a success observation, because the "
        "training plans are ground truth and therefore succeeded. The parser "
        "uses the observation to advance its step counter."
    ),
}


def _location_text(location: Any) -> str:
    """init_pos stores a location as a one-element list; the query builder wants
    a string."""
    if isinstance(location, (list, tuple)):
        return ", ".join(str(item) for item in location)
    return str(location)


def robot_to_agent(robot_id: str) -> int:
    match = ROBOT_PATTERN.match(robot_id)
    if not match:
        raise ValueError(f"Unexpected VIKI robot id: {robot_id}")
    return int(match.group(1)) - 1


def parse_plan(trajectory: str) -> List[Dict[str, Any]]:
    """Recover the plan list from the rendered trajectory text produced by
    Amendment 7, so both memory arms read the identical source records."""
    start = trajectory.find("[")
    end = trajectory.rfind("]")
    if start < 0 or end < 0:
        raise ValueError("Trajectory carries no plan array")
    return json.loads(trajectory[start : end + 1])


def action_parts(value: Any) -> Tuple[str, List[str]]:
    """VIKI stores actions either as a list or as a numpy repr string."""
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value]
    else:
        text = str(value).strip()
        try:
            parts = [str(item) for item in ast.literal_eval(text)]
        except (ValueError, SyntaxError):
            parts = [item.strip("'\" ") for item in text.strip("[]").split()]
    if not parts:
        raise ValueError(f"Empty VIKI action: {value!r}")
    return parts[0], parts[1:]


def render_trace(record: Dict[str, Any]) -> str:
    """One Amendment 7 train interaction -> PARTNR-style trace text."""
    plan = parse_plan(record["trajectory"])
    lines = [f"Task: {record['task_main']}", "Furniture:", "Objects:"]
    for step in plan:
        for robot_id in sorted(step.get("actions", {})):
            value = step["actions"][robot_id]
            if value is None:
                continue
            verb, arguments = action_parts(value)
            agent = robot_to_agent(robot_id)
            lines.append(f"Agent_{agent}_Action: {verb}[{','.join(arguments)}]")
            lines.append(f"Agent_{agent}_Observation: Successful execution!")
    return "\n".join(lines) + "\n"


def agents_in_record(record: Dict[str, Any]) -> List[int]:
    """Every agent that actually acts, replacing the hardcoded [0, 1]."""
    plan = parse_plan(record["trajectory"])
    agents = set()
    for step in plan:
        for robot_id, value in step.get("actions", {}).items():
            if value is not None:
                agents.add(robot_to_agent(robot_id))
    return sorted(agents)


def retrieval_inputs(
    ground_truth: Dict[str, Any], exposed_steps: int, self_robot: str
) -> Dict[str, Any]:
    """One VIKI-L2-Interactive row -> HierarchicalRetriever.retrieve arguments."""
    steps = sorted(ground_truth["time_steps"], key=lambda item: int(item["step"]))
    init_pos = ground_truth.get("init_pos") or {}

    per_partner: Dict[str, Any] = {}
    for step in steps[:exposed_steps]:
        for robot_id, value in sorted(step.get("actions", {}).items()):
            if value is None or robot_id == self_robot:
                continue
            verb, arguments = action_parts(value)
            per_partner[f"agent_{robot_to_agent(robot_id)}"] = {
                "action": verb,
                "target": arguments[0] if arguments else "",
                "step": int(step["step"]),
            }

    # generate_query expects one flat mapping with action/moved_objects, not a
    # per-agent nesting, so partner_effects.get("action") on the nested form
    # returned None and no partner information ever reached the query. VIKI
    # exposes more than one partner, so their actions are joined in agent order.
    partner_effects: Dict[str, Any] = dict(per_partner)
    if per_partner:
        ordered = [per_partner[key] for key in sorted(per_partner)]
        partner_effects["action"] = "; ".join(
            f"{entry['action']} {entry['target']}".strip() for entry in ordered
        )
        moved = [entry["target"] for entry in ordered if entry["target"]]
        if moved:
            partner_effects["moved_objects"] = moved

    objects = {
        name: {"name": name, "location": location}
        for name, location in init_pos.items()
        if location is not None and not ROBOT_PATTERN.match(name)
    }
    # generate_query reads seen_objects/known_rooms while _build_context_query
    # reads objects, so both spellings are supplied from the same source. Without
    # seen_objects the abstract-matching stage saw nothing but the goal string.
    seen_objects = {
        name: {"location": _location_text(entry["location"])}
        for name, entry in objects.items()
    }
    environment_state = {
        "objects": objects,
        "seen_objects": seen_objects,
        "furniture": {},
        "layout_id": ground_truth.get("layout_id"),
    }
    agent_state = {
        "agent_id": robot_to_agent(self_robot),
        "robot_type": (ground_truth.get("robots") or {}).get(self_robot),
        "holding": None,
        "location": None,
    }
    return {
        "agent_state": agent_state,
        "environment_state": environment_state,
        "partner_effects": partner_effects,
        "goal": ground_truth.get("description", ""),
    }
