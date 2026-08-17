# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import ast
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from omegaconf import OmegaConf

from habitat_llm.agent.agent import Agent
from habitat_llm.llm.base_llm import BaseLLM, Prompt
from habitat_llm.planner.centralized_llm_planner import CentralizedLLMPlanner
from habitat_llm.tools.tool import Tool
from habitat_llm.utils.grammar import (
    FURNITURE,
    NAV_TARGET,
    OBJECT,
    OBJECT_OR_FURNITURE,
    SPATIAL_CONSTRAINT,
    SPATIAL_RELATION,
)
from habitat_llm.world_model import Furniture, House, Object, Room, SpotRobot
from habitat_llm.world_model.world_graph import WorldGraph


def _guided_action_regex(prompt: str) -> str:
    furniture_match = re.search(
        r"Furniture:\n(.*?)\n\nThe following furnitures",
        prompt,
        flags=re.DOTALL,
    )
    objects_match = re.search(
        r"Objects:\n(.*?)\n\nActive agents", prompt, flags=re.DOTALL
    )
    agent_blocks = re.findall(
        r"Agent ID: (\d+)\n(.*?)(?=\nAgent ID:|\n\nRespond|\Z)",
        prompt,
        flags=re.DOTALL,
    )
    if furniture_match is None or objects_match is None or not agent_blocks:
        raise ValueError("Unable to derive PARTNR action grammar from the prompt")

    rooms = []
    furniture: List[str] = []
    for line in furniture_match.group(1).splitlines():
        if ":" not in line:
            continue
        room, contents = line.split(":", 1)
        rooms.append(room.strip())
        furniture.extend(item.strip() for item in contents.split(",") if item.strip())
    objects = [
        line.split(":", 1)[0].strip()
        for line in objects_match.group(1).splitlines()
        if ":" in line
    ]

    def choices(values: List[str]) -> str:
        return "(?:" + "|".join(re.escape(value) for value in values) + ")"

    navigation_targets = rooms + furniture + objects
    references = ["None"] + objects
    action_lines = []
    for agent_id, block in agent_blocks:
        tool_names = set(re.findall(r"^- ([A-Za-z]+):", block, flags=re.MULTILINE))
        calls = []
        if "Navigate" in tool_names and navigation_targets:
            calls.append(rf"Navigate\[{choices(navigation_targets)}\]")
        if "Pick" in tool_names and objects:
            calls.append(rf"Pick\[{choices(objects)}\]")
        for tool_name in ("Place", "Rearrange"):
            if tool_name in tool_names and objects and furniture:
                calls.append(
                    rf"{tool_name}\[{choices(objects)}, (?:on|within), "
                    rf"{choices(furniture)}, (?:None|next_to), "
                    rf"{choices(references)}\]"
                )
        for tool_name in ("Open", "Close"):
            if tool_name in tool_names and furniture:
                calls.append(rf"{tool_name}\[{choices(furniture)}\]")
        if "Explore" in tool_names and furniture + objects:
            calls.append(rf"Explore\[{choices(furniture + objects)}\]")
        if "Wait" in tool_names:
            calls.append(r"Wait\[\]")
        if not calls:
            raise ValueError(f"No valid PARTNR calls available for Agent {agent_id}")
        action_lines.append(rf"Agent_{agent_id}_Action: (?:{'|'.join(calls)})")
    joined_action_lines = "\n".join(action_lines)
    return (
        rf"(?:Final Thought: [^\n]{{1,200}}\nDone\[\]|"
        rf"Thought: [^\n]{{1,200}}\n{joined_action_lines})"
    )


class VikiEndpointLLM(BaseLLM):
    """Use a local OpenAI-compatible endpoint as a PARTNR planner backend."""

    def __init__(self, conf) -> None:
        from openai import OpenAI

        super().__init__(conf)
        self.client = OpenAI(
            base_url=conf.base_url,
            api_key=conf.api_key,
            max_retries=int(conf.get("max_retries", 2)),
        )

    def generate(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
    ) -> str:
        if not isinstance(prompt, str):
            raise TypeError("The oracle-state PARTNR planner expects a text prompt")
        params = self.generation_params
        completion = self.client.chat.completions.create(
            model=params.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_length or params.max_tokens,
            temperature=params.temperature,
            stop=stop,
            extra_body={"guided_regex": _guided_action_regex(prompt)},
        )
        return completion.choices[0].message.content or ""


class VikiDryRunTool(Tool):
    """Expose a PARTNR planning tool without requiring a Habitat motor skill."""

    def __init__(
        self,
        name_arg: str,
        description_arg: str,
        argument_types_arg: List[str],
        agent_uid_arg: int = 0,
    ) -> None:
        super().__init__(name_arg, agent_uid_arg)
        self._description = description_arg
        self._argument_types = argument_types_arg
        self.world_graph: Optional[WorldGraph] = None

    @property
    def description(self) -> str:
        return self._description

    @property
    def argument_types(self) -> List[str]:
        return self._argument_types

    def process_high_level_action(self, input_query, observations):
        if self.world_graph is None:
            return None, f"{self.name} was a success"
        arguments = [item.strip() for item in (input_query or "").split(",")]
        if self.name == "Pick" and arguments[0]:
            object_node = self.world_graph.get_node_from_name(arguments[0])
            agent_node = self.world_graph.get_node_from_name(f"agent_{self.agent_uid}")
            held_objects = [
                node
                for node in self.world_graph.get_all_objects()
                if agent_node in self.world_graph.graph[node]
            ]
            if held_objects and object_node not in held_objects:
                return None, (
                    f"Pick failed; this agent is already holding "
                    f"{held_objects[0].name}"
                )
            self.world_graph.remove_all_edges(object_node)
            self.world_graph.add_edge(agent_node, object_node, "holds", "held by")
            return None, f"Pick was a success; {object_node.name} is held by this agent"
        if self.name in {"Place", "Rearrange"} and len(arguments) >= 3:
            object_node = self.world_graph.get_node_from_name(arguments[0])
            destination = self.world_graph.get_node_from_name(arguments[2])
            agent_node = self.world_graph.get_node_from_name(f"agent_{self.agent_uid}")
            if (
                self.name == "Place"
                and agent_node not in self.world_graph.graph[object_node]
            ):
                return None, (
                    f"Place failed; {object_node.name} must be picked before placing"
                )
            self.world_graph.remove_all_edges(object_node)
            self.world_graph.add_edge(destination, object_node, "under", "on")
            return None, (
                f"{self.name} was a success; {object_node.name} is now on "
                f"{destination.name}"
            )
        return None, f"{self.name} was a success"

    def get_state_description(self) -> str:
        return "Idle"


TOOL_SPECS = {
    "Navigate": (
        "Navigate to an object, furniture, or room. Example: Navigate[table_0]",
        [NAV_TARGET],
    ),
    "Pick": ("Pick up an object. Example: Pick[apple_0]", [OBJECT]),
    "Place": (
        "Place a held object at furniture. Example: "
        "Place[apple_0, on, table_0, None, None]",
        [OBJECT, SPATIAL_RELATION, FURNITURE, SPATIAL_CONSTRAINT, OBJECT],
    ),
    "Rearrange": (
        "Move an object to furniture in one high-level operation. Example: "
        "Rearrange[apple_0, on, table_0, None, None]",
        [OBJECT, SPATIAL_RELATION, FURNITURE, SPATIAL_CONSTRAINT, OBJECT],
    ),
    "Open": ("Open furniture. Example: Open[cabinet_0]", [FURNITURE]),
    "Close": ("Close furniture. Example: Close[cabinet_0]", [FURNITURE]),
    "Explore": (
        "Interact with an object or furniture. Example: Explore[toaster_0]",
        [OBJECT_OR_FURNITURE],
    ),
    "Wait": ("Remain idle for this turn. Example: Wait[]", []),
}


def _parse_prompt_mapping(text: str, label: str) -> Dict[str, Any]:
    match = re.search(rf"^{re.escape(label)}: (.+)$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"VIKI prompt is missing {label!r}")
    value = ast.literal_eval(match.group(1))
    if not isinstance(value, dict):
        raise ValueError(f"VIKI prompt field {label!r} is not a mapping")
    return value


def _prompt_context(
    sample: Dict[str, Any]
) -> Tuple[str, Dict[str, str], Dict[str, List[str]]]:
    system_text = next(
        message["content"]
        for message in sample["prompt"]
        if message["role"] == "system"
    )
    instruction = next(
        message["content"].replace("<image>", "").strip()
        for message in sample["prompt"]
        if message["role"] == "user"
    )
    robots = _parse_prompt_mapping(system_text, "Available robot set")
    APIs = _parse_prompt_mapping(system_text, "Their available operation APIs")
    available_actions = {
        robot: list(APIs[robot_type]) for robot, robot_type in robots.items()
    }
    return instruction, robots, available_actions


def _as_strings(value: Any) -> List[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def load_train_location_types(benchmark_root: Path) -> Set[str]:
    frame = pd.read_parquet(
        benchmark_root / "data/VIKI-R/viki/VIKI-L2/train.parquet",
        columns=["reward_model"],
    )
    locations = set()
    for reward_model in frame["reward_model"]:
        for value in reward_model["ground_truth"]["init_pos"].values():
            locations.update(_as_strings(value))
    return {location for location in locations if not re.fullmatch(r"R\d+", location)}


def _asset_type(instance_name: str) -> str:
    match = re.fullmatch(r"(.+)_\d+", instance_name)
    if match is None:
        raise ValueError(f"Unexpected VIKI asset instance name: {instance_name!r}")
    return match.group(1)


def build_oracle_state_world_graph(
    sample: Dict[str, Any], location_types: Set[str]
) -> Tuple[
    WorldGraph, List[Agent], Dict[int, str], Dict[str, str], Dict[str, List[str]]
]:
    """Build a PARTNR graph from prompt metadata and only VIKI's initial state."""
    instruction, robots, available_actions = _prompt_context(sample)
    initial_positions = sample["reward_model"]["ground_truth"]["init_pos"]

    graph = WorldGraph()
    house = House("house", {"type": "house"})
    room = Room("room_0", {"type": "room"})
    graph.add_node(house)
    graph.add_node(room)
    graph.add_edge(house, room, "has", "in")

    entity_map: Dict[str, str] = {room.name: "room"}
    furniture_by_type = {}
    current_locations = {
        location
        for value in initial_positions.values()
        for location in _as_strings(value)
        if not re.fullmatch(r"R\d+", location)
    }
    mentioned_locations = {
        location
        for location in location_types
        if location.lower() in instruction.lower()
    }
    for location in sorted(current_locations | mentioned_locations):
        name = f"{location}_0"
        furniture = Furniture(name, {"type": location})
        graph.add_node(furniture)
        graph.add_edge(room, furniture, "has", "in")
        furniture_by_type[location] = furniture
        entity_map[name] = location

    agent_map = {}
    agents = []
    robot_nodes = {}
    for agent_id, robot_name in enumerate(sorted(robots)):
        robot_node = SpotRobot(
            f"agent_{agent_id}",
            {"type": robots[robot_name], "translation": [0.0, 0.0, 0.0]},
        )
        graph.add_node(robot_node)
        graph.add_edge(room, robot_node, "has", "in")
        robot_nodes[robot_name] = robot_node
        agent_map[agent_id] = robot_name
        entity_map[robot_name] = robot_name

        APIs = set(available_actions[robot_name])
        tool_names = {"Wait"}
        if "Move" in APIs:
            tool_names.add("Navigate")
        if {"Reach", "Grasp"} <= APIs:
            tool_names.add("Pick")
        if "Place" in APIs:
            tool_names.add("Place")
        if {"Move", "Reach", "Grasp", "Place"} <= APIs:
            tool_names.add("Rearrange")
        tool_names.update(APIs & {"Open", "Close"})
        if "Interact" in APIs:
            tool_names.add("Explore")
        tools = {
            tool_name: {
                "_target_": "habitat_llm.evaluation.viki_partnr_planner.VikiDryRunTool",
                "name_arg": tool_name,
                "description_arg": TOOL_SPECS[tool_name][0],
                "argument_types_arg": TOOL_SPECS[tool_name][1],
            }
            for tool_name in sorted(tool_names)
        }
        agent = Agent(agent_id, OmegaConf.create({"tools": {"planning": tools}}))
        agents.append(agent)

    robot_names = set(robots)
    for instance_name, value in initial_positions.items():
        if instance_name in robot_names or not _as_strings(value):
            continue
        asset_type = _asset_type(instance_name)
        object_node = Object(instance_name, {"type": asset_type})
        graph.add_node(object_node)
        entity_map[instance_name] = asset_type
        location = _as_strings(value)[0]
        if location in robot_nodes:
            graph.add_edge(robot_nodes[location], object_node, "holds", "held by")
        elif location in furniture_by_type:
            graph.add_edge(furniture_by_type[location], object_node, "under", "on")
        else:
            raise ValueError(f"Unknown VIKI initial location: {location!r}")

    for agent in agents:
        for tool in agent.tools.values():
            tool.world_graph = graph  # type: ignore[attr-defined]

    return graph, agents, agent_map, entity_map, available_actions


def _planner_prompt(agent_ids: List[int]) -> str:
    action_lines = "\n".join(
        f"Agent_{agent_id}_Action: <one tool call>" for agent_id in agent_ids
    )
    return f"""{{system_tag}}You are the centralized PARTNR planner. Solve the task using the world model and tools below. Select exactly one high-level tool for each active agent per turn. Do not invent entity names: use the instance names shown in the world model. Continue from successful observations and finish only after the complete task is done.

Task: {{input}}

World model:
{{world_description}}

Active agents and tools:
{{agent_descriptions}}
Respond in exactly one of these forms:

Thought: <brief planning reason>
{action_lines}
Assigned!

or, only when the task is complete:

Final Thought: <brief completion check>
Done[]{{eot_tag}}{{assistant_tag}}"""


class PartnrOracleStatePlannerProvider:
    """Run PARTNR planning with privileged VIKI initial state and no answer plan."""

    def __init__(
        self,
        benchmark_root: Path,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        max_retries: int,
        max_steps: int,
        location_types: Optional[Set[str]] = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_steps = max_steps
        self.location_types = location_types or load_train_location_types(
            benchmark_root
        )
        self.metadata: Dict[int, Dict[str, Any]] = {}

    def _create_planner(self, graph: WorldGraph, agents: List[Agent]):
        llm_config = {
            "llm": {
                "_target_": "habitat_llm.evaluation.viki_partnr_planner.VikiEndpointLLM",
                "_partial_": True,
            },
            "base_url": self.base_url,
            "api_key": self.api_key,
            "max_retries": self.max_retries,
            "system_tag": "",
            "user_tag": "",
            "assistant_tag": "",
            "eot_tag": "",
            "generation_params": {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        plan_config = OmegaConf.create(
            {
                "replanning_threshold": self.max_steps,
                "planning_mode": "cot",
                "constrained_generation": False,
                "objects_response": False,
                "objects_response_include_states": False,
                "centralized": True,
                "llm": llm_config,
                "instruct": {
                    "prompt": _planner_prompt([agent.uid for agent in agents]),
                    "stopword": "Assigned!",
                    "end_expression": "Done[]",
                    "actions_parser": {
                        "_target_": "habitat_llm.llm.instruct.utils.actions_parser",
                        "_partial_": True,
                    },
                },
            }
        )
        environment: Any = SimpleNamespace(
            _single_agent_mode=len(agents) == 1,
            partial_obs=False,
            full_world_graph=graph,
            sim=SimpleNamespace(agents_mgr=SimpleNamespace(articulated_agents_iter=[])),
        )
        planner = CentralizedLLMPlanner(plan_config, environment)
        planner.agents = agents
        planner.reset()
        return planner

    def generate(self, sample: Dict[str, Any], index: int) -> str:
        from habitat_llm.evaluation.viki_bench import convert_partnr_trace_to_viki

        instruction, _, _ = _prompt_context(sample)
        (
            graph,
            agents,
            agent_map,
            entity_map,
            available_actions,
        ) = build_oracle_state_world_graph(sample, self.location_types)
        planner = self._create_planner(graph, agents)
        world_graphs = {agent.uid: graph for agent in agents}
        trace = []
        done = False
        planner_info = {}
        for _ in range(self.max_steps + 1):
            _, planner_info, done = planner.get_next_action(
                instruction, {}, world_graphs
            )
            if done:
                break
            trace.append(planner_info["high_level_actions"])
        plan = convert_partnr_trace_to_viki(
            trace, agent_map, entity_map, available_actions
        )
        self.metadata[index] = {
            "track": "oracle-state-planner-ablation",
            "planner_stopped": done,
            "termination_reason": (
                "model_done"
                if "Done[]" in planner_info.get("traces", {}).get(agents[0].uid, "")
                else "max_steps"
            ),
            "instruction": instruction,
            "agent_map": agent_map,
            "entity_map": entity_map,
            "partnr_actions": trace,
            "partnr_trace": planner_info.get("traces", {}).get(agents[0].uid, ""),
            "viki_plan": plan,
        }
        return (
            "<think>PARTNR oracle-state planner trace converted to VIKI actions.</think>"
            f"<answer>{plan!r}</answer>"
        )

    def get_metadata(self, index: int) -> Dict[str, Any]:
        return self.metadata[index]
