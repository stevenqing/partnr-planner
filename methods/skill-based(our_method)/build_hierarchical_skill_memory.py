#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill-Structured Hierarchical Memory Builder

This module implements the memory building stage of the Decentralized Skill-Structured
Hierarchical Memory framework for Multi-Agent Coordination.

Key components:
1. Skill extraction (individual and cooperation) - supports LLM-based extraction
2. Instance grounding with context
3. Hierarchical organization

Reference: /doc/our_method
"""

import csv
import gzip
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

# Try to import LLM skill extractor
try:
    from llm_skill_extractor import LLMSkillExtractor, FallbackSkillExtractor, create_skill_extractor
    HAS_LLM_EXTRACTOR = True
except ImportError:
    HAS_LLM_EXTRACTOR = False


@dataclass
class SkillInstance:
    """
    A skill instance i = (context, demo, e_src)

    context: Scenario-specific parameters (location, object, etc.)
    demo: Natural language demonstration
    e_src: Pointer to source episodic memory
    """
    context: Dict[str, Any]
    demo: str
    e_src: str  # episode_id reference
    success: bool = True


@dataclass
class Skill:
    """
    A skill s = (name, type, I)

    name: Natural language identifier (e.g., "navigate", "pickup")
    type: Either "individual" or "cooperation"
    I: Set of grounded instances
    """
    name: str
    skill_type: str  # "individual" or "cooperation"
    instances: List[SkillInstance] = field(default_factory=list)
    description: str = ""
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)


@dataclass
class CooperationSkill(Skill):
    """
    Cooperation skill s_coop = (name, cooperation, precond_joint, I)

    Extends Skill with joint preconditions and partner context
    """
    precond_joint: Dict[str, Any] = field(default_factory=dict)
    partner_context_pattern: Dict[str, Any] = field(default_factory=dict)
    trigger_conditions: List[str] = field(default_factory=list)


@dataclass
class AgentAction:
    """Represents a single agent action in the trajectory"""
    agent_id: int
    action_type: str
    action_args: List[str]
    observation: str
    success: bool
    step_number: int
    raw_action: str


@dataclass
class EnvironmentState:
    """Represents the environment state at a given step"""
    furniture: Dict[str, str]  # furniture_name -> room
    objects: Dict[str, Dict[str, Any]]  # object_name -> {location, states, distance, room}
    agent_states: Dict[int, Dict[str, Any]]  # agent_id -> state


@dataclass
class ObjectLocationTracker:
    """
    Tracks all object locations seen by each agent over time.

    For each step, maintains cumulative knowledge of object locations
    as observed by each agent.
    """
    # agent_id -> {object_name -> {location, room, last_seen_step}}
    seen_objects: Dict[int, Dict[str, Dict[str, Any]]] = field(default_factory=lambda: {0: {}, 1: {}})

    def update_from_observation(self, agent_id: int, objects: Dict[str, Dict], step: int):
        """Update seen objects from an observation"""
        if agent_id not in self.seen_objects:
            self.seen_objects[agent_id] = {}

        for obj_name, obj_info in objects.items():
            self.seen_objects[agent_id][obj_name] = {
                "location": obj_info.get("location", "unknown"),
                "room": obj_info.get("room", "unknown"),
                "last_seen_step": step,
                "distance": obj_info.get("distance", 0.0),
                "states": obj_info.get("states", {})
            }

    def get_all_seen_locations(self, agent_id: int) -> Dict[str, Dict[str, str]]:
        """Get all object locations seen by an agent"""
        return self.seen_objects.get(agent_id, {})

    def get_location_summary(self, agent_id: int) -> Dict[str, Any]:
        """Get a summary of locations for skill context"""
        seen = self.seen_objects.get(agent_id, {})

        # Group by room
        objects_by_room = {}
        objects_by_furniture = {}

        for obj_name, info in seen.items():
            room = info.get("room", "unknown")
            location = info.get("location", "unknown")

            if room not in objects_by_room:
                objects_by_room[room] = []
            objects_by_room[room].append(obj_name)

            if location not in objects_by_furniture:
                objects_by_furniture[location] = []
            objects_by_furniture[location].append(obj_name)

        return {
            "objects_by_room": objects_by_room,
            "objects_by_furniture": objects_by_furniture,
            "all_objects": {k: {"location": v["location"], "room": v["room"]}
                          for k, v in seen.items()}
        }


class HierarchicalSkillMemory:
    """
    Skill-structured hierarchical memory system

    L = L_ind ∪ L_coop
    - L_ind: Individual skills (single-agent competencies)
    - L_coop: Cooperation skills (partner-aware conditional policies)

    Supports both rule-based and LLM-based skill extraction.
    """

    def __init__(
        self,
        use_llm: bool = False,
        use_local: bool = True,
        model_path: str = "/home/a5l/shuqing.a5l/models/Llama-3.3-70B-Instruct",
        vllm_host: str = "localhost",
        vllm_port: int = 8000,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        patch_failed: bool = False
    ):
        """
        Initialize hierarchical skill memory

        Args:
            use_llm: Use LLM (Llama 3.3 70B) for skill extraction
            use_local: Use local model (True) or vLLM API (False)
            model_path: Path to local model
            vllm_host: vLLM server host (if use_local=False)
            vllm_port: vLLM server port (if use_local=False)
            model_name: Model name for vLLM
            patch_failed: Use LLM to patch/analyze failed episodes
        """
        self.L_ind: Dict[str, Skill] = {}  # Individual skills
        self.L_coop: Dict[str, CooperationSkill] = {}  # Cooperation skills
        self.episodic_memory: Dict[str, Dict] = {}  # Full episode data
        self.skill_embeddings: Dict[str, np.ndarray] = {}  # For retrieval
        self.failure_analyses: Dict[str, Dict] = {}  # Failure analyses for patched episodes

        # LLM-based extraction
        self.use_llm = use_llm
        self.use_local = use_local
        self.patch_failed = patch_failed
        self.llm_extractor = None
        if use_llm and HAS_LLM_EXTRACTOR:
            if use_local:
                print(f"Initializing local LLM skill extractor from {model_path}...")
            else:
                print(f"Initializing LLM skill extractor with vLLM API: {model_name}...")
            self.llm_extractor = create_skill_extractor(
                use_llm=True,
                use_local=use_local,
                model_path=model_path,
                vllm_host=vllm_host,
                vllm_port=vllm_port,
                model_name=model_name
            )
            print("LLM extractor initialized successfully")
        elif use_llm:
            print("Warning: LLM extractor not available, using rule-based extraction")

        # Skill templates for extraction
        self.action_to_skill_map = {
            "Navigate": "navigation",
            "Pick": "pickup",
            "Place": "placement",
            "Open": "open_furniture",
            "Close": "close_furniture",
            "Clean": "cleaning",
            "Fill": "filling",
            "Pour": "pouring",
            "PowerOn": "power_on",
            "PowerOff": "power_off",
            "Rearrange": "rearrangement",
            "Wait": "waiting",
            "Explore": "exploration",
        }

        # Cooperation patterns
        self.cooperation_patterns = {
            "sequential_handoff": {
                "description": "Agent passes task completion to partner",
                "trigger": "partner_waiting_near_target"
            },
            "complementary_work": {
                "description": "Agents work on different subtasks simultaneously",
                "trigger": "parallel_different_targets"
            },
            "assistance": {
                "description": "Agent assists partner with task",
                "trigger": "partner_struggling_with_object"
            },
            "division_of_labor": {
                "description": "Agents divide task based on capabilities",
                "trigger": "heterogeneous_skills"
            }
        }

    def parse_trace_file(self, trace_path: str) -> Tuple[str, EnvironmentState, List[AgentAction]]:
        """Parse a trace file into structured components"""
        with open(trace_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')

        # Extract task instruction
        task_instruction = ""
        furniture = {}
        objects = {}
        actions = []

        current_section = None
        step_number = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Task:"):
                task_instruction = line.replace("Task:", "").strip()
            elif line.startswith("Furniture:"):
                current_section = "furniture"
            elif line.startswith("Objects:"):
                current_section = "objects"
            elif line.startswith("Agent_"):
                current_section = "actions"

                # Parse agent action
                action = self._parse_action_line(line, step_number)
                if action:
                    actions.append(action)
                    if action.observation:
                        step_number += 1
            elif current_section == "furniture":
                # Parse furniture line: "Floor : floor_porch_1 in porch, ..."
                parts = line.split(" : ")
                if len(parts) == 2:
                    furn_type = parts[0].strip()
                    items = parts[1].split(", ")
                    for item in items:
                        match = re.match(r"(\w+) in (\w+)", item.strip())
                        if match:
                            furniture[match.group(1)] = match.group(2)
            elif current_section == "objects":
                # Parse object line
                obj_info = self._parse_object_line(line)
                if obj_info:
                    objects[obj_info["name"]] = obj_info

        env_state = EnvironmentState(
            furniture=furniture,
            objects=objects,
            agent_states={}
        )

        return task_instruction, env_state, actions

    def _parse_action_line(self, line: str, step_number: int) -> Optional[AgentAction]:
        """Parse an action line from trace"""
        # Agent_0_Action: Navigate[plant_container_0]
        # Agent_1_observation:Successful execution!

        if "_Action:" in line:
            match = re.match(r"Agent_(\d+)_Action:\s*(.+)", line)
            if match:
                agent_id = int(match.group(1))
                action_str = match.group(2).strip()

                # Parse action type and args
                action_match = re.match(r"(\w+)\[([^\]]*)\]", action_str)
                if action_match:
                    action_type = action_match.group(1)
                    args = [a.strip() for a in action_match.group(2).split(",")] if action_match.group(2) else []
                else:
                    action_type = action_str
                    args = []

                return AgentAction(
                    agent_id=agent_id,
                    action_type=action_type,
                    action_args=args,
                    observation="",
                    success=True,
                    step_number=step_number,
                    raw_action=action_str
                )
        elif "_observation:" in line:
            match = re.match(r"Agent_(\d+)_observation:\s*(.+)", line)
            if match:
                observation = match.group(2).strip()
                success = "Successful" in observation
                # This updates the last action for this agent
                return AgentAction(
                    agent_id=int(match.group(1)),
                    action_type="observation",
                    action_args=[],
                    observation=observation,
                    success=success,
                    step_number=step_number,
                    raw_action=line
                )
        return None

    def _parse_object_line(self, line: str) -> Optional[Dict]:
        """Parse an object description line"""
        # plant_container_0 is in/on table_9 and 4.41 meters away from the agent in living_room_1
        match = re.match(
            r"(\w+) is in/on (\w+) and ([\d.]+) meters away from the agent in (\w+)",
            line
        )
        if match:
            obj_name = match.group(1)
            location = match.group(2)
            distance = float(match.group(3))
            room = match.group(4)

            # Parse states
            states = {}
            state_match = re.search(r"It has the following states: (.+)", line)
            if state_match:
                state_str = state_match.group(1)
                for state in state_str.split(", "):
                    parts = state.split(": ")
                    if len(parts) == 2:
                        states[parts[0].strip()] = parts[1].strip() == "True"

            return {
                "name": obj_name,
                "location": location,
                "distance": distance,
                "room": room,
                "states": states
            }
        return None

    def extract_individual_skills(
        self,
        actions: List[AgentAction],
        env_state: EnvironmentState,
        agent_id: int
    ) -> List[Skill]:
        """
        Extract individual skills (L_ind) from agent actions

        Individual skills capture single-agent competencies that depend only on
        the agent's own state and environment.
        """
        skills = []
        agent_actions = [a for a in actions if a.agent_id == agent_id and a.action_type != "observation"]

        # Group consecutive related actions into skill sequences
        skill_sequences = self._identify_skill_sequences(agent_actions)

        for seq in skill_sequences:
            skill_name = self._determine_skill_name(seq)

            # Get objects involved in this skill sequence
            involved_objects = list(set(
                arg for action in seq
                for arg in action.action_args
                if arg and not arg.lower() in ["none", "on", "within", "next_to"]
            ))

            # Extract detailed location info for each involved object
            object_locations = {}
            rooms_involved = set()
            furniture_involved = set()

            for obj in involved_objects:
                if obj in env_state.objects:
                    obj_info = env_state.objects[obj]
                    loc = obj_info.get("location", "unknown")
                    room = obj_info.get("room", "unknown")
                    object_locations[obj] = {
                        "location": loc,
                        "room": room
                    }
                    if loc and loc != "unknown":
                        furniture_involved.add(loc)
                    if room and room != "unknown":
                        rooms_involved.add(room)
                # Also check if the arg is a furniture/location (for Navigate actions)
                elif obj in env_state.furniture:
                    room = env_state.furniture[obj]
                    furniture_involved.add(obj)
                    if room:
                        rooms_involved.add(room)

            # Create context with enhanced location information
            context = {
                "objects": involved_objects,
                "locations": list(furniture_involved),  # Furniture/receptacles involved
                "rooms": list(rooms_involved),  # Rooms involved
                "object_locations": object_locations,  # Detailed: {obj: {location, room}}
                # Use raw_action to preserve full format: "Navigate[object_0]", "Pick[item_1]", etc.
                "action_sequence": [a.raw_action for a in seq if a.raw_action],
            }

            # Create demonstration text
            demo = self._generate_skill_demo(seq, env_state)

            skill = Skill(
                name=skill_name,
                skill_type="individual",
                instances=[SkillInstance(
                    context=context,
                    demo=demo,
                    e_src=f"agent_{agent_id}",
                    success=all(a.success for a in seq if a.observation)
                )],
                description=f"Individual skill for {skill_name}",
                preconditions=self._extract_preconditions(seq, env_state),
                effects=self._extract_effects(seq, env_state)
            )
            skills.append(skill)

        return skills

    def extract_cooperation_skills(
        self,
        all_actions: List[AgentAction],
        env_state: EnvironmentState,
        task_instruction: str
    ) -> List[CooperationSkill]:
        """
        Extract cooperation skills (L_coop) from multi-agent trajectory

        Cooperation skills encode conditional policies triggered by observed
        partner behavior.
        """
        cooperation_skills = []

        # Separate actions by agent
        agent_0_actions = [a for a in all_actions if a.agent_id == 0 and a.action_type != "observation"]
        agent_1_actions = [a for a in all_actions if a.agent_id == 1 and a.action_type != "observation"]

        # Identify coordination patterns
        patterns = self._identify_coordination_patterns(agent_0_actions, agent_1_actions, env_state)

        for pattern in patterns:
            coop_skill = CooperationSkill(
                name=pattern["name"],
                skill_type="cooperation",
                instances=[SkillInstance(
                    context=pattern["context"],
                    demo=pattern["demo"],
                    e_src="multi_agent",
                    success=pattern["success"]
                )],
                description=pattern["description"],
                precond_joint=pattern["precond_joint"],
                partner_context_pattern=pattern["partner_context"],
                trigger_conditions=pattern["triggers"]
            )
            cooperation_skills.append(coop_skill)

        return cooperation_skills

    def _identify_skill_sequences(self, actions: List[AgentAction]) -> List[List[AgentAction]]:
        """Group actions into logical skill sequences"""
        sequences = []
        current_seq = []

        for action in actions:
            if action.action_type == "Wait":
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = []
            else:
                current_seq.append(action)

        if current_seq:
            sequences.append(current_seq)

        # Further split long sequences by objective
        refined_sequences = []
        for seq in sequences:
            if len(seq) > 1:
                # Split by Place/Rearrange actions (completion points)
                sub_seq = []
                for action in seq:
                    sub_seq.append(action)
                    if action.action_type in ["Place", "Rearrange", "Clean", "Fill", "PowerOn", "PowerOff"]:
                        refined_sequences.append(sub_seq)
                        sub_seq = []
                if sub_seq:
                    refined_sequences.append(sub_seq)
            else:
                refined_sequences.append(seq)

        return refined_sequences

    def _determine_skill_name(self, actions: List[AgentAction]) -> str:
        """Determine the primary skill name from action sequence"""
        if not actions:
            return "unknown"

        # Priority order for skill naming
        priority_actions = ["Rearrange", "Place", "Clean", "Fill", "PowerOn", "PowerOff", "Pick", "Navigate"]

        for priority in priority_actions:
            for action in actions:
                if action.action_type == priority:
                    return self.action_to_skill_map.get(priority, priority.lower())

        return self.action_to_skill_map.get(actions[0].action_type, actions[0].action_type.lower())

    def _generate_skill_demo(self, actions: List[AgentAction], env_state: EnvironmentState) -> str:
        """Generate natural language demonstration from action sequence"""
        demo_parts = []

        for action in actions:
            if action.action_type == "Navigate":
                target = action.action_args[0] if action.action_args else "target"
                demo_parts.append(f"Navigate to {target}")
            elif action.action_type == "Pick":
                obj = action.action_args[0] if action.action_args else "object"
                demo_parts.append(f"Pick up {obj}")
            elif action.action_type == "Place":
                obj = action.action_args[0] if len(action.action_args) > 0 else "object"
                loc = action.action_args[2] if len(action.action_args) > 2 else "location"
                demo_parts.append(f"Place {obj} on {loc}")
            elif action.action_type == "Open":
                furn = action.action_args[0] if action.action_args else "furniture"
                demo_parts.append(f"Open {furn}")
            elif action.action_type == "Close":
                furn = action.action_args[0] if action.action_args else "furniture"
                demo_parts.append(f"Close {furn}")
            elif action.action_type == "Clean":
                obj = action.action_args[0] if action.action_args else "object"
                demo_parts.append(f"Clean {obj}")
            elif action.action_type == "Fill":
                obj = action.action_args[0] if action.action_args else "container"
                demo_parts.append(f"Fill {obj} with water")
            elif action.action_type == "PowerOn":
                obj = action.action_args[0] if action.action_args else "appliance"
                demo_parts.append(f"Turn on {obj}")
            elif action.action_type == "PowerOff":
                obj = action.action_args[0] if action.action_args else "appliance"
                demo_parts.append(f"Turn off {obj}")

        return " -> ".join(demo_parts)

    def _extract_preconditions(self, actions: List[AgentAction], env_state: EnvironmentState) -> List[str]:
        """Extract preconditions for a skill sequence"""
        preconditions = []

        for action in actions:
            if action.action_type == "Pick":
                obj = action.action_args[0] if action.action_args else None
                if obj and obj in env_state.objects:
                    loc = env_state.objects[obj].get("location", "")
                    preconditions.append(f"agent_near_{loc}")
                    preconditions.append(f"{obj}_reachable")
                    preconditions.append("agent_hands_empty")
            elif action.action_type == "Place":
                preconditions.append("agent_holding_object")
            elif action.action_type == "Fill":
                preconditions.append("near_water_source")
            elif action.action_type == "Clean":
                preconditions.append("agent_can_clean")

        return list(set(preconditions))

    def _extract_effects(self, actions: List[AgentAction], env_state: EnvironmentState) -> List[str]:
        """Extract effects of a skill sequence"""
        effects = []

        for action in actions:
            if action.action_type == "Pick":
                obj = action.action_args[0] if action.action_args else "object"
                effects.append(f"holding_{obj}")
            elif action.action_type == "Place":
                obj = action.action_args[0] if len(action.action_args) > 0 else "object"
                loc = action.action_args[2] if len(action.action_args) > 2 else "location"
                effects.append(f"{obj}_on_{loc}")
                effects.append("agent_hands_empty")
            elif action.action_type == "Clean":
                obj = action.action_args[0] if action.action_args else "object"
                effects.append(f"{obj}_is_clean")
            elif action.action_type == "Fill":
                obj = action.action_args[0] if action.action_args else "container"
                effects.append(f"{obj}_is_filled")
            elif action.action_type == "PowerOn":
                obj = action.action_args[0] if action.action_args else "appliance"
                effects.append(f"{obj}_is_on")

        return list(set(effects))

    def _identify_coordination_patterns(
        self,
        agent_0_actions: List[AgentAction],
        agent_1_actions: List[AgentAction],
        env_state: EnvironmentState
    ) -> List[Dict]:
        """Identify coordination patterns between agents"""
        patterns = []

        # Extract full action sequences for both agents (raw format with brackets)
        agent_0_raw_actions = [a.raw_action for a in agent_0_actions if a.raw_action]
        agent_1_raw_actions = [a.raw_action for a in agent_1_actions if a.raw_action]

        # Check for division of labor (heterogeneous skills)
        agent_0_types = set(a.action_type for a in agent_0_actions)
        agent_1_types = set(a.action_type for a in agent_1_actions)

        unique_to_0 = agent_0_types - agent_1_types
        unique_to_1 = agent_1_types - agent_0_types

        if unique_to_0 or unique_to_1:
            patterns.append({
                "name": "division_of_labor",
                "description": "Agents divide tasks based on heterogeneous capabilities",
                "context": {
                    "agent_0_unique": list(unique_to_0),
                    "agent_1_unique": list(unique_to_1),
                    "shared": list(agent_0_types & agent_1_types),
                    # Include full action sequences for both agents
                    "action_sequence": agent_0_raw_actions[:10] + agent_1_raw_actions[:10],
                    "agent_0_actions": agent_0_raw_actions[:10],
                    "agent_1_actions": agent_1_raw_actions[:10],
                },
                "demo": f"Agent 0 handles {unique_to_0 or 'shared tasks'}, Agent 1 handles {unique_to_1 or 'shared tasks'}",
                "success": True,
                "precond_joint": {"requires_heterogeneous_skills": True},
                "partner_context": {"skill_difference": True},
                "triggers": ["heterogeneous_task_requirements"]
            })

        # Check for sequential handoff
        handoff_detected = self._detect_sequential_handoff(agent_0_actions, agent_1_actions, env_state)
        if handoff_detected:
            patterns.append(handoff_detected)

        # Check for complementary work
        complementary = self._detect_complementary_work(agent_0_actions, agent_1_actions, env_state)
        if complementary:
            patterns.append(complementary)

        # Check for waiting/synchronization
        sync_pattern = self._detect_synchronization(agent_0_actions, agent_1_actions)
        if sync_pattern:
            patterns.append(sync_pattern)

        # Always add a task-level cooperation pattern with full action sequences
        # This ensures we have concrete action examples for any task
        task_pattern = self._create_task_cooperation_pattern(
            agent_0_actions, agent_1_actions, env_state
        )
        if task_pattern:
            patterns.append(task_pattern)

        return patterns

    def _create_task_cooperation_pattern(
        self,
        agent_0_actions: List[AgentAction],
        agent_1_actions: List[AgentAction],
        env_state: EnvironmentState
    ) -> Optional[Dict]:
        """Create a general task cooperation pattern with full action sequences"""
        if not agent_0_actions and not agent_1_actions:
            return None

        # Extract full action sequences
        agent_0_raw = [a.raw_action for a in agent_0_actions if a.raw_action]
        agent_1_raw = [a.raw_action for a in agent_1_actions if a.raw_action]

        # Get all objects involved and their locations
        all_objects = set()
        object_locations = {}
        rooms_involved = set()
        furniture_involved = set()

        for action in agent_0_actions + agent_1_actions:
            for arg in action.action_args:
                if arg and arg.lower() not in ["none", "on", "within", "next_to"]:
                    all_objects.add(arg)
                    # Get location info for objects
                    if arg in env_state.objects:
                        obj_info = env_state.objects[arg]
                        loc = obj_info.get("location", "unknown")
                        room = obj_info.get("room", "unknown")
                        object_locations[arg] = {"location": loc, "room": room}
                        if loc and loc != "unknown":
                            furniture_involved.add(loc)
                        if room and room != "unknown":
                            rooms_involved.add(room)
                    # Check if arg is furniture (for Navigate actions)
                    elif arg in env_state.furniture:
                        room = env_state.furniture[arg]
                        furniture_involved.add(arg)
                        if room:
                            rooms_involved.add(room)

        # Determine task type based on dominant actions
        action_types = [a.action_type for a in agent_0_actions + agent_1_actions]
        if "Place" in action_types or "Rearrange" in action_types:
            task_name = "Object Rearrangement"
        elif "Clean" in action_types:
            task_name = "Cleaning Task"
        elif "Fill" in action_types:
            task_name = "Filling Task"
        else:
            task_name = "Multi-Agent Task"

        return {
            "name": task_name,
            "description": f"Multi-agent cooperation for {task_name.lower()}",
            "context": {
                "objects": list(all_objects)[:10],
                "locations": list(furniture_involved),  # Furniture/receptacles involved
                "rooms": list(rooms_involved),  # Rooms involved
                "object_locations": object_locations,  # Detailed: {obj: {location, room}}
                # Full action sequences - this is key for the LLM to copy
                "action_sequence": agent_0_raw[:15] + agent_1_raw[:15],
                "agent_0_actions": agent_0_raw[:15],
                "agent_1_actions": agent_1_raw[:15],
            },
            "demo": self._generate_cooperation_demo(agent_0_actions, agent_1_actions),
            "success": True,
            "precond_joint": {"multi_agent_task": True},
            "partner_context": {"collaborative_execution": True},
            "triggers": ["task_requires_cooperation"]
        }

    def _generate_cooperation_demo(
        self,
        agent_0_actions: List[AgentAction],
        agent_1_actions: List[AgentAction]
    ) -> str:
        """Generate a demonstration text that shows both agents' actions"""
        demo_parts = []

        # Get key actions from each agent
        agent_0_key = []
        agent_1_key = []

        for action in agent_0_actions[:5]:
            if action.action_type in ["Pick", "Place", "Navigate"]:
                agent_0_key.append(action.raw_action)

        for action in agent_1_actions[:5]:
            if action.action_type in ["Pick", "Place", "Navigate"]:
                agent_1_key.append(action.raw_action)

        if agent_0_key:
            demo_parts.append(f"Agent 0: {' -> '.join(agent_0_key[:3])}")
        if agent_1_key:
            demo_parts.append(f"Agent 1: {' -> '.join(agent_1_key[:3])}")

        return " | ".join(demo_parts) if demo_parts else "Multi-agent cooperation"

    def _detect_sequential_handoff(
        self,
        agent_0_actions: List[AgentAction],
        agent_1_actions: List[AgentAction],
        env_state: EnvironmentState
    ) -> Optional[Dict]:
        """Detect sequential handoff patterns (one agent sets up, other completes)"""
        # Look for patterns where agent 0 places something and agent 1 acts on it
        placed_objects = set()
        for action in agent_0_actions:
            if action.action_type == "Place" and len(action.action_args) > 0:
                placed_objects.add(action.action_args[0])

        acted_on = set()
        for action in agent_1_actions:
            if action.action_type in ["Clean", "Fill", "PowerOn"] and len(action.action_args) > 0:
                acted_on.add(action.action_args[0])

        handoff_objects = placed_objects & acted_on
        if handoff_objects:
            # Extract full action sequences
            agent_0_raw = [a.raw_action for a in agent_0_actions if a.raw_action]
            agent_1_raw = [a.raw_action for a in agent_1_actions if a.raw_action]
            return {
                "name": "sequential_handoff",
                "description": "Agent 0 positions objects, Agent 1 performs operations",
                "context": {
                    "handoff_objects": list(handoff_objects),
                    "setup_agent": 0,
                    "completion_agent": 1,
                    # Include full action sequences
                    "action_sequence": agent_0_raw[:10] + agent_1_raw[:10],
                    "agent_0_actions": agent_0_raw[:10],
                    "agent_1_actions": agent_1_raw[:10],
                },
                "demo": f"Agent 0 places {handoff_objects}, Agent 1 performs operations on them",
                "success": True,
                "precond_joint": {"object_positioned": True},
                "partner_context": {"partner_placed_object": True},
                "triggers": ["object_moved_to_target_location"]
            }
        return None

    def _detect_complementary_work(
        self,
        agent_0_actions: List[AgentAction],
        agent_1_actions: List[AgentAction],
        env_state: EnvironmentState
    ) -> Optional[Dict]:
        """Detect complementary work patterns (agents working on different subtasks)"""
        # Check if agents work on different objects simultaneously
        agent_0_objects = set()
        agent_1_objects = set()

        for action in agent_0_actions:
            if action.action_args:
                agent_0_objects.add(action.action_args[0])

        for action in agent_1_actions:
            if action.action_args:
                agent_1_objects.add(action.action_args[0])

        unique_to_0 = agent_0_objects - agent_1_objects
        unique_to_1 = agent_1_objects - agent_0_objects

        if unique_to_0 and unique_to_1:
            # Extract full action sequences
            agent_0_raw = [a.raw_action for a in agent_0_actions if a.raw_action]
            agent_1_raw = [a.raw_action for a in agent_1_actions if a.raw_action]
            return {
                "name": "complementary_work",
                "description": "Agents work on different objects simultaneously",
                "context": {
                    "agent_0_objects": list(unique_to_0),
                    "agent_1_objects": list(unique_to_1),
                    # Include full action sequences
                    "action_sequence": agent_0_raw[:10] + agent_1_raw[:10],
                    "agent_0_actions": agent_0_raw[:10],
                    "agent_1_actions": agent_1_raw[:10],
                },
                "demo": f"Agent 0 handles {unique_to_0}, Agent 1 handles {unique_to_1}",
                "success": True,
                "precond_joint": {"multiple_subtasks": True},
                "partner_context": {"working_on_different_objects": True},
                "triggers": ["task_decomposable_into_subtasks"]
            }
        return None

    def _detect_synchronization(
        self,
        agent_0_actions: List[AgentAction],
        agent_1_actions: List[AgentAction]
    ) -> Optional[Dict]:
        """Detect synchronization/waiting patterns"""
        agent_0_waits = sum(1 for a in agent_0_actions if a.action_type == "Wait")
        agent_1_waits = sum(1 for a in agent_1_actions if a.action_type == "Wait")

        total_actions = len(agent_0_actions) + len(agent_1_actions)
        total_waits = agent_0_waits + agent_1_waits

        if total_waits > 0 and total_actions > 0:
            wait_ratio = total_waits / total_actions
            if wait_ratio > 0.1:  # Significant waiting
                # Extract full action sequences
                agent_0_raw = [a.raw_action for a in agent_0_actions if a.raw_action]
                agent_1_raw = [a.raw_action for a in agent_1_actions if a.raw_action]
                return {
                    "name": "synchronization",
                    "description": "Agents synchronize through waiting",
                    "context": {
                        "agent_0_waits": agent_0_waits,
                        "agent_1_waits": agent_1_waits,
                        "wait_ratio": wait_ratio,
                        # Include full action sequences
                        "action_sequence": agent_0_raw[:10] + agent_1_raw[:10],
                        "agent_0_actions": agent_0_raw[:10],
                        "agent_1_actions": agent_1_raw[:10],
                    },
                    "demo": f"Agents synchronize with {total_waits} wait actions",
                    "success": True,
                    "precond_joint": {"requires_synchronization": True},
                    "partner_context": {"partner_busy": True},
                    "triggers": ["partner_action_in_progress"]
                }
        return None

    def _convert_llm_skills_to_internal(self, llm_skills, agent_id: int) -> List[Skill]:
        """Convert LLM-extracted skills to internal Skill format"""
        skills = []
        for llm_skill in llm_skills:
            skill = Skill(
                name=llm_skill.name,
                skill_type="individual",
                instances=[SkillInstance(
                    context={
                        "objects": llm_skill.objects_involved,
                        "locations": llm_skill.locations_involved,
                        "action_sequence": llm_skill.action_sequence
                    },
                    demo=llm_skill.demo,
                    e_src=f"agent_{agent_id}_llm",
                    success=True
                )],
                description=llm_skill.description,
                preconditions=llm_skill.preconditions,
                effects=llm_skill.effects
            )
            skills.append(skill)
        return skills

    def _convert_llm_coop_skills_to_internal(self, llm_skills, llm_patterns) -> List[CooperationSkill]:
        """Convert LLM-extracted cooperation skills to internal format"""
        coop_skills = []

        # Convert skills
        for llm_skill in llm_skills:
            skill = CooperationSkill(
                name=llm_skill.name,
                skill_type="cooperation",
                instances=[SkillInstance(
                    context={
                        "objects": llm_skill.objects_involved,
                        "action_sequence": llm_skill.action_sequence
                    },
                    demo=llm_skill.demo,
                    e_src="multi_agent_llm",
                    success=True
                )],
                description=llm_skill.description,
                preconditions=llm_skill.preconditions,
                effects=llm_skill.effects,
                precond_joint={},
                partner_context_pattern={},
                trigger_conditions=[]
            )
            coop_skills.append(skill)

        # Convert patterns to skills
        for pattern in llm_patterns:
            skill = CooperationSkill(
                name=pattern.pattern_name,
                skill_type="cooperation",
                instances=[SkillInstance(
                    context={
                        "agent_0_role": pattern.agent_0_role,
                        "agent_1_role": pattern.agent_1_role,
                        "coordination_mechanism": pattern.coordination_mechanism
                    },
                    demo=pattern.description,
                    e_src="multi_agent_llm_pattern",
                    success=True
                )],
                description=pattern.description,
                preconditions=[],
                effects=[],
                precond_joint={"trigger": pattern.trigger_condition},
                partner_context_pattern=pattern.tom_reasoning,
                trigger_conditions=[pattern.trigger_condition]
            )
            coop_skills.append(skill)

        return coop_skills

    def add_episode_to_memory(
        self,
        episode_id: str,
        trace_paths: Dict[int, str],
        episode_info: Dict
    ):
        """Add an episode to the hierarchical memory"""
        # Parse trace for agent 0
        if 0 in trace_paths and os.path.exists(trace_paths[0]):
            task, env_state, actions = self.parse_trace_file(trace_paths[0])

            # Read trace content for LLM extraction
            with open(trace_paths[0], 'r', encoding='utf-8') as f:
                trace_content = f.read()

            # Also parse agent 1 trace if available
            all_actions = actions.copy()
            if 1 in trace_paths and os.path.exists(trace_paths[1]):
                _, _, agent_1_actions = self.parse_trace_file(trace_paths[1])
                # Merge actions (they're already interleaved in trace)
                # Just use agent 0's trace which contains both

            # Check if this is a failed episode
            is_failed = episode_info.get("success", 0) < 1.0
            task_percent_complete = episode_info.get("task_percent_complete", 0)

            # Handle failed episodes with LLM patching
            if is_failed and self.patch_failed and self.use_llm and self.llm_extractor:
                # Use LLM to analyze failure and extract patched skills
                partial_skills, patched_skills, failure_analysis = \
                    self.llm_extractor.patch_failed_episode(
                        trace_content, task, task_percent_complete
                    )

                # Add partial skills (successfully executed parts before failure)
                for llm_skill in partial_skills:
                    skill = Skill(
                        name=llm_skill.name,
                        skill_type="individual",
                        instances=[SkillInstance(
                            context={
                                "objects": llm_skill.objects_involved,
                                "locations": llm_skill.locations_involved,
                                "action_sequence": llm_skill.action_sequence,
                                "from_failed_episode": True
                            },
                            demo=llm_skill.demo,
                            e_src=f"episode_{episode_id}_partial",
                            success=True  # Partial skills are the successful parts
                        )],
                        description=llm_skill.description,
                        preconditions=llm_skill.preconditions,
                        effects=llm_skill.effects
                    )
                    skill_key = f"{skill.name}_{skill.skill_type}"
                    if skill_key not in self.L_ind:
                        self.L_ind[skill_key] = skill
                    else:
                        self.L_ind[skill_key].instances.extend(skill.instances)

                # Add patched skills (corrected versions)
                for llm_skill in patched_skills:
                    coop_skill = CooperationSkill(
                        name=llm_skill.name,
                        skill_type="cooperation",
                        instances=[SkillInstance(
                            context={
                                "objects": llm_skill.objects_involved,
                                "action_sequence": llm_skill.action_sequence,
                                "patched": True,
                                "original_episode": episode_id
                            },
                            demo=llm_skill.demo,
                            e_src=f"episode_{episode_id}_patched",
                            success=True  # Patched skills are corrected versions
                        )],
                        description=llm_skill.description,
                        preconditions=llm_skill.preconditions,
                        effects=llm_skill.effects,
                        precond_joint={},
                        partner_context_pattern={},
                        trigger_conditions=[]
                    )
                    skill_key = f"{coop_skill.name}_cooperation_patched"
                    if skill_key not in self.L_coop:
                        self.L_coop[skill_key] = coop_skill
                    else:
                        self.L_coop[skill_key].instances.extend(coop_skill.instances)

                # Store failure analysis
                if failure_analysis:
                    self.failure_analyses[episode_id] = failure_analysis

            else:
                # Standard extraction (for successful episodes or when patching is disabled)
                # Extract individual skills for each agent
                for agent_id in [0, 1]:
                    if self.use_llm and self.llm_extractor:
                        # Use LLM-based extraction
                        llm_skills = self.llm_extractor.extract_individual_skills(
                            trace_content, agent_id, task
                        )
                        agent_skills = self._convert_llm_skills_to_internal(llm_skills, agent_id)
                    else:
                        # Use rule-based extraction
                        agent_skills = self.extract_individual_skills(actions, env_state, agent_id)

                    for skill in agent_skills:
                        skill_key = f"{skill.name}_{skill.skill_type}"
                        if skill_key not in self.L_ind:
                            self.L_ind[skill_key] = skill
                        else:
                            # Merge instances
                            self.L_ind[skill_key].instances.extend(skill.instances)

                # Extract cooperation skills
                if self.use_llm and self.llm_extractor:
                    # Use LLM-based cooperation extraction
                    llm_coop_skills, llm_patterns = self.llm_extractor.extract_cooperation_skills(
                        trace_content, task
                    )
                    coop_skills = self._convert_llm_coop_skills_to_internal(llm_coop_skills, llm_patterns)
                else:
                    # Use rule-based extraction
                    coop_skills = self.extract_cooperation_skills(actions, env_state, task)

                for skill in coop_skills:
                    skill_key = f"{skill.name}_cooperation"
                    if skill_key not in self.L_coop:
                        self.L_coop[skill_key] = skill
                    else:
                        self.L_coop[skill_key].instances.extend(skill.instances)

            # Store episodic memory
            self.episodic_memory[episode_id] = {
                "task": task,
                "env_state": asdict(env_state) if hasattr(env_state, '__dict__') else {
                    "furniture": env_state.furniture,
                    "objects": env_state.objects,
                    "agent_states": env_state.agent_states
                },
                "actions": [asdict(a) for a in actions],
                "success": episode_info.get("success", 0),
                "task_percent_complete": task_percent_complete,
                "patched": is_failed and self.patch_failed
            }

    def save_memory(self, output_path: str):
        """Save the hierarchical memory to disk"""
        os.makedirs(output_path, exist_ok=True)

        # Save individual skills
        ind_skills_data = {}
        for key, skill in self.L_ind.items():
            ind_skills_data[key] = {
                "name": skill.name,
                "skill_type": skill.skill_type,
                "description": skill.description,
                "preconditions": skill.preconditions,
                "effects": skill.effects,
                "instances": [asdict(inst) for inst in skill.instances],
                "instance_count": len(skill.instances)
            }

        with gzip.open(os.path.join(output_path, "L_ind_skills.json.gz"), 'wt', encoding='utf-8') as f:
            json.dump(ind_skills_data, f, indent=2)

        # Save cooperation skills
        coop_skills_data = {}
        for key, skill in self.L_coop.items():
            coop_skills_data[key] = {
                "name": skill.name,
                "skill_type": skill.skill_type,
                "description": skill.description,
                "preconditions": skill.preconditions,
                "effects": skill.effects,
                "precond_joint": skill.precond_joint,
                "partner_context_pattern": skill.partner_context_pattern,
                "trigger_conditions": skill.trigger_conditions,
                "instances": [asdict(inst) for inst in skill.instances],
                "instance_count": len(skill.instances)
            }

        with gzip.open(os.path.join(output_path, "L_coop_skills.json.gz"), 'wt', encoding='utf-8') as f:
            json.dump(coop_skills_data, f, indent=2)

        # Save episodic memory
        with gzip.open(os.path.join(output_path, "episodic_memory.json.gz"), 'wt', encoding='utf-8') as f:
            json.dump(self.episodic_memory, f, indent=2)

        # Save failure analyses if any
        if self.failure_analyses:
            with gzip.open(os.path.join(output_path, "failure_analyses.json.gz"), 'wt', encoding='utf-8') as f:
                json.dump(self.failure_analyses, f, indent=2)

        # Save summary
        patched_episodes = sum(1 for ep in self.episodic_memory.values() if ep.get("patched", False))
        summary = {
            "total_individual_skills": len(self.L_ind),
            "total_cooperation_skills": len(self.L_coop),
            "total_episodes": len(self.episodic_memory),
            "patched_episodes": patched_episodes,
            "failure_analyses_count": len(self.failure_analyses),
            "individual_skill_names": list(self.L_ind.keys()),
            "cooperation_skill_names": list(self.L_coop.keys()),
            "created_at": datetime.now().isoformat()
        }

        with open(os.path.join(output_path, "memory_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"Memory saved to {output_path}")
        print(f"  - Individual skills: {len(self.L_ind)}")
        print(f"  - Cooperation skills: {len(self.L_coop)}")
        print(f"  - Episodes: {len(self.episodic_memory)}")
        if patched_episodes > 0:
            print(f"  - Patched episodes: {patched_episodes}")
            print(f"  - Failure analyses: {len(self.failure_analyses)}")


def build_memory_from_heuristic_results(
    results_dir: str,
    output_dir: str,
    filter_successful: bool = True,
    use_llm: bool = False,
    use_local: bool = True,
    model_path: str = "/home/a5l/shuqing.a5l/models/Llama-3.3-70B-Instruct",
    vllm_host: str = "localhost",
    vllm_port: int = 8000,
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    patch_failed: bool = False
):
    """
    Build hierarchical skill memory from heuristic trajectory results

    Args:
        results_dir: Path to heuristic results directory
        output_dir: Path to save the memory
        filter_successful: Only include successful episodes (task_state_success >= 1.0)
        use_llm: Use Llama 3.3 70B for skill extraction
        use_local: Use local model (True) or vLLM API (False)
        model_path: Path to local model
        vllm_host: vLLM server host (if use_local=False)
        vllm_port: vLLM server port (if use_local=False)
        model_name: Model name for vLLM
        patch_failed: Use LLM to patch/analyze failed episodes (requires use_llm and not filter_successful)
    """
    print("=" * 80)
    print("Building Hierarchical Skill Memory from Heuristic Results")
    if use_llm:
        if use_local:
            print(f"Using local LLM: {model_path}")
        else:
            print(f"Using vLLM API: {model_name}")
    else:
        print("Using rule-based extraction")
    if not filter_successful:
        print("Including failed episodes")
        if patch_failed and use_llm:
            print("LLM patching enabled for failed episodes")
    print("=" * 80)

    memory = HierarchicalSkillMemory(
        use_llm=use_llm,
        use_local=use_local,
        model_path=model_path,
        vllm_host=vllm_host,
        vllm_port=vllm_port,
        model_name=model_name,
        patch_failed=patch_failed
    )

    # Find episode result log
    episode_log_path = os.path.join(results_dir, "episode_result_log.csv")
    if not os.path.exists(episode_log_path):
        print(f"Error: Episode log not found at {episode_log_path}")
        return

    # Read episode results
    episodes = {}
    with open(episode_log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode_id = row.get("episode_id", "").strip()
            success = float(row.get("task_state_success", 0))

            if filter_successful and success < 1.0:
                continue

            episodes[episode_id] = {
                "instruction": row.get("instruction", ""),
                "success": success,
                "task_percent_complete": float(row.get("task_percent_complete", 0)),
                "runtime": float(row.get("runtime", 0)),
            }

    print(f"Found {len(episodes)} {'successful ' if filter_successful else ''}episodes")

    # Find trace directories
    traces_base = None
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and item.endswith(".gz"):
            traces_dir = os.path.join(item_path, "traces")
            if os.path.exists(traces_dir):
                traces_base = traces_dir
                break

    if not traces_base:
        print(f"Error: Traces directory not found in {results_dir}")
        return

    print(f"Using traces from: {traces_base}")

    # Process each episode with progress bar
    processed = 0
    for episode_id, info in tqdm(episodes.items(), desc="Processing episodes", unit="ep"):
        trace_paths = {}
        for agent_id in [0, 1]:
            trace_pattern = f"trace-episode_{episode_id}_0-{agent_id}.txt"
            trace_path = os.path.join(traces_base, str(agent_id), trace_pattern)
            if os.path.exists(trace_path):
                trace_paths[agent_id] = trace_path

        if trace_paths:
            memory.add_episode_to_memory(episode_id, trace_paths, info)
            processed += 1

    print(f"Processed {processed} episodes with traces")

    # Save memory
    memory.save_memory(output_dir)

    return memory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build hierarchical skill memory")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner/outputs/habitat_llm/2025-12-30_21-10-23-heterogeneous+rerange.json/results",
        help="Path to heuristic results directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/a5l/shuqing.a5l/partnr-planner/data/hierarchical_skill_memory",
        help="Path to save the memory"
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed episodes in memory"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Llama 3.3 70B for skill extraction"
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        default=True,
        help="Use local model (default: True)"
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use vLLM API instead of local model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/a5l/shuqing.a5l/models/Llama-3.3-70B-Instruct",
        help="Path to local model"
    )
    parser.add_argument(
        "--vllm-host",
        type=str,
        default="localhost",
        help="vLLM server host (if using API)"
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8000,
        help="vLLM server port (if using API)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name for vLLM API"
    )
    parser.add_argument(
        "--patch-failed",
        action="store_true",
        help="Use LLM to patch/analyze failed episodes (requires --use-llm and --include-failed)"
    )

    args = parser.parse_args()

    # Determine if using local or API
    use_local = not args.use_api

    build_memory_from_heuristic_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        filter_successful=not args.include_failed,
        use_llm=args.use_llm,
        use_local=use_local,
        model_path=args.model_path,
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
        model_name=args.model_name,
        patch_failed=args.patch_failed
    )
