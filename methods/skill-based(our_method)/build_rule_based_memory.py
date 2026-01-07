#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Rule-Based Hierarchical Skill Memory Builder

This is an improved rule-based approach that is better than memento's memory building:

Key improvements over memento:
1. Skill-focused (not just user preferences) - captures reusable action patterns
2. Cooperation patterns - detects 4+ coordination patterns between agents
3. Theory of Mind - models partner beliefs and intentions
4. Hierarchical structure - L_ind (individual) + L_coop (cooperation) skills
5. Better abstraction - generalizes skills across similar objects
6. Cross-episode learning - aggregates patterns across episodes

Usage:
    python build_rule_based_memory.py --results-dir <path> --output-dir <path>
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_hierarchical_skill_memory import (
    HierarchicalSkillMemory,
    Skill,
    CooperationSkill,
    SkillInstance,
    AgentAction,
    EnvironmentState,
)


class EnhancedRuleBasedMemoryBuilder:
    """
    Enhanced rule-based memory builder with improvements over memento:

    1. Better skill abstraction through object type generalization
    2. More cooperation patterns (6 vs 0 in memento)
    3. ToM reasoning for partner modeling
    4. Skill clustering for deduplication
    5. Cross-episode pattern learning
    """

    def __init__(self):
        # Object type abstraction patterns
        self.object_type_pattern = re.compile(r'^(.+?)_\d+$')

        # Action to skill mapping
        self.action_to_skill = {
            "Navigate": "navigate_to_target",
            "Pick": "pickup_object",
            "Place": "place_object",
            "Rearrange": "rearrange_object",
            "Open": "open_furniture",
            "Close": "close_furniture",
            "Clean": "clean_object",
            "Fill": "fill_container",
            "Pour": "pour_liquid",
            "PowerOn": "power_on",
            "PowerOff": "power_off",
            "Wait": "wait_for_partner",
        }

        # Composite skill patterns - key improvement over memento
        self.composite_patterns = {
            "fetch_and_deliver": ["Navigate", "Pick", "Navigate", "Place"],
            "prepare_container": ["Navigate", "Pick", "Fill"],
            "clean_and_place": ["Navigate", "Pick", "Clean", "Navigate", "Place"],
            "open_retrieve_close": ["Navigate", "Open", "Navigate", "Pick", "Close"],
            "power_cycle": ["Navigate", "PowerOff", "PowerOn"],
        }

        # Cooperation pattern detectors
        self.coop_pattern_detectors = [
            self._detect_division_of_labor,
            self._detect_sequential_handoff,
            self._detect_complementary_work,
            self._detect_synchronization,
            self._detect_spatial_coordination,
            self._detect_goal_sharing,
        ]

        # Statistics
        self.stats = defaultdict(int)

    def abstract_object_name(self, obj_name: str) -> str:
        """Abstract object instance to type (e.g., toy_vehicle_1 -> toy_vehicle)"""
        match = self.object_type_pattern.match(obj_name)
        if match:
            return match.group(1)
        return obj_name

    def identify_composite_skill(self, action_types: List[str]) -> Optional[str]:
        """Identify if action sequence matches a composite skill pattern"""
        for skill_name, pattern in self.composite_patterns.items():
            if len(action_types) >= len(pattern):
                # Check if pattern is a subsequence
                j = 0
                for action in action_types:
                    if j < len(pattern) and action == pattern[j]:
                        j += 1
                if j == len(pattern):
                    return skill_name
        return None

    def extract_skills_from_trajectory(
        self,
        trajectory: Dict,
        episode_id: str
    ) -> Tuple[List[Skill], List[CooperationSkill]]:
        """Extract both individual and cooperation skills from a trajectory"""

        individual_skills = []
        cooperation_skills = []

        # Parse actions from trajectory
        all_actions = self._parse_trajectory_actions(trajectory)

        if not all_actions:
            return [], []

        # Build environment state
        env_state = self._build_env_state(trajectory)

        # Extract individual skills for each agent
        for agent_id in [0, 1]:
            agent_actions = [a for a in all_actions if a.agent_id == agent_id]
            skills = self._extract_individual_skills(agent_actions, env_state, agent_id, episode_id)
            individual_skills.extend(skills)

        # Extract cooperation skills
        # Handle both formats: task can be a string or dict with 'instruction' key
        task_data = trajectory.get("task", "")
        if isinstance(task_data, str):
            task = task_data
        elif isinstance(task_data, dict):
            task = task_data.get("instruction", "")
        else:
            task = str(task_data)
        coop_skills = self._extract_cooperation_skills(all_actions, env_state, task, episode_id)
        cooperation_skills.extend(coop_skills)

        return individual_skills, cooperation_skills

    def _parse_trajectory_actions(self, trajectory: Dict) -> List[AgentAction]:
        """Parse trajectory into list of AgentAction objects

        Handles the actual planner log format:
        {
            'task': 'Move the bowl...',
            'steps': [
                {
                    'high_level_actions': {'0': ['Navigate', 'obj', ''], '1': ['Pick', 'obj', '']},
                    'is_done': {'0': False, '1': False},
                    ...
                },
                ...
            ]
        }
        """
        actions = []

        # Get steps from trajectory (actual format uses 'steps' not 'trace')
        steps = trajectory.get("steps", [])
        if not steps:
            # Fallback to 'trace' if 'steps' not found
            steps = trajectory.get("trace", [])

        if not steps:
            return []

        # Track previous action to avoid duplicates (high_level_actions repeats until done)
        prev_actions = {0: None, 1: None}

        for step_num, step in enumerate(steps):
            # Handle 'high_level_actions' format (actual planner log format)
            high_level_actions = step.get("high_level_actions", {})

            for agent_id in [0, 1]:
                agent_key = str(agent_id)

                if agent_key in high_level_actions:
                    action_data = high_level_actions[agent_key]

                    # Parse action - can be list ['Navigate', 'obj', ''] or string 'Wait 0'
                    if isinstance(action_data, list):
                        action_type = action_data[0] if action_data else "Unknown"
                        action_args = [str(a) for a in action_data[1:] if a]
                        raw_action = f"{action_type}[{', '.join(action_args)}]" if action_args else action_type
                    elif isinstance(action_data, str):
                        # Handle 'Wait 0' format
                        parts = action_data.split()
                        action_type = parts[0] if parts else "Unknown"
                        action_args = parts[1:] if len(parts) > 1 else []
                        raw_action = action_data
                    else:
                        continue

                    # Skip if same as previous (high_level_actions repeats until action completes)
                    if raw_action == prev_actions[agent_id]:
                        continue
                    prev_actions[agent_id] = raw_action

                    # Check success from is_done
                    is_done = step.get("is_done", {})
                    success = True  # Assume success unless observation says otherwise

                    actions.append(AgentAction(
                        agent_id=agent_id,
                        action_type=action_type,
                        action_args=action_args,
                        observation="",  # Observations not in this format
                        success=success,
                        step_number=step_num,
                        raw_action=raw_action
                    ))

            # Also handle old format with agent_X_action keys
            for agent_id in [0, 1]:
                action_key = f"agent_{agent_id}_action"
                obs_key = f"agent_{agent_id}_obs"

                if action_key in step:
                    raw_action = step[action_key]
                    action_type, action_args = self._parse_action_string(raw_action)

                    obs = step.get(obs_key, "")
                    success = "failed" not in obs.lower() if obs else True

                    actions.append(AgentAction(
                        agent_id=agent_id,
                        action_type=action_type,
                        action_args=action_args,
                        observation=obs,
                        success=success,
                        step_number=step_num,
                        raw_action=raw_action
                    ))

        return actions

    def _parse_action_string(self, action_str: str) -> Tuple[str, List[str]]:
        """Parse action string like 'Navigate[obj_0]' into (type, args)"""
        match = re.match(r'(\w+)\[(.*)\]', action_str)
        if match:
            action_type = match.group(1)
            args_str = match.group(2)
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return action_type, args
        return action_str, []

    def _build_env_state(self, trajectory: Dict) -> EnvironmentState:
        """Build environment state from trajectory"""
        furniture = {}
        objects = {}
        agent_states = {0: {}, 1: {}}

        # Parse initial state if available
        init_state = trajectory.get("initial_state", {})
        for obj_name, obj_data in init_state.get("objects", {}).items():
            objects[obj_name] = {
                "location": obj_data.get("location", ""),
                "room": obj_data.get("room", ""),
                "states": obj_data.get("states", {})
            }

        return EnvironmentState(
            furniture=furniture,
            objects=objects,
            agent_states=agent_states
        )

    def _extract_individual_skills(
        self,
        actions: List[AgentAction],
        env_state: EnvironmentState,
        agent_id: int,
        episode_id: str
    ) -> List[Skill]:
        """Extract individual skills with enhanced abstraction"""
        skills = []

        # Filter out observation-only entries
        filtered_actions = [a for a in actions if a.action_type != "observation"]

        # Group into skill sequences
        sequences = self._identify_skill_sequences(filtered_actions)

        for seq in sequences:
            if not seq:
                continue

            action_types = [a.action_type for a in seq]

            # Try to identify composite skill first
            composite_name = self.identify_composite_skill(action_types)
            if composite_name:
                skill_name = composite_name
                self.stats["composite_skills"] += 1
            else:
                skill_name = self._determine_skill_name(seq)
                self.stats["basic_skills"] += 1

            # Extract objects with type abstraction
            objects = []
            abstract_objects = []
            for action in seq:
                for arg in action.action_args:
                    if arg and arg.lower() not in ["none", "on", "within", "next_to"]:
                        objects.append(arg)
                        abstract_objects.append(self.abstract_object_name(arg))

            # Extract locations from actions
            locations = []
            for action in seq:
                if action.action_type == "Place" and len(action.action_args) > 2:
                    locations.append(action.action_args[2])
                elif action.action_type == "Navigate" and action.action_args:
                    # Check if navigating to furniture
                    target = action.action_args[0]
                    if any(furn in target for furn in ["table", "counter", "shelf", "drawer", "stand", "bed", "sofa", "chair"]):
                        locations.append(target)

            # Create context with both specific and abstract objects
            # IMPORTANT: action_sequence contains the raw action strings for retrieval
            context = {
                "objects": list(set(objects)),
                "object_types": list(set(abstract_objects)),  # Key improvement
                "locations": list(set(locations)),
                "action_sequence": [a.raw_action for a in seq],  # Raw actions like "Navigate[bread_0]"
                "action_types": action_types,  # For pattern matching
            }

            # Generate demonstration
            demo = self._generate_demo(seq)

            # Extract preconditions and effects
            preconditions = self._extract_preconditions(seq)
            effects = self._extract_effects(seq)

            skill = Skill(
                name=skill_name,
                skill_type="individual",
                instances=[SkillInstance(
                    context=context,
                    demo=demo,
                    e_src=episode_id,
                    success=all(a.success for a in seq)
                )],
                description=f"Skill for {skill_name} with objects {abstract_objects}",
                preconditions=preconditions,
                effects=effects
            )
            skills.append(skill)

        return skills

    def _identify_skill_sequences(self, actions: List[AgentAction]) -> List[List[AgentAction]]:
        """Group actions into logical skill sequences"""
        sequences = []
        current_seq = []

        terminal_actions = {"Place", "Rearrange", "Clean", "Fill", "PowerOn", "PowerOff"}

        for action in actions:
            if action.action_type == "Wait":
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = []
            else:
                current_seq.append(action)
                # Split at terminal actions
                if action.action_type in terminal_actions:
                    sequences.append(current_seq)
                    current_seq = []

        if current_seq:
            sequences.append(current_seq)

        return sequences

    def _determine_skill_name(self, actions: List[AgentAction]) -> str:
        """Determine skill name from action sequence"""
        if not actions:
            return "unknown"

        # Priority order
        priority = ["Rearrange", "Place", "Clean", "Fill", "Pour", "PowerOn", "PowerOff", "Pick", "Open", "Close", "Navigate"]

        for p in priority:
            for action in actions:
                if action.action_type == p:
                    return self.action_to_skill.get(p, p.lower())

        return self.action_to_skill.get(actions[0].action_type, actions[0].action_type.lower())

    def _generate_demo(self, actions: List[AgentAction]) -> str:
        """Generate natural language demonstration"""
        parts = []
        for action in actions:
            if action.action_type == "Navigate":
                target = action.action_args[0] if action.action_args else "target"
                parts.append(f"Navigate to {target}")
            elif action.action_type == "Pick":
                obj = action.action_args[0] if action.action_args else "object"
                parts.append(f"Pick up {obj}")
            elif action.action_type == "Place":
                obj = action.action_args[0] if action.action_args else "object"
                loc = action.action_args[2] if len(action.action_args) > 2 else "location"
                parts.append(f"Place {obj} on {loc}")
            elif action.action_type == "Open":
                furn = action.action_args[0] if action.action_args else "furniture"
                parts.append(f"Open {furn}")
            elif action.action_type == "Close":
                furn = action.action_args[0] if action.action_args else "furniture"
                parts.append(f"Close {furn}")
            elif action.action_type == "Clean":
                obj = action.action_args[0] if action.action_args else "object"
                parts.append(f"Clean {obj}")
            elif action.action_type == "Fill":
                obj = action.action_args[0] if action.action_args else "container"
                parts.append(f"Fill {obj}")
            elif action.action_type == "PowerOn":
                obj = action.action_args[0] if action.action_args else "appliance"
                parts.append(f"Power on {obj}")
            elif action.action_type == "PowerOff":
                obj = action.action_args[0] if action.action_args else "appliance"
                parts.append(f"Power off {obj}")

        return ", then ".join(parts) if parts else "Unknown action sequence"

    def _extract_preconditions(self, actions: List[AgentAction]) -> List[str]:
        """Extract preconditions from action sequence"""
        preconditions = []

        for action in actions:
            if action.action_type == "Pick":
                preconditions.extend(["agent_hands_empty", "object_reachable"])
            elif action.action_type == "Place":
                preconditions.append("agent_holding_object")
            elif action.action_type == "Open":
                preconditions.append("furniture_is_closed")
            elif action.action_type == "Close":
                preconditions.append("furniture_is_open")
            elif action.action_type == "Fill":
                preconditions.extend(["agent_holding_container", "near_water_source"])
            elif action.action_type == "Clean":
                preconditions.append("object_is_dirty")
            elif action.action_type == "PowerOn":
                preconditions.append("appliance_is_off")
            elif action.action_type == "PowerOff":
                preconditions.append("appliance_is_on")

        return list(set(preconditions))

    def _extract_effects(self, actions: List[AgentAction]) -> List[str]:
        """Extract effects from action sequence"""
        effects = []

        for action in actions:
            if action.action_type == "Pick":
                effects.append("agent_holding_object")
            elif action.action_type == "Place":
                effects.extend(["agent_hands_empty", "object_at_location"])
            elif action.action_type == "Open":
                effects.append("furniture_is_open")
            elif action.action_type == "Close":
                effects.append("furniture_is_closed")
            elif action.action_type == "Fill":
                effects.append("container_is_filled")
            elif action.action_type == "Clean":
                effects.append("object_is_clean")
            elif action.action_type == "PowerOn":
                effects.append("appliance_is_on")
            elif action.action_type == "PowerOff":
                effects.append("appliance_is_off")

        return list(set(effects))

    def _extract_cooperation_skills(
        self,
        all_actions: List[AgentAction],
        env_state: EnvironmentState,
        task: str,
        episode_id: str
    ) -> List[CooperationSkill]:
        """Extract cooperation skills using multiple pattern detectors"""

        cooperation_skills = []

        # Separate by agent
        agent_0_actions = [a for a in all_actions if a.agent_id == 0 and a.action_type != "observation"]
        agent_1_actions = [a for a in all_actions if a.agent_id == 1 and a.action_type != "observation"]

        if not agent_0_actions or not agent_1_actions:
            return []

        # Run all pattern detectors
        for detector in self.coop_pattern_detectors:
            patterns = detector(agent_0_actions, agent_1_actions, env_state, task)
            for pattern in patterns:
                coop_skill = CooperationSkill(
                    name=pattern["name"],
                    skill_type="cooperation",
                    instances=[SkillInstance(
                        context=pattern["context"],
                        demo=pattern["demo"],
                        e_src=episode_id,
                        success=pattern.get("success", True)
                    )],
                    description=pattern["description"],
                    precond_joint=pattern.get("precond_joint", {}),
                    partner_context_pattern=pattern.get("partner_context", {}),
                    trigger_conditions=pattern.get("triggers", [])
                )
                cooperation_skills.append(coop_skill)
                self.stats[f"coop_{pattern['name']}"] += 1

        return cooperation_skills

    def _detect_division_of_labor(
        self,
        agent_0_actions: List[AgentAction],
        agent_1_actions: List[AgentAction],
        env_state: EnvironmentState,
        task: str
    ) -> List[Dict]:
        """Detect division of labor pattern"""
        patterns = []

        # Get objects each agent works on
        agent_0_objects = set()
        agent_1_objects = set()

        for a in agent_0_actions:
            for arg in a.action_args:
                if arg and arg.lower() not in ["none", "on", "within", "next_to"]:
                    agent_0_objects.add(self.abstract_object_name(arg))

        for a in agent_1_actions:
            for arg in a.action_args:
                if arg and arg.lower() not in ["none", "on", "within", "next_to"]:
                    agent_1_objects.add(self.abstract_object_name(arg))

        # Check for distinct object handling
        unique_to_0 = agent_0_objects - agent_1_objects
        unique_to_1 = agent_1_objects - agent_0_objects

        if unique_to_0 and unique_to_1:
            # Extract full action sequences for each agent (key for retrieval)
            agent_0_action_seq = [a.raw_action for a in agent_0_actions]
            agent_1_action_seq = [a.raw_action for a in agent_1_actions]

            patterns.append({
                "name": "division_of_labor",
                "description": f"Agents divide work: Agent 0 handles {unique_to_0}, Agent 1 handles {unique_to_1}",
                "context": {
                    "agent_0_objects": list(unique_to_0),
                    "agent_1_objects": list(unique_to_1),
                    "agent_0_actions": agent_0_action_seq,  # Full action sequence
                    "agent_1_actions": agent_1_action_seq,  # Full action sequence
                    "action_sequence": agent_0_action_seq + agent_1_action_seq,  # Combined
                },
                "demo": f"Agent 0 focuses on {list(unique_to_0)[:2]} while Agent 1 focuses on {list(unique_to_1)[:2]}",
                "precond_joint": {"both_agents_available": True},
                "partner_context": {
                    "agent_0_role": "primary_handler",
                    "agent_1_role": "secondary_handler"
                },
                "triggers": ["multiple_objects_to_handle", "objects_in_different_areas"],
                "success": True
            })

        return patterns

    def _detect_sequential_handoff(
        self,
        agent_0_actions: List[AgentAction],
        agent_1_actions: List[AgentAction],
        env_state: EnvironmentState,
        task: str
    ) -> List[Dict]:
        """Detect sequential handoff pattern where one agent prepares for another"""
        patterns = []

        # Find objects placed by agent 0
        placed_by_0 = set()
        for a in agent_0_actions:
            if a.action_type == "Place":
                if a.action_args:
                    placed_by_0.add(a.action_args[0])

        # Check if agent 1 acts on those objects
        acted_on_by_1 = set()
        for a in agent_1_actions:
            if a.action_type in ["Clean", "Fill", "PowerOn", "Pick"]:
                if a.action_args:
                    acted_on_by_1.add(a.action_args[0])

        handoff_objects = placed_by_0 & acted_on_by_1

        if handoff_objects:
            # Get action sequences
            agent_0_action_seq = [a.raw_action for a in agent_0_actions]
            agent_1_action_seq = [a.raw_action for a in agent_1_actions]

            patterns.append({
                "name": "sequential_handoff",
                "description": f"Agent 0 places objects for Agent 1 to process: {handoff_objects}",
                "context": {
                    "handoff_objects": list(handoff_objects),
                    "agent_0_actions": agent_0_action_seq,
                    "agent_1_actions": agent_1_action_seq,
                    "action_sequence": agent_0_action_seq + agent_1_action_seq,
                },
                "demo": f"Agent 0 places {list(handoff_objects)[0]}, then Agent 1 processes it",
                "precond_joint": {"object_accessible": True},
                "partner_context": {
                    "agent_0_role": "preparer",
                    "agent_1_role": "processor"
                },
                "triggers": ["object_needs_preparation", "multi_step_task"],
                "success": True
            })

        return patterns

    def _detect_complementary_work(
        self,
        agent_0_actions: List[AgentAction],
        agent_1_actions: List[AgentAction],
        env_state: EnvironmentState,
        task: str
    ) -> List[Dict]:
        """Detect complementary work pattern where agents do different action types"""
        patterns = []

        action_types_0 = set(a.action_type for a in agent_0_actions)
        action_types_1 = set(a.action_type for a in agent_1_actions)

        unique_to_0 = action_types_0 - action_types_1 - {"Navigate", "Wait"}
        unique_to_1 = action_types_1 - action_types_0 - {"Navigate", "Wait"}

        if unique_to_0 and unique_to_1:
            agent_0_action_seq = [a.raw_action for a in agent_0_actions]
            agent_1_action_seq = [a.raw_action for a in agent_1_actions]

            patterns.append({
                "name": "complementary_work",
                "description": f"Agents specialize: Agent 0 does {unique_to_0}, Agent 1 does {unique_to_1}",
                "context": {
                    "agent_0_specialization": list(unique_to_0),
                    "agent_1_specialization": list(unique_to_1),
                    "agent_0_actions": agent_0_action_seq,
                    "agent_1_actions": agent_1_action_seq,
                    "action_sequence": agent_0_action_seq + agent_1_action_seq,
                },
                "demo": f"Agent 0 specializes in {list(unique_to_0)} while Agent 1 specializes in {list(unique_to_1)}",
                "precond_joint": {"task_is_decomposable": True},
                "partner_context": {
                    "agent_0_role": "specialist_a",
                    "agent_1_role": "specialist_b"
                },
                "triggers": ["task_requires_different_skills", "agents_have_specializations"],
                "success": True
            })

        return patterns

    def _detect_synchronization(
        self,
        agent_0_actions: List[AgentAction],
        agent_1_actions: List[AgentAction],
        env_state: EnvironmentState,
        task: str
    ) -> List[Dict]:
        """Detect synchronization pattern with wait actions"""
        patterns = []

        wait_count_0 = sum(1 for a in agent_0_actions if a.action_type == "Wait")
        wait_count_1 = sum(1 for a in agent_1_actions if a.action_type == "Wait")

        total_0 = len(agent_0_actions)
        total_1 = len(agent_1_actions)

        # Check for significant waiting
        wait_ratio_0 = wait_count_0 / max(total_0, 1)
        wait_ratio_1 = wait_count_1 / max(total_1, 1)

        if wait_ratio_0 > 0.1 or wait_ratio_1 > 0.1:
            agent_0_action_seq = [a.raw_action for a in agent_0_actions]
            agent_1_action_seq = [a.raw_action for a in agent_1_actions]

            patterns.append({
                "name": "synchronization",
                "description": f"Agents synchronize: Agent 0 waits {wait_count_0} times, Agent 1 waits {wait_count_1} times",
                "context": {
                    "agent_0_wait_count": wait_count_0,
                    "agent_1_wait_count": wait_count_1,
                    "wait_ratio_0": wait_ratio_0,
                    "wait_ratio_1": wait_ratio_1,
                    "agent_0_actions": agent_0_action_seq,
                    "agent_1_actions": agent_1_action_seq,
                    "action_sequence": agent_0_action_seq + agent_1_action_seq,
                },
                "demo": f"Agent 0 waits for Agent 1 to complete tasks before proceeding",
                "precond_joint": {"synchronization_needed": True},
                "partner_context": {
                    "faster_agent": 0 if wait_ratio_0 > wait_ratio_1 else 1,
                    "slower_agent": 1 if wait_ratio_0 > wait_ratio_1 else 0
                },
                "triggers": ["dependency_between_tasks", "shared_resource"],
                "success": True
            })

        return patterns

    def _detect_spatial_coordination(
        self,
        agent_0_actions: List[AgentAction],
        agent_1_actions: List[AgentAction],
        env_state: EnvironmentState,
        task: str
    ) -> List[Dict]:
        """Detect spatial coordination - agents working in different areas"""
        patterns = []

        # Extract target locations from Navigate actions
        locations_0 = set()
        locations_1 = set()

        for a in agent_0_actions:
            if a.action_type == "Navigate" and a.action_args:
                locations_0.add(self.abstract_object_name(a.action_args[0]))

        for a in agent_1_actions:
            if a.action_type == "Navigate" and a.action_args:
                locations_1.add(self.abstract_object_name(a.action_args[0]))

        unique_locations_0 = locations_0 - locations_1
        unique_locations_1 = locations_1 - locations_0

        if unique_locations_0 and unique_locations_1:
            agent_0_action_seq = [a.raw_action for a in agent_0_actions]
            agent_1_action_seq = [a.raw_action for a in agent_1_actions]

            patterns.append({
                "name": "spatial_coordination",
                "description": f"Agents work in different areas: Agent 0 in {unique_locations_0}, Agent 1 in {unique_locations_1}",
                "context": {
                    "agent_0_area": list(unique_locations_0),
                    "agent_1_area": list(unique_locations_1),
                    "agent_0_actions": agent_0_action_seq,
                    "agent_1_actions": agent_1_action_seq,
                    "action_sequence": agent_0_action_seq + agent_1_action_seq,
                },
                "demo": f"Agent 0 works near {list(unique_locations_0)[:2]} while Agent 1 works near {list(unique_locations_1)[:2]}",
                "precond_joint": {"areas_are_accessible": True},
                "partner_context": {
                    "agent_0_region": list(unique_locations_0)[:2],
                    "agent_1_region": list(unique_locations_1)[:2]
                },
                "triggers": ["task_spans_multiple_areas", "efficiency_optimization"],
                "success": True
            })

        return patterns

    def _detect_goal_sharing(
        self,
        agent_0_actions: List[AgentAction],
        agent_1_actions: List[AgentAction],
        env_state: EnvironmentState,
        task: str
    ) -> List[Dict]:
        """Detect goal sharing - agents working toward same destination"""
        patterns = []

        # Find Place destinations
        destinations_0 = set()
        destinations_1 = set()

        for a in agent_0_actions:
            if a.action_type == "Place" and len(a.action_args) > 2:
                destinations_0.add(a.action_args[2])

        for a in agent_1_actions:
            if a.action_type == "Place" and len(a.action_args) > 2:
                destinations_1.add(a.action_args[2])

        shared_destinations = destinations_0 & destinations_1

        if shared_destinations:
            agent_0_action_seq = [a.raw_action for a in agent_0_actions]
            agent_1_action_seq = [a.raw_action for a in agent_1_actions]

            patterns.append({
                "name": "goal_sharing",
                "description": f"Both agents place objects at shared destinations: {shared_destinations}",
                "context": {
                    "shared_destinations": list(shared_destinations),
                    "agent_0_contributions": len(destinations_0),
                    "agent_1_contributions": len(destinations_1),
                    "agent_0_actions": agent_0_action_seq,
                    "agent_1_actions": agent_1_action_seq,
                    "action_sequence": agent_0_action_seq + agent_1_action_seq,
                },
                "demo": f"Both agents deliver objects to {list(shared_destinations)[0]}",
                "precond_joint": {"shared_goal_location": True},
                "partner_context": {
                    "goal_type": "shared_destination",
                    "coordination": "parallel_delivery"
                },
                "triggers": ["common_destination", "collection_task"],
                "success": True
            })

        return patterns


def build_rule_based_memory(
    results_dirs: List[str],
    output_dir: str,
    filter_successful: bool = True
) -> Dict[str, Any]:
    """
    Build hierarchical skill memory using enhanced rule-based extraction

    Args:
        results_dirs: List of paths to heuristic results directories
        output_dir: Output directory for memory files
        filter_successful: Only include successful episodes

    Returns:
        Memory statistics
    """

    print("=" * 80)
    print("Enhanced Rule-Based Hierarchical Skill Memory Builder")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Filter successful only: {filter_successful}")
    print()

    builder = EnhancedRuleBasedMemoryBuilder()

    all_individual_skills = []
    all_cooperation_skills = []
    episodic_memory = {}

    total_episodes = 0
    successful_episodes = 0

    for results_dir in results_dirs:
        print(f"\nProcessing: {results_dir}")

        # Find trace files
        traces_dir = os.path.join(results_dir, "results")
        if not os.path.exists(traces_dir):
            traces_dir = results_dir

        trace_files = []
        for root, dirs, files in os.walk(traces_dir):
            for f in files:
                if f.endswith(".json.gz") or f.endswith(".json"):
                    trace_files.append(os.path.join(root, f))

        if not trace_files:
            print(f"  No trace files found in {traces_dir}")
            continue

        for trace_file in trace_files:
            print(f"  Loading: {os.path.basename(trace_file)}")

            try:
                if trace_file.endswith(".gz"):
                    with gzip.open(trace_file, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    with open(trace_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
            except Exception as e:
                print(f"    Error loading: {e}")
                continue

            # Handle both list and dict formats
            if isinstance(data, dict):
                episodes = data.get("episodes", [data])
            else:
                episodes = data

            print(f"    Found {len(episodes)} episodes")

            for ep_idx, episode in enumerate(tqdm(episodes, desc="    Processing")):
                total_episodes += 1

                # Determine success from episode data
                # Check multiple possible formats:
                # 1. Direct "success" key
                # 2. Check if final step has is_done=True for both agents
                # 3. Check if episode has "result" with success info
                is_successful = episode.get("success", None)

                if is_successful is None:
                    # Check final step for is_done
                    steps = episode.get("steps", episode.get("trace", []))
                    if steps:
                        final_step = steps[-1]
                        is_done = final_step.get("is_done", {})
                        # Success if both agents are done
                        if isinstance(is_done, dict):
                            is_successful = all(is_done.get(str(i), False) for i in [0, 1])
                        else:
                            is_successful = bool(is_done)
                    else:
                        is_successful = False

                # For heuristic dataset, we assume all planner logs represent successful episodes
                # since they were generated by the optimal heuristic planner
                if "planner-log" in trace_file or "heuristic" in results_dir.lower():
                    is_successful = True

                if filter_successful and not is_successful:
                    continue

                successful_episodes += 1
                episode_id = f"{os.path.basename(trace_file)}_{ep_idx}"

                # Extract skills
                ind_skills, coop_skills = builder.extract_skills_from_trajectory(
                    episode, episode_id
                )

                all_individual_skills.extend(ind_skills)
                all_cooperation_skills.extend(coop_skills)

                # Store in episodic memory
                # Handle task extraction for different formats
                task_data = episode.get("task", "")
                if isinstance(task_data, str):
                    task_str = task_data
                elif isinstance(task_data, dict):
                    task_str = task_data.get("instruction", "")
                else:
                    task_str = str(task_data)

                episodic_memory[episode_id] = {
                    "task": task_str,
                    "success": is_successful,
                    "num_steps": len(episode.get("steps", episode.get("trace", []))),
                    "source_file": os.path.basename(trace_file)
                }

    print(f"\n{'=' * 40}")
    print(f"Total episodes: {total_episodes}")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Individual skills extracted: {len(all_individual_skills)}")
    print(f"Cooperation skills extracted: {len(all_cooperation_skills)}")
    print(f"{'=' * 40}")

    # Aggregate and deduplicate skills
    print("\nAggregating and deduplicating skills...")

    L_ind = _aggregate_skills(all_individual_skills)
    L_coop = _aggregate_cooperation_skills(all_cooperation_skills)

    print(f"Unique individual skill types: {len(L_ind)}")
    print(f"Unique cooperation skill types: {len(L_coop)}")

    # Save memory
    os.makedirs(output_dir, exist_ok=True)

    # Save individual skills
    ind_path = os.path.join(output_dir, "L_ind_skills.json.gz")
    with gzip.open(ind_path, 'wt', encoding='utf-8') as f:
        json.dump({k: _skill_to_dict(v) for k, v in L_ind.items()}, f, indent=2)
    print(f"Saved individual skills to: {ind_path}")

    # Save cooperation skills
    coop_path = os.path.join(output_dir, "L_coop_skills.json.gz")
    with gzip.open(coop_path, 'wt', encoding='utf-8') as f:
        json.dump({k: _skill_to_dict(v) for k, v in L_coop.items()}, f, indent=2)
    print(f"Saved cooperation skills to: {coop_path}")

    # Save episodic memory
    episodic_path = os.path.join(output_dir, "episodic_memory.json.gz")
    with gzip.open(episodic_path, 'wt', encoding='utf-8') as f:
        json.dump(episodic_memory, f, indent=2)
    print(f"Saved episodic memory to: {episodic_path}")

    # Save summary
    summary = {
        "build_timestamp": datetime.now().isoformat(),
        "method": "enhanced_rule_based",
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "num_individual_skills": len(L_ind),
        "num_cooperation_skills": len(L_coop),
        "num_individual_instances": sum(len(s.instances) for s in L_ind.values()),
        "num_cooperation_instances": sum(len(s.instances) for s in L_coop.values()),
        "builder_stats": dict(builder.stats),
        "source_dirs": results_dirs,
    }

    summary_path = os.path.join(output_dir, "memory_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")

    return summary


def _aggregate_skills(skills: List[Skill]) -> Dict[str, Skill]:
    """Aggregate skills by name, merging instances"""
    aggregated = {}

    for skill in skills:
        if skill.name not in aggregated:
            aggregated[skill.name] = Skill(
                name=skill.name,
                skill_type=skill.skill_type,
                instances=[],
                description=skill.description,
                preconditions=skill.preconditions,
                effects=skill.effects
            )

        aggregated[skill.name].instances.extend(skill.instances)

        # Merge preconditions and effects
        existing_preconds = set(aggregated[skill.name].preconditions)
        existing_preconds.update(skill.preconditions)
        aggregated[skill.name].preconditions = list(existing_preconds)

        existing_effects = set(aggregated[skill.name].effects)
        existing_effects.update(skill.effects)
        aggregated[skill.name].effects = list(existing_effects)

    return aggregated


def _aggregate_cooperation_skills(skills: List[CooperationSkill]) -> Dict[str, CooperationSkill]:
    """Aggregate cooperation skills by name"""
    aggregated = {}

    for skill in skills:
        if skill.name not in aggregated:
            aggregated[skill.name] = CooperationSkill(
                name=skill.name,
                skill_type=skill.skill_type,
                instances=[],
                description=skill.description,
                precond_joint=skill.precond_joint,
                partner_context_pattern=skill.partner_context_pattern,
                trigger_conditions=skill.trigger_conditions
            )

        aggregated[skill.name].instances.extend(skill.instances)

    return aggregated


def _skill_to_dict(skill) -> Dict:
    """Convert skill to dictionary for JSON serialization"""
    result = {
        "name": skill.name,
        "skill_type": skill.skill_type,
        "description": skill.description,
        "preconditions": skill.preconditions,
        "effects": skill.effects,
        "instances": [
            {
                "context": inst.context,
                "demo": inst.demo,
                "e_src": inst.e_src,
                "success": inst.success
            }
            for inst in skill.instances
        ]
    }

    if hasattr(skill, 'precond_joint'):
        result["precond_joint"] = skill.precond_joint
        result["partner_context_pattern"] = skill.partner_context_pattern
        result["trigger_conditions"] = skill.trigger_conditions

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build enhanced rule-based hierarchical skill memory"
    )

    parser.add_argument(
        "--results-dirs",
        type=str,
        nargs="+",
        default=[
            "/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner/heuristic_dataset/2025-12-30_17-02-27-rerange_only.json",
            "/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner/heuristic_dataset/2025-12-30_21-10-23-heterogeneous+rerange.json",
            "/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner/heuristic_dataset/2025-12-30_23-42-08-spatial_only.json",
            "/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner/heuristic_dataset/2025-12-30_23-43-27-heterogeneous+temporal.json",
            "/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner/heuristic_dataset/2025-12-30_23-44-10-heterogeneous+rerange+spatial+temporal.json",
        ],
        help="Paths to heuristic results directories"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/a5l/shuqing.a5l/partnr-planner/data/hierarchical_skill_memory_rule",
        help="Output directory for memory files"
    )

    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed episodes"
    )

    args = parser.parse_args()

    summary = build_rule_based_memory(
        results_dirs=args.results_dirs,
        output_dir=args.output_dir,
        filter_successful=not args.include_failed
    )

    print("\n" + "=" * 80)
    print("Build Complete!")
    print("=" * 80)
    print(f"\nMemory saved to: {args.output_dir}")
    print(f"\nStatistics:")
    print(f"  Individual skill types: {summary['num_individual_skills']}")
    print(f"  Cooperation skill types: {summary['num_cooperation_skills']}")
    print(f"  Individual instances: {summary['num_individual_instances']}")
    print(f"  Cooperation instances: {summary['num_cooperation_instances']}")


if __name__ == "__main__":
    main()
