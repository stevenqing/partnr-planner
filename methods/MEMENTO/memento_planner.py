#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEMENTO Planner: Integration with LLM Planner for Personalized Assistance

This module integrates MEMENTO's user profile memory with the partnr-planner
system for personalized embodied agent planning.

Key capabilities:
1. Retrieve personalized knowledge for task planning
2. Ground instructions in user-specific context
3. Handle ambiguous references using memory
4. Coordinate multi-agent planning with personalized goals
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .user_profile_memory import (
    UserProfileMemory,
    KnowledgeNode,
    KnowledgeType,
)
from .knowledge_graph_builder import (
    KnowledgeGraphBuilder,
)
from .memento_retriever import (
    MementoRetriever,
    RetrievedKnowledge,
)


@dataclass
class PersonalizedGoal:
    """A goal grounded in personalized knowledge."""
    original_instruction: str
    grounded_instruction: str
    target_objects: List[str]
    target_locations: List[str]
    relevant_memories: List[RetrievedKnowledge]
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_instruction": self.original_instruction,
            "grounded_instruction": self.grounded_instruction,
            "target_objects": self.target_objects,
            "target_locations": self.target_locations,
            "relevant_memories": [m.to_dict() for m in self.relevant_memories],
            "confidence": self.confidence,
        }


class MementoPlanner:
    """
    MEMENTO-enhanced planner for personalized task execution.

    This planner uses user profile memory to:
    1. Resolve ambiguous object references
    2. Complete partial instructions using user patterns
    3. Ground planning in personalized knowledge
    """

    def __init__(
        self,
        memory: UserProfileMemory,
        retriever: Optional[MementoRetriever] = None,
        builder: Optional[KnowledgeGraphBuilder] = None,
        update_memory: bool = True,
        llm_model: Optional[Any] = None,
    ):
        """
        Initialize the MEMENTO planner.

        Args:
            memory: User profile memory
            retriever: Memory retriever (created if not provided)
            builder: Knowledge graph builder (for memory updates)
            update_memory: Whether to update memory after task completion
            llm_model: LLM model for grounding
        """
        self.memory = memory
        self.retriever = retriever or MementoRetriever(memory)
        self.builder = builder or KnowledgeGraphBuilder(memory)
        self.update_memory = update_memory
        self.llm_model = llm_model

        # Track episodic memory for current session
        self.current_episode: List[Dict[str, Any]] = []
        self.session_memories: List[RetrievedKnowledge] = []

    def derive_goal(
        self,
        instruction: str,
        user_id: str = "user_0",
    ) -> PersonalizedGoal:
        """
        Derive a grounded goal from instruction using memory.

        Implements: phi(I, M) -> g = (o_i, l_i)_{i=1}^k

        Args:
            instruction: Natural language instruction
            user_id: User ID for personalization

        Returns:
            PersonalizedGoal with grounded objects and locations
        """
        # Step 1: Retrieve relevant memories
        memories = self.retriever.retrieve(instruction)
        self.session_memories.extend(memories)

        # Step 2: Extract target objects and locations
        target_objects = []
        target_locations = []

        for memory in memories:
            target_objects.extend(memory.objects)
            target_locations.extend(memory.locations)

        # Deduplicate
        target_objects = list(dict.fromkeys(target_objects))
        target_locations = list(dict.fromkeys(target_locations))

        # Step 3: Ground the instruction
        grounded_instruction = self._ground_instruction(
            instruction, memories, target_objects, target_locations
        )

        # Calculate confidence based on memory relevance
        if memories:
            confidence = sum(m.similarity_score for m in memories) / len(memories)
        else:
            confidence = 0.5  # Lower confidence without memory support

        return PersonalizedGoal(
            original_instruction=instruction,
            grounded_instruction=grounded_instruction,
            target_objects=target_objects,
            target_locations=target_locations,
            relevant_memories=memories,
            confidence=confidence,
        )

    def _ground_instruction(
        self,
        instruction: str,
        memories: List[RetrievedKnowledge],
        objects: List[str],
        locations: List[str],
    ) -> str:
        """Ground instruction with specific objects and locations from memory."""
        grounded = instruction

        if not memories:
            return grounded

        # Build substitution map from memories
        substitutions = {}

        for memory in memories:
            # Map aliases to actual objects
            if memory.alias:
                alias_lower = memory.alias.lower()
                if memory.objects:
                    substitutions[alias_lower] = memory.objects[0]
                if memory.locations:
                    # Check if alias refers to a location pattern
                    if "place" in alias_lower or "spot" in alias_lower:
                        substitutions[alias_lower] = memory.locations[0]

        # Apply substitutions
        grounded_lower = grounded.lower()
        for alias, replacement in substitutions.items():
            if alias in grounded_lower:
                # Find case-insensitive and replace
                import re
                pattern = re.compile(re.escape(alias), re.IGNORECASE)
                grounded = pattern.sub(replacement, grounded)

        # Add context if we have specific objects/locations
        if objects and "object" not in grounded.lower():
            grounded += f" (objects: {', '.join(objects[:3])})"

        return grounded

    def plan_step(
        self,
        observation: str,
        goal: PersonalizedGoal,
        previous_actions: List[str] = None,
        agent_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Plan the next action step.

        Args:
            observation: Current observation
            goal: Personalized goal to achieve
            previous_actions: Previous actions taken
            agent_id: The agent planning the action

        Returns:
            Dict with 'thought' and 'action' keys
        """
        # Build context from memories
        memory_context = self.retriever.format_for_prompt(goal.relevant_memories)

        # Record step in episode
        step = {
            "observation": observation,
            "goal": goal.grounded_instruction,
            "agent_id": agent_id,
        }
        self.current_episode.append(step)

        # If LLM is available, use it for planning
        if self.llm_model:
            return self._llm_plan_step(
                observation, goal, memory_context, previous_actions, agent_id
            )
        else:
            # Return a basic planning structure
            return {
                "thought": f"Working towards: {goal.grounded_instruction}",
                "action": None,  # To be filled by downstream planner
                "memory_context": memory_context,
            }

    def _llm_plan_step(
        self,
        observation: str,
        goal: PersonalizedGoal,
        memory_context: str,
        previous_actions: List[str],
        agent_id: int,
    ) -> Dict[str, Any]:
        """Use LLM to plan the next step."""
        prompt = f"""You are an embodied agent executing a personalized task.

{memory_context}

## Current Goal
{goal.grounded_instruction}

## Target Objects
{', '.join(goal.target_objects) if goal.target_objects else 'Not specified'}

## Target Locations
{', '.join(goal.target_locations) if goal.target_locations else 'Not specified'}

## Current Observation
{observation}

## Previous Actions
{chr(10).join(previous_actions[-5:]) if previous_actions else 'None'}

Based on the personalized knowledge from memory and current observation, what should Agent_{agent_id} do next?

Respond in JSON format:
{{"thought": "your reasoning using personalized knowledge", "action": "the next action to take"}}
"""

        try:
            response = self.llm_model.generate(prompt)
            result = json.loads(response)
            return result
        except Exception as e:
            print(f"LLM planning failed: {e}")
            return {
                "thought": f"Attempting to achieve: {goal.grounded_instruction}",
                "action": None,
            }

    def complete_episode(
        self,
        success: bool,
        final_observation: str,
    ):
        """
        Complete the current episode and optionally update memory.

        Args:
            success: Whether the episode was successful
            final_observation: Final observation after episode
        """
        # Record final step
        self.current_episode.append({
            "final_observation": final_observation,
            "success": success,
        })

        # Update memory if enabled and successful
        if self.update_memory and success and self.current_episode:
            # Extract instruction from first step
            first_step = self.current_episode[0]
            instruction = first_step.get("goal", "")

            # Build trajectory for memory update
            trajectory = []
            for step in self.current_episode:
                if "action" in step:
                    trajectory.append(step)

            # Update knowledge graph
            self.builder.update(instruction, trajectory=trajectory)

        # Clear current episode
        self.current_episode = []

    def get_prompt_context(
        self,
        instruction: str,
        max_memories: int = 3,
    ) -> str:
        """
        Get memory context for inclusion in LLM prompt.

        Args:
            instruction: The instruction to get context for
            max_memories: Maximum number of memories to include

        Returns:
            Formatted context string
        """
        memories = self.retriever.retrieve(instruction)[:max_memories]
        return self.retriever.format_for_prompt(memories)


class MementoRAGIntegration:
    """
    Integration layer between MEMENTO and the existing RAG system.

    This class provides a compatible interface for using MEMENTO
    with the existing partnr-planner infrastructure.
    """

    def __init__(
        self,
        memory_path: Union[str, Path],
        llm_config: Optional[Any] = None,
        top_k: int = 5,
    ):
        """
        Initialize RAG integration.

        Args:
            memory_path: Path to user profile memory
            llm_config: LLM configuration for compatibility
            top_k: Number of memories to retrieve
        """
        self.memory_path = Path(memory_path)
        self.llm_config = llm_config
        self.top_k = top_k

        # Load or create memory
        if self.memory_path.exists():
            self.memory = UserProfileMemory.load(self.memory_path)
        else:
            self.memory = UserProfileMemory()

        self.retriever = MementoRetriever(
            memory=self.memory,
            top_k=top_k,
        )

        self.planner = MementoPlanner(
            memory=self.memory,
            retriever=self.retriever,
        )

        # RAG-compatible data structures
        self._example_type = "memento"
        self.data_dict = {}
        self._build_data_dict()

    def _build_data_dict(self):
        """Build data_dict for RAG compatibility."""
        idx = 0
        for node_id, node in self.memory.nodes.items():
            if isinstance(node, KnowledgeNode):
                reformulated = self.memory.reformulate_knowledge(node_id)
                elements = self.memory.get_elements_for_knowledge(node_id)

                self.data_dict[idx] = {
                    "knowledge_id": node_id,
                    "instruction": reformulated,
                    "knowledge_type": node.knowledge_type.value,
                    "alias": node.alias,
                    "description": node.description,
                    "trace": reformulated,
                    "agent_id": 0,
                    "embedding": None,  # Will be computed on demand
                }
                idx += 1

    def retrieve_top_k_given_query(
        self,
        query: str,
        top_k: int = 1,
        agent_id: int = 0,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RAG-compatible retrieval interface.

        Args:
            query: The query instruction
            top_k: Number of results
            agent_id: Agent ID (for compatibility)

        Returns:
            Tuple of (scores, indices)
        """
        # Derive personalized goal
        goal = self.planner.derive_goal(query)

        # Get memories
        memories = goal.relevant_memories[:top_k]

        scores = []
        indices = []

        for memory in memories:
            # Find matching entry in data_dict
            for idx, entry in self.data_dict.items():
                if entry.get("knowledge_id") == memory.knowledge_id:
                    indices.append(idx)
                    scores.append(memory.similarity_score)
                    break

        # If fewer results than requested, return what we have
        if len(indices) < top_k:
            # Pad with zeros
            while len(indices) < min(top_k, len(self.data_dict)):
                for idx in self.data_dict:
                    if idx not in indices:
                        indices.append(idx)
                        scores.append(0.0)
                        break

        return np.array(scores), np.array(indices)

    def get_example(self, index: int) -> Dict[str, Any]:
        """Get a specific example from data_dict."""
        return self.data_dict.get(index, {})

    def format_examples_for_prompt(
        self,
        indices: np.ndarray,
        agent_id: int = 0,
    ) -> str:
        """Format retrieved examples for LLM prompt."""
        parts = ["## Personalized Context from Memory\n"]

        for i, idx in enumerate(indices[:3]):
            if idx in self.data_dict:
                entry = self.data_dict[idx]

                parts.append(f"### Context {i+1}")
                if entry.get("alias"):
                    parts.append(f"**User Reference:** {entry['alias']}")

                parts.append(f"**Type:** {entry['knowledge_type'].replace('_', ' ').title()}")

                if entry.get("description"):
                    parts.append(f"{entry['description']}")

                parts.append(f"\n{entry.get('trace', '')}")
                parts.append("")

        return "\n".join(parts)

    def update_from_episode(
        self,
        instruction: str,
        trajectory: List[Dict[str, Any]],
        success: bool,
    ):
        """
        Update memory from a completed episode.

        Args:
            instruction: The task instruction
            trajectory: The execution trajectory
            success: Whether the task succeeded
        """
        if success:
            builder = KnowledgeGraphBuilder(memory=self.memory)
            builder.update(instruction, trajectory=trajectory)
            self._build_data_dict()  # Rebuild for new knowledge

    def save_memory(self, path: Optional[Union[str, Path]] = None):
        """Save memory to file."""
        save_path = path or self.memory_path
        self.memory.save(save_path)


def create_planner(
    memory_path: Union[str, Path],
    **kwargs,
) -> MementoPlanner:
    """
    Create a MEMENTO planner from saved memory.

    Args:
        memory_path: Path to user profile memory
        **kwargs: Additional arguments for MementoPlanner

    Returns:
        Configured MementoPlanner
    """
    memory = UserProfileMemory.load(memory_path)
    retriever = MementoRetriever(memory=memory)
    builder = KnowledgeGraphBuilder(memory=memory)

    return MementoPlanner(
        memory=memory,
        retriever=retriever,
        builder=builder,
        **kwargs,
    )


def create_rag_integration(
    memory_path: Union[str, Path],
    **kwargs,
) -> MementoRAGIntegration:
    """
    Create a MEMENTO RAG integration.

    Args:
        memory_path: Path to user profile memory
        **kwargs: Additional arguments

    Returns:
        Configured MementoRAGIntegration
    """
    return MementoRAGIntegration(memory_path=memory_path, **kwargs)
