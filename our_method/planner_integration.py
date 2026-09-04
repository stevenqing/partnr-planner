#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Planner Integration Module

Integrates the Hierarchical Skill Memory with the existing LLM planner infrastructure.
This module provides the bridge between our method and habitat_llm/planner/.

Reference: /doc/our_method
"""

import os
from typing import Any, Dict, List, Optional

from .hierarchical_retrieval import HierarchicalRetriever, RetrievedSkill
from .theory_of_mind import TheoryOfMindReasoner, ToMReasoning


class HierarchicalSkillPlanner:
    """
    Planner that uses Hierarchical Skill Memory for decision making

    Implements the Memory Utilization Stage:
    1. PERCEPTION: Observe environmental change from partner action
    2. DECISION: Hierarchical retrieval
    3. ACTION: Synthesize and execute
    """

    def __init__(
        self,
        memory_path: str,
        enable_tom: bool = True,
        abstract_threshold: float = 0.3,
        instance_top_k: int = 5
    ):
        """
        Initialize the planner

        Args:
            memory_path: Path to hierarchical skill memory
            enable_tom: Enable Theory of Mind reasoning
            abstract_threshold: Threshold for abstract skill matching
            instance_top_k: Number of instances to retrieve per skill
        """
        self.memory_path = memory_path
        self.enable_tom = enable_tom

        # Initialize retriever
        self.retriever = HierarchicalRetriever(
            memory_path=memory_path,
            abstract_threshold=abstract_threshold,
            instance_top_k=instance_top_k
        )

        # Initialize ToM reasoner if enabled
        self.tom_reasoner = TheoryOfMindReasoner() if enable_tom else None

        # State tracking
        self.current_agent_state = {"holding": None, "position": None}
        self.environment_state = {}
        self.partner_effects = {}
        self.last_partner_action = None

    def perceive(
        self,
        world_state: Dict[str, Any],
        partner_action: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        PERCEPTION Stage: Observe environmental change from partner action

        delta_w_t = w_{t+1} - w_t

        Args:
            world_state: Current world state observation
            partner_action: Partner's last action (if observed)

        Returns:
            Perceived effects (delta_w)
        """
        effects = {}

        # Calculate state change
        if self.environment_state:
            # Object position changes
            moved_objects = []
            for obj_name, obj_state in world_state.get("objects", {}).items():
                old_state = self.environment_state.get("objects", {}).get(obj_name, {})
                if old_state.get("location") != obj_state.get("location"):
                    moved_objects.append(obj_name)

            effects["moved_objects"] = moved_objects

            # State changes (clean, filled, powered)
            state_changes = []
            for obj_name, obj_state in world_state.get("objects", {}).items():
                old_state = self.environment_state.get("objects", {}).get(obj_name, {})
                for state_key, state_val in obj_state.get("states", {}).items():
                    if old_state.get("states", {}).get(state_key) != state_val:
                        state_changes.append(f"{obj_name}_{state_key}_{state_val}")

            effects["state_changes"] = state_changes

        # Partner action observation
        if partner_action:
            effects["action"] = partner_action
            self.last_partner_action = partner_action

        # Update stored state
        self.environment_state = world_state.copy()
        self.partner_effects = effects

        return effects

    def decide(
        self,
        goal: str,
        include_individual: bool = True,
        include_cooperation: bool = True
    ) -> List[RetrievedSkill]:
        """
        DECISION Stage: Hierarchical retrieval

        1. Generate query from current state + observed effects
        2. Match against abstract skill categories
        3. Retrieve relevant instances
        4. Filter for executability

        Args:
            goal: Current goal/task instruction
            include_individual: Include individual skills
            include_cooperation: Include cooperation skills

        Returns:
            List of executable skills sorted by relevance
        """
        # Perform hierarchical retrieval
        retrieved = self.retriever.retrieve(
            agent_state=self.current_agent_state,
            environment_state=self.environment_state,
            partner_effects=self.partner_effects,
            goal=goal,
            include_individual=include_individual,
            include_cooperation=include_cooperation
        )

        return retrieved

    def reason_about_partner(self) -> Optional[ToMReasoning]:
        """
        Perform Theory of Mind reasoning about partner

        Returns:
            ToMReasoning if partner action observed, None otherwise
        """
        if not self.enable_tom or not self.tom_reasoner:
            return None

        if not self.last_partner_action:
            return None

        # Create partner observation
        from .theory_of_mind import PartnerObservation
        obs = self.tom_reasoner.observe_partner(
            self.last_partner_action,
            self.environment_state
        )

        # Perform ToM reasoning
        reasoning = self.tom_reasoner.reason(
            obs,
            current_task="",  # Will be provided by caller
            agent_state=self.current_agent_state
        )

        return reasoning

    def act(
        self,
        retrieved_skills: List[RetrievedSkill],
        tom_reasoning: Optional[ToMReasoning] = None
    ) -> Dict[str, Any]:
        """
        ACTION Stage: Synthesize retrieved memories + current state

        Select complementary action grounded in task progress

        Args:
            retrieved_skills: Skills retrieved in decision stage
            tom_reasoning: ToM reasoning result (if available)

        Returns:
            Action recommendation with context
        """
        if not retrieved_skills:
            return {
                "action": "Wait",
                "reasoning": "No applicable skills found",
                "skill_used": None,
                "cooperation_mode": False
            }

        # Select top skill
        top_skill = retrieved_skills[0]

        # Build action recommendation
        action_info = {
            "skill_name": top_skill.skill_name,
            "skill_type": top_skill.skill_type,
            "demo": top_skill.demo,
            "confidence": top_skill.abstract_score,
            "cooperation_mode": top_skill.skill_type == "cooperation"
        }

        # Add ToM context if available
        if tom_reasoning:
            action_info["tom_context"] = {
                "hypothesis": tom_reasoning.hypothesis,
                "recommended_action": tom_reasoning.recommended_action,
                "cooperation_pattern": tom_reasoning.cooperation_pattern
            }

        return action_info

    def step(
        self,
        world_state: Dict[str, Any],
        goal: str,
        partner_action: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete perception-decision-action loop

        Args:
            world_state: Current world state
            goal: Current goal
            partner_action: Partner's last action

        Returns:
            Action recommendation
        """
        # PERCEPTION
        effects = self.perceive(world_state, partner_action)

        # DECISION
        retrieved = self.decide(goal)

        # ToM reasoning
        tom = self.reason_about_partner()

        # ACTION
        action = self.act(retrieved, tom)

        return action

    def format_for_prompt(
        self,
        retrieved_skills: List[RetrievedSkill],
        tom_reasoning: Optional[ToMReasoning] = None,
        max_examples: int = 3
    ) -> str:
        """
        Format retrieved skills and ToM for LLM prompt

        Args:
            retrieved_skills: Retrieved skills
            tom_reasoning: ToM reasoning result
            max_examples: Max examples to include

        Returns:
            Formatted string for prompt insertion
        """
        parts = []

        # Add ToM section if available
        if tom_reasoning:
            parts.append("## Theory of Mind Analysis\n")
            parts.append(f"**Partner Hypothesis:** {tom_reasoning.hypothesis}")
            parts.append(f"**Predicted Action:** {tom_reasoning.prediction}")
            parts.append(f"**Recommended Response:** {tom_reasoning.recommended_action}")
            parts.append(f"**Cooperation Pattern:** {tom_reasoning.cooperation_pattern}")
            parts.append("")

        # Add retrieved skills
        parts.append(self.retriever.format_for_prompt(retrieved_skills, max_examples))

        return "\n".join(parts)

    def update_agent_state(
        self,
        holding: Optional[str] = None,
        position: Optional[str] = None
    ):
        """Update the agent's internal state"""
        if holding is not None:
            self.current_agent_state["holding"] = holding
        if position is not None:
            self.current_agent_state["position"] = position


class RAGIntegration:
    """
    Integration with existing RAG system in habitat_llm/planner/rag.py

    Extends the RAG class to use hierarchical skill memory
    """

    def __init__(
        self,
        memory_path: str,
        rag_data_dir: Optional[str] = None
    ):
        """
        Initialize RAG integration

        Args:
            memory_path: Path to hierarchical skill memory
            rag_data_dir: Optional path to existing RAG dataset
        """
        self.memory_path = memory_path
        self.rag_data_dir = rag_data_dir

        # Initialize components
        self.skill_planner = HierarchicalSkillPlanner(memory_path)
        self.skill_retriever = self.skill_planner.retriever

    def get_examples(
        self,
        instruction: str,
        world_state: Dict[str, Any],
        partner_action: Optional[str] = None,
        num_examples: int = 3
    ) -> str:
        """
        Get examples for RAG prompt

        Combines hierarchical skill retrieval with ToM

        Args:
            instruction: Task instruction
            world_state: Current world state
            partner_action: Partner's last action
            num_examples: Number of examples to return

        Returns:
            Formatted examples string for prompt
        """
        # Update planner state
        self.skill_planner.perceive(world_state, partner_action)

        # Retrieve skills
        retrieved = self.skill_planner.decide(instruction)

        # ToM reasoning
        tom = self.skill_planner.reason_about_partner()

        # Format for prompt
        examples = self.skill_planner.format_for_prompt(
            retrieved,
            tom,
            max_examples=num_examples
        )

        return examples

    def get_skill_guidance(
        self,
        instruction: str,
        world_state: Dict[str, Any],
        partner_action: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get skill-based guidance for action selection

        Args:
            instruction: Task instruction
            world_state: Current world state
            partner_action: Partner's last action

        Returns:
            Action recommendation with reasoning
        """
        result = self.skill_planner.step(world_state, instruction, partner_action)
        return result


def create_planner(memory_path: str) -> HierarchicalSkillPlanner:
    """Factory function to create planner"""
    return HierarchicalSkillPlanner(memory_path)


def create_rag_integration(
    memory_path: str,
    rag_data_dir: Optional[str] = None
) -> RAGIntegration:
    """Factory function to create RAG integration"""
    return RAGIntegration(memory_path, rag_data_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test planner integration")
    parser.add_argument(
        "--memory-path",
        type=str,
        default="/home/a5l/shuqing.a5l/partnr-planner/data/hierarchical_skill_memory",
        help="Path to hierarchical skill memory"
    )

    args = parser.parse_args()

    # Create planner
    planner = create_planner(args.memory_path)

    # Test step
    test_world_state = {
        "objects": {
            "cup_0": {"location": "table_1", "states": {"is_clean": False}},
            "plate_0": {"location": "counter_1", "states": {"is_clean": True}}
        },
        "furniture": {
            "table_1": "living_room",
            "counter_1": "kitchen"
        }
    }

    result = planner.step(
        world_state=test_world_state,
        goal="Move the cup to the kitchen counter and clean it",
        partner_action="Navigate[plate_0]"
    )

    print("Action Recommendation:")
    print(f"  Skill: {result.get('skill_name', 'N/A')}")
    print(f"  Type: {result.get('skill_type', 'N/A')}")
    print(f"  Demo: {result.get('demo', 'N/A')}")
    print(f"  Cooperation Mode: {result.get('cooperation_mode', False)}")

    if "tom_context" in result:
        print("\nToM Context:")
        print(f"  Hypothesis: {result['tom_context']['hypothesis']}")
        print(f"  Recommended: {result['tom_context']['recommended_action']}")
        print(f"  Pattern: {result['tom_context']['cooperation_pattern']}")
