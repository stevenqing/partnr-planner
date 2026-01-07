#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theory of Mind (ToM) Module

This module implements Theory of Mind capabilities for multi-agent coordination.
The framework implicitly cultivates ToM capabilities through:

1. Agent observes partner action a_partner
2. Queries library for skills whose precondition matches observation
3. Infers partner intentions from behavioral cues
4. Selects complementary response

Reference: /doc/our_method Section 3 (Cooperation Skills)
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PartnerObservation:
    """
    Observation of partner's action and state
    context_partner = (a_partner, w_partner, traj_partner)
    """
    action: str  # a_partner: Observed partner action
    state: Dict[str, Any]  # w_partner: Partner state
    trajectory: List[str]  # traj_partner: Recent movement trajectory
    location: Optional[str] = None
    held_object: Optional[str] = None


@dataclass
class InferredIntention:
    """Inferred intention from partner observation"""
    intention: str
    confidence: float
    evidence: List[str]
    suggested_response: str


@dataclass
class ToMReasoning:
    """
    Theory of Mind reasoning result

    Contains the full ToM reasoning process:
    1. Belief Formation: Understanding partner's knowledge
    2. Hypothesis Generation: Inferring goals/strategy
    3. Prediction & Planning: Coordinating next actions
    """
    belief_formation: Dict[str, Any]
    hypothesis: str
    prediction: str
    recommended_action: str
    cooperation_pattern: str
    confidence: float


class TheoryOfMindReasoner:
    """
    Theory of Mind Reasoner for Multi-Agent Coordination

    Enables agents to:
    - Infer partner intentions from behavioral cues
    - Select complementary responses
    - Coordinate through effect-based observation
    """

    def __init__(self):
        # Intention mapping patterns from observed behaviors
        self.intention_patterns = {
            "navigate_to_object": {
                "indicators": ["Navigate"],
                "intentions": [
                    ("may_pick_object", 0.8),
                    ("exploring_area", 0.5),
                    ("preparing_for_task", 0.6)
                ]
            },
            "navigate_to_furniture": {
                "indicators": ["Navigate"],
                "intentions": [
                    ("may_place_object", 0.7),
                    ("may_open_furniture", 0.6),
                    ("preparing_workspace", 0.5)
                ]
            },
            "picking_object": {
                "indicators": ["Pick"],
                "intentions": [
                    ("will_move_object", 0.9),
                    ("collecting_items", 0.7),
                    ("needs_object_for_task", 0.8)
                ]
            },
            "placing_object": {
                "indicators": ["Place"],
                "intentions": [
                    ("completed_subtask", 0.9),
                    ("organizing_objects", 0.7),
                    ("preparing_for_next_step", 0.6)
                ]
            },
            "waiting": {
                "indicators": ["Wait"],
                "intentions": [
                    ("waiting_for_partner", 0.8),
                    ("blocking_avoided", 0.6),
                    ("synchronizing", 0.7)
                ]
            },
            "cleaning": {
                "indicators": ["Clean"],
                "intentions": [
                    ("completing_clean_task", 0.9),
                    ("maintenance_subtask", 0.7)
                ]
            },
            "filling": {
                "indicators": ["Fill"],
                "intentions": [
                    ("preparing_container", 0.9),
                    ("water_task_subtask", 0.8)
                ]
            }
        }

        # Complementary action mappings
        self.complementary_actions = {
            "may_pick_object": ["wait_or_different_object", "prepare_destination"],
            "may_place_object": ["wait_nearby", "prepare_next_subtask"],
            "will_move_object": ["clear_path", "prepare_destination"],
            "completed_subtask": ["start_next_subtask", "complementary_work"],
            "waiting_for_partner": ["complete_current_task", "signal_completion"],
            "needs_assistance": ["provide_assistance", "open_furniture"],
        }

    def observe_partner(
        self,
        action_line: str,
        env_state: Dict,
        agent_id: int = 1
    ) -> PartnerObservation:
        """
        Create observation from partner's action

        Args:
            action_line: Raw action line from trace
            env_state: Current environment state
            agent_id: Partner's agent ID

        Returns:
            PartnerObservation object
        """
        # Parse action
        action_match = re.match(r"(\w+)\[([^\]]*)\]", action_line)
        if action_match:
            action_type = action_match.group(1)
            args = [a.strip() for a in action_match.group(2).split(",")] if action_match.group(2) else []
        else:
            action_type = action_line.strip()
            args = []

        # Determine partner state from action
        state = {}
        location = None
        held_object = None

        if action_type == "Navigate" and args:
            target = args[0]
            # Check if target is object or furniture
            if target in env_state.get("objects", {}):
                state["target_type"] = "object"
                state["target_object"] = target
                location = env_state["objects"][target].get("location")
            elif target in env_state.get("furniture", {}):
                state["target_type"] = "furniture"
                state["target_furniture"] = target
                location = target
        elif action_type == "Pick" and args:
            held_object = args[0]
            state["picking"] = args[0]
        elif action_type == "Place" and args:
            state["placing"] = args[0]
            if len(args) > 2:
                state["destination"] = args[2]
        elif action_type == "Wait":
            state["waiting"] = True

        return PartnerObservation(
            action=action_type,
            state=state,
            trajectory=[action_line],
            location=location,
            held_object=held_object
        )

    def infer_intention(
        self,
        observation: PartnerObservation
    ) -> List[InferredIntention]:
        """
        Infer partner intentions from observation

        Uses intention mapping patterns to infer likely goals:
        | Observed Behavior | Inferred Intention |
        |-------------------|-------------------|
        | Partner navigating toward heavy object | May need assistance moving it |
        | Partner waiting near closed door | May need door opened |

        Args:
            observation: Partner observation

        Returns:
            List of inferred intentions with confidence scores
        """
        intentions = []

        action = observation.action
        state = observation.state

        # Match against intention patterns
        for pattern_name, pattern in self.intention_patterns.items():
            if action in pattern["indicators"]:
                for intention, base_confidence in pattern["intentions"]:
                    # Adjust confidence based on context
                    confidence = base_confidence

                    # Higher confidence if we know the target
                    if state.get("target_object") or state.get("target_furniture"):
                        confidence = min(1.0, confidence + 0.1)

                    # Generate evidence
                    evidence = [f"Action: {action}"]
                    if state.get("target_object"):
                        evidence.append(f"Target object: {state['target_object']}")
                    if state.get("target_furniture"):
                        evidence.append(f"Target furniture: {state['target_furniture']}")
                    if state.get("waiting"):
                        evidence.append("Partner is waiting")

                    # Suggest complementary response
                    responses = self.complementary_actions.get(intention, ["continue_task"])
                    suggested = responses[0] if responses else "continue_task"

                    intentions.append(InferredIntention(
                        intention=intention,
                        confidence=confidence,
                        evidence=evidence,
                        suggested_response=suggested
                    ))

        # Sort by confidence
        intentions.sort(key=lambda x: x.confidence, reverse=True)

        return intentions

    def reason(
        self,
        observation: PartnerObservation,
        current_task: str,
        agent_state: Dict
    ) -> ToMReasoning:
        """
        Full Theory of Mind reasoning process

        Three-step process:
        1. Belief Formation: Analyze state changes to understand partner's knowledge
        2. Hypothesis Generation: Predict partner's goals/strategy
        3. Prediction & Planning: Plan coordinated action

        Args:
            observation: Partner observation
            current_task: Current task instruction
            agent_state: Current agent state

        Returns:
            ToMReasoning with complete analysis
        """
        # Step 1: Belief Formation
        belief_formation = self._form_beliefs(observation, current_task)

        # Step 2: Hypothesis Generation
        intentions = self.infer_intention(observation)
        top_intention = intentions[0] if intentions else None

        hypothesis = "Unknown partner goal"
        if top_intention:
            hypothesis = f"Partner likely intends to: {top_intention.intention}"

        # Step 3: Prediction & Planning
        prediction = self._predict_partner_plan(observation, top_intention)
        recommended_action = self._select_complementary_action(
            observation, top_intention, agent_state, current_task
        )

        # Determine cooperation pattern
        cooperation_pattern = self._determine_cooperation_pattern(
            observation, top_intention, agent_state
        )

        confidence = top_intention.confidence if top_intention else 0.3

        return ToMReasoning(
            belief_formation=belief_formation,
            hypothesis=hypothesis,
            prediction=prediction,
            recommended_action=recommended_action,
            cooperation_pattern=cooperation_pattern,
            confidence=confidence
        )

    def _form_beliefs(
        self,
        observation: PartnerObservation,
        current_task: str
    ) -> Dict[str, Any]:
        """
        Step 1: Belief Formation

        Analyze state changes to understand what partner knows about the world
        """
        beliefs = {
            "partner_action": observation.action,
            "partner_location": observation.location,
            "partner_holding": observation.held_object,
            "partner_knows_task": True,  # Assume shared task knowledge
            "partner_state": observation.state
        }

        # Infer what partner knows about objects
        if observation.state.get("target_object"):
            beliefs["partner_found_object"] = observation.state["target_object"]

        if observation.state.get("target_furniture"):
            beliefs["partner_targeting_furniture"] = observation.state["target_furniture"]

        return beliefs

    def _predict_partner_plan(
        self,
        observation: PartnerObservation,
        intention: Optional[InferredIntention]
    ) -> str:
        """
        Step 3a: Predict partner's next actions
        """
        if not intention:
            return "Unable to predict partner plan"

        action = observation.action

        if action == "Navigate":
            if observation.state.get("target_object"):
                return f"Partner will likely pick up {observation.state['target_object']}"
            elif observation.state.get("target_furniture"):
                return f"Partner will interact with {observation.state['target_furniture']}"

        elif action == "Pick":
            if observation.state.get("picking"):
                return f"Partner will move {observation.state['picking']} to destination"

        elif action == "Place":
            return "Partner completed placement, will start next subtask"

        elif action == "Wait":
            return "Partner is synchronizing, waiting for completion"

        return f"Partner is executing: {intention.intention}"

    def _select_complementary_action(
        self,
        observation: PartnerObservation,
        intention: Optional[InferredIntention],
        agent_state: Dict,
        current_task: str
    ) -> str:
        """
        Step 3b: Select complementary action for coordination
        """
        if not intention:
            return "Continue with current subtask"

        suggested = intention.suggested_response

        # Map suggested response to concrete action
        if suggested == "wait_or_different_object":
            if agent_state.get("holding"):
                return "Complete current move, then find different object"
            else:
                return "Navigate to different object to work in parallel"

        elif suggested == "prepare_destination":
            return "Navigate to destination and prepare for placement"

        elif suggested == "wait_nearby":
            return "Wait for partner to complete placement"

        elif suggested == "prepare_next_subtask":
            return "Start next subtask while partner completes current one"

        elif suggested == "complete_current_task":
            return "Complete current subtask, partner is waiting"

        elif suggested == "complementary_work":
            return "Work on different objects/subtasks in parallel"

        elif suggested == "provide_assistance":
            return "Navigate to partner location to assist"

        elif suggested == "open_furniture":
            return "Open furniture for partner access"

        return "Continue with planned action"

    def _determine_cooperation_pattern(
        self,
        observation: PartnerObservation,
        intention: Optional[InferredIntention],
        agent_state: Dict
    ) -> str:
        """
        Determine the cooperation pattern being used
        """
        if observation.action == "Wait":
            return "synchronization"

        if intention:
            if "completed_subtask" in intention.intention:
                return "sequential_handoff"
            if "different_object" in intention.suggested_response:
                return "complementary_work"
            if "assistance" in intention.intention:
                return "assistance"

        # Check if agents working on different things
        if agent_state.get("target_object") and observation.state.get("target_object"):
            if agent_state["target_object"] != observation.state["target_object"]:
                return "complementary_work"

        return "division_of_labor"

    def extract_tom_from_trace(
        self,
        trace_content: str,
        agent_id: int = 0
    ) -> List[ToMReasoning]:
        """
        Extract ToM reasoning from a complete trace

        Args:
            trace_content: Full trace content
            agent_id: Agent doing the reasoning (observing partner)

        Returns:
            List of ToM reasoning for each observation point
        """
        partner_id = 1 - agent_id
        tom_reasoning = []

        lines = trace_content.split('\n')

        # Extract task
        task = ""
        env_state = {"objects": {}, "furniture": {}}
        agent_state = {"holding": None, "target_object": None}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Task:"):
                task = line.replace("Task:", "").strip()

            # Parse environment (simplified)
            if "is in/on" in line:
                match = re.match(r"(\w+) is in/on (\w+)", line)
                if match:
                    env_state["objects"][match.group(1)] = {"location": match.group(2)}

            # Parse partner actions
            if f"Agent_{partner_id}_Action:" in line:
                action_str = line.replace(f"Agent_{partner_id}_Action:", "").strip()

                # Create observation
                obs = self.observe_partner(action_str, env_state, partner_id)

                # Perform ToM reasoning
                reasoning = self.reason(obs, task, agent_state)
                tom_reasoning.append(reasoning)

            # Update agent state from own actions
            if f"Agent_{agent_id}_Action:" in line:
                action_str = line.replace(f"Agent_{agent_id}_Action:", "").strip()
                action_match = re.match(r"(\w+)\[([^\]]*)\]", action_str)
                if action_match:
                    action_type = action_match.group(1)
                    args = [a.strip() for a in action_match.group(2).split(",")] if action_match.group(2) else []

                    if action_type == "Pick" and args:
                        agent_state["holding"] = args[0]
                    elif action_type == "Place":
                        agent_state["holding"] = None
                    elif action_type == "Navigate" and args:
                        agent_state["target_object"] = args[0]

        return tom_reasoning


def create_tom_reasoner() -> TheoryOfMindReasoner:
    """Factory function to create ToM reasoner"""
    return TheoryOfMindReasoner()


if __name__ == "__main__":
    # Test ToM reasoning
    reasoner = create_tom_reasoner()

    # Test observation
    obs = PartnerObservation(
        action="Navigate",
        state={"target_object": "cup_0"},
        trajectory=["Navigate[cup_0]"],
        location="table_1",
        held_object=None
    )

    intentions = reasoner.infer_intention(obs)
    print("Inferred Intentions:")
    for intent in intentions:
        print(f"  - {intent.intention} (confidence: {intent.confidence:.2f})")
        print(f"    Evidence: {intent.evidence}")
        print(f"    Suggested: {intent.suggested_response}")

    # Test full reasoning
    agent_state = {"holding": None, "target_object": None}
    reasoning = reasoner.reason(obs, "Move the cup to the kitchen", agent_state)

    print("\nToM Reasoning:")
    print(f"  Beliefs: {reasoning.belief_formation}")
    print(f"  Hypothesis: {reasoning.hypothesis}")
    print(f"  Prediction: {reasoning.prediction}")
    print(f"  Recommended Action: {reasoning.recommended_action}")
    print(f"  Cooperation Pattern: {reasoning.cooperation_pattern}")
    print(f"  Confidence: {reasoning.confidence:.2f}")
