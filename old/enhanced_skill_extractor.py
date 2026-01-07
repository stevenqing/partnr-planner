#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ActionStep:
    """Represents a single action step in the trajectory"""

    action_type: str
    action_content: str
    observation: str
    step_number: int
    success: bool
    objects_involved: List[str]
    locations_mentioned: List[str]


@dataclass
class DecisionPoint:
    """Represents a critical decision point in the trajectory"""

    step_number: int
    decision_type: str  # exploration, manipulation, coordination, recovery
    context: str
    alternatives: List[str]
    chosen_action: str
    outcome: str
    reasoning: str


@dataclass
class SkillPattern:
    """Represents an identified skill pattern"""

    skill_name: str
    skill_type: str  # navigation, manipulation, coordination, planning
    description: str
    action_sequence: List[str]
    success_indicators: List[str]
    failure_indicators: List[str]
    contextual_conditions: Dict[str, str]


class EnhancedSkillExtractor:
    """Enhanced skill extraction with multi-stage analysis"""

    def __init__(self, use_llm: bool = True, cache_results: bool = True):
        self.use_llm = use_llm
        self.cache_results = cache_results
        self.skill_templates = self._load_skill_templates()
        self.action_patterns = self._load_action_patterns()
        self._trace_cache = {} if cache_results else None

    def extract_enhanced_skills(
        self, trace_content: str, agent_id: str, instruction: str
    ) -> Dict:
        """Main method for enhanced skill extraction with caching"""

        # Check cache first
        if self.cache_results and self._trace_cache is not None:
            cache_key = f"{hash(trace_content)}_{agent_id}_{hash(instruction)}"
            if cache_key in self._trace_cache:
                return self._trace_cache[cache_key]

        # Stage 1: Parse and structure the trace
        action_steps = self._parse_trace_to_steps(trace_content, agent_id)

        # Early exit for empty traces
        if not action_steps:
            return self._create_empty_result()

        # Stage 2: Identify decision points (optimized)
        decision_points = self._identify_decision_points_fast(action_steps, instruction)

        # Stage 3: Extract skill patterns (optimized)
        skill_patterns = self._extract_skill_patterns_fast(
            action_steps, decision_points
        )

        # Stage 4: Generate contextual skills (optimized)
        contextual_skills = self._generate_contextual_skills_fast(
            action_steps, decision_points, skill_patterns, instruction, agent_id
        )

        # Stage 5: Analyze coordination requirements (optimized)
        coordination_analysis = self._analyze_coordination_requirements_fast(
            action_steps, agent_id, instruction
        )

        result = {
            "enhanced_skill_description": contextual_skills["primary_description"],
            "skill_patterns": [pattern.__dict__ for pattern in skill_patterns],
            "decision_points": [point.__dict__ for point in decision_points],
            "coordination_requirements": coordination_analysis,
            "action_efficiency": self._calculate_action_efficiency_fast(action_steps),
            "skill_categories": contextual_skills["categories"],
            "learning_insights": contextual_skills["insights"],
        }

        # Cache result
        if self.cache_results and self._trace_cache is not None:
            self._trace_cache[cache_key] = result

        return result

    def _parse_trace_to_steps(
        self, trace_content: str, agent_id: str
    ) -> List[ActionStep]:
        """Parse trace content into structured action steps"""
        lines = trace_content.split("\n")
        steps = []
        current_step = 0
        current_action = None
        current_observation = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith("==="):
                continue

            if line.startswith(f"Agent_{agent_id}_Action:"):
                if current_action is not None:
                    # Process previous step
                    step = self._create_action_step(
                        current_action, current_observation, current_step
                    )
                    if step:
                        steps.append(step)
                    current_step += 1

                current_action = line.replace(f"Agent_{agent_id}_Action:", "").strip()
                current_observation = None

            elif line.startswith(f"Agent_{agent_id}_Observation:"):
                current_observation = line.replace(
                    f"Agent_{agent_id}_Observation:", ""
                ).strip()

        # Process final step
        if current_action is not None:
            step = self._create_action_step(
                current_action, current_observation, current_step
            )
            if step:
                steps.append(step)

        return steps

    def _create_action_step(
        self, action: str, observation: str, step_num: int
    ) -> Optional[ActionStep]:
        """Create an ActionStep from action and observation"""
        if not action:
            return None

        action_type = self._extract_action_type(action)
        objects = self._extract_objects(action, observation or "")
        locations = self._extract_locations(action, observation or "")
        success = self._determine_step_success(observation or "")

        return ActionStep(
            action_type=action_type,
            action_content=action,
            observation=observation or "",
            step_number=step_num,
            success=success,
            objects_involved=objects,
            locations_mentioned=locations,
        )

    def _extract_action_type(self, action: str) -> str:
        """Extract the type of action"""
        action_lower = action.lower()

        if any(word in action_lower for word in ["explore", "search", "look", "scan"]):
            return "exploration"
        elif any(
            word in action_lower for word in ["navigate", "move to", "go to", "walk"]
        ):
            return "navigation"
        elif any(word in action_lower for word in ["pick", "grab", "take", "grasp"]):
            return "manipulation_pick"
        elif any(word in action_lower for word in ["place", "put", "drop", "set"]):
            return "manipulation_place"
        elif any(word in action_lower for word in ["open", "close"]):
            return "container_interaction"
        elif any(word in action_lower for word in ["wait", "pause"]):
            return "coordination"
        elif "done" in action_lower:
            return "task_completion"
        else:
            return "other"

    def _extract_objects(self, action: str, observation: str) -> List[str]:
        """Extract objects mentioned in action and observation"""
        text = f"{action} {observation}".lower()

        # Common household objects
        object_patterns = [
            r"\b(apple|banana|orange|fruit)\b",
            r"\b(plate|bowl|cup|glass|dish)\b",
            r"\b(book|magazine|newspaper)\b",
            r"\b(toy|ball|doll)\b",
            r"\b(vase|candle|lamp)\b",
            r"\b(pillow|blanket|cushion)\b",
            r"\b(bottle|container|box)\b",
            r"\b(remote|phone|device)\b",
        ]

        objects = set()
        for pattern in object_patterns:
            matches = re.findall(pattern, text)
            objects.update(matches)

        return list(objects)

    def _extract_locations(self, action: str, observation: str) -> List[str]:
        """Extract locations mentioned in action and observation"""
        text = f"{action} {observation}".lower()

        location_patterns = [
            r"\b(kitchen|bedroom|living room|bathroom|dining room|hallway)\b",
            r"\b(table|counter|bed|couch|chair|desk|shelf)\b",
            r"\b(cabinet|drawer|closet|pantry)\b",
            r"\b(floor|wall|ceiling)\b",
        ]

        locations = set()
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            locations.update(matches)

        return list(locations)

    def _determine_step_success(self, observation: str) -> bool:
        """Determine if the step was successful based on observation"""
        if not observation:
            return True  # Assume success if no observation

        obs_lower = observation.lower()

        # Failure indicators
        failure_indicators = [
            "failed",
            "error",
            "cannot",
            "unable",
            "not found",
            "blocked",
            "occupied",
            "collision",
            "invalid",
        ]

        # Success indicators
        success_indicators = [
            "successfully",
            "completed",
            "found",
            "picked up",
            "placed",
            "moved",
            "opened",
            "closed",
        ]

        if any(indicator in obs_lower for indicator in failure_indicators):
            return False
        elif any(indicator in obs_lower for indicator in success_indicators):
            return True
        else:
            return True  # Assume success if ambiguous

    def _identify_decision_points(
        self, steps: List[ActionStep], instruction: str
    ) -> List[DecisionPoint]:
        """Identify critical decision points in the trajectory"""
        decision_points = []

        for i, step in enumerate(steps):
            decision_point = None

            # Exploration decisions
            if step.action_type == "exploration" and i > 0:
                prev_step = steps[i - 1]
                if not prev_step.success or "not found" in step.observation.lower():
                    decision_point = DecisionPoint(
                        step_number=i,
                        decision_type="exploration",
                        context="Previous exploration failed, deciding where to search next",
                        alternatives=[
                            "continue current room",
                            "try different room",
                            "ask other agent",
                        ],
                        chosen_action=step.action_content,
                        outcome="success" if step.success else "failure",
                        reasoning=f"Agent chose to explore after {prev_step.action_type} failed",
                    )

            # Recovery decisions
            elif not step.success and i < len(steps) - 1:
                next_step = steps[i + 1]
                decision_point = DecisionPoint(
                    step_number=i,
                    decision_type="recovery",
                    context=f"Action failed: {step.observation}",
                    alternatives=[
                        "retry same action",
                        "try different approach",
                        "explore more",
                    ],
                    chosen_action=next_step.action_content,
                    outcome="recovery_attempt",
                    reasoning=f"After failure, agent decided to {next_step.action_type}",
                )

            # Coordination decisions
            elif (
                step.action_type == "coordination"
                or "wait" in step.action_content.lower()
            ):
                decision_point = DecisionPoint(
                    step_number=i,
                    decision_type="coordination",
                    context="Multi-agent coordination required",
                    alternatives=[
                        "wait for other agent",
                        "proceed independently",
                        "communicate",
                    ],
                    chosen_action=step.action_content,
                    outcome="coordination",
                    reasoning="Agent chose to coordinate with other agent",
                )

            if decision_point:
                decision_points.append(decision_point)

        return decision_points

    def _extract_skill_patterns(
        self, steps: List[ActionStep], decision_points: List[DecisionPoint]
    ) -> List[SkillPattern]:
        """Extract skill patterns from the action sequence"""
        patterns = []

        # Pattern 1: Efficient object location
        location_pattern = self._identify_location_pattern(steps)
        if location_pattern:
            patterns.append(location_pattern)

        # Pattern 2: Object manipulation sequence
        manipulation_pattern = self._identify_manipulation_pattern(steps)
        if manipulation_pattern:
            patterns.append(manipulation_pattern)

        # Pattern 3: Recovery strategies
        recovery_patterns = self._identify_recovery_patterns(steps, decision_points)
        patterns.extend(recovery_patterns)

        # Pattern 4: Coordination strategies
        coordination_patterns = self._identify_coordination_patterns(
            steps, decision_points
        )
        patterns.extend(coordination_patterns)

        return patterns

    def _identify_location_pattern(
        self, steps: List[ActionStep]
    ) -> Optional[SkillPattern]:
        """Identify object location strategies"""
        exploration_steps = [s for s in steps if s.action_type == "exploration"]

        if len(exploration_steps) < 2:
            return None

        # Analyze exploration sequence
        rooms_explored = []
        for step in exploration_steps:
            locations = step.locations_mentioned
            if locations:
                rooms_explored.extend(locations)

        if len(set(rooms_explored)) > 1:  # Multiple rooms explored
            return SkillPattern(
                skill_name="systematic_exploration",
                skill_type="navigation",
                description=f"Systematically explore multiple locations: {' -> '.join(rooms_explored[:3])}",
                action_sequence=[s.action_content for s in exploration_steps[:3]],
                success_indicators=["object found", "systematic coverage"],
                failure_indicators=["repeated exploration", "random searching"],
                contextual_conditions={
                    "rooms_explored": str(len(set(rooms_explored))),
                    "exploration_efficiency": "high"
                    if len(exploration_steps) <= 3
                    else "medium",
                },
            )

        return None

    def _identify_manipulation_pattern(
        self, steps: List[ActionStep]
    ) -> Optional[SkillPattern]:
        """Identify object manipulation patterns"""
        manipulation_steps = [
            s
            for s in steps
            if s.action_type in ["manipulation_pick", "manipulation_place"]
        ]

        if len(manipulation_steps) < 2:
            return None

        # Analyze manipulation sequence
        pick_steps = [
            s for s in manipulation_steps if s.action_type == "manipulation_pick"
        ]
        place_steps = [
            s for s in manipulation_steps if s.action_type == "manipulation_place"
        ]

        if pick_steps and place_steps:
            success_rate = sum(1 for s in manipulation_steps if s.success) / len(
                manipulation_steps
            )

            return SkillPattern(
                skill_name="pick_and_place_sequence",
                skill_type="manipulation",
                description=f"Execute {len(pick_steps)} pick and {len(place_steps)} place operations",
                action_sequence=[s.action_content for s in manipulation_steps],
                success_indicators=[
                    "smooth transitions",
                    "no drops",
                    "accurate placement",
                ],
                failure_indicators=["failed grasps", "dropped objects", "misplacement"],
                contextual_conditions={
                    "success_rate": f"{success_rate:.2f}",
                    "objects_handled": str(
                        len(
                            {
                                obj
                                for s in manipulation_steps
                                for obj in s.objects_involved
                            }
                        )
                    ),
                },
            )

        return None

    def _identify_recovery_patterns(
        self, steps: List[ActionStep], decision_points: List[DecisionPoint]
    ) -> List[SkillPattern]:
        """Identify recovery strategy patterns"""
        recovery_patterns = []
        recovery_decisions = [
            dp for dp in decision_points if dp.decision_type == "recovery"
        ]

        for recovery in recovery_decisions:
            step_idx = recovery.step_number
            if step_idx + 1 < len(steps):
                recovery_action = steps[step_idx + 1]

                pattern = SkillPattern(
                    skill_name="failure_recovery",
                    skill_type="planning",
                    description=f"Recover from {steps[step_idx].action_type} failure with {recovery_action.action_type}",
                    action_sequence=[
                        steps[step_idx].action_content,
                        recovery_action.action_content,
                    ],
                    success_indicators=["successful recovery", "alternative approach"],
                    failure_indicators=["repeated failures", "stuck in loop"],
                    contextual_conditions={
                        "failure_type": steps[step_idx].action_type,
                        "recovery_strategy": recovery_action.action_type,
                        "recovery_success": str(recovery_action.success),
                    },
                )
                recovery_patterns.append(pattern)

        return recovery_patterns

    def _identify_coordination_patterns(
        self, steps: List[ActionStep], decision_points: List[DecisionPoint]
    ) -> List[SkillPattern]:
        """Identify coordination strategy patterns"""
        coordination_patterns = []
        coordination_decisions = [
            dp for dp in decision_points if dp.decision_type == "coordination"
        ]

        for coord in coordination_decisions:
            step = steps[coord.step_number]

            pattern = SkillPattern(
                skill_name="multi_agent_coordination",
                skill_type="coordination",
                description=f"Coordinate with other agent: {coord.reasoning}",
                action_sequence=[step.action_content],
                success_indicators=["successful coordination", "no conflicts"],
                failure_indicators=["agent collision", "task interference"],
                contextual_conditions={
                    "coordination_type": coord.context,
                    "timing": f"step_{coord.step_number}",
                },
            )
            coordination_patterns.append(pattern)

        return coordination_patterns

    def _generate_contextual_skills(
        self,
        steps: List[ActionStep],
        decision_points: List[DecisionPoint],
        patterns: List[SkillPattern],
        instruction: str,
        agent_id: str,
    ) -> Dict:
        """Generate enhanced contextual skill descriptions"""

        # Analyze task complexity and requirements
        task_analysis = self._analyze_task_complexity(instruction, steps)

        # Generate primary skill description
        if self.use_llm:
            primary_description = self._generate_llm_enhanced_description(
                steps, decision_points, patterns, instruction, agent_id, task_analysis
            )
        else:
            primary_description = self._generate_heuristic_description(
                steps, patterns, instruction, agent_id, task_analysis
            )

        # Extract skill categories
        categories = self._extract_enhanced_categories(patterns, task_analysis)

        # Generate learning insights
        insights = self._generate_learning_insights(steps, decision_points, patterns)

        return {
            "primary_description": primary_description,
            "categories": categories,
            "insights": insights,
        }

    def _analyze_task_complexity(
        self, instruction: str, steps: List[ActionStep]
    ) -> Dict:
        """Analyze task complexity and requirements"""

        # Count different types of actions required
        action_types = set(step.action_type for step in steps)

        # Count objects and locations involved
        all_objects = set()
        all_locations = set()
        for step in steps:
            all_objects.update(step.objects_involved)
            all_locations.update(step.locations_mentioned)

        # Analyze instruction complexity
        instruction_words = instruction.split()
        conjunctions = sum(
            1
            for word in instruction_words
            if word.lower() in ["and", "then", "also", "after"]
        )

        return {
            "action_diversity": len(action_types),
            "objects_count": len(all_objects),
            "locations_count": len(all_locations),
            "instruction_complexity": len(instruction_words),
            "multi_step": conjunctions > 0,
            "success_rate": sum(1 for s in steps if s.success) / len(steps)
            if steps
            else 0,
        }

    def _generate_llm_enhanced_description(
        self,
        steps: List[ActionStep],
        decision_points: List[DecisionPoint],
        patterns: List[SkillPattern],
        instruction: str,
        agent_id: str,
        task_analysis: Dict,
    ) -> str:
        """Generate enhanced description using LLM"""

        # Create comprehensive context for LLM
        context_data = {
            "instruction": instruction,
            "agent_id": agent_id,
            "total_steps": len(steps),
            "decision_points_count": len(decision_points),
            "skill_patterns": [p.skill_name for p in patterns],
            "task_complexity": task_analysis,
            "key_actions": [s.action_content for s in steps if s.success][:5],
            "critical_decisions": [dp.reasoning for dp in decision_points][:3],
        }

        enhanced_prompt = f"""You are analyzing a robotic agent's successful task execution. Generate a comprehensive, specific skill description that captures the agent's capabilities and decision-making.

Task: {instruction}
Agent: {agent_id}

Execution Analysis:
- Total Steps: {context_data['total_steps']}
- Success Rate: {task_analysis['success_rate']:.2f}
- Action Diversity: {task_analysis['action_diversity']} different action types
- Objects Handled: {task_analysis['objects_count']} unique objects
- Locations Visited: {task_analysis['locations_count']} unique locations

Key Actions Performed:
{chr(10).join([f"- {action}" for action in context_data['key_actions']])}

Critical Decision Points:
{chr(10).join([f"- {decision}" for decision in context_data['critical_decisions']])}

Identified Skill Patterns:
{chr(10).join([f"- {pattern}" for pattern in context_data['skill_patterns']])}

Generate a detailed skill description that:
1. Specifies EXACT capabilities demonstrated
2. Describes decision-making strategies used
3. Explains coordination approaches if applicable
4. Highlights efficiency and success factors
5. Provides actionable insights for similar tasks

Format: "Agent demonstrates advanced [skill type] capabilities including [specific abilities]. Key strategies: [decision approaches]. Success factors: [what made it work]."

Enhanced Skill Description:"""

        try:
            result = subprocess.run(
                ["ollama", "run", "llama3.1:8b"],
                input=enhanced_prompt,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                description = result.stdout.strip()
                # Clean up response
                if description.startswith("Enhanced Skill Description:"):
                    description = description[28:].strip()
                return description
            else:
                return self._generate_heuristic_description(
                    steps, patterns, instruction, agent_id, task_analysis
                )

        except Exception:
            return self._generate_heuristic_description(
                steps, patterns, instruction, agent_id, task_analysis
            )

    def _generate_heuristic_description(
        self,
        steps: List[ActionStep],
        patterns: List[SkillPattern],
        instruction: str,
        agent_id: str,
        task_analysis: Dict,
    ) -> str:
        """Generate enhanced description using heuristics"""

        # Identify primary capabilities
        capabilities = []
        if task_analysis["action_diversity"] >= 3:
            capabilities.append("multi-modal task execution")
        if task_analysis["success_rate"] > 0.8:
            capabilities.append("high-reliability action execution")
        if task_analysis["locations_count"] > 2:
            capabilities.append("spatial reasoning and navigation")
        if task_analysis["objects_count"] > 1:
            capabilities.append("multi-object manipulation")

        # Identify key strategies
        strategies = []
        for pattern in patterns:
            if pattern.skill_type == "navigation":
                strategies.append("systematic exploration patterns")
            elif pattern.skill_type == "manipulation":
                strategies.append("efficient pick-and-place sequences")
            elif pattern.skill_type == "coordination":
                strategies.append("multi-agent coordination protocols")
            elif pattern.skill_type == "planning":
                strategies.append("adaptive recovery strategies")

        # Generate description
        capability_str = (
            ", ".join(capabilities) if capabilities else "basic task execution"
        )
        strategy_str = (
            ", ".join(strategies) if strategies else "direct action execution"
        )

        return (
            f"Agent {agent_id} demonstrates advanced {capability_str} with {task_analysis['success_rate']:.1%} "
            f"success rate. Key strategies include {strategy_str}. Successfully handled "
            f"{task_analysis['objects_count']} objects across {task_analysis['locations_count']} locations "
            f"using {task_analysis['action_diversity']} different action types to complete: {instruction}"
        )

    def _extract_enhanced_categories(
        self, patterns: List[SkillPattern], task_analysis: Dict
    ) -> List[str]:
        """Extract enhanced skill categories"""
        categories = set()

        # From patterns
        for pattern in patterns:
            if pattern.skill_type == "navigation":
                categories.add("Advanced Navigation")
            elif pattern.skill_type == "manipulation":
                categories.add("Object Manipulation Mastery")
            elif pattern.skill_type == "coordination":
                categories.add("Multi-Agent Coordination")
            elif pattern.skill_type == "planning":
                categories.add("Adaptive Planning")

        # From task analysis
        if task_analysis["multi_step"]:
            categories.add("Multi-Step Task Execution")
        if task_analysis["success_rate"] > 0.9:
            categories.add("High-Reliability Execution")
        if task_analysis["action_diversity"] >= 4:
            categories.add("Versatile Action Selection")

        return list(categories)

    def _generate_learning_insights(
        self,
        steps: List[ActionStep],
        decision_points: List[DecisionPoint],
        patterns: List[SkillPattern],
    ) -> List[str]:
        """Generate actionable learning insights"""
        insights = []

        # Efficiency insights
        total_steps = len(steps)
        successful_steps = sum(1 for s in steps if s.success)
        if successful_steps / total_steps > 0.8:
            insights.append(
                f"High efficiency: {successful_steps}/{total_steps} actions successful"
            )

        # Decision-making insights
        if decision_points:
            recovery_decisions = [
                dp for dp in decision_points if dp.decision_type == "recovery"
            ]
            if recovery_decisions:
                insights.append(
                    f"Demonstrates resilience with {len(recovery_decisions)} successful recovery strategies"
                )

        # Pattern insights
        for pattern in patterns:
            if pattern.skill_name == "systematic_exploration":
                insights.append(
                    "Uses systematic exploration rather than random searching"
                )
            elif pattern.skill_name == "pick_and_place_sequence":
                insights.append(
                    "Executes smooth manipulation sequences with minimal errors"
                )
            elif pattern.skill_name == "multi_agent_coordination":
                insights.append(
                    "Actively coordinates with other agents to prevent conflicts"
                )

        # Strategic insights
        action_types = set(step.action_type for step in steps)
        if "exploration" in action_types and "manipulation_pick" in action_types:
            insights.append(
                "Combines exploration and manipulation effectively for task completion"
            )

        return insights

    def _analyze_coordination_requirements(
        self, steps: List[ActionStep], agent_id: str, instruction: str
    ) -> Dict:
        """Analyze coordination requirements and patterns"""

        coordination_actions = [s for s in steps if s.action_type == "coordination"]
        wait_actions = [s for s in steps if "wait" in s.action_content.lower()]

        # Check if instruction mentions multiple agents or coordination
        instruction_lower = instruction.lower()
        multi_agent_indicators = [
            "other agent",
            "both",
            "together",
            "coordinate",
            "help",
        ]
        requires_coordination = any(
            indicator in instruction_lower for indicator in multi_agent_indicators
        )

        return {
            "coordination_actions_count": len(coordination_actions),
            "wait_actions_count": len(wait_actions),
            "requires_coordination": requires_coordination,
            "coordination_points": [s.step_number for s in coordination_actions],
            "coordination_effectiveness": len(coordination_actions) > 0
            and all(s.success for s in coordination_actions),
        }

    def _calculate_action_efficiency(self, steps: List[ActionStep]) -> Dict:
        """Calculate various efficiency metrics"""
        if not steps:
            return {"efficiency_score": 0.0, "metrics": {}}

        total_steps = len(steps)
        successful_steps = sum(1 for s in steps if s.success)
        unique_action_types = len(set(s.action_type for s in steps))

        # Calculate efficiency metrics
        success_rate = successful_steps / total_steps
        action_diversity = unique_action_types / 7.0  # Normalize by max action types

        # Penalty for repeated failed actions
        failed_sequences = 0
        for i in range(1, len(steps)):
            if (
                not steps[i - 1].success
                and steps[i].action_type == steps[i - 1].action_type
            ):
                failed_sequences += 1

        repetition_penalty = failed_sequences / total_steps if total_steps > 0 else 0
        efficiency_score = (
            success_rate * 0.6 + action_diversity * 0.3 + (1 - repetition_penalty) * 0.1
        )

        return {
            "efficiency_score": efficiency_score,
            "metrics": {
                "success_rate": success_rate,
                "action_diversity": action_diversity,
                "repetition_penalty": repetition_penalty,
                "total_steps": total_steps,
                "successful_steps": successful_steps,
            },
        }

    def _load_skill_templates(self) -> Dict:
        """Load skill description templates"""
        return {
            "navigation": "Navigate efficiently to {location} using {strategy}",
            "manipulation": "Manipulate {objects} with {precision} using {approach}",
            "coordination": "Coordinate with other agents through {method}",
            "planning": "Execute {complexity} planning with {adaptability}",
        }

    def _load_action_patterns(self) -> Dict:
        """Load common action pattern definitions"""
        return {
            "exploration_patterns": ["systematic", "targeted", "opportunistic"],
            "manipulation_patterns": ["precise", "efficient", "adaptive"],
            "coordination_patterns": ["proactive", "reactive", "collaborative"],
            "planning_patterns": ["linear", "adaptive", "hierarchical"],
        }

    # ============= OPTIMIZED FAST METHODS =============

    def _create_empty_result(self) -> Dict:
        """Create empty result for traces with no actions"""
        return {
            "enhanced_skill_description": "No actions recorded for this episode.",
            "skill_patterns": [],
            "decision_points": [],
            "coordination_requirements": {
                "coordination_actions_count": 0,
                "wait_actions_count": 0,
                "requires_coordination": False,
                "coordination_points": [],
                "coordination_effectiveness": False,
            },
            "action_efficiency": {"efficiency_score": 0.0, "metrics": {}},
            "skill_categories": [],
            "learning_insights": [],
        }

    def _identify_decision_points_fast(
        self, steps: List[ActionStep], instruction: str
    ) -> List[DecisionPoint]:
        """Fast decision point identification"""
        decision_points = []

        for i, step in enumerate(steps):
            # Only check critical decision types for speed
            if not step.success and i < len(steps) - 1:
                # Recovery decision
                next_step = steps[i + 1]
                decision_points.append(
                    DecisionPoint(
                        step_number=i,
                        decision_type="recovery",
                        context=f"Action failed: {step.observation[:50]}...",
                        alternatives=["retry", "different approach"],
                        chosen_action=next_step.action_content[:50],
                        outcome="recovery_attempt",
                        reasoning=f"After {step.action_type} failed, chose {next_step.action_type}",
                    )
                )
            elif (
                step.action_type == "coordination"
                or "wait" in step.action_content.lower()
            ):
                # Coordination decision
                decision_points.append(
                    DecisionPoint(
                        step_number=i,
                        decision_type="coordination",
                        context="Multi-agent coordination",
                        alternatives=["wait", "proceed"],
                        chosen_action=step.action_content[:50],
                        outcome="coordination",
                        reasoning="Agent coordinated with other agent",
                    )
                )

        return decision_points

    def _extract_skill_patterns_fast(
        self, steps: List[ActionStep], decision_points: List[DecisionPoint]
    ) -> List[SkillPattern]:
        """Fast skill pattern extraction"""
        patterns = []

        # Quick pattern detection
        action_types = [s.action_type for s in steps]

        # Navigation pattern
        if "exploration" in action_types and "navigation" in action_types:
            patterns.append(
                SkillPattern(
                    skill_name="navigation_sequence",
                    skill_type="navigation",
                    description="Combines exploration and navigation effectively",
                    action_sequence=action_types[:3],
                    success_indicators=["successful navigation"],
                    failure_indicators=["failed navigation"],
                    contextual_conditions={"pattern_type": "navigation"},
                )
            )

        # Manipulation pattern
        if "manipulation_pick" in action_types and "manipulation_place" in action_types:
            patterns.append(
                SkillPattern(
                    skill_name="pick_and_place_sequence",
                    skill_type="manipulation",
                    description="Executes pick and place operations",
                    action_sequence=[a for a in action_types if "manipulation" in a],
                    success_indicators=["successful manipulation"],
                    failure_indicators=["failed manipulation"],
                    contextual_conditions={"pattern_type": "manipulation"},
                )
            )

        # Recovery pattern
        if len(decision_points) > 0:
            patterns.append(
                SkillPattern(
                    skill_name="adaptive_recovery",
                    skill_type="planning",
                    description=f"Demonstrates {len(decision_points)} adaptive decisions",
                    action_sequence=[],
                    success_indicators=["successful adaptation"],
                    failure_indicators=["repeated failures"],
                    contextual_conditions={"decisions": str(len(decision_points))},
                )
            )

        return patterns

    def _generate_contextual_skills_fast(
        self,
        steps: List[ActionStep],
        decision_points: List[DecisionPoint],
        patterns: List[SkillPattern],
        instruction: str,
        agent_id: str,
    ) -> Dict:
        """Fast contextual skill generation"""

        # Quick analysis
        success_rate = sum(1 for s in steps if s.success) / len(steps) if steps else 0
        action_diversity = len(set(s.action_type for s in steps))

        # Fast description generation
        if success_rate > 0.8:
            primary_desc = f"Agent {agent_id} demonstrates high-efficiency execution ({success_rate:.1%} success) with {action_diversity} action types for: {instruction[:100]}"
        else:
            primary_desc = f"Agent {agent_id} shows adaptive behavior ({success_rate:.1%} success) across {action_diversity} action types for: {instruction[:100]}"

        # Fast categorization
        categories = []
        if action_diversity >= 3:
            categories.append("Multi-Modal Execution")
        if success_rate > 0.8:
            categories.append("High Reliability")
        if len(patterns) > 1:
            categories.append("Pattern Recognition")
        if len(decision_points) > 0:
            categories.append("Adaptive Decision Making")

        # Fast insights
        insights = []
        if success_rate > 0.8:
            insights.append(
                f"High success rate: {len([s for s in steps if s.success])}/{len(steps)} actions successful"
            )
        if len(patterns) > 0:
            insights.append(f"Identified {len(patterns)} skill patterns")
        if len(decision_points) > 0:
            insights.append(f"Made {len(decision_points)} adaptive decisions")

        return {
            "primary_description": primary_desc,
            "categories": categories,
            "insights": insights,
        }

    def _analyze_coordination_requirements_fast(
        self, steps: List[ActionStep], agent_id: str, instruction: str
    ) -> Dict:
        """Fast coordination analysis"""

        coordination_actions = [s for s in steps if s.action_type == "coordination"]
        wait_actions = [s for s in steps if "wait" in s.action_content.lower()]

        # Quick check for coordination indicators
        requires_coordination = (
            len(coordination_actions) > 0
            or len(wait_actions) > 0
            or any(
                word in instruction.lower()
                for word in ["other agent", "together", "coordinate"]
            )
        )

        return {
            "coordination_actions_count": len(coordination_actions),
            "wait_actions_count": len(wait_actions),
            "requires_coordination": requires_coordination,
            "coordination_points": [s.step_number for s in coordination_actions],
            "coordination_effectiveness": len(coordination_actions) > 0
            and all(s.success for s in coordination_actions),
        }

    def _calculate_action_efficiency_fast(self, steps: List[ActionStep]) -> Dict:
        """Fast action efficiency calculation"""
        if not steps:
            return {"efficiency_score": 0.0, "metrics": {}}

        total_steps = len(steps)
        successful_steps = sum(1 for s in steps if s.success)
        success_rate = successful_steps / total_steps

        # Simple efficiency score
        efficiency_score = success_rate * 0.8 + min(
            len(set(s.action_type for s in steps)) / 5.0, 0.2
        )

        return {
            "efficiency_score": efficiency_score,
            "metrics": {
                "success_rate": success_rate,
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "action_diversity": len(set(s.action_type for s in steps)),
            },
        }


# Usage example and testing functions
def test_enhanced_extractor():
    """Test the enhanced skill extractor with sample data"""

    sample_trace = """
=== SKILL ANALYSIS ===
=== ORIGINAL TRACE ===

Agent_0_Action: Explore[living room]
Agent_0_Observation: In living room, I can see: couch, coffee table, lamp. No apple found.

Agent_0_Action: Explore[kitchen]
Agent_0_Observation: In kitchen, I can see: counter, sink, fruit bowl with apple, plates.

Agent_0_Action: Navigate[kitchen counter]
Agent_0_Observation: Successfully moved to kitchen counter.

Agent_0_Action: Pick[apple from fruit bowl]
Agent_0_Observation: Successfully picked up the apple.

Agent_0_Action: Navigate[dining table]
Agent_0_Observation: Successfully moved to dining table.

Agent_0_Action: Place[apple on dining table]
Agent_0_Observation: Successfully placed apple on dining table.

Agent_0_Action: Done[]
Agent_0_Observation: Task completed successfully.
"""

    extractor = EnhancedSkillExtractor(use_llm=False)  # Use heuristics for testing
    instruction = "Move the apple from the kitchen to the dining table"

    result = extractor.extract_enhanced_skills(sample_trace, "0", instruction)

    print("Enhanced Skill Extraction Test Results:")
    print("=" * 50)
    print(f"Primary Description: {result['enhanced_skill_description']}")
    print(f"\nSkill Categories: {result['skill_categories']}")
    print(f"\nLearning Insights: {result['learning_insights']}")
    print(f"\nEfficiency Score: {result['action_efficiency']['efficiency_score']:.3f}")
    print(f"\nDecision Points: {len(result['decision_points'])}")
    print(f"Skill Patterns: {len(result['skill_patterns'])}")

    return result


if __name__ == "__main__":
    test_enhanced_extractor()
