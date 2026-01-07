#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-Based Skill Extractor

Uses local Llama 3.3 70B Instruct model to extract individual and cooperation skills
from trajectories with better semantic understanding.

Supports both local models (transformers) and vLLM API.

Reference: /doc/our_method
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Try to import transformers for local model
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers/torch not found for local model support")

# Try to import openai for vLLM API fallback
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class ExtractedSkill:
    """Skill extracted by LLM"""
    name: str
    skill_type: str  # "individual" or "cooperation"
    description: str
    action_sequence: List[str]
    objects_involved: List[str]
    locations_involved: List[str]
    preconditions: List[str]
    effects: List[str]
    demo: str
    confidence: float


@dataclass
class ExtractedCooperationPattern:
    """Cooperation pattern extracted by LLM"""
    pattern_name: str
    description: str
    agent_0_role: str
    agent_1_role: str
    trigger_condition: str
    coordination_mechanism: str
    tom_reasoning: Dict[str, str]


class LLMSkillExtractor:
    """
    LLM-based skill extractor using Llama 3.3 70B

    Supports both local model inference (transformers) and vLLM API.
    """

    def __init__(
        self,
        model_path: str = "/home/a5l/shuqing.a5l/models/Llama-3.3-70B-Instruct",
        use_local: bool = True,
        vllm_host: str = "localhost",
        vllm_port: int = 8000,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        device_map: str = "auto"
    ):
        self.model_path = model_path
        self.use_local = use_local
        self.vllm_host = vllm_host
        self.vllm_port = vllm_port
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device_map = device_map

        # Initialize model/client
        self.model = None
        self.tokenizer = None
        self.client = None

        if use_local and HAS_TRANSFORMERS:
            self._init_local_model()
        elif HAS_OPENAI:
            self._init_api_client()
        else:
            raise RuntimeError("Neither transformers nor openai available for LLM inference")

    def _init_local_model(self):
        """Initialize local model with transformers"""
        print(f"Loading local model from {self.model_path}...")
        print(f"Device map: {self.device_map}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Local model loaded successfully!")

    def _init_api_client(self):
        """Initialize OpenAI client for vLLM API"""
        base_url = f"http://{self.vllm_host}:{self.vllm_port}/v1"
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url
        )

    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call LLM and return response"""
        if self.use_local and self.model is not None:
            return self._call_local_model(prompt, system_prompt)
        elif self.client is not None:
            return self._call_api(prompt, system_prompt)
        else:
            raise RuntimeError("No LLM backend available")

    def _call_local_model(self, prompt: str, system_prompt: str = None) -> str:
        """Call local model for inference"""
        # Build chat messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode response (only new tokens)
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def _call_api(self, prompt: str, system_prompt: str = None) -> str:
        """Call vLLM API for inference"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            return ""

    def extract_individual_skills(
        self,
        trace_content: str,
        agent_id: int,
        instruction: str
    ) -> List[ExtractedSkill]:
        """
        Extract individual skills from agent's trajectory using LLM

        Args:
            trace_content: Full trace content
            agent_id: Agent ID to extract skills for
            instruction: Task instruction

        Returns:
            List of extracted skills
        """
        # Parse agent actions from trace
        agent_actions = self._parse_agent_actions(trace_content, agent_id)

        if not agent_actions:
            return []

        system_prompt = """You are an expert at analyzing robotic agent behaviors and extracting reusable skills.
Your task is to identify distinct, reusable skills from an agent's action sequence.

A skill should be:
1. A coherent sequence of actions that accomplishes a sub-goal
2. Reusable across different contexts
3. Clearly defined with preconditions and effects

Output your analysis as JSON."""

        prompt = f"""Analyze this agent's behavior and extract individual skills.

Task: {instruction}
Agent ID: {agent_id}

Action Sequence:
{chr(10).join(f"- {a}" for a in agent_actions)}

Extract skills from this trajectory. For each skill, provide:
1. name: A short, descriptive name (e.g., "pickup_and_place", "navigate_to_target")
2. description: What the skill accomplishes
3. action_sequence: The specific actions that make up this skill
4. objects_involved: Objects the agent interacts with
5. locations_involved: Locations/furniture involved
6. preconditions: What must be true before executing
7. effects: What changes after execution
8. demo: Natural language demonstration of the skill

Output as JSON array:
```json
[
  {{
    "name": "skill_name",
    "description": "what it does",
    "action_sequence": ["Navigate[x]", "Pick[y]", ...],
    "objects_involved": ["object1", "object2"],
    "locations_involved": ["table_1", "counter_2"],
    "preconditions": ["agent_hands_empty", "object_reachable"],
    "effects": ["holding_object", "object_moved"],
    "demo": "Navigate to table_1, pick up cup_0, then move to counter_2 and place it"
  }}
]
```

Extract skills (output JSON only):"""

        response = self._call_llm(prompt, system_prompt)
        skills = self._parse_skills_response(response, "individual")

        return skills

    def extract_cooperation_skills(
        self,
        trace_content: str,
        instruction: str
    ) -> Tuple[List[ExtractedSkill], List[ExtractedCooperationPattern]]:
        """
        Extract cooperation skills and patterns from multi-agent trajectory

        Args:
            trace_content: Full trace content with both agents
            instruction: Task instruction

        Returns:
            Tuple of (cooperation skills, cooperation patterns)
        """
        # Parse both agents' actions
        agent_0_actions = self._parse_agent_actions(trace_content, 0)
        agent_1_actions = self._parse_agent_actions(trace_content, 1)

        system_prompt = """You are an expert at analyzing multi-agent coordination and extracting cooperation patterns.
Your task is to identify how agents coordinate to accomplish shared tasks.

Focus on:
1. Division of labor - how tasks are split
2. Sequential handoffs - when one agent sets up for another
3. Synchronization - how agents wait and coordinate timing
4. Theory of Mind - how agents reason about partner's actions

Output your analysis as JSON."""

        prompt = f"""Analyze this multi-agent trajectory and extract cooperation skills and patterns.

Task: {instruction}

Agent 0 Actions:
{chr(10).join(f"- {a}" for a in agent_0_actions)}

Agent 1 Actions:
{chr(10).join(f"- {a}" for a in agent_1_actions)}

Extract:

1. COOPERATION SKILLS - Joint skills that require both agents:
```json
{{
  "cooperation_skills": [
    {{
      "name": "skill_name",
      "description": "what the joint skill accomplishes",
      "agent_0_role": "what agent 0 does",
      "agent_1_role": "what agent 1 does",
      "agent_0_actions": ["Navigate[obj_0]", "Pick[obj_0]", "Place[obj_0, on, table_1, None, None]"],
      "agent_1_actions": ["Navigate[obj_1]", "Pick[obj_1]", "Place[obj_1, on, table_1, None, None]"],
      "action_sequence": ["Navigate[obj_0]", "Pick[obj_0]", "Navigate[obj_1]", "Pick[obj_1]", "Place[...]"],
      "objects_involved": ["obj1", "obj2"],
      "preconditions": ["both_agents_available"],
      "effects": ["task_completed"],
      "demo": "Agent 0 picks up X while Agent 1 prepares Y, then Agent 0 places X near Y"
    }}
  ]
}}
```

2. COOPERATION PATTERNS - How agents coordinate:
```json
{{
  "cooperation_patterns": [
    {{
      "pattern_name": "division_of_labor|sequential_handoff|synchronization|complementary_work",
      "description": "how this pattern manifests",
      "agent_0_role": "Agent 0's contribution",
      "agent_1_role": "Agent 1's contribution",
      "trigger_condition": "when this pattern is used",
      "coordination_mechanism": "how coordination happens (waiting, sequencing, etc.)",
      "tom_reasoning": {{
        "belief_formation": "what each agent knows about partner's state",
        "intention_inference": "how agents predict partner's goals",
        "action_selection": "how agents choose complementary actions"
      }}
    }}
  ]
}}
```

Output JSON only:"""

        response = self._call_llm(prompt, system_prompt)
        skills, patterns = self._parse_cooperation_response(response)

        return skills, patterns

    def patch_failed_episode(
        self,
        trace_content: str,
        instruction: str,
        task_percent_complete: float
    ) -> Tuple[List[ExtractedSkill], List[ExtractedSkill], Dict[str, Any]]:
        """
        Analyze and patch a failed episode using LLM

        For failed trajectories, the LLM:
        1. Extracts partial skills that were successfully executed
        2. Identifies what went wrong (failure analysis)
        3. Suggests corrected action sequences (patches)

        Args:
            trace_content: Full trace content
            instruction: Task instruction
            task_percent_complete: How much of the task was completed (0-1)

        Returns:
            Tuple of (partial_skills, patched_skills, failure_analysis)
        """
        # Parse both agents' actions
        agent_0_actions = self._parse_agent_actions(trace_content, 0)
        agent_1_actions = self._parse_agent_actions(trace_content, 1)

        system_prompt = """You are an expert at analyzing failed robotic task executions and extracting useful knowledge.
Even when tasks fail, there are often:
1. Partial successes - skills that were executed correctly before failure
2. Learnable patterns - what approach was attempted
3. Correctable mistakes - how the failure could be fixed

Your goal is to extract maximum value from failed trajectories by:
1. Identifying successfully executed sub-skills
2. Analyzing the failure point and cause
3. Suggesting corrected action sequences that would have succeeded

Output your analysis as JSON."""

        prompt = f"""Analyze this FAILED multi-agent trajectory and extract useful skills.

Task: {instruction}
Completion: {task_percent_complete * 100:.1f}% complete (FAILED)

Agent 0 Actions:
{chr(10).join(f"- {a}" for a in agent_0_actions)}

Agent 1 Actions:
{chr(10).join(f"- {a}" for a in agent_1_actions)}

Extract:

1. PARTIAL SKILLS - Skills that were successfully executed before failure:
```json
{{
  "partial_skills": [
    {{
      "name": "skill_name",
      "description": "what this partial skill accomplished",
      "action_sequence": ["Navigate[x]", "Pick[y]", ...],
      "objects_involved": ["object1"],
      "success": true,
      "demo": "Navigate to X, pick up Y"
    }}
  ]
}}
```

2. FAILURE ANALYSIS:
```json
{{
  "failure_analysis": {{
    "failure_point": "where in the trajectory the failure occurred",
    "failure_cause": "why the task failed (e.g., wrong object, wrong location, coordination issue)",
    "missing_actions": ["actions that should have been taken"],
    "incorrect_actions": ["actions that were wrong and why"]
  }}
}}
```

3. PATCHED SKILLS - Corrected versions that would succeed:
```json
{{
  "patched_skills": [
    {{
      "name": "corrected_skill_name",
      "description": "what the corrected skill does",
      "original_issue": "what was wrong in the failed attempt",
      "action_sequence": ["Navigate[correct_target]", "Pick[correct_object]", ...],
      "agent_0_actions": ["Navigate[x]", "Pick[y]"],
      "agent_1_actions": ["Navigate[z]", "Clean[w]"],
      "objects_involved": ["object1", "object2"],
      "correction_applied": "description of how this differs from failed attempt",
      "demo": "Corrected approach: Navigate to X, pick up Y, then place on Z"
    }}
  ]
}}
```

Output complete JSON with all three sections:"""

        response = self._call_llm(prompt, system_prompt)
        return self._parse_patch_response(response)

    def _parse_patch_response(
        self, response: str
    ) -> Tuple[List[ExtractedSkill], List[ExtractedSkill], Dict[str, Any]]:
        """Parse the patch response from LLM"""
        partial_skills = []
        patched_skills = []
        failure_analysis = {}

        # Try to extract JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return partial_skills, patched_skills, failure_analysis

        try:
            data = json.loads(json_match.group())

            # Parse partial skills
            for s in data.get("partial_skills", []):
                partial_skills.append(ExtractedSkill(
                    name=s.get("name", "partial_skill"),
                    skill_type="individual",
                    description=s.get("description", ""),
                    action_sequence=s.get("action_sequence", []),
                    objects_involved=s.get("objects_involved", []),
                    locations_involved=s.get("locations_involved", []),
                    preconditions=s.get("preconditions", []),
                    effects=s.get("effects", []),
                    demo=s.get("demo", ""),
                    confidence=0.7  # Lower confidence for partial skills
                ))

            # Parse patched skills
            for s in data.get("patched_skills", []):
                # Combine agent actions for action_sequence
                action_seq = s.get("action_sequence", [])
                if not action_seq:
                    agent_0_acts = s.get("agent_0_actions", [])
                    agent_1_acts = s.get("agent_1_actions", [])
                    action_seq = agent_0_acts + agent_1_acts

                patched_skills.append(ExtractedSkill(
                    name=s.get("name", "patched_skill"),
                    skill_type="cooperation",
                    description=s.get("description", "") + f" [Correction: {s.get('correction_applied', '')}]",
                    action_sequence=action_seq,
                    objects_involved=s.get("objects_involved", []),
                    locations_involved=s.get("locations_involved", []),
                    preconditions=s.get("preconditions", []),
                    effects=s.get("effects", []),
                    demo=s.get("demo", ""),
                    confidence=0.8  # Patched skills have reasonable confidence
                ))

            # Parse failure analysis
            failure_analysis = data.get("failure_analysis", {})

        except json.JSONDecodeError as e:
            print(f"Failed to parse patch JSON: {e}")

        return partial_skills, patched_skills, failure_analysis

    def extract_tom_reasoning(
        self,
        trace_content: str,
        agent_id: int,
        instruction: str
    ) -> Dict[str, Any]:
        """
        Extract Theory of Mind reasoning from trajectory

        Args:
            trace_content: Full trace content
            agent_id: Agent doing the reasoning
            instruction: Task instruction

        Returns:
            ToM reasoning analysis
        """
        partner_id = 1 - agent_id
        partner_actions = self._parse_agent_actions(trace_content, partner_id)

        system_prompt = """You are analyzing an agent's Theory of Mind reasoning.
Identify how the agent reasons about its partner's:
1. Beliefs - what does partner know?
2. Goals - what is partner trying to achieve?
3. Actions - what will partner do next?
4. Coordination - how should I respond?"""

        prompt = f"""Analyze the Theory of Mind reasoning for Agent {agent_id} observing Agent {partner_id}.

Task: {instruction}

Partner (Agent {partner_id}) Actions:
{chr(10).join(f"- {a}" for a in partner_actions)}

For each of partner's key actions, analyze:
1. What can Agent {agent_id} infer about partner's beliefs?
2. What goal is partner pursuing?
3. What will partner likely do next?
4. How should Agent {agent_id} respond to coordinate?

Output as JSON:
```json
{{
  "tom_steps": [
    {{
      "partner_action": "the observed action",
      "belief_formation": "what agent infers partner knows",
      "goal_inference": "what agent thinks partner is trying to do",
      "action_prediction": "what partner will likely do next",
      "coordination_response": "how agent should respond",
      "cooperation_pattern": "division_of_labor|sequential_handoff|synchronization"
    }}
  ],
  "overall_coordination_strategy": "summary of how agents coordinate",
  "tom_quality": {{
    "completeness": 0.0-1.0,
    "accuracy": 0.0-1.0,
    "coordination_effectiveness": 0.0-1.0
  }}
}}
```

Output JSON only:"""

        response = self._call_llm(prompt, system_prompt)
        return self._parse_tom_response(response)

    def _parse_agent_actions(self, trace_content: str, agent_id: int) -> List[str]:
        """Parse agent actions from trace content"""
        actions = []
        for line in trace_content.split('\n'):
            if f"Agent_{agent_id}_Action:" in line:
                action = line.split("Action:")[-1].strip()
                if action:
                    actions.append(action)
        return actions

    def _parse_skills_response(self, response: str, skill_type: str) -> List[ExtractedSkill]:
        """Parse LLM response into skill objects"""
        skills = []

        # Try to extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            return skills

        try:
            skill_data = json.loads(json_match.group())
            for s in skill_data:
                skills.append(ExtractedSkill(
                    name=s.get("name", "unknown"),
                    skill_type=skill_type,
                    description=s.get("description", ""),
                    action_sequence=s.get("action_sequence", []),
                    objects_involved=s.get("objects_involved", []),
                    locations_involved=s.get("locations_involved", []),
                    preconditions=s.get("preconditions", []),
                    effects=s.get("effects", []),
                    demo=s.get("demo", ""),
                    confidence=0.9
                ))
        except json.JSONDecodeError as e:
            print(f"Failed to parse skills JSON: {e}")

        return skills

    def _parse_cooperation_response(
        self, response: str
    ) -> Tuple[List[ExtractedSkill], List[ExtractedCooperationPattern]]:
        """Parse cooperation skills and patterns from LLM response"""
        skills = []
        patterns = []

        # Try to extract JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return skills, patterns

        try:
            data = json.loads(json_match.group())

            # Parse cooperation skills
            for s in data.get("cooperation_skills", []):
                # Combine agent actions into action_sequence if not already provided
                action_seq = s.get("action_sequence", [])
                if not action_seq:
                    # Fallback: combine individual agent actions
                    agent_0_acts = s.get("agent_0_actions", [])
                    agent_1_acts = s.get("agent_1_actions", [])
                    action_seq = agent_0_acts + agent_1_acts

                skills.append(ExtractedSkill(
                    name=s.get("name", "unknown"),
                    skill_type="cooperation",
                    description=s.get("description", ""),
                    action_sequence=action_seq,
                    objects_involved=s.get("objects_involved", []),
                    locations_involved=[],
                    preconditions=s.get("preconditions", []),
                    effects=s.get("effects", []),
                    demo=s.get("demo", ""),
                    confidence=0.9
                ))

            # Parse cooperation patterns
            for p in data.get("cooperation_patterns", []):
                patterns.append(ExtractedCooperationPattern(
                    pattern_name=p.get("pattern_name", "unknown"),
                    description=p.get("description", ""),
                    agent_0_role=p.get("agent_0_role", ""),
                    agent_1_role=p.get("agent_1_role", ""),
                    trigger_condition=p.get("trigger_condition", ""),
                    coordination_mechanism=p.get("coordination_mechanism", ""),
                    tom_reasoning=p.get("tom_reasoning", {})
                ))

        except json.JSONDecodeError as e:
            print(f"Failed to parse cooperation JSON: {e}")

        return skills, patterns

    def _parse_tom_response(self, response: str) -> Dict[str, Any]:
        """Parse ToM reasoning from LLM response"""
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return {}

        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return {}


class FallbackSkillExtractor:
    """
    Fallback rule-based skill extractor when LLM is not available
    """

    def __init__(self):
        self.action_to_skill = {
            "Navigate": "navigation",
            "Pick": "pickup",
            "Place": "placement",
            "Open": "open_furniture",
            "Close": "close_furniture",
            "Clean": "cleaning",
            "Fill": "filling",
            "PowerOn": "power_on",
            "PowerOff": "power_off",
            "Rearrange": "rearrangement",
            "Wait": "synchronization"
        }

    def extract_individual_skills(
        self,
        trace_content: str,
        agent_id: int,
        instruction: str
    ) -> List[ExtractedSkill]:
        """Extract skills using rule-based approach"""
        actions = self._parse_actions(trace_content, agent_id)

        # Group into skill sequences
        sequences = self._group_into_sequences(actions)

        skills = []
        for seq in sequences:
            skill = self._sequence_to_skill(seq, agent_id)
            if skill:
                skills.append(skill)

        return skills

    def _parse_actions(self, trace_content: str, agent_id: int) -> List[Dict]:
        """Parse actions from trace"""
        actions = []
        for line in trace_content.split('\n'):
            if f"Agent_{agent_id}_Action:" in line:
                action_str = line.split("Action:")[-1].strip()
                match = re.match(r'(\w+)\[([^\]]*)\]', action_str)
                if match:
                    actions.append({
                        "type": match.group(1),
                        "args": [a.strip() for a in match.group(2).split(",") if a.strip()],
                        "raw": action_str
                    })
                else:
                    actions.append({"type": action_str, "args": [], "raw": action_str})
        return actions

    def _group_into_sequences(self, actions: List[Dict]) -> List[List[Dict]]:
        """Group actions into skill sequences"""
        sequences = []
        current = []

        for action in actions:
            if action["type"] == "Wait":
                if current:
                    sequences.append(current)
                    current = []
            else:
                current.append(action)
                # Split on terminal actions
                if action["type"] in ["Place", "Clean", "Fill", "PowerOn", "PowerOff"]:
                    sequences.append(current)
                    current = []

        if current:
            sequences.append(current)

        return sequences

    def _sequence_to_skill(self, sequence: List[Dict], agent_id: int) -> Optional[ExtractedSkill]:
        """Convert action sequence to skill"""
        if not sequence:
            return None

        # Determine skill name from terminal action
        terminal = sequence[-1]
        skill_name = self.action_to_skill.get(terminal["type"], terminal["type"].lower())

        # Extract objects and locations
        objects = []
        locations = []
        for action in sequence:
            for arg in action["args"]:
                if arg.lower() not in ["none", "on", "within", "next_to"]:
                    if re.match(r'\w+_\d+', arg):  # Object pattern
                        if any(f in arg for f in ["table", "counter", "cabinet", "shelf", "bed", "stand"]):
                            locations.append(arg)
                        else:
                            objects.append(arg)

        # Generate demo
        demo_parts = []
        for action in sequence:
            if action["type"] == "Navigate" and action["args"]:
                demo_parts.append(f"Navigate to {action['args'][0]}")
            elif action["type"] == "Pick" and action["args"]:
                demo_parts.append(f"Pick up {action['args'][0]}")
            elif action["type"] == "Place" and len(action["args"]) >= 3:
                demo_parts.append(f"Place {action['args'][0]} on {action['args'][2]}")
            elif action["type"] == "Clean" and action["args"]:
                demo_parts.append(f"Clean {action['args'][0]}")
            elif action["type"] == "Fill" and action["args"]:
                demo_parts.append(f"Fill {action['args'][0]}")

        return ExtractedSkill(
            name=skill_name,
            skill_type="individual",
            description=f"Agent {agent_id} performs {skill_name}",
            action_sequence=[a["raw"] for a in sequence],
            objects_involved=list(set(objects)),
            locations_involved=list(set(locations)),
            preconditions=self._infer_preconditions(sequence),
            effects=self._infer_effects(sequence),
            demo=" -> ".join(demo_parts),
            confidence=0.7
        )

    def _infer_preconditions(self, sequence: List[Dict]) -> List[str]:
        """Infer preconditions from sequence"""
        preconditions = []
        for action in sequence:
            if action["type"] == "Pick":
                preconditions.extend(["agent_hands_empty", "object_reachable"])
            elif action["type"] == "Place":
                preconditions.append("agent_holding_object")
            elif action["type"] == "Fill":
                preconditions.append("near_water_source")
        return list(set(preconditions))

    def _infer_effects(self, sequence: List[Dict]) -> List[str]:
        """Infer effects from sequence"""
        effects = []
        for action in sequence:
            if action["type"] == "Pick" and action["args"]:
                effects.append(f"holding_{action['args'][0]}")
            elif action["type"] == "Place":
                effects.append("agent_hands_empty")
                if len(action["args"]) >= 3:
                    effects.append(f"{action['args'][0]}_on_{action['args'][2]}")
            elif action["type"] == "Clean" and action["args"]:
                effects.append(f"{action['args'][0]}_is_clean")
            elif action["type"] == "Fill" and action["args"]:
                effects.append(f"{action['args'][0]}_is_filled")
        return list(set(effects))


def create_skill_extractor(
    use_llm: bool = True,
    use_local: bool = True,
    model_path: str = "/home/a5l/shuqing.a5l/models/Llama-3.3-70B-Instruct",
    vllm_host: str = "localhost",
    vllm_port: int = 8000,
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    device_map: str = "auto"
) -> Union[LLMSkillExtractor, FallbackSkillExtractor]:
    """
    Factory function to create skill extractor

    Args:
        use_llm: Whether to use LLM-based extraction
        use_local: Use local model (True) or vLLM API (False)
        model_path: Path to local model
        vllm_host: vLLM server host (if use_local=False)
        vllm_port: vLLM server port (if use_local=False)
        model_name: Model name for vLLM API
        device_map: Device map for local model ("auto", "cuda:0", etc.)

    Returns:
        Skill extractor instance
    """
    if use_llm:
        try:
            if use_local and HAS_TRANSFORMERS:
                extractor = LLMSkillExtractor(
                    model_path=model_path,
                    use_local=True,
                    device_map=device_map
                )
                if extractor.model is not None:
                    return extractor
            elif HAS_OPENAI:
                extractor = LLMSkillExtractor(
                    use_local=False,
                    vllm_host=vllm_host,
                    vllm_port=vllm_port,
                    model_name=model_name
                )
                if extractor.client:
                    return extractor
        except Exception as e:
            print(f"Failed to initialize LLM extractor: {e}")

    print("Using fallback rule-based skill extractor")
    return FallbackSkillExtractor()


if __name__ == "__main__":
    # Test with sample trace
    sample_trace = """Task: Move the cup to the kitchen counter

Agent_0_Action: Navigate[cup_0]
Agent_0_Action: Pick[cup_0]
Agent_0_Action: Navigate[counter_1]
Agent_0_Action: Place[cup_0, on, counter_1, none, none]
Agent_1_Action: Wait
Agent_1_Action: Navigate[table_1]
Agent_1_Action: Clean[table_1]
"""

    # Try LLM extractor first, fall back to rules
    extractor = create_skill_extractor(use_llm=True)

    print("Extracting individual skills...")
    skills = extractor.extract_individual_skills(sample_trace, 0, "Move the cup to the kitchen counter")

    for skill in skills:
        print(f"\nSkill: {skill.name}")
        print(f"  Description: {skill.description}")
        print(f"  Actions: {skill.action_sequence}")
        print(f"  Demo: {skill.demo}")
