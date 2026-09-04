#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Retrieval Pipeline

This module implements the memory utilization stage with hierarchical retrieval
for the Decentralized Skill-Structured Hierarchical Memory framework.

The pipeline follows:
1. Query Generation: q_t = f_query(w_t^self, w_t^env, delta_w_{t-1}, g)
2. Abstract Skill Matching: S_candidate = {s in L : sim(q_t, name(s)) > theta_abstract}
3. Instance Retrieval: I_retrieved = union top-k{sim(context(i), w_t)}
4. Executability Filtering

Reference: /doc/our_method Section 4
"""

import gzip
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not available. Using fallback similarity.")


@dataclass
class Query:
    """
    Query representation for retrieval
    q_t = f_query(w_t^self, w_t^env, delta_w_{t-1}, g)
    """
    agent_state: Dict[str, Any]  # w_t^self
    environment_state: Dict[str, Any]  # w_t^env
    partner_effects: Dict[str, Any]  # delta_w_{t-1}
    goal: str  # g
    text_query: str  # Natural language query


# Amendment 9 switches, off by default so every earlier run reproduces exactly.
ROLE_AWARE_CONTEXT = os.environ.get("A9_ROLE_AWARE") == "1"


@dataclass
class RetrievedSkill:
    """A skill retrieved from memory with relevance score"""
    skill_name: str
    skill_type: str  # "individual" or "cooperation"
    instances: List[Dict]
    abstract_score: float
    instance_scores: List[float]
    is_executable: bool
    demo: str


class HierarchicalRetriever:
    """
    Hierarchical Retrieval Pipeline

    Implements the four-stage retrieval process:
    1. Query Generation
    2. Abstract Skill Matching
    3. Instance Retrieval
    4. Executability Filtering
    """

    def __init__(
        self,
        memory_path: str,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        abstract_threshold: float = 0.3,
        instance_top_k: int = 5
    ):
        self.memory_path = memory_path
        self.abstract_threshold = abstract_threshold
        self.instance_top_k = instance_top_k

        # Load memory
        self.L_ind = {}
        self.L_coop = {}
        self.episodic_memory = {}
        self._load_memory()

        # Initialize embedding model
        self.embedding_model = None
        self.skill_embeddings = {}
        self.instance_embeddings = {}

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self._build_embeddings()
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")

    def _load_memory(self):
        """Load hierarchical memory from disk"""
        # Load individual skills
        ind_path = os.path.join(self.memory_path, "L_ind_skills.json.gz")
        if os.path.exists(ind_path):
            with gzip.open(ind_path, 'rt', encoding='utf-8') as f:
                self.L_ind = json.load(f)
            print(f"Loaded {len(self.L_ind)} individual skills")

        # Load cooperation skills
        coop_path = os.path.join(self.memory_path, "L_coop_skills.json.gz")
        if os.path.exists(coop_path):
            with gzip.open(coop_path, 'rt', encoding='utf-8') as f:
                self.L_coop = json.load(f)
            print(f"Loaded {len(self.L_coop)} cooperation skills")

        # Load episodic memory
        episodic_path = os.path.join(self.memory_path, "episodic_memory.json.gz")
        if os.path.exists(episodic_path):
            with gzip.open(episodic_path, 'rt', encoding='utf-8') as f:
                self.episodic_memory = json.load(f)
            print(f"Loaded {len(self.episodic_memory)} episodes")

    def _build_embeddings(self):
        """Build embeddings for skills and instances"""
        if not self.embedding_model:
            return

        print("Building skill embeddings...")

        # Embed skill names and descriptions
        all_skills = {**self.L_ind, **self.L_coop}

        skill_texts = []
        skill_keys = []
        for key, skill in all_skills.items():
            text = f"{skill['name']} {skill['description']}"
            skill_texts.append(text)
            skill_keys.append(key)

        if skill_texts:
            embeddings = self.embedding_model.encode(skill_texts, show_progress_bar=False)
            for key, emb in zip(skill_keys, embeddings):
                self.skill_embeddings[key] = emb

        # Embed instance contexts
        print("Building instance embeddings...")
        for key, skill in all_skills.items():
            self.instance_embeddings[key] = []
            for inst in skill.get("instances", []):
                context_text = self._context_to_text(inst.get("context", {}))
                demo_text = inst.get("demo", "")
                text = f"{context_text} {demo_text}"
                if text.strip():
                    emb = self.embedding_model.encode(text, show_progress_bar=False)
                    self.instance_embeddings[key].append(emb)

        print(f"Built embeddings for {len(self.skill_embeddings)} skills")

    def _context_to_text(self, context: Dict) -> str:
        """Convert context dict to text for embedding"""
        parts = []
        if "objects" in context:
            parts.append(f"objects: {', '.join(context['objects'])}")
        if "locations" in context and context["locations"]:
            parts.append(f"locations: {', '.join(context['locations'])}")
        if "rooms" in context and context["rooms"]:
            parts.append(f"rooms: {', '.join(context['rooms'])}")
        if "object_locations" in context and context["object_locations"]:
            # Format: "obj1 at loc1 in room1, obj2 at loc2 in room2"
            loc_strs = []
            for obj, info in context["object_locations"].items():
                loc = info.get("location", "unknown")
                room = info.get("room", "unknown")
                loc_strs.append(f"{obj} at {loc} in {room}")
            parts.append(f"object positions: {', '.join(loc_strs)}")
        if "action_sequence" in context:
            parts.append(f"actions: {', '.join(context['action_sequence'])}")
        if ROLE_AWARE_CONTEXT:
            # Amendment 9 C1. A cooperation instance that stores per-agent roles
            # carries none of the object-centric keys above, so its text was empty,
            # its embedding uninformative, and instance_top_k always cut it: 0 of
            # 705 retrieved cooperation skills carried role fields. The roles are
            # the coordination content, so they are embedded too.
            roles = [
                f"{key}: {value}"
                for key, value in sorted(context.items())
                if key.startswith("agent_") and key.endswith("_role") and value
            ]
            if roles:
                parts.append("roles: " + "; ".join(roles))
            if context.get("coordination_mechanism"):
                parts.append(f"coordination: {context['coordination_mechanism']}")
        return " ".join(parts)

    def generate_query(
        self,
        agent_state: Dict[str, Any],
        environment_state: Dict[str, Any],
        partner_effects: Dict[str, Any],
        goal: str
    ) -> Query:
        """
        Stage 1: Query Generation
        q_t = f_query(w_t^self, w_t^env, delta_w_{t-1}, g)

        Args:
            agent_state: Agent's internal state (held objects, position)
            environment_state: Observable environment state (includes seen_objects)
            partner_effects: Observed effects of partner's action (delta_w)
            goal: Current goal

        Returns:
            Query object for retrieval
        """
        # Build text query from components
        query_parts = []

        # Goal
        query_parts.append(f"Goal: {goal}")

        # Agent state
        if agent_state.get("holding"):
            query_parts.append(f"Holding: {agent_state['holding']}")
        if agent_state.get("position"):
            query_parts.append(f"At: {agent_state['position']}")
        if agent_state.get("room"):
            query_parts.append(f"In room: {agent_state['room']}")

        # Environment state with seen object locations
        if environment_state:
            # Include seen objects with their locations
            seen_objects = environment_state.get("seen_objects", {})
            if seen_objects:
                loc_strs = []
                for obj_name, obj_info in list(seen_objects.items())[:5]:  # Limit to 5 for query
                    if isinstance(obj_info, dict):
                        loc = obj_info.get("location", "unknown")
                        room = obj_info.get("room", "unknown")
                        loc_strs.append(f"{obj_name} at {loc} in {room}")
                if loc_strs:
                    query_parts.append(f"Seen objects: {', '.join(loc_strs)}")

            # Include rooms the agent knows about
            known_rooms = environment_state.get("known_rooms", [])
            if known_rooms:
                query_parts.append(f"Known rooms: {', '.join(known_rooms[:5])}")

        # Partner effects (Theory of Mind component)
        if partner_effects:
            if partner_effects.get("action"):
                query_parts.append(f"Partner action: {partner_effects['action']}")
            if partner_effects.get("moved_objects"):
                query_parts.append(f"Partner moved: {partner_effects['moved_objects']}")
            if partner_effects.get("completed_subtasks"):
                query_parts.append(f"Partner completed: {partner_effects['completed_subtasks']}")
            if partner_effects.get("location"):
                query_parts.append(f"Partner at: {partner_effects['location']}")

        text_query = " | ".join(query_parts)

        # Debug: Print query with location info
        seen_objs = environment_state.get("seen_objects", {}) if environment_state else {}
        known_rooms = environment_state.get("known_rooms", []) if environment_state else []
        print(f"\n[LOCATION TRACKING] Generated Query:", flush=True)
        print(f"  Text Query: {text_query[:200]}...", flush=True)
        print(f"  Seen Objects: {len(seen_objs)} objects tracked", flush=True)
        print(f"  Known Rooms: {known_rooms[:5] if known_rooms else 'none'}", flush=True)

        return Query(
            agent_state=agent_state,
            environment_state=environment_state,
            partner_effects=partner_effects,
            goal=goal,
            text_query=text_query
        )

    def match_abstract_skills(self, query: Query) -> Dict[str, float]:
        """
        Stage 2: Abstract Skill Matching
        S_candidate = {s in L : sim(q_t, name(s)) > theta_abstract}

        Args:
            query: Query object

        Returns:
            Dict mapping skill keys to similarity scores
        """
        candidates = {}

        if self.embedding_model and self.skill_embeddings:
            # Embed query
            query_emb = self.embedding_model.encode(query.text_query, show_progress_bar=False)

            # Compute similarities
            for skill_key, skill_emb in self.skill_embeddings.items():
                sim = self._cosine_similarity(query_emb, skill_emb)
                if sim > self.abstract_threshold:
                    candidates[skill_key] = sim
        else:
            # Fallback: keyword matching
            candidates = self._keyword_matching(query)

        # Sort by similarity
        candidates = dict(sorted(candidates.items(), key=lambda x: x[1], reverse=True))

        return candidates

    def retrieve_instances(
        self,
        candidates: Dict[str, float],
        query: Query
    ) -> List[RetrievedSkill]:
        """
        Stage 3: Instance Retrieval
        I_retrieved = union_{s in S_candidate} top-k{sim(context(i), w_t)}

        Args:
            candidates: Dict of candidate skills with abstract scores
            query: Query object

        Returns:
            List of RetrievedSkill objects with instances
        """
        retrieved = []

        # Prepare query context embedding
        query_context_text = self._build_context_query(query)
        query_context_emb = None
        if self.embedding_model:
            query_context_emb = self.embedding_model.encode(query_context_text, show_progress_bar=False)

        for skill_key, abstract_score in candidates.items():
            # Get skill data
            if skill_key in self.L_ind:
                skill = self.L_ind[skill_key]
                skill_type = "individual"
            elif skill_key in self.L_coop:
                skill = self.L_coop[skill_key]
                skill_type = "cooperation"
            else:
                continue

            instances = skill.get("instances", [])
            instance_scores = []

            # Score instances by context similarity
            if query_context_emb is not None and skill_key in self.instance_embeddings:
                inst_embeddings = self.instance_embeddings[skill_key]
                for i, inst_emb in enumerate(inst_embeddings):
                    if i < len(instances):
                        sim = self._cosine_similarity(query_context_emb, inst_emb)
                        instance_scores.append((i, sim))
            else:
                # Fallback: use all instances with equal score
                instance_scores = [(i, 1.0) for i in range(len(instances))]

            # Sort and take top-k
            instance_scores.sort(key=lambda x: x[1], reverse=True)
            top_k_indices = [idx for idx, _ in instance_scores[:self.instance_top_k]]
            top_k_scores = [score for _, score in instance_scores[:self.instance_top_k]]

            # Get top-k instances
            top_instances = [instances[i] for i in top_k_indices if i < len(instances)]

            # Build demo from top instance
            demo = top_instances[0].get("demo", "") if top_instances else ""

            retrieved.append(RetrievedSkill(
                skill_name=skill["name"],
                skill_type=skill_type,
                instances=top_instances,
                abstract_score=abstract_score,
                instance_scores=top_k_scores,
                is_executable=True,  # Will be filtered in next stage
                demo=demo
            ))

        return retrieved

    def filter_executable(
        self,
        retrieved: List[RetrievedSkill],
        query: Query
    ) -> List[RetrievedSkill]:
        """
        Stage 4: Executability Filtering

        For individual skills:
        S_executable = {s in S_candidate : precond_s(w_t^self) = true}

        For cooperation skills:
        S_executable^coop = {s in S_candidate ∩ L_coop : trigger_s(delta_w_{t-1}) and precond_s(w_t^self)}

        Args:
            retrieved: List of retrieved skills
            query: Query object

        Returns:
            Filtered list of executable skills
        """
        executable = []

        for skill in retrieved:
            is_exec = True

            if skill.skill_type == "individual":
                # Check individual preconditions
                is_exec = self._check_individual_preconditions(skill, query)
            else:
                # Check cooperation preconditions
                is_exec = self._check_cooperation_preconditions(skill, query)

            if is_exec:
                skill.is_executable = True
                executable.append(skill)
            else:
                skill.is_executable = False

        return executable

    def _check_individual_preconditions(self, skill: RetrievedSkill, query: Query) -> bool:
        """Check if individual skill preconditions are met"""
        # Get preconditions from memory
        skill_data = self.L_ind.get(f"{skill.skill_name}_individual", {})
        preconditions = skill_data.get("preconditions", [])

        agent_state = query.agent_state

        for precond in preconditions:
            if "agent_hands_empty" in precond:
                if agent_state.get("holding"):
                    return False
            if "agent_holding_object" in precond:
                if not agent_state.get("holding"):
                    return False

        return True

    def _check_cooperation_preconditions(self, skill: RetrievedSkill, query: Query) -> bool:
        """
        Check if cooperation skill preconditions and triggers are met.

        Cooperation skills are important for multi-agent coordination and should be
        retrieved even when partner_effects is not yet available, as they provide
        useful coordination patterns for the agent to follow.
        """
        # Get skill data
        skill_key = f"{skill.skill_name}_cooperation"
        skill_data = self.L_coop.get(skill_key, {})

        # Check trigger conditions based on partner effects
        trigger_conditions = skill_data.get("trigger_conditions", [])
        partner_effects = query.partner_effects

        # IMPORTANT: Don't filter out cooperation skills just because partner_effects is empty.
        # Cooperation skills provide valuable patterns even at the start of execution.
        # The agent needs to see these patterns to coordinate effectively.

        # If no trigger conditions defined, skill is always applicable
        if len(trigger_conditions) == 0:
            return True

        # If partner effects exist, check if any trigger is satisfied
        if partner_effects:
            for trigger in trigger_conditions:
                if "object_moved_to_target_location" in trigger:
                    if partner_effects.get("moved_objects"):
                        return True
                if "partner_action_in_progress" in trigger:
                    if partner_effects.get("action"):
                        return True
                if "heterogeneous_task_requirements" in trigger:
                    return True
                if "task_decomposable_into_subtasks" in trigger:
                    return True

        # Even without partner effects, allow cooperation skills if:
        # 1. The goal suggests multi-agent coordination (e.g., "together", "both", "each")
        # 2. The skill is highly relevant (abstract_score > 0.5)
        goal_lower = query.goal.lower()
        cooperation_keywords = ["together", "both", "each", "while", "and", "then", "after"]
        if any(kw in goal_lower for kw in cooperation_keywords):
            return True

        # Allow cooperation skills with high relevance even without explicit triggers
        # This ensures agents can learn coordination patterns proactively
        if skill.abstract_score > 0.5:
            return True

        return False

    def retrieve(
        self,
        agent_state: Dict[str, Any],
        environment_state: Dict[str, Any],
        partner_effects: Dict[str, Any],
        goal: str,
        include_individual: bool = True,
        include_cooperation: bool = True
    ) -> List[RetrievedSkill]:
        """
        Full hierarchical retrieval pipeline

        Args:
            agent_state: Agent's internal state
            environment_state: Observable environment state
            partner_effects: Observed effects of partner's action
            goal: Current goal
            include_individual: Whether to include individual skills
            include_cooperation: Whether to include cooperation skills

        Returns:
            List of executable skills sorted by relevance
        """
        # Stage 1: Query Generation
        query = self.generate_query(agent_state, environment_state, partner_effects, goal)

        # Stage 2: Abstract Skill Matching
        candidates = self.match_abstract_skills(query)

        # Filter by skill type if needed
        if not include_individual:
            candidates = {k: v for k, v in candidates.items() if k in self.L_coop}
        if not include_cooperation:
            candidates = {k: v for k, v in candidates.items() if k in self.L_ind}

        # Stage 3: Instance Retrieval
        retrieved = self.retrieve_instances(candidates, query)

        # Stage 4: Executability Filtering
        executable = self.filter_executable(retrieved, query)

        # Sort by combined score
        executable.sort(key=lambda s: s.abstract_score * (sum(s.instance_scores) / len(s.instance_scores) if s.instance_scores else 0), reverse=True)

        return executable

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _keyword_matching(self, query: Query) -> Dict[str, float]:
        """Fallback keyword matching for abstract skill matching"""
        candidates = {}
        query_words = set(query.text_query.lower().split())

        all_skills = {**self.L_ind, **self.L_coop}

        for skill_key, skill in all_skills.items():
            skill_words = set(skill["name"].lower().split())
            skill_words.update(skill["description"].lower().split())

            # Jaccard similarity
            intersection = len(query_words & skill_words)
            union = len(query_words | skill_words)

            if union > 0:
                sim = intersection / union
                if sim > self.abstract_threshold:
                    candidates[skill_key] = sim

        return candidates

    def _build_context_query(self, query: Query) -> str:
        """Build context text from query for instance matching"""
        parts = []

        # Objects in environment
        env_objects = query.environment_state.get("objects", {})
        if env_objects:
            obj_names = list(env_objects.keys())[:10]  # Limit to 10
            parts.append(f"objects: {', '.join(obj_names)}")

        # Locations
        furniture = query.environment_state.get("furniture", {})
        if furniture:
            locs = list(furniture.keys())[:10]
            parts.append(f"locations: {', '.join(locs)}")

        if ROLE_AWARE_CONTEXT:
            # The query side must speak the same vocabulary as the instance side,
            # otherwise embedding roles into instances only adds noise. The partner
            # actions are the observed half of the coordination pattern.
            effects = query.partner_effects or {}
            if effects.get("action"):
                parts.append(f"roles: partner performs {effects['action']}")
            if effects.get("moved_objects"):
                parts.append(
                    "coordination: partner handles "
                    + ", ".join(str(item) for item in effects["moved_objects"])
                )

        # Goal
        parts.append(f"goal: {query.goal}")

        return " ".join(parts)

    def format_for_prompt(
        self,
        retrieved_skills: List[RetrievedSkill],
        max_examples: int = 3
    ) -> str:
        """
        Format retrieved skills for inclusion in LLM prompt

        Args:
            retrieved_skills: List of retrieved skills
            max_examples: Maximum number of examples to include

        Returns:
            Formatted string for prompt
        """
        if not retrieved_skills:
            return "No relevant skills found in memory."

        parts = ["## Retrieved Skills from Memory\n"]

        for i, skill in enumerate(retrieved_skills[:max_examples]):
            parts.append(f"### Skill {i+1}: {skill.skill_name} ({skill.skill_type})")
            parts.append(f"Relevance: {skill.abstract_score:.2f}")
            parts.append(f"Demo: {skill.demo}")

            if skill.instances:
                inst = skill.instances[0]
                context = inst.get("context", {})
                if context.get("objects"):
                    parts.append(f"Objects: {', '.join(context['objects'][:5])}")
                if context.get("locations"):
                    parts.append(f"Locations: {', '.join(context['locations'][:5])}")
                if context.get("rooms"):
                    parts.append(f"Rooms: {', '.join(context['rooms'][:5])}")
                if context.get("object_locations"):
                    obj_loc_strs = []
                    for obj, info in list(context["object_locations"].items())[:5]:
                        if isinstance(info, dict):
                            loc = info.get("location", "unknown")
                            room = info.get("room", "")
                            if room:
                                obj_loc_strs.append(f"{obj} at {loc} in {room}")
                            else:
                                obj_loc_strs.append(f"{obj} at {loc}")
                        else:
                            obj_loc_strs.append(f"{obj} at {info}")
                    if obj_loc_strs:
                        parts.append(f"Object Locations: {'; '.join(obj_loc_strs)}")
                if context.get("action_sequence"):
                    # The stored sequence runs 11-12 actions for a two-agent skill.
                    # Cutting it at five removed the tail of every clear_table skill,
                    # and the tail is where Open[cabinet] lives -- the one action that
                    # family cannot be solved without. The cap stays at five by
                    # default so the filed arms reproduce byte for byte; set
                    # A9_ACTION_CAP=0 to render the whole sequence.
                    cap = int(os.environ.get("A9_ACTION_CAP", "5"))
                    shown = context["action_sequence"]
                    if cap > 0:
                        shown = shown[:cap]
                    parts.append(f"Actions: {' -> '.join(shown)}")

            parts.append("")

        return "\n".join(parts)


def create_retriever(memory_path: str) -> HierarchicalRetriever:
    """Factory function to create a retriever"""
    return HierarchicalRetriever(memory_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test hierarchical retrieval")
    parser.add_argument(
        "--memory-path",
        type=str,
        default="/home/a5l/shuqing.a5l/partnr-planner/data/hierarchical_skill_memory",
        help="Path to hierarchical memory"
    )

    args = parser.parse_args()

    # Create retriever
    retriever = create_retriever(args.memory_path)

    # Test retrieval
    test_goal = "Move the cup from the table to the kitchen counter"
    test_agent_state = {"holding": None, "position": "living_room"}
    test_env_state = {"objects": {"cup_0": {"location": "table_1"}}, "furniture": {"table_1": "living_room"}}
    test_partner_effects = {"action": "Navigate", "moved_objects": []}

    results = retriever.retrieve(
        agent_state=test_agent_state,
        environment_state=test_env_state,
        partner_effects=test_partner_effects,
        goal=test_goal
    )

    print("\nRetrieval Results:")
    print(retriever.format_for_prompt(results))
