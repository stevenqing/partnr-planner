#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Graph Builder: Update and Build User Profile Memory

This module implements the knowledge graph update algorithm from MEMENTO.

Algorithm: UpdateKnowledgeGraph
Input: User instruction I, Knowledge graph G
Output: Updated knowledge graph G'

1. EXTRACT CANDIDATE KNOWLEDGE
   K <- ExtractKnowledge(I)    // K = {(k_1, t_1), (k_2, t_2), ...}
   C <- empty

2. FOR EACH KNOWLEDGE (k, t_k) in K:
   a. EXTRACT ELEMENTS
      E <- ExtractElements(I, k, t_k)

   b. SELECT SUBGRAPH BY TYPE
      G_t <- Subgraph(G, type = t_k)

   c. RETRIEVE CANDIDATES
      C <- C union SimilaritySearch(G_t, k, t_k)
      FOR EACH (e, t_e) in E:
         C <- C union SimilaritySearch(G_t, e, t_e)

3. LLM RESOLUTION
   op <- LLMResolve(I, C)

   IF op = update THEN
      RETURN G' <- UpdateKnowledgeNode(G, Expand(C), I)
   ELSE IF op = add THEN
      RETURN G' <- AddKnowledgeNode(G, Expand(C), I)
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from .user_profile_memory import (
    UserProfileMemory,
    KnowledgeNode,
    ObjectNode,
    PatternNode,
    LocationNode,
    KnowledgeType,
    EdgeType,
    ObjectSemanticSubtype,
    UserPatternSubtype,
)


@dataclass
class ExtractedKnowledge:
    """Extracted knowledge from instruction."""
    name: str
    knowledge_type: KnowledgeType
    subtype: Optional[str] = None
    alias: Optional[str] = None
    description: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ExtractedElement:
    """Extracted element (object, pattern, location) from instruction."""
    element_type: str  # "object", "pattern", "location"
    name: str
    attributes: Dict[str, Any] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


class KnowledgeExtractor:
    """
    Extract personalized knowledge from user instructions.

    This class identifies two types of personalized knowledge:
    1. Object Semantics: Ownership, preference, history, groups
    2. User Patterns: Routine, preference
    """

    # Patterns for detecting object semantics
    OWNERSHIP_PATTERNS = [
        r"\bmy\s+(\w+(?:\s+\w+)*)",
        r"\bmine\b",
        r"\bour\s+(\w+(?:\s+\w+)*)",
    ]

    PREFERENCE_PATTERNS = [
        r"\bfavorite\s+(\w+(?:\s+\w+)*)",
        r"\bpreferred\s+(\w+(?:\s+\w+)*)",
        r"\bi\s+(?:like|love|prefer)\s+(\w+(?:\s+\w+)*)",
    ]

    HISTORY_PATTERNS = [
        r"\bgift\s+from\s+(\w+(?:\s+\w+)*)",
        r"\bremember(?:s)?\s+(\w+(?:\s+\w+)*)",
        r"\bfrom\s+(?:that|the)\s+time",
    ]

    GROUP_PATTERNS = [
        r"\b(\w+)\s+set\b",
        r"\b(\w+)\s+collection\b",
        r"\bsetup\b",
        r"\bessentials\b",
    ]

    # Patterns for detecting user patterns
    ROUTINE_PATTERNS = [
        r"\bmorning\s+(?:routine|ritual)",
        r"\bevening\s+(?:routine|ritual)",
        r"\bdaily\s+(?:routine|ritual)",
        r"\bwhen\s+(?:i|we)\s+(\w+)",
        r"\balways\s+(\w+)",
        r"\busually\s+(\w+)",
    ]

    PATTERN_PREFERENCE_PATTERNS = [
        r"\bi\s+(?:like|want|prefer)\s+(?:it|them)\s+(\w+)",
        r"\bkeep\s+(?:it|them)\s+(\w+)",
        r"\bplace\s+(?:it|them)\s+(\w+)",
    ]

    def __init__(self, use_llm: bool = False, llm_model: Optional[Any] = None):
        """
        Initialize the knowledge extractor.

        Args:
            use_llm: Whether to use LLM for extraction
            llm_model: LLM model for extraction (if use_llm is True)
        """
        self.use_llm = use_llm
        self.llm_model = llm_model

    def extract_knowledge(
        self,
        instruction: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractedKnowledge]:
        """
        Extract personalized knowledge from instruction.

        Args:
            instruction: User instruction
            context: Optional context information

        Returns:
            List of extracted knowledge items
        """
        if self.use_llm and self.llm_model:
            return self._extract_with_llm(instruction, context)
        else:
            return self._extract_with_patterns(instruction)

    def _extract_with_patterns(self, instruction: str) -> List[ExtractedKnowledge]:
        """Extract knowledge using pattern matching."""
        knowledge_list = []
        instruction_lower = instruction.lower()

        # Check for object semantics
        for pattern in self.OWNERSHIP_PATTERNS:
            matches = re.findall(pattern, instruction_lower)
            for match in matches:
                knowledge_list.append(ExtractedKnowledge(
                    name=f"ownership_{match}" if isinstance(match, str) else f"ownership_{match[0]}",
                    knowledge_type=KnowledgeType.OBJECT_SEMANTICS,
                    subtype=ObjectSemanticSubtype.OWNERSHIP.value,
                    alias=f"my {match}" if isinstance(match, str) else f"my {match[0]}",
                    confidence=0.8,
                ))

        for pattern in self.PREFERENCE_PATTERNS:
            matches = re.findall(pattern, instruction_lower)
            for match in matches:
                knowledge_list.append(ExtractedKnowledge(
                    name=f"preference_{match}" if isinstance(match, str) else f"preference_{match[0]}",
                    knowledge_type=KnowledgeType.OBJECT_SEMANTICS,
                    subtype=ObjectSemanticSubtype.PREFERENCE.value,
                    alias=f"favorite {match}" if isinstance(match, str) else f"favorite {match[0]}",
                    confidence=0.8,
                ))

        for pattern in self.HISTORY_PATTERNS:
            matches = re.findall(pattern, instruction_lower)
            for match in matches:
                knowledge_list.append(ExtractedKnowledge(
                    name=f"history_{match}" if isinstance(match, str) else f"history_{match[0]}",
                    knowledge_type=KnowledgeType.OBJECT_SEMANTICS,
                    subtype=ObjectSemanticSubtype.HISTORY.value,
                    confidence=0.7,
                ))

        for pattern in self.GROUP_PATTERNS:
            matches = re.findall(pattern, instruction_lower)
            for match in matches:
                knowledge_list.append(ExtractedKnowledge(
                    name=f"group_{match}" if isinstance(match, str) else f"group_{match[0]}",
                    knowledge_type=KnowledgeType.OBJECT_SEMANTICS,
                    subtype=ObjectSemanticSubtype.GROUPS.value,
                    alias=f"{match} set" if isinstance(match, str) else f"{match[0]} set",
                    confidence=0.7,
                ))

        # Check for user patterns
        for pattern in self.ROUTINE_PATTERNS:
            if re.search(pattern, instruction_lower):
                knowledge_list.append(ExtractedKnowledge(
                    name=f"routine_{instruction[:30]}",
                    knowledge_type=KnowledgeType.USER_PATTERN,
                    subtype=UserPatternSubtype.ROUTINE.value,
                    description=instruction,
                    confidence=0.7,
                ))
                break

        for pattern in self.PATTERN_PREFERENCE_PATTERNS:
            if re.search(pattern, instruction_lower):
                knowledge_list.append(ExtractedKnowledge(
                    name=f"pattern_preference_{instruction[:30]}",
                    knowledge_type=KnowledgeType.USER_PATTERN,
                    subtype=UserPatternSubtype.PREFERENCE.value,
                    description=instruction,
                    confidence=0.7,
                ))
                break

        return knowledge_list

    def _extract_with_llm(
        self,
        instruction: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge using LLM."""
        prompt = f"""Extract personalized knowledge from the following instruction.

Instruction: {instruction}

Identify any of the following types of knowledge:
1. Object Semantics:
   - Ownership: References to user's belongings ("my cup")
   - Preference: User's favorite items
   - History: Objects linked to personal memories
   - Groups: Collections or sets of related objects

2. User Patterns:
   - Routine: Regular activities or sequences
   - Preference: Preferred arrangements or placements

Return a JSON list of extracted knowledge with fields:
- name: short identifier
- knowledge_type: "object_semantics" or "user_pattern"
- subtype: specific category
- alias: how the user refers to it
- description: what it means
- confidence: 0.0-1.0

JSON output:"""

        try:
            response = self.llm_model.generate(prompt)
            knowledge_data = json.loads(response)

            return [
                ExtractedKnowledge(
                    name=k["name"],
                    knowledge_type=KnowledgeType(k["knowledge_type"]),
                    subtype=k.get("subtype"),
                    alias=k.get("alias"),
                    description=k.get("description"),
                    confidence=k.get("confidence", 0.8),
                )
                for k in knowledge_data
            ]
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return self._extract_with_patterns(instruction)


class ElementExtractor:
    """
    Extract elements (objects, patterns, locations) from instructions.
    """

    # Common object patterns
    OBJECT_PATTERNS = [
        r"\b(cup|mug|plate|bowl|spoon|fork|knife|glass)\b",
        r"\b(book|laptop|phone|tablet|charger)\b",
        r"\b(toy|ball|doll|truck|car)\b",
        r"\b(plant|flower|vase)\b",
        r"\b(lamp|clock|picture|frame)\b",
        r"\b(\w+)_\d+\b",  # Object instances like "cup_0"
    ]

    # Location patterns
    LOCATION_PATTERNS = [
        r"\b(?:on|in|at|near|next to)\s+(?:the\s+)?(\w+(?:\s+\w+)?)\b",
        r"\b(kitchen|bedroom|living room|bathroom|dining room|office)\b",
        r"\b(table|desk|shelf|counter|couch|sofa|bed|chair)\b",
        r"\b(\w+)_\d+\b",  # Furniture instances like "table_0"
    ]

    # Action patterns
    ACTION_PATTERNS = [
        r"\b(move|place|put|pick|pick up|take|bring|rearrange)\b",
        r"\b(arrange|organize|set up|prepare)\b",
    ]

    def __init__(self, use_llm: bool = False, llm_model: Optional[Any] = None):
        self.use_llm = use_llm
        self.llm_model = llm_model

    def extract_elements(
        self,
        instruction: str,
        knowledge: ExtractedKnowledge,
    ) -> List[ExtractedElement]:
        """
        Extract elements associated with a knowledge item.

        Args:
            instruction: User instruction
            knowledge: The knowledge this element relates to

        Returns:
            List of extracted elements
        """
        if self.use_llm and self.llm_model:
            return self._extract_with_llm(instruction, knowledge)
        else:
            return self._extract_with_patterns(instruction)

    def _extract_with_patterns(self, instruction: str) -> List[ExtractedElement]:
        """Extract elements using pattern matching."""
        elements = []
        instruction_lower = instruction.lower()

        # Extract objects
        for pattern in self.OBJECT_PATTERNS:
            matches = re.findall(pattern, instruction_lower)
            for match in matches:
                elements.append(ExtractedElement(
                    element_type="object",
                    name=match,
                    attributes={"source": "pattern"},
                ))

        # Extract locations
        for pattern in self.LOCATION_PATTERNS:
            matches = re.findall(pattern, instruction_lower)
            for match in matches:
                # Skip if already added as object
                if not any(e.name == match and e.element_type == "object" for e in elements):
                    elements.append(ExtractedElement(
                        element_type="location",
                        name=match,
                        attributes={"source": "pattern"},
                    ))

        # Extract actions as patterns
        for pattern in self.ACTION_PATTERNS:
            matches = re.findall(pattern, instruction_lower)
            for match in matches:
                elements.append(ExtractedElement(
                    element_type="pattern",
                    name=match,
                    attributes={"source": "pattern"},
                ))

        return elements

    def _extract_with_llm(
        self,
        instruction: str,
        knowledge: ExtractedKnowledge,
    ) -> List[ExtractedElement]:
        """Extract elements using LLM."""
        prompt = f"""Extract specific elements from the following instruction related to the knowledge item.

Instruction: {instruction}
Knowledge: {knowledge.name} ({knowledge.knowledge_type.value})

Identify:
1. Objects: Specific items mentioned
2. Locations: Places or furniture mentioned
3. Patterns/Actions: Actions or sequences described

Return a JSON list with fields:
- element_type: "object", "location", or "pattern"
- name: name of the element
- attributes: any relevant attributes

JSON output:"""

        try:
            response = self.llm_model.generate(prompt)
            element_data = json.loads(response)

            return [
                ExtractedElement(
                    element_type=e["element_type"],
                    name=e["name"],
                    attributes=e.get("attributes", {}),
                )
                for e in element_data
            ]
        except Exception as e:
            print(f"LLM element extraction failed: {e}")
            return self._extract_with_patterns(instruction)


class KnowledgeGraphBuilder:
    """
    Build and update the user profile memory knowledge graph.

    This implements the UpdateKnowledgeGraph algorithm from MEMENTO.
    """

    def __init__(
        self,
        memory: Optional[UserProfileMemory] = None,
        similarity_threshold: float = 0.7,
        embedding_model: str = "all-mpnet-base-v2",
        use_llm: bool = False,
        llm_model: Optional[Any] = None,
    ):
        """
        Initialize the knowledge graph builder.

        Args:
            memory: Existing user profile memory (or None to create new)
            similarity_threshold: Threshold for similarity matching
            embedding_model: Sentence transformer model name
            use_llm: Whether to use LLM for extraction and resolution
            llm_model: LLM model instance
        """
        self.memory = memory or UserProfileMemory()
        self.similarity_threshold = similarity_threshold
        self.use_llm = use_llm
        self.llm_model = llm_model

        # Initialize extractors
        self.knowledge_extractor = KnowledgeExtractor(use_llm, llm_model)
        self.element_extractor = ElementExtractor(use_llm, llm_model)

        # Initialize embedding model
        self._embedding_model = None
        self._embedding_model_name = embedding_model
        self._knowledge_embeddings: Dict[str, np.ndarray] = {}

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def update(
        self,
        instruction: str,
        user_id: str = "user_0",
        trajectory: Optional[List[Dict[str, Any]]] = None,
    ) -> UserProfileMemory:
        """
        Update the knowledge graph with a new instruction.

        Args:
            instruction: User instruction
            user_id: ID of the user
            trajectory: Optional trajectory data (thoughts, actions, observations)

        Returns:
            Updated UserProfileMemory
        """
        # Ensure user exists
        if user_id not in self.memory.users:
            self.memory.add_user(user_id, name=user_id)

        # Step 1: Extract candidate knowledge
        knowledge_list = self.knowledge_extractor.extract_knowledge(instruction)

        if not knowledge_list:
            # Try to extract from trajectory if available
            if trajectory:
                knowledge_list = self._extract_from_trajectory(trajectory)

        for knowledge in knowledge_list:
            # Step 2a: Extract elements
            elements = self.element_extractor.extract_elements(instruction, knowledge)

            # Step 2b: Get subgraph by type
            subgraph = self.memory.get_subgraph_by_type(knowledge.knowledge_type)

            # Step 2c: Retrieve candidates via similarity search
            candidates = self._similarity_search(subgraph, knowledge, elements)

            # Step 3: Resolve with LLM or heuristics
            operation, target_id = self._resolve_operation(
                instruction, knowledge, candidates
            )

            if operation == "update":
                self._update_knowledge_node(target_id, knowledge, elements, instruction)
            else:  # operation == "add"
                self._add_knowledge_node(user_id, knowledge, elements, instruction)

        return self.memory

    def _extract_from_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge from trajectory data."""
        knowledge_list = []

        # Analyze trajectory for patterns
        action_sequence = []
        objects_involved = set()
        locations_involved = set()

        for step in trajectory:
            if "action" in step:
                action = step["action"]
                action_sequence.append(action)

                # Parse action to extract objects and locations
                if "Rearrange" in action or "Pick" in action or "Place" in action:
                    # Extract object and location from action string
                    match = re.search(r"\[([^\]]+)\]", action)
                    if match:
                        args = match.group(1).split(",")
                        if args:
                            objects_involved.add(args[0].strip())
                        if len(args) > 2:
                            locations_involved.add(args[2].strip())

        # Create user pattern knowledge if we have a sequence
        if len(action_sequence) >= 2:
            knowledge_list.append(ExtractedKnowledge(
                name=f"pattern_{'_'.join(action_sequence[:3])}",
                knowledge_type=KnowledgeType.USER_PATTERN,
                subtype=UserPatternSubtype.ROUTINE.value,
                description=f"Action sequence: {' -> '.join(action_sequence)}",
                confidence=0.6,
            ))

        # Create object semantics if objects are mentioned together
        if len(objects_involved) > 1:
            knowledge_list.append(ExtractedKnowledge(
                name=f"group_{'_'.join(list(objects_involved)[:2])}",
                knowledge_type=KnowledgeType.OBJECT_SEMANTICS,
                subtype=ObjectSemanticSubtype.GROUPS.value,
                alias=f"set of {', '.join(objects_involved)}",
                confidence=0.5,
            ))

        return knowledge_list

    def _similarity_search(
        self,
        subgraph: UserProfileMemory,
        knowledge: ExtractedKnowledge,
        elements: List[ExtractedElement],
    ) -> List[Tuple[str, float]]:
        """
        Perform similarity search in the subgraph.

        Returns list of (node_id, similarity_score) tuples.
        """
        candidates = []

        # Get query embedding
        query_text = knowledge.alias or knowledge.name
        query_embedding = self.embedding_model.encode(query_text)

        # Search knowledge nodes
        for node_id, node in subgraph.nodes.items():
            if isinstance(node, KnowledgeNode):
                # Get or compute embedding for this node
                node_embedding = self._get_node_embedding(node)
                similarity = self._cosine_similarity(query_embedding, node_embedding)

                if similarity >= self.similarity_threshold:
                    candidates.append((node_id, similarity))

        # Also search based on elements
        for element in elements:
            element_embedding = self.embedding_model.encode(element.name)
            for node_id, node in subgraph.nodes.items():
                if isinstance(node, (ObjectNode, LocationNode, PatternNode)):
                    node_embedding = self._get_node_embedding(node)
                    similarity = self._cosine_similarity(element_embedding, node_embedding)

                    if similarity >= self.similarity_threshold * 0.8:  # Lower threshold for elements
                        # Find parent knowledge node
                        parent_ids = self._find_parent_knowledge(subgraph, node_id)
                        for parent_id in parent_ids:
                            if parent_id not in [c[0] for c in candidates]:
                                candidates.append((parent_id, similarity * 0.9))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates

    def _get_node_embedding(self, node) -> np.ndarray:
        """Get or compute embedding for a node."""
        if node.id in self._knowledge_embeddings:
            return self._knowledge_embeddings[node.id]

        # Build text representation
        if isinstance(node, KnowledgeNode):
            text = node.alias or node.description or node.name
        elif isinstance(node, ObjectNode):
            text = f"{node.name} {' '.join(node.object_attributes)}"
        elif isinstance(node, LocationNode):
            text = f"{node.name} {node.expression}"
        elif isinstance(node, PatternNode):
            text = f"{node.action_name} {' '.join(node.args)}"
        else:
            text = node.name

        embedding = self.embedding_model.encode(text)
        self._knowledge_embeddings[node.id] = embedding

        return embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _find_parent_knowledge(
        self,
        subgraph: UserProfileMemory,
        node_id: str,
    ) -> List[str]:
        """Find parent knowledge nodes for a given node."""
        parents = []
        for edge in subgraph._edges_by_target.get(node_id, []):
            if edge.source_id in subgraph.nodes:
                source = subgraph.nodes[edge.source_id]
                if isinstance(source, KnowledgeNode):
                    parents.append(edge.source_id)
                else:
                    # Recurse to find knowledge parent
                    parents.extend(self._find_parent_knowledge(subgraph, edge.source_id))
        return parents

    def _resolve_operation(
        self,
        instruction: str,
        knowledge: ExtractedKnowledge,
        candidates: List[Tuple[str, float]],
    ) -> Tuple[str, Optional[str]]:
        """
        Resolve whether to update existing or add new knowledge.

        Returns:
            Tuple of (operation, target_id) where operation is "update" or "add"
        """
        if not candidates:
            return ("add", None)

        top_candidate_id, top_similarity = candidates[0]

        if self.use_llm and self.llm_model:
            return self._resolve_with_llm(instruction, knowledge, candidates)
        else:
            # Heuristic: update if similarity is very high
            if top_similarity >= 0.85:
                return ("update", top_candidate_id)
            else:
                return ("add", None)

    def _resolve_with_llm(
        self,
        instruction: str,
        knowledge: ExtractedKnowledge,
        candidates: List[Tuple[str, float]],
    ) -> Tuple[str, Optional[str]]:
        """Use LLM to resolve operation."""
        # Format candidates for prompt
        candidate_descriptions = []
        for node_id, sim in candidates[:3]:
            node = self.memory.nodes.get(node_id)
            if isinstance(node, KnowledgeNode):
                desc = f"- {node.alias or node.name}: {node.description or 'No description'} (similarity: {sim:.2f})"
                candidate_descriptions.append(desc)

        prompt = f"""Determine if the new knowledge should update an existing entry or create a new one.

New Knowledge:
- Name: {knowledge.name}
- Type: {knowledge.knowledge_type.value}
- Alias: {knowledge.alias}
- Description: {knowledge.description}

Existing Candidates:
{chr(10).join(candidate_descriptions) if candidate_descriptions else "None"}

Instruction: {instruction}

Should we UPDATE an existing entry or ADD a new one?
Answer with JSON: {{"operation": "update" or "add", "target_id": "id if update else null"}}

JSON output:"""

        try:
            response = self.llm_model.generate(prompt)
            result = json.loads(response)
            return (result["operation"], result.get("target_id"))
        except Exception as e:
            print(f"LLM resolution failed: {e}")
            if candidates and candidates[0][1] >= 0.85:
                return ("update", candidates[0][0])
            return ("add", None)

    def _add_knowledge_node(
        self,
        user_id: str,
        knowledge: ExtractedKnowledge,
        elements: List[ExtractedElement],
        instruction: str,
    ):
        """Add a new knowledge node to the graph."""
        # Create knowledge node
        knowledge_id = self.memory.generate_id("k")
        self.memory.add_knowledge(
            knowledge_id=knowledge_id,
            name=knowledge.name,
            knowledge_type=knowledge.knowledge_type,
            user_id=user_id,
            subtype=knowledge.subtype,
            alias=knowledge.alias,
            description=knowledge.description or instruction,
        )

        # Add elements
        pattern_nodes = []
        for element in elements:
            if element.element_type == "object":
                obj_id = self.memory.generate_id("o")
                self.memory.add_object(
                    object_id=obj_id,
                    name=element.name,
                    knowledge_id=knowledge_id,
                    granularity=element.attributes.get("granularity", "instance"),
                    object_attributes=element.attributes.get("attributes", []),
                )

            elif element.element_type == "location":
                loc_id = self.memory.generate_id("l")
                self.memory.add_location(
                    location_id=loc_id,
                    name=element.name,
                    expression=element.attributes.get("expression", f"at {element.name}"),
                )
                # Link to knowledge
                self.memory.add_edge(knowledge_id, loc_id, EdgeType.TARGET_LOCATION)

            elif element.element_type == "pattern":
                pattern_id = self.memory.generate_id("p")
                pattern_node = self.memory.add_pattern(
                    pattern_id=pattern_id,
                    name=element.name,
                    knowledge_id=knowledge_id,
                    action_name=element.name,
                    args=element.attributes.get("args", []),
                )
                pattern_nodes.append(pattern_node)

        # Add temporal edges between patterns
        for i in range(len(pattern_nodes) - 1):
            self.memory.add_temporal_edge(pattern_nodes[i].id, pattern_nodes[i + 1].id)

    def _update_knowledge_node(
        self,
        knowledge_id: str,
        knowledge: ExtractedKnowledge,
        elements: List[ExtractedElement],
        instruction: str,
    ):
        """Update an existing knowledge node."""
        node = self.memory.nodes.get(knowledge_id)
        if not isinstance(node, KnowledgeNode):
            return

        # Update node attributes
        if knowledge.alias and not node.alias:
            node.alias = knowledge.alias
        if knowledge.description:
            if node.description:
                node.description = f"{node.description}; {knowledge.description}"
            else:
                node.description = knowledge.description

        # Add new elements
        existing_elements = self.memory.get_elements_for_knowledge(knowledge_id)
        existing_object_names = {obj.name for obj in existing_elements["objects"]}
        existing_location_names = {loc.name for loc in existing_elements["locations"]}

        for element in elements:
            if element.element_type == "object" and element.name not in existing_object_names:
                obj_id = self.memory.generate_id("o")
                self.memory.add_object(
                    object_id=obj_id,
                    name=element.name,
                    knowledge_id=knowledge_id,
                )

            elif element.element_type == "location" and element.name not in existing_location_names:
                loc_id = self.memory.generate_id("l")
                self.memory.add_location(
                    location_id=loc_id,
                    name=element.name,
                    expression=element.attributes.get("expression", f"at {element.name}"),
                )
                self.memory.add_edge(knowledge_id, loc_id, EdgeType.TARGET_LOCATION)

    def build_from_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
        user_id: str = "user_0",
    ) -> UserProfileMemory:
        """
        Build memory from multiple trajectories.

        Args:
            trajectories: List of trajectory dictionaries with 'instruction' and 'steps'
            user_id: User ID to associate with knowledge

        Returns:
            Built UserProfileMemory
        """
        for trajectory in trajectories:
            instruction = trajectory.get("instruction", "")
            steps = trajectory.get("steps", [])

            self.update(instruction, user_id, steps)

        return self.memory

    def save(self, path: Union[str, Path]):
        """Save the memory to a file."""
        self.memory.save(path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        **kwargs,
    ) -> "KnowledgeGraphBuilder":
        """Load a builder with existing memory."""
        memory = UserProfileMemory.load(path)
        return cls(memory=memory, **kwargs)


def build_memory_from_trajectories(
    trajectory_path: Union[str, Path],
    output_path: Union[str, Path],
    user_id: str = "user_0",
    use_llm: bool = False,
    llm_model: Optional[Any] = None,
) -> UserProfileMemory:
    """
    Build user profile memory from trajectory files.

    Args:
        trajectory_path: Path to trajectory file or directory
        output_path: Path to save the built memory
        user_id: User ID
        use_llm: Whether to use LLM for extraction
        llm_model: LLM model instance

    Returns:
        Built UserProfileMemory
    """
    import glob

    trajectory_path = Path(trajectory_path)
    output_path = Path(output_path)

    builder = KnowledgeGraphBuilder(use_llm=use_llm, llm_model=llm_model)

    # Load trajectories
    if trajectory_path.is_file():
        with open(trajectory_path, 'r') as f:
            trajectories = json.load(f)
    else:
        trajectories = []
        for trace_file in glob.glob(str(trajectory_path / "**/*.json"), recursive=True):
            with open(trace_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    trajectories.extend(data)
                else:
                    trajectories.append(data)

    print(f"Building memory from {len(trajectories)} trajectories...")

    # Build memory
    memory = builder.build_from_trajectories(trajectories, user_id)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    memory.save(output_path)

    return memory
