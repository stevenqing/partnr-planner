#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEMENTO Retriever: Retrieve Personalized Knowledge from User Profile Memory

This module implements the retrieval procedure from MEMENTO.

Algorithm: RetrievalProcedure
Input: User instruction I, User profile memory graph G
Output: Retrieved natural language descriptions R

1. PARSE INSTRUCTION
   K <- ExtractKnowledge(I)
   C, R <- empty

2. FILTER AND SEARCH
   FOR EACH (k, t_k) in K:
      G_t <- Subgraph(G, type = t)
      C <- C union SimilaritySearch(G_t, k, t_k)

   C <- RemoveDuplicates(C)

3. REFORMULATE RESULTS
   FOR EACH (c, t^c) in C:
      R <- R union Reformulate(Expand(c))

RETURN R
"""

import json
from dataclasses import dataclass, field
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
)
from .knowledge_graph_builder import (
    KnowledgeExtractor,
    ExtractedKnowledge,
)


@dataclass
class RetrievedKnowledge:
    """Retrieved knowledge with relevance information."""
    knowledge_id: str
    knowledge_type: KnowledgeType
    alias: Optional[str]
    description: Optional[str]
    reformulated_text: str
    similarity_score: float
    objects: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "knowledge_id": self.knowledge_id,
            "knowledge_type": self.knowledge_type.value,
            "alias": self.alias,
            "description": self.description,
            "reformulated_text": self.reformulated_text,
            "similarity_score": self.similarity_score,
            "objects": self.objects,
            "patterns": self.patterns,
            "locations": self.locations,
        }


class MementoRetriever:
    """
    Retrieve personalized knowledge from user profile memory.

    This implements the full retrieval procedure from MEMENTO:
    1. Parse instruction to extract knowledge references
    2. Filter and search by knowledge type
    3. Reformulate results into natural language
    """

    def __init__(
        self,
        memory: UserProfileMemory,
        embedding_model: str = "all-mpnet-base-v2",
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        use_llm: bool = False,
        llm_model: Optional[Any] = None,
    ):
        """
        Initialize the retriever.

        Args:
            memory: User profile memory to search
            embedding_model: Sentence transformer model name
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity for retrieval
            use_llm: Whether to use LLM for extraction
            llm_model: LLM model instance
        """
        self.memory = memory
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_llm = use_llm
        self.llm_model = llm_model

        # Initialize knowledge extractor
        self.knowledge_extractor = KnowledgeExtractor(use_llm, llm_model)

        # Initialize embedding model lazily
        self._embedding_model = None
        self._embedding_model_name = embedding_model

        # Cache for embeddings
        self._knowledge_embeddings: Dict[str, np.ndarray] = {}
        self._build_knowledge_embeddings()

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def _build_knowledge_embeddings(self):
        """Pre-compute embeddings for all knowledge nodes."""
        for node_id, node in self.memory.nodes.items():
            if isinstance(node, KnowledgeNode):
                text = self._get_knowledge_text(node)
                self._knowledge_embeddings[node_id] = self.embedding_model.encode(text)

    def _get_knowledge_text(self, node: KnowledgeNode) -> str:
        """Get text representation of a knowledge node for embedding."""
        parts = []
        if node.alias:
            parts.append(node.alias)
        if node.description:
            parts.append(node.description)
        if node.name and node.name not in parts:
            parts.append(node.name)

        # Include element information
        elements = self.memory.get_elements_for_knowledge(node.id)
        for obj in elements["objects"]:
            parts.append(obj.name)
        for loc in elements["locations"]:
            parts.append(loc.expression or loc.name)

        return " ".join(parts)

    def retrieve(
        self,
        instruction: str,
        user_id: Optional[str] = None,
        knowledge_type: Optional[KnowledgeType] = None,
    ) -> List[RetrievedKnowledge]:
        """
        Retrieve relevant personalized knowledge for an instruction.

        Args:
            instruction: User instruction
            user_id: Optional user ID to filter by
            knowledge_type: Optional knowledge type to filter by

        Returns:
            List of RetrievedKnowledge sorted by relevance
        """
        # Step 1: Parse instruction to extract knowledge references
        extracted_knowledge = self.knowledge_extractor.extract_knowledge(instruction)

        candidates: List[Tuple[str, float]] = []

        # Step 2: Filter and search
        if extracted_knowledge:
            for knowledge in extracted_knowledge:
                # Get subgraph by type
                if knowledge_type:
                    subgraph = self.memory.get_subgraph_by_type(knowledge_type)
                else:
                    subgraph = self.memory.get_subgraph_by_type(knowledge.knowledge_type)

                # Similarity search
                query_text = knowledge.alias or knowledge.description or knowledge.name
                matches = self._similarity_search(query_text, subgraph, knowledge.knowledge_type)
                candidates.extend(matches)
        else:
            # Fallback: search using the full instruction
            if knowledge_type:
                subgraph = self.memory.get_subgraph_by_type(knowledge_type)
                matches = self._similarity_search(instruction, subgraph, knowledge_type)
            else:
                # Search both types
                for kt in [KnowledgeType.OBJECT_SEMANTICS, KnowledgeType.USER_PATTERN]:
                    subgraph = self.memory.get_subgraph_by_type(kt)
                    matches = self._similarity_search(instruction, subgraph, kt)
                    candidates.extend(matches)

        # Remove duplicates and sort
        unique_candidates = self._remove_duplicates(candidates)

        # Step 3: Reformulate results
        results = []
        for knowledge_id, score in unique_candidates[:self.top_k]:
            node = self.memory.nodes.get(knowledge_id)
            if isinstance(node, KnowledgeNode):
                reformulated = self._reformulate(knowledge_id)
                elements = self.memory.get_elements_for_knowledge(knowledge_id)

                results.append(RetrievedKnowledge(
                    knowledge_id=knowledge_id,
                    knowledge_type=node.knowledge_type,
                    alias=node.alias,
                    description=node.description,
                    reformulated_text=reformulated,
                    similarity_score=score,
                    objects=[obj.name for obj in elements["objects"]],
                    patterns=[p.action_name for p in elements["patterns"]],
                    locations=[loc.expression or loc.name for loc in elements["locations"]],
                ))

        return results

    def _similarity_search(
        self,
        query: str,
        subgraph: UserProfileMemory,
        knowledge_type: Optional[KnowledgeType] = None,
    ) -> List[Tuple[str, float]]:
        """
        Perform similarity search in a subgraph.

        Args:
            query: Query text
            subgraph: Subgraph to search in
            knowledge_type: Type of knowledge to filter

        Returns:
            List of (knowledge_id, similarity_score) tuples
        """
        candidates = []

        # Encode query
        query_embedding = self.embedding_model.encode(query)

        # Search knowledge nodes
        for node_id, node in subgraph.nodes.items():
            if not isinstance(node, KnowledgeNode):
                continue

            if knowledge_type and node.knowledge_type != knowledge_type:
                continue

            # Get embedding
            if node_id in self._knowledge_embeddings:
                node_embedding = self._knowledge_embeddings[node_id]
            else:
                text = self._get_knowledge_text(node)
                node_embedding = self.embedding_model.encode(text)
                self._knowledge_embeddings[node_id] = node_embedding

            # Compute similarity
            similarity = self._cosine_similarity(query_embedding, node_embedding)

            if similarity >= self.similarity_threshold:
                candidates.append((node_id, similarity))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _remove_duplicates(
        self,
        candidates: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Remove duplicate knowledge IDs, keeping highest score."""
        seen = {}
        for knowledge_id, score in candidates:
            if knowledge_id not in seen or score > seen[knowledge_id]:
                seen[knowledge_id] = score

        # Sort by score
        return sorted(seen.items(), key=lambda x: x[1], reverse=True)

    def _reformulate(self, knowledge_id: str) -> str:
        """
        Reformulate a knowledge node into natural language.

        This expands the knowledge node and converts it to readable text.
        """
        node = self.memory.nodes.get(knowledge_id)
        if not isinstance(node, KnowledgeNode):
            return ""

        # Use memory's reformulate function
        base_text = self.memory.reformulate_knowledge(knowledge_id)

        # Enhance with additional context
        if node.knowledge_type == KnowledgeType.OBJECT_SEMANTICS:
            return self._reformulate_object_semantics(node, base_text)
        else:
            return self._reformulate_user_pattern(node, base_text)

    def _reformulate_object_semantics(
        self,
        node: KnowledgeNode,
        base_text: str,
    ) -> str:
        """Reformulate object semantics knowledge."""
        parts = []

        if node.subtype == "ownership":
            parts.append(f"User's personal item: {base_text}")
        elif node.subtype == "preference":
            parts.append(f"User's preferred item: {base_text}")
        elif node.subtype == "history":
            parts.append(f"Item with personal significance: {base_text}")
        elif node.subtype == "groups":
            parts.append(f"Collection of related items: {base_text}")
        else:
            parts.append(f"Object semantics: {base_text}")

        return " ".join(parts)

    def _reformulate_user_pattern(
        self,
        node: KnowledgeNode,
        base_text: str,
    ) -> str:
        """Reformulate user pattern knowledge."""
        parts = []

        if node.subtype == "routine":
            parts.append(f"User's routine: {base_text}")
        elif node.subtype == "preference":
            parts.append(f"User's preferred arrangement: {base_text}")
        else:
            parts.append(f"User pattern: {base_text}")

        # Get pattern sequence
        elements = self.memory.get_elements_for_knowledge(node.id)
        if elements["patterns"]:
            pattern_seq = []
            for pattern in elements["patterns"]:
                if pattern.action_name:
                    if pattern.args:
                        pattern_seq.append(f"{pattern.action_name}({', '.join(pattern.args)})")
                    else:
                        pattern_seq.append(pattern.action_name)

            if pattern_seq:
                parts.append(f"Action sequence: {' -> '.join(pattern_seq)}")

        return " ".join(parts)

    def format_for_prompt(
        self,
        retrieved: List[RetrievedKnowledge],
        max_items: int = 3,
    ) -> str:
        """
        Format retrieved knowledge for inclusion in LLM prompt.

        Args:
            retrieved: List of retrieved knowledge
            max_items: Maximum number of items to include

        Returns:
            Formatted string for prompt
        """
        if not retrieved:
            return "No relevant personalized knowledge found."

        parts = ["## Retrieved Personalized Knowledge\n"]

        for i, item in enumerate(retrieved[:max_items]):
            parts.append(f"### Memory {i+1}")

            if item.alias:
                parts.append(f"**Reference:** {item.alias}")

            parts.append(f"**Type:** {item.knowledge_type.value.replace('_', ' ').title()}")
            parts.append(f"**Relevance:** {item.similarity_score:.2f}")

            parts.append(f"\n{item.reformulated_text}")

            if item.objects:
                parts.append(f"\n**Related Objects:** {', '.join(item.objects)}")

            if item.locations:
                parts.append(f"**Related Locations:** {', '.join(item.locations)}")

            parts.append("")

        return "\n".join(parts)

    def get_episodic_context(
        self,
        instruction: str,
        top_k: int = 3,
    ) -> str:
        """
        Get episodic context by retrieving relevant memories.

        This combines object semantics and user patterns to provide
        comprehensive context for task execution.

        Args:
            instruction: User instruction
            top_k: Number of memories to retrieve

        Returns:
            Formatted context string
        """
        # Retrieve from both knowledge types
        object_memories = self.retrieve(
            instruction,
            knowledge_type=KnowledgeType.OBJECT_SEMANTICS,
        )[:top_k // 2 + 1]

        pattern_memories = self.retrieve(
            instruction,
            knowledge_type=KnowledgeType.USER_PATTERN,
        )[:top_k // 2 + 1]

        # Combine and format
        all_memories = object_memories + pattern_memories

        # Sort by relevance and deduplicate
        seen_ids = set()
        unique_memories = []
        for m in sorted(all_memories, key=lambda x: x.similarity_score, reverse=True):
            if m.knowledge_id not in seen_ids:
                seen_ids.add(m.knowledge_id)
                unique_memories.append(m)

        return self.format_for_prompt(unique_memories[:top_k])

    def update_memory(self, memory: UserProfileMemory):
        """Update the memory and rebuild embeddings."""
        self.memory = memory
        self._knowledge_embeddings.clear()
        self._build_knowledge_embeddings()


def create_retriever(
    memory_path: Union[str, Path],
    **kwargs,
) -> MementoRetriever:
    """
    Create a retriever from a saved memory file.

    Args:
        memory_path: Path to saved UserProfileMemory
        **kwargs: Additional arguments for MementoRetriever

    Returns:
        Configured MementoRetriever
    """
    memory = UserProfileMemory.load(memory_path)
    return MementoRetriever(memory=memory, **kwargs)


class MementoRAG:
    """
    RAG interface compatible with the existing partnr-planner RAG system.

    This class provides a drop-in replacement for the RAG class
    that uses MEMENTO's user profile memory.
    """

    def __init__(
        self,
        memory_path: Union[str, Path],
        embedding_model: str = "all-mpnet-base-v2",
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        llm_config: Optional[Any] = None,
    ):
        """
        Initialize MEMENTO RAG.

        Args:
            memory_path: Path to user profile memory
            embedding_model: Embedding model name
            top_k: Default number of results to retrieve
            similarity_threshold: Minimum similarity threshold
            llm_config: LLM configuration (for compatibility)
        """
        self.memory = UserProfileMemory.load(memory_path)
        self.retriever = MementoRetriever(
            memory=self.memory,
            embedding_model=embedding_model,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        self._llm_config = llm_config
        self._example_type = "memento"

        # Build data_dict for compatibility
        self.data_dict = {}
        self._build_data_dict()

    def _build_data_dict(self):
        """Build data_dict for compatibility with existing RAG interface."""
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
                    "trace": self._format_as_trace(node, elements),
                    "agent_id": 0,
                    "objects": [obj.name for obj in elements["objects"]],
                    "patterns": [p.action_name for p in elements["patterns"]],
                    "locations": [loc.name for loc in elements["locations"]],
                }
                idx += 1

    def _format_as_trace(
        self,
        node: KnowledgeNode,
        elements: Dict[str, List],
    ) -> str:
        """Format knowledge as a trace-like string."""
        lines = []

        # Add knowledge description
        if node.alias:
            lines.append(f"Personalized Reference: {node.alias}")
        if node.description:
            lines.append(f"Description: {node.description}")

        # Add objects
        if elements["objects"]:
            obj_list = [obj.name for obj in elements["objects"]]
            lines.append(f"Related Objects: {', '.join(obj_list)}")

        # Add patterns
        if elements["patterns"]:
            for pattern in elements["patterns"]:
                if pattern.action_name:
                    args_str = ', '.join(pattern.args) if pattern.args else ''
                    lines.append(f"Action: {pattern.action_name}({args_str})")

        # Add locations
        if elements["locations"]:
            loc_list = [loc.expression or loc.name for loc in elements["locations"]]
            lines.append(f"Locations: {', '.join(loc_list)}")

        return "\n".join(lines)

    def retrieve_top_k_given_query(
        self,
        query: str,
        top_k: int = 1,
        agent_id: int = 0,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k personalized knowledge for a query.

        Compatible with the standard RAG interface.

        Args:
            query: The query/instruction
            top_k: Number of results to retrieve
            agent_id: Agent ID (for compatibility)

        Returns:
            Tuple of (scores, indices)
        """
        retrieved = self.retriever.retrieve(query)[:top_k]

        scores = []
        indices = []

        for item in retrieved:
            # Find the index in data_dict
            for idx, entry in self.data_dict.items():
                if entry["knowledge_id"] == item.knowledge_id:
                    indices.append(idx)
                    scores.append(item.similarity_score)
                    break

        return np.array(scores), np.array(indices)

    def format_for_prompt(
        self,
        indices: np.ndarray,
        max_examples: int = 3,
    ) -> str:
        """Format retrieved entries for prompt."""
        parts = ["## Personalized Knowledge from Memory\n"]

        for i, idx in enumerate(indices[:max_examples]):
            if idx in self.data_dict:
                entry = self.data_dict[idx]
                parts.append(f"### Memory {i+1}")

                if entry.get("alias"):
                    parts.append(f"**Reference:** {entry['alias']}")

                parts.append(f"**Type:** {entry['knowledge_type'].replace('_', ' ').title()}")

                if entry.get("description"):
                    parts.append(f"{entry['description']}")

                if entry.get("objects"):
                    parts.append(f"**Objects:** {', '.join(entry['objects'])}")

                if entry.get("locations"):
                    parts.append(f"**Locations:** {', '.join(entry['locations'])}")

                parts.append("")

        return "\n".join(parts)
