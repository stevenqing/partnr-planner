#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEMENTO: Memory-Enhanced Multi-Agent Personalized Assistance

Implementation based on:
"Embodied Agents Meet Personalization: Investigating Challenges and Solutions
Through the Lens of Memory Utilization" (Kwon et al., 2025)

Key Components:
- UserProfileMemory: Hierarchical knowledge graph for personalized knowledge
- KnowledgeGraphBuilder: Build and update user profile memory from interactions
- MementoRetriever: Retrieve relevant personalized knowledge
- MementoPlanner: Integration with LLM planner

Memory Architecture:
    USER LEVEL          -> User nodes (user_0, user_1, ...)
    KNOWLEDGE LEVEL     -> Object Semantics & User Patterns
    ELEMENTS LEVEL      -> Objects, Patterns, Locations

Reference: /doc/memory_method
"""

from .user_profile_memory import (
    UserProfileMemory,
    UserNode,
    KnowledgeNode,
    ObjectNode,
    PatternNode,
    LocationNode,
    Edge,
    EdgeType,
    KnowledgeType,
)

from .knowledge_graph_builder import (
    KnowledgeGraphBuilder,
    KnowledgeExtractor,
    ElementExtractor,
    build_memory_from_trajectories,
)

from .memento_retriever import (
    MementoRetriever,
    RetrievedKnowledge,
    create_retriever,
)

from .memento_planner import (
    MementoPlanner,
    MementoRAGIntegration,
    create_planner,
)

__version__ = "1.0.0"

__all__ = [
    # User Profile Memory
    "UserProfileMemory",
    "UserNode",
    "KnowledgeNode",
    "ObjectNode",
    "PatternNode",
    "LocationNode",
    "Edge",
    "EdgeType",
    "KnowledgeType",

    # Knowledge Graph Builder
    "KnowledgeGraphBuilder",
    "KnowledgeExtractor",
    "ElementExtractor",
    "build_memory_from_trajectories",

    # Retrieval
    "MementoRetriever",
    "RetrievedKnowledge",
    "create_retriever",

    # Planner Integration
    "MementoPlanner",
    "MementoRAGIntegration",
    "create_planner",
]
