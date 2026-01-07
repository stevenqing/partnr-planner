#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decentralized Skill-Structured Hierarchical Memory for Multi-Agent Coordination

This package implements the hierarchical memory framework for multi-agent
coordination as described in /doc/our_method.

Key Components:
- HierarchicalSkillMemory: Skill library (L_ind ∪ L_coop)
- HierarchicalRetriever: Four-stage retrieval pipeline
- TheoryOfMindReasoner: ToM capabilities for partner intention inference
- RAGDatasetBuilder: Build RAG datasets from heuristic trajectories
- HierarchicalSkillPlanner: Integration with LLM planner

Usage:
    from our_method import (
        HierarchicalSkillMemory,
        HierarchicalRetriever,
        TheoryOfMindReasoner,
        HierarchicalSkillPlanner,
        create_planner,
        create_rag_integration
    )

Reference: /doc/our_method
"""

from .build_hierarchical_skill_memory import (
    HierarchicalSkillMemory,
    Skill,
    SkillInstance,
    CooperationSkill,
    AgentAction,
    EnvironmentState,
    build_memory_from_heuristic_results
)

from .hierarchical_retrieval import (
    HierarchicalRetriever,
    Query,
    RetrievedSkill,
    create_retriever
)

from .theory_of_mind import (
    TheoryOfMindReasoner,
    PartnerObservation,
    InferredIntention,
    ToMReasoning,
    create_tom_reasoner
)

from .planner_integration import (
    HierarchicalSkillPlanner,
    RAGIntegration,
    create_planner,
    create_rag_integration
)

from .build_rag_dataset import (
    RAGDatasetBuilder,
    build_rag_dataset
)

# LLM-based skill extraction
try:
    from .llm_skill_extractor import (
        LLMSkillExtractor,
        FallbackSkillExtractor,
        ExtractedSkill,
        ExtractedCooperationPattern,
        create_skill_extractor
    )
    HAS_LLM_EXTRACTOR = True
except ImportError:
    HAS_LLM_EXTRACTOR = False

__version__ = "1.0.0"

__all__ = [
    # Memory building
    "HierarchicalSkillMemory",
    "Skill",
    "SkillInstance",
    "CooperationSkill",
    "AgentAction",
    "EnvironmentState",
    "build_memory_from_heuristic_results",

    # Retrieval
    "HierarchicalRetriever",
    "Query",
    "RetrievedSkill",
    "create_retriever",

    # Theory of Mind
    "TheoryOfMindReasoner",
    "PartnerObservation",
    "InferredIntention",
    "ToMReasoning",
    "create_tom_reasoner",

    # Planner integration
    "HierarchicalSkillPlanner",
    "RAGIntegration",
    "create_planner",
    "create_rag_integration",

    # Dataset building
    "RAGDatasetBuilder",
    "build_rag_dataset",

    # LLM-based extraction (if available)
    "LLMSkillExtractor",
    "FallbackSkillExtractor",
    "ExtractedSkill",
    "ExtractedCooperationPattern",
    "create_skill_extractor",
    "HAS_LLM_EXTRACTOR",
]
