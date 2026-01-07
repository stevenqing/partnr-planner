#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Profile Memory: Hierarchical Knowledge Graph Architecture

This module implements the three-level hierarchical knowledge graph structure
for managing personalized user knowledge as described in MEMENTO.

Structure:
    ┌─────────────────────────────────────────────────────────────┐
    │                         USER LEVEL                          │
    │                    (user_0, user_1, ...)                   │
    └─────────────────────────┬───────────────────────────────────┘
                              │ refers_to
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      KNOWLEDGE LEVEL                        │
    │     Object Semantics          │        User Patterns        │
    │   ("my coffee set")           │   ("my morning routine")    │
    └──────────┬────────────────────┴──────────────┬──────────────┘
               │ composed_of                        │ entails
               ▼                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      ELEMENTS LEVEL                         │
    │   Objects    │    Patterns    │    Locations                │
    │ (red mug)    │   (place on)   │ (kitchen counter)          │
    └─────────────────────────────────────────────────────────────┘
"""

import json
import gzip
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    USER = "user"
    KNOWLEDGE = "knowledge"
    OBJECT = "object"
    PATTERN = "pattern"
    LOCATION = "location"


class KnowledgeType(str, Enum):
    """Subtypes of knowledge nodes."""
    OBJECT_SEMANTICS = "object_semantics"
    USER_PATTERN = "user_pattern"


class ObjectSemanticSubtype(str, Enum):
    """Subtypes of object semantics knowledge."""
    OWNERSHIP = "ownership"      # "my cup", "my laptop"
    PREFERENCE = "preference"    # "bread from my favorite bakery"
    HISTORY = "history"          # "graduation gift from my grandma"
    GROUPS = "groups"            # "my home office setup"


class UserPatternSubtype(str, Enum):
    """Subtypes of user pattern knowledge."""
    ROUTINE = "routine"          # "meal time setting"
    PREFERENCE = "preference"    # "my coffee break setup"


class EdgeType(str, Enum):
    """Types of edges in the knowledge graph."""
    # Hierarchical edges (structural relationships)
    REFERS_TO = "refers_to"           # User -> Knowledge
    ENTAILS = "entails"               # Knowledge -> Pattern
    COMPOSED_OF = "composed_of"       # Knowledge -> Object
    TARGET = "target"                 # Pattern -> Object/Location
    ALIAS_OF = "alias_of"             # Knowledge -> Object
    TARGET_OBJECT = "target_object"   # Knowledge -> Object
    TARGET_LOCATION = "target_location"  # Knowledge -> Location

    # Temporal edges (sequential ordering)
    BEFORE = "before"                 # Pattern_n -> Pattern_{n+1}


@dataclass
class BaseNode:
    """Base class for all nodes in the knowledge graph."""
    id: str
    node_type: NodeType
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type.value,
            "name": self.name,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseNode":
        return cls(
            id=data["id"],
            node_type=NodeType(data["type"]),
            name=data["name"],
            attributes=data.get("attributes", {}),
        )


@dataclass
class UserNode(BaseNode):
    """User level node."""

    def __init__(self, id: str, name: str, attributes: Dict[str, Any] = None):
        super().__init__(
            id=id,
            node_type=NodeType.USER,
            name=name,
            attributes=attributes or {},
        )


@dataclass
class KnowledgeNode(BaseNode):
    """Knowledge level node (Object Semantics or User Pattern)."""
    knowledge_type: KnowledgeType = KnowledgeType.OBJECT_SEMANTICS
    subtype: Optional[str] = None
    alias: Optional[str] = None
    description: Optional[str] = None

    def __init__(
        self,
        id: str,
        name: str,
        knowledge_type: KnowledgeType,
        subtype: Optional[str] = None,
        alias: Optional[str] = None,
        description: Optional[str] = None,
        attributes: Dict[str, Any] = None,
    ):
        super().__init__(
            id=id,
            node_type=NodeType.KNOWLEDGE,
            name=name,
            attributes=attributes or {},
        )
        self.knowledge_type = knowledge_type
        self.subtype = subtype
        self.alias = alias
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "knowledge_type": self.knowledge_type.value,
            "subtype": self.subtype,
            "alias": self.alias,
            "description": self.description,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeNode":
        return cls(
            id=data["id"],
            name=data["name"],
            knowledge_type=KnowledgeType(data.get("knowledge_type", "object_semantics")),
            subtype=data.get("subtype"),
            alias=data.get("alias"),
            description=data.get("description"),
            attributes=data.get("attributes", {}),
        )


@dataclass
class ObjectNode(BaseNode):
    """Element level object node."""
    granularity: str = "instance"  # "instance" or "category"
    object_attributes: List[str] = field(default_factory=list)

    def __init__(
        self,
        id: str,
        name: str,
        granularity: str = "instance",
        object_attributes: List[str] = None,
        attributes: Dict[str, Any] = None,
    ):
        super().__init__(
            id=id,
            node_type=NodeType.OBJECT,
            name=name,
            attributes=attributes or {},
        )
        self.granularity = granularity
        self.object_attributes = object_attributes or []

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "granularity": self.granularity,
            "object_attributes": self.object_attributes,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectNode":
        return cls(
            id=data["id"],
            name=data["name"],
            granularity=data.get("granularity", "instance"),
            object_attributes=data.get("object_attributes", []),
            attributes=data.get("attributes", {}),
        )


@dataclass
class PatternNode(BaseNode):
    """Element level pattern node."""
    action_name: str = ""
    args: List[str] = field(default_factory=list)

    def __init__(
        self,
        id: str,
        name: str,
        action_name: str = "",
        args: List[str] = None,
        attributes: Dict[str, Any] = None,
    ):
        super().__init__(
            id=id,
            node_type=NodeType.PATTERN,
            name=name,
            attributes=attributes or {},
        )
        self.action_name = action_name
        self.args = args or []

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "action_name": self.action_name,
            "args": self.args,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternNode":
        return cls(
            id=data["id"],
            name=data["name"],
            action_name=data.get("action_name", ""),
            args=data.get("args", []),
            attributes=data.get("attributes", {}),
        )


@dataclass
class LocationNode(BaseNode):
    """Element level location node."""
    expression: str = ""  # Natural language expression

    def __init__(
        self,
        id: str,
        name: str,
        expression: str = "",
        attributes: Dict[str, Any] = None,
    ):
        super().__init__(
            id=id,
            node_type=NodeType.LOCATION,
            name=name,
            attributes=attributes or {},
        )
        self.expression = expression

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "expression": self.expression,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocationNode":
        return cls(
            id=data["id"],
            name=data["name"],
            expression=data.get("expression", ""),
            attributes=data.get("attributes", {}),
        )


@dataclass
class Edge:
    """Edge connecting two nodes in the knowledge graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        return cls(
            source_id=data["source"],
            target_id=data["target"],
            edge_type=EdgeType(data["type"]),
            attributes=data.get("attributes", {}),
        )


class UserProfileMemory:
    """
    Hierarchical Knowledge Graph-based User Profile Memory.

    This class manages the three-level hierarchical structure for storing
    and retrieving personalized user knowledge.

    Attributes:
        nodes: Dictionary mapping node IDs to node objects
        edges: List of edges connecting nodes
        users: Set of user IDs
        knowledge_index: Index for quick knowledge lookup by type
    """

    def __init__(self):
        self.nodes: Dict[str, BaseNode] = {}
        self.edges: List[Edge] = []
        self.users: Set[str] = set()

        # Indices for efficient lookup
        self._knowledge_by_type: Dict[KnowledgeType, Set[str]] = {
            KnowledgeType.OBJECT_SEMANTICS: set(),
            KnowledgeType.USER_PATTERN: set(),
        }
        self._edges_by_source: Dict[str, List[Edge]] = {}
        self._edges_by_target: Dict[str, List[Edge]] = {}

    def add_user(self, user_id: str, name: str, attributes: Dict[str, Any] = None) -> UserNode:
        """Add a user node to the graph."""
        node = UserNode(id=user_id, name=name, attributes=attributes)
        self.nodes[user_id] = node
        self.users.add(user_id)
        return node

    def add_knowledge(
        self,
        knowledge_id: str,
        name: str,
        knowledge_type: KnowledgeType,
        user_id: str,
        subtype: Optional[str] = None,
        alias: Optional[str] = None,
        description: Optional[str] = None,
        attributes: Dict[str, Any] = None,
    ) -> KnowledgeNode:
        """Add a knowledge node and connect it to a user."""
        node = KnowledgeNode(
            id=knowledge_id,
            name=name,
            knowledge_type=knowledge_type,
            subtype=subtype,
            alias=alias,
            description=description,
            attributes=attributes,
        )
        self.nodes[knowledge_id] = node
        self._knowledge_by_type[knowledge_type].add(knowledge_id)

        # Add edge from user to knowledge
        self.add_edge(user_id, knowledge_id, EdgeType.REFERS_TO)

        return node

    def add_object(
        self,
        object_id: str,
        name: str,
        knowledge_id: str,
        granularity: str = "instance",
        object_attributes: List[str] = None,
        edge_type: EdgeType = EdgeType.COMPOSED_OF,
        attributes: Dict[str, Any] = None,
    ) -> ObjectNode:
        """Add an object node and connect it to a knowledge node."""
        node = ObjectNode(
            id=object_id,
            name=name,
            granularity=granularity,
            object_attributes=object_attributes,
            attributes=attributes,
        )
        self.nodes[object_id] = node

        # Add edge from knowledge to object
        self.add_edge(knowledge_id, object_id, edge_type)

        return node

    def add_pattern(
        self,
        pattern_id: str,
        name: str,
        knowledge_id: str,
        action_name: str = "",
        args: List[str] = None,
        attributes: Dict[str, Any] = None,
    ) -> PatternNode:
        """Add a pattern node and connect it to a knowledge node."""
        node = PatternNode(
            id=pattern_id,
            name=name,
            action_name=action_name,
            args=args,
            attributes=attributes,
        )
        self.nodes[pattern_id] = node

        # Add edge from knowledge to pattern
        self.add_edge(knowledge_id, pattern_id, EdgeType.ENTAILS)

        return node

    def add_location(
        self,
        location_id: str,
        name: str,
        expression: str = "",
        attributes: Dict[str, Any] = None,
    ) -> LocationNode:
        """Add a location node."""
        node = LocationNode(
            id=location_id,
            name=name,
            expression=expression,
            attributes=attributes,
        )
        self.nodes[location_id] = node
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        attributes: Dict[str, Any] = None,
    ) -> Edge:
        """Add an edge between two nodes."""
        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            attributes=attributes or {},
        )
        self.edges.append(edge)

        # Update indices
        if source_id not in self._edges_by_source:
            self._edges_by_source[source_id] = []
        self._edges_by_source[source_id].append(edge)

        if target_id not in self._edges_by_target:
            self._edges_by_target[target_id] = []
        self._edges_by_target[target_id].append(edge)

        return edge

    def add_temporal_edge(
        self,
        pattern1_id: str,
        pattern2_id: str,
        attributes: Dict[str, Any] = None,
    ) -> Edge:
        """Add a temporal ordering edge between two patterns."""
        return self.add_edge(pattern1_id, pattern2_id, EdgeType.BEFORE, attributes)

    def get_subgraph_by_type(self, knowledge_type: KnowledgeType) -> "UserProfileMemory":
        """
        Get a subgraph containing only knowledge of the specified type.

        Args:
            knowledge_type: Type of knowledge to filter

        Returns:
            New UserProfileMemory containing the filtered subgraph
        """
        subgraph = UserProfileMemory()

        # Get all knowledge IDs of this type
        knowledge_ids = self._knowledge_by_type.get(knowledge_type, set())

        # Add related nodes and edges
        visited_nodes = set()
        for knowledge_id in knowledge_ids:
            self._collect_subgraph_nodes(knowledge_id, visited_nodes, subgraph)

        return subgraph

    def _collect_subgraph_nodes(
        self,
        node_id: str,
        visited: Set[str],
        subgraph: "UserProfileMemory",
    ):
        """Recursively collect nodes for a subgraph."""
        if node_id in visited:
            return
        visited.add(node_id)

        # Add the node
        if node_id in self.nodes:
            node = self.nodes[node_id]
            subgraph.nodes[node_id] = node

            if isinstance(node, UserNode):
                subgraph.users.add(node_id)
            elif isinstance(node, KnowledgeNode):
                subgraph._knowledge_by_type[node.knowledge_type].add(node_id)

        # Add outgoing edges and recurse
        for edge in self._edges_by_source.get(node_id, []):
            subgraph.edges.append(edge)
            self._collect_subgraph_nodes(edge.target_id, visited, subgraph)

    def expand(self, node_ids: List[str]) -> Tuple[List[BaseNode], List[Edge]]:
        """
        Expand a set of nodes to include all connected edges and descendant nodes.

        Args:
            node_ids: List of node IDs to expand

        Returns:
            Tuple of (nodes, edges) reachable from the input nodes
        """
        visited_nodes = set()
        collected_nodes = []
        collected_edges = []

        for node_id in node_ids:
            self._expand_node(node_id, visited_nodes, collected_nodes, collected_edges)

        return collected_nodes, collected_edges

    def _expand_node(
        self,
        node_id: str,
        visited: Set[str],
        nodes: List[BaseNode],
        edges: List[Edge],
    ):
        """Recursively expand a node."""
        if node_id in visited:
            return
        visited.add(node_id)

        if node_id in self.nodes:
            nodes.append(self.nodes[node_id])

        for edge in self._edges_by_source.get(node_id, []):
            edges.append(edge)
            self._expand_node(edge.target_id, visited, nodes, edges)

    def get_knowledge_for_user(self, user_id: str) -> List[KnowledgeNode]:
        """Get all knowledge nodes associated with a user."""
        knowledge_nodes = []
        for edge in self._edges_by_source.get(user_id, []):
            if edge.edge_type == EdgeType.REFERS_TO:
                node = self.nodes.get(edge.target_id)
                if isinstance(node, KnowledgeNode):
                    knowledge_nodes.append(node)
        return knowledge_nodes

    def get_elements_for_knowledge(
        self,
        knowledge_id: str,
    ) -> Dict[str, List[BaseNode]]:
        """Get all element nodes (objects, patterns, locations) for a knowledge node."""
        elements = {
            "objects": [],
            "patterns": [],
            "locations": [],
        }

        for edge in self._edges_by_source.get(knowledge_id, []):
            node = self.nodes.get(edge.target_id)
            if isinstance(node, ObjectNode):
                elements["objects"].append(node)
            elif isinstance(node, PatternNode):
                elements["patterns"].append(node)
            elif isinstance(node, LocationNode):
                elements["locations"].append(node)

        return elements

    def reformulate_knowledge(self, knowledge_id: str) -> str:
        """
        Reformulate a knowledge subgraph into natural language description.

        Args:
            knowledge_id: ID of the knowledge node to reformulate

        Returns:
            Natural language description of the knowledge
        """
        knowledge = self.nodes.get(knowledge_id)
        if not isinstance(knowledge, KnowledgeNode):
            return ""

        parts = []

        # Add knowledge description
        if knowledge.alias:
            parts.append(f"'{knowledge.alias}'")
        if knowledge.description:
            parts.append(f": {knowledge.description}")

        # Get elements
        elements = self.get_elements_for_knowledge(knowledge_id)

        # Add objects
        if elements["objects"]:
            obj_names = [obj.name for obj in elements["objects"]]
            if knowledge.knowledge_type == KnowledgeType.OBJECT_SEMANTICS:
                parts.append(f" refers to: {', '.join(obj_names)}")
            else:
                parts.append(f" involves: {', '.join(obj_names)}")

        # Add patterns (for user patterns)
        if elements["patterns"]:
            pattern_descs = []
            for pattern in elements["patterns"]:
                if pattern.action_name and pattern.args:
                    pattern_descs.append(f"{pattern.action_name}({', '.join(pattern.args)})")
            if pattern_descs:
                parts.append(f" actions: {' -> '.join(pattern_descs)}")

        # Add locations
        if elements["locations"]:
            loc_exprs = [loc.expression or loc.name for loc in elements["locations"]]
            parts.append(f" at: {', '.join(loc_exprs)}")

        return "".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the graph to a dictionary for serialization."""
        return {
            "nodes": [self._node_to_dict(node) for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": {
                "num_users": len(self.users),
                "num_object_semantics": len(self._knowledge_by_type[KnowledgeType.OBJECT_SEMANTICS]),
                "num_user_patterns": len(self._knowledge_by_type[KnowledgeType.USER_PATTERN]),
            },
        }

    def _node_to_dict(self, node: BaseNode) -> Dict[str, Any]:
        """Convert a node to dictionary based on its type."""
        if isinstance(node, KnowledgeNode):
            return node.to_dict()
        elif isinstance(node, ObjectNode):
            return node.to_dict()
        elif isinstance(node, PatternNode):
            return node.to_dict()
        elif isinstance(node, LocationNode):
            return node.to_dict()
        else:
            return node.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfileMemory":
        """Create a UserProfileMemory from a dictionary."""
        memory = cls()

        # Load nodes
        for node_data in data.get("nodes", []):
            node_type = NodeType(node_data["type"])

            if node_type == NodeType.USER:
                node = UserNode(
                    id=node_data["id"],
                    name=node_data["name"],
                    attributes=node_data.get("attributes", {}),
                )
                memory.nodes[node.id] = node
                memory.users.add(node.id)

            elif node_type == NodeType.KNOWLEDGE:
                node = KnowledgeNode.from_dict(node_data)
                memory.nodes[node.id] = node
                memory._knowledge_by_type[node.knowledge_type].add(node.id)

            elif node_type == NodeType.OBJECT:
                node = ObjectNode.from_dict(node_data)
                memory.nodes[node.id] = node

            elif node_type == NodeType.PATTERN:
                node = PatternNode.from_dict(node_data)
                memory.nodes[node.id] = node

            elif node_type == NodeType.LOCATION:
                node = LocationNode.from_dict(node_data)
                memory.nodes[node.id] = node

        # Load edges
        for edge_data in data.get("edges", []):
            edge = Edge.from_dict(edge_data)
            memory.edges.append(edge)

            # Update indices
            if edge.source_id not in memory._edges_by_source:
                memory._edges_by_source[edge.source_id] = []
            memory._edges_by_source[edge.source_id].append(edge)

            if edge.target_id not in memory._edges_by_target:
                memory._edges_by_target[edge.target_id] = []
            memory._edges_by_target[edge.target_id].append(edge)

        return memory

    def save(self, path: Union[str, Path], compress: bool = True):
        """Save the memory to a file."""
        path = Path(path)
        data = self.to_dict()

        if compress:
            if not str(path).endswith('.gz'):
                path = Path(str(path) + '.gz')
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        print(f"Saved UserProfileMemory to {path}")
        print(f"  Users: {len(self.users)}")
        print(f"  Object Semantics: {len(self._knowledge_by_type[KnowledgeType.OBJECT_SEMANTICS])}")
        print(f"  User Patterns: {len(self._knowledge_by_type[KnowledgeType.USER_PATTERN])}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "UserProfileMemory":
        """Load a memory from a file."""
        path = Path(path)

        if str(path).endswith('.gz'):
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        memory = cls.from_dict(data)
        print(f"Loaded UserProfileMemory from {path}")
        print(f"  Users: {len(memory.users)}")
        print(f"  Object Semantics: {len(memory._knowledge_by_type[KnowledgeType.OBJECT_SEMANTICS])}")
        print(f"  User Patterns: {len(memory._knowledge_by_type[KnowledgeType.USER_PATTERN])}")

        return memory

    def generate_id(self, prefix: str = "") -> str:
        """Generate a unique ID for a new node."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def __repr__(self) -> str:
        return (
            f"UserProfileMemory("
            f"users={len(self.users)}, "
            f"knowledge={len(self._knowledge_by_type[KnowledgeType.OBJECT_SEMANTICS]) + len(self._knowledge_by_type[KnowledgeType.USER_PATTERN])}, "
            f"nodes={len(self.nodes)}, "
            f"edges={len(self.edges)})"
        )
