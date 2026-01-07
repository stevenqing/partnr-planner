#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build MEMENTO User Profile Memory from Trajectories

This script builds the hierarchical knowledge graph-based user profile memory
from heuristic agent trajectories.

Usage:
    python build_memory.py --data_dir /path/to/trajectories --output_dir /path/to/output

The script will:
1. Load successful trajectories from the data directory
2. Extract personalized knowledge (object semantics and user patterns)
3. Build the hierarchical knowledge graph
4. Save the memory for use in planning
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from methods.MEMENTO.user_profile_memory import (
    UserProfileMemory,
    KnowledgeType,
    ObjectSemanticSubtype,
    UserPatternSubtype,
)
from methods.MEMENTO.knowledge_graph_builder import (
    KnowledgeGraphBuilder,
    KnowledgeExtractor,
    ElementExtractor,
    ExtractedKnowledge,
    ExtractedElement,
)


def parse_action(action_str: str) -> Dict[str, Any]:
    """Parse an action string to extract components."""
    result = {
        "action_type": None,
        "object": None,
        "location": None,
        "reference": None,
        "raw": action_str,
    }

    # Parse Rearrange action
    match = re.match(r"Rearrange\[([^\]]+)\]", action_str)
    if match:
        args = [a.strip() for a in match.group(1).split(",")]
        result["action_type"] = "Rearrange"
        if len(args) >= 1:
            result["object"] = args[0]
        if len(args) >= 3:
            result["location"] = args[2]
        if len(args) >= 5:
            result["reference"] = args[4]
        return result

    # Parse Pick action
    match = re.match(r"Pick\[([^\]]+)\]", action_str)
    if match:
        result["action_type"] = "Pick"
        result["object"] = match.group(1).strip()
        return result

    # Parse Place action
    match = re.match(r"Place\[([^\]]+)\]", action_str)
    if match:
        args = [a.strip() for a in match.group(1).split(",")]
        result["action_type"] = "Place"
        if len(args) >= 1:
            result["object"] = args[0]
        if len(args) >= 3:
            result["location"] = args[2]
        return result

    # Parse Navigate action
    match = re.match(r"Navigate\[([^\]]+)\]", action_str)
    if match:
        result["action_type"] = "Navigate"
        result["location"] = match.group(1).strip()
        return result

    return result


def parse_trace_file(trace_path: str) -> Dict[str, Any]:
    """Parse a trace file to extract trajectory information."""
    with open(trace_path, 'r') as f:
        content = f.read()

    result = {
        "instruction": "",
        "steps": [],
        "objects_involved": set(),
        "locations_involved": set(),
        "action_sequence": [],
    }

    # Extract instruction
    task_match = re.search(r"Task:\s*(.+?)(?:\n|$)", content)
    if task_match:
        result["instruction"] = task_match.group(1).strip()

    # Extract thoughts and actions
    thought_pattern = r"Thought:\s*(.+?)(?=\n(?:Agent_|Thought:|$))"
    action_pattern = r"Agent_\d+_Action:\s*(.+?)(?:\n|$)"

    thoughts = re.findall(thought_pattern, content, re.DOTALL)
    actions = re.findall(action_pattern, content)

    for i, action in enumerate(actions):
        thought = thoughts[i] if i < len(thoughts) else ""
        parsed_action = parse_action(action.strip())

        step = {
            "thought": thought.strip(),
            "action": action.strip(),
            "parsed": parsed_action,
        }
        result["steps"].append(step)

        # Track objects and locations
        if parsed_action["object"]:
            result["objects_involved"].add(parsed_action["object"])
        if parsed_action["location"]:
            result["locations_involved"].add(parsed_action["location"])
        if parsed_action["action_type"]:
            result["action_sequence"].append(parsed_action["action_type"])

    # Convert sets to lists for JSON serialization
    result["objects_involved"] = list(result["objects_involved"])
    result["locations_involved"] = list(result["locations_involved"])

    return result


def extract_user_patterns(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract user patterns from trajectories."""
    patterns = []

    # Group trajectories by similar instructions
    instruction_groups = {}
    for traj in trajectories:
        # Normalize instruction for grouping
        instruction = traj["instruction"].lower()
        # Remove specific object names for grouping
        normalized = re.sub(r'\b\w+_\d+\b', 'OBJ', instruction)

        if normalized not in instruction_groups:
            instruction_groups[normalized] = []
        instruction_groups[normalized].append(traj)

    # Find recurring patterns
    for normalized, group in instruction_groups.items():
        if len(group) >= 2:
            # This is a recurring pattern
            # Analyze action sequences
            action_sequences = [tuple(t["action_sequence"]) for t in group]
            most_common_seq = max(set(action_sequences), key=action_sequences.count)

            # Get common objects and locations
            all_objects = set()
            all_locations = set()
            for t in group:
                all_objects.update(t["objects_involved"])
                all_locations.update(t["locations_involved"])

            patterns.append({
                "type": "routine",
                "description": group[0]["instruction"],
                "action_sequence": list(most_common_seq),
                "objects": list(all_objects),
                "locations": list(all_locations),
                "frequency": len(group),
            })

    return patterns


def extract_object_semantics(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract object semantics from trajectories."""
    semantics = []

    # Track objects that appear together
    object_cooccurrence = {}

    for traj in trajectories:
        objects = traj["objects_involved"]
        if len(objects) >= 2:
            # Objects appear together in the same task
            key = tuple(sorted(objects))
            if key not in object_cooccurrence:
                object_cooccurrence[key] = {
                    "objects": objects,
                    "instructions": [],
                    "locations": set(),
                }
            object_cooccurrence[key]["instructions"].append(traj["instruction"])
            object_cooccurrence[key]["locations"].update(traj["locations_involved"])

    # Create object groups
    for key, data in object_cooccurrence.items():
        if len(data["instructions"]) >= 2:
            # These objects are frequently used together
            semantics.append({
                "type": "groups",
                "alias": f"set of {', '.join(data['objects'][:3])}",
                "objects": data["objects"],
                "locations": list(data["locations"]),
                "description": f"Objects commonly rearranged together",
            })

    # Track individual object patterns
    object_patterns = {}
    for traj in trajectories:
        for obj in traj["objects_involved"]:
            if obj not in object_patterns:
                object_patterns[obj] = {
                    "locations": set(),
                    "frequency": 0,
                }
            object_patterns[obj]["frequency"] += 1
            object_patterns[obj]["locations"].update(traj["locations_involved"])

    # Create object-location associations
    for obj, data in object_patterns.items():
        if data["frequency"] >= 2 and data["locations"]:
            semantics.append({
                "type": "preference",
                "alias": obj,
                "objects": [obj],
                "locations": list(data["locations"]),
                "description": f"Frequently placed at: {', '.join(data['locations'])}",
            })

    return semantics


def build_memory_from_data(
    data_dir: str,
    output_dir: str,
    user_id: str = "user_0",
    use_llm: bool = False,
    llm_model: Optional[Any] = None,
) -> UserProfileMemory:
    """
    Build MEMENTO user profile memory from trajectory data.

    Args:
        data_dir: Directory containing trajectory data
        output_dir: Directory to save the memory
        user_id: User ID for the memory
        use_llm: Whether to use LLM for extraction
        llm_model: LLM model instance

    Returns:
        Built UserProfileMemory
    """
    print("=" * 60)
    print("MEMENTO: Building User Profile Memory")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"User ID: {user_id}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize memory and builder
    memory = UserProfileMemory()
    builder = KnowledgeGraphBuilder(
        memory=memory,
        use_llm=use_llm,
        llm_model=llm_model,
    )

    # Add user
    memory.add_user(user_id, name=user_id)

    # Load successful trajectories
    trajectories = load_trajectories(data_dir)
    print(f"\nLoaded {len(trajectories)} successful trajectories")

    # Process each trajectory
    for i, traj in enumerate(trajectories):
        if i % 100 == 0:
            print(f"Processing trajectory {i}/{len(trajectories)}...")

        # Update memory using builder
        builder.update(
            instruction=traj["instruction"],
            user_id=user_id,
            trajectory=traj["steps"],
        )

    # Extract additional patterns from aggregated data
    print("\nExtracting user patterns...")
    user_patterns = extract_user_patterns(trajectories)
    print(f"Found {len(user_patterns)} recurring patterns")

    for pattern in user_patterns:
        # Add pattern knowledge
        knowledge_id = memory.generate_id("k")
        memory.add_knowledge(
            knowledge_id=knowledge_id,
            name=f"pattern_{pattern['type']}",
            knowledge_type=KnowledgeType.USER_PATTERN,
            user_id=user_id,
            subtype=pattern["type"],
            description=pattern["description"],
        )

        # Add associated elements
        for obj in pattern.get("objects", [])[:5]:
            obj_id = memory.generate_id("o")
            memory.add_object(obj_id, obj, knowledge_id)

        for i, action in enumerate(pattern.get("action_sequence", [])[:5]):
            pattern_id = memory.generate_id("p")
            memory.add_pattern(
                pattern_id, action, knowledge_id, action_name=action
            )

    # Extract object semantics
    print("\nExtracting object semantics...")
    object_semantics = extract_object_semantics(trajectories)
    print(f"Found {len(object_semantics)} object semantic groups")

    for sem in object_semantics:
        # Add object semantics knowledge
        knowledge_id = memory.generate_id("k")
        memory.add_knowledge(
            knowledge_id=knowledge_id,
            name=f"object_sem_{sem['type']}",
            knowledge_type=KnowledgeType.OBJECT_SEMANTICS,
            user_id=user_id,
            subtype=sem["type"],
            alias=sem.get("alias"),
            description=sem.get("description"),
        )

        # Add associated objects
        for obj in sem.get("objects", [])[:5]:
            obj_id = memory.generate_id("o")
            memory.add_object(obj_id, obj, knowledge_id)

    # Save memory
    output_path = os.path.join(output_dir, "user_profile_memory.json.gz")
    memory.save(output_path)

    # Save summary
    summary = {
        "num_users": len(memory.users),
        "num_object_semantics": len(memory._knowledge_by_type[KnowledgeType.OBJECT_SEMANTICS]),
        "num_user_patterns": len(memory._knowledge_by_type[KnowledgeType.USER_PATTERN]),
        "num_total_nodes": len(memory.nodes),
        "num_edges": len(memory.edges),
        "trajectories_processed": len(trajectories),
    }

    summary_path = os.path.join(output_dir, "memory_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Memory building complete!")
    print(f"  Users: {summary['num_users']}")
    print(f"  Object Semantics: {summary['num_object_semantics']}")
    print(f"  User Patterns: {summary['num_user_patterns']}")
    print(f"  Total Nodes: {summary['num_total_nodes']}")
    print(f"  Total Edges: {summary['num_edges']}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    return memory


def load_trajectories(data_dir: str) -> List[Dict[str, Any]]:
    """Load successful trajectories from data directory."""
    trajectories = []

    # Check for episode result log
    csv_path = os.path.join(data_dir, "episode_result_log.csv")
    successful_episodes = set()

    if os.path.exists(csv_path):
        print(f"Loading successful episodes from {csv_path}")
        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    try:
                        epid = int(row[0])
                        success = float(row[-1])
                        if success == 1.0:
                            successful_episodes.add(epid)
                    except (ValueError, IndexError):
                        continue
        print(f"Found {len(successful_episodes)} successful episodes")

    # Find trace files
    trace_patterns = [
        os.path.join(data_dir, "**/traces/*/trace-*.txt"),
        os.path.join(data_dir, "**/trace-*.txt"),
        os.path.join(data_dir, "traces/*/trace-*.txt"),
    ]

    trace_files = []
    for pattern in trace_patterns:
        trace_files.extend(glob.glob(pattern, recursive=True))

    print(f"Found {len(trace_files)} trace files")

    # Process trace files
    for trace_path in trace_files:
        # Extract episode ID from filename
        match = re.search(r"episode_(\d+)_", os.path.basename(trace_path))
        if match:
            epid = int(match.group(1))
            # Skip if we have a success filter and this episode isn't successful
            if successful_episodes and epid not in successful_episodes:
                continue

        try:
            traj = parse_trace_file(trace_path)
            if traj["instruction"] and traj["steps"]:
                trajectories.append(traj)
        except Exception as e:
            print(f"Warning: Failed to parse {trace_path}: {e}")
            continue

    return trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Build MEMENTO User Profile Memory from trajectories"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing trajectory data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the memory",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        default="user_0",
        help="User ID for the memory",
    )
    parser.add_argument(
        "--use_llm",
        action="store_true",
        help="Use LLM for extraction (requires model)",
    )

    args = parser.parse_args()

    build_memory_from_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        user_id=args.user_id,
        use_llm=args.use_llm,
    )


if __name__ == "__main__":
    main()
