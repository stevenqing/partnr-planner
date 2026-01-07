#!/usr/bin/env python3

"""
Create MEMENTO-style memory system from trajectory files.
This script processes the detailed trajectory pickle files and creates
a structured memory system similar to MEMENTO for RAG-based planning.
"""

import glob
import json
import os
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add the project to Python path
sys.path.insert(0, "/home/shuqing/partnr-planner")


def extract_episode_id_from_trace_file(filename: str) -> Optional[str]:
    """Extract episode ID from trace filename."""
    match = re.search(r"detailed_trace-episode_(\d+)_\d+\.pkl", filename)
    return match.group(1) if match else None


def load_episode_results(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load episode results from CSV files."""
    episode_results = {}

    # Load episode result log
    episode_log_path = os.path.join(results_dir, "episode_result_log.csv")
    if os.path.exists(episode_log_path):
        try:
            with open(episode_log_path, "r") as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        episode_id = parts[0]
                        success = (
                            float(parts[-1])
                            if parts[-1].replace(".", "").isdigit()
                            else 0.0
                        )
                        episode_results[episode_id] = {
                            "success": success >= 1.0,
                            "score": success,
                        }
        except Exception as e:
            print(f"Warning: Could not parse episode results: {e}")

    return episode_results


def process_trajectory_pickle(pickle_path: str) -> Dict[str, Any]:
    """Process a single trajectory pickle file and extract relevant information."""
    try:
        with open(pickle_path, "rb") as f:
            trace_data = pickle.load(f)

        # Extract key information
        instruction = trace_data.get("instruction", "")
        action_history = trace_data.get("action_history", [])

        # Create a formatted trace from action history
        trace_lines = [f"Task: {instruction}\n"]

        for _i, action_element in enumerate(action_history):
            if hasattr(action_element, "action") and hasattr(
                action_element, "response"
            ):
                action = action_element.action
                response = action_element.response
                agent_uid = getattr(action_element, "agent_uid", 0)

                # Format action
                if isinstance(action, tuple) and len(action) >= 2:
                    action_name, *action_args = action
                    action_str = f"{action_name}[{', '.join(str(arg) for arg in action_args if arg)}]"
                else:
                    action_str = str(action)

                trace_lines.append(f"Agent_{agent_uid}_Action: {action_str}")
                trace_lines.append(f"Agent_{agent_uid}_observation: {response}")

        # Extract prompts if available
        prompts = {}
        if (
            action_history
            and hasattr(action_history[0], "info")
            and "planner_info" in action_history[0].info
        ):
            planner_info = action_history[0].info["planner_info"]
            if "prompts" in planner_info:
                prompts = planner_info["prompts"]

        return {
            "instruction": instruction,
            "trace": "\n".join(trace_lines),
            "prompts": prompts,
            "action_count": len(action_history),
            "success": False,  # Will be updated from results
        }

    except Exception as e:
        print(f"Error processing {pickle_path}: {e}")
        return None


def create_memento_memory(
    trajectory_dir: str,
    output_memory_dir: str = "/home/shuqing/partnr-planner/memory_from_trajectories",
    filter_successful_only: bool = False,
):
    """
    Create MEMENTO-style memory structure from trajectory files.

    Args:
        trajectory_dir: Path to directory containing trajectory pickle files
        output_memory_dir: Output directory for memory structure
        filter_successful_only: Whether to only include successful episodes
    """

    print(f"Creating MEMENTO memory from trajectories in: {trajectory_dir}")

    # Find all detailed trace files
    trace_pattern = os.path.join(trajectory_dir, "**/detailed_trace-episode_*.pkl")
    trace_files = glob.glob(trace_pattern, recursive=True)

    print(f"Found {len(trace_files)} trajectory files")

    if not trace_files:
        print("No trajectory files found!")
        return None

    # Load episode results - find the actual results directory
    sample_trace = trace_files[0]
    # Navigate from detailed_traces back to results directory
    parts = sample_trace.split("/")
    results_idx = None
    for i, part in enumerate(parts):
        if part == "results":
            results_idx = i
            break

    if results_idx:
        results_dir = "/".join(parts[: results_idx + 1])
    else:
        # Fallback to going up levels from trace file
        results_dir = os.path.dirname(os.path.dirname(os.path.dirname(trace_files[0])))

    episode_results = load_episode_results(results_dir)
    print(f"Loaded results for {len(episode_results)} episodes")

    # Process each trajectory file
    memory_data = {}
    successful_count = 0

    for trace_file in trace_files:
        episode_id = extract_episode_id_from_trace_file(os.path.basename(trace_file))
        if not episode_id:
            continue

        # Check if we should filter by success
        episode_result = episode_results.get(episode_id, {})
        is_successful = episode_result.get("success", False)

        if filter_successful_only and not is_successful:
            continue

        # Process the trajectory
        trajectory_data = process_trajectory_pickle(trace_file)
        if trajectory_data is None:
            continue

        # Update success status
        trajectory_data["success"] = is_successful
        trajectory_data["score"] = episode_result.get("score", 0.0)

        # Use episode_id as scene_id for simplicity (can be enhanced later)
        scene_id = f"episode_{episode_id}"

        memory_data[scene_id] = trajectory_data

        if is_successful:
            successful_count += 1

    print(f"Processed {len(memory_data)} episodes ({successful_count} successful)")

    if filter_successful_only:
        print(f"Filtered to {len(memory_data)} successful episodes only")

    # Create output directory structure
    os.makedirs(output_memory_dir, exist_ok=True)

    # Save memory data in MEMENTO-style structure
    experiment_name = "habitat_trajectories"
    run_name = "trajectory_memory"

    base_memory_path = os.path.join(output_memory_dir, experiment_name, run_name)

    for scene_id, data in memory_data.items():
        # Create scene directory
        scene_dir = os.path.join(base_memory_path, scene_id)

        # Create prompts and traces directories
        prompts_dir = os.path.join(scene_dir, "prompts", "0")
        traces_dir = os.path.join(scene_dir, "traces", "0")

        os.makedirs(prompts_dir, exist_ok=True)
        os.makedirs(traces_dir, exist_ok=True)

        # Extract episode_id from scene_id
        episode_id = scene_id.replace("episode_", "")

        # Save trace file
        trace_filename = f"trace-episode_{episode_id}_0-0.txt"
        trace_path = os.path.join(traces_dir, trace_filename)
        with open(trace_path, "w") as f:
            f.write(data["trace"])

        # Save prompt files (if available)
        if data["prompts"]:
            for agent_id, prompt_content in data["prompts"].items():
                prompt_filename = f"prompt-episode_{episode_id}_0-{agent_id}.txt"
                prompt_path = os.path.join(prompts_dir, prompt_filename)
                with open(prompt_path, "w") as f:
                    f.write(prompt_content)

    # Create base memory directory structure
    os.makedirs(base_memory_path, exist_ok=True)

    # Create summary file
    summary_data = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "source_trajectory_dir": trajectory_dir,
        "total_episodes": len(memory_data),
        "successful_episodes": successful_count,
        "filter_successful_only": filter_successful_only,
        "memory_structure": "MEMENTO-style",
        "created_at": Path().cwd().as_posix(),
    }

    summary_path = os.path.join(base_memory_path, "memory_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Copy episode results if available
    episode_log_path = os.path.join(results_dir, "episode_result_log.csv")
    if os.path.exists(episode_log_path):
        shutil.copy(
            episode_log_path, os.path.join(base_memory_path, "episode_result_log.csv")
        )

    print("\nMEMENTO memory created successfully!")
    print(f"Location: {base_memory_path}")
    print(f"Episodes: {len(memory_data)}")
    print(f"Successful: {successful_count}")

    # Show directory structure sample
    print("\nSample directory structure:")
    sample_scenes = list(memory_data.keys())[:3]
    for scene in sample_scenes:
        scene_path = os.path.join(base_memory_path, scene)
        print(f"  {scene}/")
        if os.path.exists(os.path.join(scene_path, "prompts", "0")):
            prompt_files = os.listdir(os.path.join(scene_path, "prompts", "0"))
            for pf in prompt_files[:2]:
                print(f"    prompts/0/{pf}")
        if os.path.exists(os.path.join(scene_path, "traces", "0")):
            trace_files = os.listdir(os.path.join(scene_path, "traces", "0"))
            for tf in trace_files[:2]:
                print(f"    traces/0/{tf}")

    return base_memory_path


def main():
    """Main function to create MEMENTO memory from trajectories."""

    # Default trajectory directory
    trajectory_dir = "/home/shuqing/partnr-planner/outputs/habitat_llm/2025-09-07_14-12-16-rerange_only.json/results/rerange_only.json.gz"

    if len(sys.argv) > 1:
        trajectory_dir = sys.argv[1]

    # Check if directory exists
    if not os.path.exists(trajectory_dir):
        print(f"Error: Trajectory directory does not exist: {trajectory_dir}")
        return

    print("🚀 Creating MEMENTO Memory from Trajectories")
    print("=" * 60)
    print(f"Source: {trajectory_dir}")

    # Create memory for both successful only and all episodes
    print("\n1. Creating memory with ALL episodes...")
    memory_path_all = create_memento_memory(
        trajectory_dir=trajectory_dir,
        output_memory_dir="/home/shuqing/partnr-planner/memory_from_trajectories_all",
        filter_successful_only=False,
    )

    print("\n2. Creating memory with SUCCESSFUL episodes only...")
    memory_path_success = create_memento_memory(
        trajectory_dir=trajectory_dir,
        output_memory_dir="/home/shuqing/partnr-planner/memory_from_trajectories_success",
        filter_successful_only=True,
    )

    print("\n✅ MEMENTO Memory Creation Complete!")
    print("\nCreated two memory variants:")
    print(f"  All episodes: {memory_path_all}")
    print(f"  Successful only: {memory_path_success}")

    print("\n📋 Usage in your planner config:")
    print("For all episodes:")
    print(f"  rag_dataset_dir=[{os.path.dirname(memory_path_all)}/]")
    print("  memory_path=habitat_trajectories/trajectory_memory")

    print("\nFor successful episodes only:")
    print(f"  rag_dataset_dir=[{os.path.dirname(memory_path_success)}/]")
    print("  memory_path=habitat_trajectories/trajectory_memory")


if __name__ == "__main__":
    main()
