#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import glob
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


def episode_scene_map(dataset) -> Dict[str, str]:
    """
    Create a mapping from episode_id to scene_id.

    Args:
        dataset: Dataset object with episodes attribute containing episode information

    Returns:
        Dict mapping episode_id to scene_id
    """
    ep_scene_map_dict = {}

    if hasattr(dataset, "episodes"):
        for episode in dataset.episodes:
            ep_scene_map_dict[str(episode.episode_id)] = episode.scene_id
    else:
        # Handle case where dataset structure is different
        print("Warning: Dataset does not have 'episodes' attribute")

    return ep_scene_map_dict


def build_memory(
    output_path: str,
    data_path: str,
    dataset: Optional[Any] = None,
    memory_base_dir: str = "/home/shuqing/partnr-planner/memory",
    filter_successful_only: bool = True,
) -> str:
    """
    Builds the memory structure for RAG-based planning by organizing episode traces and prompts by scene.
    Based on the MEMENTO memory generation method.

    Args:
        output_path (str): Path to the experiment output directory containing results
        data_path (str): Path to the dataset file
        dataset: Dataset object containing episode information (optional)
        memory_base_dir (str): Base directory for storing memory files
        filter_successful_only (bool): Whether to only include successful episodes

    Returns:
        str: The constructed memory path

    Example:
        Input structure:
        /outputs/experiment/run_name/results/dataset_name/prompts/0/prompt-episode_X_0-0.txt
        /outputs/experiment/run_name/results/dataset_name/traces/0/trace-episode_X_0-0.txt

        Output structure:
        /memory/experiment/run_name/scene_id/prompts/0/prompt-episode_X_0-0.txt
        /memory/experiment/run_name/scene_id/traces/0/trace-episode_X_0-0.txt
    """

    # Parse job and run names from output path
    # Example: /outputs/experiment/run_name/results -> experiment, run_name
    path_parts = output_path.rstrip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError(f"Invalid output_path format: {output_path}")

    # Find the results directory and extract job/run names
    try:
        results_idx = path_parts.index("results")
        run_name = path_parts[results_idx - 1]
        job_name = path_parts[results_idx - 2] if results_idx >= 2 else "default"
    except (ValueError, IndexError):
        # Fallback: use last two directories before results
        if "results" in output_path:
            job_name = path_parts[-3] if len(path_parts) >= 3 else "experiment"
            run_name = path_parts[-2] if len(path_parts) >= 2 else "default_run"
        else:
            job_name = path_parts[-2] if len(path_parts) >= 2 else "experiment"
            run_name = path_parts[-1] if len(path_parts) >= 1 else "default_run"

    print(f"Building memory for job: {job_name}, run: {run_name}")

    # Extract dataset name from data_path
    dataset_name = Path(data_path).stem
    if dataset_name.endswith(".json"):
        dataset_name = dataset_name[:-5]  # Remove .json extension
    if dataset_name.endswith(".gz"):
        dataset_name = dataset_name[:-3]  # Remove .gz extension

    # Find prompt and trace files
    prompt_pattern = f"{output_path}/{dataset_name}/prompts/0/prompt-episode_*.txt"
    trace_pattern = f"{output_path}/{dataset_name}/traces/0/trace-episode_*.txt"

    prompt_files = glob.glob(prompt_pattern, recursive=True)
    trace_files = glob.glob(trace_pattern, recursive=True)

    print(f"Found {len(prompt_files)} prompt files and {len(trace_files)} trace files")

    if not prompt_files and not trace_files:
        # Try alternative patterns
        prompt_files = glob.glob(
            f"{output_path}/**/prompt-episode_*.txt", recursive=True
        )
        trace_files = glob.glob(f"{output_path}/**/trace-episode_*.txt", recursive=True)
        print(
            f"Alternative search found {len(prompt_files)} prompt files and {len(trace_files)} trace files"
        )

    if not prompt_files and not trace_files:
        raise ValueError(f"No episode files found in {output_path}")

    # Create episode-scene mapping
    ep_scene_map_dict = {}
    if dataset:
        ep_scene_map_dict = episode_scene_map(dataset)
        print(f"Created episode-scene mapping for {len(ep_scene_map_dict)} episodes")

    # Filter successful episodes if requested
    successful_episodes = set()
    if filter_successful_only:
        episode_log_path = f"{output_path}/episode_result_log.csv"
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
                            if success >= 1.0:
                                successful_episodes.add(episode_id)
                print(f"Found {len(successful_episodes)} successful episodes")
            except Exception as e:
                print(f"Warning: Could not parse episode results: {e}")
                filter_successful_only = False

    # Create base memory directory
    base_memory_path = os.path.join(memory_base_dir, job_name, run_name)
    os.makedirs(base_memory_path, exist_ok=True)

    # Copy episode result log if it exists
    episode_log_path = f"{output_path}/episode_result_log.csv"
    if os.path.exists(episode_log_path):
        shutil.copy(
            episode_log_path, os.path.join(base_memory_path, "episode_result_log.csv")
        )

    # Process prompt files
    processed_episodes = set()
    for file_path in prompt_files:
        file_name = os.path.basename(file_path)

        # Extract episode_id from filename
        match = re.search(r"prompt-episode_(\d+)_\d+-0\.txt", file_name)
        if match:
            episode_id = match.group(1)

            # Skip if filtering successful episodes and this episode is not successful
            if (
                filter_successful_only
                and successful_episodes
                and episode_id not in successful_episodes
            ):
                continue

            # Get scene_id from mapping or use default
            scene_id = ep_scene_map_dict.get(episode_id, f"scene_{episode_id}")

            # Create destination directory
            dest_dir = os.path.join(base_memory_path, scene_id, "prompts", "0")
            os.makedirs(dest_dir, exist_ok=True)

            # Copy file
            dest_path = os.path.join(dest_dir, file_name)
            shutil.copy(file_path, dest_path)
            processed_episodes.add(episode_id)

    # Process trace files
    for file_path in trace_files:
        file_name = os.path.basename(file_path)

        # Extract episode_id from filename
        match = re.search(r"trace-episode_(\d+)_\d+-0\.txt", file_name)
        if match:
            episode_id = match.group(1)

            # Skip if filtering successful episodes and this episode is not successful
            if (
                filter_successful_only
                and successful_episodes
                and episode_id not in successful_episodes
            ):
                continue

            # Get scene_id from mapping or use default
            scene_id = ep_scene_map_dict.get(episode_id, f"scene_{episode_id}")

            # Create destination directory
            dest_dir = os.path.join(base_memory_path, scene_id, "traces", "0")
            os.makedirs(dest_dir, exist_ok=True)

            # Copy file
            dest_path = os.path.join(dest_dir, file_name)
            shutil.copy(file_path, dest_path)
            processed_episodes.add(episode_id)

    print(f"Processed {len(processed_episodes)} episodes into memory structure")
    print(f"Memory built at: {base_memory_path}")

    # Create a summary file
    summary_data = {
        "job_name": job_name,
        "run_name": run_name,
        "dataset_name": dataset_name,
        "total_episodes": len(processed_episodes),
        "successful_episodes_only": filter_successful_only,
        "created_at": str(Path().cwd()),
        "source_path": output_path,
    }

    with open(os.path.join(base_memory_path, "memory_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    return base_memory_path


def list_available_memories(
    memory_base_dir: str = "/home/shuqing/partnr-planner/memory",
) -> Dict[str, Dict[str, Any]]:
    """
    List all available memory structures.

    Args:
        memory_base_dir (str): Base directory containing memory structures

    Returns:
        Dict containing information about available memories
    """
    memories = {}

    if not os.path.exists(memory_base_dir):
        return memories

    for job_name in os.listdir(memory_base_dir):
        job_path = os.path.join(memory_base_dir, job_name)
        if os.path.isdir(job_path):
            memories[job_name] = {}

            for run_name in os.listdir(job_path):
                run_path = os.path.join(job_path, run_name)
                if os.path.isdir(run_path):
                    # Count scenes and episodes
                    scene_count = 0
                    episode_count = 0

                    for scene_name in os.listdir(run_path):
                        scene_path = os.path.join(run_path, scene_name)
                        if os.path.isdir(scene_path) and scene_name not in [
                            "memory_summary.json",
                            "episode_result_log.csv",
                        ]:
                            scene_count += 1

                            # Count episodes in this scene
                            traces_path = os.path.join(scene_path, "traces", "0")
                            if os.path.exists(traces_path):
                                trace_files = glob.glob(
                                    f"{traces_path}/trace-episode_*.txt"
                                )
                                episode_count += len(trace_files)

                    memories[job_name][run_name] = {
                        "path": run_path,
                        "scenes": scene_count,
                        "episodes": episode_count,
                    }

                    # Load summary if available
                    summary_path = os.path.join(run_path, "memory_summary.json")
                    if os.path.exists(summary_path):
                        try:
                            with open(summary_path, "r") as f:
                                summary = json.load(f)
                                memories[job_name][run_name].update(summary)
                        except Exception as e:
                            print(
                                f"Warning: Could not load summary from {summary_path}: {e}"
                            )

    return memories


if __name__ == "__main__":
    # Example usage
    output_path = "/home/shuqing/partnr-planner/outputs/experiment/run_001/results"
    data_path = "/home/shuqing/partnr-planner/data/dataset.json"

    try:
        memory_path = build_memory(output_path, data_path)
        print(f"Memory built successfully at: {memory_path}")

        # List available memories
        memories = list_available_memories()
        print("Available memories:")
        for job, runs in memories.items():
            for run, info in runs.items():
                print(
                    f"  {job}/{run}: {info['episodes']} episodes across {info['scenes']} scenes"
                )

    except Exception as e:
        print(f"Error building memory: {e}")
