#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Episode:
    """Represents a single episode with scene and episode information."""

    episode_id: str
    scene_id: str

    def __init__(self, episode_id: str, scene_id: str):
        self.episode_id = str(episode_id)
        self.scene_id = scene_id


class MemoryManagementDatasetV0:
    """
    Dataset interface for memory management compatible with MEMENTO memory building.
    Provides episode-to-scene mapping functionality.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        episodes_data: Optional[List[Dict]] = None,
    ):
        """
        Initialize the memory management dataset.

        Args:
            dataset_path: Path to dataset file (JSON or JSON.gz)
            episodes_data: Pre-loaded episodes data as list of dicts
        """
        self.episodes: List[Episode] = []
        self._episode_scene_map: Dict[str, str] = {}

        if dataset_path:
            self.load_from_file(dataset_path)
        elif episodes_data:
            self.load_from_data(episodes_data)

    def load_from_file(self, dataset_path: str):
        """
        Load dataset from file (supports .json and .json.gz).

        Args:
            dataset_path: Path to the dataset file
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Load data based on file extension
        if dataset_path.suffix == ".gz":
            with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        self._parse_dataset(data)

    def load_from_data(self, episodes_data: List[Dict]):
        """
        Load dataset from pre-loaded data.

        Args:
            episodes_data: List of episode dictionaries
        """
        self._parse_dataset({"episodes": episodes_data})

    def _parse_dataset(self, data: Dict):
        """
        Parse dataset structure and extract episodes.

        Args:
            data: Dataset dictionary containing episodes
        """
        episodes_list = []

        # Handle different possible data structures
        if "episodes" in data:
            episodes_list = data["episodes"]
        elif isinstance(data, list):
            episodes_list = data
        else:
            # Try to find episodes in the data structure
            for _key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    # Check if first item looks like an episode
                    first_item = value[0]
                    if isinstance(first_item, dict) and (
                        "episode_id" in first_item or "scene_id" in first_item
                    ):
                        episodes_list = value
                        break

        # Parse episodes
        for episode_data in episodes_list:
            if isinstance(episode_data, dict):
                # Extract episode_id and scene_id with various possible keys
                episode_id = self._extract_episode_id(episode_data)
                scene_id = self._extract_scene_id(episode_data)

                if episode_id is not None and scene_id is not None:
                    episode = Episode(episode_id, scene_id)
                    self.episodes.append(episode)
                    self._episode_scene_map[str(episode_id)] = scene_id

        print(f"Loaded {len(self.episodes)} episodes from dataset")

    def _extract_episode_id(self, episode_data: Dict) -> Optional[str]:
        """Extract episode ID from episode data with various possible key names."""
        possible_keys = ["episode_id", "id", "episode", "ep_id", "episodeId"]

        for key in possible_keys:
            if key in episode_data:
                return str(episode_data[key])

        # If no direct key found, try to extract from other fields
        if "info" in episode_data and isinstance(episode_data["info"], dict):
            for key in possible_keys:
                if key in episode_data["info"]:
                    return str(episode_data["info"][key])

        return None

    def _extract_scene_id(self, episode_data: Dict) -> Optional[str]:
        """Extract scene ID from episode data with various possible key names."""
        possible_keys = ["scene_id", "scene", "scene_name", "sceneId"]

        for key in possible_keys:
            if key in episode_data:
                return str(episode_data[key])

        # Try nested structures
        if "info" in episode_data and isinstance(episode_data["info"], dict):
            for key in possible_keys:
                if key in episode_data["info"]:
                    return str(episode_data["info"][key])

        # Try to extract from scene path
        if "scene_path" in episode_data:
            scene_path = episode_data["scene_path"]
            if isinstance(scene_path, str):
                # Extract scene name from path
                return Path(scene_path).stem

        return None

    def get_episode_scene_map(self) -> Dict[str, str]:
        """
        Get mapping from episode_id to scene_id.

        Returns:
            Dictionary mapping episode_id to scene_id
        """
        return self._episode_scene_map.copy()

    def get_episodes_by_scene(self, scene_id: str) -> List[Episode]:
        """
        Get all episodes for a specific scene.

        Args:
            scene_id: Scene identifier

        Returns:
            List of episodes in the specified scene
        """
        return [ep for ep in self.episodes if ep.scene_id == scene_id]

    def get_scenes(self) -> List[str]:
        """
        Get list of all unique scene IDs in the dataset.

        Returns:
            List of unique scene identifiers
        """
        return list(set(ep.scene_id for ep in self.episodes))

    def add_episode(self, episode_id: str, scene_id: str):
        """
        Add a new episode to the dataset.

        Args:
            episode_id: Episode identifier
            scene_id: Scene identifier
        """
        episode = Episode(episode_id, scene_id)
        self.episodes.append(episode)
        self._episode_scene_map[str(episode_id)] = scene_id

    def __len__(self) -> int:
        """Return number of episodes in dataset."""
        return len(self.episodes)

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return f"MemoryManagementDatasetV0({len(self.episodes)} episodes, {len(self.get_scenes())} scenes)"


def create_dataset_from_traces(traces_dir: str) -> MemoryManagementDatasetV0:
    """
    Create a dataset from existing trace files by parsing episode and scene information.

    Args:
        traces_dir: Directory containing trace files

    Returns:
        MemoryManagementDatasetV0 instance
    """
    import glob
    import re

    dataset = MemoryManagementDatasetV0()

    # Find all trace files
    trace_files = glob.glob(f"{traces_dir}/**/trace-episode_*.txt", recursive=True)

    for trace_file in trace_files:
        # Extract episode_id from filename
        match = re.search(r"trace-episode_(\d+)_", Path(trace_file).name)
        if match:
            episode_id = match.group(1)

            # Try to extract scene_id from directory structure
            # Assume structure: .../scene_id/traces/0/trace-episode_X.txt
            parts = Path(trace_file).parts
            if len(parts) >= 3:
                # Look for scene_id in the path
                scene_id = None
                for i, part in enumerate(parts):
                    if part == "traces" and i > 0:
                        scene_id = parts[i - 1]
                        break

                if scene_id:
                    dataset.add_episode(episode_id, scene_id)
                else:
                    # Use filename as fallback
                    scene_id = f"scene_{episode_id}"
                    dataset.add_episode(episode_id, scene_id)

    return dataset


if __name__ == "__main__":
    # Example usage

    # Create dataset from sample data
    sample_episodes = [
        {"episode_id": "001", "scene_id": "bedroom_scene"},
        {"episode_id": "002", "scene_id": "kitchen_scene"},
        {"episode_id": "003", "scene_id": "bedroom_scene"},
    ]

    dataset = MemoryManagementDatasetV0(episodes_data=sample_episodes)
    print(f"Created dataset: {dataset}")

    # Test episode-scene mapping
    mapping = dataset.get_episode_scene_map()
    print(f"Episode-Scene mapping: {mapping}")

    # Test scene-specific episodes
    bedroom_episodes = dataset.get_episodes_by_scene("bedroom_scene")
    print(f"Bedroom episodes: {[ep.episode_id for ep in bedroom_episodes]}")

    # Test unique scenes
    scenes = dataset.get_scenes()
    print(f"Unique scenes: {scenes}")
