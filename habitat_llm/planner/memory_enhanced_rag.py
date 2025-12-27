#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


class MemoryEnhancedRAG:
    """
    Enhanced RAG class that supports MEMENTO-style memory loading.
    Provides scene-specific retrieval and memory-based example loading.
    """

    def __init__(
        self,
        example_type: str,
        data_dir: List[str],
        data_source_name: List[str],
        llm_config: Any,
        skills_filter: Optional[List[str]] = None,
        # Memory-specific parameters
        scene_id: Optional[str] = None,
        memory_path: Optional[str] = None,
        ensure_same_scene: bool = True,
        corresponding_memory: bool = False,
    ):
        """
        Initialize the Memory Enhanced RAG system.

        Args:
            example_type: Type of examples ("react", "summary", "zero_shot", "skills")
            data_dir: List of data directories
            data_source_name: List of data source names
            llm_config: LLM configuration
            skills_filter: Filter for specific skills (optional)
            scene_id: Current scene ID for scene-specific retrieval
            memory_path: Path to memory structure (relative to data_dir)
            ensure_same_scene: Whether to only load examples from same scene
            corresponding_memory: Whether to use corresponding memory lookup
        """
        self._device = "cuda"
        self._llm_config = llm_config
        self._example_type = example_type
        self._skills_filter = skills_filter

        # Memory-specific attributes
        self.scene_id = scene_id
        self.memory_path = memory_path
        self.ensure_same_scene = ensure_same_scene
        self.corresponding_memory = corresponding_memory

        # Determine the start header index
        if example_type == "react" or example_type == "zero_shot":
            self.start_header_idx = 1
        elif example_type == "summary":
            self.start_header_idx = 0
        elif example_type == "skills":
            self.start_header_idx = 0

        self.data_dict = {}
        self.index = 0

        # Load data from all directories
        for i in range(len(data_dir)):
            self._data_dir = data_dir[i]
            self._data_source_name = data_source_name[i]

            is_dir_exist = Path(self._data_dir)
            if not is_dir_exist.is_dir():
                raise ValueError(
                    f"The rag dataset path {self._data_dir} does not exist"
                )

            # Choose loading method based on memory configuration
            if self.memory_path:
                print(f"Loading data from memory: {self.memory_path}")
                self.load_data_from_memory()
            else:
                print("Loading data from standard format")
                if example_type == "skills":
                    self.load_skills_data()
                else:
                    self.load_data_llm()

        # Build sentence embedding
        self.build_data_embedding()

    def load_data_from_memory(self):
        """
        Load data from MEMENTO-style memory structure.
        Memory structure: memory_path/{scene_id}/prompts|traces/0/files
        """
        memory_base_path = os.path.join(self._data_dir, self.memory_path)

        if not os.path.exists(memory_base_path):
            raise ValueError(f"Memory path {memory_base_path} does not exist")

        print(f"Loading from memory base path: {memory_base_path}")

        # Find all available memory directories (model runs)
        valid_directories = []
        if os.path.isdir(memory_base_path):
            # Check if this directory contains scene subdirectories
            potential_scenes = [
                d
                for d in os.listdir(memory_base_path)
                if os.path.isdir(os.path.join(memory_base_path, d))
                and d not in ["memory_summary.json", "episode_result_log.csv"]
            ]

            if potential_scenes:
                # Check if any of these look like scene directories
                sample_scene_path = os.path.join(memory_base_path, potential_scenes[0])
                if os.path.exists(
                    os.path.join(sample_scene_path, "prompts")
                ) or os.path.exists(os.path.join(sample_scene_path, "traces")):
                    # This is already a memory run directory
                    valid_directories = [memory_base_path]
                else:
                    # This might be a directory containing multiple memory runs
                    valid_directories = [
                        os.path.join(memory_base_path, d)
                        for d in potential_scenes
                        if os.path.isdir(os.path.join(memory_base_path, d))
                    ]

        print(f"Found {len(valid_directories)} valid memory directories")

        # Collect all episode files
        prompt_files = []
        trace_files = []

        if self.ensure_same_scene and self.scene_id:
            print(f"Loading examples for scene: {self.scene_id}")
            # Load only from the specified scene
            for memory_dir in valid_directories:
                scene_path = os.path.join(memory_dir, self.scene_id)
                if os.path.exists(scene_path):
                    print(
                        f"Found scene {self.scene_id} in {os.path.basename(memory_dir)}"
                    )

                    # Collect prompt files
                    prompt_pattern = f"{scene_path}/prompts/0/prompt-episode_*.txt"
                    scene_prompts = glob.glob(prompt_pattern)
                    prompt_files.extend(scene_prompts)

                    # Collect trace files
                    trace_pattern = f"{scene_path}/traces/0/trace-episode_*.txt"
                    scene_traces = glob.glob(trace_pattern)
                    trace_files.extend(scene_traces)

                    print(
                        f"  Found {len(scene_prompts)} prompts and {len(scene_traces)} traces"
                    )
        else:
            print("Loading examples from all scenes")
            # Load from all scenes
            for memory_dir in valid_directories:
                scene_dirs = [
                    d
                    for d in os.listdir(memory_dir)
                    if os.path.isdir(os.path.join(memory_dir, d))
                    and d not in ["memory_summary.json", "episode_result_log.csv"]
                ]

                for scene_dir in scene_dirs:
                    scene_path = os.path.join(memory_dir, scene_dir)

                    prompt_pattern = f"{scene_path}/prompts/0/prompt-episode_*.txt"
                    trace_pattern = f"{scene_path}/traces/0/trace-episode_*.txt"

                    prompt_files.extend(glob.glob(prompt_pattern))
                    trace_files.extend(glob.glob(trace_pattern))

        print(
            f"Total files found: {len(prompt_files)} prompts, {len(trace_files)} traces"
        )

        if not prompt_files and not trace_files:
            raise ValueError("No episode files found in memory structure")

        # Process files based on example type
        if self._example_type == "zero_shot":
            self._process_zero_shot_memory_files(prompt_files)
        else:
            self._process_react_summary_memory_files(trace_files)

    def _process_zero_shot_memory_files(self, prompt_files: List[str]):
        """Process prompt files for zero-shot examples."""
        for prompt_file in prompt_files:
            try:
                with open(prompt_file, "r") as f:
                    prompt_content = f.read()

                # Extract episode ID from filename
                filename = os.path.basename(prompt_file)
                match = re.search(r"prompt-episode_(\d+)_", filename)
                episode_id = match.group(1) if match else "unknown"

                info = {"file": prompt_file, "agent_id": 0, "episode_id": episode_id}

                # Process prompt content
                if "Task: " in prompt_content:
                    task_index = prompt_content.index("Task: ")
                    prompt_content = prompt_content[task_index:]

                    # Extract instruction
                    first_line = prompt_content.split("\n")[0]
                    if ":" in first_line:
                        info["instruction"] = first_line.split(":", 1)[1].strip()
                    else:
                        info["instruction"] = first_line.strip()

                    info["trace"] = prompt_content

                    self.data_dict[self.index] = info
                    self.index += 1

            except Exception as e:
                print(f"Warning: Failed to process {prompt_file}: {e}")

    def _process_react_summary_memory_files(self, trace_files: List[str]):
        """Process trace files for react/summary examples."""
        agent_ids = [0, 1] if self._example_type in ["react", "summary"] else [0]

        for trace_file in trace_files:
            try:
                with open(trace_file, "r") as f:
                    trace_content = f.read()

                # Extract episode ID from filename
                filename = os.path.basename(trace_file)
                match = re.search(r"trace-episode_(\d+)_", filename)
                episode_id = match.group(1) if match else "unknown"

                # Extract agent ID from filename or default to 0
                agent_match = re.search(r"trace-episode_\d+_\d+-(\d+)\.txt", filename)
                agent_id = int(agent_match.group(1)) if agent_match else 0

                # Only process if agent_id is in our consideration list
                if agent_id not in agent_ids:
                    continue

                info = {
                    "file": trace_file,
                    "agent_id": agent_id,
                    "episode_id": episode_id,
                }

                # Extract instruction from trace
                if "Task:" in trace_content:
                    task_line = trace_content.split("Task:")[-1].split("\n")[0]
                    info["instruction"] = task_line.strip()
                else:
                    info["instruction"] = "Task not found"

                # Process trace content based on example type
                if self._example_type == "react":
                    processed_trace = self._process_react_trace(trace_content, agent_id)
                    info["trace"] = processed_trace
                elif self._example_type == "summary":
                    processed_traces = self._process_summary_trace(trace_content)
                    info["trace"] = processed_traces

                self.data_dict[self.index] = info
                self.index += 1

            except Exception as e:
                print(f"Warning: Failed to process {trace_file}: {e}")

    def _process_react_trace(self, trace_content: str, agent_id: int) -> str:
        """Process trace content for ReAct format."""
        # Replace agent-specific markers with generic ones
        processed = trace_content.replace(f"Agent_{agent_id}_", "Agent_{id}_")

        # Handle successful completion
        if "Final Thought" in processed:
            last_index = processed.rfind("\nAssigned!")
            if last_index >= 0:
                processed = (
                    processed[:last_index]
                    + " Exit!"
                    + processed[last_index + len("\nAssigned!") :]
                )
        else:
            # Add completion marker
            addition_text = f"{self._llm_config.user_tag}Agent_{agent_id}_Observation:Successful execution!\n{self._llm_config.eot_tag}{self._llm_config.assistant_tag}Thought:All objects were successfully moved, so I am done!\nFinal Thought: Exit!\n{self._llm_config.eot_tag}"
            processed = processed + addition_text

        return processed

    def _process_summary_trace(self, trace_content: str) -> List[str]:
        """Process trace content for summary format."""
        # Extract house description
        house_match = re.search(
            r"House description:(.*?)Objects in the house", trace_content, re.DOTALL
        )
        house_description = house_match.group(1).strip() if house_match else ""

        # Extract object observations
        object_matches = re.findall(
            r"Objects in the house:(.*?)Assigned!", trace_content, re.DOTALL
        )

        # Extract success/failure indicators
        success_pattern = re.compile(
            r"^" + re.escape("Assigned!") + r"\s*\n\s*(.*)", re.MULTILINE
        )
        success_indicators = success_pattern.findall(trace_content)

        # Build successful examples
        successful_traces = []
        for _i, (objects, success) in enumerate(
            zip(object_matches, success_indicators)
        ):
            if "successful" in success.lower():
                formatted_trace = (
                    f"Task:\n{self.data_dict.get(self.index-1, {}).get('instruction', '')}\n"
                    f"House description:\n{house_description}\n\n"
                    f"Objects in the house:\n{objects.strip()}\nAssigned!"
                )
                successful_traces.append(formatted_trace)

        return successful_traces

    def build_data_embedding(self):
        """Build embeddings for loaded data."""
        if not self.data_dict:
            raise ValueError("No data loaded for embedding")

        print(f"Building embeddings for {len(self.data_dict)} examples...")

        # Load embedding model
        self.embedding_model = SentenceTransformer(
            model_name_or_path="all-mpnet-base-v2", device=self._device
        )

        # Extract instructions
        instruction_list = [
            self.data_dict[index]["instruction"] for index in self.data_dict
        ]

        # Generate embeddings
        instruction_embeddings = self.embedding_model.encode(
            instruction_list,
            batch_size=32,
            convert_to_tensor=True,
        )

        # Add embeddings back to data
        for index in self.data_dict:
            self.data_dict[index]["embedding"] = instruction_embeddings[index]

    def load_data_llm(self):
        """Fallback to original data loading method."""
        # This preserves the original functionality
        # Implementation would be similar to the original RAG class
        print("Using fallback data loading method...")
        # ... (implement original loading logic)

    def load_skills_data(self):
        """Load skill-based data."""
        # Implementation for skills data loading
        print("Loading skills data...")
        # ... (implement skills loading logic)

    def retrieve_top_k_given_query(
        self,
        query: str,
        top_k: int = 1,
        agent_id: int = 0,
        related_episode_id: List[int] = None,
    ):
        """
        Retrieve top-k examples given a query.
        Enhanced with corresponding memory support.
        """
        if not query.strip():
            raise ValueError("Query text is empty")

        if len(self.data_dict) < top_k:
            raise ValueError(
                f"top_k ({top_k}) exceeds dataset size ({len(self.data_dict)})"
            )

        if agent_id not in [0, 1]:
            raise ValueError(f"Unsupported agent_id: {agent_id}")

        # Embed the query
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)

        # Filter by agent_id if available
        use_agent_id = False
        if "agent_id" in next(iter(self.data_dict.values()), {}):
            use_agent_id = True
            # Get embeddings for matching agent_id
            embeddings = torch.stack(
                [
                    self.data_dict[index]["embedding"]
                    for index in self.data_dict
                    if self.data_dict[index]["agent_id"] == agent_id
                ]
            )

            # Create index mapping
            embed_id_to_true_id = {}
            ind = 0
            for index in self.data_dict:
                if self.data_dict[index]["agent_id"] == agent_id:
                    embed_id_to_true_id[ind] = index
                    ind += 1
        else:
            # Use all embeddings
            embeddings = torch.stack(
                [self.data_dict[index]["embedding"] for index in self.data_dict]
            )

        # Compute similarity scores
        dot_scores = util.dot_score(query_embedding, embeddings)[0]
        scores, indices = torch.topk(input=dot_scores, k=top_k)

        scores = scores.cpu().numpy()
        indices = indices.cpu().numpy()

        if use_agent_id:
            indices = np.array([embed_id_to_true_id[ind] for ind in indices])

        # Handle corresponding memory if enabled
        if self.corresponding_memory and related_episode_id:
            self._handle_corresponding_memory(
                indices, embed_id_to_true_id, related_episode_id
            )

        # Log retrieved episodes
        print("Retrieved episodes (top-k):")
        used_episodes = []
        for idx in indices:
            true_idx = embed_id_to_true_id[idx] if use_agent_id else idx
            episode_id = self.data_dict[true_idx].get("episode_id", "unknown")
            used_episodes.append(episode_id)
        print(used_episodes)

        return scores, indices

    def _handle_corresponding_memory(
        self, indices, embed_id_to_true_id, related_episode_id
    ):
        """Handle corresponding memory lookup for related episodes."""
        found_episodes = set()

        # Check which related episodes are already in results
        for _i, idx in enumerate(indices):
            true_idx = embed_id_to_true_id[idx] if embed_id_to_true_id else idx
            current_episode = str(self.data_dict[true_idx].get("episode_id", ""))
            if current_episode in [str(ep_id) for ep_id in related_episode_id]:
                found_episodes.add(current_episode)

        # Add missing related episodes
        missing_episodes = (
            set(str(ep_id) for ep_id in related_episode_id) - found_episodes
        )
        for missing_ep in missing_episodes:
            # Find the episode in the dataset
            for embed_id, true_id in embed_id_to_true_id.items():
                episode_id = str(self.data_dict[true_id].get("episode_id", ""))
                if episode_id == missing_ep:
                    # Replace a random index
                    random_pos = np.random.randint(len(indices))
                    indices[random_pos] = embed_id
                    print(f"Added related episode {missing_ep} to results")
                    break

    def get_example_trace(self, index: int) -> str:
        """Get the trace content for a specific example."""
        if index not in self.data_dict:
            raise ValueError(f"Index {index} not found in dataset")

        trace = self.data_dict[index]["trace"]
        if isinstance(trace, list):
            # For summary format, return the first example
            return trace[0] if trace else ""
        return trace

    def get_example_instruction(self, index: int) -> str:
        """Get the instruction for a specific example."""
        if index not in self.data_dict:
            raise ValueError(f"Index {index} not found in dataset")

        return self.data_dict[index]["instruction"]

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset."""
        info = {
            "total_examples": len(self.data_dict),
            "example_type": self._example_type,
            "memory_enabled": self.memory_path is not None,
            "scene_filtering": self.ensure_same_scene,
            "current_scene": self.scene_id,
        }

        if self.data_dict:
            # Count by agent_id if available
            agent_counts = {}
            for data in self.data_dict.values():
                agent_id = data.get("agent_id", 0)
                agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
            info["agent_distribution"] = agent_counts

        return info
