#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import csv
import glob
import gzip
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# Import hierarchical retrieval module if available
try:
    from our_method.hierarchical_retrieval import (
        HierarchicalRetriever,
        Query,
        RetrievedSkill,
    )
    HAS_HIERARCHICAL_RETRIEVAL = True
except ImportError:
    HAS_HIERARCHICAL_RETRIEVAL = False
    print("Warning: our_method.hierarchical_retrieval not available. 'hierarchical' example_type will not work.")

# Import MEMENTO retrieval module if available
try:
    from methods.MEMENTO.memento_retriever import MementoRetriever, MementoRAG
    from methods.MEMENTO.user_profile_memory import UserProfileMemory, KnowledgeNode
    HAS_MEMENTO_RETRIEVAL = True
except ImportError:
    HAS_MEMENTO_RETRIEVAL = False
    print("Warning: methods.MEMENTO not available. 'memento' example_type will not work.")


class RAG:
    def __init__(
        self,
        example_type,
        data_dir,
        data_source_name,
        llm_config,
        skills_filter=None,
        scene_id=None,
        memory_path=None,
        ensure_same_scene=True,
        # Ablation configuration for hierarchical retrieval
        ablation_config=None,
    ):
        self._device = "cuda"
        self._llm_config = llm_config
        self._example_type = example_type
        self._skills_filter = skills_filter

        # Memory-specific parameters (MEMENTO integration)
        self.scene_id = scene_id
        self.memory_path = memory_path
        self.ensure_same_scene = ensure_same_scene

        # Ablation configuration for hierarchical retrieval experiments
        # Default: full method with all components enabled
        self.ablation_config = ablation_config or {
            'include_coop_skills': True,      # Include L_coop
            'include_ind_skills': True,       # Include L_ind
            'use_hierarchical_structure': True,  # Use hierarchical vs flat
            'random_retrieval': False,        # Random vs similarity-based
        }

        # Determine the start header index
        if example_type == "react" or example_type == "zero_shot":
            self.start_header_idx = 1
        elif example_type == "summary":
            # In summary based, the initial prompt is not included
            # in the trace file
            self.start_header_idx = 0
        elif example_type == "skills":
            self.start_header_idx = 0
        elif example_type == "hierarchical":
            self.start_header_idx = 0
        elif example_type == "memento":
            self.start_header_idx = 0

        self.data_dict = {}
        self.index = 0

        # Hierarchical retriever for hierarchical example_type
        self.hierarchical_retriever = None

        # MEMENTO retriever for memento example_type
        self.memento_retriever = None
        self.memento_rag = None

        # Handle hierarchical example_type specially
        if example_type == "hierarchical":
            if not HAS_HIERARCHICAL_RETRIEVAL:
                raise ImportError(
                    "example_type='hierarchical' requires our_method.hierarchical_retrieval module. "
                    "Make sure the our_method package is in your Python path."
                )
            self._init_hierarchical_retrieval(data_dir)
            return  # Skip standard data loading for hierarchical mode

        # Handle MEMENTO example_type specially
        if example_type == "memento":
            if not HAS_MEMENTO_RETRIEVAL:
                raise ImportError(
                    "example_type='memento' requires methods.MEMENTO module. "
                    "Make sure the methods.MEMENTO package is in your Python path."
                )
            self._init_memento_retrieval(data_dir)
            return  # Skip standard data loading for memento mode

        for i in range(len(data_dir)):
            self._data_dir = data_dir[i]
            self._data_source_name = data_source_name[i]
            is_dir_exist = Path(self._data_dir)
            if not is_dir_exist.is_dir():
                raise ValueError(
                    f"The rag dataset path {self._data_dir} does not exist"
                )
            # Load the data - use memory if available
            if self.memory_path:
                print(f"Loading from memory: {self.memory_path}")
                self.load_data_from_memory()
            elif example_type == "skills":
                self.load_skills_data()
            else:
                self.load_data_llm()

        # Build sentence embedding
        self.build_data_embedding()

    def build_data_embedding(self):
        """Index the obtain the embedding of the dataset"""

        # Load embedding model
        self.embedding_model = SentenceTransformer(
            model_name_or_path="all-mpnet-base-v2", device=self._device
        )

        # Turn text files into a single list
        instruction_list = [
            self.data_dict[index]["instruction"] for index in self.data_dict
        ]
        # Get instruction_embeddings with size of num_of_instruction X embedding size
        instruction_embeddings = self.embedding_model.encode(
            instruction_list,
            batch_size=32,  # you can use different batch sizes here for speed/performance, I found 32 works well for this use case
            convert_to_tensor=True,
        )  # optional to return embeddings as tensor instead of array

        # Add instruction back to the dict
        for index in self.data_dict:
            info = self.data_dict[index]
            info["embedding"] = instruction_embeddings[index]
            self.data_dict[index] = info

    def load_data_llm(self):
        """Load the example prompts based on LLM's dataset format"""
        # First find the csv file and filter out the trace that is successful
        file_csv = glob.glob(f"{self._data_dir}episode_result_log.csv")[0]
        first_enter = True
        select_epid_list = []
        with open(file_csv, newline="") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
            for row in spamreader:
                if not first_enter:
                    state_success = float(row[-1].split(",")[-1])
                    epid = int(row[0].split(",")[0])
                    if state_success == 1.0:
                        select_epid_list.append(epid)
                else:
                    first_enter = False

        if self._example_type == "react" or self._example_type == "summary":
            consider_agent_id_list = [0, 1]  # 支持两个agent
        if self._example_type == "zero_shot":
            agent_id = 0
            for epid in select_epid_list:
                prompt_file = f"{self._data_dir}/{self._data_source_name}/prompts/0/prompt-episode_{epid}_0-0.txt"
                prompt_content = ""
                info = {}
                with open(prompt_file, "r") as f:
                    # Get the line
                    if "file" not in info:
                        info["file"] = prompt_file
                    for line in f:
                        prompt_content += line
                    info["agent_id"] = agent_id
                    # Remove system header instructions
                    task_index = prompt_content.index("Task: ")
                    prompt_content = prompt_content[task_index:]
                    # removes the possible actions and format description
                    possible_actions_index = prompt_content.index("Possible Actions:")
                    assistant_index = prompt_content.index(
                        "<|start_header_id|>assistant<|end_header_id|>"
                    )
                    prompt_content = (
                        prompt_content[:possible_actions_index]
                        + prompt_content[assistant_index:]
                    )
                    # Get the instruction from the first line
                    info["instruction"] = (
                        prompt_content.split("\n")[0].split(":", 1)[1].strip()
                    )
                    info["trace"] = prompt_content
                self.data_dict[self.index] = info
                self.index += 1
            assert (
                len(self.data_dict) > 0
            ), "Loading RAG dataset is not successful -- the dataset is zero length"
            return

        # Then, read the text file and process the trace
        for agent_id in consider_agent_id_list:
            for epid in select_epid_list:
                trace_dir = f"{self._data_dir}/{self._data_source_name}/traces/{agent_id}/trace-episode_{epid}_0-{agent_id}.txt"
                prompt_content = ""
                info = {}
                with open(trace_dir, "r") as f:
                    # Get the line
                    for line in f:
                        prompt_content += line
                        if "instruction" not in info:
                            info["instruction"] = line.split("Task:")[-1][
                                self.start_header_idx : -2
                            ]
                        if "file" not in info:
                            info["file"] = trace_dir

                    # Keep the specific agent_id for better agent-specific examples
                    # Don't replace agent_id with generic {id} placeholder
                    # Store original agent_id for later use
                    original_agent_id = agent_id

                    # Replace for compatibility with current prompt format
                    prompt_content = prompt_content.replace(
                        f"Agent_{agent_id}_Observation", "Agent_{id}_Observation"
                    )
                    prompt_content = prompt_content.replace(
                        f"Agent_{agent_id}_Action", "Agent_{id}_Action"
                    )
                    if self._example_type == "react":
                        # Remove the final assign string
                        if "Final Thought" in prompt_content:
                            last_index = prompt_content.rfind("\nAssigned!")
                            info["trace"] = (
                                prompt_content[:last_index]
                                + " Exit!"
                                + prompt_content[last_index + len("\nAssigned!") :]
                            )
                        else:
                            addition_text = f"{self._llm_config.user_tag}Agent_{agent_id}_Observation:Successful execution!\n{self._llm_config.eot_tag}{self._llm_config.assistant_tag}Thought:All objects were successfully moved, so I am done!\nFinal Thought: Exit!\n{self._llm_config.eot_tag}"
                            info["trace"] = prompt_content + addition_text

                    elif self._example_type == "summary":
                        # For summary based approach the trace is a list of the example
                        # that we can sample from.
                        # Group the content into several smaller examples

                        # Select the text for house description
                        match = re.search(
                            r"House description:(.*?)Objects in the house",
                            prompt_content,
                            re.DOTALL,
                        )
                        house_description = match.group(1).strip()

                        # Now find the assigned between text chunk
                        matches = re.findall(
                            r"Objects in the house:(.*?)Assigned!",
                            prompt_content,
                            re.DOTALL,
                        )
                        object_obs_summary = []
                        for _, match in enumerate(
                            matches
                        ):  # Start from 1 to skip text before the first "Assigned!"
                            object_obs_summary.append(match.strip())

                        # For the line next to "Assigned!". This is used to select if we want to add
                        # that example in the dataset
                        pattern = re.compile(
                            r"^" + re.escape("Assigned!") + r"\s*\n\s*(.*)",
                            re.MULTILINE,
                        )
                        succeed_or_fail = pattern.findall(prompt_content)

                        # Filter out successful traces
                        prompt_content_list = []
                        for i in range(len(succeed_or_fail)):
                            if "successful" in succeed_or_fail[i].lower():
                                # Format the prompt
                                content_i = (
                                    "Task:\n"
                                    + info["instruction"]
                                    + "\n"
                                    + "House description:\n"
                                    + house_description
                                    + "\n\n"
                                    + "Objects in the house:\n"
                                    + object_obs_summary[i]
                                    + "\nAssigned!"
                                )
                                prompt_content_list.append(content_i)

                        info["trace"] = prompt_content_list

                        # Example
                        """
                        Task:
                        Move all objects from sofa to bedroom and place them next to the toy truck.

                        House description:
                        living_room_0: chair_0, chair_1, chair_2, chair_3, table_0, couch_0, couch_1, table_1, table_2, table_3
                        closet_0: shelves_0
                        bedroom_0: bed_0, chest_of_drawers_0, chest_of_drawers_1
                        kitchen_1: cabinet_0, table_4, chair_4, chair_5, chair_6, chair_7
                        bedroom_1: bed_1, chest_of_drawers_2, chest_of_drawers_3
                        bedroom_2: bed_2, chest_of_drawers_4, chest_of_drawers_5, wardrobe_0, wardrobe_1
                        laundryroom/mudroom_0: washer_dryer_0, washer_dryer_1, shelves_1, shelves_2
                        bathroom_0: toilet_0
                        bathroom_2: toilet_1
                        bathroom_1: toilet_2
                        kitchen_0: fridge_0
                        garage_0: fridge_1

                        Objects in the house:
                        cherry_0: couch_0
                        apple_0: agent_0
                        banana_0: couch_0
                        toy_fire_truck_0: bed_1

                        Task progress:
                        Agent 0 picked apple_0 and is currently walking.
                        Agent 1 is walking somewhere.

                        Your agent's observations of the last executed action (if available):
                        Agent_1_observation: Unexpected failure! - Failed to pick! This object is with another agent.

                        Thought: Based on the task and the list of objects in the house, the current task-relevant objects are cherry_0, banana_0, apple_0 located on the couch_0, couch_0, and agent_0 respectively. The desired location for these objects on the bed, specifically next to the toy truck based on the task description. So I will choose the bed where toy truck is located as target location for these objects. I will use the exact name of the bed provided in house description. Based on the object locations provided in the object list and the task progress summary, Agent 0 is rearranging apple_0. Agent 1's previous action execution failed because Agent 1 was already rearranging that object. So, I will ask my Agent 1 to rearrange one of the other task-relevant objects cherry_0 or banana_0.
                        Agent_1_Action: Rearrange[cherry_0, on, bed_1, next_to, toy_fire_truck_0]
                        Assigned!
                        """

                    info["agent_id"] = agent_id
                    info[
                        "original_agent_id"
                    ] = original_agent_id  # Store original agent_id for reference

                self.data_dict[self.index] = info
                self.index += 1

        # Make sure we store the data
        assert (
            len(self.data_dict) > 0
        ), "Loading RAG dataset is not successful -- the dataset is zero length"

    def load_skills_data(self):
        """Load skill-based organized dataset format from compressed JSON files"""

        # Find all skill files in the directory
        skill_files = glob.glob(f"{self._data_dir}/skill_*.json.gz")

        if not skill_files:
            raise ValueError(f"No skill files found in {self._data_dir}")

        for skill_file in skill_files:
            # Apply skills filter if specified
            if self._skills_filter:
                skill_name = (
                    Path(skill_file).stem.replace("skill_", "").replace(".json", "")
                )
                if not any(
                    skill.lower().replace(" ", "_") in skill_name.lower()
                    for skill in self._skills_filter
                ):
                    continue

            try:
                with gzip.open(skill_file, "rt") as f:
                    data = json.load(f)

                    skill_category = data["metadata"]["skill_category"]

                    for episode in data["episodes"]:
                        # Create entries for each agent if skills are available
                        if "skills" in episode:
                            for agent_id_str, skill_description in episode[
                                "skills"
                            ].items():
                                agent_id = int(agent_id_str)

                                info = {
                                    "instruction": episode["instruction"],
                                    "skill_category": skill_category,
                                    "task_type": episode.get("task_type", "Unknown"),
                                    "complexity": episode.get("complexity", "Unknown"),
                                    "agent_id": agent_id,
                                    "trace": skill_description,
                                    "file": skill_file,
                                    "episode_id": episode["episode_id"],
                                }

                                self.data_dict[self.index] = info
                                self.index += 1
                        else:
                            # Fallback: create a general entry if no agent-specific skills
                            info = {
                                "instruction": episode["instruction"],
                                "skill_category": skill_category,
                                "task_type": episode.get("task_type", "Unknown"),
                                "complexity": episode.get("complexity", "Unknown"),
                                "agent_id": 0,  # default to agent 0
                                "trace": f"Task: {episode['instruction']}",
                                "file": skill_file,
                                "episode_id": episode["episode_id"],
                            }

                            self.data_dict[self.index] = info
                            self.index += 1

            except (json.JSONDecodeError, gzip.BadGzipFile) as e:
                print(f"Warning: Could not load {skill_file}: {e}")
                continue

        # Make sure we loaded some data
        assert (
            len(self.data_dict) > 0
        ), "Loading skills dataset is not successful -- the dataset is zero length"

    def _init_hierarchical_retrieval(self, data_dir: List[str]):
        """
        Initialize hierarchical retrieval for our method.

        The hierarchical retrieval pipeline follows:
        1. Query Generation: q_t = f_query(w_t^self, w_t^env, delta_w_{t-1}, g)
        2. Abstract Skill Matching: S_candidate = {s in L : sim(q_t, name(s)) > theta_abstract}
        3. Instance Retrieval: I_retrieved = union top-k{sim(context(i), w_t)}
        4. Executability Filtering

        Args:
            data_dir: List of directories containing hierarchical skill memory
        """
        print("=" * 60)
        print("Initializing Hierarchical Skill Memory Retrieval")
        print(f"Loading from {len(data_dir)} memory directories")
        print("=" * 60)

        # Use the first data_dir as the primary memory path
        memory_path = data_dir[0] if data_dir else None

        if not memory_path or not os.path.exists(memory_path):
            raise ValueError(
                f"Hierarchical memory path does not exist: {memory_path}"
            )

        # Initialize the hierarchical retriever with the first directory
        self.hierarchical_retriever = HierarchicalRetriever(
            memory_path=memory_path,
            abstract_threshold=0.3,  # theta_abstract
            instance_top_k=5,
        )

        print(f"Loaded primary memory from: {memory_path}")
        print(f"  - Individual skills: {len(self.hierarchical_retriever.L_ind)}")
        print(f"  - Cooperation skills: {len(self.hierarchical_retriever.L_coop)}")

        # Load and merge additional memory directories
        for additional_path in data_dir[1:]:
            if additional_path and os.path.exists(additional_path):
                self._merge_hierarchical_memory(additional_path)

        # Also load a simple embedding model for instruction-based fallback
        self.embedding_model = SentenceTransformer(
            model_name_or_path="all-mpnet-base-v2", device=self._device
        )

        # Store memory paths for later use
        self.hierarchical_memory_path = memory_path
        self.hierarchical_memory_paths = data_dir

        # Rebuild embeddings after merging all memories
        self.hierarchical_retriever._build_embeddings()

        # Build data_dict from hierarchical memory for fallback/compatibility
        self._build_data_dict_from_hierarchical_memory()

        print("=" * 60)
        print(f"Total after merging all memories:")
        print(f"  - Individual skills: {len(self.hierarchical_retriever.L_ind)}")
        print(f"  - Cooperation skills: {len(self.hierarchical_retriever.L_coop)}")
        print("=" * 60)

    def _merge_hierarchical_memory(self, additional_path: str):
        """
        Merge additional hierarchical memory into the existing retriever.

        Args:
            additional_path: Path to additional hierarchical memory directory
        """
        print(f"\nMerging additional memory from: {additional_path}")

        # Load the additional memory files
        ind_path = os.path.join(additional_path, "L_ind_skills.json.gz")
        coop_path = os.path.join(additional_path, "L_coop_skills.json.gz")

        additional_ind = {}
        additional_coop = {}

        if os.path.exists(ind_path):
            with gzip.open(ind_path, 'rt', encoding='utf-8') as f:
                additional_ind = json.load(f)

        if os.path.exists(coop_path):
            with gzip.open(coop_path, 'rt', encoding='utf-8') as f:
                additional_coop = json.load(f)

        # Merge into the retriever's memory, handling duplicate keys
        merged_ind = 0
        merged_coop = 0

        for skill_key, skill_data in additional_ind.items():
            if skill_key not in self.hierarchical_retriever.L_ind:
                self.hierarchical_retriever.L_ind[skill_key] = skill_data
                merged_ind += 1
            else:
                # Merge instances for existing skills
                existing_instances = self.hierarchical_retriever.L_ind[skill_key].get("instances", [])
                new_instances = skill_data.get("instances", [])
                # Add non-duplicate instances
                for inst in new_instances:
                    if inst not in existing_instances:
                        existing_instances.append(inst)
                self.hierarchical_retriever.L_ind[skill_key]["instances"] = existing_instances

        for skill_key, skill_data in additional_coop.items():
            if skill_key not in self.hierarchical_retriever.L_coop:
                self.hierarchical_retriever.L_coop[skill_key] = skill_data
                merged_coop += 1
            else:
                # Merge instances for existing skills
                existing_instances = self.hierarchical_retriever.L_coop[skill_key].get("instances", [])
                new_instances = skill_data.get("instances", [])
                for inst in new_instances:
                    if inst not in existing_instances:
                        existing_instances.append(inst)
                self.hierarchical_retriever.L_coop[skill_key]["instances"] = existing_instances

        print(f"  - Added {merged_ind} new individual skills, {merged_coop} new cooperation skills")
        print(f"  - Total now: {len(self.hierarchical_retriever.L_ind)} ind, {len(self.hierarchical_retriever.L_coop)} coop")

    def _build_data_dict_from_hierarchical_memory(self):
        """
        Build a data_dict from hierarchical memory for compatibility with
        standard RAG interface.
        """
        if not self.hierarchical_retriever:
            return

        # Combine individual and cooperation skills
        all_skills = {}
        all_skills.update(self.hierarchical_retriever.L_ind)
        all_skills.update(self.hierarchical_retriever.L_coop)

        for skill_key, skill in all_skills.items():
            skill_type = "cooperation" if skill_key in self.hierarchical_retriever.L_coop else "individual"

            # Get all instances for this skill
            for i, instance in enumerate(skill.get("instances", [])):
                info = {
                    "instruction": f"{skill['description']} (from {skill['name']})",
                    "skill_name": skill["name"],
                    "skill_type": skill_type,
                    "skill_category": skill.get("category", "Unknown"),
                    "context": instance.get("context", {}),
                    "trace": instance.get("demo", ""),
                    "agent_id": 0,  # Will be determined during retrieval
                    "file": self.hierarchical_memory_path,
                    "skill_key": skill_key,
                    "instance_idx": i,
                }

                self.data_dict[self.index] = info
                self.index += 1

        # Build embeddings for the data_dict
        if self.data_dict:
            instruction_list = [
                self.data_dict[index]["instruction"] for index in self.data_dict
            ]
            instruction_embeddings = self.embedding_model.encode(
                instruction_list,
                batch_size=32,
                convert_to_tensor=True,
            )
            for index in self.data_dict:
                self.data_dict[index]["embedding"] = instruction_embeddings[index]

        print(f"Built data_dict with {len(self.data_dict)} entries from hierarchical memory")

    def _init_memento_retrieval(self, data_dir: List[str]):
        """
        Initialize MEMENTO retrieval for personalized user profile memory.

        MEMENTO uses a hierarchical knowledge graph-based user profile memory:
        1. User Level: User identification
        2. Knowledge Level: Object Semantics & User Patterns
        3. Elements Level: Objects, Patterns, Locations

        Retrieval procedure:
        1. Parse instruction to extract knowledge references
        2. Filter and search by knowledge type in subgraph
        3. Reformulate results into natural language

        Args:
            data_dir: List of directories containing MEMENTO user profile memory
        """
        print("=" * 60)
        print("Initializing MEMENTO User Profile Memory Retrieval")
        print("=" * 60)

        if not data_dir:
            raise ValueError("MEMENTO memory path not provided")

        # Find all memory files from the provided directories
        memory_files = []
        for memory_path in data_dir:
            memory_file = None
            if os.path.isfile(memory_path):
                memory_file = memory_path
            else:
                # Look for memory file in the directory
                possible_files = [
                    os.path.join(memory_path, "user_profile_memory.json.gz"),
                    os.path.join(memory_path, "user_profile_memory.json"),
                ]
                for pf in possible_files:
                    if os.path.exists(pf):
                        memory_file = pf
                        break

            if memory_file and os.path.exists(memory_file):
                memory_files.append(memory_file)
                print(f"Found MEMENTO memory: {memory_file}")
            else:
                print(f"Warning: MEMENTO memory not found in: {memory_path}")

        if not memory_files:
            raise ValueError(
                f"No MEMENTO memory files found. Looked in: {data_dir}"
            )

        # Load and merge all user profile memories
        print(f"Loading {len(memory_files)} MEMENTO memory file(s)...")
        self.memento_memory = UserProfileMemory.load(memory_files[0])

        # Merge additional memories if there are multiple
        for memory_file in memory_files[1:]:
            additional_memory = UserProfileMemory.load(memory_file)
            self._merge_memento_memories(additional_memory)
            print(f"  Merged memory from: {memory_file}")

        # Initialize the MEMENTO retriever
        self.memento_retriever = MementoRetriever(
            memory=self.memento_memory,
            embedding_model="all-mpnet-base-v2",
            top_k=5,
            similarity_threshold=0.3,
        )

        # Store memory paths for later use
        self.memento_memory_path = memory_files[0]
        self.memento_memory_paths = memory_files

        # Build data_dict from MEMENTO memory for compatibility
        self._build_data_dict_from_memento_memory()

        # Initialize embedding model for fallback
        self.embedding_model = SentenceTransformer(
            model_name_or_path="all-mpnet-base-v2", device=self._device
        )

        # Count knowledge nodes
        knowledge_count = sum(
            1 for node in self.memento_memory.nodes.values()
            if isinstance(node, KnowledgeNode)
        )

        print(f"  - Total nodes in memory: {len(self.memento_memory.nodes)}")
        print(f"  - Knowledge nodes: {knowledge_count}")
        print(f"  - Data dict entries: {len(self.data_dict)}")
        print("=" * 60)

    def _merge_memento_memories(self, additional_memory: "UserProfileMemory"):
        """
        Merge an additional UserProfileMemory into the main memory.

        Args:
            additional_memory: The UserProfileMemory to merge
        """
        # Merge nodes with unique IDs
        for node_id, node in additional_memory.nodes.items():
            # Create a unique ID if there's a conflict
            if node_id in self.memento_memory.nodes:
                new_id = f"{node_id}_{len(self.memento_memory.nodes)}"
                self.memento_memory.nodes[new_id] = node
            else:
                self.memento_memory.nodes[node_id] = node

        # Merge edges
        for edge in additional_memory.edges:
            if edge not in self.memento_memory.edges:
                self.memento_memory.edges.append(edge)

    def _build_data_dict_from_memento_memory(self):
        """
        Build a data_dict from MEMENTO user profile memory for compatibility
        with standard RAG interface.
        """
        if not self.memento_memory:
            return

        for node_id, node in self.memento_memory.nodes.items():
            if isinstance(node, KnowledgeNode):
                # Get reformulated text
                reformulated = self.memento_memory.reformulate_knowledge(node_id)

                # Get related elements
                elements = self.memento_memory.get_elements_for_knowledge(node_id)

                # Format as trace
                trace_parts = []
                if node.alias:
                    trace_parts.append(f"Personalized Reference: {node.alias}")
                if node.description:
                    trace_parts.append(f"Description: {node.description}")

                if elements.get("objects"):
                    obj_names = [obj.name for obj in elements["objects"]]
                    trace_parts.append(f"Related Objects: {', '.join(obj_names)}")

                if elements.get("patterns"):
                    for pattern in elements["patterns"]:
                        if pattern.action_name:
                            args_str = ', '.join(pattern.args) if pattern.args else ''
                            trace_parts.append(f"Action: {pattern.action_name}({args_str})")

                if elements.get("locations"):
                    loc_names = [loc.expression or loc.name for loc in elements["locations"]]
                    trace_parts.append(f"Locations: {', '.join(loc_names)}")

                info = {
                    "knowledge_id": node_id,
                    "instruction": reformulated,
                    "knowledge_type": node.knowledge_type.value,
                    "alias": node.alias,
                    "description": node.description,
                    "trace": "\n".join(trace_parts),
                    "agent_id": 0,
                    "file": self.memento_memory_path if hasattr(self, 'memento_memory_path') else "memento",
                    "objects": [obj.name for obj in elements.get("objects", [])],
                    "patterns": [p.action_name for p in elements.get("patterns", [])],
                    "locations": [loc.name for loc in elements.get("locations", [])],
                }

                self.data_dict[self.index] = info
                self.index += 1

        # Build embeddings for the data_dict
        if self.data_dict:
            # Ensure we have embedding model
            if not hasattr(self, 'embedding_model') or self.embedding_model is None:
                self.embedding_model = SentenceTransformer(
                    model_name_or_path="all-mpnet-base-v2", device=self._device
                )

            instruction_list = [
                self.data_dict[index]["instruction"] for index in self.data_dict
            ]
            instruction_embeddings = self.embedding_model.encode(
                instruction_list,
                batch_size=32,
                convert_to_tensor=True,
            )
            for index in self.data_dict:
                self.data_dict[index]["embedding"] = instruction_embeddings[index]

        print(f"Built data_dict with {len(self.data_dict)} entries from MEMENTO memory")

    def retrieve_memento(
        self,
        query: str,
        top_k: int = 3,
        agent_id: int = 0,
    ) -> List[Any]:
        """
        Perform MEMENTO retrieval using personalized knowledge graph.

        This implements MEMENTO's retrieval procedure:
        1. Parse instruction to extract knowledge references
        2. Filter and search by knowledge type
        3. Reformulate results into natural language

        Args:
            query: The task instruction
            top_k: Maximum number of memories to retrieve
            agent_id: The agent performing the retrieval

        Returns:
            List of RetrievedKnowledge objects sorted by relevance
        """
        if not self.memento_retriever:
            raise RuntimeError("MEMENTO retriever not initialized")

        # Retrieve from MEMENTO memory
        retrieved = self.memento_retriever.retrieve(query)

        # Limit to top_k
        return retrieved[:top_k]

    def format_memento_for_prompt(
        self,
        retrieved_memories: List[Any],
        max_examples: int = 3,
    ) -> str:
        """
        Format MEMENTO retrieved memories for inclusion in LLM prompt.

        Args:
            retrieved_memories: List of RetrievedKnowledge objects
            max_examples: Maximum number of examples to include

        Returns:
            Formatted string for prompt
        """
        if not retrieved_memories:
            return "No relevant personalized knowledge found in memory."

        return self.memento_retriever.format_for_prompt(
            retrieved_memories[:max_examples],
            max_items=max_examples,
        )

    def retrieve_hierarchical(
        self,
        query: str,
        agent_state: Optional[Dict[str, Any]] = None,
        environment_state: Optional[Dict[str, Any]] = None,
        partner_effects: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
        agent_id: int = 0,
    ) -> List["RetrievedSkill"]:
        """
        Perform hierarchical retrieval using our method's pipeline.

        This implements the full 4-stage retrieval:
        1. Query Generation
        2. Abstract Skill Matching
        3. Instance Retrieval
        4. Executability Filtering

        Args:
            query: The task instruction (goal)
            agent_state: Agent's internal state (holding, position)
            environment_state: Observable environment state
            partner_effects: Observed effects of partner's action
            top_k: Maximum number of skills to retrieve
            agent_id: The agent performing the retrieval

        Returns:
            List of RetrievedSkill objects sorted by relevance
        """
        if not self.hierarchical_retriever:
            raise RuntimeError("Hierarchical retriever not initialized")

        # Default states if not provided
        if agent_state is None:
            agent_state = {"holding": None, "position": None}
        if environment_state is None:
            environment_state = {}
        if partner_effects is None:
            partner_effects = {}

        # Perform hierarchical retrieval
        retrieved = self.hierarchical_retriever.retrieve(
            agent_state=agent_state,
            environment_state=environment_state,
            partner_effects=partner_effects,
            goal=query,
            include_individual=True,
            include_cooperation=True,
        )

        # Limit to top_k
        return retrieved[:top_k]

    def format_hierarchical_for_prompt(
        self,
        retrieved_skills: List["RetrievedSkill"],
        max_examples: int = 3,
    ) -> str:
        """
        Format hierarchically retrieved skills for inclusion in LLM prompt.

        IMPORTANT: This formatting includes explicit action format examples to prevent
        syntax errors in LLM output. The LLM must generate actions in exact format:
        ActionName[parameter] or ActionName[param1, param2, ...]

        Args:
            retrieved_skills: List of RetrievedSkill objects
            max_examples: Maximum number of examples to include

        Returns:
            Formatted string for prompt
        """
        if not retrieved_skills:
            return "No relevant skills found in hierarchical memory."

        parts = ["## Retrieved Skills from Hierarchical Memory"]
        parts.append("\n**CRITICAL: Action Format Reminder**")
        parts.append("All actions MUST follow this exact format: ActionName[parameter]")
        parts.append("Examples: Navigate[living_room_1], Pick[kettle_0], Place[cup_0, on, counter_29, None, None]")
        parts.append("Do NOT omit the square brackets [] around parameters!\n")

        for i, skill in enumerate(retrieved_skills[:max_examples]):
            parts.append(f"### Skill {i+1}: {skill.skill_name} ({skill.skill_type})")

            # Always show action sequence first if available (concrete examples)
            if skill.instances:
                inst = skill.instances[0]
                context = inst.get("context", {})

                # Show concrete action patterns first - this is the most important for format
                if context.get("action_sequence"):
                    action_seq = context["action_sequence"]
                    parts.append(f"\n**Concrete Action Pattern (follow this format exactly):**")
                    for action in action_seq[:5]:  # Show up to 5 actions
                        parts.append(f"  {action}")
                    if len(action_seq) > 5:
                        parts.append(f"  ... and {len(action_seq) - 5} more actions")

                if context.get("objects"):
                    parts.append(f"\nRelevant Objects: {', '.join(context['objects'][:5])}")

                # Show locations (furniture IDs where objects are/should go)
                if context.get("locations"):
                    parts.append(f"Relevant Locations: {', '.join(context['locations'][:5])}")

                # Show room information if available
                if context.get("rooms"):
                    parts.append(f"Rooms: {', '.join(context['rooms'][:5])}")

                # Show object-to-location mapping if available
                if context.get("object_locations"):
                    obj_loc_strs = []
                    for obj, info in list(context["object_locations"].items())[:5]:
                        if isinstance(info, dict):
                            loc = info.get("location", "unknown")
                            room = info.get("room", "")
                            if room:
                                obj_loc_strs.append(f"{obj} at {loc} in {room}")
                            else:
                                obj_loc_strs.append(f"{obj} at {loc}")
                        else:
                            obj_loc_strs.append(f"{obj} at {info}")
                    if obj_loc_strs:
                        parts.append(f"Object Locations: {'; '.join(obj_loc_strs)}")

            # Show demo description (high-level strategy)
            if skill.demo:
                parts.append(f"\nStrategy: {skill.demo}")

            # Add cooperation-specific guidance with action format examples
            if skill.skill_type == "cooperation":
                parts.append("\n**Cooperation Guidance:**")
                parts.append("- Coordinate with partner agent to divide tasks")
                parts.append("- Use Wait[] if partner is handling current subtask")
                parts.append("- Example actions: Navigate[object_0], Pick[object_0], Place[object_0, on, furniture_0, None, None]")

            parts.append("")

        # Add final format reminder
        parts.append("**Remember:** Every action must have format ActionName[param]. Never omit brackets!")

        return "\n".join(parts)

    def load_data_from_memory(self):
        """
        Load data from MEMENTO-style memory structure.
        Memory structure: memory_path/{scene_id}/prompts|traces/0/files
        """
        memory_base_path = os.path.join(self._data_dir, self.memory_path)

        if not os.path.exists(memory_base_path):
            raise ValueError(f"Memory path {memory_base_path} does not exist")

        print(f"Loading from memory base path: {memory_base_path}")

        # Find available scenes/episodes in memory
        prompt_files = []
        trace_files = []

        if self.ensure_same_scene and self.scene_id:
            print(f"Loading examples for scene: {self.scene_id}")
            # Load only from the specified scene
            scene_path = os.path.join(memory_base_path, self.scene_id)
            if os.path.exists(scene_path):
                prompt_pattern = f"{scene_path}/prompts/0/prompt-episode_*.txt"
                trace_pattern = f"{scene_path}/traces/0/trace-episode_*.txt"
                prompt_files = glob.glob(prompt_pattern)
                trace_files = glob.glob(trace_pattern)
                print(
                    f"Found {len(prompt_files)} prompts and {len(trace_files)} traces in scene {self.scene_id}"
                )
            else:
                print(
                    f"Warning: Scene {self.scene_id} not found in memory. Using all available scenes."
                )
                self.ensure_same_scene = False  # Fall back to all scenes

        if not self.ensure_same_scene or not self.scene_id:
            print("Loading examples from all available scenes")
            # Find all scene directories
            scene_dirs = [
                d
                for d in os.listdir(memory_base_path)
                if os.path.isdir(os.path.join(memory_base_path, d))
                and d not in ["memory_summary.json", "episode_result_log.csv"]
            ]

            for scene_dir in scene_dirs:
                scene_path = os.path.join(memory_base_path, scene_dir)
                prompt_pattern = f"{scene_path}/prompts/0/prompt-episode_*.txt"
                trace_pattern = f"{scene_path}/traces/0/trace-episode_*.txt"
                prompt_files.extend(glob.glob(prompt_pattern))
                trace_files.extend(glob.glob(trace_pattern))

            print(
                f"Found {len(prompt_files)} prompts and {len(trace_files)} traces across all scenes"
            )

        if not prompt_files and not trace_files:
            raise ValueError("No episode files found in memory structure")

        # Process files based on example type
        if self._example_type == "zero_shot":
            self._process_memory_prompt_files(prompt_files)
        else:
            self._process_memory_trace_files(trace_files)

        print(f"Loaded {len(self.data_dict)} examples from memory")

    def _process_memory_prompt_files(self, prompt_files):
        """Process prompt files from memory for zero-shot examples."""
        agent_id = 0
        for prompt_file in prompt_files:
            try:
                with open(prompt_file, "r") as f:
                    prompt_content = f.read()

                # Extract episode ID from filename
                filename = os.path.basename(prompt_file)
                match = re.search(r"prompt-episode_(\d+)_", filename)
                episode_id = match.group(1) if match else "unknown"

                info = {
                    "file": prompt_file,
                    "agent_id": agent_id,
                    "episode_id": episode_id,
                }

                # Process prompt content similar to original zero_shot logic
                if "Task: " in prompt_content:
                    task_index = prompt_content.index("Task: ")
                    prompt_content = prompt_content[task_index:]

                    # Remove possible actions if present
                    if (
                        "Possible Actions:" in prompt_content
                        and "<|start_header_id|>assistant<|end_header_id|>"
                        in prompt_content
                    ):
                        possible_actions_index = prompt_content.index(
                            "Possible Actions:"
                        )
                        assistant_index = prompt_content.index(
                            "<|start_header_id|>assistant<|end_header_id|>"
                        )
                        prompt_content = (
                            prompt_content[:possible_actions_index]
                            + prompt_content[assistant_index:]
                        )

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

    def _process_memory_trace_files(self, trace_files):
        """Process trace files from memory for react/summary examples."""
        agent_ids = [0, 1] if self._example_type in ["react", "summary"] else [0]

        for trace_file in trace_files:
            try:
                with open(trace_file, "r") as f:
                    trace_content = f.read()

                # Extract episode ID from filename
                filename = os.path.basename(trace_file)
                episode_match = re.search(r"trace-episode_(\d+)_", filename)
                episode_id = episode_match.group(1) if episode_match else "unknown"

                # For our MEMENTO memory structure, create entries for both agents
                # since the traces contain multi-agent information
                for agent_id in agent_ids:
                    info = {
                        "file": trace_file,
                        "agent_id": agent_id,
                        "episode_id": episode_id,
                    }

                    # Extract instruction from trace
                    if "Task:" in trace_content:
                        task_line = trace_content.split("Task:")[-1]
                        # Get the instruction part
                        instruction_part = (
                            task_line[self.start_header_idx :]
                            if len(task_line) > self.start_header_idx
                            else task_line.strip()
                        )
                        info["instruction"] = instruction_part.split("\n")[0].strip()
                    else:
                        info["instruction"] = "Task not found"

                    # Process trace content based on example type
                    if self._example_type == "react":
                        # Replace specific agent references with generic ones
                        processed_trace = trace_content
                        processed_trace = processed_trace.replace(
                            "Agent_0_", "Agent_{id}_"
                        )
                        processed_trace = processed_trace.replace(
                            "Agent_1_", "Agent_{id}_"
                        )

                        # Add proper completion if missing
                        if (
                            "Final Thought" not in processed_trace
                            and "Exit!" not in processed_trace
                        ):
                            addition_text = "\nAgent_{id}_observation: Successful execution!\nThought: All tasks completed successfully!\nFinal Thought: Exit!"
                            processed_trace = processed_trace.strip() + addition_text

                        info["trace"] = processed_trace

                    elif self._example_type == "summary":
                        # For summary, use the trace as is
                        info["trace"] = [trace_content]

                    else:
                        # For other types, use trace as is
                        info["trace"] = trace_content

                    info["original_agent_id"] = agent_id

                    self.data_dict[self.index] = info
                    self.index += 1

            except Exception as e:
                print(f"Warning: Failed to process {trace_file}: {e}")

    def retrieve_top_k_given_query(
        self,
        query: str,
        top_k: int = 1,
        agent_id: int = 0,
        agent_state: Optional[Dict[str, Any]] = None,
        environment_state: Optional[Dict[str, Any]] = None,
        partner_effects: Optional[Dict[str, Any]] = None,
    ):
        """
        Return the top k text/index of the examples given query and agent id.

        For hierarchical example_type, uses the 4-stage hierarchical retrieval pipeline.
        For other types, uses standard embedding-based similarity search.

        Args:
            query: The task instruction/query
            top_k: Number of examples to retrieve
            agent_id: The agent performing retrieval (0 or 1)
            agent_state: (hierarchical only) Agent's internal state
            environment_state: (hierarchical only) Observable environment state
            partner_effects: (hierarchical only) Observed effects of partner's action

        Returns:
            Tuple of (scores, indices) where indices point to entries in data_dict
        """
        assert query != "", "query text is an empty string"
        assert agent_id in [0, 1], "Do not support agent_id other than 0 and 1"

        # Handle hierarchical retrieval specially
        if self._example_type == "hierarchical" and self.hierarchical_retriever:
            return self._retrieve_hierarchical_top_k(
                query=query,
                top_k=top_k,
                agent_id=agent_id,
                agent_state=agent_state,
                environment_state=environment_state,
                partner_effects=partner_effects,
            )

        # Handle MEMENTO retrieval specially
        if self._example_type == "memento" and self.memento_retriever:
            return self._retrieve_memento_top_k(
                query=query,
                top_k=top_k,
                agent_id=agent_id,
            )

        # Standard retrieval for other example types
        assert (
            len(self.data_dict) >= top_k
        ), "top_k value exceeds the size of the RAG examples"

        # Embed the query
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)

        use_agent_id = False
        # Process embedding
        if "agent_id" in self.data_dict[0]:
            # Select the embeddings given agent_id
            agent_specific_embeddings = [
                self.data_dict[index]["embedding"]
                for index in self.data_dict
                if self.data_dict[index]["agent_id"] == agent_id
            ]

            # If no examples for this agent_id, fall back to all examples
            if len(agent_specific_embeddings) == 0:
                print(
                    f"Warning: No examples found for agent_id {agent_id}, using all available examples"
                )
                embeddings = torch.stack(
                    [self.data_dict[index]["embedding"] for index in self.data_dict]
                )
                use_agent_id = False
            else:
                embeddings = torch.stack(agent_specific_embeddings)
                # Record index conversion
                ind = 0
                embed_id_to_true_id = {}
                for index in self.data_dict:
                    if self.data_dict[index]["agent_id"] == agent_id:
                        embed_id_to_true_id[ind] = index
                        ind += 1
                use_agent_id = True
        else:
            embeddings = torch.stack(
                [self.data_dict[index]["embedding"] for index in self.data_dict]
            )

        # Compute score
        dot_scores = util.dot_score(query_embedding, embeddings)[0]

        scores = None
        # Sort based on the score
        scores, indices = torch.topk(input=dot_scores, k=top_k)

        assert scores is not None, "Cannot retrieve the information"

        scores = scores.cpu().numpy()
        indices = indices.cpu().numpy()
        if use_agent_id:
            indices = np.array([embed_id_to_true_id[ind] for ind in indices])

        return scores, indices

    def _retrieve_hierarchical_top_k(
        self,
        query: str,
        top_k: int = 1,
        agent_id: int = 0,
        agent_state: Optional[Dict[str, Any]] = None,
        environment_state: Optional[Dict[str, Any]] = None,
        partner_effects: Optional[Dict[str, Any]] = None,
    ):
        """
        Perform hierarchical retrieval and return results in standard RAG format.

        This method implements the 4-stage hierarchical retrieval pipeline:
        1. Query Generation: q_t = f_query(w_t^self, w_t^env, delta_w_{t-1}, g)
        2. Abstract Skill Matching: S_candidate = {s in L : sim(q_t, name(s)) > theta}
        3. Instance Retrieval: I_retrieved = union top-k{sim(context(i), w_t)}
        4. Executability Filtering

        Supports ablation variants via self.ablation_config:
        - include_coop_skills: Whether to include cooperation skills (L_coop)
        - include_ind_skills: Whether to include individual skills (L_ind)
        - use_hierarchical_structure: Use hierarchical vs flat retrieval
        - random_retrieval: Use random selection instead of similarity-based

        Args:
            query: The task instruction (goal)
            top_k: Number of skills to retrieve
            agent_id: The agent performing retrieval
            agent_state: Agent's internal state (holding, position)
            environment_state: Observable environment state
            partner_effects: Observed effects of partner's action

        Returns:
            Tuple of (scores, indices) compatible with standard RAG interface
        """
        print("\n" + "=" * 60)
        print("HIERARCHICAL RETRIEVAL - 4-Stage Pipeline")
        print("=" * 60)
        print(f"Query: {query}")
        print(f"Agent ID: {agent_id}")

        # Get ablation configuration
        ablation = getattr(self, 'ablation_config', {})
        include_coop = ablation.get('include_coop_skills', True)
        include_ind = ablation.get('include_ind_skills', True)
        use_hierarchical = ablation.get('use_hierarchical_structure', True)
        random_retrieval = ablation.get('random_retrieval', False)

        print(f"\nAblation Config:")
        print(f"  - include_coop_skills: {include_coop}")
        print(f"  - include_ind_skills: {include_ind}")
        print(f"  - use_hierarchical_structure: {use_hierarchical}")
        print(f"  - random_retrieval: {random_retrieval}")

        # Default states if not provided
        if agent_state is None:
            agent_state = {"holding": None, "position": None}
        if environment_state is None:
            environment_state = {}
        if partner_effects is None:
            partner_effects = {}

        # Handle different retrieval modes based on ablation config
        if random_retrieval:
            # Random retrieval ablation: randomly select from data_dict
            return self._random_retrieval(top_k)

        if not use_hierarchical:
            # Flat memory ablation: use standard embedding-based retrieval
            # on the flattened data_dict (no hierarchical structure)
            return self._flat_retrieval(query, top_k, agent_id)

        # Full hierarchical retrieval with conditional skill inclusion
        # Stage 1-4: Perform hierarchical retrieval
        retrieved_skills = self.hierarchical_retriever.retrieve(
            agent_state=agent_state,
            environment_state=environment_state,
            partner_effects=partner_effects,
            goal=query,
            include_individual=include_ind,
            include_cooperation=include_coop,
        )

        print(f"\nRetrieved {len(retrieved_skills)} skills")

        if not retrieved_skills:
            print("No skills retrieved, falling back to flat retrieval")
            return self._flat_retrieval(query, top_k, agent_id)

        # Map retrieved skills to data_dict indices
        scores = []
        indices = []

        for skill in retrieved_skills[:top_k]:
            skill_key = f"{skill.skill_name}_{skill.skill_type}"

            # Find matching entry in data_dict
            found = False
            for idx, entry in self.data_dict.items():
                if entry.get("skill_key") == skill_key or \
                   (entry.get("skill_name") == skill.skill_name and
                    entry.get("skill_type") == skill.skill_type):
                    indices.append(idx)
                    scores.append(skill.abstract_score)

                    # Update trace with hierarchical demo
                    self.data_dict[idx]["trace"] = self.format_hierarchical_for_prompt(
                        [skill], max_examples=1
                    )
                    found = True
                    break

            if not found:
                # Create a virtual entry for this skill
                virtual_idx = self._create_virtual_entry(skill, agent_id)
                indices.append(virtual_idx)
                scores.append(skill.abstract_score)

            print(f"  - {skill.skill_name} ({skill.skill_type}): score={skill.abstract_score:.3f}")

        # Print cooperation skill info if retrieved
        coop_skills = [s for s in retrieved_skills[:top_k] if s.skill_type == "cooperation"]
        if coop_skills:
            print(f"\n** Cooperation skills detected: {len(coop_skills)} **")
            for s in coop_skills:
                demo_preview = s.demo[:100] + "..." if len(s.demo) > 100 else s.demo
                print(f"   - {s.skill_name}: {demo_preview}")

        print("=" * 60 + "\n")

        return np.array(scores), np.array(indices)

    def _retrieve_memento_top_k(
        self,
        query: str,
        top_k: int = 1,
        agent_id: int = 0,
    ):
        """
        Perform MEMENTO retrieval and return results in standard RAG format.

        This method implements MEMENTO's retrieval procedure:
        1. Parse instruction to extract knowledge references
        2. Filter and search by knowledge type in subgraph
        3. Reformulate results into natural language descriptions

        Args:
            query: The task instruction
            top_k: Number of memories to retrieve
            agent_id: The agent performing retrieval

        Returns:
            Tuple of (scores, indices) compatible with standard RAG interface
        """
        print("\n" + "=" * 60)
        print("MEMENTO RETRIEVAL - User Profile Memory")
        print("=" * 60)
        print(f"Query: {query}")
        print(f"Agent ID: {agent_id}")

        # Retrieve from MEMENTO memory
        retrieved_memories = self.memento_retriever.retrieve(query)

        print(f"\nRetrieved {len(retrieved_memories)} personalized memories")

        if not retrieved_memories:
            print("No memories retrieved, returning empty results")
            return np.array([]), np.array([])

        # Map retrieved memories to data_dict indices
        scores = []
        indices = []

        for memory in retrieved_memories[:top_k]:
            # Find matching entry in data_dict by knowledge_id
            found = False
            for idx, entry in self.data_dict.items():
                if entry.get("knowledge_id") == memory.knowledge_id:
                    indices.append(idx)
                    scores.append(memory.similarity_score)

                    # Update trace with formatted memory
                    self.data_dict[idx]["trace"] = self.format_memento_for_prompt(
                        [memory], max_examples=1
                    )
                    found = True
                    break

            if not found:
                # Create a virtual entry for this memory
                virtual_idx = self._create_virtual_memento_entry(memory, agent_id)
                indices.append(virtual_idx)
                scores.append(memory.similarity_score)

            # Print retrieved memory info
            alias_str = memory.alias if memory.alias else "N/A"
            print(f"  - {memory.knowledge_type.value}: {alias_str} (score={memory.similarity_score:.3f})")
            if memory.objects:
                print(f"    Objects: {', '.join(memory.objects[:3])}")

        print("=" * 60 + "\n")

        return np.array(scores), np.array(indices)

    def _create_virtual_memento_entry(self, memory: Any, agent_id: int) -> int:
        """
        Create a virtual data_dict entry for a MEMENTO retrieved memory.

        Args:
            memory: The RetrievedKnowledge object
            agent_id: The agent ID

        Returns:
            The index of the new entry
        """
        virtual_idx = self.index
        self.index += 1

        # Format the memory for the trace
        formatted_trace = self.format_memento_for_prompt([memory], max_examples=1)

        info = {
            "knowledge_id": memory.knowledge_id,
            "instruction": memory.reformulated_text,
            "knowledge_type": memory.knowledge_type.value,
            "alias": memory.alias,
            "description": memory.description,
            "trace": formatted_trace,
            "agent_id": agent_id,
            "file": self.memento_memory_path if hasattr(self, 'memento_memory_path') else "memento",
            "objects": memory.objects,
            "patterns": memory.patterns,
            "locations": memory.locations,
            "similarity_score": memory.similarity_score,
        }

        # Add embedding for compatibility
        if hasattr(self, 'embedding_model') and self.embedding_model:
            embedding = self.embedding_model.encode(
                info["instruction"], convert_to_tensor=True
            )
            info["embedding"] = embedding

        self.data_dict[virtual_idx] = info
        return virtual_idx

    def _flat_retrieval(self, query: str, top_k: int, agent_id: int):
        """
        Flat memory retrieval (ablation: -Hierarchical Structure).
        Uses standard embedding-based similarity on flattened data_dict.

        Args:
            query: The task instruction
            top_k: Number of examples to retrieve
            agent_id: The agent ID

        Returns:
            Tuple of (scores, indices)
        """
        print("\n[ABLATION] Using FLAT retrieval (no hierarchical structure)")

        if len(self.data_dict) == 0:
            return np.array([]), np.array([])

        # Ensure we have an embedding model
        if not hasattr(self, 'embedding_model') or self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                model_name_or_path="all-mpnet-base-v2", device=self._device
            )

        # Embed the query
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)

        # Get all embeddings
        embeddings = torch.stack(
            [self.data_dict[index]["embedding"] for index in self.data_dict]
        )

        # Compute scores
        dot_scores = util.dot_score(query_embedding, embeddings)[0]

        # Get top-k
        actual_k = min(top_k, len(self.data_dict))
        scores, indices_tensor = torch.topk(input=dot_scores, k=actual_k)

        scores = scores.cpu().numpy()
        indices = indices_tensor.cpu().numpy()

        # Map tensor indices to data_dict keys
        data_dict_keys = list(self.data_dict.keys())
        indices = np.array([data_dict_keys[i] for i in indices])

        return scores, indices

    def _random_retrieval(self, top_k: int):
        """
        Random retrieval (ablation: Random Retrieval).
        Randomly selects entries from data_dict.

        Args:
            top_k: Number of examples to retrieve

        Returns:
            Tuple of (scores, indices)
        """
        print("\n[ABLATION] Using RANDOM retrieval")

        if len(self.data_dict) == 0:
            return np.array([]), np.array([])

        # Randomly select indices
        data_dict_keys = list(self.data_dict.keys())
        actual_k = min(top_k, len(data_dict_keys))

        random_indices = np.random.choice(data_dict_keys, size=actual_k, replace=False)
        random_scores = np.ones(actual_k)  # All scores = 1 for random

        return random_scores, random_indices

    def _create_virtual_entry(self, skill: "RetrievedSkill", agent_id: int) -> int:
        """
        Create a virtual data_dict entry for a hierarchically retrieved skill.

        Args:
            skill: The RetrievedSkill object
            agent_id: The agent ID

        Returns:
            The index of the new entry
        """
        virtual_idx = self.index
        self.index += 1

        # Format the skill demo for the trace
        formatted_trace = self.format_hierarchical_for_prompt([skill], max_examples=1)

        info = {
            "instruction": f"{skill.skill_name} skill for agent coordination",
            "skill_name": skill.skill_name,
            "skill_type": skill.skill_type,
            "skill_category": "Hierarchical",
            "context": skill.instances[0].get("context", {}) if skill.instances else {},
            "trace": formatted_trace,
            "agent_id": agent_id,
            "file": self.hierarchical_memory_path if hasattr(self, 'hierarchical_memory_path') else "hierarchical",
            "skill_key": f"{skill.skill_name}_{skill.skill_type}",
            "instance_idx": 0,
            "abstract_score": skill.abstract_score,
            "is_cooperation": skill.skill_type == "cooperation",
        }

        # Add embedding for compatibility
        if hasattr(self, 'embedding_model') and self.embedding_model:
            embedding = self.embedding_model.encode(
                info["instruction"], convert_to_tensor=True
            )
            info["embedding"] = embedding

        self.data_dict[virtual_idx] = info
        return virtual_idx

    def set_ablation_config(self, config: Dict[str, Any]):
        """
        Set ablation configuration for experimental variants.

        Ablation variants (from paper):
        - Full Method (Ours): All True, random_retrieval=False
        - -Hierarchical Structure: use_hierarchical_structure=False
        - -Cooperation Skills (L_coop): include_coop_skills=False
        - -Individual Skills (L_ind): include_ind_skills=False
        - Flat Memory (Raw RAG): use_hierarchical_structure=False
        - Random Retrieval: random_retrieval=True

        Args:
            config: Dict with keys:
                - include_coop_skills: bool (default True)
                - include_ind_skills: bool (default True)
                - use_hierarchical_structure: bool (default True)
                - random_retrieval: bool (default False)
        """
        self.ablation_config = config
        print(f"Ablation config set: {config}")


if __name__ == "__main__":
    example_type = "skills"  # react, summary, or skills

    _llm_config = dict(
        system_tag="<|start_header_id|>system<|end_header_id|>\n",
        user_tag="<|start_header_id|>user<|end_header_id|>\n",
        assistant_tag="<|start_header_id|>assistant<|end_header_id|>\n",
        eot_tag="<|eot_id|>\n",
    )
    llm_config = SimpleNamespace(**_llm_config)
    if example_type == "react":
        data_dir = ["path_to_rag_react_dataset/"]
        data_source_name = [
            "2024_08_01_train_mini.json.gz",
        ]
        skills_filter = None
    elif example_type == "summary":
        data_dir = [
            "path_to_rag_summary_dataset/",
        ]
        data_source_name = [
            "2024_08_01_train_mini.json.gz",
        ]
        skills_filter = None
    elif example_type == "skills":
        data_dir = ["data/rag_datasets/rerange_only_organized_by_skills/"]
        data_source_name = ["organized_skills"]  # Not used for skills format
        skills_filter = [
            "Navigation",
            "Task Planning",
        ]  # Optional: filter specific skills
    else:
        raise NotImplementedError

    rag = RAG(example_type, data_dir, data_source_name, llm_config, skills_filter)
    scores, indices = rag.retrieve_top_k_given_query(
        "Move something to something", 10, 0
    )
    i = 0
    ins_key = "instruction"
    print("====Result====")
    for index in indices:
        print(f"{index}: {rag.data_dict[index][ins_key]}; score: {scores[i]}")
        i += 1

    # Find the closest instructions to the current task
    # Some heuristic based sampling of "candidate" states from the rollouts of these episodes
    # Add these samples to the prompt and then continue planning with this static prompt.
