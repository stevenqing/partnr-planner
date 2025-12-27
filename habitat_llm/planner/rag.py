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

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


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
    ):
        self._device = "cuda"
        self._llm_config = llm_config
        self._example_type = example_type
        self._skills_filter = skills_filter

        # Memory-specific parameters (MEMENTO integration)
        self.scene_id = scene_id
        self.memory_path = memory_path
        self.ensure_same_scene = ensure_same_scene

        # Determine the start header index
        if example_type == "react" or example_type == "zero_shot":
            self.start_header_idx = 1
        elif example_type == "summary":
            # In summary based, the initial prompt is not included
            # in the trace file
            self.start_header_idx = 0
        elif example_type == "skills":
            self.start_header_idx = 0

        self.data_dict = {}
        self.index = 0
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

    def retrieve_top_k_given_query(self, query: str, top_k: int = 1, agent_id: int = 0):
        """Return the top k text/index of the examples given query and agent id."""

        assert query != "", "query text is an empty string"
        assert (
            len(self.data_dict) >= top_k
        ), "top_k value exceeds the size of the RAG examples"
        assert agent_id in [0, 1], "Do not support agent_id other than 0 and 1"

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
