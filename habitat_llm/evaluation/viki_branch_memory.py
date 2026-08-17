# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from habitat_llm.evaluation.viki_branch_conditions import AvailabilityPredicate
from habitat_llm.evaluation.viki_memory_skill import (
    MemoryRetrieval,
    RetrievedInstance,
    VikiMemorySkillLibrary,
    format_memory_prompt,
    get_prompt_context,
)

ABSTRACT_SKILL_DESCRIPTIONS = {
    "clear_table_with_two_robots_and_put_in_cabinet": (
        "Move two portable items into a closed storage destination after making "
        "the destination accessible."
    ),
    "cut_fruit_on_board": (
        "Move one portable item to a work surface, acquire the required tool, "
        "then apply the requested state-changing operation."
    ),
    "cut_two_fruits_on_board": (
        "Move two portable items to a shared work surface, acquire the required "
        "tool, then apply the requested operation to both items."
    ),
    "dog_check_environment": (
        "Navigate to the referenced region and inspect its surroundings."
    ),
    "dog_push_box_for_two_panda_transport": (
        "Use a mobile carrier when direct transport is unsuitable, then complete "
        "delivery to the requested destination."
    ),
    "ensure_all_fruits_on_table": (
        "Inspect the required destination, identify every unmet item condition, "
        "and satisfy each one."
    ),
    "parallel_human_dual_asset_to_plate_or_bowl": (
        "Assign two agents to relocate two portable items concurrently to one "
        "destination."
    ),
    "sequential_pick_two_and_place": (
        "Acquire two portable items sequentially, then deliver both to one "
        "destination."
    ),
    "serve_bread_after_checking_cabinet": (
        "Make a possible storage source accessible before retrieving and "
        "delivering the requested item."
    ),
    "serve_bread_from_counter": (
        "Retrieve the requested item from its available source and deliver it to "
        "the requested destination."
    ),
    "set_plate_and_fork_on_table": (
        "In parallel, relocate a destination receptacle and retrieve another item "
        "from storage, then combine them."
    ),
    "single_move_asset_to_target": (
        "Retrieve one portable item and relocate it to the requested destination."
    ),
    "toast_bread_and_set_plate": (
        "In parallel, load one item into an appliance, activate it, and relocate "
        "another item."
    ),
    "wash_fruit_and_serve": (
        "Move one portable item through an intermediate state-changing station, "
        "apply the operation, then deliver it to the final destination."
    ),
}


@dataclass(frozen=True)
class BranchRetrieval:
    retrieval: MemoryRetrieval
    current_branch: str
    absent_assets: Sequence[str]
    tier: str
    similarity_bar: float

    def to_metadata(self) -> Dict[str, Any]:
        return {
            **self.retrieval.to_metadata(),
            "current_branch": self.current_branch,
            "current_absent_assets": list(self.absent_assets),
            "injection_tier": self.tier,
            "train_calibrated_similarity_bar": self.similarity_bar,
        }


def validate_abstract_descriptions(asset_vocabulary: Sequence[str]) -> None:
    for skill_name, description in ABSTRACT_SKILL_DESCRIPTIONS.items():
        for asset in asset_vocabulary:
            if re.search(
                rf"(?<![a-z0-9]){re.escape(asset.lower())}(?![a-z0-9])",
                description.lower(),
            ):
                raise ValueError(
                    f"Abstract description {skill_name!r} contains asset {asset!r}"
                )


def format_abstract_prompt(skill_name: str, description: str) -> str:
    return (
        "Memory-as-Skill abstract guidance:\n"
        f"Reusable procedure: {description}\n"
        "Apply only this procedural scaffold. Determine all concrete entities, "
        "conditions, and actions from the current task, image, and robot APIs."
    )


class BranchIndexedMemory:
    def __init__(
        self,
        library: VikiMemorySkillLibrary,
        predicate: AvailabilityPredicate,
        instance_branches: Mapping[int, str],
        asset_vocabulary: Sequence[str],
        calibration_quantile: float = 0.10,
    ) -> None:
        self.library = library
        self.predicate = predicate
        self.instance_branches = dict(instance_branches)
        self.asset_vocabulary = sorted(asset_vocabulary)
        self.query_embedding_cache: Dict[str, np.ndarray] = {}
        self.executable_indices_cache: Dict[tuple, List[int]] = {}
        self.skill_match_cache: Dict[tuple, tuple] = {}
        validate_abstract_descriptions(self.asset_vocabulary)
        if set(self.instance_branches) != {
            instance.train_index for instance in self.library.instances
        }:
            raise ValueError("Branch labels do not cover the frozen train memory")
        self.calibration_quantile = calibration_quantile
        self.similarity_bar = self._calibrate_similarity_bar()

    def _calibrate_similarity_bar(self) -> float:
        nearest = []
        for _skill_name, indices in self.library.skill_indices.items():
            by_branch: Dict[str, List[int]] = {}
            for index in indices:
                train_index = self.library.instances[index].train_index
                branch = self.instance_branches[train_index]
                by_branch.setdefault(branch, []).append(index)
            for branch_indices in by_branch.values():
                if len(branch_indices) < 2:
                    continue
                embeddings = self.library.embeddings[branch_indices]
                similarities = embeddings @ embeddings.T
                for position, library_index in enumerate(branch_indices):
                    context = self.library.instances[library_index].context
                    valid_positions = [
                        candidate_position
                        for candidate_position, candidate_index in enumerate(
                            branch_indices
                        )
                        if candidate_index != library_index
                        and self.library.instances[candidate_index].context != context
                    ]
                    if valid_positions:
                        nearest.append(
                            float(similarities[position, valid_positions].max())
                        )
        if not nearest:
            raise ValueError("No train-only neighbors available for calibration")
        return float(np.quantile(np.asarray(nearest), self.calibration_quantile))

    def retrieve(
        self,
        sample: Dict[str, Any],
        top_k: int,
        predicted_skill: str,
        branch_indexing: bool = True,
        graded_injection: bool = True,
        allowed_train_indices: Optional[Sequence[int]] = None,
    ) -> BranchRetrieval:
        branch = self.predicate.evaluate(sample)
        instruction, robots, available_actions = get_prompt_context(sample)
        capability_key = (
            tuple(sorted(robots.items())),
            tuple(
                (robot, tuple(actions))
                for robot, actions in sorted(available_actions.items())
            ),
        )
        executable_indices = self.executable_indices_cache.get(capability_key)
        if executable_indices is None:
            executable_indices = self.library._executable_indices(sample)
            self.executable_indices_cache[capability_key] = executable_indices
        if allowed_train_indices is not None:
            allowed = set(int(index) for index in allowed_train_indices)
            executable_indices = [
                index
                for index in executable_indices
                if self.library.instances[index].train_index in allowed
            ]
        executable_skills = sorted(
            {self.library.instances[index].skill_name for index in executable_indices}
        )
        if not executable_skills:
            empty = MemoryRetrieval(None, 0.0, [])
            return BranchRetrieval(
                empty, branch.branch, branch.absent_assets, "none", self.similarity_bar
            )
        match_key = (predicted_skill, tuple(executable_skills))
        matched = self.skill_match_cache.get(match_key)
        if matched is None:
            matched = self.library._match_skill_name(predicted_skill, executable_skills)
            self.skill_match_cache[match_key] = matched
        skill_name, skill_similarity = matched
        query = self.query_embedding_cache.get(instruction)
        if query is None:
            query = np.asarray(
                self.library.embedding_model.encode(
                    [instruction],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]
            )
            self.query_embedding_cache[instruction] = query
        candidates = [
            index
            for index in executable_indices
            if self.library.instances[index].skill_name == skill_name
            and (
                not branch_indexing
                or self.instance_branches[self.library.instances[index].train_index]
                == branch.branch
            )
        ]
        ranked = sorted(
            (
                RetrievedInstance(
                    self.library.instances[index],
                    float(np.dot(query, self.library.embeddings[index])),
                )
                for index in candidates
            ),
            key=lambda item: (-item.similarity, item.instance.train_index),
        )
        instances = []
        seen = set()
        for item in ranked:
            key = (item.instance.context, item.instance.demonstration)
            if key in seen:
                continue
            seen.add(key)
            instances.append(item)
            if len(instances) == top_k:
                break
        retrieval = MemoryRetrieval(skill_name, skill_similarity, instances)
        if not graded_injection:
            tier = "grounded" if instances else "abstract"
        elif instances and instances[0].similarity >= self.similarity_bar:
            tier = "grounded"
        else:
            tier = "abstract"
            retrieval = MemoryRetrieval(skill_name, skill_similarity, [])
        return BranchRetrieval(
            retrieval,
            branch.branch,
            branch.absent_assets,
            tier,
            self.similarity_bar,
        )

    def format_prompt(self, result: BranchRetrieval) -> str:
        if result.tier == "grounded":
            return format_memory_prompt(result.retrieval)
        if result.tier == "abstract" and result.retrieval.skill_name is not None:
            skill_name = result.retrieval.skill_name
            return format_abstract_prompt(
                skill_name, ABSTRACT_SKILL_DESCRIPTIONS[skill_name]
            )
        return ""
