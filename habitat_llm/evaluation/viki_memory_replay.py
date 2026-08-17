# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set

import pandas as pd

from habitat_llm.evaluation.viki_memory_skill import (
    MemoryRetrieval,
    RetrievedInstance,
    SkillInstance,
    _to_native,
    build_skill_instances,
)

FILTERED_ACTIONS = {
    "Move",
    "Reach",
    "Grasp",
    "Place",
    "Open",
    "Close",
    "Interact",
}


def _load_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    records = {}
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record["index"])
            if index in records:
                raise ValueError(f"Duplicate index {index} in {path}")
            records[index] = record
    return records


def _asset_type(entity_id: str) -> str:
    match = re.fullmatch(r"(.+)_([0-9]+)", entity_id)
    if match is None:
        raise ValueError(f"Unexpected VIKI asset instance id {entity_id!r}")
    return match.group(1)


def _initial_state_vocabulary(
    ground_truth: Mapping[str, Any],
) -> tuple[Set[str], Set[str]]:
    asset_types = set()
    locations: Set[str] = set()
    for entity_id, positions in ground_truth["init_pos"].items():
        if entity_id.startswith("R") and entity_id[1:].isdigit():
            continue
        if positions is None:
            continue
        asset_types.add(_asset_type(entity_id))
        locations.update(str(position) for position in _to_native(positions))
    return asset_types, locations


def _instance_targets(instance: SkillInstance) -> Set[str]:
    plan = json.loads(instance.demonstration)
    return {
        str(target)
        for step in plan
        for action in step["actions"].values()
        if action is not None and action[0] in FILTERED_ACTIONS
        for target in action[1:]
    }


@dataclass(frozen=True)
class ReplayFilterResult:
    retrieval: MemoryRetrieval
    original_demo_ids: Sequence[int]
    dropped_demo_ids: Sequence[int]
    dropped_targets: Mapping[int, Sequence[str]]

    def to_metadata(self) -> Dict[str, Any]:
        kept_ids = [item.instance.train_index for item in self.retrieval.instances]
        return {
            **self.retrieval.to_metadata(),
            "replay_original_demo_ids": list(self.original_demo_ids),
            "replay_kept_demo_ids": kept_ids,
            "replay_dropped_demo_ids": list(self.dropped_demo_ids),
            "replay_dropped_targets": {
                str(index): list(targets)
                for index, targets in self.dropped_targets.items()
            },
            "object_filter_fallback": not kept_ids,
        }


class FrozenOODMemoryReplay:
    def __init__(
        self,
        benchmark_root: Path,
        v0_table: Path,
        frozen_memory_log: Path,
    ) -> None:
        train_path = benchmark_root / "data/VIKI-R/viki/VIKI-L2/train.parquet"
        train_frame = pd.read_parquet(train_path, columns=["prompt", "reward_model"])
        self.instances = {
            instance.train_index: instance
            for instance in build_skill_instances(train_frame)
        }
        self.global_asset_types = set()
        self.global_locations = set()
        for reward_model in train_frame["reward_model"]:
            ground_truth = _to_native(reward_model)["ground_truth"]
            assets, locations = _initial_state_vocabulary(ground_truth)
            self.global_asset_types.update(assets)
            self.global_locations.update(locations)

        self.v0 = pd.read_parquet(v0_table).set_index("index")
        self.memory_records = _load_jsonl(frozen_memory_log)
        if set(self.v0.index) != set(self.memory_records):
            raise ValueError("V0 table and frozen memory log indices differ")

    def _frozen_retrieval(self, index: int) -> MemoryRetrieval:
        row = self.v0.loc[index]
        metadata = self.memory_records[index]["provider_metadata"]
        metadata_by_id = {
            int(item["train_index"]): item for item in metadata["instances"]
        }
        logged_ids = [int(value) for value in row["injected_demo_ids"]]
        source_ids = [int(item["train_index"]) for item in metadata["instances"]]
        if logged_ids != source_ids:
            raise ValueError(
                f"V0 and frozen memory demo ids differ at index {index}: "
                f"{logged_ids} != {source_ids}"
            )
        instances = [
            RetrievedInstance(
                self.instances[train_index],
                float(metadata_by_id[train_index]["similarity"]),
            )
            for train_index in logged_ids
        ]
        return MemoryRetrieval(
            str(row["routed_skill"]),
            float(metadata["skill_similarity"]),
            instances,
        )

    def retrieve(
        self,
        index: int,
        sample: Dict[str, Any],
        filter_objects: bool,
    ) -> ReplayFilterResult:
        frozen = self._frozen_retrieval(index)
        original_ids = [item.instance.train_index for item in frozen.instances]
        if not filter_objects:
            return ReplayFilterResult(frozen, original_ids, [], {})

        ground_truth = _to_native(sample["reward_model"])["ground_truth"]
        current_assets, current_locations = _initial_state_vocabulary(ground_truth)
        allowed_locations = self.global_locations | current_locations
        kept: List[RetrievedInstance] = []
        dropped_ids = []
        dropped_targets = {}
        for item in frozen.instances:
            unavailable = sorted(
                target
                for target in _instance_targets(item.instance)
                if (target in self.global_asset_types and target not in current_assets)
                or (
                    target not in self.global_asset_types
                    and target not in allowed_locations
                )
            )
            if unavailable:
                dropped_ids.append(item.instance.train_index)
                dropped_targets[item.instance.train_index] = unavailable
            else:
                kept.append(item)
        retrieval = MemoryRetrieval(
            frozen.skill_name,
            frozen.skill_similarity,
            kept,
        )
        return ReplayFilterResult(
            retrieval,
            original_ids,
            dropped_ids,
            dropped_targets,
        )
