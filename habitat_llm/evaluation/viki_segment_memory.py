import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from habitat_llm.evaluation.viki_branch_conditions import get_instruction, initial_state

SUBGOAL_SYSTEM_PROMPT = """Predict the ordered individual-skill subgoals needed to solve the task.
Use short snake_case skill phrases. Do not output robot assignments or low-level actions.
Return only a JSON array of strings in execution order."""


def _fingerprint(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_subgoal_messages(sample: Mapping[str, Any]) -> List[Dict[str, str]]:
    context = {
        "instruction": get_instruction(sample),
        "initial_state": initial_state(sample),
    }
    return [
        {"role": "system", "content": SUBGOAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                context, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ),
        },
    ]


def build_retrieval_context(sample: Mapping[str, Any]) -> str:
    state = json.dumps(initial_state(sample), sort_keys=True, separators=(",", ":"))
    return f"Instruction: {get_instruction(sample)}\nInitial state: {state}"


def parse_subgoal_prediction(text: str) -> List[str]:
    value = text.strip()
    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", value, flags=re.DOTALL)
    if answer is not None:
        value = answer.group(1).strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL)
    if fence is not None:
        value = fence.group(1).strip()
    parsed = json.loads(value)
    if isinstance(parsed, dict) and set(parsed) == {"subgoals"}:
        parsed = parsed["subgoals"]
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("Subgoal prediction must be a non-empty JSON array")
    subgoals = []
    for item in parsed:
        if not isinstance(item, str) or not _normalize_name(item):
            raise ValueError("Every predicted subgoal must be a non-empty string")
        subgoals.append(_normalize_name(item))
    return subgoals


@dataclass(frozen=True)
class SegmentInstance:
    instance_id: str
    source_train_index: int
    skill_name: str
    description: str
    context: str
    demo: Sequence[Mapping[str, Any]]
    self_cond: Mapping[str, Any]
    ordered_units: Sequence[str]
    unit_kinds: Sequence[str]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SegmentInstance":
        return cls(
            instance_id=str(value["instance_id"]),
            source_train_index=int(value["source_train_index"]),
            skill_name=str(value["skill_name"]),
            description=str(value["description"]),
            context=str(value["context"]),
            demo=value["demo"],
            self_cond=value["self_cond"],
            ordered_units=value.get("ordered_units", []),
            unit_kinds=value.get("unit_kinds", []),
        )


@dataclass(frozen=True)
class RetrievedSegment:
    instance: SegmentInstance
    context_similarity: float


@dataclass(frozen=True)
class RetrievedGroup:
    predicted_subgoal: str
    skill_name: Optional[str]
    skill_similarity: float
    instances: Sequence[RetrievedSegment]
    dropped_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_subgoal": self.predicted_subgoal,
            "skill_name": self.skill_name,
            "skill_similarity": self.skill_similarity,
            "dropped_reason": self.dropped_reason,
            "instances": [
                {
                    "instance_id": item.instance.instance_id,
                    "source_train_index": item.instance.source_train_index,
                    "skill_name": item.instance.skill_name,
                    "context_similarity": item.context_similarity,
                }
                for item in self.instances
            ],
        }


class SegmentMemoryBank:
    def __init__(
        self,
        instances_path: Path,
        skills_path: Path,
        embedding_model: str = "all-mpnet-base-v2",
        embedding_device: str = "cpu",
        cache_path: Optional[Path] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.instances = [
            SegmentInstance.from_dict(json.loads(line))
            for line in instances_path.read_text().splitlines()
            if line.strip()
        ]
        skills = json.loads(skills_path.read_text())["skills"]
        self.skill_names = [str(skill["name"]) for skill in skills]
        self.skill_descriptions = {
            str(skill["name"]): str(skill["description"]) for skill in skills
        }
        if not self.instances or not self.skill_names:
            raise ValueError("M0 bank is empty")
        self.instances_by_skill: Dict[str, List[int]] = {
            name: [] for name in self.skill_names
        }
        for index, instance in enumerate(self.instances):
            self.instances_by_skill[instance.skill_name].append(index)
        model = SentenceTransformer(embedding_model, device=embedding_device)
        cache_key = _fingerprint(
            self.skill_names + [instance.instance_id for instance in self.instances]
        )
        loaded = False
        if cache_path is not None and cache_path.is_file():
            cached = np.load(cache_path, allow_pickle=False)
            if str(cached["cache_key"].item()) == cache_key:
                self.skill_embeddings = cached["skill_embeddings"]
                self.context_embeddings = cached["context_embeddings"]
                loaded = True
        if not loaded:
            self.skill_embeddings = np.asarray(
                model.encode(
                    self.skill_names,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
            unique_contexts = list(
                dict.fromkeys(instance.context for instance in self.instances)
            )
            unique_embeddings = np.asarray(
                model.encode(
                    unique_contexts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
            context_index = {
                context: index for index, context in enumerate(unique_contexts)
            }
            self.context_embeddings = np.asarray(
                [
                    unique_embeddings[context_index[instance.context]]
                    for instance in self.instances
                ]
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    cache_key=np.asarray(cache_key),
                    skill_embeddings=self.skill_embeddings,
                    context_embeddings=self.context_embeddings,
                )
        self.embedding_model = model
        self.embedding_lock = Lock()

    def retrieve(
        self,
        predicted_subgoals: Sequence[str],
        query_context: str,
        threshold: float = 0.3,
        per_skill: int = 2,
        total: int = 6,
        allowed_source_indices: Optional[Sequence[int]] = None,
        allowed_instance_ids: Optional[Sequence[str]] = None,
        excluded_source_indices: Optional[Sequence[int]] = None,
    ) -> List[RetrievedGroup]:
        if per_skill < 1 or total < 1:
            raise ValueError("Retrieval limits must be positive")
        normalized = [_normalize_name(value) for value in predicted_subgoals]
        with self.embedding_lock:
            query_embeddings = np.asarray(
                self.embedding_model.encode(
                    normalized,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
            context_embedding = np.asarray(
                self.embedding_model.encode(
                    [query_context],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]
            )
        allowed = (
            None if allowed_source_indices is None else set(allowed_source_indices)
        )
        allowed_instances = (
            None if allowed_instance_ids is None else set(allowed_instance_ids)
        )
        excluded = set(excluded_source_indices or [])
        groups = []
        selected = 0
        seen_instances = set()
        for subgoal, embedding in zip(normalized, query_embeddings):
            similarities = self.skill_embeddings @ embedding
            skill_index = int(np.argmax(similarities))
            skill_name = self.skill_names[skill_index]
            skill_similarity = float(similarities[skill_index])
            if skill_similarity < threshold:
                groups.append(
                    RetrievedGroup(
                        subgoal,
                        skill_name,
                        skill_similarity,
                        [],
                        "skill_below_threshold",
                    )
                )
                continue
            candidates = []
            for instance_index in self.instances_by_skill[skill_name]:
                instance = self.instances[instance_index]
                if allowed is not None and instance.source_train_index not in allowed:
                    continue
                if (
                    allowed_instances is not None
                    and instance.instance_id not in allowed_instances
                ):
                    continue
                if instance.source_train_index in excluded:
                    continue
                similarity = float(
                    np.dot(context_embedding, self.context_embeddings[instance_index])
                )
                if (
                    similarity >= threshold
                    and instance.instance_id not in seen_instances
                ):
                    candidates.append((similarity, instance.instance_id, instance))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            available = max(0, total - selected)
            retrieved = [
                RetrievedSegment(instance, similarity)
                for similarity, _, instance in candidates[: min(per_skill, available)]
            ]
            if not retrieved:
                reason = "total_limit" if available == 0 else "context_below_threshold"
                groups.append(
                    RetrievedGroup(subgoal, skill_name, skill_similarity, [], reason)
                )
                continue
            for item in retrieved:
                seen_instances.add(item.instance.instance_id)
            selected += len(retrieved)
            groups.append(
                RetrievedGroup(subgoal, skill_name, skill_similarity, retrieved, None)
            )
        return groups


def format_grouped_memory(groups: Sequence[RetrievedGroup]) -> str:
    active = [group for group in groups if group.instances]
    if not active:
        return ""
    lines = [
        "Grounded skill memory, grouped in predicted subgoal order:",
    ]
    for group_index, group in enumerate(active, 1):
        lines.append(
            f"Subgoal {group_index}: {group.predicted_subgoal} "
            f"(matched skill: {group.skill_name})"
        )
        for instance_index, retrieved in enumerate(group.instances, 1):
            instance = retrieved.instance
            lines.extend(
                [
                    f"  Instance {instance_index} context: {instance.context}",
                    "  Instance availability: "
                    + json.dumps(instance.self_cond, sort_keys=True, ensure_ascii=True),
                    "  Instance demonstration: "
                    + json.dumps(
                        instance.demo, separators=(",", ":"), ensure_ascii=True
                    ),
                ]
            )
    lines.extend(
        [
            "Use the examples only as grounded sub-plan guidance.",
            "Produce one complete plan for the current task using only its available robot APIs.",
        ]
    )
    return "\n".join(lines)
