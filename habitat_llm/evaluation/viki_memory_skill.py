# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_native(value.tolist())
    return value


def _parse_prompt_mapping(text: str, label: str) -> Dict[str, Any]:
    match = re.search(rf"^{re.escape(label)}: (.+)$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"VIKI prompt is missing {label!r}")
    value = ast.literal_eval(match.group(1))
    if not isinstance(value, dict):
        raise ValueError(f"VIKI prompt field {label!r} is not a mapping")
    return value


def get_prompt_context(
    sample: Dict[str, Any],
) -> Tuple[str, Dict[str, str], Dict[str, List[str]]]:
    messages = _to_native(sample["prompt"])
    system_text = next(
        message["content"] for message in messages if message["role"] == "system"
    )
    instruction = next(
        message["content"].replace("<image>", "").strip()
        for message in messages
        if message["role"] == "user"
    )
    robots = _parse_prompt_mapping(system_text, "Available robot set")
    robot_apis = _parse_prompt_mapping(system_text, "Their available operation APIs")
    available_actions = {
        robot: list(robot_apis[robot_type]) for robot, robot_type in robots.items()
    }
    return instruction, robots, available_actions


@dataclass(frozen=True)
class SkillInstance:
    train_index: int
    skill_name: str
    context: str
    demonstration: str
    required_actions: frozenset
    robot_count: int


@dataclass(frozen=True)
class RetrievedInstance:
    instance: SkillInstance
    similarity: float


@dataclass(frozen=True)
class MemoryRetrieval:
    skill_name: Optional[str]
    skill_similarity: float
    instances: Sequence[RetrievedInstance]

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "method": "memory-as-skill-individual",
            "predicted_skill": self.skill_name,
            "skill_similarity": self.skill_similarity,
            "instances": [
                {
                    "train_index": item.instance.train_index,
                    "skill_name": item.instance.skill_name,
                    "context": item.instance.context,
                    "similarity": item.similarity,
                    "required_actions": sorted(item.instance.required_actions),
                    "robot_count": item.instance.robot_count,
                }
                for item in self.instances
            ],
        }


def _format_plan(ground_truth: Dict[str, Any]) -> str:
    plan = []
    for step in ground_truth["time_steps"]:
        actions = {
            robot: action
            for robot, action in step["actions"].items()
            if action is not None
        }
        plan.append({"step": int(step["step"]), "actions": actions})
    return json.dumps(plan, separators=(",", ":"))


def _required_actions(ground_truth: Dict[str, Any]) -> frozenset:
    return frozenset(
        action[0]
        for step in ground_truth["time_steps"]
        for action in step["actions"].values()
        if action is not None
    )


def build_skill_instances(frame: pd.DataFrame) -> List[SkillInstance]:
    instances = []
    for train_index, row in frame.iterrows():
        sample = _to_native(row.to_dict())
        instruction, robots, _ = get_prompt_context(sample)
        ground_truth = _to_native(sample["reward_model"])["ground_truth"]
        instances.append(
            SkillInstance(
                train_index=int(train_index),
                skill_name=str(ground_truth["task_name"]),
                context=instruction,
                demonstration=_format_plan(ground_truth),
                required_actions=_required_actions(ground_truth),
                robot_count=len(robots),
            )
        )
    return instances


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class VikiMemorySkillLibrary:
    def __init__(
        self,
        benchmark_root: Path,
        embedding_model_name: str,
        device: str = "cpu",
        embedding_model: Optional[Any] = None,
        cache_path: Optional[Path] = None,
    ) -> None:
        train_path = benchmark_root / "data/VIKI-R/viki/VIKI-L2/train.parquet"
        frame = pd.read_parquet(train_path, columns=["prompt", "reward_model"])
        self.instances = build_skill_instances(frame)
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer

            embedding_model = SentenceTransformer(
                model_name_or_path=embedding_model_name,
                device=device,
            )
        self.embedding_model = embedding_model
        fingerprint = self._fingerprint(embedding_model_name)
        self.embeddings = self._load_cached_embeddings(cache_path, fingerprint)
        if self.embeddings is None:
            contexts = [instance.context for instance in self.instances]
            unique_contexts = list(dict.fromkeys(contexts))
            unique_embeddings = np.asarray(
                self.embedding_model.encode(
                    unique_contexts,
                    batch_size=32,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
            context_embeddings = dict(zip(unique_contexts, unique_embeddings))
            self.embeddings = np.asarray(
                [context_embeddings[context] for context in contexts]
            )
            self._write_cached_embeddings(cache_path, fingerprint)
        self.skill_indices: Dict[str, List[int]] = {}
        for index, instance in enumerate(self.instances):
            self.skill_indices.setdefault(instance.skill_name, []).append(index)
        self.skill_centroids = {
            skill_name: _normalize(self.embeddings[indices].mean(axis=0))
            for skill_name, indices in self.skill_indices.items()
        }

    def _fingerprint(self, embedding_model_name: str) -> str:
        digest = hashlib.sha256(embedding_model_name.encode("utf-8"))
        for instance in self.instances:
            digest.update(
                json.dumps(
                    {
                        "train_index": instance.train_index,
                        "skill_name": instance.skill_name,
                        "context": instance.context,
                        "demonstration": instance.demonstration,
                        "required_actions": sorted(instance.required_actions),
                        "robot_count": instance.robot_count,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        return digest.hexdigest()

    def _load_cached_embeddings(
        self, cache_path: Optional[Path], fingerprint: str
    ) -> Optional[np.ndarray]:
        if cache_path is None or not cache_path.is_file():
            return None
        with np.load(cache_path, allow_pickle=False) as cache:
            cached_fingerprint = str(cache["fingerprint"].item())
            embeddings = np.asarray(cache["embeddings"])
        if cached_fingerprint != fingerprint or len(embeddings) != len(self.instances):
            return None
        return embeddings

    def _write_cached_embeddings(
        self, cache_path: Optional[Path], fingerprint: str
    ) -> None:
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(cache_path.name + ".tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                fingerprint=np.asarray(fingerprint),
                embeddings=self.embeddings,
            )
        temporary.replace(cache_path)

    def _executable_indices(self, sample: Dict[str, Any]) -> List[int]:
        _, robots, available_actions = get_prompt_context(sample)
        allowed_actions = {
            action for actions in available_actions.values() for action in actions
        }
        return [
            index
            for index, instance in enumerate(self.instances)
            if instance.robot_count == len(robots)
            and instance.required_actions <= allowed_actions
        ]

    def executable_skill_names(self, sample: Dict[str, Any]) -> List[str]:
        return sorted(
            {
                self.instances[index].skill_name
                for index in self._executable_indices(sample)
            }
        )

    def executable_skill_descriptions(
        self, sample: Dict[str, Any], examples_per_skill: int = 2
    ) -> Dict[str, List[str]]:
        executable_indices = self._executable_indices(sample)
        descriptions = {}
        for skill_name in self.executable_skill_names(sample):
            contexts = {
                self.instances[index].context
                for index in executable_indices
                if self.instances[index].skill_name == skill_name
            }
            descriptions[skill_name] = sorted(
                contexts, key=lambda context: (len(context), context)
            )[:examples_per_skill]
        return descriptions

    def _match_skill_name(
        self,
        predicted_skill: str,
        executable_skills: Sequence[str],
    ) -> Tuple[str, float]:
        answers = re.findall(
            r"<answer>(.*?)</answer>", predicted_skill, flags=re.DOTALL
        )
        selection = answers[-1] if answers else ""
        normalized_prediction = selection.strip().lower().replace(" ", "_")
        for skill_name in executable_skills:
            if skill_name.lower() in normalized_prediction:
                return skill_name, 1.0
        prediction_embedding = np.asarray(
            self.embedding_model.encode(
                [predicted_skill],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        )
        similarities = np.asarray(
            [
                float(np.dot(prediction_embedding, self.skill_centroids[skill_name]))
                for skill_name in executable_skills
            ]
        )
        best_position = int(np.argmax(similarities))
        return executable_skills[best_position], float(similarities[best_position])

    def retrieve(
        self,
        sample: Dict[str, Any],
        top_k: int,
        similarity_threshold: float,
        predicted_skill: Optional[str] = None,
    ) -> MemoryRetrieval:
        instruction, _, _ = get_prompt_context(sample)
        executable_indices = self._executable_indices(sample)
        if not executable_indices:
            return MemoryRetrieval(None, 0.0, [])

        query = np.asarray(
            self.embedding_model.encode(
                [instruction],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        )
        executable_skills = {
            self.instances[index].skill_name for index in executable_indices
        }
        if predicted_skill is None:
            skill_name, skill_similarity = max(
                (
                    skill,
                    float(np.dot(query, self.skill_centroids[skill])),
                )
                for skill in executable_skills
            )
        else:
            skill_name, skill_similarity = self._match_skill_name(
                predicted_skill, sorted(executable_skills)
            )
        if skill_similarity < similarity_threshold:
            return MemoryRetrieval(None, skill_similarity, [])
        candidates = [
            index
            for index in executable_indices
            if self.instances[index].skill_name == skill_name
        ]
        ranked = sorted(
            (
                RetrievedInstance(
                    self.instances[index],
                    float(np.dot(query, self.embeddings[index])),
                )
                for index in candidates
            ),
            key=lambda item: (-item.similarity, item.instance.train_index),
        )
        unique_instances = []
        seen_demonstrations = set()
        for item in ranked:
            key = (item.instance.context, item.instance.demonstration)
            if key in seen_demonstrations:
                continue
            seen_demonstrations.add(key)
            unique_instances.append(item)
            if len(unique_instances) == top_k:
                break
        return MemoryRetrieval(skill_name, skill_similarity, unique_instances)


def get_skill_prediction_messages(
    sample: Dict[str, Any], skill_descriptions: Mapping[str, Sequence[str]]
) -> List[Dict[str, Any]]:
    instruction, _, _ = get_prompt_context(sample)
    candidate_blocks = []
    for skill_name, contexts in skill_descriptions.items():
        candidate_blocks.append(
            f"- {skill_name}\n  Successful contexts: {' | '.join(contexts)}"
        )
    candidates = "\n".join(candidate_blocks)
    return [
        {
            "role": "system",
            "content": (
                "Route the complete task instruction to one reusable individual skill. "
                "Choose the skill that covers the full conditional policy, including checking and satisfying every requirement. "
                "Do not infer which conditions currently hold and do not reduce the task to one next action. "
                "Reason in <think> tags, then output the exact candidate name in <answer> tags. "
                "Compare procedures, not shared words. "
                "Example valid output: <answer>ensure_all_fruits_on_table</answer>."
            ),
        },
        {
            "role": "user",
            "content": f"Current task:\n{instruction}\n\nCandidate skills:\n{candidates}",
        },
    ]


def format_memory_prompt(retrieval: MemoryRetrieval) -> str:
    if not retrieval.instances:
        return ""
    lines = [
        "Memory-as-Skill guidance from successful training trajectories:",
        f"Predicted abstract skill: {retrieval.skill_name}",
        "Executable grounded instances:",
    ]
    for number, item in enumerate(retrieval.instances, 1):
        lines.extend(
            [
                f"{number}. Context: {item.instance.context}",
                f"   Demonstration: {item.instance.demonstration}",
            ]
        )
    lines.extend(
        [
            "Use these as structural skill examples only.",
            "Ground the plan in the current image, current task entities, and current robot APIs; do not copy training entity names blindly.",
        ]
    )
    return "\n".join(lines)


def add_memory_to_messages(
    messages: List[Dict[str, Any]], memory_prompt: str
) -> List[Dict[str, Any]]:
    if not memory_prompt:
        return messages
    user_message = next(
        message for message in reversed(messages) if message["role"] == "user"
    )
    content = user_message["content"]
    if isinstance(content, list):
        text_item = next(item for item in content if item["type"] == "text")
        text_item["text"] = f"{memory_prompt}\n\nCurrent task:\n{text_item['text']}"
    else:
        user_message["content"] = f"{memory_prompt}\n\nCurrent task:\n{content}"
    return messages
