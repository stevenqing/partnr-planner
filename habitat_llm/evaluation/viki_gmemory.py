from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import runpy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def parse_rule_operations(text: str) -> List[Tuple[str, str]]:
    pattern = r"((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)"
    result = []
    for operation, rule in re.findall(pattern, text):
        rule = rule.strip()
        if (
            rule
            and not any(word in rule for word in ("ADD", "AGREE", "EDIT"))
            and rule.endswith(".")
        ):
            result.append(("ADD" if "ADD" in operation else operation.strip(), rule))
    return result


def parse_numbered_list(text: str) -> List[str]:
    pattern = r"\d+\.\s+(.*?)(?=\n\d+\.|\Z)"
    return [item.strip() for item in re.findall(pattern, text.strip(), re.DOTALL)]


def parse_relevance_score(text: str) -> int:
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 0


def update_rules(
    insights: List[Dict[str, Any]],
    relative_tasks: Sequence[str],
    operations: Sequence[Tuple[str, str]],
    local_insight_ids: Optional[Sequence[int]] = None,
) -> None:
    def existing(rule: str) -> bool:
        return any(str(item["rule"]) in rule for item in insights)

    def matching_index(rule: str) -> int:
        for index, item in enumerate(insights):
            if str(item["rule"]) in rule:
                return index
        return -1

    mapped = []
    for operation, rule in operations:
        parts = operation.split()
        if len(parts) == 2 and local_insight_ids is not None:
            local_index = int(parts[1]) - 1
            if local_index < 0 or local_index >= len(local_insight_ids):
                continue
            operation = f"{parts[0]} {local_insight_ids[local_index] + 1}"
        mapped.append((operation, rule))

    filtered = []
    for operation, rule in mapped:
        parts = operation.split()
        kind = parts[0]
        number = int(parts[1]) if len(parts) == 2 else None
        if kind == "ADD" and existing(rule):
            continue
        if kind == "EDIT" and existing(rule):
            filtered.append((f"AGREE {matching_index(rule) + 1}", rule))
            continue
        if kind in {"EDIT", "REMOVE", "AGREE"} and (
            number is None or number > len(insights) or number <= 0
        ):
            continue
        if kind in {"ADD", "EDIT", "REMOVE", "AGREE"}:
            filtered.append((operation, rule))

    list_full = len(insights) >= 10
    tasks = sorted(set(relative_tasks))
    for kind in ("REMOVE", "AGREE", "EDIT", "ADD"):
        for operation, rule in filtered:
            if operation.split()[0] != kind:
                continue
            if kind == "ADD":
                insights.append(
                    {
                        "rule": rule,
                        "score": 2,
                        "positive_correlation_tasks": tasks,
                        "negative_correlation_tasks": [],
                    }
                )
                continue
            index = (
                matching_index(rule)
                if kind == "AGREE"
                else int(operation.split()[1]) - 1
            )
            item = insights[index]
            if kind == "REMOVE":
                item["score"] -= 3 if list_full else 1
                item["negative_correlation_tasks"] = sorted(
                    set(item["negative_correlation_tasks"] + tasks)
                )
            elif kind == "AGREE":
                item["score"] += 1
                item["positive_correlation_tasks"] = sorted(
                    set(item["positive_correlation_tasks"] + tasks)
                )
            else:
                item["rule"] = rule
                item["score"] += 1
                item["positive_correlation_tasks"] = sorted(
                    set(item["positive_correlation_tasks"] + tasks)
                )
    insights[:] = [item for item in insights if item["score"] > 0]


def reward_retrieved_insights(
    insights: List[Dict[str, Any]], retrieved_rules: Sequence[str], success: bool
) -> None:
    reward = 1 if success else -2
    for retrieved_rule in retrieved_rules:
        for insight in insights:
            if retrieved_rule in str(insight["rule"]):
                insight["score"] += reward
    insights[:] = [insight for insight in insights if insight["score"] > 0]


def load_author_prompts(gmemory_root: Path) -> Dict[str, str]:
    prompt_path = gmemory_root / "mas/memory/mas_memory/prompt.py"
    format_path = gmemory_root / "tasks/mas_workflow/format.py"
    values = runpy.run_path(str(prompt_path))
    names = (
        "generative_task_system_prompt",
        "generative_task_user_prompt",
        "extract_true_traj_system_prompt",
        "extract_true_traj_user_prompt",
        "critique_compare_rules_system_prompt",
        "critique_compare_rules_user_prompt",
        "critique_success_rules_system_prompt",
        "critique_success_rules_user_prompt",
        "detect_mistakes_system_prompt",
        "detect_mistakes_user_prompt",
        "merge_rules_system_prompt",
        "merge_rules_user_prompt",
    )
    prompts = {name: str(values[name]) for name in names}
    format_values = runpy.run_path(str(format_path))
    prompts["task_solve_with_insights"] = str(format_values["task_solve_with_insights"])
    prompts["task_format"] = str(format_values["task_format"])
    return prompts


def render_task_context(record: Mapping[str, Any]) -> str:
    return (
        "### Task description:\n"
        f"{record['task_description']}\n\n"
        "### Key steps:\n"
        f"{record['key_steps']}\n\n"
        "### Detailed trajectory:\n"
        f"{record['trajectory']}"
    )


def render_retrieval_prompt(
    successful_records: Sequence[Mapping[str, Any]], insights: Sequence[str]
) -> str:
    memory_text = "\n\n".join(
        f"Task {index + 1}:\n{render_task_context(record)}"
        for index, record in enumerate(successful_records)
    )
    insight_text = "\n".join(
        f"{index + 1}. {insight}" for index, insight in enumerate(insights)
    )
    return (
        "## Your Own Past Successes (Execution Patterns)\n"
        "Here are successful execution processes from similar completed tasks.\n"
        f"{memory_text}\n---\n\n"
        "## Key Insights from Related Tasks\n"
        "The following general insights were distilled from related executions.\n"
        f"{insight_text}\n---\n\n"
        "Use these memories only as reference. Produce one complete plan for the current "
        "image and instruction using only the activated robot APIs."
    )


def finch_labels(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=np.int64)
    from finch import FINCH

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            labels = FINCH(metric="cosine").fit_predict(embeddings)
        return np.asarray(labels, dtype=np.int64)
    except (UnboundLocalError, ValueError):
        return np.zeros(len(embeddings), dtype=np.int64)


@dataclass
class GMemoryState:
    records: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    graph: Dict[str, Dict[str, float]] = field(default_factory=dict)
    insights: List[Dict[str, Any]] = field(default_factory=list)
    query_edge_threshold: float = 0.7
    hop: int = 1

    @classmethod
    def from_canonical_state(
        cls, value: Mapping[str, Any], embeddings: Sequence[np.ndarray]
    ) -> "GMemoryState":
        records = [dict(record) for record in value["records"]]
        # add_record stores unit vectors, so canonical_state hashes normalized
        # embeddings. Restore from the same representation, otherwise the cached
        # raw MiniLM rows hash differently wherever the norm is not exactly 1.
        vectors = []
        for vector in embeddings:
            candidate = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(candidate))
            if norm == 0:
                raise ValueError("G-Memory embedding cannot be zero")
            vectors.append(candidate / norm)
        if len(records) != len(vectors):
            raise ValueError("Canonical G-Memory records and embeddings differ")
        observed_hashes = [
            hashlib.sha256(vector.astype("<f4").tobytes()).hexdigest()
            for vector in vectors
        ]
        if observed_hashes != list(value["embedding_hashes"]):
            raise ValueError("Canonical G-Memory embedding hashes differ")
        graph: Dict[str, Dict[str, float]] = {
            str(node): {} for node in value["query_nodes"]
        }
        for left, right, weight in value["query_edges"]:
            graph[str(left)][str(right)] = float(weight)
            graph[str(right)][str(left)] = float(weight)
        state = cls(
            records=records,
            embeddings=vectors,
            graph=graph,
            insights=[dict(insight) for insight in value["insights"]],
            query_edge_threshold=float(value["query_edge_threshold"]),
            hop=int(value["hop"]),
        )
        if state.canonical_state() != dict(value):
            raise ValueError("Canonical G-Memory restoration changed state")
        return state

    def add_record(self, record: Mapping[str, Any], embedding: np.ndarray) -> None:
        item = dict(record)
        vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("G-Memory embedding cannot be zero")
        vector = vector / norm
        task_main = str(item["task_main"])
        if task_main not in self.graph:
            self.graph[task_main] = {}
            if self.records:
                ranking = self._rank_record_indices(vector)[:10]
                for index in ranking:
                    neighbor = str(self.records[index]["task_main"])
                    similarity = float(self.embeddings[index] @ vector)
                    if similarity < self.query_edge_threshold:
                        continue
                    self.graph.setdefault(neighbor, {})[task_main] = similarity
                    self.graph[task_main][neighbor] = similarity
        self.records.append(item)
        self.embeddings.append(vector)

    def _rank_record_indices(
        self, query_embedding: np.ndarray, label: Optional[bool] = None
    ) -> List[int]:
        query = np.asarray(query_embedding, dtype=np.float32)
        query = query / np.linalg.norm(query)
        return sorted(
            [
                index
                for index, record in enumerate(self.records)
                if label is None or bool(record["label"]) is label
            ],
            key=lambda index: (
                -float(self.embeddings[index] @ query),
                str(self.records[index]["memory_id"]),
            ),
        )

    def _expand_nodes(self, start: str) -> List[str]:
        seen = {start}
        frontier = {start}
        for _ in range(self.hop):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier.update(self.graph.get(node, {}))
            next_frontier -= seen
            seen.update(next_frontier)
            frontier = next_frontier
        return sorted(seen)

    def raw_success_candidates(
        self, query_embedding: np.ndarray, count: int
    ) -> List[int]:
        if not self.records or count <= 0:
            return []
        nearest = self._rank_record_indices(query_embedding)[:1]
        related_nodes = set()
        for index in nearest:
            related_nodes.update(
                self._expand_nodes(str(self.records[index]["task_main"]))
            )
        candidates = []
        for node in sorted(related_nodes):
            node_indices = [
                index
                for index, record in enumerate(self.records)
                if str(record["task_main"]) == node
            ]
            if not node_indices:
                continue
            candidates.append(
                min(
                    node_indices,
                    key=lambda index: str(self.records[index]["memory_id"]),
                )
            )
        candidates = [index for index in candidates if self.records[index]["label"]]
        if len(candidates) < count:
            candidates = self._rank_record_indices(query_embedding, label=True)[:count]
        query = np.asarray(query_embedding, dtype=np.float32)
        query = query / np.linalg.norm(query)
        return sorted(
            set(candidates),
            key=lambda index: (
                -float(self.embeddings[index] @ query),
                str(self.records[index]["memory_id"]),
            ),
        )[:count]

    def nearest_record_indices(
        self, query_embedding: np.ndarray, count: int, label: Optional[bool] = None
    ) -> List[int]:
        return self._rank_record_indices(query_embedding, label=label)[:count]

    def related_insights(self, query_embedding: np.ndarray, count: int) -> List[str]:
        success = self._rank_record_indices(query_embedding, label=True)[:4]
        failed = self._rank_record_indices(query_embedding, label=False)[:2]
        tasks = [str(self.records[index]["task_main"]) for index in success + failed]
        scored = []
        for index, insight in enumerate(self.insights):
            score = sum(
                task in insight.get("positive_correlation_tasks", []) for task in tasks
            )
            if score >= 1:
                scored.append((str(insight["rule"]), score, index))
        return [
            rule
            for rule, _, _ in sorted(scored, key=lambda item: (-item[1], item[2]))[
                :count
            ]
        ]

    def related_insight_ids(
        self, tasks: Iterable[str], threshold: float = 1
    ) -> List[int]:
        task_list = list(tasks)
        scored = []
        for index, insight in enumerate(self.insights):
            score = sum(
                task in insight.get("positive_correlation_tasks", [])
                for task in task_list
            )
            if score >= threshold:
                scored.append((index, score))
        return [index for index, _ in sorted(scored, key=lambda item: -item[1])]

    def cluster_tasks(self) -> Dict[int, List[str]]:
        nodes = list(self.graph)
        first_record = {
            node: next(
                index
                for index, record in enumerate(self.records)
                if str(record["task_main"]) == node
            )
            for node in nodes
        }
        matrix = np.vstack([self.embeddings[first_record[node]] for node in nodes])
        labels = finch_labels(matrix)
        clusters: Dict[int, List[str]] = {}
        for node, label in zip(nodes, labels.tolist()):
            clusters.setdefault(int(label), []).append(node)
        return clusters

    def canonical_state(self) -> Dict[str, Any]:
        edges = sorted(
            [
                [left, right, self.graph[left][right]]
                for left in self.graph
                for right in self.graph[left]
                if left < right
            ]
        )
        embedding_hashes = [
            hashlib.sha256(vector.astype("<f4").tobytes()).hexdigest()
            for vector in self.embeddings
        ]
        return {
            "records": self.records,
            "embedding_hashes": embedding_hashes,
            "query_nodes": list(self.graph),
            "query_edges": edges,
            "insights": self.insights,
            "query_edge_threshold": self.query_edge_threshold,
            "hop": self.hop,
        }

    def hierarchy_sha256(self) -> str:
        return sha256_text(canonical_json(self.canonical_state()))
