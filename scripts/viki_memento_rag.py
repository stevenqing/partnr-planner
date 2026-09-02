#!/usr/bin/env python3
"""MEMENTO's user-profile memory, ported to VIKI-L2 as a baseline.

MEMENTO (Kim et al., arXiv 2505.16348) proposes a hierarchical knowledge-graph
user-profile memory that keeps personalized knowledge in separate types so that
retrieving one kind does not drown the other. Its pipeline, read off
`src/planner/user_profile_rag.py` in the authors' repository: a language model reads the
instruction and names the personalized entities it mentions, split into
`object_semantics` and `user_pattern`; each type is then retrieved separately against
sentence-transformer embeddings (all-mpnet-base-v2) of the knowledge entries; the
retrieved subgraph is rendered into natural language and handed to the planner, which
writes the plan.

What has to be said plainly about this port. MEMENTO's memory holds knowledge about a
person -- a grandmother's vase, a bedtime routine -- and VIKI-L2 has no person in it. No
user, no history, no preferences, no personal names for objects. There is nothing here
for that memory to hold, so the entries are built from the same training episodes every
other arm on this benchmark is built from, mapped onto MEMENTO's two types:

  object_semantics   what is known about a thing: where it is usually found, whether it
                     is a container, whether it starts shut
  user_pattern       what is known about a kind of task: what has to end up true, what
                     must precede what, and the shape of the work

That is MEMENTO's architecture carrying this benchmark's content. It is not MEMENTO
evaluated on its own benchmark, and no result from it should be reported as one. What it
does support is the comparison worth making: type-separated knowledge-graph retrieval
rendered into a prompt, against the same knowledge held as operators and executed. The
contribution being isolated is executability, with the type separation held constant.

Built from the same half of the training split as skill memory v2, so the two arms差
only in representation and consumption, not in what they were allowed to see.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

EMBEDDING_MODEL = "all-mpnet-base-v2"
TOP_K = 5


def _status(predicate: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (predicate.get("status") or {}).items() if v is not None}


def _flatten(node: Any) -> List[Dict[str, Any]]:
    if isinstance(node, dict):
        return [node]
    if isinstance(node, list):
        out: List[Dict[str, Any]] = []
        for item in node:
            out.extend(_flatten(item))
        return out
    return []


def build(episodes: Iterable[Dict[str, Any]], containers: set, exclude_family: Optional[str] = None,
          per_family: int = 250, per_episode: bool = True) -> Dict[str, Any]:
    """Two node types over the training episodes, and the edges between them.

    `user_pattern` nodes are written per episode rather than per family. The authors are
    explicit that episodic memory carries both the personalized knowledge and an
    in-context learning benefit, and folding a family into one averaged node throws the
    second away -- which would understate their method on a benchmark where the nearest
    episode is very close to the answer. Each node is keyed on its own instruction, so
    retrieval can find that neighbour if one exists.
    """
    where: Dict[str, Counter] = defaultdict(Counter)
    is_container: Dict[str, bool] = {}
    families: Dict[str, Dict[str, Any]] = {}
    family_objects: Dict[str, Counter] = defaultdict(Counter)
    seen = Counter()

    for truth in episodes:
        if not isinstance(truth, dict) or not truth.get("time_steps"):
            continue
        family = truth.get("task_name", "?")
        if exclude_family and family == exclude_family:
            continue
        if seen[family] >= per_family:
            continue
        seen[family] += 1
        for name, positions in (truth.get("init_pos") or {}).items():
            if positions is None or (name.startswith("R") and name[1:].isdigit()):
                continue
            kind = name.rsplit("_", 1)[0]
            is_container[kind] = kind in containers
            for item in positions:
                if isinstance(item, str):
                    where[kind][item] += 1
            family_objects[family][kind] += 1

        record = families.setdefault(family, {"goals": Counter(), "order": Counter(),
                                              "sketch": None, "steps": Counter(),
                                              "episodes": []})
        for predicate in _flatten(truth.get("goal_constraints") or []):
            for key, value in _status(predicate).items():
                record["goals"][f"{predicate['name']} must be "
                                f"{'at the ' + str(value) if key == 'pos.name' else 'used'}"] += 1
        for constraint in truth.get("temporal_constraints") or []:
            stages = [_flatten(stage) for stage in constraint]
            for earlier, later in zip(stages, stages[1:]):
                for a in earlier:
                    for b in later:
                        sa, sb = _status(a), _status(b)
                        record["order"][
                            f"{a['name']} "
                            f"{'reaches the ' + str(sa.get('pos.name')) if 'pos.name' in sa else 'is used'}"
                            f" before {b['name']} "
                            f"{'reaches the ' + str(sb.get('pos.name')) if 'pos.name' in sb else 'is used'}"
                        ] += 1
        record["steps"][len(truth["time_steps"])] += 1
        # Written in the shape the answer is asked for, rather than a compressed
        # paraphrase. Rendering a remembered plan into prose would cost this arm the
        # in-context learning its authors say episodic memory provides, and the cost
        # would be a choice of this port rather than a property of the method.
        sketch = json.dumps([
            {"step": step["step"],
             "actions": {r: a for r, a in (step.get("actions") or {}).items() if a is not None}}
            for step in truth["time_steps"]
        ])
        if record["sketch"] is None:
            record["sketch"] = sketch
        if per_episode:
            goals_here = "; ".join(
                f"{p['name']} must be "
                f"{'at the ' + str(_status(p).get('pos.name')) if 'pos.name' in _status(p) else 'used'}"
                for p in _flatten(truth.get("goal_constraints") or [])
            )
            record["episodes"].append({
                "instruction": (truth.get("description") or "").strip(),
                "goals": goals_here,
                "steps": len(truth["time_steps"]),
                "sketch": sketch,
                "family": family,
            })

    nodes: List[Dict[str, Any]] = []
    for kind, counter in sorted(where.items()):
        spots = ", ".join(name for name, _ in counter.most_common(3))
        text = f"The {kind} is usually found at: {spots}."
        if is_container.get(kind):
            text += f" The {kind} is a container that other things can be put into."
        if kind == "cabinet":
            text += " The cabinet starts shut and has to be opened before anything crosses it."
        nodes.append({"knowledge": f"{kind}: {spots}", "text": text,
                      "knowledge_type": "object_semantics", "entity": kind,
                      "support": int(sum(counter.values()))})
    if per_episode:
        for family, record in sorted(families.items()):
            for episode in record["episodes"]:
                if not episode["instruction"]:
                    continue
                text = (f"Asked: \"{episode['instruction']}\"\n  what had to end up true: "
                        f"{episode['goals']}\n  the plan that was carried out "
                        f"({episode['steps']} steps): {episode['sketch']}")
                nodes.append({"knowledge": episode["instruction"], "text": text,
                              "knowledge_type": "user_pattern", "entity": family,
                              "support": 1})
    for family, record in ([] if per_episode else sorted(families.items())):
        goals = "; ".join(name for name, _ in record["goals"].most_common(4))
        order = "; ".join(name for name, _ in record["order"].most_common(3))
        steps = record["steps"].most_common(1)[0][0] if record["steps"] else None
        text = f"Routine '{family}': {goals}."
        if order:
            text += f" Order that matters: {order}."
        if steps:
            text += f" Typically done in {steps} steps."
        if record["sketch"]:
            text += f" A past run went: {record['sketch']}"
        nodes.append({"knowledge": f"{family}: {goals}", "text": text,
                      "knowledge_type": "user_pattern", "entity": family,
                      "support": int(seen[family]),
                      "objects": [name for name, _ in family_objects[family].most_common(6)]})
    return {"nodes": nodes, "excluded_family": exclude_family,
            "counts": {t: sum(1 for n in nodes if n["knowledge_type"] == t)
                       for t in ("object_semantics", "user_pattern")}}


EXTRACTION_PROMPT = """Analyse the following instruction and extract the information it \
relies on, grouped by type.

- object_semantics: the specific objects, items and places the instruction names
- user_pattern: the routine or kind of task it describes

Output format (use empty arrays if no items for that type), and nothing else:
{{
    "object_semantics": [list of objects and places],
    "user_pattern": [list of routines or task kinds]
}}

[Example 1]
Instruction: Please put the ceramic bowl and the wooden cutting board back on the kitchen counter.
Output:
{{
    "object_semantics": ["ceramic bowl", "wooden cutting board", "kitchen counter"],
    "user_pattern": []
}}

[Example 2]
Instruction: Insert the bread into the toaster and start it, then set the plate down.
Output:
{{
    "object_semantics": ["bread", "toaster", "plate"],
    "user_pattern": ["toasting and setting a plate"]
}}

Instruction: {instruction}
Output:"""


class MementoRAG:
    """Type-separated retrieval over the ported knowledge graph."""

    def __init__(self, record: Dict[str, Any], device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.nodes = record["nodes"]
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        self.by_type: Dict[str, List[int]] = defaultdict(list)
        for index, node in enumerate(self.nodes):
            self.by_type[node["knowledge_type"]].append(index)
        self.embeddings = self.model.encode(
            [node["knowledge"] for node in self.nodes], convert_to_tensor=True,
            show_progress_bar=False, normalize_embeddings=True,
        )

    def _retrieve_for_type(self, texts: List[str], knowledge_type: str, top_k: int) -> List[int]:
        """One type at a time, which is the whole point of the module being hierarchical."""
        pool = self.by_type.get(knowledge_type, [])
        if not pool or not texts:
            return []
        from sentence_transformers import util

        query = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False,
                                  normalize_embeddings=True)
        scores = util.cos_sim(query, self.embeddings[pool])
        best: Dict[int, float] = {}
        for row in range(scores.shape[0]):
            for column in range(scores.shape[1]):
                index = pool[column]
                value = float(scores[row][column])
                if value > best.get(index, -1):
                    best[index] = value
        return [index for index, _ in sorted(best.items(), key=lambda kv: -kv[1])[:top_k]]

    def retrieve(self, extracted: Dict[str, List[str]], top_k: int = TOP_K) -> Dict[str, List[int]]:
        return {
            "object_semantics": self._retrieve_for_type(
                [t for t in extracted.get("object_semantics") or [] if isinstance(t, str)],
                "object_semantics", top_k),
            "user_pattern": self._retrieve_for_type(
                [t for t in extracted.get("user_pattern") or [] if isinstance(t, str)],
                "user_pattern", top_k),
        }

    def to_natural_language(self, retrieved: Dict[str, List[int]]) -> str:
        """The retrieved subgraph written out, kept in its two sections."""
        lines: List[str] = []
        objects = retrieved.get("object_semantics") or []
        patterns = retrieved.get("user_pattern") or []
        if objects:
            lines.append("What is known about the things involved:")
            lines += [f"- {self.nodes[i]['text']}" for i in objects]
        if patterns:
            if lines:
                lines.append("")
            lines.append("What is known about this kind of task:")
            lines += [f"- {self.nodes[i]['text']}" for i in patterns]
        return "\n".join(lines)
