#!/usr/bin/env python3
"""G-Memory ported from VIKI-L2 to PARTNR.

`GMemoryState` and the authors' own prompts are reused from
`habitat_llm/evaluation/viki_gmemory.py` -- the state machine is generic over records and
embeddings, so nothing about the method is reimplemented here. What is written here is
the PARTNR side: the rollouts turned into G-Memory records, and the periodic insight
distillation run over them with the same hyperparameters the VIKI-L2 arm used
(`START_INSIGHTS_THRESHOLD=5`, `ROUNDS_PER_INSIGHTS=5`, `INSIGHTS_POINT_NUM=5`,
`SUCCESSFUL_TOPK=1`, `INSIGHTS_TOPK=3`, `hop=1`, `use_projector=False`).

Built from the same rearrange-only half of `train_mini` that skill memory v2 induces its
operators from, so the two arms differ in representation and consumption, not in what
they were allowed to see. That restriction is the point: an unrestricted pool has seen
spatial and temporal work the operators never did, and comparing against it answers
"is retrieval enough" rather than "does this memory compose".

The VIKI-L2 arm carried a freeze/ledger protocol for reproducibility of a headline
number; this one does not, because it is a baseline inside a sweep rather than a
published figure. The retrieval path, prompts and hyperparameters are the same.
"""

from __future__ import annotations

import base64
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from habitat_llm.evaluation.viki_gmemory import (  # noqa: E402
    GMemoryState,
    load_author_prompts,
    parse_rule_operations,
    render_retrieval_prompt,
    update_rules,
)

from . import substrate  # noqa: E402

GMEMORY_ROOT = substrate.ROOT / "third_party/GMemory"
EMBEDDING_MODEL = "all-mpnet-base-v2"

SUCCESSFUL_TOPK = 1
INSIGHTS_TOPK = 3
GRAPH_HOP = 1
SIMILARITY_THRESHOLD = 0.0
START_INSIGHTS_THRESHOLD = 5
ROUNDS_PER_INSIGHTS = 5
INSIGHTS_POINT_NUM = 5
SEED = 20260829


def render_trajectory(record: Dict[str, Any]) -> str:
    """What the two agents actually did, one high-level action per line."""
    lines = []
    for step in record["sketch"]:
        for agent, action in sorted(step["actions"].items()):
            verb = action[0]
            argument = action[1] if len(action) > 1 else ""
            lines.append(f"step {step['step']} agent {agent}: {verb}[{argument}]")
    return "\n".join(lines)


def as_records(rollouts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """G-Memory's record shape over PARTNR rollouts.

    Every rollout in the pile comes from the privileged scripted planner, so all of them
    carry `label=True`. G-Memory's failed-trajectory branch therefore never fires here,
    which is the same situation the VIKI-L2 port was in (`FAILED_TOPK=0`) and is stated
    rather than hidden: this port gives the method successes only.
    """
    return [
        {
            "memory_id": record["episode_id"],
            "task_main": record["instruction"],
            "task_description": (f"{record['instruction']}\nWhat had to end up true: "
                                 + "; ".join(record["goals"])),
            "trajectory": render_trajectory(record),
            "key_steps": "",
            "label": True,
        }
        for record in rollouts
        if record["instruction"]
    ]


def build(rollouts: List[Dict[str, Any]], complete: Callable[[List[Dict[str, str]]], str],
          embed: Callable[[List[str]], np.ndarray],
          progress: Optional[Callable[[str], None]] = None,
          workers: int = 8) -> Tuple[GMemoryState, np.ndarray]:
    """Add every record, distilling insights on the authors' schedule.

    `complete` takes chat messages and returns the model's text; `embed` takes strings and
    returns a matrix. Both are injected so the build can be driven by whatever endpoint is
    up without this module knowing about it.
    """
    say = progress or (lambda message: None)
    prompts = load_author_prompts(GMEMORY_ROOT)
    records = as_records(rollouts)
    say(f"condensing {len(records)} trajectories")

    def condense(record: Dict[str, Any]) -> str:
        return complete([
            {"role": "system", "content": prompts["extract_true_traj_system_prompt"]},
            {"role": "user", "content": prompts["extract_true_traj_user_prompt"].format(
                task=record["task_description"], trajectory=record["trajectory"])},
        ])

    # Condensation is per record and order-independent, so it is the one part of the
    # build that parallelizes. The insight rounds below cannot: each one reads the rules
    # the previous round wrote.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record, key_steps in zip(records, pool.map(condense, records)):
            record["key_steps"] = key_steps
    say(f"  condensed {len(records)}")

    embeddings = embed([record["task_description"] for record in records])
    state = GMemoryState(query_edge_threshold=SIMILARITY_THRESHOLD, hop=GRAPH_HOP)
    rng = random.Random(SEED)

    for position, record in enumerate(records):
        state.add_record(record, embeddings[position])
        size = position + 1
        if size < START_INSIGHTS_THRESHOLD or size % ROUNDS_PER_INSIGHTS:
            continue
        for _ in range(INSIGHTS_POINT_NUM):
            anchor = rng.choice(range(size))
            selected = state.nearest_record_indices(
                state.embeddings[anchor], count=3, label=True) + [anchor]
            rng.shuffle(selected)
            tasks = [str(state.records[i]["task_main"]) for i in selected]
            local_ids = state.related_insight_ids(tasks, threshold=len(tasks) / 2)
            local_rules = [state.insights[i]["rule"] for i in local_ids] or [""]
            history = "\n".join(
                f"task{index}:\n{state.records[i]['task_description']}{state.records[i]['key_steps']}"
                for index, i in enumerate(selected))
            suffix = ("Focus on REMOVE or EDIT or AGREE rules first, and stop ADD rule "
                      "unless the new rule is VERY insightful and different from "
                      "EXISTING RULES.\n" if len(state.insights) > 10 else "")
            response = complete([
                {"role": "system",
                 "content": prompts["critique_success_rules_system_prompt"] + suffix},
                {"role": "user", "content": prompts["critique_success_rules_user_prompt"].format(
                    success_history=history,
                    existing_rules="\n".join(
                        f"{i}. {rule}" for i, rule in enumerate(local_rules, 1)))},
            ])
            update_rules(state.insights, tasks, parse_rule_operations(response),
                         local_insight_ids=local_ids)
        say(f"  {size} records, {len(state.insights)} insights")
    return state, embeddings


VIKI_CLOSING = ("Use these memories only as reference. Produce one complete plan for the "
                "current image and instruction using only the activated robot APIs.")
PARTNR_CLOSING = "Use these memories only as reference."


def retarget(text: str) -> str:
    """Drop the closing line that only makes sense on VIKI-L2.

    `render_retrieval_prompt` is the authors' own rendering and is reused unchanged, but
    its last sentence tells the model to produce one complete plan for an image using the
    activated robot APIs. PARTNR's ReAct emits one action at a time and there is no image,
    so leaving it in would hand this arm an instruction that contradicts the format it is
    scored on -- a handicap invented by the port rather than a property of the method.
    Everything above that line, which is the memory itself, is untouched.
    """
    return text.replace(VIKI_CLOSING, PARTNR_CLOSING)


class PartnrGMemory:
    """Answer an instruction with G-Memory's own retrieval prompt."""

    def __init__(self, state: GMemoryState, embed: Callable[[List[str]], np.ndarray],
                 successful_topk: int = SUCCESSFUL_TOPK, insights_topk: int = INSIGHTS_TOPK):
        self.state = state
        self.embed = embed
        self.successful_topk = successful_topk
        self.insights_topk = insights_topk
        self._cache: Dict[str, str] = {}

    def render(self, instruction: str) -> str:
        if instruction in self._cache:
            return self._cache[instruction]
        query = self.embed([instruction])[0]
        candidates = self.state.raw_success_candidates(query, self.successful_topk)
        insights = self.state.related_insights(query, self.insights_topk)
        text = retarget(render_retrieval_prompt(
            [self.state.records[index] for index in candidates], insights))
        self._cache[instruction] = text
        return text


def save(state: GMemoryState, embeddings: np.ndarray, path: Path) -> None:
    """State as JSON, embeddings as exact bytes.

    `from_canonical_state` re-normalizes what it is handed and checks the result against
    hashes of the vectors `add_record` normalized, so the restored bytes have to be the
    bytes that went in. Writing them as JSON decimals does not survive that: the float32
    values round-trip through text and one bit is enough to fail the check. Base64 of the
    raw little-endian float32 buffer round-trips exactly, and the check stays a real check
    rather than something switched off to make the load work.
    """
    raw = np.asarray(embeddings, dtype="<f4")
    path.write_text(json.dumps({
        "state": state.canonical_state(),
        "embedding_shape": list(raw.shape),
        "embedding_b64": base64.b64encode(raw.tobytes()).decode("ascii"),
    }, indent=1) + "\n")


def restore(path: Path) -> GMemoryState:
    stored = json.loads(Path(path).read_text())
    raw = np.frombuffer(base64.b64decode(stored["embedding_b64"]), dtype="<f4")
    raw = raw.reshape(tuple(stored["embedding_shape"]))
    return GMemoryState.from_canonical_state(stored["state"], list(raw))
