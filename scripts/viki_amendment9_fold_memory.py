#!/usr/bin/env python3
"""Fold-aware providers: every arm loses the same held-out family.

A held-out split only means something if it is held out from every arm. Each arm
hides the family differently because each stores its memory differently:

  skill_memory    a bank rebuilt without the family's episodes
  trajectory_rag  the family's rows dropped from the retrieval pool
  gmemory         the family's records masked at retrieval, and any insight whose
                  positive_correlation_tasks name a held-out task dropped
  zero_shot       carries no memory, so it is unaffected

The G-Memory hierarchy is frozen and its embeddings, graph and neighbour indices
are positional, so records are masked at query time rather than deleted -- deleting
would renumber everything the frozen certificate covers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

from habitat_llm.evaluation.viki_branch_conditions import get_instruction
from habitat_llm.evaluation.viki_gmemory import render_retrieval_prompt
from viki_amendment6 import GateFailure
from viki_amendment8_memory import (
    EMBEDDER_MINILM,
    TRAJECTORY_TOP_K,
    GMemory,
    TrajectoryRag,
    _encode,
)
from viki_amendment9_folds import (
    episode_families,
    held_out_ids,
    train_family_by_index,
)

# A fold masks every record of the held-out family, and G-Memory's nearest
# neighbours are overwhelmingly drawn from exactly that family: with a pool of 8
# the first fold could not find two permitted candidates for a single row. The pool
# is widened so the arm still gets its two candidates from the material it is
# allowed to see. This changes which records it ranks, not how it ranks them.
# Rank every record, then drop the held-out ones, then take the best two. Taking a
# fixed-size pool first does not work here: G-Memory's neighbours are so
# concentrated in the query's own family that even the top 256 were entirely
# masked on some rows. Ranking everything gives the arm its two best permitted
# candidates, which is what the fold is supposed to measure.
CANDIDATE_POOL = int(os.environ.get("A9_GMEM_POOL", "0")) or None

# Control for what G-Memory's retrieval actually contributes on the folds. Three
# separate structural analyses found nothing in the bank for a held-out family to
# reuse: the ground-truth plan never survives the fold, nor does the object-typed
# coordination template, and the rows whose verb skeleton does survive are the rows
# every arm fails. So its 20.78% may not be retrieval at all, but the base model
# primed by one well-formed example. Setting A9_GMEM_SHUFFLE=1 keeps the arm
# identical except that the trajectory is drawn uniformly from the permitted
# records instead of by similarity. If the score holds up, the retrieval is doing
# no work on this split.
SHUFFLE = os.environ.get("A9_GMEM_SHUFFLE") == "1"


class FoldTrajectoryRag(TrajectoryRag):
    """Nearest-neighbour retrieval with the held-out family removed."""

    def __init__(self, family: str) -> None:
        super().__init__()
        # load_source_rows is keyed by source_train_index, not memory_id, so the
        # family map has to be the by-index one. Using episode_families() here
        # matched nothing and would have left the held-out family in the pool.
        families = train_family_by_index()
        keep = [
            position
            for position, source_id in enumerate(self.source_ids)
            if families.get(source_id) != family
        ]
        if len(keep) == len(self.source_ids):
            raise GateFailure(f"fold {family} removed nothing from trajectory_rag")
        self.source_ids = [self.source_ids[i] for i in keep]
        self.instructions = [self.instructions[i] for i in keep]
        self.matrix = self.matrix[keep]
        self.removed = len(keep)


class FoldGMemory(GMemory):
    """The frozen hierarchy, with the held-out family masked at retrieval."""

    def __init__(self, client, family: str) -> None:
        super().__init__(client)
        self.family = family
        removed = held_out_ids(family)
        self.blocked = {
            position
            for position, record in enumerate(self.state.records)
            if record.get("memory_id") in removed
        }
        if not self.blocked:
            raise GateFailure(f"fold {family} masked no gmemory records")
        held_tasks = {
            self.state.records[position].get("task_main") for position in self.blocked
        }
        self.blocked_insights = {
            index
            for index, insight in enumerate(self.state.insights)
            if held_tasks.intersection(insight.get("positive_correlation_tasks") or [])
        }

    def prompt(self, index: int, sample: Dict[str, Any]) -> str:
        from habitat_llm.evaluation.viki_gmemory import parse_relevance_score

        a7 = self.a7
        instruction = get_instruction(sample)
        query = _encode(self.model, [instruction])[0]
        if SHUFFLE:
            import random as _random

            permitted = [
                position
                for position in range(len(self.state.records))
                if position not in self.blocked
            ]
            chosen = _random.Random(a7.SEED + index).choice(permitted)
            insights = self._allowed_insights(query, a7.INSIGHTS_TOPK)
            return render_retrieval_prompt([self.state.records[chosen]], insights)
        pool = self.state.raw_success_candidates(
            query, count=CANDIDATE_POOL or len(self.state.records)
        )
        allowed = [position for position in pool if position not in self.blocked][:2]
        if len(allowed) < 2:
            raise GateFailure(
                f"gmemory {index} found {len(allowed)} permitted candidates in fold "
                f"{self.family} after ranking all {len(self.state.records)} records"
            )
        scores = []
        for position, candidate_index in enumerate(allowed):
            candidate = self.state.records[candidate_index]
            messages = [
                {
                    "role": "system",
                    "content": self.prompts["generative_task_system_prompt"],
                },
                {
                    "role": "user",
                    "content": self.prompts["generative_task_user_prompt"].format(
                        trajectory=(
                            str(candidate["task_description"])
                            + "\n"
                            + str(candidate["trajectory"])
                        ),
                        query_scenario=instruction,
                    ),
                },
            ]
            completion = self.client.chat.completions.create(
                model=a7.SERVED_MODEL,
                messages=messages,
                max_tokens=a7.MEMORY_CONTROL_MAX_TOKENS,
                temperature=a7.MEMORY_CONTROL_TEMPERATURE,
                seed=a7.SEED + index * 2 + position,
            )
            scores.append(
                parse_relevance_score(completion.choices[0].message.content or "")
            )
        best = allowed[int(np.argmax(scores))]
        insights = self._allowed_insights(query, a7.INSIGHTS_TOPK)
        return render_retrieval_prompt([self.state.records[best]], insights)

    def _allowed_insights(self, query: np.ndarray, count: int) -> List[str]:
        if not self.blocked_insights:
            return self.state.related_insights(query, count)
        wider = self.state.related_insights(query, count + len(self.blocked_insights))
        blocked_rules = {
            self.state.insights[index]["rule"] for index in self.blocked_insights
        }
        return [rule for rule in wider if rule not in blocked_rules][:count]


def fold_bank(family: str) -> Path:
    from viki_amendment9_fold_build import FOLD_ROOT

    bank = FOLD_ROOT / family
    if not (bank / "memory_summary.json").is_file():
        raise GateFailure(f"fold bank for {family} has not been built")
    return bank


def make_fold_memory(arm: str, client, family: str, exposed: int, self_robot: str):
    if arm == "zero_shot":
        return None
    if arm == "trajectory_rag":
        return FoldTrajectoryRag(family)
    if arm == "gmemory":
        return FoldGMemory(client, family)
    if arm == "skill_memory":
        mode = os.environ.get("A9_MODE", "")
        if mode in ("grounded", "rescore"):
            import viki_amendment9_grounding as grounding

            provider = grounding.GroundedSkillMemory(
                fold_bank(family),
                exposed,
                self_robot,
                client=client,
                rescore=mode == "rescore",
            )
            # The grounding pool is the training set, so it needs the same cut.
            families = train_family_by_index()
            keep = [
                position
                for position, source_id in enumerate(provider.source_ids)
                if families.get(source_id) != family
            ]
            if len(keep) == len(provider.source_ids):
                raise GateFailure(
                    f"fold {family} removed nothing from the grounding pool"
                )
            provider.source_ids = [provider.source_ids[i] for i in keep]
            provider.instructions = [provider.instructions[i] for i in keep]
            provider.matrix = provider.matrix[keep]
            provider.verbs = [provider.verbs[i] for i in keep]
            return provider
        import viki_amendment8_memory as memories

        return memories.SkillMemory(fold_bank(family), exposed, self_robot)
    raise GateFailure(f"Unknown arm: {arm}")
