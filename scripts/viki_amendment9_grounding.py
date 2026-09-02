#!/usr/bin/env python3
"""Amendment 9 C2/C3: grounding a retrieved skill in a concrete training episode.

Amendment 8b showed the retrieved skills are fragments drawn from different
episodes whose action sequences contradict each other, while the scorer measures a
step-aligned multi-robot plan. The skills therefore select, and one training
episode supplies the concrete realisation.

Retrieval stays ours: the skill hierarchy, queried with state and partner effects,
decides which structural pattern applies. Grounding then picks the episode that
instantiates that pattern and is closest to the current query. C3 adds the LLM
relevance rescoring step, which is taken from G-Memory and reported as taken.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

from habitat_llm.evaluation.viki_branch_conditions import get_instruction
from viki_amendment6 import GateFailure, format_trajectory_prompt, load_source_rows
from viki_amendment8_memory import EMBEDDER_MINILM, SkillMemory, _encode, render_skills

GROUND_TOP_K = int(os.environ.get("A9_GROUND_TOPK", "3"))
RESCORE_CANDIDATES = int(os.environ.get("A9_RESCORE_N", "2"))
VERB = re.compile(r"^([A-Za-z_]+)")


def _verb(action_text: str) -> str:
    match = VERB.match(str(action_text).strip())
    return match.group(1) if match else ""


def _is_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    position = 0
    for item in haystack:
        if position < len(needle) and item == needle[position]:
            position += 1
    return position == len(needle)


class GroundedSkillMemory(SkillMemory):
    """Skill retrieval, then one concrete episode that instantiates the skill."""

    name = "skill_memory"

    def __init__(
        self,
        memory_dir: Path,
        exposed_steps: int,
        self_robot: str,
        client: Any = None,
        rescore: bool = False,
    ) -> None:
        super().__init__(memory_dir, exposed_steps, self_robot)
        from sentence_transformers import SentenceTransformer

        import viki_adapter as adapter

        self.client = client
        self.rescore = rescore
        self.source_ids, self.source_rows, self.instructions = load_source_rows()
        self.model = SentenceTransformer(EMBEDDER_MINILM, device="cpu")
        self.matrix = _encode(self.model, self.instructions)
        # Verb signature per episode, so a skill's action_sequence can be tested
        # for containment without re-parsing the plan on every query.
        # load_source_rows maps a source_train_index to a list of SegmentInstance,
        # each carrying its demo as [{"actions": {...}, "step": n}, ...]. An earlier
        # version indexed these as record["trajectory"] inside a bare except, so
        # every episode silently became an empty verb sequence and grounding
        # attached a plan to 0 of 924 rows. Failures are raised now.
        self.verbs: List[Tuple[str, ...]] = []
        for source_id in self.source_ids:
            sequence: List[str] = []
            for instance in self.source_rows[source_id]:
                for step in instance.demo:
                    actions = step.get("actions") or {}
                    for robot_id in sorted(actions):
                        value = actions[robot_id]
                        if value is None:
                            continue
                        verb, _ = adapter.action_parts(value)
                        sequence.append(verb)
            self.verbs.append(tuple(sequence))
        if not any(self.verbs):
            raise GateFailure("grounding built no verb signatures from the source rows")

    def _skill_verbs(self, skill: Any) -> List[Tuple[str, ...]]:
        patterns = []
        for instance in skill.instances or []:
            sequence = (instance.get("context") or {}).get("action_sequence") or []
            verbs = tuple(_verb(item) for item in sequence if _verb(item))
            if verbs:
                patterns.append(verbs)
        return patterns

    def _candidates(self, skills: List[Any]) -> Tuple[List[int], str]:
        """Episodes that instantiate any retrieved skill's action pattern.

        Skills are merged across episodes, so a skill's action_sequence need not
        appear verbatim anywhere: ordered matching alone grounded 15 of 24 sampled
        rows. The filter therefore relaxes in fixed steps and reports which step it
        used, so a run can say how often the strict form was enough rather than
        hiding the fallback.
        """
        wanted = [pattern for skill in skills for pattern in self._skill_verbs(skill)]
        if not wanted:
            return list(range(len(self.verbs))), "unfiltered"

        ordered = [
            position
            for position, episode in enumerate(self.verbs)
            if episode and any(_is_subsequence(p, episode) for p in wanted)
        ]
        if ordered:
            return ordered, "ordered"

        # Same verbs, order ignored: still skill-mediated, just weaker.
        sets = [set(pattern) for pattern in wanted]
        unordered = [
            position
            for position, episode in enumerate(self.verbs)
            if episode and any(wanted_set <= set(episode) for wanted_set in sets)
        ]
        if unordered:
            return unordered, "unordered"
        return list(range(len(self.verbs))), "unfiltered"

    def _query_text(self, sample: Dict[str, Any]) -> str:
        ground_truth = sample["reward_model"]["ground_truth"]
        inputs = self.adapter.retrieval_inputs(
            ground_truth, self.exposed_steps, self.self_robot
        )
        partner = (inputs["partner_effects"] or {}).get("action")
        text = get_instruction(sample)
        return f"{text} | partner: {partner}" if partner else text

    def ground(self, sample: Dict[str, Any], skills: List[Any]) -> Tuple[List[int], str]:
        """Rank the skill-compatible episodes by similarity to the query."""
        candidates, tier = self._candidates(skills)
        if not candidates:
            return [], tier
        query = _encode(self.model, [self._query_text(sample)])[0]
        scores = self.matrix[candidates] @ query
        order = np.argsort(-scores)
        return [candidates[int(position)] for position in order], tier

    def _pick(self, index: int, sample: Dict[str, Any], ranked: List[int]) -> int:
        if not self.rescore or self.client is None or len(ranked) < 2:
            return ranked[0]
        # G-Memory's relevance step, applied to our skill-filtered candidates.
        import viki_amendment7 as a7
        from habitat_llm.evaluation.viki_gmemory import (
            load_author_prompts,
            parse_relevance_score,
        )

        prompts = load_author_prompts(a7.GMEMORY_ROOT)
        instruction = get_instruction(sample)
        best, best_score = ranked[0], None
        for position, candidate in enumerate(ranked[:RESCORE_CANDIDATES]):
            segments = self.source_rows[self.source_ids[candidate]]
            messages = [
                {"role": "system", "content": prompts["generative_task_system_prompt"]},
                {
                    "role": "user",
                    "content": prompts["generative_task_user_prompt"].format(
                        trajectory=format_trajectory_prompt([segments]),
                        query_scenario=instruction,
                    ),
                },
            ]
            completion = self.client.chat.completions.create(
                model=a7.SERVED_MODEL,
                messages=messages,
                max_tokens=a7.MEMORY_CONTROL_MAX_TOKENS,
                temperature=a7.MEMORY_CONTROL_TEMPERATURE,
                seed=a7.SEED + index * 4 + position,
            )
            score = parse_relevance_score(completion.choices[0].message.content or "")
            if best_score is None or score > best_score:
                best, best_score = candidate, score
        return best

    def prompt(self, index: int, sample: Dict[str, Any]) -> str:
        ground_truth = sample["reward_model"]["ground_truth"]
        inputs = self.adapter.retrieval_inputs(
            ground_truth, self.exposed_steps, self.self_robot
        )
        skills = self.retriever.retrieve(
            agent_state=inputs["agent_state"],
            environment_state=inputs["environment_state"],
            partner_effects=inputs["partner_effects"],
            goal=inputs["goal"],
        )
        if not skills:
            return ""
        # _select applies the reserved coordination-pattern slots. Sorting and
        # slicing here instead, as an earlier version did, silently dropped the
        # pattern skills again: role text reached 0 of 24 rows in grounded mode
        # even though the same check read 23 of 24 without grounding.
        ranked_skills = sorted(skills, key=lambda item: -item.abstract_score)
        skills = self._select(ranked_skills)[:GROUND_TOP_K]
        body = render_skills(self.retriever, skills)

        ranked, self.last_tier = self.ground(sample, skills)
        if not ranked:
            return body + self.TRAILER
        chosen = self._pick(index, sample, ranked)
        grounded = format_trajectory_prompt(
            [self.source_rows[self.source_ids[chosen]]]
        )
        return (
            body
            + "\n\n## One completed plan that follows this pattern\n"
            + grounded
            + self.TRAILER
        )
