#!/usr/bin/env python3
"""Memory prompts for the three Amendment 8 memory arms.

Each arm owns one class with a single `prompt(index, sample)` entry point, so
the run loop stays identical across arms and the only difference between them
is how the memory text is produced.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

from habitat_llm.evaluation.viki_branch_conditions import get_instruction
from habitat_llm.evaluation.viki_gmemory import render_retrieval_prompt
from viki_amendment6 import GateFailure, format_trajectory_prompt, load_source_rows

EMBEDDER_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
TRAJECTORY_TOP_K = 1

# The retriever returns every skill above its abstract_threshold. On a 6699-episode
# bank that is ~716 skills, which renders to ~99k characters and overflows the
# 16384-token context. The threshold belongs to the method, so it is left alone;
# the render layer instead keeps the highest-scoring skills, mirroring how the
# G-Memory arm keeps SUCCESSFUL_TOPK=1 record plus INSIGHTS_TOPK=3 insights.
# Overridable so a budget-matched variant can be run without editing the file;
# the effective value is recorded in every variant run's metadata.
SKILL_RENDER_TOP_K = int(os.environ.get("A8B_SKILL_TOPK", "4"))

# Amendment 9. The coordination structure -- trigger, per-agent roles, coordination
# mechanism -- lives in 5 of the 412 cooperation skills. Those five pass abstract
# matching on every sampled row, scoring 0.31 to 0.48, but the task-specific skills
# score 0.70 to 0.83, so a plain top-K by abstract_score never reaches them: they
# ranked in the hundreds. Reserving slots is what puts the method's distinguishing
# content in the prompt at all. Off by default.
PATTERN_SLOTS = int(os.environ.get("A9_PATTERN_SLOTS", "0"))
# The stored action list flattens two agents into one chain and drops the step
# timing, which is the only thing VIKI-L2 scores. A9_STEP_ALIGNED=1 replaces that
# line with the real per-step structure, recovered from the training parquet, for
# the instances whose candidate source plans agree on it. Off by default.
STEP_ALIGNED = os.environ.get("A9_STEP_ALIGNED") == "1"
STEP_ALIGNED_HITS = {"rewritten": 0, "left_flat": 0}


def _encode(model, texts: List[str]) -> np.ndarray:
    return np.asarray(
        model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


class TrajectoryRag:
    """Flat nearest-neighbour retrieval over the shared training plans."""

    name = "trajectory_rag"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.source_ids, self.source_rows, self.instructions = load_source_rows()
        self.model = SentenceTransformer(EMBEDDER_MINILM, device="cpu")
        self.matrix = _encode(self.model, self.instructions)

    def prompt(self, index: int, sample: Dict[str, Any]) -> str:
        query = _encode(self.model, [get_instruction(sample)])[0]
        ranking = np.argsort(-(self.matrix @ query))[:TRAJECTORY_TOP_K]
        selected = [self.source_rows[self.source_ids[position]] for position in ranking]
        return format_trajectory_prompt(selected)


class GMemory:
    """The Amendment 7 hierarchy, reused unchanged: same memory bank, new queries."""

    name = "gmemory"

    def __init__(self, client) -> None:
        from sentence_transformers import SentenceTransformer

        import viki_amendment7 as a7
        from habitat_llm.evaluation.viki_gmemory import load_author_prompts

        self.a7 = a7
        self.client = client
        self.state = a7.load_frozen_hierarchy()
        self.prompts = load_author_prompts(a7.GMEMORY_ROOT)
        self.model = SentenceTransformer(EMBEDDER_MINILM, device="cpu")

    def prompt(self, index: int, sample: Dict[str, Any]) -> str:
        from habitat_llm.evaluation.viki_gmemory import parse_relevance_score

        a7 = self.a7
        instruction = get_instruction(sample)
        query = _encode(self.model, [instruction])[0]
        candidate_indices = self.state.raw_success_candidates(query, count=2)
        if len(candidate_indices) != 2:
            raise GateFailure(f"gmemory {index} did not retrieve two candidates")
        scores = []
        for position, candidate_index in enumerate(candidate_indices):
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
        best = candidate_indices[int(np.argmax(scores))]
        insights = self.state.related_insights(query, a7.INSIGHTS_TOPK)
        return render_retrieval_prompt([self.state.records[best]], insights)


# format_for_prompt only reads the object-centric context fields (objects,
# locations, rooms, object_locations, action_sequence). A cooperation skill
# stores none of those: its content lives in trigger_conditions, precond_joint,
# partner_context_pattern and per-agent roles, so the generic renderer reduces it
# to a name and one prose line. Those fields are the part of this method that
# G-Memory has no counterpart for, so they are surfaced here. Nothing is
# invented: every line below is read straight out of the stored skill.
COOP_ROLE_PREFIX = "agent_"
COOP_ROLE_SUFFIX = "_role"


def _first_context(instances, predicate) -> Dict[str, Any]:
    """A cooperation skill stores two flavours of instance: one carrying
    objects/action_sequence, one carrying the per-agent roles. The retriever never
    returns the role flavour: _context_to_text builds an instance's embedding from
    objects/locations/rooms/object_locations/action_sequence only, so a role
    instance embeds near-empty text, scores last, and is cut by instance_top_k.
    The roles are therefore read from the stored skill rather than from the
    retrieved subset. Retrieval and scoring are left untouched."""
    for instance in instances or []:
        context = instance.get("context") or {}
        if predicate(context):
            return context
    return {}


def _has_role(context: Dict[str, Any]) -> bool:
    return any(
        key.startswith(COOP_ROLE_PREFIX) and key.endswith(COOP_ROLE_SUFFIX) and value
        for key, value in context.items()
    )


_COOP_DETAIL_CACHE: Dict[str, List[str]] = {}


def _coop_details(retriever, skill) -> List[str]:
    # The lines depend only on the stored skill, never on the query, but the scan
    # walks every stored instance and the reserved pattern skills are the largest
    # in the bank -- division_of_labor alone holds 6682. Recomputing per row made
    # this process the machine's top CPU consumer, so the result is memoised.
    cached = _COOP_DETAIL_CACHE.get(skill.skill_name)
    if cached is not None:
        return cached
    entry = getattr(retriever, "L_coop", {}).get(f"{skill.skill_name}_cooperation", {})
    lines: List[str] = []

    triggers = entry.get("trigger_conditions") or []
    joint = (entry.get("precond_joint") or {}).get("trigger")
    trigger = triggers[0] if triggers else joint
    if trigger:
        lines.append(f"Applies when: {trigger}")

    stored = entry.get("instances") or []
    context = _first_context(stored, _has_role)
    roles = [
        f"{key[len(COOP_ROLE_PREFIX):-len(COOP_ROLE_SUFFIX)]}={value}"
        for key, value in sorted(context.items())
        if key.startswith(COOP_ROLE_PREFIX)
        and key.endswith(COOP_ROLE_SUFFIX)
        and value
    ]
    if roles:
        lines.append("Agent roles: " + "; ".join(roles))
    coordination = context.get("coordination_mechanism") or _first_context(
        stored, lambda ctx: bool(ctx.get("coordination_mechanism"))
    ).get("coordination_mechanism")
    if coordination:
        lines.append(f"Coordination: {coordination}")

    pattern = entry.get("partner_context_pattern") or {}
    reasoning = [
        f"{key.replace('_', ' ')}: {value}"
        for key, value in sorted(pattern.items())
        if value
    ]
    if reasoning:
        lines.append("Partner reasoning -- " + "; ".join(reasoning))
    _COOP_DETAIL_CACHE[skill.skill_name] = lines
    return lines


def render_skills(retriever, skills: List[Any]) -> str:
    """One block per skill: the method's own renderer for the shared fields, plus
    the cooperation fields it cannot emit. Rendering each skill separately keeps
    format_for_prompt as the single source of truth for the shared part."""
    blocks = []
    for position, skill in enumerate(skills):
        text = retriever.format_for_prompt([skill], max_examples=1)
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.replace("### Skill 1:", f"### Skill {position + 1}:", 1).strip()
        if STEP_ALIGNED:
            import viki_amendment9_stepalign as stepalign

            sequence = None
            if skill.instances:
                sequence = skill.instances[0].get("context", {}).get(
                    "action_sequence"
                )
            text, done = stepalign.rewrite(text, sequence)
            STEP_ALIGNED_HITS["rewritten" if done else "left_flat"] += 1
        if skill.skill_type == "cooperation":
            extra = _coop_details(retriever, skill)
            if extra:
                text = text + "\n" + "\n".join(extra)
        blocks.append(text)
    return "## Retrieved Skills from Memory\n\n" + "\n\n".join(blocks)


class SkillMemory:
    """The hierarchical skill memory, queried through the VIKI adapter."""

    name = "skill_memory"
    TRAILER = (
        "\n---\n\n"
        "Use these skills only as reference. Produce one complete plan for the "
        "current image and instruction using only the activated robot APIs."
    )

    def __init__(self, memory_dir: Path, exposed_steps: int, self_robot: str) -> None:
        import viki_adapter as adapter
        from hierarchical_retrieval import create_retriever

        if not Path(memory_dir).exists():
            raise GateFailure("Build the Amendment 8 skill memory bank first")
        self.adapter = adapter
        self.retriever = create_retriever(str(memory_dir))
        self.exposed_steps = exposed_steps
        self.self_robot = self_robot
        # Skills whose stored instances carry per-agent roles: the coordination
        # patterns. Identified from the bank once, not guessed from the name.
        self.pattern_skills = {
            key[: -len("_cooperation")]
            for key, entry in getattr(self.retriever, "L_coop", {}).items()
            if key.endswith("_cooperation")
            and any(
                any(
                    field.startswith(COOP_ROLE_PREFIX)
                    and field.endswith(COOP_ROLE_SUFFIX)
                    and value
                    for field, value in ((instance.get("context") or {}).items())
                )
                for instance in (entry.get("instances") or [])
            )
        }

    def _select(self, ranked: List[Any]) -> List[Any]:
        """Top-K by score, except that PATTERN_SLOTS of the places are held for
        coordination patterns when any were retrieved. The reserved places are
        filled by the highest-scoring patterns, so the ordering within each group
        is still the method's own."""
        if PATTERN_SLOTS <= 0:
            return ranked[:SKILL_RENDER_TOP_K]
        patterns = [s for s in ranked if s.skill_name in self.pattern_skills]
        others = [s for s in ranked if s.skill_name not in self.pattern_skills]
        reserved = patterns[:PATTERN_SLOTS]
        remaining = SKILL_RENDER_TOP_K - len(reserved)
        return reserved + others[:remaining]

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
        skills = sorted(skills, key=lambda item: -item.abstract_score)
        skills = self._select(skills)
        # The retriever ships its own renderer, which emits each skill's concrete
        # action_sequence, objects and object_locations alongside the prose demo.
        # The earlier hand-rolled block here forwarded only skill_name and demo,
        # so the action sequences never reached the model at all.
        body = render_skills(self.retriever, skills)
        return body + self.TRAILER
