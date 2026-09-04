#!/usr/bin/env python3
"""Who runs each requirement: the memory's own search, or the robots the model named.

The intent prompt asks for a `robots` field on every work item, and the two delegation
arms differ only in whether the consumer reads it. Because the prompt is one shared
constant, an archived run can be re-scored as either arm without asking the model
anything, which is what makes the whole comparison free -- and it is why this lives in one
place: if the two arms were assembled separately they could drift apart in some detail and
the 24-point gap between them would stop being attributable to delegation alone.

Mirrors `viki_eval_v2_intent_choice.py` exactly, including the recast fallback: a casting
the world cannot honour falls back to a free search rather than scoring the row zero, so
the arm is not punished for the scheduler's inability to use a legal assignment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from our_method.skill_memory_v2 import planner

CREW_CHOICES = ("memory", "model")


def collect(work: List[Any], to_requirement, memory, scene, metadata
            ) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
    """Requirements and the robots named for each, kept in step with one another."""
    requirements: List[Dict[str, Any]] = []
    crew: List[List[str]] = []
    for item in work:
        requirement = to_requirement(memory, item, scene) if isinstance(item, dict) else None
        if requirement is None:
            continue
        requirements.append(requirement)
        crew.append([name for name in (item.get("robots") or []) if name in metadata["agents"]])
    return requirements, crew


def casting_of(requirements, crew, use_crew: bool) -> Optional[Dict[str, str]]:
    if not use_crew:
        return None
    return {
        planner.predicate_key(requirement): names[0]
        for requirement, names in zip(requirements, crew)
        if names
    }


def solve(blind, memory, sim, seed, requirements, crew, temporal, use_crew: bool):
    """Plan under the chosen delegation. Returns (plan, reason, recast)."""
    casting = casting_of(requirements, crew, use_crew)
    blind["goal_constraints"] = [[requirement] for requirement in requirements]
    blind["temporal_constraints"] = temporal
    plan, reason = planner.plan(blind, memory, sim, seed, crew=casting)
    if plan is None:
        blind["temporal_constraints"] = []
        plan, reason = planner.plan(blind, memory, sim, seed, crew=casting)
    recast = False
    if plan is None and casting:
        recast = True
        blind["temporal_constraints"] = temporal
        plan, reason = planner.plan(blind, memory, sim, seed)
    return plan, reason, recast
