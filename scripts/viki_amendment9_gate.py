#!/usr/bin/env python3
"""Mechanism gates and the running ledger for the Amendment 9 ladder.

Each rung changes one thing. A rung's accuracy is only interpretable if the change
actually took effect, so every rung has a mechanism gate that is independent of the
score: the query must really carry the partner action, the role instances must
really be retrievable, the grounded plan must really be attached. A rung whose
mechanism gate fails is reported as a defect, not as evidence about the method.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment5 import atomic_json
from viki_amendment8b import (
    EXPOSED_STEPS,
    OUTPUT_DIR,
    SELF_ROBOT,
    SOURCE_PARQUET,
    SELF_ROBOT as _SELF,
    load_manifest,
    native,
)

LEDGER = OUTPUT_DIR / "amendment9_ledger.json"
SAMPLE_ROWS = 24


def _provider():
    mode = os.environ.get("A9_MODE", "")
    if mode in ("grounded", "rescore"):
        import viki_amendment9_grounding as grounding

        return grounding.GroundedSkillMemory(
            OUTPUT_DIR / "skill_memory_bank", EXPOSED_STEPS, SELF_ROBOT
        )
    import viki_amendment8_memory as memories

    return memories.SkillMemory(
        OUTPUT_DIR / "skill_memory_bank", EXPOSED_STEPS, SELF_ROBOT
    )


def check(rung: str) -> Dict[str, Any]:
    provider = _provider()
    dataset = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()
    rows = sorted(manifest)[:: max(1, len(manifest) // SAMPLE_ROWS)][:SAMPLE_ROWS]

    partner_in_query = 0
    pattern_rows = 0
    role_skills = 0
    coop_skills = 0
    grounded = 0
    tiers: Dict[str, int] = {}
    lengths = []
    import viki_amendment8_memory as memories

    for index in rows:
        sample = native(dataset.iloc[index].to_dict())
        ground_truth = sample["reward_model"]["ground_truth"]
        inputs = provider.adapter.retrieval_inputs(
            ground_truth, EXPOSED_STEPS, SELF_ROBOT
        )
        query = provider.retriever.generate_query(
            agent_state=inputs["agent_state"],
            environment_state=inputs["environment_state"],
            partner_effects=inputs["partner_effects"],
            goal=inputs["goal"],
        )
        if "Partner action:" in query.text_query:
            partner_in_query += 1
        skills = provider.retriever.retrieve(
            agent_state=inputs["agent_state"],
            environment_state=inputs["environment_state"],
            partner_effects=inputs["partner_effects"],
            goal=inputs["goal"],
        )
        selected = provider._select(
            sorted(skills, key=lambda item: -item.abstract_score)
        )
        coop_skills += sum(1 for s in selected if s.skill_type == "cooperation")
        if any(s.skill_name in provider.pattern_skills for s in selected):
            pattern_rows += 1
        # The end-to-end question is whether the coordination content reaches the
        # model, so the check reads the rendered prompt rather than an instance
        # field. An earlier version inspected instances[0] and reported 0 even
        # when the cause was the top-K cut, several stages upstream.
        text = provider.prompt(index, sample)
        lengths.append(len(text))
        if "Agent roles:" in text or "Coordination:" in text:
            role_skills += 1
        if "## One completed plan that follows this pattern" in text:
            grounded += 1
        tier = getattr(provider, "last_tier", None)
        if tier:
            tiers[tier] = tiers.get(tier, 0) + 1

    result = {
        "rung": rung,
        "sampled_rows": len(rows),
        "env": {
            key: os.environ.get(key, "")
            for key in ("A9_MODE", "A9_ROLE_AWARE", "A8B_SKILL_TOPK", "A9_GROUND_TOPK")
        },
        "partner_in_query": f"{partner_in_query}/{len(rows)}",
        "coop_skills_selected": coop_skills,
        "rows_with_a_pattern_skill": f"{pattern_rows}/{len(rows)}",
        "rows_with_role_text_in_prompt": f"{role_skills}/{len(rows)}",
        "grounded_rows": f"{grounded}/{len(rows)}",
        "grounding_filter_tier": tiers,
        "memory_chars_median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
    }

    gates = {
        "query_carries_partner": partner_in_query >= 0.9 * len(rows),
        "pattern_selected": pattern_rows >= 0.9 * len(rows),
        "roles_in_prompt": role_skills >= 0.9 * len(rows),
        "plan_attached": grounded >= 0.9 * len(rows),
    }
    # Each rung requires only the mechanism it actually introduces. The first
    # ladder run required roles_retrieved for the grounded and rescore rungs too,
    # so a C1 failure blocked two rungs whose mechanism does not depend on C1 at
    # all, and neither was measured.
    required = {
        "queryfix": ["query_carries_partner"],
        "patternslot": ["query_carries_partner", "pattern_selected", "roles_in_prompt"],
        "grounded": ["query_carries_partner", "plan_attached"],
        "rescore": ["query_carries_partner", "plan_attached"],
    }.get(rung, [])
    result["gates"] = gates
    result["required"] = required
    result["pass"] = all(gates[name] for name in required)

    ledger = json.loads(LEDGER.read_text()) if LEDGER.is_file() else {"rungs": []}
    ledger["rungs"] = [r for r in ledger["rungs"] if r.get("rung") != rung] + [result]
    atomic_json(LEDGER, ledger)
    return result


if __name__ == "__main__":
    rung_name = sys.argv[1] if len(sys.argv) > 1 else "queryfix"
    outcome = check(rung_name)
    # Written here rather than by redirecting stdout: the simulator prints a banner
    # to stdout on import, so the redirected file was not valid JSON and every
    # later attempt to read a gate result failed.
    atomic_json(OUTPUT_DIR / f"gate_{rung_name}.json", outcome)
    print(json.dumps(outcome, indent=2))
    raise SystemExit(0 if outcome["pass"] else 3)
