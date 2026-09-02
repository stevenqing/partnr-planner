"""Layer 2: which orderings are real, learned rather than asserted.

Asked what order things must happen in, a model narrates the instruction back, and an
order the task does not impose serialises work that could have run at once -- which
costs steps against a budget measured in steps. Asked for no order at all, the tasks
that genuinely need one fail outright. So the ordering has to come from somewhere that
is neither the model's paraphrase nor a rule someone wrote after seeing which families
broke.

The training episodes state their temporal constraints outright, so the rule can be
mined. Every ordered pair of an episode's requirements is labelled by whether the
episode ordered them, and described by a handful of structural relations plus one fact
that only Layer 1 can supply: whether the body that achieves the later requirement goes
to the place the earlier one puts something. That last feature is what separates cutting
-- where the fruit must be on the board because the knife is used at the board, and the
two predicates share no argument at all -- from two objects that merely end up in the
same cupboard and are independent.

A pattern is kept only if the episodes that show it order it nearly always. Precision is
what matters here and recall is not: a missed ordering costs one task, an invented one
serialises every task that matches it.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .induction import _segment_start, replay, requirements_of
from .simulator import Simulator, predicate_status

MIN_SUPPORT = 30
MIN_PRECISION = 0.9


def _describe(a: Dict[str, Any], b: Dict[str, Any], b_visits_a_target: bool) -> str:
    sa, sb = predicate_status(a), predicate_status(b)
    a_target, b_target = sa.get("pos.name"), sb.get("pos.name")
    return json.dumps(
        {
            "a_key": "pos" if "pos.name" in sa else "act",
            "b_key": "pos" if "pos.name" in sb else "act",
            "a_subject_is_b_target": a["name"] == b_target,
            "a_target_is_b_subject": a_target == b["name"],
            "a_target_is_b_target": a_target is not None and a_target == b_target,
            "b_visits_a_target": bool(b_visits_a_target),
        },
        sort_keys=True,
    )


def _ordered_pairs(truth: Dict[str, Any]) -> set:
    """The pairs this episode actually ordered, as (earlier, later) predicate keys."""
    from .simulator import flatten_predicates

    def key(predicate):
        return json.dumps([predicate["name"], predicate_status(predicate)], sort_keys=True)

    edges = set()
    for constraint in truth.get("temporal_constraints") or []:
        stages = [[key(p) for p in flatten_predicates(stage)] for stage in constraint]
        for position, earlier in enumerate(stages):
            for later in stages[position + 1:]:
                for one in earlier:
                    for two in later:
                        edges.add((one, two))
    return edges


def mine(
    episodes: Iterable[Dict[str, Any]],
    sim: Simulator,
    seed: int,
    per_family: int = 250,
    exclude_family: Optional[str] = None,
) -> Dict[str, Any]:
    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # [ordered, seen]
    families: Dict[str, set] = defaultdict(set)
    seen_family = Counter()

    def key(predicate):
        return json.dumps([predicate["name"], predicate_status(predicate)], sort_keys=True)

    for truth in episodes:
        if not isinstance(truth, dict) or not truth.get("time_steps"):
            continue
        family = truth.get("task_name", "?")
        if exclude_family and family == exclude_family:
            continue
        if seen_family[family] >= per_family:
            continue
        seen_family[family] += 1
        trace, status = replay(truth, sim, seed)
        if trace is None:
            continue
        edges = _ordered_pairs(truth)
        requirements = requirements_of(truth)

        # Where each requirement's work went, so "the later body visits the earlier
        # target" can be read off the plan rather than guessed at.
        visits: Dict[str, set] = {}
        for index, actor, predicate in trace["completions"]:
            if actor is None:
                continue
            start = _segment_start(trace["completions"], index, actor)
            targets = set()
            for step in range(start, index + 1):
                action = trace["history"][step]["actions"].get(actor)
                if action:
                    targets.update(action[1:])
            visits[key(predicate)] = targets

        for a in requirements:
            for b in requirements:
                if a is b or key(a) == key(b):
                    continue
                a_target = predicate_status(a).get("pos.name")
                pattern = _describe(a, b, a_target is not None and a_target in visits.get(key(b), set()))
                counts[pattern][1] += 1
                if (key(a), key(b)) in edges:
                    counts[pattern][0] += 1
                    families[pattern].add(family)

    rules = []
    for pattern, (ordered, seen) in counts.items():
        precision = ordered / seen if seen else 0.0
        rules.append({
            "pattern": json.loads(pattern),
            "ordered": ordered,
            "seen": seen,
            "precision": round(precision, 4),
            "families": sorted(families.get(pattern, ())),
            "kept": bool(ordered >= MIN_SUPPORT and precision >= MIN_PRECISION),
        })
    rules.sort(key=lambda item: (-item["ordered"], -item["precision"]))
    kept = [item for item in rules if item["kept"]]
    covered = sum(item["ordered"] for item in kept)
    total = sum(item["ordered"] for item in rules)
    return {
        "rules": rules,
        "kept_patterns": [json.dumps(item["pattern"], sort_keys=True) for item in kept],
        "min_support": MIN_SUPPORT,
        "min_precision": MIN_PRECISION,
        "orderings_recalled": covered,
        "orderings_total": total,
        "recall": round(covered / total, 4) if total else 0.0,
        "false_orderings": sum(item["seen"] - item["ordered"] for item in kept),
    }


def order_for(
    requirements: List[Dict[str, Any]],
    visits: Dict[int, set],
    kept_patterns: List[str],
) -> List[Any]:
    """The orderings the learned patterns say this requirement set must respect.

    `visits` says, per requirement index, which objects the body chosen for it will
    address -- the one thing the patterns need that the predicates alone do not carry.
    """
    kept = set(kept_patterns)
    constraints = []
    for i, a in enumerate(requirements):
        for j, b in enumerate(requirements):
            if i == j:
                continue
            a_target = predicate_status(a).get("pos.name")
            pattern = _describe(a, b, a_target is not None and a_target in visits.get(j, set()))
            if pattern in kept:
                constraints.append([[a], [b]])
    return constraints
