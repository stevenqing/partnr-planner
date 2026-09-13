#!/usr/bin/env python3
"""Layer 2 for PartNR: the orderings a requirement set must respect, mined from train.

`partnr_planner.temporal_order` reads the episode's own `TemporalConstraint` DAG. That is
privileged -- the same privilege as reading its propositions -- and the planner says so at
`partnr_planner.py:461-463`: switching it off is the ablation that says what an ordering is
worth, "and so what the mined ordering rules will have to recover for the arm that has
none". This script mines those rules.

The shape is VIKI's `our_method/skill_memory_v2/dependencies.py:mine`, with the predicate
vocabulary swapped for PartNR's. A pattern is a pair of proposition keys plus which of
their arguments coincide; a pattern is kept when the episodes that show that pair almost
always order it the same way.

Mining reads train only. `val_mini` is the reporting split and is never opened here -- the
ledger's rule is that the reporting pool stays untouched until the arm runs.

Usage:
  python scripts/partnr_mine_orderings.py --source train_mini --out results/partnr_orderings_train_mini.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
EPISODES = ROOT / "data/datasets/partnr_episodes/v0_0"

# The two knobs, fixed here rather than swept. VIKI's layer 2 kept 30 / 0.9 and recovered
# 1277 of 1277 orderings; the same pair is the starting point so the two sides stay
# comparable, and any change to them is a change to the method, not a tuning step.
MIN_SUPPORT = 30
MIN_PRECISION = 0.9


def load(name: str) -> List[Dict[str, Any]]:
    path = EPISODES / ("%s.json.gz" % name)
    with gzip.open(path, "rt") as handle:
        data = json.load(handle)
    return data["episodes"] if isinstance(data, dict) else data


def _args(proposition: Dict[str, Any]) -> Dict[str, Any]:
    return proposition.get("args") or {}


def subjects_of(proposition: Dict[str, Any]) -> Set[str]:
    a = _args(proposition)
    return {str(h) for h in (a.get("object_handles") or a.get("entity_handles_a") or [])}


def targets_of(proposition: Dict[str, Any]) -> Set[str]:
    a = _args(proposition)
    return {
        str(h)
        for h in (
            a.get("receptacle_handles")
            or a.get("entity_handles_b")
            or a.get("room_ids")
            or []
        )
    }


def describe(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    """The pattern that a pair of propositions instantiates.

    Only the argument coincidences, never the object names themselves: a rule that names
    a mug is a rule about that episode, not about the ordering.
    """
    sa, ta = subjects_of(a), targets_of(a)
    sb, tb = subjects_of(b), targets_of(b)
    return json.dumps(
        {
            "a_key": a.get("function_name"),
            "b_key": b.get("function_name"),
            "a_subject_is_b_subject": bool(sa & sb),
            "a_subject_is_b_target": bool(sa & tb),
            "a_target_is_b_subject": bool(ta & sb),
            "a_target_is_b_target": bool(ta & tb),
        },
        sort_keys=True,
    )


def ordered_pairs(episode: Dict[str, Any]) -> Set[Tuple[int, int]]:
    """(earlier, later) proposition indices this episode actually orders.

    `dag_edges` are direct edges; the judge applies their transitive closure, so mining
    reads the closure too -- otherwise a chain i->j->k would count (i, k) as evidence that
    the pair is unordered, which is the opposite of what the episode says.
    """
    direct: Dict[int, Set[int]] = defaultdict(set)
    for constraint in episode.get("evaluation_constraints") or []:
        if not isinstance(constraint, dict):
            continue
        if constraint.get("type") != "TemporalConstraint":
            continue
        for edge in (constraint.get("args") or {}).get("dag_edges") or []:
            earlier, later = int(edge[0]), int(edge[1])
            direct[earlier].add(later)
    closure: Set[Tuple[int, int]] = set()
    for start in list(direct):
        stack, seen = list(direct[start]), set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            closure.add((start, node))
            stack.extend(direct.get(node, ()))
    return closure


def mine(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # [ordered, seen]
    episodes_with_dag = 0
    edges_total = 0
    for episode in episodes:
        propositions = episode.get("evaluation_propositions") or []
        if len(propositions) < 2:
            continue
        edges = ordered_pairs(episode)
        if edges:
            episodes_with_dag += 1
            edges_total += len(edges)
        for i, j in permutations(range(len(propositions)), 2):
            pattern = describe(propositions[i], propositions[j])
            counts[pattern][1] += 1
            if (i, j) in edges:
                counts[pattern][0] += 1
    rules = []
    for pattern, (ordered, seen) in counts.items():
        precision = ordered / seen if seen else 0.0
        rules.append(
            {
                "pattern": json.loads(pattern),
                "ordered": ordered,
                "seen": seen,
                "precision": precision,
                "kept": seen >= MIN_SUPPORT and precision >= MIN_PRECISION,
            }
        )
    rules.sort(key=lambda r: (-r["kept"], -r["ordered"]))
    kept = [r for r in rules if r["kept"]]
    return {
        "rules": rules,
        "kept_patterns": [json.dumps(r["pattern"], sort_keys=True) for r in kept],
        "min_support": MIN_SUPPORT,
        "min_precision": MIN_PRECISION,
        "episodes_read": len(episodes),
        "episodes_with_dag": episodes_with_dag,
        "edges_total": edges_total,
    }


def apply_rules(episode: Dict[str, Any], kept: Set[str]) -> Set[Tuple[int, int]]:
    """What the kept patterns would order for this episode, reading no DAG."""
    propositions = episode.get("evaluation_propositions") or []
    out: Set[Tuple[int, int]] = set()
    for i, j in permutations(range(len(propositions)), 2):
        if describe(propositions[i], propositions[j]) in kept:
            out.add((i, j))
    return out


def score(episodes: List[Dict[str, Any]], kept: Set[str]) -> Dict[str, Any]:
    """Recall and precision of the kept rules against the episodes' own DAGs."""
    hit = miss = false = 0
    episodes_fully_recovered = 0
    episodes_with_dag = 0
    for episode in episodes:
        truth = ordered_pairs(episode)
        predicted = apply_rules(episode, kept)
        if truth:
            episodes_with_dag += 1
            if truth <= predicted:
                episodes_fully_recovered += 1
        hit += len(truth & predicted)
        miss += len(truth - predicted)
        false += len(predicted - truth)
    return {
        "orderings_recalled": hit,
        "orderings_total": hit + miss,
        "recall": hit / (hit + miss) if (hit + miss) else 0.0,
        "false_orderings": false,
        "precision": hit / (hit + false) if (hit + false) else 0.0,
        "episodes_with_dag": episodes_with_dag,
        "episodes_fully_recovered": episodes_fully_recovered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="train_mini",
                        help="episode pool to mine; must be a train pool")
    parser.add_argument("--score-on", action="append", default=[],
                        help="pools to score the kept rules on; repeatable")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    forbidden = {"val_mini", "val"}
    if args.source in forbidden or set(args.score_on) & forbidden:
        print("refusing: %s is the reporting split and stays shut until the arm runs"
              % (args.source if args.source in forbidden else sorted(set(args.score_on) & forbidden)))
        return 2

    episodes = load(args.source)
    report = mine(episodes)
    report["source"] = args.source
    kept = set(report["kept_patterns"])
    report["in_sample"] = score(episodes, kept)
    # train_2k contains train_mini whole (403 of 403), so a pool scored as it comes is
    # part in-sample. Scoring drops the mined episodes by id and says how many it dropped.
    mined_ids = {str(e.get("episode_id")) for e in episodes}
    report["held_out"] = {}
    for name in args.score_on:
        pool = load(name)
        rest = [e for e in pool if str(e.get("episode_id")) not in mined_ids]
        row = score(rest, kept)
        row["episodes_scored"] = len(rest)
        row["episodes_dropped_as_in_sample"] = len(pool) - len(rest)
        report["held_out"][name] = row

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=False))
    print("wrote %s" % args.out)
    print("  read %d episodes, %d with a DAG, %d ordered pairs"
          % (report["episodes_read"], report["episodes_with_dag"], report["edges_total"]))
    print("  patterns: %d seen, %d kept (support>=%d, precision>=%.2f)"
          % (len(report["rules"]), len(kept), MIN_SUPPORT, MIN_PRECISION))
    s = report["in_sample"]
    print("  in-sample : recall %d/%d = %.3f | false orderings %d | episodes fully recovered %d/%d"
          % (s["orderings_recalled"], s["orderings_total"], s["recall"], s["false_orderings"],
             s["episodes_fully_recovered"], s["episodes_with_dag"]))
    for name, h in report["held_out"].items():
        print("  %-10s: recall %d/%d = %.3f | false orderings %d | episodes fully recovered %d/%d"
              % (name, h["orderings_recalled"], h["orderings_total"], h["recall"],
                 h["false_orderings"], h["episodes_fully_recovered"], h["episodes_with_dag"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
