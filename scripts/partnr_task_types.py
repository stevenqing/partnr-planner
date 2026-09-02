#!/usr/bin/env python3
"""Classify PARTNR episodes by the constraints their propositions impose.

The four elementary types are read straight off the evaluation functions, the way the
Memory-as-Skill paper defines them: rearrangement is a location predicate, spatial is a
`next to` relation, heterogeneous is a state change, and temporal is a real edge in the
proposition DAG. A composite type is the sorted union, so `H_R_S_T` names an episode
that carries all four.

This is what picks the cell to work in -- memory built from Rearrange-only episodes,
evaluated on Rearrange+Spatial ones -- and it has to be computed before anything is
collected, since it decides which episodes are collected from. Reconstructing it
independently is also the check that the cell is the paper's cell and not a similar one:
three of the paper's reported episode counts fall out of this classification exactly.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REARRANGE = {"is_on_top", "is_inside", "is_in_room", "is_on_floor"}
SPATIAL = {"is_next_to", "is_clustered"}
HETEROGENEOUS = {"is_clean", "is_dirty", "is_powered_on", "is_powered_off",
                 "is_filled", "is_empty"}


def classify(episode: Dict[str, Any]) -> str:
    kinds = set()
    for proposition in episode.get("evaluation_propositions") or []:
        name = proposition.get("function_name", "")
        if name in REARRANGE:
            kinds.add("R")
        elif name in SPATIAL:
            kinds.add("S")
        elif name in HETEROGENEOUS:
            kinds.add("H")
    # Temporal is a real edge in the proposition DAG and nothing else. An episode can
    # also carry `evaluation_proposition_dependencies`, and counting those as temporal
    # too seemed natural until the counts were checked against the paper's: with DAG
    # edges alone, val_mini holds exactly 76 R_S episodes, val holds 11 H_R_T, and
    # train_2k holds 197 H_R -- the three numbers the paper reports. Counting
    # dependencies as well collapses R_S from 76 to a handful. The rule is fixed by
    # that agreement, not by taste.
    for constraint in episode.get("evaluation_constraints") or []:
        if constraint.get("type") != "TemporalConstraint":
            continue
        if constraint.get("args", {}).get("dag_edges"):
            kinds.add("T")
    return "_".join(sorted(kinds)) or "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/datasets/partnr_episodes/v0_0")
    parser.add_argument("--splits", nargs="*", default=["train_mini", "train_2k", "val_mini", "val"])
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    everything: Dict[str, Dict[str, List[Any]]] = {}
    for split in arguments.splits:
        path = Path(arguments.root) / f"{split}.json.gz"
        if not path.is_file():
            continue
        episodes = json.loads(gzip.open(path).read())["episodes"]
        by_type: Dict[str, List[Any]] = {}
        for episode in episodes:
            by_type.setdefault(classify(episode), []).append(episode["episode_id"])
        everything[split] = by_type
        counts = Counter({k: len(v) for k, v in by_type.items()})
        print(f"\n{split}  ({len(episodes)} episodes)")
        for kind, count in counts.most_common():
            print(f"  {kind:<10} {count:>5}  {count / len(episodes) * 100:5.1f}%")
    if arguments.out:
        Path(arguments.out).write_text(json.dumps(everything, indent=2) + "\n")
        print(f"\nwrote {arguments.out}")


if __name__ == "__main__":
    main()
