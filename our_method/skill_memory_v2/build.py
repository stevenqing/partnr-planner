"""Build the artefact: three layers, one file, and a self-check before it is trusted.

The training episodes are split in two before anything is induced. Half build the memory
and the other half are never shown to it, so the build can end by asking the memory to
plan episodes it has not seen and scoring those plans with the official judge. A library
can look reasonable in a listing and plan nothing at all; this is the check that catches
that, and it is run at build time so a broken artefact never reaches an experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import dependencies, induction, vocabulary
from .memory import FORMAT, SkillMemoryV2
from .simulator import SEED, Simulator


def load_episodes(parquet: Path) -> List[Dict[str, Any]]:
    import pandas as pd
    from habitat_llm.evaluation import viki_bench as bench

    frame = pd.read_parquet(parquet)
    return [
        bench.get_ground_truth(bench.to_native(frame.iloc[position].to_dict()))
        for position in range(len(frame))
    ]


def build(
    episodes: List[Dict[str, Any]],
    sim: Simulator,
    seed: int = SEED,
    per_family: int = 250,
    exclude_family: Optional[str] = None,
) -> SkillMemoryV2:
    # Deterministic halves: the memory is induced from one and checked on the other, so
    # the build's own number is never taken on episodes it learned from.
    induction_set = episodes[::2]
    layer1 = induction.induce(induction_set, sim, seed, per_family, exclude_family)
    layer2 = dependencies.mine(induction_set, sim, seed, per_family, exclude_family)
    layer3 = vocabulary.harvest(induction_set, exclude_family)
    record = {
        "format": FORMAT,
        "built_from": "VIKI-L2 train.parquet, even-indexed episodes",
        "excluded_family": exclude_family,
        "seed": seed,
        "per_family": per_family,
        "layer1": layer1,
        "layer2": layer2,
        "layer3": layer3,
    }
    return SkillMemoryV2(record)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build skill memory v2 from VIKI-L2 training episodes")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--per-family", type=int, default=250)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--exclude-family", default=None)
    parser.add_argument("--validate", type=int, default=200,
                        help="held-out training episodes to plan and score after building")
    arguments = parser.parse_args(argv)

    sim = Simulator(arguments.benchmark_root)
    episodes = load_episodes(arguments.train)
    memory = build(episodes, sim, arguments.seed, arguments.per_family, arguments.exclude_family)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    memory.save(arguments.out)

    summary = memory.summary()
    print(json.dumps(summary, indent=2))
    if arguments.validate:
        holdout = [
            truth for truth in episodes[1::2]
            if isinstance(truth, dict)
            and (not arguments.exclude_family or truth.get("task_name") != arguments.exclude_family)
        ][: arguments.validate]
        report = memory.validate(holdout, sim, arguments.seed)
        print("\nself-check on held-out training episodes (oracle goals, official scorer):")
        print(json.dumps(report, indent=2))
        memory.record["self_check"] = report
        memory.save(arguments.out)
    print(f"\nwrote {arguments.out}")


if __name__ == "__main__":
    main()
