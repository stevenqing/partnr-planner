#!/usr/bin/env python3
"""Is the ported MEMENTO retrieval finding the right memories, or is the port broken?

Reporting a low number for somebody else's method is only honest if the method was given
a working chance. This re-runs the retrieval on the answers already collected and asks
the one question that separates the two explanations: of the episodic memories it pulled
back, how many come from the same task family as the row it was retrieving for. A high
rate means retrieval works and the score is about what the model does with what it got;
a low rate means the query and the index do not speak the same language, and the port
needs fixing before any number is quoted.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_memento_rag import MementoRAG, build
from our_method.skill_memory_v2 import Simulator
from our_method.skill_memory_v2.build import load_episodes

OUT = ROOT / "results/viki_memory_experiments/amendment11"
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "memento_smoke2"


def main() -> None:
    sim = Simulator(BENCHMARK_ROOT)
    episodes = load_episodes(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/train.parquet")
    graph = build(episodes[::2], set(sim.viki2.CONTAINER_ASSETS))
    rag = MementoRAG(graph)
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    records = [json.loads(l) for l in (OUT / f"{SOURCE}.jsonl").read_text().splitlines() if l.strip()]

    hits, seen, empty = 0, 0, 0
    by_query = Counter()
    shown = 0
    for record in records:
        truth = bench.get_ground_truth(bench.to_native(frame.iloc[record["index"]].to_dict()))
        extracted = record.get("extracted") or {}
        by_query["object_semantics terms"] += len(extracted.get("object_semantics") or [])
        by_query["user_pattern terms"] += len(extracted.get("user_pattern") or [])
        retrieved = rag.retrieve(extracted)
        patterns = retrieved.get("user_pattern") or []
        if not patterns:
            empty += 1
            continue
        same = sum(1 for i in patterns if rag.nodes[i]["entity"] == truth["task_name"])
        hits += same
        seen += len(patterns)
        if shown < 3:
            shown += 1
            print("=" * 74)
            print(f"[{truth['task_name']}] score={record.get('score')}")
            print(f"  instruction : {(truth.get('description') or '')[:110]}")
            print(f"  extracted   : {json.dumps(extracted)[:200]}")
            print(f"  retrieved   : {same}/{len(patterns)} from the right family")
            for i in patterns[:2]:
                print(f"    - [{rag.nodes[i]['entity']}] {rag.nodes[i]['knowledge'][:90]}")

    print(f"\nrows                                 {len(records)}")
    print(f"rows with no pattern retrieved       {empty}")
    print(f"retrieved episodic memories          {seen}")
    print(f"  from the row's own family          {hits}  ({hits / max(1, seen) * 100:.1f}%)")
    print(f"extraction yield                     {dict(by_query)}")


if __name__ == "__main__":
    main()
