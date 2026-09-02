#!/usr/bin/env python3
"""Which family does G-Memory actually retrieve from once a fold masks its own?

The shuffled control shows retrieval is worth about eleven points on the folds, yet
three structural analyses found nothing in the permitted bank that shares the row's
plan or its object-typed coordination template. Both can hold if the family
partition is finer than the task: cut_fruit_on_board is held out while
cut_two_fruits_on_board stays, and the two are nearly the same task. Then the fold
removes a label, not a distribution, and what retrieval finds is a sibling.

This reads the retrieved record for every fold row -- no model calls, since only the
ranking is needed, not G-Memory's choice between its top two -- and reports which
family it came from and how close its instruction is.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import numpy as np
import pandas as pd

from habitat_llm.evaluation.viki_branch_conditions import get_instruction
from viki_amendment8b import OUTPUT_DIR, SOURCE_PARQUET, load_manifest, native
from viki_amendment8_memory import _encode
from viki_amendment9_diag_content import OfflineGMemory
from viki_amendment9_folds import episode_families, folds, rows_of


def main() -> None:
    manifest = load_manifest()
    test = pd.read_parquet(SOURCE_PARQUET)
    by_episode = episode_families()

    tally: Counter = Counter()
    similarity = defaultdict(list)
    for family in folds():
        rows = [i for i in rows_of(family) if i in manifest]
        theirs = OfflineGMemory(family)
        model = theirs.model
        for index in rows:
            sample = native(test.iloc[index].to_dict())
            instruction = get_instruction(sample)
            query = _encode(model, [instruction])[0]
            pool = theirs.state.raw_success_candidates(
                query, count=len(theirs.state.records)
            )
            best = next(p for p in pool if p not in theirs.blocked)
            record = theirs.state.records[best]
            source = by_episode.get(record.get("memory_id"), "unknown")
            tally[(family, source)] += 1
            other = _encode(model, [str(record.get("task_description", ""))])[0]
            similarity[family].append(
                float(
                    np.dot(query, other)
                    / (np.linalg.norm(query) * np.linalg.norm(other) + 1e-9)
                )
            )

    print("Held-out family -> the family its top permitted neighbour comes from:")
    for family in folds():
        picks = Counter(
            {src: n for (fam, src), n in tally.items() if fam == family}
        )
        total = sum(picks.values()) or 1
        top = ", ".join(
            f"{src} {100*n/total:.0f}%" for src, n in picks.most_common(3)
        )
        print(
            f"  {family:46s} sim={median(similarity[family]):.3f}  {top}"
        )

    out = OUTPUT_DIR / "folds" / "sibling_retrieval.json"
    out.write_text(
        json.dumps(
            {f"{fam}->{src}": n for (fam, src), n in tally.items()}, indent=2
        )
    )
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
