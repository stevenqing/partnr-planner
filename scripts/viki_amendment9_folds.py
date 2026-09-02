#!/usr/bin/env python3
"""Held-out family folds for Amendment 9.

VIKI-L2 ships no out-of-distribution split: all 14 test families appear in train
and every one of the 1800 test rows has a same-family neighbour in the memory bank.
Nearest-trajectory retrieval is close to optimal in that regime, so it cannot
separate a memory that abstracts from one that copies. These folds construct the
missing split: for each family present in the evaluation manifest, every training
episode of that family is removed from the bank and the family's own rows are then
evaluated against it.

The 924-row manifest covers 8 of the 14 families, so there are 8 folds. Each row
belongs to exactly one fold, so the folds together score the same 924 rows once.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment6 import ROOT, GateFailure
from viki_amendment8b import MEMORY_PARQUET, SOURCE_PARQUET, load_manifest, native

TRAIN_INTERACTIONS = (
    ROOT / "results/viki_memory_experiments/amendment7/train_interactions.jsonl"
)


@lru_cache(maxsize=1)
def train_family_by_index() -> Dict[int, str]:
    frame = pd.read_parquet(MEMORY_PARQUET)
    return {
        i: native(frame.iloc[i].to_dict())["reward_model"]["ground_truth"].get(
            "task_name"
        )
        for i in range(len(frame))
    }


@lru_cache(maxsize=1)
def episode_families() -> Dict[str, str]:
    """memory_id -> task family, via the source_train_index the record carries."""
    by_index = train_family_by_index()
    families: Dict[str, str] = {}
    with TRAIN_INTERACTIONS.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            source = record.get("source_train_index")
            if source is None or source not in by_index:
                raise GateFailure(
                    f"{record.get('memory_id')} has no usable source_train_index"
                )
            families[record["memory_id"]] = by_index[source]
    return families


@lru_cache(maxsize=1)
def eval_families() -> Dict[int, str]:
    frame = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()
    return {
        i: native(frame.iloc[i].to_dict())["reward_model"]["ground_truth"].get(
            "task_name"
        )
        for i in sorted(manifest)
    }


@lru_cache(maxsize=1)
def folds() -> List[str]:
    """Evaluation families, largest first, so the costly folds surface early."""
    counts: Dict[str, int] = {}
    for family in eval_families().values():
        counts[family] = counts.get(family, 0) + 1
    return [family for family, _ in sorted(counts.items(), key=lambda kv: -kv[1])]


def rows_of(family: str) -> List[int]:
    return sorted(i for i, name in eval_families().items() if name == family)


def held_out_ids(family: str) -> set:
    return {mid for mid, name in episode_families().items() if name == family}


def summary() -> Dict[str, object]:
    families = episode_families()
    counts: Dict[str, int] = {}
    for name in families.values():
        counts[name] = counts.get(name, 0) + 1
    return {
        "eval_families": len(folds()),
        "train_families": len(counts),
        "folds": [
            {
                "family": family,
                "eval_rows": len(rows_of(family)),
                "train_episodes_removed": counts.get(family, 0),
                "train_episodes_kept": len(families) - counts.get(family, 0),
            }
            for family in folds()
        ],
    }


if __name__ == "__main__":
    # --names prints one family per line for shell drivers; the default prints the
    # full summary. Keeping the machine-readable form explicit avoids a driver
    # parsing JSON out of a stream that also carries simulator banner text.
    if "--names" in sys.argv:
        print("\n".join(folds()))
    else:
        print(json.dumps(summary(), indent=2))
