#!/usr/bin/env python3

import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CENSUS = (
    ROOT / "results/viki_memory_experiments/amendment1/c0_composition_census.parquet"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment1"
SEED = 20260814
CHANNEL_CAP = 400
PAIR_MIN_ROWS = 30


def unit_kinds(row: pd.Series) -> List[str]:
    return list(json.loads(row["unit_kinds"]))


def unordered_pairs(values: Iterable[str]) -> Set[Tuple[str, str]]:
    unique = sorted(set(values))
    return set(itertools.combinations(unique, 2))


def seeded_cap(indices: Sequence[int], cap: int, seed_offset: int) -> List[int]:
    values = sorted(int(index) for index in indices)
    if len(values) <= cap:
        return values
    rng = random.Random(SEED + seed_offset)
    return sorted(rng.sample(values, cap))


def manifest_rows(
    channel: str,
    test_indices: Sequence[int],
    memory_indices: Any,
    test: pd.DataFrame,
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    test_by_index = test.set_index("index")
    for index in sorted(test_indices):
        row = test_by_index.loc[index]
        row_memory_indices = (
            memory_indices[int(index)]
            if isinstance(memory_indices, dict)
            else memory_indices
        )
        memory_json = json.dumps(sorted(row_memory_indices), separators=(",", ":"))
        rows.append(
            {
                "channel": channel,
                "test_split": "test",
                "test_index": int(index),
                "task_name": row["task_name"],
                "full_signature": row["full_signature"],
                "unit_kinds": row["unit_kinds"],
                "memory_split": "train",
                "allowed_memory_indices": memory_json,
                "allowed_memory_count": len(row_memory_indices),
                "channel_metadata": json.dumps(
                    metadata, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    return rows


def build_splits(args: argparse.Namespace) -> Dict[str, Any]:
    census = pd.read_parquet(args.census)
    train = census[census["split"] == "train"].copy()
    test = census[census["split"] == "test"].copy()
    if len(train) != 7196 or len(test) != 1800:
        raise ValueError(
            f"C1 input coverage mismatch: train={len(train)}, test={len(test)}"
        )
    train["unit_count"] = train.apply(lambda row: len(unit_kinds(row)), axis=1)
    test["unit_count"] = test.apply(lambda row: len(unit_kinds(row)), axis=1)

    rows = []
    episode_candidates = []
    episode_memory_by_test: Dict[int, List[int]] = {}
    for task, test_group in test.groupby("task_name"):
        train_group = train[train["task_name"] == task]
        if train_group.empty:
            continue
        task_memory = sorted(int(index) for index in train_group["index"])
        for index in test_group["index"]:
            test_index = int(index)
            episode_candidates.append(test_index)
            episode_memory_by_test[test_index] = task_memory
    episode_test = seeded_cap(episode_candidates, CHANNEL_CAP, 1)
    rows.extend(
        manifest_rows(
            "episode_heldout",
            episode_test,
            episode_memory_by_test,
            test,
            {"seed": SEED, "cap": CHANNEL_CAP},
        )
    )

    productivity_memory = sorted(
        int(index) for index in train.loc[train["unit_count"] == 1, "index"]
    )
    productivity_candidates = [
        int(index) for index in test.loc[test["unit_count"] > 1, "index"]
    ]
    productivity_test = seeded_cap(productivity_candidates, CHANNEL_CAP, 2)
    rows.extend(
        manifest_rows(
            "task_heldout_productivity",
            productivity_test,
            productivity_memory,
            test,
            {
                "seed": SEED,
                "cap": CHANNEL_CAP,
                "memory_rule": "unit_count == 1",
                "test_rule": "unit_count > 1",
            },
        )
    )

    pair_counts: Counter[Tuple[str, str]] = Counter()
    row_pairs: Dict[int, Set[Tuple[str, str]]] = {}
    for _, row in test.iterrows():
        pairs = unordered_pairs(unit_kinds(row))
        row_pairs[int(row["index"])] = pairs
        pair_counts.update(pairs)
    heldout_pairs = {
        pair for pair, count in pair_counts.items() if count >= PAIR_MIN_ROWS
    }
    systematicity_candidates = [
        index for index, pairs in row_pairs.items() if pairs & heldout_pairs
    ]
    systematicity_test = seeded_cap(systematicity_candidates, CHANNEL_CAP, 3)
    systematicity_memory = []
    for _, row in train.iterrows():
        if not (unordered_pairs(unit_kinds(row)) & heldout_pairs):
            systematicity_memory.append(int(row["index"]))
    heldout_pair_strings = ["+".join(pair) for pair in sorted(heldout_pairs)]
    rows.extend(
        manifest_rows(
            "task_heldout_systematicity",
            systematicity_test,
            systematicity_memory,
            test,
            {
                "seed": SEED,
                "cap": CHANNEL_CAP,
                "pair_min_rows": PAIR_MIN_ROWS,
                "heldout_pairs": heldout_pair_strings,
            },
        )
    )

    manifest = pd.DataFrame(rows)
    leakage = {}
    train_signatures = train.set_index("index")["full_signature"].to_dict()
    for channel, group in manifest.groupby("channel"):
        overlap_count = 0
        memory_counts = []
        for _, row in group.iterrows():
            memory_indices = json.loads(row["allowed_memory_indices"])
            memory_counts.append(len(memory_indices))
            memory_signatures = {
                train_signatures[int(index)] for index in memory_indices
            }
            overlap_count += row["full_signature"] in memory_signatures
        leakage[channel] = {
            "test_rows": len(group),
            "memory_rows_min": min(memory_counts),
            "memory_rows_max": max(memory_counts),
            "full_signature_overlap_rows": int(overlap_count),
        }
        if channel.startswith("task_heldout") and overlap_count:
            raise ValueError(
                f"C1 leakage: {channel} has {overlap_count} signature overlaps"
            )

    summary = {
        "task": "C1",
        "seed": SEED,
        "channel_cap": CHANNEL_CAP,
        "pair_min_rows": PAIR_MIN_ROWS,
        "channels": leakage,
        "productivity": {
            "single_unit_train_rows": len(productivity_memory),
            "single_unit_train_tasks": {
                str(key): int(value)
                for key, value in train.loc[train["unit_count"] == 1, "task_name"]
                .value_counts()
                .items()
            },
            "multi_unit_id_rows": len(productivity_candidates),
            "degenerate": bool(
                len(productivity_memory) > 0
                and train.loc[train["unit_count"] == 1, "task_name"].nunique() == 1
            ),
        },
        "systematicity": {
            "heldout_pairs": {
                "+".join(pair): int(pair_counts[pair]) for pair in sorted(heldout_pairs)
            },
            "candidate_id_rows": len(systematicity_candidates),
            "allowed_train_rows": len(systematicity_memory),
            "allowed_train_tasks": {
                str(key): int(value)
                for key, value in train[train["index"].isin(systematicity_memory)][
                    "task_name"
                ]
                .value_counts()
                .items()
            },
            "degenerate": len(systematicity_memory) == len(productivity_memory),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "c1_split_manifest.parquet"
    csv_path = args.output_dir / "c1_split_manifest.csv"
    summary_path = args.output_dir / "c1_split_manifest.summary.json"
    manifest.to_parquet(parquet_path, index=False)
    manifest.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "parquet": str(parquet_path),
                "csv": str(csv_path),
                "summary": str(summary_path),
                **summary,
            },
            indent=2,
        )
    )
    return summary


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict VIKI C1 splits")
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build_splits(parse_args())
