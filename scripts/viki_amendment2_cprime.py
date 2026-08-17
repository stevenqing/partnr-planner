#!/usr/bin/env python3

import argparse
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT.parent / "VIKI-R"
DATA_ROOT = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2"
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment2"
M0_SUMMARY = OUTPUT_DIR / "m0.summary.json"
M0_INSTANCES = OUTPUT_DIR / "m0_instances.jsonl"
C0_CENSUS = (
    ROOT / "results/viki_memory_experiments/amendment1/c0_composition_census.parquet"
)
MANIFEST_PATH = OUTPUT_DIR / "cprime_manifest.parquet"
AUDIT_PATH = OUTPUT_DIR / "cprime_candidate_leakage_audit.parquet"
SEED = 20260814
CHANNEL_CAP = 400
PAIR_MIN_ROWS = 30
UNIT_MIN_INSTANCES = 30


class GateFailure(RuntimeError):
    pass


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if hasattr(value, "tolist"):
        return native(value.tolist())
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def require_m0() -> None:
    if not M0_SUMMARY.is_file() or not M0_INSTANCES.is_file():
        raise GateFailure("C-prime requires M0 artifacts")
    summary = json.loads(M0_SUMMARY.read_text())
    if summary.get("gate", {}).get("status") != "PASS":
        raise GateFailure("C-prime blocked by failed GATE M0")


def load_instances() -> List[Dict[str, Any]]:
    with M0_INSTANCES.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def parse_json_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return list(json.loads(value))
    return list(value)


def unordered_pairs(values: Iterable[str]) -> Set[Tuple[str, str]]:
    return set(itertools.combinations(sorted(set(values)), 2))


def seeded_cap(indices: Sequence[int], offset: int) -> List[int]:
    values = sorted(set(int(index) for index in indices))
    if len(values) <= CHANNEL_CAP:
        return values
    return sorted(random.Random(SEED + offset).sample(values, CHANNEL_CAP))


def grounded_plan_tokens(ground_truth: Mapping[str, Any]) -> List[str]:
    tokens = []
    for step in native(ground_truth)["time_steps"]:
        for robot, action in sorted(step["actions"].items()):
            if action is not None:
                tokens.append(":".join([str(robot), *map(str, action)]))
    return tokens


def demo_tokens(demo: Sequence[Mapping[str, Any]]) -> List[str]:
    return grounded_plan_tokens({"time_steps": demo})


def levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def ngrams(tokens: Sequence[str], size: int = 3) -> Set[Tuple[str, ...]]:
    if not tokens:
        return set()
    if len(tokens) < size:
        return {tuple(tokens)}
    return {
        tuple(tokens[position : position + size])
        for position in range(len(tokens) - size + 1)
    }


def ngram_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_ngrams = ngrams(left)
    right_ngrams = ngrams(right)
    union = left_ngrams | right_ngrams
    return len(left_ngrams & right_ngrams) / len(union) if union else 1.0


def covers_full_signature(instance_units: Sequence[str], test_signature: str) -> bool:
    test_units = list(json.loads(test_signature)["ordered_units"])
    if not test_units or len(instance_units) < len(test_units):
        return False
    width = len(test_units)
    return any(
        list(instance_units[position : position + width]) == test_units
        for position in range(len(instance_units) - width + 1)
    )


def coverage_counts(instances: Sequence[Mapping[str, Any]]) -> Counter:
    return Counter(
        str(instance["unit_kinds"][0])
        for instance in instances
        if len(instance.get("ordered_units", [])) == 1
        and len(instance.get("unit_kinds", [])) == 1
    )


def pool_audit(
    channel: str,
    test_row: Mapping[str, Any],
    instances: Sequence[Mapping[str, Any]],
    test_tokens: Sequence[str],
) -> Dict[str, Any]:
    full_cover_ids = [
        str(instance["instance_id"])
        for instance in instances
        if covers_full_signature(
            instance.get("ordered_units", []), test_row["full_signature"]
        )
    ]
    distances = []
    overlaps = []
    for instance in instances:
        tokens = demo_tokens(instance["demo"])
        distances.append(
            (levenshtein(tokens, test_tokens), str(instance["instance_id"]))
        )
        overlaps.append(
            (ngram_jaccard(tokens, test_tokens), str(instance["instance_id"]))
        )
    nearest_distance = min(distances) if distances else (None, None)
    maximum_overlap = max(overlaps) if overlaps else (None, None)
    return {
        "channel": channel,
        "index": int(test_row["index"]),
        "candidate_instances": len(instances),
        "full_signature_cover_count": len(full_cover_ids),
        "full_signature_cover_ids": json.dumps(full_cover_ids, separators=(",", ":")),
        "min_plan_edit_distance": nearest_distance[0],
        "nearest_edit_instance_id": nearest_distance[1],
        "max_plan_trigram_jaccard": maximum_overlap[0],
        "nearest_trigram_instance_id": maximum_overlap[1],
    }


def prepare() -> Dict[str, Any]:
    require_m0()
    if not C0_CENSUS.is_file():
        raise GateFailure(f"Missing frozen C0 census: {C0_CENSUS}")
    instances = load_instances()
    census = pd.read_parquet(C0_CENSUS)
    train = census[census["split"] == "train"].copy()
    test = census[census["split"] == "test"].copy()
    if len(train) != 7196 or len(test) != 1800:
        raise GateFailure(
            f"C-prime census coverage mismatch: train={len(train)}, test={len(test)}"
        )
    train["unit_kinds_value"] = train["unit_kinds"].map(parse_json_list)
    test["unit_kinds_value"] = test["unit_kinds"].map(parse_json_list)
    test["unit_count"] = test["unit_kinds_value"].map(len)
    train_by_index = train.set_index("index")
    test_by_index = test.set_index("index")

    pair_counts: Counter[Tuple[str, str]] = Counter()
    test_pairs: Dict[int, Set[Tuple[str, str]]] = {}
    for index, row in test_by_index.iterrows():
        pairs = unordered_pairs(row["unit_kinds_value"])
        test_pairs[int(index)] = pairs
        pair_counts.update(pairs)
    heldout_pairs = {
        pair for pair, count in pair_counts.items() if count >= PAIR_MIN_ROWS
    }

    single_instances = [
        instance
        for instance in instances
        if len(instance.get("ordered_units", [])) == 1
        and len(instance.get("unit_kinds", [])) == 1
    ]
    productivity_counts = coverage_counts(single_instances)
    productivity_covered = {
        kind
        for kind, count in productivity_counts.items()
        if count >= UNIT_MIN_INSTANCES
    }
    systematicity_instances = []
    for instance in single_instances:
        source_index = int(instance["source_train_index"])
        source_pairs = unordered_pairs(
            train_by_index.loc[source_index, "unit_kinds_value"]
        )
        if not (source_pairs & heldout_pairs):
            systematicity_instances.append(instance)
    systematicity_counts = coverage_counts(systematicity_instances)
    systematicity_covered = {
        kind
        for kind, count in systematicity_counts.items()
        if count >= UNIT_MIN_INSTANCES
    }

    task_instances: Dict[str, List[Dict[str, Any]]] = {}
    for instance in instances:
        source_index = int(instance["source_train_index"])
        task_name = str(train_by_index.loc[source_index, "task_name"])
        task_instances.setdefault(task_name, []).append(instance)
    instance_candidates = [
        int(index)
        for index, row in test_by_index.iterrows()
        if str(row["task_name"]) in task_instances
    ]
    multi_unit = test[test["unit_count"] > 1]
    productivity_before = [int(value) for value in multi_unit["index"]]
    productivity_after = [
        int(row["index"])
        for _, row in multi_unit.iterrows()
        if set(row["unit_kinds_value"]) <= productivity_covered
    ]
    systematicity_before = [
        int(row["index"])
        for _, row in multi_unit.iterrows()
        if test_pairs[int(row["index"])] & heldout_pairs
    ]
    systematicity_after = [
        int(row["index"])
        for _, row in multi_unit.iterrows()
        if test_pairs[int(row["index"])] & heldout_pairs
        and set(row["unit_kinds_value"]) <= systematicity_covered
    ]
    selected = {
        "instance": seeded_cap(instance_candidates, 1),
        "productivity": seeded_cap(productivity_after, 2),
        "systematicity": seeded_cap(systematicity_after, 3),
    }
    short_channels = {
        channel: len(indices)
        for channel, indices in selected.items()
        if len(indices) != CHANNEL_CAP
    }
    if short_channels:
        failure_summary = {
            "task": "C-prime-0",
            "seed": SEED,
            "channel_cap": CHANNEL_CAP,
            "pair_min_rows": PAIR_MIN_ROWS,
            "unit_min_instances": UNIT_MIN_INSTANCES,
            "gate": {
                "status": "FAIL",
                "reason": "covered signatures cannot form 400 rows per channel",
                "short_channels": short_channels,
                "generation_halted": True,
            },
            "heldout_pairs": {
                "+".join(pair): int(pair_counts[pair]) for pair in sorted(heldout_pairs)
            },
            "coverage": {
                "productivity": {
                    "single_unit_instances_by_kind": dict(productivity_counts),
                    "covered_unit_kinds": sorted(productivity_covered),
                },
                "systematicity": {
                    "single_unit_instances_by_kind": dict(systematicity_counts),
                    "covered_unit_kinds": sorted(systematicity_covered),
                },
            },
            "narrowing": {
                "productivity": {
                    "before": len(productivity_before),
                    "after": len(productivity_after),
                    "selected": len(selected["productivity"]),
                },
                "systematicity": {
                    "before": len(systematicity_before),
                    "after": len(systematicity_after),
                    "selected": len(selected["systematicity"]),
                },
            },
            "channel_samples": {
                channel: len(indices) for channel, indices in selected.items()
            },
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        atomic_json(MANIFEST_PATH.with_suffix(".summary.json"), failure_summary)
        raise GateFailure(
            f"GATE C-prime-0 cannot form 400 rows per channel: {short_channels}"
        )

    pools: Dict[str, Any] = {
        "productivity": single_instances,
        "systematicity": systematicity_instances,
    }
    test_dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    manifest_rows = []
    audit_rows = []
    for channel, indices in selected.items():
        for index in indices:
            row = test_by_index.loc[index]
            current_pool = (
                task_instances[str(row["task_name"])]
                if channel == "instance"
                else pools[channel]
            )
            instance_ids = sorted(str(item["instance_id"]) for item in current_pool)
            source_indices = sorted(
                {int(item["source_train_index"]) for item in current_pool}
            )
            manifest_rows.append(
                {
                    "channel": channel,
                    "test_split": "test",
                    "index": index,
                    "task_name": str(row["task_name"]),
                    "full_signature": str(row["full_signature"]),
                    "unit_kinds": json.dumps(
                        row["unit_kinds_value"], separators=(",", ":")
                    ),
                    "unit_count": int(row["unit_count"]),
                    "heldout_pairs": json.dumps(
                        [
                            "+".join(pair)
                            for pair in sorted(test_pairs[index] & heldout_pairs)
                        ],
                        separators=(",", ":"),
                    ),
                    "allowed_instance_ids": json.dumps(
                        instance_ids, separators=(",", ":")
                    ),
                    "allowed_instance_count": len(instance_ids),
                    "allowed_source_indices": json.dumps(
                        source_indices, separators=(",", ":")
                    ),
                    "allowed_source_count": len(source_indices),
                    "scorer_seed": index,
                }
            )
            if channel != "instance":
                ground_truth = native(test_dataset.iloc[index]["reward_model"])[
                    "ground_truth"
                ]
                audit_rows.append(
                    pool_audit(
                        channel,
                        {"index": index, "full_signature": row["full_signature"]},
                        current_pool,
                        grounded_plan_tokens(ground_truth),
                    )
                )

    manifest = pd.DataFrame(manifest_rows).sort_values(["channel", "index"])
    audit = pd.DataFrame(audit_rows).sort_values(["channel", "index"])
    strict_cover_rows = int((audit["full_signature_cover_count"] > 0).sum())
    if strict_cover_rows:
        raise GateFailure(
            f"C-prime instance leakage construction failed for {strict_cover_rows} rows"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(MANIFEST_PATH, index=False)
    manifest.to_csv(MANIFEST_PATH.with_suffix(".csv"), index=False)
    audit.to_parquet(AUDIT_PATH, index=False)
    audit.to_csv(AUDIT_PATH.with_suffix(".csv"), index=False)
    summary = {
        "task": "C-prime-0",
        "seed": SEED,
        "channel_cap": CHANNEL_CAP,
        "pair_min_rows": PAIR_MIN_ROWS,
        "unit_min_instances": UNIT_MIN_INSTANCES,
        "gate": {
            "status": "PASS",
            "strict_full_signature_cover_rows": strict_cover_rows,
        },
        "heldout_pairs": {
            "+".join(pair): int(pair_counts[pair]) for pair in sorted(heldout_pairs)
        },
        "coverage": {
            "productivity": {
                "single_unit_instances_by_kind": dict(productivity_counts),
                "covered_unit_kinds": sorted(productivity_covered),
            },
            "systematicity": {
                "single_unit_instances_by_kind": dict(systematicity_counts),
                "covered_unit_kinds": sorted(systematicity_covered),
            },
        },
        "narrowing": {
            "productivity": {
                "before": len(productivity_before),
                "after": len(productivity_after),
                "excluded": len(productivity_before) - len(productivity_after),
            },
            "systematicity": {
                "before": len(systematicity_before),
                "after": len(systematicity_after),
                "excluded": len(systematicity_before) - len(systematicity_after),
            },
        },
        "channels": {
            channel: {
                "samples": len(group),
                "multi_unit_rows": int((group["unit_count"] > 1).sum()),
                "allowed_instances_min": int(group["allowed_instance_count"].min()),
                "allowed_instances_max": int(group["allowed_instance_count"].max()),
            }
            for channel, group in manifest.groupby("channel")
        },
        "audit": {
            channel: {
                "samples": len(group),
                "full_signature_cover_rows": int(
                    (group["full_signature_cover_count"] > 0).sum()
                ),
                "min_edit_distance": int(group["min_plan_edit_distance"].min()),
                "max_trigram_jaccard": float(group["max_plan_trigram_jaccard"].max()),
            }
            for channel, group in audit.groupby("channel")
        },
        "manifest": str(MANIFEST_PATH),
        "candidate_leakage_audit": str(AUDIT_PATH),
    }
    atomic_json(MANIFEST_PATH.with_suffix(".summary.json"), summary)
    return summary


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VIKI Amendment 2 C-prime")
    parser.add_argument("command", choices=("prepare",))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        result = prepare() if args.command == "prepare" else None
    except GateFailure as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
