#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from viki_amendment3_f2 import (
    F2Provider,
    SymmetricDrop,
    exact_mcnemar_p,
    file_sha256,
    fingerprint,
    load_bench,
    load_certified_cprime_preflight,
    messages_sha256,
    require_frozen_args,
    server_metadata,
    validate_server_metadata,
)

from habitat_llm.evaluation.viki_branch_conditions import get_instruction
from habitat_llm.evaluation.viki_composition import parse_composition
from habitat_llm.evaluation.viki_memory_skill import add_memory_to_messages
from habitat_llm.evaluation.viki_segment_memory import (
    RetrievedGroup,
    RetrievedSegment,
    SegmentInstance,
    format_grouped_memory,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT.parent / "VIKI-R"
DATA_ROOT = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2"
AMENDMENT1_DIR = ROOT / "results/viki_memory_experiments/amendment1"
AMENDMENT2_DIR = ROOT / "results/viki_memory_experiments/amendment2"
AMENDMENT3_DIR = ROOT / "results/viki_memory_experiments/amendment3"
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment4"
CENSUS_PATH = AMENDMENT1_DIR / "c0_composition_census.parquet"
INSTANCES_PATH = AMENDMENT2_DIR / "m0_instances.jsonl"
MANIFEST_PATH = AMENDMENT3_DIR / "f2_cprime_manifest.parquet"
C_PRIME_CHANNELS = ("instance", "productivity")
PRIMITIVE_KINDS = ("fetch", "relocate", "state_change")
TOKEN_TOLERANCE = 0.05
G2_CHANNEL = "productivity"
G2_ROWS = 400
G2_PREFLIGHT_PATH = OUTPUT_DIR / "g2_clean_flat.preflight.jsonl"
G2_OUTPUT_PATH = OUTPUT_DIR / "g2_clean_flat.jsonl"
G2B_PREFLIGHT_PATH = OUTPUT_DIR / "g2b_segment_flat.preflight.jsonl"
G2B_OUTPUT_PATH = OUTPUT_DIR / "g2b_segment_flat.jsonl"
G2B_POSTTRIM_RENDER_PATH = OUTPUT_DIR / "g2b_posttrim_render_gate.parquet"
GENERATION_CLOSED_PATH = OUTPUT_DIR / "VIKI_GENERATION_CLOSED.json"
SEED = 20260814
BOOTSTRAP_DRAWS = 100000


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


def write_frame(frame: pd.DataFrame, stem: str) -> Dict[str, str]:
    parquet = OUTPUT_DIR / f"{stem}.parquet"
    csv = OUTPUT_DIR / f"{stem}.csv"
    frame.to_parquet(parquet, index=False)
    frame.to_csv(csv, index=False)
    return {"parquet": str(parquet), "csv": str(csv)}


def require_inputs() -> None:
    required = [CENSUS_PATH, INSTANCES_PATH, MANIFEST_PATH]
    for channel in C_PRIME_CHANNELS:
        required.extend(
            [
                AMENDMENT3_DIR / f"f2_cprime_{channel}.jsonl",
                AMENDMENT3_DIR / f"f2_cprime_{channel}.summary.json",
            ]
        )
    required.extend(
        [
            AMENDMENT3_DIR / "f2_ood.jsonl",
            AMENDMENT3_DIR / "f2_ood.summary.json",
        ]
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise GateFailure("Missing Amendment 4 inputs: " + ", ".join(missing))
    for name in ("f2_cprime_instance", "f2_cprime_productivity", "f2_ood"):
        summary = json.loads((AMENDMENT3_DIR / f"{name}.summary.json").read_text())
        if summary.get("status") != "PASS":
            raise GateFailure(f"Amendment 4 requires passing input: {name}")


def load_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        index = int(record["index"])
        if index in records:
            raise GateFailure(f"Duplicate index in {path}:{line_number}: {index}")
        records[index] = record
    return records


def parse_json_list(value: Any) -> List[Any]:
    return list(json.loads(value)) if isinstance(value, str) else list(value)


def grounded_plan_tokens(ground_truth: Mapping[str, Any]) -> List[str]:
    tokens = []
    for step in native(ground_truth)["time_steps"]:
        for robot, action in sorted(step["actions"].items()):
            if action is not None:
                tokens.append(":".join([str(robot), *map(str, action)]))
    return tokens


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


def covers_full_signature(source_units: Sequence[str], test_signature: str) -> bool:
    test_units = list(json.loads(test_signature)["ordered_units"])
    if not test_units or len(source_units) < len(test_units):
        return False
    width = len(test_units)
    return any(
        list(source_units[position : position + width]) == test_units
        for position in range(len(source_units) - width + 1)
    )


def paired_summary(frame: pd.DataFrame, left: str, right: str) -> Dict[str, Any]:
    left_success = frame[f"{left}_success"].astype(bool)
    right_success = frame[f"{right}_success"].astype(bool)
    return {
        "samples": len(frame),
        f"{left}_successes": int(left_success.sum()),
        f"{right}_successes": int(right_success.sum()),
        f"{left}_accuracy": float(left_success.mean()) if len(frame) else None,
        f"{right}_accuracy": float(right_success.mean()) if len(frame) else None,
        "right_minus_left": (
            float(right_success.mean() - left_success.mean()) if len(frame) else None
        ),
        "left_only": int((left_success & ~right_success).sum()),
        "right_only": int((~left_success & right_success).sum()),
    }


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for position, right_value in enumerate(right, 1):
            current.append(
                previous[position - 1] + 1
                if left_value == right_value
                else max(previous[position], current[-1])
            )
        previous = current
    return previous[-1]


def multiset_f1(left: Sequence[str], right: Sequence[str]) -> float:
    left_counts = Counter(left)
    right_counts = Counter(right)
    overlap = sum((left_counts & right_counts).values())
    denominator = len(left) + len(right)
    return 2 * overlap / denominator if denominator else 1.0


def classify_subgoal(value: str) -> Optional[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    tokens = set(normalized.split("_"))
    state_words = {
        "activate",
        "check",
        "clean",
        "close",
        "cut",
        "handover",
        "inspect",
        "interact",
        "open",
        "push",
        "slice",
        "survey",
        "toast",
        "turn",
        "wash",
    }
    if tokens & state_words:
        return "state_change"
    relocate_words = {
        "bring",
        "deliver",
        "load",
        "place",
        "put",
        "relocate",
        "serve",
        "set",
        "store",
        "transfer",
    }
    if tokens & relocate_words:
        return "relocate"
    if normalized.startswith("move_") and not normalized.startswith("move_to_"):
        return "relocate"
    fetch_words = {"fetch", "find", "get", "grab", "grasp", "pick", "retrieve", "take"}
    if tokens & fetch_words:
        return "fetch"
    return None


def load_sources() -> Tuple[Dict[str, Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    instances = [
        json.loads(line)
        for line in INSTANCES_PATH.read_text().splitlines()
        if line.strip()
    ]
    by_id = {str(instance["instance_id"]): instance for instance in instances}
    by_source: Dict[int, List[Dict[str, Any]]] = {}
    for instance in instances:
        by_source.setdefault(int(instance["source_train_index"]), []).append(instance)
    return by_id, by_source


def flat_instance_block(number: int, instance: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"{number}. Context: {instance['context']}",
            "   Availability: "
            + json.dumps(instance["self_cond"], sort_keys=True, ensure_ascii=True),
            "   Demonstration: "
            + json.dumps(instance["demo"], separators=(",", ":"), ensure_ascii=True),
        ]
    )


def injected_whole_source_rows(
    flat: Mapping[str, Any],
    instances_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[int], List[int]]:
    prompt = str(flat["prompt"])
    selected_ids = [
        str(value) for value in flat["candidate_instance_ids_before_prefix_trim"]
    ]
    by_source: Dict[int, List[bool]] = {}
    for number, instance_id in enumerate(selected_ids, 1):
        instance = instances_by_id[instance_id]
        source = int(instance["source_train_index"])
        present = flat_instance_block(number, instance) in prompt
        by_source.setdefault(source, []).append(present)
    whole = [source for source, present in by_source.items() if all(present)]
    partial = [
        source
        for source, present in by_source.items()
        if any(present) and not all(present)
    ]
    return whole, partial


def run_g0(
    manifest: pd.DataFrame,
    census: pd.DataFrame,
    test_dataset: pd.DataFrame,
    train_dataset: pd.DataFrame,
    instances_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    train_census = census[census["split"] == "train"].set_index("index")
    test_census = census[census["split"] == "test"].set_index("index")
    row_records = []
    exemplar_records = []
    pool_records = []
    for channel in C_PRIME_CHANNELS:
        channel_manifest = manifest[manifest["channel"] == channel].sort_values("index")
        records = load_jsonl(AMENDMENT3_DIR / f"f2_cprime_{channel}.jsonl")
        for manifest_row in channel_manifest.to_dict("records"):
            index = int(manifest_row["index"])
            record = records[index]
            selected_ids = [
                str(value)
                for value in record["flat"]["candidate_instance_ids_before_prefix_trim"]
            ]
            pretrim_source_indices = list(
                dict.fromkeys(
                    int(instances_by_id[instance_id]["source_train_index"])
                    for instance_id in selected_ids
                )
            )
            source_indices, partial_source_indices = injected_whole_source_rows(
                record["flat"], instances_by_id
            )
            test_signature = str(test_census.loc[index, "full_signature"])
            test_tokens = grounded_plan_tokens(
                native(test_dataset.iloc[index]["reward_model"])["ground_truth"]
            )
            covering = 0
            multi_unit = 0
            maximum_overlap = 0.0
            for rank, source_index in enumerate(source_indices, 1):
                source_signature = str(train_census.loc[source_index, "full_signature"])
                source_units = parse_json_list(
                    train_census.loc[source_index, "ordered_units"]
                )
                source_kinds = parse_json_list(
                    train_census.loc[source_index, "unit_kinds"]
                )
                is_covering = covers_full_signature(source_units, test_signature)
                is_multi_unit = len(source_units) > 1
                source_tokens = grounded_plan_tokens(
                    native(train_dataset.iloc[source_index]["reward_model"])[
                        "ground_truth"
                    ]
                )
                overlap = ngram_jaccard(source_tokens, test_tokens)
                covering += int(is_covering)
                multi_unit += int(is_multi_unit)
                maximum_overlap = max(maximum_overlap, overlap)
                exemplar_records.append(
                    {
                        "channel": channel,
                        "index": index,
                        "exemplar_rank": rank,
                        "source_train_index": source_index,
                        "source_full_signature": source_signature,
                        "source_ordered_units": json.dumps(
                            source_units, separators=(",", ":")
                        ),
                        "source_unit_kinds": json.dumps(
                            source_kinds, separators=(",", ":")
                        ),
                        "source_is_multi_unit": is_multi_unit,
                        "covers_test_full_signature": is_covering,
                        "grounded_plan_trigram_jaccard": overlap,
                    }
                )
            flat_success = float(record["arms"]["flat_memory"]["task_score"]) == 1
            skill_success = float(record["arms"]["skill_memory"]["task_score"]) == 1
            row_records.append(
                {
                    "channel": channel,
                    "index": index,
                    "test_full_signature": test_signature,
                    "source_train_indices_before_token_trim": json.dumps(
                        pretrim_source_indices, separators=(",", ":")
                    ),
                    "injected_whole_source_train_indices": json.dumps(
                        source_indices, separators=(",", ":")
                    ),
                    "partially_injected_source_train_indices": json.dumps(
                        partial_source_indices, separators=(",", ":")
                    ),
                    "exemplar_count": len(source_indices),
                    "pretrim_source_count": len(pretrim_source_indices),
                    "partially_injected_source_count": len(partial_source_indices),
                    "multi_unit_exemplar_count": multi_unit,
                    "covering_exemplar_count": covering,
                    "has_covering_exemplar": covering > 0,
                    "maximum_grounded_plan_trigram_jaccard": maximum_overlap,
                    "flat_success": flat_success,
                    "skill_success": skill_success,
                    "flat_success_skill_failure": flat_success and not skill_success,
                }
            )
            allowed_ids = parse_json_list(manifest_row["allowed_instance_ids"])
            allowed_sources = parse_json_list(manifest_row["allowed_source_indices"])
            pool_records.append(
                {
                    "channel": channel,
                    "index": index,
                    "allowed_instance_count_field": int(
                        manifest_row["allowed_instance_count"]
                    ),
                    "allowed_instance_ids_length": len(allowed_ids),
                    "allowed_source_count_field": int(
                        manifest_row["allowed_source_count"]
                    ),
                    "allowed_source_indices_length": len(allowed_sources),
                    "allowed_source_min": min(allowed_sources),
                    "allowed_source_max": max(allowed_sources),
                }
            )
    rows = pd.DataFrame(row_records).sort_values(["channel", "index"])
    exemplars = pd.DataFrame(exemplar_records).sort_values(
        ["channel", "index", "exemplar_rank"]
    )
    pools = pd.DataFrame(pool_records).sort_values(["channel", "index"])
    artifacts = {
        "rows": write_frame(rows, "g0_flat_prompt_rows"),
        "exemplars": write_frame(exemplars, "g0_flat_prompt_exemplars"),
        "pools": write_frame(pools, "g0_manifest_pools"),
    }
    by_channel: Dict[str, Dict[str, Any]] = {}
    for channel, group in rows.groupby("channel", sort=True):
        discordant = group[group["flat_success_skill_failure"]]
        pool_group = pools[pools["channel"] == channel]
        by_channel[str(channel)] = {
            "rows": len(group),
            "manifest_fields": {
                "allowed_instance_count_unique": sorted(
                    int(value)
                    for value in pool_group["allowed_instance_count_field"].unique()
                ),
                "allowed_source_count_unique": sorted(
                    int(value)
                    for value in pool_group["allowed_source_count_field"].unique()
                ),
            },
            "selected_source_rows_unique": int(
                exemplars.loc[
                    exemplars["channel"] == channel, "source_train_index"
                ].nunique()
            ),
            "rows_with_covering_exemplar": int(group["has_covering_exemplar"].sum()),
            "fraction_with_covering_exemplar": float(
                group["has_covering_exemplar"].mean()
            ),
            "flat_success_skill_failure_rows": len(discordant),
            "flat_success_skill_failure_with_cover": int(
                discordant["has_covering_exemplar"].sum()
            ),
            "flat_success_skill_failure_cover_fraction": float(
                discordant["has_covering_exemplar"].mean()
            ),
            "multi_unit_exemplars": int(
                exemplars.loc[
                    exemplars["channel"] == channel, "source_is_multi_unit"
                ].sum()
            ),
            "maximum_grounded_plan_trigram_jaccard": float(
                group["maximum_grounded_plan_trigram_jaccard"].max()
            ),
        }
    trigger = by_channel["productivity"]["rows_with_covering_exemplar"] > 0
    summary = {
        "task": "Amendment4_G0",
        "status": "PASS",
        "audit_scope": (
            "whole source_train_index rows whose complete formatted segment "
            "blocks remain present after Amendment 3.1 token-level prefix trimming"
        ),
        "coverage_definition": (
            "C2 contiguous ordered-unit coverage of the test full signature"
        ),
        "pool_fact_basis": [
            "allowed_instance_ids",
            "allowed_instance_count",
            "allowed_source_indices",
            "allowed_source_count",
        ],
        "m0_source_train_rows": int(
            len(
                {
                    instance["source_train_index"]
                    for instance in instances_by_id.values()
                }
            )
        ),
        "by_channel": by_channel,
        "branch": {
            "productivity_covering_exemplar_present": trigger,
            "g2_required": trigger,
            "instance_rerun_required": False,
        },
        "artifacts": artifacts,
    }
    atomic_json(OUTPUT_DIR / "g0_summary.json", summary)
    return summary


def route_metrics(
    record: Mapping[str, Any],
    expected_kinds: Sequence[str],
    instances_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    route_kinds = []
    subgoal_rows = []
    scorable_hits = []
    for position, group in enumerate(record["route"]["groups"]):
        predicted_subgoal = str(group["predicted_subgoal"])
        predicted_kind = classify_subgoal(predicted_subgoal)
        injected_ids = [str(item["instance_id"]) for item in group["instances"]]
        injected_kinds = sorted(
            {
                str(kind)
                for instance_id in injected_ids
                for kind in instances_by_id[instance_id].get("unit_kinds", [])
            }
        )
        hit = predicted_kind in injected_kinds if predicted_kind is not None else None
        if predicted_kind is not None:
            route_kinds.append(predicted_kind)
            scorable_hits.append(bool(hit))
        subgoal_rows.append(
            {
                "index": int(record["index"]),
                "subgoal_position": position,
                "predicted_subgoal": predicted_subgoal,
                "predicted_primitive_kind": predicted_kind,
                "matched_skill_name": group.get("skill_name"),
                "injected_instance_ids": json.dumps(
                    injected_ids, separators=(",", ":")
                ),
                "injected_unit_kinds": json.dumps(
                    injected_kinds, separators=(",", ":")
                ),
                "retrieval_hit": hit,
            }
        )
    multiset_agreement = Counter(route_kinds) == Counter(expected_kinds)
    order_agreement = route_kinds == list(expected_kinds)
    all_hits = bool(scorable_hits) and all(scorable_hits)
    lcs = lcs_length(route_kinds, expected_kinds)
    metrics = {
        "expected_primitive_kinds": json.dumps(
            list(expected_kinds), separators=(",", ":")
        ),
        "route_primitive_kinds": json.dumps(route_kinds, separators=(",", ":")),
        "unit_multiset_agreement": multiset_agreement,
        "unit_multiset_f1": multiset_f1(route_kinds, expected_kinds),
        "order_agreement": order_agreement,
        "order_lcs_ratio": lcs / len(expected_kinds) if expected_kinds else 1.0,
        "scorable_subgoals": len(scorable_hits),
        "retrieval_hits": sum(scorable_hits),
        "retrieval_hit_rate": (
            sum(scorable_hits) / len(scorable_hits) if scorable_hits else None
        ),
        "all_subgoals_retrieval_hit": all_hits,
        "route_correct": multiset_agreement and order_agreement and all_hits,
    }
    return metrics, subgoal_rows


def summarize_route_frame(frame: pd.DataFrame) -> Dict[str, Any]:
    strata = {}
    for route_correct, group in frame.groupby("route_correct", sort=False):
        label = "route_correct" if route_correct else "route_incorrect"
        strata[label] = {
            **paired_summary(group, "flat", "skill"),
            "unit_multiset_agreement_rate": float(
                group["unit_multiset_agreement"].mean()
            ),
            "order_agreement_rate": float(group["order_agreement"].mean()),
            "mean_retrieval_hit_rate": float(
                group["retrieval_hit_rate"].fillna(0).mean()
            ),
        }
    return {
        "rows": len(frame),
        "route_correct_rows": int(frame["route_correct"].sum()),
        "route_correct_fraction": float(frame["route_correct"].mean()),
        "unit_multiset_agreement_fraction": float(
            frame["unit_multiset_agreement"].mean()
        ),
        "order_agreement_fraction": float(frame["order_agreement"].mean()),
        "mean_unit_multiset_f1": float(frame["unit_multiset_f1"].mean()),
        "mean_order_lcs_ratio": float(frame["order_lcs_ratio"].mean()),
        "mean_retrieval_hit_rate": float(frame["retrieval_hit_rate"].fillna(0).mean()),
        "strata": strata,
    }


def run_g1(
    census: pd.DataFrame,
    test_dataset: pd.DataFrame,
    val_dataset: pd.DataFrame,
    instances_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    test_census = census[census["split"] == "test"].set_index("index")
    route_rows = []
    subgoal_rows = []

    for channel in C_PRIME_CHANNELS:
        records = load_jsonl(AMENDMENT3_DIR / f"f2_cprime_{channel}.jsonl")
        for index, record in records.items():
            expected = [
                str(kind)
                for kind in parse_json_list(test_census.loc[index, "unit_kinds"])
                if kind in PRIMITIVE_KINDS
            ]
            metrics, groups = route_metrics(record, expected, instances_by_id)
            for group in groups:
                group["channel"] = channel
            subgoal_rows.extend(groups)
            route_rows.append(
                {
                    "channel": channel,
                    "index": index,
                    "c0_full_signature": str(test_census.loc[index, "full_signature"]),
                    **metrics,
                    "skill_success": float(record["arms"]["skill_memory"]["task_score"])
                    == 1,
                    "flat_success": float(record["arms"]["flat_memory"]["task_score"])
                    == 1,
                }
            )
    routes = pd.DataFrame(route_rows).sort_values(["channel", "index"])
    subgoals = pd.DataFrame(subgoal_rows).sort_values(
        ["channel", "index", "subgoal_position"]
    )
    ood_records = load_jsonl(AMENDMENT3_DIR / "f2_ood.jsonl")
    ood_rows = []
    ood_subgoals = []
    for index, record in ood_records.items():
        ground_truth = native(val_dataset.iloc[index]["reward_model"])["ground_truth"]
        signature = parse_composition(ground_truth)
        expected = [
            unit.kind for unit in signature.units if unit.kind in PRIMITIVE_KINDS
        ]
        metrics, groups = route_metrics(record, expected, instances_by_id)
        for group in groups:
            group["channel"] = "ood"
        ood_subgoals.extend(groups)
        zero_success = float(record["arms"]["zero_shot"]["task_score"]) == 1
        skill_success = float(record["arms"]["skill_memory"]["task_score"]) == 1
        transition = (
            "fix"
            if skill_success and not zero_success
            else "regression"
            if zero_success and not skill_success
            else "unchanged_success"
            if zero_success and skill_success
            else "unchanged_failure"
        )
        ood_rows.append(
            {
                "channel": "ood",
                "index": index,
                "c0_full_signature": signature.full_signature(),
                **metrics,
                "zero_shot_success": zero_success,
                "skill_success": skill_success,
                "transition": transition,
            }
        )
    ood = pd.DataFrame(ood_rows).sort_values("index")
    ood_subgoal_frame = pd.DataFrame(ood_subgoals).sort_values(
        ["index", "subgoal_position"]
    )
    artifacts = {
        "cprime_rows": write_frame(routes, "g1_cprime_route_rows"),
        "cprime_subgoals": write_frame(subgoals, "g1_cprime_route_subgoals"),
        "ood_rows": write_frame(ood, "g1_ood_route_rows"),
        "ood_subgoals": write_frame(ood_subgoal_frame, "g1_ood_route_subgoals"),
    }
    ood_transitions = {}
    for transition in ("fix", "regression"):
        group = ood[ood["transition"] == transition]
        ood_transitions[transition] = {
            "rows": len(group),
            "route_correct_rows": int(group["route_correct"].sum()),
            "route_correct_fraction": float(group["route_correct"].mean()),
            "unit_multiset_agreement_fraction": float(
                group["unit_multiset_agreement"].mean()
            ),
            "order_agreement_fraction": float(group["order_agreement"].mean()),
            "mean_retrieval_hit_rate": float(
                group["retrieval_hit_rate"].fillna(0).mean()
            ),
        }
    summary = {
        "task": "Amendment4_G1",
        "status": "PASS",
        "route_definition": {
            "c0_projection": list(PRIMITIVE_KINDS),
            "navigation_subgoals": "unscored",
            "derived_c0_composites": "excluded to avoid double-counting primitives",
            "route_correct": (
                "exact primitive-kind multiset, exact primitive-kind order, and "
                "every scorable predicted subgoal retrieves at least one instance "
                "containing its predicted primitive kind"
            ),
        },
        "by_channel": {
            channel: summarize_route_frame(routes[routes["channel"] == channel])
            for channel in C_PRIME_CHANNELS
        },
        "ood_transitions": ood_transitions,
        "artifacts": artifacts,
    }
    atomic_json(OUTPUT_DIR / "g1_summary.json", summary)
    return summary


def run_audit() -> Dict[str, Any]:
    require_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    census = pd.read_parquet(CENSUS_PATH)
    manifest = pd.read_parquet(MANIFEST_PATH)
    train_dataset = pd.read_parquet(DATA_ROOT / "train.parquet")
    test_dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    val_dataset = pd.read_parquet(DATA_ROOT / "val.parquet")
    instances_by_id, _ = load_sources()
    g0 = run_g0(
        manifest,
        census,
        test_dataset,
        train_dataset,
        instances_by_id,
    )
    g1 = run_g1(census, test_dataset, val_dataset, instances_by_id)
    result = {
        "task": "Amendment4_zero_call_audits",
        "model_calls": 0,
        "g0": g0,
        "g1": g1,
        "branch": g0["branch"],
        "status": "PASS",
    }
    atomic_json(OUTPUT_DIR / "audit_summary.json", result)
    return result


def write_jsonl_snapshot(
    path: Path,
    records: Mapping[int, Mapping[str, Any]],
    indices: Sequence[int],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as destination:
        for index in indices:
            if index in records:
                destination.write(json.dumps(records[index], ensure_ascii=True) + "\n")
    temporary.replace(path)


def require_generation_open(command: str) -> None:
    if GENERATION_CLOSED_PATH.is_file():
        closure = json.loads(GENERATION_CLOSED_PATH.read_text())
        raise GateFailure(
            f"VIKI generation is closed; {command} is unauthorized: {closure}"
        )


def clean_flat_pool(
    census: pd.DataFrame,
    instances_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[str], Set[int], List[int]]:
    train = census[census["split"] == "train"].set_index("index")
    c0_single_unit_sources = {
        int(index)
        for index, row in train.iterrows()
        if len(parse_json_list(row["ordered_units"])) == 1
    }
    bank_sources = {
        int(instance["source_train_index"]) for instance in instances_by_id.values()
    }
    single_unit_sources = c0_single_unit_sources & bank_sources
    unavailable_sources = sorted(c0_single_unit_sources - bank_sources)
    allowed_ids = sorted(
        instance_id
        for instance_id, instance in instances_by_id.items()
        if int(instance["source_train_index"]) in single_unit_sources
    )
    represented_sources = {
        int(instances_by_id[instance_id]["source_train_index"])
        for instance_id in allowed_ids
    }
    if not allowed_ids or represented_sources != single_unit_sources:
        missing = sorted(single_unit_sources - represented_sources)
        raise GateFailure(
            "G2 clean pool does not represent every single-unit source row: "
            f"instances={len(allowed_ids)}, sources={len(represented_sources)}, "
            f"missing={missing[:10]}"
        )
    return allowed_ids, represented_sources, unavailable_sources


def g2_indices(manifest: pd.DataFrame) -> List[int]:
    indices = sorted(
        int(value) for value in manifest.loc[manifest["channel"] == G2_CHANNEL, "index"]
    )
    if len(indices) != G2_ROWS or len(set(indices)) != G2_ROWS:
        raise GateFailure(f"G2 requires {G2_ROWS} unique productivity rows")
    return indices


def g2_metadata(
    args: argparse.Namespace,
    task: str,
    indices: Sequence[int],
    clean_ids: Sequence[str],
    clean_sources: Set[int],
    unavailable_sources: Sequence[int],
    preflight_path: Optional[Path] = None,
) -> Dict[str, Any]:
    original_run_path = AMENDMENT3_DIR / "f2_cprime_productivity.jsonl.run.json"
    original_metadata = json.loads(original_run_path.read_text())
    metadata = {
        "task": task,
        "channel": G2_CHANNEL,
        "indices": list(indices),
        "model": args.model,
        "base_urls": list(args.base_urls),
        "server_metadata": server_metadata(args.base_urls),
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "token_tolerance": TOKEN_TOLERANCE,
        "token_match_criterion": "per-row",
        "flat_builder": "Amendment 3.1 extend-whole-rows-then-token-trim",
        "pool_rule": "C0 full signature has exactly one ordered unit",
        "clean_pool_instance_count": len(clean_ids),
        "clean_pool_source_count": len(clean_sources),
        "c0_single_unit_sources_absent_from_m0": list(unavailable_sources),
        "original_productivity_run_fingerprint": fingerprint(original_metadata),
        "reused_arms": ["zero_shot", "skill_memory"],
        "discarded_arms": ["flat_memory"],
        "artifacts": {
            "audit_summary_sha256": file_sha256(OUTPUT_DIR / "audit_summary.json"),
            "census_sha256": file_sha256(CENSUS_PATH),
            "instances_sha256": file_sha256(INSTANCES_PATH),
            "manifest_sha256": file_sha256(MANIFEST_PATH),
            "original_results_sha256": file_sha256(
                AMENDMENT3_DIR / "f2_cprime_productivity.jsonl"
            ),
            "original_run_sha256": file_sha256(original_run_path),
            "original_preflight_sha256": file_sha256(
                AMENDMENT3_DIR / "f2_cprime_productivity.preflight.jsonl"
            ),
        },
    }
    validate_server_metadata(metadata["server_metadata"])
    if preflight_path is not None:
        metadata["artifacts"]["g2_preflight_sha256"] = file_sha256(preflight_path)
        metadata["artifacts"][
            "amendment3_1_preflight_certificate_sha256"
        ] = file_sha256(AMENDMENT3_DIR / "f2_cprime_preflight.summary.json")
    return metadata


def original_productivity_inputs() -> (
    Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], str]
):
    result_path = AMENDMENT3_DIR / "f2_cprime_productivity.jsonl"
    run_path = result_path.with_suffix(result_path.suffix + ".run.json")
    records = load_jsonl(result_path)
    metadata = json.loads(run_path.read_text())
    run_hash = fingerprint(metadata)
    if len(records) != G2_ROWS:
        raise GateFailure("G2 requires all 400 original productivity records")
    for index, record in records.items():
        if record.get("run_fingerprint") != run_hash:
            raise GateFailure(f"Original productivity fingerprint mismatch at {index}")
        if set(record.get("arms", {})) != {
            "zero_shot",
            "skill_memory",
            "flat_memory",
        }:
            raise GateFailure(f"Original productivity arms are malformed at {index}")
    original_preflight = load_jsonl(
        AMENDMENT3_DIR / "f2_cprime_productivity.preflight.jsonl"
    )
    if len(original_preflight) != G2_ROWS:
        raise GateFailure("Original productivity preflight is incomplete")
    return records, original_preflight, run_hash


def validate_prompt_reuse(
    bench: Any,
    sample: Dict[str, Any],
    index: int,
    original: Mapping[str, Any],
    original_preflight: Mapping[str, Any],
) -> Dict[str, str]:
    zero_messages = bench.get_messages(sample)
    skill_messages = add_memory_to_messages(
        bench.get_messages(sample), original["route"]["memory_prompt"]
    )
    observed = {
        "zero_shot": messages_sha256(zero_messages),
        "skill_memory": messages_sha256(skill_messages),
    }
    expected = original_preflight["prompt_sha256"]
    if any(observed[arm] != expected.get(arm) for arm in observed):
        raise GateFailure(f"G2 reused prompt changed at index {index}")
    return observed


def run_g2_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    require_inputs()
    require_frozen_args(args)
    audit = json.loads((OUTPUT_DIR / "audit_summary.json").read_text())
    if audit.get("status") != "PASS" or not audit["branch"]["g2_required"]:
        raise GateFailure("G2 is not authorized by the completed G0 branch")
    census = pd.read_parquet(CENSUS_PATH)
    manifest = pd.read_parquet(MANIFEST_PATH)
    test_census = census[census["split"] == "test"].set_index("index")
    test_dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    instances_by_id, _ = load_sources()
    clean_ids, clean_sources, unavailable_sources = clean_flat_pool(
        census, instances_by_id
    )
    indices = g2_indices(manifest)
    original, original_preflight, original_hash = original_productivity_inputs()
    bench = load_bench()
    provider = F2Provider(args, bench)
    run_path = G2_PREFLIGHT_PATH.with_suffix(G2_PREFLIGHT_PATH.suffix + ".run.json")
    metadata = g2_metadata(
        args,
        "Amendment4_G2_preflight",
        indices,
        clean_ids,
        clean_sources,
        unavailable_sources,
    )
    run_hash = fingerprint(metadata)
    if run_path.is_file():
        observed_metadata = json.loads(run_path.read_text())
        if observed_metadata != metadata:
            observed_comparable = native(observed_metadata)
            expected_comparable = native(metadata)
            observed_comparable["artifacts"].pop("audit_summary_sha256", None)
            expected_comparable["artifacts"].pop("audit_summary_sha256", None)
            if observed_comparable != expected_comparable:
                raise GateFailure("Cannot resume G2 preflight: metadata differs")
            prior_hash = fingerprint(observed_metadata)
            prior_records = load_jsonl(G2_PREFLIGHT_PATH)
            if len(prior_records) != len(indices) or any(
                record.get("run_fingerprint") != prior_hash
                for record in prior_records.values()
            ):
                raise GateFailure("Cannot rebind incomplete G2 preflight records")
            for record in prior_records.values():
                record["run_fingerprint"] = run_hash
            write_jsonl_snapshot(G2_PREFLIGHT_PATH, prior_records, indices)
            atomic_json(run_path, metadata)
            atomic_json(
                OUTPUT_DIR / "g2_clean_flat.preflight.rebind.json",
                {
                    "reason": "corrected G0 post-trim whole-row audit",
                    "prior_run_fingerprint": prior_hash,
                    "new_run_fingerprint": run_hash,
                    "rows_reused": len(prior_records),
                    "token_calls_repeated": 0,
                    "status": "PASS",
                },
            )
    else:
        atomic_json(run_path, metadata)
    records = load_jsonl(G2_PREFLIGHT_PATH) if G2_PREFLIGHT_PATH.is_file() else {}
    for index, record in records.items():
        if index not in indices or record.get("run_fingerprint") != run_hash:
            raise GateFailure(f"Invalid G2 preflight resume record at {index}")
        if (
            record.get("dropped")
            or record["clean_flat"]["relative_difference"] > TOKEN_TOLERANCE
        ):
            raise GateFailure(f"Invalid G2 preflight token gate at {index}")
    pending = [index for index in indices if index not in records]

    def inspect(index: int) -> Tuple[int, Dict[str, Any]]:
        sample = native(test_dataset.iloc[index].to_dict())
        prompt_hashes = validate_prompt_reuse(
            bench, sample, index, original[index], original_preflight[index]
        )
        route = original[index]["route"]
        has_skill_memory = any(group["instances"] for group in route["groups"])
        try:
            clean_flat = provider.build_flat_control(
                sample,
                index,
                clean_ids,
                int(route["memory_input_tokens"]),
                has_skill_memory,
            )
        except SymmetricDrop as error:
            raise GateFailure(f"G2 symmetric drop at index {index}: {error}") from error
        selected_sources = list(
            dict.fromkeys(
                int(instances_by_id[instance_id]["source_train_index"])
                for instance_id in clean_flat[
                    "candidate_instance_ids_before_prefix_trim"
                ]
            )
        )
        non_single = [
            source for source in selected_sources if source not in clean_sources
        ]
        if non_single:
            raise GateFailure(
                f"G2 clean pool contains non-single-unit sources at {index}: "
                f"{non_single[:10]}"
            )
        test_signature = str(test_census.loc[index, "full_signature"])
        train_census = census[census["split"] == "train"].set_index("index")
        covering = [
            source
            for source in selected_sources
            if covers_full_signature(
                parse_json_list(train_census.loc[source, "ordered_units"]),
                test_signature,
            )
        ]
        if covering:
            raise GateFailure(
                f"G2 clean flat has covering sources at {index}: {covering[:10]}"
            )
        flat_value = {
            key: value for key, value in clean_flat.items() if key != "messages"
        }
        return index, {
            "index": index,
            "run_fingerprint": run_hash,
            "original_run_fingerprint": original_hash,
            "prompt_sha256": {
                **prompt_hashes,
                "clean_flat": messages_sha256(clean_flat["messages"]),
            },
            "clean_flat": flat_value,
            "selected_source_indices": selected_sources,
            "selected_source_count": len(selected_sources),
            "selected_multi_unit_source_count": 0,
            "covering_source_count": 0,
            "dropped": False,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(inspect, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                _, record = future.result()
            except Exception:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            records[index] = record
            write_jsonl_snapshot(G2_PREFLIGHT_PATH, records, indices)
    if set(records) != set(indices):
        raise GateFailure("G2 preflight is incomplete")
    differences = [
        float(records[index]["clean_flat"]["relative_difference"]) for index in indices
    ]
    summary = {
        "task": "Amendment4_G2_preflight",
        "status": "PASS",
        "rows": len(indices),
        "model_generation_calls": 0,
        "token_tolerance": TOKEN_TOLERANCE,
        "in_band_rows": sum(value <= TOKEN_TOLERANCE for value in differences),
        "maximum_relative_difference": max(differences),
        "symmetric_drops": 0,
        "pool": {
            "rule": "C0 full signature has exactly one ordered unit",
            "allowed_instance_count": len(clean_ids),
            "allowed_source_count": len(clean_sources),
            "c0_single_unit_sources_absent_from_m0": unavailable_sources,
            "selected_multi_unit_sources": 0,
            "covering_sources": 0,
            "single_unit_source_fraction": 1.0,
        },
        "reuse": {
            "verified_zero_shot_prompts": len(indices),
            "verified_skill_memory_prompts": len(indices),
            "original_run_fingerprint": original_hash,
        },
        "results": str(G2_PREFLIGHT_PATH),
        "results_sha256": file_sha256(G2_PREFLIGHT_PATH),
    }
    atomic_json(G2_PREFLIGHT_PATH.with_suffix(".summary.json"), summary)
    return summary


def validate_g2_preflight(
    args: argparse.Namespace,
    indices: Sequence[int],
    clean_ids: Sequence[str],
    clean_sources: Set[int],
    unavailable_sources: Sequence[int],
) -> Dict[int, Dict[str, Any]]:
    summary_path = G2_PREFLIGHT_PATH.with_suffix(".summary.json")
    run_path = G2_PREFLIGHT_PATH.with_suffix(G2_PREFLIGHT_PATH.suffix + ".run.json")
    if not summary_path.is_file() or not run_path.is_file():
        raise GateFailure("G2 run requires its completed preflight")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("status") != "PASS"
        or summary.get("rows") != G2_ROWS
        or summary.get("in_band_rows") != G2_ROWS
        or summary.get("symmetric_drops") != 0
        or summary.get("pool", {}).get("covering_sources") != 0
        or summary.get("pool", {}).get("single_unit_source_fraction") != 1.0
        or file_sha256(G2_PREFLIGHT_PATH) != summary.get("results_sha256")
    ):
        raise GateFailure("G2 preflight certificate is invalid")
    expected_metadata = g2_metadata(
        args,
        "Amendment4_G2_preflight",
        indices,
        clean_ids,
        clean_sources,
        unavailable_sources,
    )
    observed_metadata = json.loads(run_path.read_text())
    if observed_metadata != expected_metadata:
        raise GateFailure("G2 preflight metadata is stale")
    run_hash = fingerprint(observed_metadata)
    records = load_jsonl(G2_PREFLIGHT_PATH)
    if set(records) != set(indices):
        raise GateFailure("G2 preflight rows are incomplete")
    for index, record in records.items():
        if (
            record.get("run_fingerprint") != run_hash
            or record.get("dropped")
            or record["covering_source_count"] != 0
            or record["selected_multi_unit_source_count"] != 0
            or record["clean_flat"]["relative_difference"] > TOKEN_TOLERANCE
        ):
            raise GateFailure(f"G2 preflight row failed at index {index}")
    return records


def comparison_summary(frame: pd.DataFrame, left: str, right: str) -> Dict[str, Any]:
    left_success = frame[f"{left}_task_score"] == 1
    right_success = frame[f"{right}_task_score"] == 1
    left_only = int((left_success & ~right_success).sum())
    right_only = int((~left_success & right_success).sum())
    return {
        "samples": len(frame),
        f"{left}_successes": int(left_success.sum()),
        f"{right}_successes": int(right_success.sum()),
        f"{left}_accuracy": float(left_success.mean()),
        f"{right}_accuracy": float(right_success.mean()),
        f"{right}_minus_{left}": float(right_success.mean() - left_success.mean()),
        "left_only": left_only,
        "right_only": right_only,
        "discordant_pairs": left_only + right_only,
        "mcnemar_exact_p": exact_mcnemar_p(right_only, left_only),
    }


def validate_g2_resume_records(
    records: Mapping[int, Mapping[str, Any]],
    indices: Sequence[int],
    run_hash: str,
    original_hash: str,
    original: Mapping[int, Mapping[str, Any]],
    preflight: Mapping[int, Mapping[str, Any]],
) -> None:
    unexpected = sorted(set(records) - set(indices))
    if unexpected:
        raise GateFailure(f"G2 resume has unexpected indices: {unexpected[:10]}")
    for index, record in records.items():
        if record.get("run_fingerprint") != run_hash:
            raise GateFailure(f"G2 resume fingerprint mismatch at {index}")
        if record.get("endpoint_error"):
            continue
        if (
            record.get("original_run_fingerprint") != original_hash
            or record.get("discarded_original_flat") is not True
            or record.get("clean_flat") != preflight[index]["clean_flat"]
            or record.get("clean_flat_prompt_sha256")
            != preflight[index]["prompt_sha256"]["clean_flat"]
            or record.get("reused_prompt_sha256")
            != {
                arm: preflight[index]["prompt_sha256"][arm]
                for arm in ("zero_shot", "skill_memory")
            }
        ):
            raise GateFailure(f"G2 resume provenance mismatch at {index}")
        arms = record.get("arms")
        if not isinstance(arms, dict) or set(arms) != {
            "zero_shot",
            "skill_memory",
            "clean_flat",
        }:
            raise GateFailure(f"G2 resume arms are malformed at {index}")
        if (
            arms["zero_shot"] != original[index]["arms"]["zero_shot"]
            or arms["skill_memory"] != original[index]["arms"]["skill_memory"]
        ):
            raise GateFailure(f"G2 reused output changed at {index}")
        if not {
            "response",
            "prompt_tokens",
            "completion_tokens",
            "task_score",
            "format_score",
        } <= set(arms["clean_flat"]):
            raise GateFailure(f"G2 clean-flat arm is malformed at {index}")


def run_g2(args: argparse.Namespace) -> Dict[str, Any]:
    require_generation_open("g2-run")
    require_inputs()
    require_frozen_args(args)
    census = pd.read_parquet(CENSUS_PATH)
    manifest = pd.read_parquet(MANIFEST_PATH)
    test_dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    instances_by_id, _ = load_sources()
    clean_ids, clean_sources, unavailable_sources = clean_flat_pool(
        census, instances_by_id
    )
    indices = g2_indices(manifest)
    certified_preflight, original_drops = load_certified_cprime_preflight(
        args, G2_CHANNEL, manifest
    )
    if original_drops:
        raise GateFailure(
            f"G2 cannot reuse dropped original rows: {original_drops[:10]}"
        )
    preflight = validate_g2_preflight(
        args, indices, clean_ids, clean_sources, unavailable_sources
    )
    original, original_preflight, original_hash = original_productivity_inputs()
    if certified_preflight != original_preflight:
        raise GateFailure("G2 original preflight differs from its certificate")
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    provider = F2Provider(args, bench)
    run_path = G2_OUTPUT_PATH.with_suffix(G2_OUTPUT_PATH.suffix + ".run.json")
    metadata = g2_metadata(
        args,
        "Amendment4_G2_clean_flat",
        indices,
        clean_ids,
        clean_sources,
        unavailable_sources,
        preflight_path=G2_PREFLIGHT_PATH,
    )
    run_hash = fingerprint(metadata)
    if run_path.is_file():
        if json.loads(run_path.read_text()) != metadata:
            raise GateFailure("Cannot resume G2 run: metadata differs")
    else:
        atomic_json(run_path, metadata)
    records = load_jsonl(G2_OUTPUT_PATH) if G2_OUTPUT_PATH.is_file() else {}
    validate_g2_resume_records(
        records,
        indices,
        run_hash,
        original_hash,
        original,
        preflight,
    )
    pending = [
        index
        for index in indices
        if index not in records or records[index].get("endpoint_error")
    ]

    def generate(index: int) -> Tuple[int, Dict[str, Any]]:
        sample = native(test_dataset.iloc[index].to_dict())
        observed_hashes = validate_prompt_reuse(
            bench, sample, index, original[index], original_preflight[index]
        )
        clean_messages = add_memory_to_messages(
            bench.get_messages(sample), preflight[index]["clean_flat"]["prompt"]
        )
        if (
            messages_sha256(clean_messages)
            != preflight[index]["prompt_sha256"]["clean_flat"]
        ):
            raise GateFailure(f"G2 clean-flat prompt changed at index {index}")
        generated = provider.call(index, clean_messages, args.max_tokens)
        metrics = bench.score_response(
            scorer,
            2,
            generated["response"],
            bench.get_ground_truth(sample),
            index,
        )
        return index, {
            "index": index,
            "run_fingerprint": run_hash,
            "original_run_fingerprint": original_hash,
            "reused_prompt_sha256": observed_hashes,
            "clean_flat_prompt_sha256": preflight[index]["prompt_sha256"]["clean_flat"],
            "clean_flat": preflight[index]["clean_flat"],
            "arms": {
                "zero_shot": original[index]["arms"]["zero_shot"],
                "skill_memory": original[index]["arms"]["skill_memory"],
                "clean_flat": {**generated, **metrics},
            },
            "discarded_original_flat": True,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                _, record = future.result()
            except GateFailure:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            except Exception as error:
                record = {
                    "index": index,
                    "run_fingerprint": run_hash,
                    "endpoint_error": repr(error),
                }
            records[index] = record
            write_jsonl_snapshot(G2_OUTPUT_PATH, records, indices)
    if set(records) != set(indices):
        raise GateFailure("G2 run is incomplete")
    errors = [record for record in records.values() if record.get("endpoint_error")]
    if errors:
        raise GateFailure(f"G2 halted with {len(errors)} endpoint errors")
    rows = []
    for index in indices:
        record = records[index]
        rows.append(
            {
                "index": index,
                "clean_flat_input_tokens": int(record["clean_flat"]["input_tokens"]),
                "clean_flat_relative_difference": float(
                    record["clean_flat"]["relative_difference"]
                ),
                **{
                    f"{arm}_{metric}": float(record["arms"][arm][metric])
                    for arm in ("zero_shot", "skill_memory", "clean_flat")
                    for metric in ("task_score", "format_score")
                },
            }
        )
    frame = pd.DataFrame(rows).sort_values("index")
    parquet_path = G2_OUTPUT_PATH.with_suffix(".parquet")
    frame.to_parquet(parquet_path, index=False)
    summary = {
        "task": "Amendment4_G2_clean_flat",
        "status": "PASS",
        "samples": len(frame),
        "model_generation_calls": len(frame),
        "pool": {
            "rule": "C0 full signature has exactly one ordered unit",
            "allowed_instance_count": len(clean_ids),
            "allowed_source_count": len(clean_sources),
            "c0_single_unit_sources_absent_from_m0": unavailable_sources,
            "single_unit_source_fraction": 1.0,
            "covering_sources": 0,
        },
        "token_match": {
            "criterion": "per-row",
            "tolerance": TOKEN_TOLERANCE,
            "maximum_relative_difference": float(
                frame["clean_flat_relative_difference"].max()
            ),
            "violations": int(
                (frame["clean_flat_relative_difference"] > TOKEN_TOLERANCE).sum()
            ),
        },
        "reuse": {
            "original_run_fingerprint": original_hash,
            "new_run_fingerprint": run_hash,
            "zero_shot_outputs_reused": len(frame),
            "skill_memory_outputs_reused": len(frame),
            "original_flat_outputs_discarded": len(frame),
        },
        "arms": {
            arm: {
                "successes": int((frame[f"{arm}_task_score"] == 1).sum()),
                "accuracy": float(frame[f"{arm}_task_score"].mean()),
                "format_compliance": float((frame[f"{arm}_format_score"] == 1).mean()),
            }
            for arm in ("zero_shot", "skill_memory", "clean_flat")
        },
        "comparisons": {
            "clean_flat_to_skill": comparison_summary(
                frame, "clean_flat", "skill_memory"
            ),
            "zero_shot_to_clean_flat": comparison_summary(
                frame, "zero_shot", "clean_flat"
            ),
            "zero_shot_to_skill": comparison_summary(
                frame, "zero_shot", "skill_memory"
            ),
        },
        "results": str(G2_OUTPUT_PATH),
        "parquet": str(parquet_path),
    }
    atomic_json(G2_OUTPUT_PATH.with_suffix(".summary.json"), summary)
    return summary


def skill_singleton_segment_block(instance: SegmentInstance) -> str:
    rendered = format_grouped_memory(
        [
            RetrievedGroup(
                "g2b_render_gate",
                instance.skill_name,
                1.0,
                [RetrievedSegment(instance, 1.0)],
                None,
            )
        ]
    )
    header = (
        "Grounded skill memory, grouped in predicted subgoal order:\n"
        f"Subgoal 1: g2b_render_gate (matched skill: {instance.skill_name})\n"
    )
    footer = (
        "\nUse the examples only as grounded sub-plan guidance.\n"
        "Produce one complete plan for the current task using only its available robot APIs."
    )
    if not rendered.startswith(header) or not rendered.endswith(footer):
        raise GateFailure(
            f"Unexpected skill renderer structure for {instance.instance_id}"
        )
    return rendered[len(header) : -len(footer)]


def segment_flat_block(instance: SegmentInstance) -> str:
    return "\n".join(
        [
            f"  Instance 1 context: {instance.context}",
            "  Instance availability: "
            + json.dumps(instance.self_cond, sort_keys=True, ensure_ascii=True),
            "  Instance demonstration: "
            + json.dumps(instance.demo, separators=(",", ":"), ensure_ascii=True),
        ]
    )


def format_segment_flat(instances: Sequence[SegmentInstance]) -> str:
    if not instances:
        return ""
    blocks = [segment_flat_block(instance) for instance in instances]
    blocks.extend(
        [
            "Use the examples only as grounded sub-plan guidance.",
            "Produce one complete plan for the current task using only its available robot APIs.",
        ]
    )
    return "\n".join(blocks)


class G2bProvider(F2Provider):
    def rank_segments(
        self,
        sample: Mapping[str, Any],
        allowed_instance_ids: Sequence[str],
    ) -> List[SegmentInstance]:
        allowed = set(str(value) for value in allowed_instance_ids)
        with self.memory.embedding_lock:
            query_embedding = np.asarray(
                self.memory.embedding_model.encode(
                    [get_instruction(sample)],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]
            )
        ranked = sorted(
            (
                (
                    float(
                        np.dot(
                            query_embedding,
                            self.memory.context_embeddings[position],
                        )
                    ),
                    instance.instance_id,
                    instance,
                )
                for position, instance in enumerate(self.memory.instances)
                if instance.instance_id in allowed
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if (
            len(ranked) != len(allowed)
            or {instance.instance_id for _, _, instance in ranked} != allowed
        ):
            raise GateFailure(
                "G2b ranking pool mismatch: "
                f"ranked={len(ranked)}, allowed={len(allowed)}"
            )
        return [instance for _, _, instance in ranked]

    def build_segment_flat(
        self,
        sample: Dict[str, Any],
        index: int,
        allowed_instance_ids: Sequence[str],
        skill_total_tokens: int,
        has_skill_memory: bool,
    ) -> Dict[str, Any]:
        base_messages = self.bench.get_messages(sample)
        base_tokens = self.tokenizer_count(index, base_messages)
        if not has_skill_memory:
            return {
                "prompt": "",
                "messages": base_messages,
                "input_tokens": base_tokens,
                "selected_instance_ids_before_token_trim": [],
                "segment_k": 0,
                "truncated": False,
                "relative_difference": 0.0,
            }
        ranked = self.rank_segments(sample, allowed_instance_ids)
        selected: List[SegmentInstance] = []
        prompt = ""
        messages = base_messages
        total = base_tokens
        for instance in ranked:
            selected.append(instance)
            prompt = format_segment_flat(selected)
            messages = add_memory_to_messages(self.bench.get_messages(sample), prompt)
            total = self.tokenizer_count(index, messages)
            if total >= skill_total_tokens:
                break
        difference = abs(total - skill_total_tokens) / skill_total_tokens
        if total < skill_total_tokens * (1 - TOKEN_TOLERANCE):
            raise SymmetricDrop(
                "G2b pool exhausted below token lower bound: "
                f"skill={skill_total_tokens}, segment_flat={total}, "
                f"segments={len(selected)}, difference={difference:.6f}"
            )
        truncated = False
        if total > skill_total_tokens:
            trimmed_prompt, trimmed_total = self.trim_flat_prompt(
                sample,
                index,
                prompt,
                max(1, skill_total_tokens - base_tokens),
                skill_total_tokens,
            )
            truncated = trimmed_prompt != prompt
            prompt = trimmed_prompt
            total = trimmed_total
            messages = add_memory_to_messages(self.bench.get_messages(sample), prompt)
            difference = abs(total - skill_total_tokens) / skill_total_tokens
        if difference > TOKEN_TOLERANCE:
            raise GateFailure(
                "G2b segment flat outside token tolerance: "
                f"skill={skill_total_tokens}, segment_flat={total}, "
                f"difference={difference:.6f}"
            )
        return {
            "prompt": prompt,
            "messages": messages,
            "input_tokens": total,
            "selected_instance_ids_before_token_trim": [
                instance.instance_id for instance in selected
            ],
            "segment_k": len(selected),
            "truncated": truncated,
            "relative_difference": difference,
        }

    def tokenizer_count(self, index: int, messages: Sequence[Mapping[str, Any]]) -> int:
        from viki_amendment3_f2 import token_count

        return token_count(self.endpoint(index), self.args.model, messages)


def g2b_manifest_rows(manifest: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    rows = manifest[manifest["channel"] == G2_CHANNEL].sort_values("index")
    result = {int(row["index"]): row for row in rows.to_dict("records")}
    if len(result) != G2_ROWS:
        raise GateFailure("G2b requires 400 productivity manifest rows")
    return result


def manifest_allowed_ids(row: Mapping[str, Any]) -> Tuple[str, List[str]]:
    raw = str(row["allowed_instance_ids"])
    values = [str(value) for value in json.loads(raw)]
    if json.dumps(values, separators=(",", ":")) != raw:
        raise GateFailure(
            f"G2b manifest pool is not byte-canonical at index {row['index']}"
        )
    if len(values) != int(row["allowed_instance_count"]):
        raise GateFailure(f"G2b manifest pool count mismatch at {row['index']}")
    return raw, values


def run_g2b_gates(args: argparse.Namespace) -> Dict[str, Any]:
    require_inputs()
    require_frozen_args(args)
    g2_summary = json.loads(G2_OUTPUT_PATH.with_suffix(".summary.json").read_text())
    if g2_summary.get("status") != "PASS":
        raise GateFailure("G2b requires completed G2")
    manifest = pd.read_parquet(MANIFEST_PATH)
    rows = g2b_manifest_rows(manifest)
    census = pd.read_parquet(CENSUS_PATH)
    test_census = census[census["split"] == "test"].set_index("index")
    test_dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    instances_by_id, _ = load_sources()
    provider = G2bProvider(args, load_bench())
    pool_rows = []
    unique_pool_hashes = set()
    unique_pool_ids: Optional[Set[str]] = None
    for index in sorted(rows):
        raw, allowed = manifest_allowed_ids(rows[index])
        allowed_set = set(allowed)
        if unique_pool_ids is None:
            unique_pool_ids = allowed_set
        elif allowed_set != unique_pool_ids:
            raise GateFailure(f"G2b productivity pool differs at index {index}")
        missing = sorted(allowed_set - set(instances_by_id))
        if missing:
            raise GateFailure(f"G2b pool has unknown instance IDs: {missing[:10]}")
        non_single = [
            instance_id
            for instance_id in allowed
            if len(instances_by_id[instance_id].get("ordered_units", [])) != 1
            or len(instances_by_id[instance_id].get("unit_kinds", [])) != 1
        ]
        if non_single:
            raise GateFailure(
                f"G2b pool has non-single-unit segments at {index}: {non_single[:10]}"
            )
        test_signature = str(test_census.loc[index, "full_signature"])
        covering = [
            instance_id
            for instance_id in allowed
            if covers_full_signature(
                instances_by_id[instance_id]["ordered_units"], test_signature
            )
        ]
        if covering:
            raise GateFailure(
                f"G2b pool has covering segments at {index}: {covering[:10]}"
            )
        pool_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        unique_pool_hashes.add(pool_hash)
        pool_rows.append(
            {
                "index": index,
                "manifest_allowed_instance_ids_sha256": pool_hash,
                "manifest_allowed_instance_count": len(allowed),
                "single_unit_segment_count": len(allowed),
                "covering_segment_count": 0,
                "pool_byte_roundtrip": True,
            }
        )
    if unique_pool_ids is None:
        raise GateFailure("G2b productivity pool is empty")
    memory_instances = {
        instance.instance_id: instance for instance in provider.memory.instances
    }
    render_mismatches = [
        instance_id
        for instance_id in sorted(unique_pool_ids)
        if segment_flat_block(memory_instances[instance_id])
        != skill_singleton_segment_block(memory_instances[instance_id])
    ]
    if render_mismatches:
        raise GateFailure(f"G2b per-segment render mismatch: {render_mismatches[:10]}")
    sampled_indices = sorted(
        int(value)
        for value in np.random.default_rng(SEED).choice(
            sorted(rows), size=20, replace=False
        )
    )
    render_rows = []
    for index in sampled_indices:
        _, allowed = manifest_allowed_ids(rows[index])
        ranked = provider.rank_segments(
            native(test_dataset.iloc[index].to_dict()), allowed
        )
        for rank, instance in enumerate(ranked[:6], 1):
            segment_render = segment_flat_block(instance)
            skill_render = skill_singleton_segment_block(instance)
            render_rows.append(
                {
                    "index": index,
                    "rank": rank,
                    "instance_id": instance.instance_id,
                    "segment_render_sha256": hashlib.sha256(
                        segment_render.encode("utf-8")
                    ).hexdigest(),
                    "skill_render_sha256": hashlib.sha256(
                        skill_render.encode("utf-8")
                    ).hexdigest(),
                    "byte_identical": segment_render == skill_render,
                }
            )
    pool_frame = pd.DataFrame(pool_rows).sort_values("index")
    render_frame = pd.DataFrame(render_rows).sort_values(["index", "rank"])
    if not render_frame["byte_identical"].all():
        raise GateFailure("G2b sampled render gate failed")
    summary = {
        "task": "Amendment4_1_G2b_zero_call_gates",
        "status": "PASS",
        "model_generation_calls": 0,
        "pool": {
            "rows": len(pool_frame),
            "allowed_instance_count": len(unique_pool_ids),
            "unique_manifest_pool_hashes": len(unique_pool_hashes),
            "byte_identical_rows": int(pool_frame["pool_byte_roundtrip"].sum()),
            "single_unit_fraction": 1.0,
            "non_single_unit_segments": 0,
            "covering_segments": 0,
        },
        "render": {
            "sample_seed": SEED,
            "sampled_rows": sampled_indices,
            "segments_checked_per_row": 6,
            "segments_checked": len(render_frame),
            "byte_identical_segments": int(render_frame["byte_identical"].sum()),
            "full_pool_template_mismatches": 0,
        },
        "artifacts": {
            "pool": write_frame(pool_frame, "g2b_pool_gate"),
            "render": write_frame(render_frame, "g2b_render_gate"),
        },
    }
    atomic_json(OUTPUT_DIR / "g2b_gates.summary.json", summary)
    return summary


def g2b_metadata(
    args: argparse.Namespace,
    task: str,
    indices: Sequence[int],
    preflight_path: Optional[Path] = None,
) -> Dict[str, Any]:
    gates_path = OUTPUT_DIR / "g2b_gates.summary.json"
    g2_run_path = G2_OUTPUT_PATH.with_suffix(G2_OUTPUT_PATH.suffix + ".run.json")
    metadata = {
        "task": task,
        "channel": G2_CHANNEL,
        "indices": list(indices),
        "model": args.model,
        "base_urls": list(args.base_urls),
        "server_metadata": server_metadata(args.base_urls),
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "embedding_model": args.embedding_model,
        "retrieval": {
            "query": "test instruction only",
            "candidate_text": "segment context",
            "metric": "normalized MPNet cosine",
            "skill_matching": False,
            "subgoal_prediction": False,
            "grouping": False,
        },
        "rendering": (
            "skill singleton per-segment block with grouping headers removed; "
            "segments concatenated in retrieval-similarity order"
        ),
        "token_tolerance": TOKEN_TOLERANCE,
        "token_match_criterion": "per-row",
        "flat_builder": "Amendment 3.1 extend-segments-then-token-trim",
        "reused_arms": ["zero_shot", "clean_flat", "skill_memory"],
        "new_arm": "segment_flat",
        "artifacts": {
            "manifest_sha256": file_sha256(MANIFEST_PATH),
            "instances_sha256": file_sha256(INSTANCES_PATH),
            "g2b_gates_sha256": file_sha256(gates_path),
            "g2_results_sha256": file_sha256(G2_OUTPUT_PATH),
            "g2_run_sha256": file_sha256(g2_run_path),
            "g2_preflight_sha256": file_sha256(G2_PREFLIGHT_PATH),
        },
    }
    validate_server_metadata(metadata["server_metadata"])
    if preflight_path is not None:
        metadata["artifacts"]["g2b_preflight_sha256"] = file_sha256(preflight_path)
        metadata["artifacts"]["g2b_posttrim_render_gate_sha256"] = file_sha256(
            G2B_POSTTRIM_RENDER_PATH
        )
        metadata["artifacts"]["g2b_posttrim_render_summary_sha256"] = file_sha256(
            G2B_POSTTRIM_RENDER_PATH.with_suffix(".summary.json")
        )
    return metadata


def g2b_reuse_inputs() -> (
    Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], str]
):
    summary = json.loads(G2_OUTPUT_PATH.with_suffix(".summary.json").read_text())
    if summary.get("status") != "PASS" or summary.get("samples") != G2_ROWS:
        raise GateFailure("G2b requires passing 400-row G2")
    records = load_jsonl(G2_OUTPUT_PATH)
    preflight = load_jsonl(G2_PREFLIGHT_PATH)
    metadata = json.loads(
        G2_OUTPUT_PATH.with_suffix(G2_OUTPUT_PATH.suffix + ".run.json").read_text()
    )
    run_hash = fingerprint(metadata)
    if len(records) != G2_ROWS or len(preflight) != G2_ROWS:
        raise GateFailure("G2b reuse inputs are incomplete")
    for index, record in records.items():
        if record.get("run_fingerprint") != run_hash or set(record.get("arms", {})) != {
            "zero_shot",
            "skill_memory",
            "clean_flat",
        }:
            raise GateFailure(f"G2b reuse record is malformed at {index}")
    return records, preflight, run_hash


def validate_g2b_reuse_prompts(
    bench: Any,
    sample: Dict[str, Any],
    index: int,
    original_f2: Mapping[str, Any],
    g2_record: Mapping[str, Any],
) -> Dict[str, str]:
    zero_messages = bench.get_messages(sample)
    skill_messages = add_memory_to_messages(
        bench.get_messages(sample), original_f2["route"]["memory_prompt"]
    )
    clean_messages = add_memory_to_messages(
        bench.get_messages(sample), g2_record["clean_flat"]["prompt"]
    )
    observed = {
        "zero_shot": messages_sha256(zero_messages),
        "skill_memory": messages_sha256(skill_messages),
        "clean_flat": messages_sha256(clean_messages),
    }
    expected = {
        **g2_record["reused_prompt_sha256"],
        "clean_flat": g2_record["clean_flat_prompt_sha256"],
    }
    if observed != expected:
        raise GateFailure(f"G2b reused prompt changed at index {index}")
    return observed


def run_g2b_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    require_inputs()
    require_frozen_args(args)
    gates_path = OUTPUT_DIR / "g2b_gates.summary.json"
    if not gates_path.is_file():
        raise GateFailure("G2b preflight requires zero-call gates")
    gates = json.loads(gates_path.read_text())
    if (
        gates.get("status") != "PASS"
        or gates.get("pool", {}).get("rows") != G2_ROWS
        or gates.get("pool", {}).get("allowed_instance_count") != 10359
        or gates.get("pool", {}).get("single_unit_fraction") != 1.0
        or gates.get("pool", {}).get("covering_segments") != 0
        or gates.get("render", {}).get("sampled_rows") is None
        or gates.get("render", {}).get("segments_checked")
        != gates.get("render", {}).get("byte_identical_segments")
    ):
        raise GateFailure("G2b zero-call gate certificate is invalid")
    manifest = pd.read_parquet(MANIFEST_PATH)
    rows = g2b_manifest_rows(manifest)
    indices = sorted(rows)
    test_dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    instances_by_id, _ = load_sources()
    original_f2, _, _ = original_productivity_inputs()
    g2_records, _, g2_hash = g2b_reuse_inputs()
    bench = load_bench()
    provider = G2bProvider(args, bench)
    memory_instances = {
        instance.instance_id: instance for instance in provider.memory.instances
    }
    run_path = G2B_PREFLIGHT_PATH.with_suffix(G2B_PREFLIGHT_PATH.suffix + ".run.json")
    metadata = g2b_metadata(args, "Amendment4_1_G2b_preflight", indices)
    run_hash = fingerprint(metadata)
    if run_path.is_file():
        if json.loads(run_path.read_text()) != metadata:
            raise GateFailure("Cannot resume G2b preflight: metadata differs")
    else:
        atomic_json(run_path, metadata)
    records = load_jsonl(G2B_PREFLIGHT_PATH) if G2B_PREFLIGHT_PATH.is_file() else {}
    for index, record in records.items():
        if index not in rows or record.get("run_fingerprint") != run_hash:
            raise GateFailure(f"Invalid G2b preflight resume row at {index}")
    pending = [index for index in indices if index not in records]
    sampled = set(int(value) for value in gates["render"]["sampled_rows"])

    def inspect(index: int) -> Tuple[int, Dict[str, Any]]:
        sample = native(test_dataset.iloc[index].to_dict())
        raw_pool, allowed = manifest_allowed_ids(rows[index])
        reuse_hashes = validate_g2b_reuse_prompts(
            bench, sample, index, original_f2[index], g2_records[index]
        )
        route = original_f2[index]["route"]
        has_skill_memory = any(group["instances"] for group in route["groups"])
        try:
            segment_flat = provider.build_segment_flat(
                sample,
                index,
                allowed,
                int(route["memory_input_tokens"]),
                has_skill_memory,
            )
        except SymmetricDrop as error:
            return index, {
                "index": index,
                "run_fingerprint": run_hash,
                "original_g2_run_fingerprint": g2_hash,
                "manifest_allowed_instance_ids_sha256": hashlib.sha256(
                    raw_pool.encode("utf-8")
                ).hexdigest(),
                "reuse_prompt_sha256": reuse_hashes,
                "dropped": True,
                "drop_reason": str(error),
            }
        selected_ids = segment_flat["selected_instance_ids_before_token_trim"]
        allowed_set = set(allowed)
        if any(instance_id not in allowed_set for instance_id in selected_ids):
            raise GateFailure(f"G2b selected outside pool at index {index}")
        non_single = [
            instance_id
            for instance_id in selected_ids
            if len(instances_by_id[instance_id].get("ordered_units", [])) != 1
            or len(instances_by_id[instance_id].get("unit_kinds", [])) != 1
        ]
        if non_single:
            raise GateFailure(f"G2b selected non-single segments at {index}")
        render_checked = index in sampled
        render_matches = (
            [
                segment_flat_block(memory_instances[instance_id])
                == skill_singleton_segment_block(memory_instances[instance_id])
                for instance_id in selected_ids
            ]
            if render_checked
            else []
        )
        if render_checked and not all(render_matches):
            raise GateFailure(f"G2b injected render mismatch at index {index}")
        flat_value = {
            key: value for key, value in segment_flat.items() if key != "messages"
        }
        return index, {
            "index": index,
            "run_fingerprint": run_hash,
            "original_g2_run_fingerprint": g2_hash,
            "manifest_allowed_instance_ids_sha256": hashlib.sha256(
                raw_pool.encode("utf-8")
            ).hexdigest(),
            "manifest_allowed_instance_count": len(allowed),
            "reuse_prompt_sha256": reuse_hashes,
            "segment_flat_prompt_sha256": messages_sha256(segment_flat["messages"]),
            "segment_flat": flat_value,
            "selected_single_unit_segments": len(selected_ids),
            "selected_covering_segments": 0,
            "render_gate_sampled_row": render_checked,
            "render_gate_segments_checked": len(render_matches),
            "render_gate_byte_identical_segments": sum(render_matches),
            "dropped": False,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(inspect, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                _, record = future.result()
            except Exception:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            records[index] = record
            write_jsonl_snapshot(G2B_PREFLIGHT_PATH, records, indices)
    if set(records) != set(indices):
        raise GateFailure("G2b preflight is incomplete")
    dropped = [record for record in records.values() if record.get("dropped")]
    retained = [record for record in records.values() if not record.get("dropped")]
    violations = [
        record
        for record in retained
        if float(record["segment_flat"]["relative_difference"]) > TOKEN_TOLERANCE
    ]
    sampled_records = [
        record for record in retained if record["render_gate_sampled_row"]
    ]
    sampled_checked = sum(
        int(record["render_gate_segments_checked"]) for record in sampled_records
    )
    sampled_identical = sum(
        int(record["render_gate_byte_identical_segments"]) for record in sampled_records
    )
    status = "PASS" if not dropped and not violations else "FAIL"
    summary = {
        "task": "Amendment4_1_G2b_preflight",
        "status": status,
        "rows": len(indices),
        "model_generation_calls": 0,
        "in_band_rows": len(retained) - len(violations),
        "maximum_relative_difference": max(
            (
                float(record["segment_flat"]["relative_difference"])
                for record in retained
            ),
            default=0.0,
        ),
        "symmetric_drops": len(dropped),
        "pool": {
            "manifest_pool_rows_verified": len(indices),
            "allowed_instance_count": 10359,
            "single_unit_fraction": 1.0,
            "covering_segments": 0,
        },
        "render": {
            "sampled_rows": len(sampled_records),
            "injected_segments_checked": sampled_checked,
            "byte_identical_segments": sampled_identical,
        },
        "reuse": {
            "zero_shot_prompts_verified": len(indices),
            "skill_memory_prompts_verified": len(indices),
            "clean_flat_prompts_verified": len(indices),
            "original_g2_run_fingerprint": g2_hash,
        },
        "results": str(G2B_PREFLIGHT_PATH),
        "results_sha256": file_sha256(G2B_PREFLIGHT_PATH),
    }
    atomic_json(G2B_PREFLIGHT_PATH.with_suffix(".summary.json"), summary)
    if status != "PASS":
        raise GateFailure(
            f"G2b preflight failed: drops={len(dropped)}, "
            f"violations={len(violations)}"
        )
    return summary


def fully_injected_segment_ids(
    prompt: str,
    selected: Sequence[SegmentInstance],
) -> Tuple[List[str], Optional[str]]:
    prefix = ""
    complete = []
    for instance in selected:
        block = segment_flat_block(instance)
        candidate = block if not prefix else f"{prefix}\n{block}"
        if prompt.startswith(candidate):
            prefix = candidate
            complete.append(instance.instance_id)
        else:
            remainder = prompt[len(prefix) :].lstrip("\n")
            return complete, instance.instance_id if remainder else None
    return complete, None


def run_g2b_posttrim_render_gate() -> Dict[str, Any]:
    gates = json.loads((OUTPUT_DIR / "g2b_gates.summary.json").read_text())
    preflight_summary = json.loads(
        G2B_PREFLIGHT_PATH.with_suffix(".summary.json").read_text()
    )
    if gates.get("status") != "PASS" or preflight_summary.get("status") != "PASS":
        raise GateFailure("G2b post-trim render gate requires passing inputs")
    preflight = load_jsonl(G2B_PREFLIGHT_PATH)
    instances_by_id, _ = load_sources()
    typed = {
        instance_id: SegmentInstance.from_dict(instance)
        for instance_id, instance in instances_by_id.items()
    }
    rows = []
    for index in gates["render"]["sampled_rows"]:
        record = preflight[int(index)]
        selected_ids = record["segment_flat"]["selected_instance_ids_before_token_trim"]
        selected = [typed[str(instance_id)] for instance_id in selected_ids]
        expected = format_segment_flat(selected)
        prompt = str(record["segment_flat"]["prompt"])
        is_prefix = expected.startswith(prompt)
        complete, partial = fully_injected_segment_ids(prompt, selected)
        complete_render_matches = all(
            segment_flat_block(typed[instance_id])
            == skill_singleton_segment_block(typed[instance_id])
            for instance_id in complete
        )
        rows.append(
            {
                "index": int(index),
                "selected_instance_count_before_trim": len(selected_ids),
                "fully_injected_instance_ids": json.dumps(
                    complete, separators=(",", ":")
                ),
                "fully_injected_instance_count": len(complete),
                "partially_injected_instance_id": partial,
                "partial_suffix_count": int(partial is not None),
                "final_prompt_is_byte_prefix_of_rendered_segments": is_prefix,
                "complete_segment_blocks_byte_identical": complete_render_matches,
                "final_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "untrimmed_render_sha256": hashlib.sha256(
                    expected.encode("utf-8")
                ).hexdigest(),
            }
        )
    frame = pd.DataFrame(rows).sort_values("index")
    if (
        len(frame) != 20
        or not frame["final_prompt_is_byte_prefix_of_rendered_segments"].all()
        or not frame["complete_segment_blocks_byte_identical"].all()
        or int(frame["partial_suffix_count"].max()) > 1
    ):
        raise GateFailure("G2b post-trim render gate failed")
    frame.to_parquet(G2B_POSTTRIM_RENDER_PATH, index=False)
    frame.to_csv(G2B_POSTTRIM_RENDER_PATH.with_suffix(".csv"), index=False)
    summary = {
        "task": "Amendment4_1_G2b_posttrim_render_gate",
        "status": "PASS",
        "model_generation_calls": 0,
        "sampled_rows": len(frame),
        "final_prompt_prefix_rows": int(
            frame["final_prompt_is_byte_prefix_of_rendered_segments"].sum()
        ),
        "fully_injected_segments": int(frame["fully_injected_instance_count"].sum()),
        "byte_identical_complete_segments": int(
            frame.loc[
                frame["complete_segment_blocks_byte_identical"],
                "fully_injected_instance_count",
            ].sum()
        ),
        "rows_with_one_partial_token_trim_suffix": int(
            frame["partial_suffix_count"].sum()
        ),
        "maximum_partial_suffixes_per_row": int(frame["partial_suffix_count"].max()),
        "results": str(G2B_POSTTRIM_RENDER_PATH),
        "results_sha256": file_sha256(G2B_POSTTRIM_RENDER_PATH),
    }
    atomic_json(G2B_POSTTRIM_RENDER_PATH.with_suffix(".summary.json"), summary)
    return summary


def validate_g2b_preflight(
    args: argparse.Namespace, indices: Sequence[int]
) -> Dict[int, Dict[str, Any]]:
    summary_path = G2B_PREFLIGHT_PATH.with_suffix(".summary.json")
    run_path = G2B_PREFLIGHT_PATH.with_suffix(G2B_PREFLIGHT_PATH.suffix + ".run.json")
    if not summary_path.is_file() or not run_path.is_file():
        raise GateFailure("G2b run requires completed preflight")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("status") != "PASS"
        or summary.get("rows") != G2_ROWS
        or summary.get("in_band_rows") != G2_ROWS
        or summary.get("symmetric_drops") != 0
        or summary.get("pool", {}).get("manifest_pool_rows_verified") != G2_ROWS
        or summary.get("pool", {}).get("allowed_instance_count") != 10359
        or summary.get("pool", {}).get("single_unit_fraction") != 1.0
        or summary.get("pool", {}).get("covering_segments") != 0
        or summary.get("render", {}).get("sampled_rows") != 20
        or summary.get("render", {}).get("injected_segments_checked")
        != summary.get("render", {}).get("byte_identical_segments")
        or file_sha256(G2B_PREFLIGHT_PATH) != summary.get("results_sha256")
    ):
        raise GateFailure("G2b preflight certificate is invalid")
    expected_metadata = g2b_metadata(args, "Amendment4_1_G2b_preflight", indices)
    observed_metadata = json.loads(run_path.read_text())
    if observed_metadata != expected_metadata:
        raise GateFailure("G2b preflight metadata is stale")
    run_hash = fingerprint(observed_metadata)
    records = load_jsonl(G2B_PREFLIGHT_PATH)
    if set(records) != set(indices):
        raise GateFailure("G2b preflight rows are incomplete")
    for index, record in records.items():
        if (
            record.get("run_fingerprint") != run_hash
            or record.get("dropped")
            or record.get("manifest_allowed_instance_count") != 10359
            or record.get("selected_covering_segments") != 0
            or record["segment_flat"]["relative_difference"] > TOKEN_TOLERANCE
            or set(record.get("reuse_prompt_sha256", {}))
            != {"zero_shot", "skill_memory", "clean_flat"}
        ):
            raise GateFailure(f"G2b preflight row failed at {index}")
    return records


def paired_delta_interval(
    frame: pd.DataFrame,
    left: str,
    right: str,
    seed_offset: int,
) -> List[float]:
    difference = (
        (frame[f"{right}_task_score"] == 1).astype(int)
        - (frame[f"{left}_task_score"] == 1).astype(int)
    ).to_numpy()
    probabilities = np.array(
        [
            float((difference == -1).mean()),
            float((difference == 0).mean()),
            float((difference == 1).mean()),
        ]
    )
    draws = np.random.default_rng(SEED + seed_offset).multinomial(
        len(frame), probabilities, size=BOOTSTRAP_DRAWS
    )
    deltas = (draws[:, 2] - draws[:, 0]) / len(frame)
    return [float(value) for value in np.quantile(deltas, [0.025, 0.975])]


def g2b_comparison_summary(
    frame: pd.DataFrame,
    left: str,
    right: str,
    seed_offset: int,
) -> Dict[str, Any]:
    result = comparison_summary(frame, left, right)
    result["paired_delta_interval"] = paired_delta_interval(
        frame, left, right, seed_offset
    )
    result["interval_method"] = (
        f"paired multinomial bootstrap, {BOOTSTRAP_DRAWS} draws, "
        f"seed {SEED + seed_offset}"
    )
    return result


def validate_g2b_resume_records(
    records: Mapping[int, Mapping[str, Any]],
    indices: Sequence[int],
    run_hash: str,
    g2_hash: str,
    g2_records: Mapping[int, Mapping[str, Any]],
    preflight: Mapping[int, Mapping[str, Any]],
) -> None:
    unexpected = sorted(set(records) - set(indices))
    if unexpected:
        raise GateFailure(f"G2b resume has unexpected indices: {unexpected[:10]}")
    for index, record in records.items():
        if record.get("run_fingerprint") != run_hash:
            raise GateFailure(f"G2b resume fingerprint mismatch at {index}")
        if record.get("endpoint_error"):
            continue
        if (
            record.get("original_g2_run_fingerprint") != g2_hash
            or record.get("reused_prompt_sha256")
            != preflight[index]["reuse_prompt_sha256"]
            or record.get("segment_flat_prompt_sha256")
            != preflight[index]["segment_flat_prompt_sha256"]
            or record.get("segment_flat") != preflight[index]["segment_flat"]
        ):
            raise GateFailure(f"G2b resume provenance mismatch at {index}")
        arms = record.get("arms")
        if not isinstance(arms, dict) or set(arms) != {
            "zero_shot",
            "clean_flat",
            "segment_flat",
            "skill_memory",
        }:
            raise GateFailure(f"G2b resume arms are malformed at {index}")
        for arm in ("zero_shot", "clean_flat", "skill_memory"):
            if arms[arm] != g2_records[index]["arms"][arm]:
                raise GateFailure(f"G2b reused output changed at {index}:{arm}")
        if not {
            "response",
            "prompt_tokens",
            "completion_tokens",
            "task_score",
            "format_score",
        } <= set(arms["segment_flat"]):
            raise GateFailure(f"G2b segment-flat output malformed at {index}")


def run_g2b(args: argparse.Namespace) -> Dict[str, Any]:
    require_generation_open("g2b-run")
    require_inputs()
    require_frozen_args(args)
    manifest = pd.read_parquet(MANIFEST_PATH)
    rows = g2b_manifest_rows(manifest)
    indices = sorted(rows)
    preflight = validate_g2b_preflight(args, indices)
    g2_records, _, g2_hash = g2b_reuse_inputs()
    test_dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    provider = G2bProvider(args, bench)
    run_path = G2B_OUTPUT_PATH.with_suffix(G2B_OUTPUT_PATH.suffix + ".run.json")
    metadata = g2b_metadata(
        args,
        "Amendment4_1_G2b_segment_flat",
        indices,
        preflight_path=G2B_PREFLIGHT_PATH,
    )
    run_hash = fingerprint(metadata)
    if run_path.is_file():
        if json.loads(run_path.read_text()) != metadata:
            raise GateFailure("Cannot resume G2b run: metadata differs")
    else:
        atomic_json(run_path, metadata)
    records = load_jsonl(G2B_OUTPUT_PATH) if G2B_OUTPUT_PATH.is_file() else {}
    validate_g2b_resume_records(
        records, indices, run_hash, g2_hash, g2_records, preflight
    )
    pending = [
        index
        for index in indices
        if index not in records or records[index].get("endpoint_error")
    ]

    def generate(index: int) -> Tuple[int, Dict[str, Any]]:
        sample = native(test_dataset.iloc[index].to_dict())
        segment_messages = add_memory_to_messages(
            bench.get_messages(sample),
            preflight[index]["segment_flat"]["prompt"],
        )
        if (
            messages_sha256(segment_messages)
            != preflight[index]["segment_flat_prompt_sha256"]
        ):
            raise GateFailure(f"G2b segment-flat prompt changed at {index}")
        generated = provider.call(index, segment_messages, args.max_tokens)
        metrics = bench.score_response(
            scorer,
            2,
            generated["response"],
            bench.get_ground_truth(sample),
            index,
        )
        return index, {
            "index": index,
            "run_fingerprint": run_hash,
            "original_g2_run_fingerprint": g2_hash,
            "reused_prompt_sha256": preflight[index]["reuse_prompt_sha256"],
            "segment_flat_prompt_sha256": preflight[index][
                "segment_flat_prompt_sha256"
            ],
            "segment_flat": preflight[index]["segment_flat"],
            "arms": {
                "zero_shot": g2_records[index]["arms"]["zero_shot"],
                "clean_flat": g2_records[index]["arms"]["clean_flat"],
                "segment_flat": {**generated, **metrics},
                "skill_memory": g2_records[index]["arms"]["skill_memory"],
            },
            "reused_arms": ["zero_shot", "clean_flat", "skill_memory"],
            "new_arm": "segment_flat",
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                _, record = future.result()
            except GateFailure:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            except Exception as error:
                record = {
                    "index": index,
                    "run_fingerprint": run_hash,
                    "endpoint_error": repr(error),
                }
            records[index] = record
            write_jsonl_snapshot(G2B_OUTPUT_PATH, records, indices)
    if set(records) != set(indices):
        raise GateFailure("G2b run is incomplete")
    errors = [record for record in records.values() if record.get("endpoint_error")]
    if errors:
        raise GateFailure(f"G2b halted with {len(errors)} endpoint errors")
    rows_out = []
    for index in indices:
        record = records[index]
        rows_out.append(
            {
                "index": index,
                "segment_flat_input_tokens": int(
                    record["segment_flat"]["input_tokens"]
                ),
                "segment_flat_relative_difference": float(
                    record["segment_flat"]["relative_difference"]
                ),
                **{
                    f"{arm}_{metric}": float(record["arms"][arm][metric])
                    for arm in (
                        "zero_shot",
                        "clean_flat",
                        "segment_flat",
                        "skill_memory",
                    )
                    for metric in ("task_score", "format_score")
                },
            }
        )
    frame = pd.DataFrame(rows_out).sort_values("index")
    parquet_path = G2B_OUTPUT_PATH.with_suffix(".parquet")
    frame.to_parquet(parquet_path, index=False)
    pair_order = [
        ("zero_shot", "clean_flat"),
        ("zero_shot", "segment_flat"),
        ("zero_shot", "skill_memory"),
        ("clean_flat", "segment_flat"),
        ("clean_flat", "skill_memory"),
        ("segment_flat", "skill_memory"),
    ]
    comparisons = {
        f"{left}_to_{right}": g2b_comparison_summary(frame, left, right, position + 1)
        for position, (left, right) in enumerate(pair_order)
    }
    organization = comparisons["segment_flat_to_skill_memory"]
    if (
        organization["skill_memory_minus_segment_flat"] > 0
        and organization["mcnemar_exact_p"] < 0.05
    ):
        selected_ending = "organization_significant"
    elif organization["mcnemar_exact_p"] >= 0.05:
        selected_ending = "segment_flat_matches_skill"
    else:
        selected_ending = "both_components"
    summary = {
        "task": "Amendment4_1_G2b_segment_flat",
        "status": "PASS",
        "campaign_status": "VIKI_GENERATION_CLOSED",
        "samples": len(frame),
        "model_generation_calls": len(frame),
        "pool": {
            "per_row_allowed_instance_ids": 10359,
            "source_rows_represented": 4993,
            "single_unit_fraction": 1.0,
            "covering_segments": 0,
        },
        "token_match": {
            "criterion": "per-row",
            "tolerance": TOKEN_TOLERANCE,
            "maximum_relative_difference": float(
                frame["segment_flat_relative_difference"].max()
            ),
            "violations": int(
                (frame["segment_flat_relative_difference"] > TOKEN_TOLERANCE).sum()
            ),
        },
        "reuse": {
            "original_g2_run_fingerprint": g2_hash,
            "new_run_fingerprint": run_hash,
            "zero_shot_outputs_reused": len(frame),
            "clean_flat_outputs_reused": len(frame),
            "skill_memory_outputs_reused": len(frame),
        },
        "arms": {
            arm: {
                "successes": int((frame[f"{arm}_task_score"] == 1).sum()),
                "accuracy": float(frame[f"{arm}_task_score"].mean()),
                "format_compliance": float((frame[f"{arm}_format_score"] == 1).mean()),
            }
            for arm in (
                "zero_shot",
                "clean_flat",
                "segment_flat",
                "skill_memory",
            )
        },
        "comparisons": comparisons,
        "decomposition": {
            "content_and_pool": comparisons["zero_shot_to_segment_flat"],
            "organization": organization,
        },
        "scope_note": (
            "Organization here means the skill arm's routing, grouping, and "
            "ordering jointly, G2b does not separate those three, and routing "
            "is part of the structure being claimed. A further routed-but-ungrouped "
            "arm is possible at another 400 calls and is not required for the "
            "paper's claim."
        ),
        "selected_ending": selected_ending,
        "further_viki_generation_authorized": False,
        "results": str(G2B_OUTPUT_PATH),
        "parquet": str(parquet_path),
    }
    atomic_json(G2B_OUTPUT_PATH.with_suffix(".summary.json"), summary)
    atomic_json(
        GENERATION_CLOSED_PATH,
        {
            "status": "CLOSED",
            "closed_by": "Amendment 4.1 G2b",
            "completed_generation_calls": len(frame),
            "final_run_fingerprint": run_hash,
            "summary": str(G2B_OUTPUT_PATH.with_suffix(".summary.json")),
            "summary_sha256": file_sha256(G2B_OUTPUT_PATH.with_suffix(".summary.json")),
            "further_viki_generation_authorized": False,
            "next_deliverable": "chapter draft",
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VIKI Amendment 4")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    preflight = subparsers.add_parser("g2-preflight")
    run = subparsers.add_parser("g2-run")
    g2b_gates = subparsers.add_parser("g2b-gates")
    g2b_preflight = subparsers.add_parser("g2b-preflight")
    g2b_run = subparsers.add_parser("g2b-run")
    subparsers.add_parser("g2b-posttrim-render")
    for subparser in (
        preflight,
        run,
        g2b_gates,
        g2b_preflight,
        g2b_run,
    ):
        subparser.add_argument("--base-url", dest="base_urls", action="append")
        subparser.add_argument("--api-key-env", default="OPENAI_API_KEY")
        subparser.add_argument("--model", default="qwen2.5-vl-72b-amendment3-f2")
        subparser.add_argument("--max-tokens", type=int, default=2000)
        subparser.add_argument("--subgoal-max-tokens", type=int, default=128)
        subparser.add_argument("--workers", type=int, default=8)
        subparser.add_argument("--max-retries", type=int, default=5)
        subparser.add_argument("--timeout", type=float, default=3600)
        subparser.add_argument("--embedding-model", default="all-mpnet-base-v2")
        subparser.add_argument("--embedding-device", default="cpu")
    args = parser.parse_args()
    if hasattr(args, "base_urls") and args.base_urls is None:
        args.base_urls = ["http://127.0.0.1:8050/v1"]
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.command == "audit":
            result = run_audit()
        elif args.command == "g2-preflight":
            result = run_g2_preflight(args)
        elif args.command == "g2b-gates":
            result = run_g2b_gates(args)
        elif args.command == "g2b-preflight":
            result = run_g2b_preflight(args)
        elif args.command == "g2b-posttrim-render":
            result = run_g2b_posttrim_render_gate()
        elif args.command == "g2b-run":
            result = run_g2b(args)
        else:
            result = run_g2(args)
    except GateFailure as error:
        print(f"GATE FAILURE: {error}")
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
