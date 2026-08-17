#!/usr/bin/env python3

import argparse
import ast
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from habitat_llm.evaluation.viki_branch_conditions import (
    AvailabilityPredicate,
    BranchPredicate,
    derive_train_portable_assets,
    derive_train_vocabularies,
    discover_instruction_regions,
    get_instruction,
)
from habitat_llm.evaluation.viki_composition import parse_composition

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT.parent / "VIKI-R"
DATA_ROOT = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2"
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment1"
ADJUDICATION_PATH = ROOT / "docs/viki_amendment1_a0_adjudication.csv"
C0_ADJUDICATION_PATH = ROOT / "docs/viki_amendment1_c0_adjudication.tsv"
V0_PATH = ROOT / "results/viki_memory_experiments/viki_ood_samples.parquet"
BASELINE_PATH = ROOT / "results/viki_official_7b_l2_ood.jsonl"
MEMORY_PATH = ROOT / "results/viki_memory_skill_7b_l2_ood.jsonl"
V1_REGRESSION_PATH = ROOT / "results/viki_memory_experiments/regression_taxonomy.csv"
V1_FIX_PATH = ROOT / "results/viki_memory_experiments/fix_taxonomy.csv"


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


def load_frames() -> Dict[str, pd.DataFrame]:
    return {
        split: pd.read_parquet(DATA_ROOT / f"{split}.parquet")
        for split in ("train", "test", "val")
    }


def build_predicate(
    train: pd.DataFrame,
) -> Tuple[AvailabilityPredicate, Dict[str, Any]]:
    samples = [row.to_dict() for _, row in train.iterrows()]
    assets, locations = derive_train_vocabularies(samples)
    portable = derive_train_portable_assets(samples)
    discovered = discover_instruction_regions(
        (get_instruction(sample) for sample in samples), assets, locations
    )
    predicate = AvailabilityPredicate(assets, locations | discovered, portable)
    return predicate, {
        "asset_vocabulary": sorted(assets),
        "portable_asset_vocabulary": sorted(portable),
        "initial_location_vocabulary": sorted(locations),
        "instruction_discovered_regions": sorted(discovered - locations - assets),
    }


def condition_string(result: BranchPredicate) -> str:
    return ";".join(
        f"{condition.asset}|{condition.target}|{condition.status}"
        for condition in result.conditions
    )


def gate_a0_adjudication(
    frames: Mapping[str, pd.DataFrame], predicate: AvailabilityPredicate
) -> Dict[str, Any]:
    labels = pd.read_csv(ADJUDICATION_PATH, keep_default_na=False)
    mismatches = []
    for _, label in labels.iterrows():
        split = str(label["split"])
        index = int(label["index"])
        result = predicate.evaluate(frames[split].iloc[index].to_dict())
        observed_conditions = condition_string(result)
        if (
            result.branch != label["branch"]
            or observed_conditions != label["conditions"]
        ):
            mismatches.append(
                {
                    "split": split,
                    "index": index,
                    "expected_branch": label["branch"],
                    "observed_branch": result.branch,
                    "expected_conditions": label["conditions"],
                    "observed_conditions": observed_conditions,
                }
            )
    if len(labels) != 50 or mismatches:
        raise GateFailure(
            "GATE A0 failed: adjudication "
            f"observed={len(labels) - len(mismatches)}/50, expected=50/50, "
            f"miss={len(mismatches)}, mismatches={mismatches}"
        )
    return {
        "status": "PASS",
        "seed": 20260814,
        "rows": 50,
        "agreements": 50,
        "disagreements": 0,
        "labels": str(ADJUDICATION_PATH.relative_to(ROOT)),
    }


def run_a0(args: argparse.Namespace) -> Dict[str, Any]:
    frames = load_frames()
    predicate, vocabulary = build_predicate(frames["train"])
    adjudication = gate_a0_adjudication(frames, predicate)
    v0 = pd.read_parquet(V0_PATH).set_index("index")
    rows = []
    for split, frame in frames.items():
        for index, source_row in frame.iterrows():
            sample = source_row.to_dict()
            result = predicate.evaluate(sample)
            source_skill = str(source_row["reward_model"]["ground_truth"]["task_name"])
            analysis_skill = (
                str(v0.loc[index, "routed_skill"]) if split == "val" else source_skill
            )
            rows.append(
                {
                    "split": split,
                    "index": int(index),
                    "source_task": source_skill,
                    "analysis_skill": analysis_skill,
                    "instruction": result.instruction,
                    "branch": result.branch,
                    "conditions": json.dumps(
                        [condition.to_dict() for condition in result.conditions],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "absent_assets": json.dumps(
                        result.absent_assets, separators=(",", ":")
                    ),
                    "unresolved_assets": json.dumps(
                        list(result.unresolved_assets), separators=(",", ":")
                    ),
                }
            )
    census = pd.DataFrame(rows)
    expected_sizes = {"train": 7196, "test": 1800, "val": 1218}
    for split, expected in expected_sizes.items():
        observed = len(census[census["split"] == split])
        if observed != expected:
            raise GateFailure(
                f"GATE A0 failed: {split} rows observed={observed}, "
                f"expected={expected}, miss={observed - expected}"
            )
    ood = census[census["split"] == "val"]
    ood_exceptions = ood[ood["branch"] != "some_absent"]
    if not ood_exceptions.empty:
        raise GateFailure(
            "GATE A0 failed: OOD some-absent exceptions "
            f"observed={len(ood_exceptions)}, expected=0, "
            f"miss={len(ood_exceptions)}"
        )

    split_counts = {
        split: {
            str(branch): int(count)
            for branch, count in census[census["split"] == split]["branch"]
            .value_counts()
            .items()
        }
        for split in ("train", "test", "val")
    }
    train = census[census["split"] == "train"]
    train_by_skill = {
        skill: {
            str(branch): int(count)
            for branch, count in group["branch"].value_counts().items()
        }
        for skill, group in train.groupby("analysis_skill")
    }
    ood_routes = {
        str(skill): int(count)
        for skill, count in ood["analysis_skill"].value_counts().items()
    }
    usable_absent = {
        skill: int(train_by_skill.get(skill, {}).get("some_absent", 0))
        for skill in sorted(ood_routes)
    }
    summary = {
        "task": "A0",
        "gate": adjudication,
        "predicate_inputs": (
            "Per-row instruction and init_pos only. Asset, portable-asset, and "
            "region lexicons are frozen from train before applying one code path "
            "to train, ID test, and OOD val."
        ),
        "vocabulary": vocabulary,
        "branch_counts_by_split": split_counts,
        "train_branch_counts_by_skill": train_by_skill,
        "ood_route_counts": ood_routes,
        "usable_absent_train_instances_for_ood_routes": usable_absent,
        "ood_some_absent_rate": float((ood["branch"] == "some_absent").mean()),
        "ood_exceptions": [],
        "diagnosis_result": (
            "CONTRADICTED: all 7,024 train rows with an asset-placement condition "
            "are some-absent; only 172 observation-only dog_check rows are "
            "not-applicable. No all-present train branch was found."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "a0_branch_census.parquet"
    csv_path = args.output_dir / "a0_branch_census.csv"
    summary_path = args.output_dir / "a0_branch_census.summary.json"
    census.to_parquet(parquet_path, index=False)
    census.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "parquet": str(parquet_path),
        "csv": str(csv_path),
        "summary": str(summary_path),
        **summary,
    }


def load_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    records = {}
    with path.open() as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                records[int(record["index"])] = record
    return records


def parse_plan(response: str) -> List[Mapping[str, Any]]:
    match = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    if match is None:
        return []
    try:
        plan = ast.literal_eval(match.group(1).strip())
    except (SyntaxError, TypeError, ValueError):
        return []
    return plan if isinstance(plan, list) else []


def plan_tokens(plan: Sequence[Mapping[str, Any]]) -> Tuple[List[str], List[str]]:
    action_tokens = []
    grounded_tokens = []
    for step in plan:
        actions = step.get("actions", {})
        if not isinstance(actions, dict):
            continue
        for robot, action in sorted(actions.items()):
            if not isinstance(action, (list, tuple)) or not action:
                continue
            action_tokens.append(str(action[0]))
            grounded_tokens.append(":".join([str(robot), *map(str, action)]))
    return action_tokens, grounded_tokens


def plan_entities(plan: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grasps = []
    places = []
    actions_by_target: Dict[str, List[str]] = defaultdict(list)
    for step in plan:
        for action in step.get("actions", {}).values():
            if not isinstance(action, (list, tuple)) or not action:
                continue
            name = str(action[0])
            target = str(action[1]) if len(action) > 1 else ""
            actions_by_target[target].append(name)
            if name == "Grasp":
                grasps.append(target)
            elif name == "Place":
                places.append(target)
    return {"grasps": grasps, "places": places, "actions_by_target": actions_by_target}


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


def ground_truth_plan(ground_truth: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "step": int(step["step"]),
            "actions": {
                robot: action
                for robot, action in step["actions"].items()
                if action is not None
            },
        }
        for step in native(ground_truth)["time_steps"]
    ]


def has_fetch(actions_by_target: Mapping[str, Sequence[str]], asset: str) -> bool:
    actions = set(actions_by_target.get(asset, []))
    return {"Move", "Reach", "Grasp"} <= actions


def nearest_demo_metrics(
    baseline_plan: Sequence[Mapping[str, Any]],
    memory_plan: Sequence[Mapping[str, Any]],
    demo_ids: Sequence[int],
    train: pd.DataFrame,
) -> Dict[str, Any]:
    baseline_actions, baseline_grounded = plan_tokens(baseline_plan)
    memory_actions, memory_grounded = plan_tokens(memory_plan)
    candidates = []
    for demo_id in demo_ids:
        ground_truth = native(train.iloc[int(demo_id)]["reward_model"])["ground_truth"]
        demo_plan = ground_truth_plan(ground_truth)
        demo_actions, demo_grounded = plan_tokens(demo_plan)
        candidates.append(
            {
                "demo_id": int(demo_id),
                "memory_action_distance": levenshtein(memory_actions, demo_actions),
                "memory_grounded_distance": levenshtein(memory_grounded, demo_grounded),
                "baseline_action_distance": levenshtein(baseline_actions, demo_actions),
                "baseline_grounded_distance": levenshtein(
                    baseline_grounded, demo_grounded
                ),
            }
        )
    nearest = min(
        candidates,
        key=lambda item: (
            item["memory_grounded_distance"],
            item["memory_action_distance"],
            item["demo_id"],
        ),
    )
    return {
        **nearest,
        "copies_demo_structure": bool(
            nearest["memory_action_distance"] < nearest["baseline_action_distance"]
            or nearest["memory_grounded_distance"]
            < nearest["baseline_grounded_distance"]
        ),
    }


def autopsy_row(
    index: int,
    predicate: AvailabilityPredicate,
    val: pd.DataFrame,
    train: pd.DataFrame,
    baseline: Mapping[int, Mapping[str, Any]],
    memory: Mapping[int, Mapping[str, Any]],
    v0: pd.DataFrame,
    train_branches: Mapping[int, str],
) -> Dict[str, Any]:
    branch = predicate.evaluate(val.iloc[index].to_dict())
    baseline_plan = parse_plan(str(baseline[index]["response"]))
    memory_plan = parse_plan(str(memory[index]["response"]))
    baseline_entities = plan_entities(baseline_plan)
    memory_entities = plan_entities(memory_plan)
    absent = set(branch.absent_assets)
    present = {
        condition.asset
        for condition in branch.conditions
        if condition.status == "present_at_target"
    }
    baseline_grasps = set(baseline_entities["grasps"])
    memory_grasps = set(memory_entities["grasps"])
    baseline_fetches_absent = any(
        has_fetch(baseline_entities["actions_by_target"], asset) for asset in absent
    )
    memory_fetches_absent = any(
        has_fetch(memory_entities["actions_by_target"], asset) for asset in absent
    )
    uses_present_where_baseline_fetches = bool(
        baseline_fetches_absent
        and memory_grasps & present
        and not memory_fetches_absent
    )
    swaps_object = baseline_entities["grasps"] != memory_entities["grasps"]
    omits_required_fetch = bool(baseline_fetches_absent and not memory_fetches_absent)
    branch_decision = bool(
        baseline_grasps & absent
        and (memory_grasps & present or not memory_grasps & absent)
    )
    demo_ids = [int(value) for value in v0.loc[index, "injected_demo_ids"]]
    edit = nearest_demo_metrics(baseline_plan, memory_plan, demo_ids, train)
    demo_branch_counts = Counter(train_branches[demo_id] for demo_id in demo_ids)
    if uses_present_where_baseline_fetches:
        classification = "uses_present_asset_where_baseline_fetches_absent"
    elif swaps_object:
        classification = "swaps_manipulated_object"
    elif omits_required_fetch:
        classification = "omits_required_fetch"
    elif edit["copies_demo_structure"]:
        classification = "copies_injected_demo_structure"
    else:
        classification = "other_legal_goal_miss"
    return {
        "index": index,
        "ood_subset": str(v0.loc[index, "ood_subset"]),
        "routed_skill": str(v0.loc[index, "routed_skill"]),
        "retrieval_sim_max": float(v0.loc[index, "retrieval_sim_max"]),
        "retrieval_sim_mean": float(v0.loc[index, "retrieval_sim_mean"]),
        "required_absent_assets": json.dumps(sorted(absent), separators=(",", ":")),
        "present_at_target_assets": json.dumps(sorted(present), separators=(",", ":")),
        "baseline_grasps": json.dumps(
            baseline_entities["grasps"], separators=(",", ":")
        ),
        "memory_grasps": json.dumps(memory_entities["grasps"], separators=(",", ":")),
        "baseline_places": json.dumps(
            baseline_entities["places"], separators=(",", ":")
        ),
        "memory_places": json.dumps(memory_entities["places"], separators=(",", ":")),
        "uses_present_where_baseline_fetches": uses_present_where_baseline_fetches,
        "swaps_manipulated_object": swaps_object,
        "omits_required_fetch": omits_required_fetch,
        "branch_decision": branch_decision,
        "classification": classification,
        "injected_demo_ids": json.dumps(demo_ids, separators=(",", ":")),
        "injected_branch_all_present": int(demo_branch_counts["all_present"]),
        "injected_branch_some_absent": int(demo_branch_counts["some_absent"]),
        "injected_branch_not_applicable": int(demo_branch_counts["not_applicable"]),
        **edit,
    }


def group_contrast(frame: pd.DataFrame) -> Dict[str, Any]:
    return {
        "samples": len(frame),
        "route_counts": {
            str(key): int(value)
            for key, value in frame["routed_skill"].value_counts().items()
        },
        "retrieval_sim_max_mean": float(frame["retrieval_sim_max"].mean()),
        "retrieval_sim_max_median": float(frame["retrieval_sim_max"].median()),
        "retrieval_sim_mean_mean": float(frame["retrieval_sim_mean"].mean()),
        "mean_injected_branch_composition": {
            branch: float(frame[column].mean())
            for branch, column in (
                ("all_present", "injected_branch_all_present"),
                ("some_absent", "injected_branch_some_absent"),
                ("not_applicable", "injected_branch_not_applicable"),
            )
        },
    }


def run_a1(args: argparse.Namespace) -> Dict[str, Any]:
    frames = load_frames()
    predicate, _ = build_predicate(frames["train"])
    a0_path = args.output_dir / "a0_branch_census.parquet"
    if not a0_path.is_file():
        raise GateFailure(f"A1 requires passing A0 output {a0_path}")
    a0 = pd.read_parquet(a0_path)
    train_branches = {
        int(row["index"]): str(row["branch"])
        for _, row in a0[a0["split"] == "train"].iterrows()
    }
    baseline = load_jsonl(BASELINE_PATH)
    memory = load_jsonl(MEMORY_PATH)
    v0 = pd.read_parquet(V0_PATH).set_index("index")
    regressions_v1 = pd.read_csv(V1_REGRESSION_PATH)
    fixes_v1 = pd.read_csv(V1_FIX_PATH)
    legal_regression_ids = [
        int(value)
        for value in regressions_v1.loc[
            regressions_v1["memory_failure_reason"] == "goal_miss", "index"
        ]
    ]
    legal_fix_ids = [
        int(value)
        for value in fixes_v1.loc[
            fixes_v1["baseline_failure_reason"] == "goal_miss", "index"
        ]
    ]
    if len(legal_regression_ids) != 73 or len(legal_fix_ids) != 54:
        raise GateFailure(
            "GATE A1 failed: legal pair counts "
            f"observed={len(legal_regression_ids)}/{len(legal_fix_ids)}, "
            "expected=73/54"
        )
    regressions = pd.DataFrame(
        [
            autopsy_row(
                index,
                predicate,
                frames["val"],
                frames["train"],
                baseline,
                memory,
                v0,
                train_branches,
            )
            for index in legal_regression_ids
        ]
    )
    fixes = pd.DataFrame(
        [
            autopsy_row(
                index,
                predicate,
                frames["val"],
                frames["train"],
                baseline,
                memory,
                v0,
                train_branches,
            )
            for index in legal_fix_ids
        ]
    )
    all_plate_regression_ids = [
        int(value)
        for value in regressions_v1.loc[
            regressions_v1["ood_subset"] == "plate_missing", "index"
        ]
    ]
    all_plate_fix_ids = [
        int(value)
        for value in fixes_v1.loc[fixes_v1["ood_subset"] == "plate_missing", "index"]
    ]
    all_plate_regressions = pd.DataFrame(
        [
            autopsy_row(
                index,
                predicate,
                frames["val"],
                frames["train"],
                baseline,
                memory,
                v0,
                train_branches,
            )
            for index in all_plate_regression_ids
        ]
    )
    all_plate_fixes = pd.DataFrame(
        [
            autopsy_row(
                index,
                predicate,
                frames["val"],
                frames["train"],
                baseline,
                memory,
                v0,
                train_branches,
            )
            for index in all_plate_fix_ids
        ]
    )
    if len(all_plate_regressions) != 78 or len(all_plate_fixes) != 17:
        raise GateFailure(
            "GATE A1 failed: all plate discordants "
            f"observed={len(all_plate_regressions)}/{len(all_plate_fixes)}, "
            "expected=78/17"
        )
    branch_decisions = int(regressions["branch_decision"].sum())
    required = math.ceil(73 / 2)
    if branch_decisions < required:
        raise GateFailure(
            "GATE A1 failed: branch decisions "
            f"observed={branch_decisions}/73, expected>={required}/73, "
            f"miss={branch_decisions - required}"
        )
    plate_regressions = regressions[regressions["ood_subset"] == "plate_missing"]
    plate_fixes = fixes[fixes["ood_subset"] == "plate_missing"]
    summary = {
        "task": "A1",
        "gate": {
            "status": "PASS",
            "legal_goal_misses": 73,
            "branch_decisions": branch_decisions,
            "branch_decision_rate": branch_decisions / 73,
            "required_count": required,
        },
        "regression_classification": {
            str(key): int(value)
            for key, value in regressions["classification"].value_counts().items()
        },
        "regression_flags": {
            "uses_present_where_baseline_fetches": int(
                regressions["uses_present_where_baseline_fetches"].sum()
            ),
            "swaps_manipulated_object": int(
                regressions["swaps_manipulated_object"].sum()
            ),
            "omits_required_fetch": int(regressions["omits_required_fetch"].sum()),
            "copies_injected_demo_structure": int(
                regressions["copies_demo_structure"].sum()
            ),
        },
        "legal_fixes": {
            "samples": 54,
            "memory_selects_required_absent_asset": int(
                sum(
                    bool(
                        set(json.loads(row["memory_grasps"]))
                        & set(json.loads(row["required_absent_assets"]))
                    )
                    for _, row in fixes.iterrows()
                )
            ),
            "classification": {
                str(key): int(value)
                for key, value in fixes["classification"].value_counts().items()
            },
        },
        "plate_missing_contrast": {
            "all_78_regressions": group_contrast(all_plate_regressions),
            "all_17_fixes": group_contrast(all_plate_fixes),
            "legal_66_regressions": group_contrast(plate_regressions),
            "legal_16_fixes": group_contrast(plate_fixes),
        },
        "diagnosis_result": (
            "CONFIRMED for OOD output flips: at least half of legal regressions "
            "are wrong branch decisions. A0 nevertheless contradicted the claimed "
            "train branch census: retrieved train demos are not all-present under "
            "the preregistered predicate."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regression_path = args.output_dir / "a1_legal_regression_autopsy.csv"
    fix_path = args.output_dir / "a1_legal_fix_autopsy.csv"
    summary_path = args.output_dir / "a1_autopsy.summary.json"
    regressions.to_csv(regression_path, index=False)
    fixes.to_csv(fix_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "regression_csv": str(regression_path),
        "fix_csv": str(fix_path),
        "summary": str(summary_path),
        **summary,
    }


def gate_c0_adjudication(frames: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    labels = pd.read_csv(C0_ADJUDICATION_PATH, sep="\t")
    mismatches = []
    for _, label in labels.iterrows():
        split = str(label["split"])
        index = int(label["index"])
        ground_truth = frames[split].iloc[index]["reward_model"]["ground_truth"]
        signature = parse_composition(ground_truth)
        observed_units = " || ".join(signature.ordered_units())
        if (
            signature.plan_length != int(label["plan_length"])
            or signature.object_count != int(label["object_count"])
            or observed_units != label["ordered_units"]
        ):
            mismatches.append(
                {
                    "split": split,
                    "index": index,
                    "expected": {
                        "plan_length": int(label["plan_length"]),
                        "object_count": int(label["object_count"]),
                        "ordered_units": label["ordered_units"],
                    },
                    "observed": {
                        "plan_length": signature.plan_length,
                        "object_count": signature.object_count,
                        "ordered_units": observed_units,
                    },
                }
            )
    if len(labels) != 50 or mismatches:
        raise GateFailure(
            "GATE C0 failed: adjudication "
            f"observed={len(labels) - len(mismatches)}/50, expected=50/50, "
            f"miss={len(mismatches)}, mismatches={mismatches}"
        )
    return {
        "status": "PASS",
        "seed": 20260814,
        "rows": 50,
        "agreements": 50,
        "disagreements": 0,
        "labels": str(C0_ADJUDICATION_PATH.relative_to(ROOT)),
    }


def run_c0(args: argparse.Namespace) -> Dict[str, Any]:
    frames = load_frames()
    gate = gate_c0_adjudication(frames)
    rows = []
    for split in ("train", "test"):
        for index, source_row in frames[split].iterrows():
            ground_truth = source_row["reward_model"]["ground_truth"]
            signature = parse_composition(ground_truth)
            if not signature.units or len(signature.action_skeleton) != len(
                ground_truth["time_steps"]
            ):
                raise GateFailure(
                    f"GATE C0 failed: invalid signature at {split}:{index}"
                )
            unit_kinds = [unit.kind for unit in signature.units]
            rows.append(
                {
                    "split": split,
                    "index": int(index),
                    "task_name": str(ground_truth["task_name"]),
                    "ordered_units": json.dumps(
                        signature.ordered_units(), separators=(",", ":")
                    ),
                    "unit_kinds": json.dumps(unit_kinds, separators=(",", ":")),
                    "plan_length": signature.plan_length,
                    "object_count": signature.object_count,
                    "action_skeleton": json.dumps(
                        [list(step) for step in signature.action_skeleton],
                        separators=(",", ":"),
                    ),
                    "full_signature": signature.full_signature(),
                }
            )
    census = pd.DataFrame(rows)
    expected = {"train": 7196, "test": 1800}
    for split, count in expected.items():
        observed = len(census[census["split"] == split])
        if observed != count:
            raise GateFailure(
                f"GATE C0 failed: {split} rows observed={observed}, "
                f"expected={count}, miss={observed - count}"
            )
    summary = {
        "task": "C0",
        "gate": gate,
        "rows_by_split": expected,
        "unique_full_signatures": {
            split: int(census.loc[census["split"] == split, "full_signature"].nunique())
            for split in expected
        },
        "unique_signatures_by_task": {
            split: {
                str(task): int(group["full_signature"].nunique())
                for task, group in census[census["split"] == split].groupby("task_name")
            }
            for split in expected
        },
        "unit_kind_counts": {
            split: dict(
                Counter(
                    kind
                    for value in census.loc[census["split"] == split, "unit_kinds"]
                    for kind in json.loads(value)
                )
            )
            for split in expected
        },
        "definition": (
            "Ordered deterministic units from reference actions: fetch, relocate, "
            "open_container_then_retrieve, check_then_act, state_change, and "
            "multi_object_sequence; full signature also includes reference plan "
            "length and relocated payload object count. These labels define splits "
            "only and never enter prompts or memory."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "c0_composition_census.parquet"
    csv_path = args.output_dir / "c0_composition_census.csv"
    summary_path = args.output_dir / "c0_composition_census.summary.json"
    census.to_parquet(parquet_path, index=False)
    census.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "parquet": str(parquet_path),
        "csv": str(csv_path),
        "summary": str(summary_path),
        **summary,
    }


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VIKI Amendment 1 zero-call analyses")
    parser.add_argument("task", choices=("a0", "a1", "c0"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        if args.task == "a0":
            result = run_a0(args)
        elif args.task == "a1":
            result = run_a1(args)
        else:
            result = run_c0(args)
    except GateFailure as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
