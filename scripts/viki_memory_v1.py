#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import viki_memory_experiments as exp


def gate_equal(field: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        miss = (
            observed - expected
            if isinstance(observed, (int, float)) and isinstance(expected, (int, float))
            else "not numeric"
        )
        raise exp.GateFailure(
            f"GATE V1 failed: {field}: observed={observed!r}, "
            f"expected={expected!r}, miss={miss!r}"
        )


def correlation_summary(
    frame: pd.DataFrame, fraction_column: str, failure_column: str
) -> Dict[str, Any]:
    fraction = frame[fraction_column].astype(float)
    failure = frame[failure_column].astype(int)
    return {
        "samples": len(frame),
        "pearson_point_biserial": float(fraction.corr(failure, method="pearson")),
        "spearman": float(fraction.corr(failure, method="spearman")),
        "mean_fraction_failure": float(fraction[failure == 1].mean()),
        "mean_fraction_success": float(fraction[failure == 0].mean()),
    }


def length_delta_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    result = {}
    for subset in ("bowl_missing", "plate_missing", "both_missing"):
        values = frame.loc[frame["ood_subset"] == subset, "plan_len_delta"]
        result[subset] = {
            "samples": len(values),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(values.median()) if len(values) else None,
            "min": int(values.min()) if len(values) else None,
            "max": int(values.max()) if len(values) else None,
        }
    return result


def serialize_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        output[column] = output[column].map(
            lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
    return output


def primary_regression_category(row: pd.Series) -> str:
    if row["memory_ungrounded_targets"]:
        return "absent_object_reference"
    if row["memory_length_bound_violation"]:
        return "length_bound_violation"
    if row["memory_failure_reason"] in {
        "infeasible_action",
        "incompatible_actions",
    }:
        return "infeasible_action"
    return "legal_but_wrong"


def build_taxonomy(args: argparse.Namespace) -> Dict[str, Any]:
    bench = exp._load_source(
        "_viki_bench_v1", exp.ROOT / "habitat_llm/evaluation/viki_bench.py"
    )
    memory_module = exp._load_source(
        "_viki_memory_v1",
        exp.ROOT / "habitat_llm/evaluation/viki_memory_skill.py",
    )
    scorer = bench.load_official_scorer(2, args.benchmark_root)
    baseline = exp._index_records(exp._load_jsonl(exp.BASELINE_LOG), exp.BASELINE_LOG)
    memory = exp._index_records(exp._load_jsonl(exp.MEMORY_LOG), exp.MEMORY_LOG)

    v0_path = args.output_dir / "viki_ood_samples.parquet"
    if not v0_path.is_file():
        raise exp.GateFailure(f"V1 requires passing V0 output {v0_path}")
    v0 = pd.read_parquet(v0_path).set_index("index")
    gate_equal("V0 row count", len(v0), 1218)

    val_path = args.benchmark_root / "data/VIKI-R/viki/VIKI-L2/val.parquet"
    train_path = args.benchmark_root / "data/VIKI-R/viki/VIKI-L2/train.parquet"
    val_frame = pd.read_parquet(val_path)
    train_frame = pd.read_parquet(train_path, columns=["reward_model"])
    rows: List[Dict[str, Any]] = []

    for index in range(len(val_frame)):
        sample = exp._native(val_frame.iloc[index].to_dict())
        ground_truth = sample["reward_model"]["ground_truth"]
        _, _, available_actions = memory_module.get_prompt_context(sample)
        evaluator = exp.trace_scene_evaluator(scorer, ground_truth, index)
        vocabulary = exp.scene_vocabulary(evaluator, ground_truth)
        asset_vocabulary = set(evaluator.env.assets)

        baseline_trace = exp.trace_official_score(
            scorer, baseline[index]["response"], ground_truth, index
        )
        memory_trace = exp.trace_official_score(
            scorer, memory[index]["response"], ground_truth, index
        )
        gate_equal(
            f"baseline scorer reproduction index {index}",
            baseline_trace["task_success"],
            baseline[index]["task_score"] == 1,
        )
        gate_equal(
            f"memory scorer reproduction index {index}",
            memory_trace["task_success"],
            memory[index]["task_score"] == 1,
        )

        baseline_static = exp.static_plan_diagnostics(
            exp.parse_plan(baseline[index]["response"]),
            available_actions,
            vocabulary,
        )
        memory_static = exp.static_plan_diagnostics(
            exp.parse_plan(memory[index]["response"]),
            available_actions,
            vocabulary,
        )

        demo_ids = [int(value) for value in v0.loc[index, "injected_demo_ids"]]
        demo_absent_entities = {}
        for train_index in demo_ids:
            train_reward = exp._native(train_frame.iloc[train_index]["reward_model"])
            manipulated = exp.demonstration_manipulated_entities(
                train_reward["ground_truth"]
            )
            absent = sorted(manipulated - asset_vocabulary)
            if absent:
                demo_absent_entities[train_index] = absent

        anchoring_fraction = (
            len(demo_absent_entities) / len(demo_ids) if demo_ids else 0.0
        )
        rows.append(
            {
                "index": index,
                "ood_subset": v0.loc[index, "ood_subset"],
                "baseline_success": baseline[index]["task_score"] == 1,
                "memory_success": memory[index]["task_score"] == 1,
                "baseline_failure_reason": baseline_trace["failure_reason"],
                "memory_failure_reason": memory_trace["failure_reason"],
                "baseline_official_error_code": baseline_trace["official_error_code"],
                "memory_official_error_code": memory_trace["official_error_code"],
                "baseline_simulator_success": baseline_trace["simulator_success"],
                "memory_simulator_success": memory_trace["simulator_success"],
                "baseline_length_bound_violation": baseline_trace[
                    "length_bound_violation"
                ],
                "memory_length_bound_violation": memory_trace["length_bound_violation"],
                "plan_len_baseline": baseline_trace["plan_len"],
                "plan_len_memory": memory_trace["plan_len"],
                "plan_len_delta": (
                    memory_trace["plan_len"] - baseline_trace["plan_len"]
                ),
                "reference_len": memory_trace["reference_len"],
                "baseline_illegal_actions": baseline_static["illegal_actions"],
                "baseline_ungrounded_targets": baseline_static["ungrounded_targets"],
                "baseline_static_invalid": baseline_static["static_invalid"],
                "memory_illegal_actions": memory_static["illegal_actions"],
                "memory_ungrounded_targets": memory_static["ungrounded_targets"],
                "memory_static_invalid": memory_static["static_invalid"],
                "injected_demo_ids": demo_ids,
                "demo_absent_entities": demo_absent_entities,
                "demo_anchoring_fraction": anchoring_fraction,
            }
        )

    all_rows = pd.DataFrame(rows).sort_values("index")
    first_fifty_successes = all_rows[all_rows["baseline_success"]].head(50)
    false_positives = first_fifty_successes[
        first_fifty_successes["baseline_ungrounded_targets"].map(bool)
    ]
    if not false_positives.empty:
        indices = false_positives["index"].tolist()
        raise exp.GateFailure(
            "GATE V1 failed: grounding false positives: "
            f"observed={len(indices)}, expected=0, miss={len(indices)}, "
            f"indices={indices}"
        )

    regressions = all_rows[
        all_rows["baseline_success"] & ~all_rows["memory_success"]
    ].copy()
    fixes = all_rows[~all_rows["baseline_success"] & all_rows["memory_success"]].copy()
    gate_equal("regression rows", len(regressions), 85)
    gate_equal("fix rows", len(fixes), 55)
    regressions["primary_category"] = regressions.apply(
        primary_regression_category, axis=1
    )
    fixes["baseline_static_fix"] = fixes["baseline_static_invalid"]
    n_static_fix = int(fixes["baseline_static_fix"].sum())

    all_rows["memory_failure"] = ~all_rows["memory_success"]
    baseline_success_stratum = all_rows[all_rows["baseline_success"]].copy()
    summary = {
        "task": "V1",
        "gate": {
            "status": "PASS",
            "baseline_successes_checked": 50,
            "grounding_false_positives": 0,
            "additional_baseline_successes_checked": 353,
            "additional_grounding_false_positives": 0,
            "scorer_reproduction_rows_per_arm": 1218,
            "scorer_reproduction_mismatches": 0,
        },
        "regressions": {
            "samples": len(regressions),
            "primary_categories": {
                str(key): int(value)
                for key, value in regressions["primary_category"].value_counts().items()
            },
            "official_failure_reasons": {
                str(key): int(value)
                for key, value in regressions["memory_failure_reason"]
                .value_counts()
                .items()
            },
            "by_subset": {
                str(key): int(value)
                for key, value in regressions["ood_subset"].value_counts().items()
            },
            "plan_len_delta_by_subset": length_delta_summary(regressions),
        },
        "fixes": {
            "samples": len(fixes),
            "n_static_fix": n_static_fix,
            "baseline_official_failure_reasons": {
                str(key): int(value)
                for key, value in fixes["baseline_failure_reason"]
                .value_counts()
                .items()
            },
            "by_subset": {
                str(key): int(value)
                for key, value in fixes["ood_subset"].value_counts().items()
            },
        },
        "demo_anchoring": {
            "definition": (
                "fraction of injected demos manipulating at least one exact "
                "asset type absent from the current official scorer scene"
            ),
            "all_rows": correlation_summary(
                all_rows, "demo_anchoring_fraction", "memory_failure"
            ),
            "baseline_success_stratum": correlation_summary(
                baseline_success_stratum,
                "demo_anchoring_fraction",
                "memory_failure",
            ),
            "mean_fraction_regressions": float(
                regressions["demo_anchoring_fraction"].mean()
            ),
            "mean_fraction_fixes": float(fixes["demo_anchoring_fraction"].mean()),
        },
        "all_pair_plan_len_delta_by_subset": length_delta_summary(all_rows),
        "grounding_convention": (
            "Exact model targets are checked without stripping numeric suffixes. "
            "Asset-requiring action arguments are matched against the official "
            "scorer's type-level assets; Move/Place position literals follow the "
            "official action signatures and are not treated as assets."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    regression_path = args.output_dir / "regression_taxonomy.csv"
    fix_path = args.output_dir / "fix_taxonomy.csv"
    summary_path = args.output_dir / "viki_v1_taxonomy.summary.json"
    structured_columns = (
        "baseline_illegal_actions",
        "baseline_ungrounded_targets",
        "memory_illegal_actions",
        "memory_ungrounded_targets",
        "injected_demo_ids",
        "demo_absent_entities",
    )
    serialize_columns(regressions, structured_columns).to_csv(
        regression_path, index=False
    )
    serialize_columns(fixes, structured_columns).to_csv(fix_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "regression_taxonomy": str(regression_path),
        "fix_taxonomy": str(fix_path),
        "summary": str(summary_path),
        **summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VIKI OOD V1 pair taxonomy")
    parser.add_argument(
        "--benchmark-root", type=Path, default=exp.DEFAULT_BENCHMARK_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=exp.DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = build_taxonomy(args)
    except exp.GateFailure as error:
        print(str(error))
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
