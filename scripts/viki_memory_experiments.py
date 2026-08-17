#!/usr/bin/env python3

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = ROOT.parent / "VIKI-R"
DEFAULT_MODEL_PATH = DEFAULT_BENCHMARK_ROOT / "models/Qwen2.5VL-7B-Instruct-VIKI-R-2"
DEFAULT_OUTPUT_DIR = ROOT / "results/viki_memory_experiments"
BASELINE_LOG = ROOT / "results/viki_official_7b_l2_ood.jsonl"
MEMORY_LOG = ROOT / "results/viki_memory_skill_7b_l2_ood.jsonl"
MEMORY_SOURCE_LOGS = (
    ROOT / "results/viki_memory_skill_7b_l2_ood_pilot10_pred768.jsonl",
    ROOT / "results/viki_memory_skill_7b_l2_ood_heldout_10_43.jsonl",
    ROOT / "results/viki_memory_skill_7b_l2_ood_heldout_44_76.jsonl",
    ROOT / "results/viki_memory_skill_7b_l2_ood_heldout_77_109.jsonl",
    ROOT / "results/viki_memory_skill_7b_l2_ood_final_110_386.jsonl",
    ROOT / "results/viki_memory_skill_7b_l2_ood_final_387_663.jsonl",
    ROOT / "results/viki_memory_skill_7b_l2_ood_final_664_940.jsonl",
    ROOT / "results/viki_memory_skill_7b_l2_ood_final_941_1217.jsonl",
)
MODEL_CONTEXT = 4096
PLAN_TOKEN_BUDGET = 2000

EXPECTED_COUNTS = {
    "overall": {
        "samples": 1218,
        "baseline_successes": 403,
        "memory_successes": 373,
    },
    "bowl_missing": {
        "samples": 409,
        "baseline_successes": 13,
        "memory_successes": 47,
    },
    "plate_missing": {
        "samples": 418,
        "baseline_successes": 387,
        "memory_successes": 326,
    },
    "both_missing": {
        "samples": 391,
        "baseline_successes": 3,
        "memory_successes": 0,
    },
}
EXPECTED_FAIL_TO_SUCCESS = 55
EXPECTED_SUCCESS_TO_FAIL = 85
EXPECTED_MCNEMAR_P = 0.01395733412484244

SCORER_REASON_MAP = {
    "INVALID_COMMAND": "invalid_command",
    "NOT_FOUND_ENTITY": "ungrounded_entity",
    "ACTION_NOT_FEASIBLE": "infeasible_action",
    "ACTION_NOT_COMPATIBLE": "incompatible_actions",
    "FAILED_GOAL_CONSTRAINT": "goal_miss",
    "FAILED_TEMPORAL_CONSTRAINT": "temporal_constraint",
}


class GateFailure(RuntimeError):
    pass


def _load_source(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def _index_records(
    records: Iterable[Mapping[str, Any]], path: Path
) -> Dict[int, Mapping[str, Any]]:
    indexed: Dict[int, Mapping[str, Any]] = {}
    for record in records:
        index = int(record["index"])
        if index in indexed:
            raise GateFailure(f"duplicate index {index} in {path}")
        indexed[index] = record
    return indexed


def _assert_equal(field: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        miss: Any
        if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
            miss = observed - expected
        else:
            miss = "not numeric"
        raise GateFailure(
            f"GATE V0 failed: {field}: observed={observed!r}, "
            f"expected={expected!r}, miss={miss!r}"
        )


def _assert_close(
    field: str, observed: float, expected: float, tolerance: float = 1e-12
) -> None:
    miss = observed - expected
    if abs(miss) > tolerance:
        raise GateFailure(
            f"GATE V0 failed: {field}: observed={observed:.15g}, "
            f"expected={expected:.15g}, miss={miss:.15g}"
        )


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if hasattr(value, "tolist"):
        return _native(value.tolist())
    return value


def derive_ood_subset(ground_truth: Mapping[str, Any]) -> str:
    positions: Dict[str, List[str]] = defaultdict(list)
    for entity_id, entity_positions in ground_truth["init_pos"].items():
        if entity_id.startswith("R") and entity_id[1:].isdigit():
            continue
        if entity_positions is None:
            continue
        entity_type = entity_id.rsplit("_", 1)[0]
        positions[entity_type].extend(_native(entity_positions))

    bowl_on_table = "table" in positions["bowl"]
    plate_on_table = "table" in positions["plate"]
    if bowl_on_table and not plate_on_table:
        return "plate_missing"
    if plate_on_table and not bowl_on_table:
        return "bowl_missing"
    if not bowl_on_table and not plate_on_table:
        return "both_missing"
    raise GateFailure("OOD row has both bowl and plate initially on the table")


def parse_plan(response: str) -> List[Mapping[str, Any]]:
    match = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    if match is None:
        return []
    try:
        parsed = ast.literal_eval(match.group(1).strip())
    except (SyntaxError, ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [
        step
        for step in parsed
        if isinstance(step, dict)
        and isinstance(step.get("actions"), dict)
        and step["actions"]
    ]


def trace_official_score(
    scorer: Any,
    response: str,
    ground_truth: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    answer_match = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    invalid = {
        "task_success": False,
        "simulator_success": False,
        "failure_reason": "invalid_answer",
        "official_error_code": None,
        "plan_len": 0,
        "reference_len": len(ground_truth["time_steps"]),
        "length_bound_violation": False,
    }
    if answer_match is None:
        return invalid
    try:
        plan = ast.literal_eval(answer_match.group(1).strip())
    except (SyntaxError, ValueError, TypeError):
        return invalid
    transformed = scorer.transform_actions(plan)
    if not transformed:
        return invalid

    evaluator_globals = scorer.eval_single.__globals__
    official_eval = evaluator_globals["Eval"]
    original_random = evaluator_globals["random"]
    trace: Dict[str, Any] = {}

    class TracingEval(official_eval):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            trace["evaluator"] = self

    try:
        evaluator_globals["Eval"] = TracingEval
        evaluator_globals["random"] = random.Random(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            simulator_success = bool(scorer.eval_single(transformed, ground_truth))
    finally:
        evaluator_globals["Eval"] = official_eval
        evaluator_globals["random"] = original_random

    evaluator = trace.get("evaluator")
    error_code = None if evaluator is None else evaluator.error_desc_code
    reference_len = len(ground_truth["time_steps"])
    plan_len = len(transformed)
    length_bound_violation = bool(simulator_success and reference_len / plan_len < 0.99)
    task_success = simulator_success and not length_bound_violation
    if length_bound_violation:
        reason = "length_bound_violation"
    elif task_success:
        reason = "success"
    else:
        reason = SCORER_REASON_MAP.get(error_code, "unknown_scorer_failure")
    return {
        "task_success": task_success,
        "simulator_success": simulator_success,
        "failure_reason": reason,
        "official_error_code": error_code,
        "plan_len": plan_len,
        "reference_len": reference_len,
        "length_bound_violation": length_bound_violation,
    }


def _iter_actions(plan: Sequence[Mapping[str, Any]]) -> Iterable[Tuple[str, Any]]:
    for step in plan:
        actions = step.get("actions", {})
        if not isinstance(actions, dict):
            continue
        for robot, action in actions.items():
            yield robot, action


def trace_scene_evaluator(
    scorer: Any, ground_truth: Mapping[str, Any], seed: int
) -> Any:
    evaluator_globals = scorer.eval_single.__globals__
    official_eval = evaluator_globals["Eval"]
    original_random = evaluator_globals["random"]
    trace: Dict[str, Any] = {}

    class TracingEval(official_eval):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            trace["evaluator"] = self

    try:
        evaluator_globals["Eval"] = TracingEval
        evaluator_globals["random"] = random.Random(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            scorer.eval_single([{"R1": "<Move,table>"}], ground_truth)
    finally:
        evaluator_globals["Eval"] = official_eval
        evaluator_globals["random"] = original_random
    return trace["evaluator"]


def scene_vocabulary(traced_evaluator: Any, ground_truth: Mapping[str, Any]) -> set:
    vocabulary = set(traced_evaluator.env.assets)
    vocabulary.update(traced_evaluator.env.agents)
    for positions in ground_truth["init_pos"].values():
        if positions is not None:
            vocabulary.update(_native(positions))
    return vocabulary


def static_plan_diagnostics(
    plan: Sequence[Mapping[str, Any]],
    available_actions: Mapping[str, Sequence[str]],
    vocabulary: set,
) -> Dict[str, Any]:
    illegal_actions = []
    ungrounded_targets = []
    for robot, action in _iter_actions(plan):
        if not isinstance(action, (list, tuple)) or not action:
            illegal_actions.append(f"{robot}:<malformed>")
            continue
        action_name = str(action[0])
        if (
            robot not in available_actions
            or action_name not in available_actions[robot]
        ):
            illegal_actions.append(f"{robot}:{action_name}")
        entity_targets = (
            action[1:]
            if action_name
            in {
                "Reach",
                "Grasp",
                "Open",
                "Close",
                "Interact",
            }
            else []
        )
        for target in entity_targets:
            target_name = str(target)
            if target_name not in vocabulary:
                ungrounded_targets.append(target_name)
    return {
        "illegal_actions": sorted(set(illegal_actions)),
        "ungrounded_targets": sorted(set(ungrounded_targets)),
        "static_invalid": bool(illegal_actions or ungrounded_targets),
    }


def demonstration_targets(ground_truth: Mapping[str, Any]) -> Set[str]:
    targets: Set[str] = set()
    for step in ground_truth["time_steps"]:
        for action in step["actions"].values():
            if action is None or not isinstance(action, (list, tuple)):
                continue
            if action and action[0] in {
                "Move",
                "Reach",
                "Grasp",
                "Place",
                "Open",
                "Close",
                "Interact",
            }:
                targets.update(str(target) for target in action[1:])
    return targets


def demonstration_manipulated_entities(ground_truth: Mapping[str, Any]) -> Set[str]:
    targets: Set[str] = set()
    for step in ground_truth["time_steps"]:
        for action in step["actions"].values():
            if action is None or not isinstance(action, (list, tuple)) or not action:
                continue
            if action[0] in {"Reach", "Grasp", "Open", "Close", "Interact"}:
                targets.update(str(target) for target in action[1:])
    return targets


def exact_mcnemar_p(fail_to_success: int, success_to_fail: int) -> float:
    discordant = fail_to_success + success_to_fail
    tail = min(fail_to_success, success_to_fail)
    return min(
        1.0,
        2.0
        * sum(math.comb(discordant, value) for value in range(tail + 1))
        / (2**discordant),
    )


def _run_metadata_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + ".run.json")


def validate_frozen_provenance(
    baseline: Mapping[int, Mapping[str, Any]],
    memory: Mapping[int, Mapping[str, Any]],
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    baseline_metadata = json.loads(_run_metadata_path(BASELINE_LOG).read_text())
    _assert_equal("baseline seed", baseline_metadata["seed"], 0)
    _assert_equal("baseline split", baseline_metadata["split"], "val")
    _assert_equal("baseline provider", baseline_metadata["provider"], "endpoint")
    _assert_equal("baseline max_tokens", baseline_metadata["max_tokens"], 2000)
    _assert_equal("baseline temperature", baseline_metadata["temperature"], 0.0)

    source_by_index: Dict[int, Mapping[str, Any]] = {}
    source_seed_by_index: Dict[int, int] = {}
    source_configs = []
    for source_path in MEMORY_SOURCE_LOGS:
        metadata_path = _run_metadata_path(source_path)
        if not source_path.is_file() or not metadata_path.is_file():
            raise GateFailure(f"missing canonical frozen memory source {source_path}")
        metadata = json.loads(metadata_path.read_text())
        source_configs.append(metadata)
        _assert_equal(f"{source_path.name} seed", metadata["seed"], 0)
        _assert_equal(f"{source_path.name} split", metadata["split"], "val")
        _assert_equal(
            f"{source_path.name} provider", metadata["provider"], "memory-endpoint"
        )
        _assert_equal(f"{source_path.name} top_k", metadata["memory_top_k"], 5)
        _assert_equal(
            f"{source_path.name} threshold",
            metadata["memory_similarity_threshold"],
            0.3,
        )
        _assert_equal(
            f"{source_path.name} routing tokens",
            metadata["memory_prediction_max_tokens"],
            768,
        )
        _assert_equal(f"{source_path.name} max_tokens", metadata["max_tokens"], 2000)
        _assert_equal(f"{source_path.name} temperature", metadata["temperature"], 0.0)
        for index, record in _index_records(
            _load_jsonl(source_path), source_path
        ).items():
            if index in source_by_index:
                raise GateFailure(f"memory source logs overlap at index {index}")
            source_by_index[index] = record
            source_seed_by_index[index] = int(metadata["seed"]) + index

    expected_indices = set(range(EXPECTED_COUNTS["overall"]["samples"]))
    _assert_equal(
        "canonical memory source indices", set(source_by_index), expected_indices
    )
    _assert_equal("merged memory indices", set(memory), expected_indices)
    _assert_equal("baseline indices", set(baseline), expected_indices)
    for index in sorted(expected_indices):
        for field in (
            "response",
            "score",
            "format_score",
            "task_score",
            "provider_metadata",
        ):
            _assert_equal(
                f"merged memory index {index} {field}",
                memory[index][field],
                source_by_index[index][field],
            )
        baseline_seed = int(baseline_metadata["seed"]) + index
        _assert_equal(
            f"scorer seed index {index}", source_seed_by_index[index], baseline_seed
        )

    frozen_config = {
        "baseline_run_metadata": baseline_metadata,
        "memory_source_run_metadata": source_configs,
    }
    return source_seed_by_index, frozen_config


def _build_retrieval(
    record: Mapping[str, Any],
    train_frame: pd.DataFrame,
    memory_module: Any,
) -> Any:
    metadata = record["provider_metadata"]
    instances = []
    for item in metadata["instances"]:
        train_sample = _native(train_frame.iloc[int(item["train_index"])].to_dict())
        instruction, robots, _ = memory_module.get_prompt_context(train_sample)
        ground_truth = train_sample["reward_model"]["ground_truth"]
        demonstration = memory_module._format_plan(ground_truth)
        required_actions = memory_module._required_actions(ground_truth)
        _assert_equal(
            f"train {item['train_index']} context", instruction, item["context"]
        )
        _assert_equal(
            f"train {item['train_index']} skill",
            ground_truth["task_name"],
            item["skill_name"],
        )
        _assert_equal(
            f"train {item['train_index']} required_actions",
            sorted(required_actions),
            sorted(item["required_actions"]),
        )
        _assert_equal(
            f"train {item['train_index']} robot_count",
            len(robots),
            item["robot_count"],
        )
        instance = memory_module.SkillInstance(
            train_index=int(item["train_index"]),
            skill_name=item["skill_name"],
            context=instruction,
            demonstration=demonstration,
            required_actions=required_actions,
            robot_count=len(robots),
        )
        instances.append(
            memory_module.RetrievedInstance(instance, float(item["similarity"]))
        )
    return memory_module.MemoryRetrieval(
        metadata["predicted_skill"],
        float(metadata["skill_similarity"]),
        instances,
    )


def _prompt_token_count(
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
    image: Image.Image,
) -> int:
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = processor(text=[rendered], images=[image], return_tensors=None)
    return len(encoded["input_ids"][0])


def _distribution(values: Sequence[float]) -> Dict[str, float]:
    series = pd.Series(values, dtype="float64")
    return {
        "median": float(series.median()),
        "p10": float(series.quantile(0.10)),
        "min": float(series.min()),
    }


def _aggregate(frame: pd.DataFrame) -> Dict[str, Any]:
    baseline_successes = int(frame["baseline_success"].sum())
    memory_successes = int(frame["memory_success"].sum())
    return {
        "samples": len(frame),
        "baseline_successes": baseline_successes,
        "memory_successes": memory_successes,
        "baseline_accuracy": baseline_successes / len(frame),
        "memory_accuracy": memory_successes / len(frame),
        "fail_to_success": int(
            ((~frame["baseline_success"]) & frame["memory_success"]).sum()
        ),
        "success_to_fail": int(
            (frame["baseline_success"] & (~frame["memory_success"])).sum()
        ),
    }


def gate_v0(frame: pd.DataFrame) -> Dict[str, Any]:
    overall = _aggregate(frame)
    by_subset = {
        subset: _aggregate(frame[frame["ood_subset"] == subset])
        for subset in ("bowl_missing", "plate_missing", "both_missing")
    }
    for field in ("samples", "baseline_successes", "memory_successes"):
        _assert_equal(
            f"overall {field}", overall[field], EXPECTED_COUNTS["overall"][field]
        )
    for subset, expected in EXPECTED_COUNTS.items():
        if subset == "overall":
            continue
        for field in ("samples", "baseline_successes", "memory_successes"):
            _assert_equal(
                f"{subset} {field}", by_subset[subset][field], expected[field]
            )
    _assert_equal(
        "overall fail_to_success", overall["fail_to_success"], EXPECTED_FAIL_TO_SUCCESS
    )
    _assert_equal(
        "overall success_to_fail", overall["success_to_fail"], EXPECTED_SUCCESS_TO_FAIL
    )
    p_value = exact_mcnemar_p(overall["fail_to_success"], overall["success_to_fail"])
    _assert_close("exact McNemar p", p_value, EXPECTED_MCNEMAR_P)
    _assert_equal(
        "format compliance baseline source",
        bool((frame["baseline_success"].notna()).all()),
        True,
    )
    return {
        "status": "PASS",
        "overall": overall,
        "by_subset": by_subset,
        "mcnemar_exact_p": p_value,
    }


def build_v0(args: argparse.Namespace) -> Dict[str, Any]:
    from transformers import AutoProcessor

    bench_module = _load_source(
        "_viki_bench_frozen", ROOT / "habitat_llm/evaluation/viki_bench.py"
    )
    memory_module = _load_source(
        "_viki_memory_frozen",
        ROOT / "habitat_llm/evaluation/viki_memory_skill.py",
    )
    baseline = _index_records(_load_jsonl(BASELINE_LOG), BASELINE_LOG)
    memory = _index_records(_load_jsonl(MEMORY_LOG), MEMORY_LOG)
    scorer_seeds, frozen_config = validate_frozen_provenance(baseline, memory)

    val_path = args.benchmark_root / "data/VIKI-R/viki/VIKI-L2/val.parquet"
    train_path = args.benchmark_root / "data/VIKI-R/viki/VIKI-L2/train.parquet"
    val_frame = pd.read_parquet(val_path)
    train_frame = pd.read_parquet(train_path, columns=["prompt", "reward_model"])
    _assert_equal("validation row count", len(val_frame), 1218)
    _assert_equal("training row count", len(train_frame), 7196)
    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, use_fast=False
    )

    rows = []
    for index in range(len(val_frame)):
        sample = _native(val_frame.iloc[index].to_dict())
        ground_truth = sample["reward_model"]["ground_truth"]
        memory_record = memory[index]
        baseline_record = baseline[index]
        retrieval = _build_retrieval(memory_record, train_frame, memory_module)
        base_messages = bench_module.get_messages(sample)
        memory_messages = memory_module.add_memory_to_messages(
            bench_module.get_messages(sample),
            memory_module.format_memory_prompt(retrieval),
        )
        with Image.open(io.BytesIO(sample["images"][0]["bytes"])) as source_image:
            image = source_image.convert("RGB")
            baseline_input_tokens = _prompt_token_count(processor, base_messages, image)
            memory_input_tokens = _prompt_token_count(processor, memory_messages, image)
        similarities = [float(item.similarity) for item in retrieval.instances]
        rows.append(
            {
                "index": index,
                "ood_subset": derive_ood_subset(ground_truth),
                "baseline_success": baseline_record["task_score"] == 1,
                "memory_success": memory_record["task_score"] == 1,
                "routed_skill": retrieval.skill_name,
                "injected_demo_ids": [
                    int(item.instance.train_index) for item in retrieval.instances
                ],
                "n_demos_injected": len(retrieval.instances),
                "retrieval_sim_max": max(similarities) if similarities else math.nan,
                "retrieval_sim_mean": (
                    sum(similarities) / len(similarities) if similarities else math.nan
                ),
                "fallback_fired": not retrieval.instances,
                "input_tokens_total": memory_input_tokens,
                "injected_tokens": memory_input_tokens - baseline_input_tokens,
                "truncation_flag": (
                    memory_input_tokens + PLAN_TOKEN_BUDGET > MODEL_CONTEXT
                ),
                "plan_len_baseline": len(parse_plan(baseline_record["response"])),
                "plan_len_memory": len(parse_plan(memory_record["response"])),
                "scorer_seed": scorer_seeds[index],
            }
        )

    frame = pd.DataFrame(rows)
    gate = gate_v0(frame)
    similarity = {
        "overall": _distribution(frame["retrieval_sim_max"].tolist()),
        "by_subset": {
            subset: _distribution(
                frame.loc[frame["ood_subset"] == subset, "retrieval_sim_max"].tolist()
            )
            for subset in ("bowl_missing", "plate_missing", "both_missing")
        },
    }
    summary = {
        "task": "V0",
        "gate": gate,
        "fallback": {
            "prediction": 0.0,
            "count": int(frame["fallback_fired"].sum()),
            "rate": float(frame["fallback_fired"].mean()),
        },
        "retrieval_similarity_max": similarity,
        "context_headroom": {
            "model_context": MODEL_CONTEXT,
            "generation_reservation": PLAN_TOKEN_BUDGET,
            "input_budget": MODEL_CONTEXT - PLAN_TOKEN_BUDGET,
            "max_input_tokens_total": int(frame["input_tokens_total"].max()),
            "min_headroom": int(
                MODEL_CONTEXT - PLAN_TOKEN_BUDGET - frame["input_tokens_total"].max()
            ),
            "truncation_count": int(frame["truncation_flag"].sum()),
        },
        "scorer_seed": {
            "rule": "run_seed + index",
            "baseline_run_seed": 0,
            "memory_run_seed": 0,
            "identical_per_index": True,
        },
        "token_counting": {
            "processor": str(args.model_path.resolve()),
            "transformers_version": __import__("transformers").__version__,
            "vllm_equivalence": (
                "vLLM 0.8.4 Qwen2_5_VLMultiModalProcessor calls the same "
                "Qwen2_5_VLProcessor and expands image tokens from image_grid_thw"
            ),
        },
        "frozen_provenance": frozen_config,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "viki_ood_samples.parquet"
    csv_path = args.output_dir / "viki_ood_samples.csv"
    summary_path = args.output_dir / "viki_ood_samples.summary.json"
    frame.to_parquet(parquet_path, index=False)
    csv_frame = frame.copy()
    csv_frame["injected_demo_ids"] = csv_frame["injected_demo_ids"].map(
        lambda values: json.dumps(values, separators=(",", ":"))
    )
    csv_frame.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "parquet": str(parquet_path),
        "csv": str(csv_path),
        "summary": str(summary_path),
        **summary,
    }


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preregistered VIKI Memory-as-Skill experiment analyses"
    )
    subparsers = parser.add_subparsers(dest="task", required=True)
    v0 = subparsers.add_parser("v0", help="Build and gate the OOD sample table")
    v0.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    v0.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    v0.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        if args.task == "v0":
            result = build_v0(args)
        else:
            raise ValueError(f"Unsupported task {args.task}")
    except GateFailure as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
