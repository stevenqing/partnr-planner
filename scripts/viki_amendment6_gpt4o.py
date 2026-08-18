#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import tiktoken
from viki_amendment5 import (
    BENCHMARK_ROOT,
    EMBEDDING_MODEL,
    H0_PATH,
    MAX_OUTPUT_TOKENS,
    SIMILARITY_THRESHOLD,
    atomic_json,
    file_sha256,
    fingerprint,
    load_bench,
    load_jsonl,
    messages_sha256,
    native,
    write_jsonl_snapshot,
)
from viki_amendment6 import (
    AMENDMENT5_1_DIR,
    AMENDMENT5_DIR,
    ARM_ORDER,
    ID_ROWS,
    OOD_ROWS,
    OUTPUT_DIR,
    ROOT,
    TOKEN_TOLERANCE,
    TOM_REASONING_TEMPLATE,
    GateFailure,
    add_tom_to_messages,
    format_trajectory_prompt,
    load_row_embeddings,
    load_source_rows,
    preregistration,
    require_preregistration,
    split_configuration,
    summarize_subset,
)

from habitat_llm.evaluation.viki_branch_conditions import get_instruction
from habitat_llm.evaluation.viki_memory_skill import (
    add_memory_to_messages,
    get_prompt_context,
)

BACKBONE = "gpt_4o_optional"
MODEL_ID = "gpt-4o-2024-08-06"
REQUEST_MODEL = "gpt-4o-20240806"
ENCODING = "o200k_base"


def source_configuration(
    split: str,
) -> Tuple[pd.DataFrame, List[int], Path, Path]:
    dataset, indices, _, _, _ = split_configuration(split)
    directory = AMENDMENT5_DIR if split == "ood" else AMENDMENT5_1_DIR
    return (
        dataset,
        indices,
        directory / f"{BACKBONE}.jsonl",
        directory / f"{BACKBONE}.summary.json",
    )


def validate_source(
    split: str, indices: Sequence[int], source_path: Path, summary_path: Path
) -> Tuple[Dict[int, Dict[str, Any]], str, Dict[str, Any]]:
    run_path = source_path.with_suffix(source_path.suffix + ".run.json")
    if any(not path.is_file() for path in (source_path, summary_path, run_path)):
        raise GateFailure(f"Missing GPT-4o source artifact for {split}")
    records = load_jsonl(source_path)
    metadata = json.loads(run_path.read_text())
    summary = json.loads(summary_path.read_text())
    run_hash = fingerprint(metadata)
    if (
        set(records) != set(indices)
        or any(record.get("run_fingerprint") != run_hash for record in records.values())
        or any(
            set(record.get("arms", {})) != {"zero_shot", "segment"}
            for record in records.values()
        )
        or any(record.get("endpoint_error") for record in records.values())
        or summary.get("samples") != len(indices)
    ):
        raise GateFailure(f"Invalid GPT-4o source artifact for {split}")
    expected = {
        arm: int(summary["arms"][arm]["successes"]) for arm in ("zero_shot", "segment")
    }
    observed = {
        arm: sum(record["arms"][arm]["task_score"] == 1 for record in records.values())
        for arm in ("zero_shot", "segment")
    }
    if observed != expected:
        raise GateFailure(f"GPT-4o source outcomes changed for {split}")
    return records, run_hash, summary


def required_72b_complete() -> None:
    missing = [
        OUTPUT_DIR / f"qwen2_5_vl_72b.{split}.{arm}.summary.json"
        for split, arm in (
            ("ood", "trajectory_rag"),
            ("ood", "tom"),
            ("id", "trajectory_rag"),
            ("id", "tom"),
        )
        if not (OUTPUT_DIR / f"qwen2_5_vl_72b.{split}.{arm}.summary.json").is_file()
    ]
    if missing:
        raise GateFailure(
            "Required 72B arms are incomplete: " + ", ".join(map(str, missing))
        )


def message_text(messages: Sequence[Mapping[str, Any]], role: str) -> str:
    message = next(item for item in messages if item["role"] == role)
    content = message["content"]
    if isinstance(content, str):
        return content
    text_item = next(item for item in content if item["type"] == "text")
    return str(text_item["text"])


def memory_delta(encoding: Any, original_text: str, memory_prompt: str) -> int:
    modified = f"{memory_prompt}\n\nCurrent task:\n{original_text}"
    return len(encoding.encode(modified)) - len(encoding.encode(original_text))


def tom_delta(encoding: Any, original_system: str) -> int:
    modified = f"{original_system}\n\n{TOM_REASONING_TEMPLATE}"
    return len(encoding.encode(modified)) - len(encoding.encode(original_system))


def trim_prompt(
    encoding: Any, prompt: str, original_text: str, target_delta: int
) -> Tuple[str, int]:
    encoded = encoding.encode(prompt)
    observed: Dict[int, Tuple[str, int]] = {}

    def evaluate(count: int) -> Tuple[str, int]:
        count = min(len(encoded), max(1, count))
        if count not in observed:
            candidate = encoding.decode(encoded[:count])
            observed[count] = (
                candidate,
                memory_delta(encoding, original_text, candidate),
            )
        return observed[count]

    lower, upper = 1, len(encoded)
    while lower <= upper:
        middle = (lower + upper) // 2
        if evaluate(middle)[1] < target_delta:
            lower = middle + 1
        else:
            upper = middle - 1
    candidates = {
        count
        for center in (lower, upper)
        for count in range(max(1, center - 16), min(len(encoded), center + 16) + 1)
    }
    for count in sorted(candidates):
        evaluate(count)
    return min(observed.values(), key=lambda item: abs(item[1] - target_delta))


def prior_optional_summaries(split: str, arm: str) -> List[Path]:
    order = [
        ("ood", "trajectory_rag"),
        ("ood", "tom"),
        ("id", "trajectory_rag"),
        ("id", "tom"),
    ]
    current = order.index((split, arm))
    return [
        OUTPUT_DIR / f"{BACKBONE}.{prior_split}.{prior_arm}.summary.json"
        for prior_split, prior_arm in order[:current]
    ]


def preflight(split: str) -> Dict[str, Any]:
    require_preregistration()
    required_72b_complete()
    dataset, indices, source_path, source_summary_path = source_configuration(split)
    source_records, source_run_hash, source_summary = validate_source(
        split, indices, source_path, source_summary_path
    )
    source_ids, source_rows, source_instructions = load_source_rows()
    row_embeddings = load_row_embeddings(source_ids, source_instructions)
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    query_embeddings = np.asarray(
        embedding_model.encode(
            [
                get_instruction(native(dataset.iloc[index].to_dict()))
                for index in indices
            ],
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )
    encoding = tiktoken.get_encoding(ENCODING)
    bench = load_bench()
    manifest_path = (
        AMENDMENT5_DIR / "deployment_manifest.jsonl"
        if split == "ood"
        else AMENDMENT5_1_DIR / "id_deployment_manifest.jsonl"
    )
    segment_manifest = load_jsonl(manifest_path)
    records: Dict[int, Dict[str, Any]] = {}
    calibration_errors: List[int] = []
    for position, index in enumerate(indices):
        sample = native(dataset.iloc[index].to_dict())
        zero_messages = bench.get_messages(sample)
        original_text = message_text(zero_messages, "user")
        original_system = message_text(zero_messages, "system")
        zero_tokens = int(source_records[index]["arms"]["zero_shot"]["prompt_tokens"])
        segment_tokens = int(source_records[index]["arms"]["segment"]["prompt_tokens"])
        segment_prompt = str(segment_manifest[index]["segment_prompt"])
        calibrated_segment_delta = memory_delta(encoding, original_text, segment_prompt)
        calibration_errors.append(
            calibrated_segment_delta - (segment_tokens - zero_tokens)
        )
        if calibration_errors[-1] != 0:
            raise GateFailure(f"GPT-4o token calibration changed at {split}:{index}")
        similarities = row_embeddings @ query_embeddings[position]
        ranking = sorted(
            range(len(source_ids)),
            key=lambda item: (-float(similarities[item]), source_ids[item]),
        )
        selected_sources: List[int] = []
        selected_rows = []
        trajectory_prompt = ""
        trajectory_delta = 0
        if float(similarities[ranking[0]]) >= SIMILARITY_THRESHOLD:
            for ranked_position in ranking:
                source = source_ids[ranked_position]
                selected_sources.append(source)
                selected_rows.append(source_rows[source])
                trajectory_prompt = format_trajectory_prompt(selected_rows)
                trajectory_delta = memory_delta(
                    encoding, original_text, trajectory_prompt
                )
                if trajectory_delta >= segment_tokens - zero_tokens:
                    break
            if trajectory_delta > segment_tokens - zero_tokens:
                trajectory_prompt, trajectory_delta = trim_prompt(
                    encoding,
                    trajectory_prompt,
                    original_text,
                    segment_tokens - zero_tokens,
                )
        trajectory_tokens = zero_tokens + trajectory_delta
        relative_difference = abs(trajectory_tokens - segment_tokens) / segment_tokens
        if relative_difference > TOKEN_TOLERANCE:
            raise GateFailure(f"GPT-4o trajectory token gate failed at {split}:{index}")
        trajectory_messages = add_memory_to_messages(
            bench.get_messages(sample), trajectory_prompt
        )
        tom_messages = add_tom_to_messages(bench.get_messages(sample))
        tom_tokens = zero_tokens + tom_delta(encoding, original_system)
        _, robots, _ = get_prompt_context(sample)
        ground_truth = bench.get_ground_truth(sample)
        records[index] = {
            "index": index,
            "split": split,
            "family": str(ground_truth["task_name"]),
            "active_robot_count": len(robots),
            "trajectory_rag": {
                "prompt": trajectory_prompt,
                "best_similarity": float(similarities[ranking[0]]),
                "fallback_bare": not selected_sources,
                "selected_source_indices_before_token_trim": selected_sources,
                "input_tokens": trajectory_tokens,
                "injected_tokens": trajectory_delta,
                "relative_difference_from_segment": relative_difference,
            },
            "tom": {
                "input_tokens": tom_tokens,
                "injected_tokens": tom_tokens - zero_tokens,
            },
            "token_counts": {"zero_shot": zero_tokens, "segment": segment_tokens},
            "prompt_sha256": {
                "zero_shot": messages_sha256(zero_messages),
                "trajectory_rag": messages_sha256(trajectory_messages),
                "tom": messages_sha256(tom_messages),
                "segment": source_records[index]["prompt_sha256"]["segment"],
            },
            "reused_response_sha256": {
                arm: hashlib.sha256(
                    source_records[index]["arms"][arm]["response"].encode()
                ).hexdigest()
                for arm in ("zero_shot", "segment")
            },
        }
    output = OUTPUT_DIR / f"{BACKBONE}.{split}.preflight.jsonl"
    write_jsonl_snapshot(output, records, indices)
    summary = {
        "task": "Amendment6_GPT4o_preflight",
        "status": "PASS",
        "split": split,
        "rows": len(records),
        "model_generation_calls": 0,
        "model_id": MODEL_ID,
        "request_model": REQUEST_MODEL,
        "tokenizer": ENCODING,
        "calibration": {
            "source": "certified GPT-4o segment-minus-zero provider usage",
            "rows": len(calibration_errors),
            "maximum_absolute_error": max(abs(value) for value in calibration_errors),
        },
        "fallback_rows": sum(
            row["trajectory_rag"]["fallback_bare"] for row in records.values()
        ),
        "maximum_trajectory_token_difference": max(
            row["trajectory_rag"]["relative_difference_from_segment"]
            for row in records.values()
        ),
        "source_run_fingerprint": source_run_hash,
        "source_results_sha256": file_sha256(source_path),
        "source_summary_sha256": file_sha256(source_summary_path),
        "source_status": source_summary.get("status", "PASS"),
        "results": str(output),
        "results_sha256": file_sha256(output),
        "h0_sha256": file_sha256(H0_PATH),
    }
    atomic_json(output.with_suffix(".summary.json"), summary)
    return summary


def runtime_identity() -> Dict[str, Any]:
    return {
        "base_url": "https://cloudgpt-openai.azure-api.net/",
        "provider": "CloudGPT AzureOpenAI",
        "authentication": "Azure CLI cached token",
        "api_version": "2024-06-01",
        "models": [{"id": REQUEST_MODEL, "root": MODEL_ID, "max_model_len": 128000}],
    }


def run_arm(split: str, arm: str, workers: int) -> Dict[str, Any]:
    require_preregistration()
    required_72b_complete()
    if os.environ.get("CLOUDGPT_USE_AZURE_CLI") != "1":
        raise GateFailure("Set CLOUDGPT_USE_AZURE_CLI=1 for CloudGPT")
    missing = [
        str(path) for path in prior_optional_summaries(split, arm) if not path.is_file()
    ]
    if missing:
        raise GateFailure("Optional GPT-4o order is incomplete: " + ", ".join(missing))
    dataset, indices, source_path, source_summary_path = source_configuration(split)
    source_records, source_run_hash, _ = validate_source(
        split, indices, source_path, source_summary_path
    )
    preflight_path = OUTPUT_DIR / f"{BACKBONE}.{split}.preflight.jsonl"
    preflight_summary_path = preflight_path.with_suffix(".summary.json")
    if any(not path.is_file() for path in (preflight_path, preflight_summary_path)):
        raise GateFailure(f"Run GPT-4o {split} preflight first")
    preflight_summary = json.loads(preflight_summary_path.read_text())
    if preflight_summary.get("results_sha256") != file_sha256(preflight_path):
        raise GateFailure(f"Stale GPT-4o {split} preflight")
    preflight_records = load_jsonl(preflight_path)
    runtime = runtime_identity()
    metadata = {
        "task": "Amendment6_GPT4o_deployment_baseline",
        "backbone": BACKBONE,
        "model_id": MODEL_ID,
        "request_model": REQUEST_MODEL,
        "split": split,
        "arm": arm,
        "indices": indices,
        "runtime": runtime,
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "workers": workers,
        "scorer_seed_rule": "original row index",
        "preflight_sha256": file_sha256(preflight_path),
        "source_run_fingerprint": source_run_hash,
        "source_results_sha256": file_sha256(source_path),
        "h0_sha256": file_sha256(H0_PATH),
    }
    run_hash = fingerprint(metadata)
    output = OUTPUT_DIR / f"{BACKBONE}.{split}.{arm}.jsonl"
    run_path = output.with_suffix(output.suffix + ".run.json")
    if run_path.is_file() and json.loads(run_path.read_text()) != metadata:
        raise GateFailure(f"Cannot resume GPT-4o {split} {arm}: metadata differs")
    atomic_json(run_path, metadata)
    records = load_jsonl(output)
    if any(
        index not in indices or row.get("run_fingerprint") != run_hash
        for index, row in records.items()
    ):
        raise GateFailure(f"Invalid GPT-4o checkpoint for {split} {arm}")
    pending = [
        index
        for index in indices
        if index not in records or records[index].get("endpoint_error")
    ]
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    from cloudgpt_aoai import get_openai_client

    client = get_openai_client(timeout=3600, max_retries=5)

    def generate(index: int) -> Tuple[int, Dict[str, Any]]:
        sample = native(dataset.iloc[index].to_dict())
        row = preflight_records[index]
        messages = (
            add_memory_to_messages(
                bench.get_messages(sample), row["trajectory_rag"]["prompt"]
            )
            if arm == "trajectory_rag"
            else add_tom_to_messages(bench.get_messages(sample))
        )
        if messages_sha256(messages) != row["prompt_sha256"][arm]:
            raise GateFailure(f"GPT-4o prompt changed at {split}:{arm}:{index}")
        completion = client.chat.completions.create(
            model=REQUEST_MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0,
        )
        usage = completion.usage
        prompt_tokens = None if usage is None else usage.prompt_tokens
        expected_tokens = int(row[arm]["input_tokens"])
        if prompt_tokens != expected_tokens:
            raise GateFailure(
                f"GPT-4o prompt token count changed at {split}:{arm}:{index}: "
                f"expected={expected_tokens}, observed={prompt_tokens}"
            )
        response = completion.choices[0].message.content or ""
        metrics = bench.score_response(
            scorer, 2, response, bench.get_ground_truth(sample), index
        )
        return index, {
            "index": index,
            "split": split,
            "arm": arm,
            "run_fingerprint": run_hash,
            "prompt_sha256": row["prompt_sha256"][arm],
            "source_run_fingerprint": source_run_hash,
            "reused_response_sha256": row["reused_response_sha256"],
            "active_robot_count": row["active_robot_count"],
            "family": row["family"],
            "response": response,
            "prompt_tokens": prompt_tokens,
            "injected_tokens": int(row[arm]["injected_tokens"]),
            "completion_tokens": None if usage is None else usage.completion_tokens,
            **metrics,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
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
            write_jsonl_snapshot(output, records, indices)
    errors = [row for row in records.values() if row.get("endpoint_error")]
    if set(records) != set(indices) or errors:
        raise GateFailure(f"GPT-4o {split} {arm} incomplete with {len(errors)} errors")
    successes = sum(row["task_score"] == 1 for row in records.values())
    formats = sum(row["format_score"] == 1 for row in records.values())
    compliance = formats / len(records)
    summary = {
        "task": "Amendment6_GPT4o_deployment_baseline",
        "status": "PASS" if compliance >= 0.9 else "FAIL",
        "backbone": BACKBONE,
        "split": split,
        "arm": arm,
        "samples": len(records),
        "successes": successes,
        "accuracy": successes / len(records),
        "format_successes": formats,
        "format_compliance": compliance,
        "format_gate": {
            "status": "PASS" if compliance >= 0.9 else "FAIL",
            "minimum": 0.9,
            "exempt": False,
        },
        "model_generation_calls": len(records),
        "run_fingerprint": run_hash,
        "source_run_fingerprint": source_run_hash,
        "runtime": runtime,
        "results": str(output),
        "results_sha256": file_sha256(output),
    }
    atomic_json(output.with_suffix(".summary.json"), summary)
    return summary


def validate_generated(
    split: str, arm: str, indices: Sequence[int]
) -> Dict[int, Dict[str, Any]]:
    output = OUTPUT_DIR / f"{BACKBONE}.{split}.{arm}.jsonl"
    run_path = output.with_suffix(output.suffix + ".run.json")
    summary_path = output.with_suffix(".summary.json")
    if any(not path.is_file() for path in (output, run_path, summary_path)):
        raise GateFailure(f"Missing GPT-4o {split} {arm} artifact")
    records = load_jsonl(output)
    run_hash = fingerprint(json.loads(run_path.read_text()))
    summary = json.loads(summary_path.read_text())
    if (
        set(records) != set(indices)
        or any(
            row.get("run_fingerprint") != run_hash or row.get("endpoint_error")
            for row in records.values()
        )
        or summary.get("run_fingerprint") != run_hash
        or summary.get("results_sha256") != file_sha256(output)
    ):
        raise GateFailure(f"Invalid GPT-4o {split} {arm} artifact")
    return records


def analyze(split: str) -> Dict[str, Any]:
    require_preregistration()
    dataset, indices, source_path, source_summary_path = source_configuration(split)
    del dataset
    source_records, source_run_hash, _ = validate_source(
        split, indices, source_path, source_summary_path
    )
    generated = {
        arm: validate_generated(split, arm, indices)
        for arm in ("trajectory_rag", "tom")
    }
    preflight_path = OUTPUT_DIR / f"{BACKBONE}.{split}.preflight.jsonl"
    preflight_records = load_jsonl(preflight_path)
    arms: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    token_rows = []
    for index in indices:
        for arm in ("zero_shot", "segment"):
            observed = hashlib.sha256(
                source_records[index]["arms"][arm]["response"].encode()
            ).hexdigest()
            if observed != preflight_records[index]["reused_response_sha256"][arm]:
                raise GateFailure(
                    f"GPT-4o reused response changed at {split}:{index}:{arm}"
                )
        arms[index] = {
            "zero_shot": source_records[index]["arms"]["zero_shot"],
            "trajectory_rag": generated["trajectory_rag"][index],
            "tom": generated["tom"][index],
            "segment": source_records[index]["arms"]["segment"],
        }
        zero_tokens = int(arms[index]["zero_shot"]["prompt_tokens"])
        row = {
            "index": index,
            "family": preflight_records[index]["family"],
            "active_robot_count": preflight_records[index]["active_robot_count"],
        }
        for arm in ARM_ORDER:
            input_tokens = int(arms[index][arm]["prompt_tokens"])
            row.update(
                {
                    f"{arm}_task_score": arms[index][arm]["task_score"],
                    f"{arm}_format_score": arms[index][arm]["format_score"],
                    f"{arm}_input_tokens": input_tokens,
                    f"{arm}_injected_tokens": 0
                    if arm == "zero_shot"
                    else input_tokens - zero_tokens,
                    f"{arm}_generated_tokens": arms[index][arm].get(
                        "completion_tokens"
                    ),
                }
            )
        token_rows.append(row)
    primary = summarize_subset(arms, 200 if split == "ood" else 300)
    breakdown_field = "family" if split == "ood" else "active_robot_count"
    breakdown: Dict[str, Dict[str, Any]] = {}
    for value in sorted({row[breakdown_field] for row in token_rows}, key=str):
        subset = {
            index: arms[index]
            for index in indices
            if preflight_records[index][breakdown_field] == value
        }
        breakdown[str(value)] = summarize_subset(subset, 2000 + len(breakdown) * 10)
    frame = pd.DataFrame(token_rows).sort_values("index")
    row_path = OUTPUT_DIR / f"{BACKBONE}.{split}.four_arm_rows.parquet"
    frame.to_parquet(row_path, index=False)
    arm_table = pd.DataFrame(
        [
            {
                "arm": arm,
                **primary["arms"][arm],
                "mean_input_tokens": float(frame[f"{arm}_input_tokens"].mean()),
                "mean_injected_tokens": float(frame[f"{arm}_injected_tokens"].mean()),
                "mean_generated_tokens": float(frame[f"{arm}_generated_tokens"].mean()),
            }
            for arm in ARM_ORDER
        ]
    )
    arm_path = OUTPUT_DIR / f"{BACKBONE}.{split}.four_arm_table.csv"
    arm_table.to_csv(arm_path, index=False)
    pairwise_path = OUTPUT_DIR / f"{BACKBONE}.{split}.pairwise_table.csv"
    pd.DataFrame(primary["comparisons"].values()).to_csv(pairwise_path, index=False)
    summary = {
        "task": "Amendment6_GPT4o_four_arm_analysis",
        "status": "PASS"
        if all(value["format_compliance"] >= 0.9 for value in primary["arms"].values())
        else "COMPLETE_WITH_FORMAT_GATE_FAILURES",
        "backbone": BACKBONE,
        "split": split,
        **primary,
        "breakdown_field": breakdown_field,
        "breakdown": breakdown,
        "model_generation_calls": 2 * len(indices),
        "reused_arms": ["zero_shot", "segment"],
        "source_run_fingerprint": source_run_hash,
        "source_results_sha256": file_sha256(source_path),
        "row_results": str(row_path),
        "row_results_sha256": file_sha256(row_path),
        "arm_table": str(arm_path),
        "arm_table_sha256": file_sha256(arm_path),
        "pairwise_table": str(pairwise_path),
        "pairwise_table_sha256": file_sha256(pairwise_path),
        "preflight_sha256": file_sha256(preflight_path),
        "h0_sha256": file_sha256(H0_PATH),
    }
    atomic_json(OUTPUT_DIR / f"{BACKBONE}.{split}.analysis.json", summary)
    return summary


def close() -> Dict[str, Any]:
    require_preregistration()
    required_72b_complete()
    generated_artifacts = []
    accepted_calls = 0
    format_failures = []
    for backbone in ("qwen2_5_vl_72b", BACKBONE):
        for split, expected_rows in (("ood", OOD_ROWS), ("id", ID_ROWS)):
            analysis_path = OUTPUT_DIR / f"{backbone}.{split}.analysis.json"
            if not analysis_path.is_file():
                raise GateFailure(f"Missing final analysis: {analysis_path}")
            analysis = json.loads(analysis_path.read_text())
            if analysis.get("samples") != expected_rows:
                raise GateFailure(f"Invalid final analysis coverage: {analysis_path}")
            if analysis.get("status") != "PASS":
                format_failures.append(f"{backbone}.{split}")
            for arm in ("trajectory_rag", "tom"):
                result_path = OUTPUT_DIR / f"{backbone}.{split}.{arm}.jsonl"
                run_path = result_path.with_suffix(result_path.suffix + ".run.json")
                summary_path = result_path.with_suffix(".summary.json")
                if any(
                    not path.is_file() for path in (result_path, run_path, summary_path)
                ):
                    raise GateFailure(f"Missing generated artifact: {result_path}")
                records = load_jsonl(result_path)
                metadata = json.loads(run_path.read_text())
                summary = json.loads(summary_path.read_text())
                run_hash = fingerprint(metadata)
                successes = sum(
                    record["task_score"] == 1 for record in records.values()
                )
                formats = sum(
                    record["format_score"] == 1 for record in records.values()
                )
                if (
                    len(records) != expected_rows
                    or any(record.get("endpoint_error") for record in records.values())
                    or {record.get("run_fingerprint") for record in records.values()}
                    != {run_hash}
                    or summary.get("run_fingerprint") != run_hash
                    or summary.get("results_sha256") != file_sha256(result_path)
                    or summary.get("successes") != successes
                    or summary.get("format_successes") != formats
                ):
                    raise GateFailure(f"Final certification failed: {result_path}")
                accepted_calls += len(records)
                generated_artifacts.append(
                    {
                        "backbone": backbone,
                        "split": split,
                        "arm": arm,
                        "rows": len(records),
                        "successes": successes,
                        "format_successes": formats,
                        "run_fingerprint": run_hash,
                        "results": str(result_path),
                        "results_sha256": file_sha256(result_path),
                        "summary_sha256": file_sha256(summary_path),
                    }
                )
    failed_path = (
        OUTPUT_DIR / "qwen2_5_vl_72b.ood.trajectory_rag.failed_token_accounting.json"
    )
    failed = json.loads(failed_path.read_text())
    failed_run_path = Path(failed["failed_run_metadata"])
    if (
        failed.get("status") != "FAIL_CLOSED"
        or failed.get("accepted_rows") != 0
        or failed.get("discarded_generation_calls") != 8
        or failed.get("primary_inference_eligible") is not False
        or failed.get("failed_run_metadata_sha256") != file_sha256(failed_run_path)
    ):
        raise GateFailure("Failed-call audit certification changed")
    qwen_ood_path = OUTPUT_DIR / "qwen2_5_vl_72b.ood.analysis.json"
    qwen_ood = json.loads(qwen_ood_path.read_text())
    ending = qwen_ood.get("ending")
    if (
        not isinstance(ending, dict)
        or ending.get("selected") not in preregistration()["endings"]
    ):
        raise GateFailure("Preregistered Amendment 6 ending is missing")
    result = {
        "task": "Amendment6_generation_closure",
        "status": ("COMPLETE_WITH_FORMAT_GATE_FAILURES" if format_failures else "PASS"),
        "generation_closed": True,
        "permanent_closure_reinstated": True,
        "completed_results_modified": False,
        "per_backbone_tuning": False,
        "required_qwen2_5_vl_72b_complete": True,
        "optional_gpt_4o_complete": True,
        "gemini_excluded": True,
        "accepted_generation_calls": accepted_calls,
        "accepted_calls_by_backbone": {
            "qwen2_5_vl_72b": 3036,
            BACKBONE: 3036,
        },
        "discarded_fail_closed_calls": failed["discarded_generation_calls"],
        "failed_call_audit": str(failed_path),
        "failed_call_audit_sha256": file_sha256(failed_path),
        "generated_artifacts_certified": len(generated_artifacts),
        "generated_artifacts": generated_artifacts,
        "analysis_artifacts": {
            f"{backbone}.{split}": {
                "path": str(OUTPUT_DIR / f"{backbone}.{split}.analysis.json"),
                "sha256": file_sha256(OUTPUT_DIR / f"{backbone}.{split}.analysis.json"),
            }
            for backbone in ("qwen2_5_vl_72b", BACKBONE)
            for split in ("ood", "id")
        },
        "format_gate_failure_analyses": format_failures,
        "selected_ending": ending,
        "preregistration": str(OUTPUT_DIR / "preregistration.json"),
        "preregistration_sha256": file_sha256(OUTPUT_DIR / "preregistration.json"),
        "supersession_sha256": file_sha256(OUTPUT_DIR / "closure_supersession.json"),
        "qwen_ood_analysis_sha256": file_sha256(qwen_ood_path),
        "runner_sha256": file_sha256(ROOT / "scripts/viki_amendment6.py"),
        "gpt4o_runner_sha256": file_sha256(ROOT / "scripts/viki_amendment6_gpt4o.py"),
        "after_scope": "No further VIKI generation is authorized.",
    }
    atomic_json(OUTPUT_DIR / "AMENDMENT6_GENERATION_CLOSED.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional GPT-4o VIKI Amendment 6")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--split", choices=("ood", "id"), required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--split", choices=("ood", "id"), required=True)
    run_parser.add_argument("--arm", choices=("trajectory_rag", "tom"), required=True)
    run_parser.add_argument("--workers", type=int, default=8)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--split", choices=("ood", "id"), required=True)
    subparsers.add_parser("close")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "preflight":
            result = preflight(args.split)
        elif args.command == "run":
            result = run_arm(args.split, args.arm, args.workers)
        elif args.command == "close":
            result = close()
        else:
            result = analyze(args.split)
    except GateFailure as error:
        print(f"GATE FAILURE: {error}")
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
