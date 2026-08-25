#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import tiktoken
from viki_amendment3_f2 import exact_mcnemar_p
from viki_amendment5 import (
    BENCHMARK_ROOT,
    H0_PATH,
    atomic_json,
    file_sha256,
    fingerprint,
    load_bench,
    load_jsonl,
    messages_sha256,
    native,
    write_jsonl_snapshot,
)
from viki_amendment6_gpt4o import (
    ENCODING,
    MODEL_ID,
    REQUEST_MODEL,
    memory_delta,
    message_text,
    runtime_identity,
    source_configuration,
    trim_prompt,
    validate_source,
)
from viki_amendment7 import (
    AMENDMENT6_DIR,
    OUTPUT_DIR,
    PLAN_MAX_TOKENS,
    PREREGISTRATION_PATH,
    REQUIRED_PLAN_CALLS,
    TOKEN_TOLERANCE,
    GateFailure,
    require_frozen,
    split_configuration,
)

from habitat_llm.evaluation.viki_memory_skill import add_memory_to_messages

BACKBONE = "gpt_4o_optional"
MODES = ("A", "B")
SPLITS = ("ood", "id")
OPTIONAL_PROTOCOL_PATH = OUTPUT_DIR / "gpt_4o_optional.protocol.json"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def record_protocol() -> Dict[str, Any]:
    frozen = require_frozen()
    existing_plan_results = [
        OUTPUT_DIR / f"mode_{mode}.{split}.jsonl" for mode in MODES for split in SPLITS
    ]
    if any(path.is_file() for path in existing_plan_results):
        raise GateFailure(
            "Optional protocol must be recorded before A7 plan generation"
        )
    optional = frozen["optional_backbone"]
    if optional.get("authorized_plan_generation_calls") != REQUIRED_PLAN_CALLS:
        raise GateFailure("Frozen optional GPT-4o call authorization changed")
    result = {
        "task": "Amendment7_GMemory_GPT4o_protocol_interpretation",
        "status": "FILED_BEFORE_A7_PLAN_GENERATION",
        "training": False,
        "gradient_updates": False,
        "fine_tuning": False,
        "authorized_plan_generation_calls": REQUIRED_PLAN_CALLS,
        "authorized_optional_memory_control_calls": 0,
        "source_clauses": [
            "Qwen2.5-VL-72B only, the sole backbone with a significant positive OOD result.",
            "Insight-extraction calls during the train build are counted separately and reported.",
            "GPT-4o is optional and runs only if the 72B arms complete.",
        ],
        "implementation": (
            "Replay each completed Qwen A7 row's memory context and replace only the "
            "plan-generation backbone with GPT-4o."
        ),
        "mode_A_label": "frozen Qwen memory-control context; GPT-4o plan generator",
        "mode_B_label": (
            "Qwen native cross-trial memory-control replay; GPT-4o plan generator; "
            "not GPT-native test-time adaptation"
        ),
        "qwen_arms_required_before_optional": True,
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
    }
    atomic_json(OPTIONAL_PROTOCOL_PATH, result)
    return result


def qwen_complete() -> None:
    for mode in MODES:
        for split in SPLITS:
            result_path = OUTPUT_DIR / f"mode_{mode}.{split}.jsonl"
            summary_path = result_path.with_suffix(".summary.json")
            analysis_path = OUTPUT_DIR / f"mode_{mode}.{split}.analysis.json"
            if any(
                not path.is_file()
                for path in (result_path, summary_path, analysis_path)
            ):
                raise GateFailure("Complete every 72B A7 arm and analysis first")
            summary = json.loads(summary_path.read_text())
            analysis = json.loads(analysis_path.read_text())
            if (
                summary.get("status") != "PASS"
                or summary.get("results_sha256") != file_sha256(result_path)
                or analysis.get("status") != "PASS"
            ):
                raise GateFailure(f"Invalid completed 72B Mode {mode} {split}")


def qwen_context(mode: str, split: str) -> Tuple[Path, Dict[int, Dict[str, Any]]]:
    path = OUTPUT_DIR / (
        f"mode_A.{split}.preflight.jsonl" if mode == "A" else f"mode_B.{split}.jsonl"
    )
    rows = load_jsonl(path)
    _, indices, _, _, _ = split_configuration(split)
    if set(rows) != set(indices):
        raise GateFailure(f"Invalid 72B context ledger for Mode {mode} {split}")
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.is_file():
        raise GateFailure(f"Missing 72B context certificate: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "PASS" or summary.get("results_sha256") != file_sha256(
        path
    ):
        raise GateFailure(f"Invalid 72B context certificate: {summary_path}")
    if mode == "B":
        run_path = path.with_suffix(path.suffix + ".run.json")
        if not run_path.is_file():
            raise GateFailure(f"Missing 72B context run metadata: {run_path}")
        run_hash = fingerprint(json.loads(run_path.read_text()))
        if any(row.get("run_fingerprint") != run_hash for row in rows.values()):
            raise GateFailure(f"Invalid 72B context row fingerprint: {path}")
    return path, rows


def optional_order(mode: str, split: str) -> List[Path]:
    order = [("A", "ood"), ("A", "id"), ("B", "ood"), ("B", "id")]
    current = order.index((mode, split))
    return [
        OUTPUT_DIR / f"{BACKBONE}.mode_{prior_mode}.{prior_split}.summary.json"
        for prior_mode, prior_split in order[:current]
    ]


def validate_optional_order(mode: str, split: str) -> None:
    for summary_path in optional_order(mode, split):
        if not summary_path.is_file():
            raise GateFailure(f"Missing prior optional summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        result_path = Path(str(summary_path).replace(".summary.json", ".jsonl"))
        if (
            summary.get("status") != "PASS"
            or summary.get("samples") not in (300, 1218)
            or not result_path.is_file()
            or summary.get("results_sha256") != file_sha256(result_path)
        ):
            raise GateFailure(f"Invalid prior optional summary: {summary_path}")


def preflight(mode: str, split: str) -> Dict[str, Any]:
    require_frozen()
    qwen_complete()
    dataset, indices, source_path, source_summary_path = source_configuration(split)
    source_records, source_run_hash, _ = validate_source(
        split, indices, source_path, source_summary_path
    )
    context_path, contexts = qwen_context(mode, split)
    encoding = tiktoken.get_encoding(ENCODING)
    bench = load_bench()
    records: Dict[int, Dict[str, Any]] = {}
    for index in indices:
        sample = native(dataset.iloc[index].to_dict())
        zero_messages = bench.get_messages(sample)
        original_text = message_text(zero_messages, "user")
        zero_tokens = int(source_records[index]["arms"]["zero_shot"]["prompt_tokens"])
        target_tokens = int(source_records[index]["arms"]["segment"]["prompt_tokens"])
        qwen_prompt = str(contexts[index]["final_prompt"])
        final_prompt = qwen_prompt
        injected_tokens = memory_delta(encoding, original_text, final_prompt)
        input_tokens = zero_tokens + injected_tokens
        if input_tokens > target_tokens:
            final_prompt, injected_tokens = trim_prompt(
                encoding,
                final_prompt,
                original_text,
                target_tokens - zero_tokens,
            )
            input_tokens = zero_tokens + injected_tokens
        relative_difference = abs(input_tokens - target_tokens) / target_tokens
        if relative_difference > TOKEN_TOLERANCE:
            raise GateFailure(f"GPT-4o Mode {mode} {split}:{index} token gate failed")
        messages = add_memory_to_messages(zero_messages, final_prompt)
        records[index] = {
            "index": index,
            "mode": mode,
            "split": split,
            "replay_semantics": "Qwen memory-control context; GPT-4o plan generator",
            "qwen_context_sha256": sha256_text(qwen_prompt),
            "qwen_context_ledger_sha256": file_sha256(context_path),
            "hierarchy_before_sha256": contexts[index]["hierarchy_before_sha256"],
            "hierarchy_after_sha256": contexts[index]["hierarchy_after_sha256"],
            "selected_memory_id": contexts[index]["selected_memory_id"],
            "candidate_memory_ids": contexts[index]["candidate_memory_ids"],
            "insights": contexts[index]["insights"],
            "final_prompt": final_prompt,
            "prompt_sha256": messages_sha256(messages),
            "input_tokens": input_tokens,
            "injected_tokens": injected_tokens,
            "segment_target_tokens": target_tokens,
            "relative_difference_from_segment": relative_difference,
            "gpt_tokenizer_trimmed": final_prompt != qwen_prompt,
        }
    output = OUTPUT_DIR / f"{BACKBONE}.mode_{mode}.{split}.preflight.jsonl"
    write_jsonl_snapshot(output, records, indices)
    summary = {
        "task": "Amendment7_GMemory_GPT4o_preflight",
        "status": "PASS",
        "backbone": BACKBONE,
        "mode": mode,
        "split": split,
        "rows": len(records),
        "plan_generation_calls": 0,
        "memory_control_calls": 0,
        "replay_semantics": "Qwen memory-control context; GPT-4o plan generator",
        "native_gpt4o_adaptation_claimed": False,
        "tokenizer": ENCODING,
        "maximum_token_difference": max(
            row["relative_difference_from_segment"] for row in records.values()
        ),
        "trimmed_rows": sum(row["gpt_tokenizer_trimmed"] for row in records.values()),
        "qwen_context_ledger": str(context_path),
        "qwen_context_ledger_sha256": file_sha256(context_path),
        "source_run_fingerprint": source_run_hash,
        "source_results_sha256": file_sha256(source_path),
        "results": str(output),
        "results_sha256": file_sha256(output),
    }
    atomic_json(output.with_suffix(".summary.json"), summary)
    return summary


def pending_path(output: Path, index: int) -> Path:
    directory = output.with_suffix(output.suffix + ".pending")
    directory.mkdir(exist_ok=True)
    return directory / f"{index}.json"


def run(mode: str, split: str, workers: int) -> Dict[str, Any]:
    require_frozen()
    qwen_complete()
    if os.environ.get("CLOUDGPT_USE_AZURE_CLI") != "1":
        raise GateFailure("Set CLOUDGPT_USE_AZURE_CLI=1 for CloudGPT")
    validate_optional_order(mode, split)
    dataset, indices, source_path, source_summary_path = source_configuration(split)
    _, source_run_hash, _ = validate_source(
        split, indices, source_path, source_summary_path
    )
    preflight_path = OUTPUT_DIR / f"{BACKBONE}.mode_{mode}.{split}.preflight.jsonl"
    preflight_summary_path = preflight_path.with_suffix(".summary.json")
    if not preflight_path.is_file() or not preflight_summary_path.is_file():
        raise GateFailure(f"Run GPT-4o Mode {mode} {split} preflight first")
    preflight_rows = load_jsonl(preflight_path)
    preflight_summary = json.loads(preflight_summary_path.read_text())
    if (
        preflight_summary.get("status") != "PASS"
        or preflight_summary.get("results_sha256") != file_sha256(preflight_path)
        or set(preflight_rows) != set(indices)
    ):
        raise GateFailure(f"Stale GPT-4o Mode {mode} {split} preflight")
    context_path, _ = qwen_context(mode, split)
    if preflight_summary.get("qwen_context_ledger_sha256") != file_sha256(context_path):
        raise GateFailure(f"Qwen Mode {mode} {split} context ledger changed")
    metadata = {
        "task": "Amendment7_GMemory_GPT4o_plan_replay",
        "backbone": BACKBONE,
        "model_id": MODEL_ID,
        "request_model": REQUEST_MODEL,
        "mode": mode,
        "split": split,
        "indices": indices,
        "runtime": runtime_identity(),
        "temperature": 0,
        "max_tokens": PLAN_MAX_TOKENS,
        "workers": workers,
        "replay_semantics": "Qwen memory-control context; GPT-4o plan generator",
        "native_gpt4o_adaptation_claimed": False,
        "memory_control_calls": 0,
        "preflight_sha256": file_sha256(preflight_path),
        "qwen_context_ledger": str(context_path),
        "qwen_context_ledger_sha256": file_sha256(context_path),
        "source_run_fingerprint": source_run_hash,
        "source_results_sha256": file_sha256(source_path),
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
        "h0_sha256": file_sha256(H0_PATH),
    }
    run_hash = fingerprint(metadata)
    output = OUTPUT_DIR / f"{BACKBONE}.mode_{mode}.{split}.jsonl"
    run_path = output.with_suffix(output.suffix + ".run.json")
    if run_path.is_file() and json.loads(run_path.read_text()) != metadata:
        raise GateFailure(f"Cannot resume GPT-4o Mode {mode} {split}: metadata differs")
    atomic_json(run_path, metadata)
    records = load_jsonl(output)
    for index, row in records.items():
        preflight_row = preflight_rows.get(index)
        if (
            index not in indices
            or preflight_row is None
            or row.get("run_fingerprint") != run_hash
            or row.get("backbone") != BACKBONE
            or row.get("mode") != mode
            or row.get("split") != split
            or row.get("prompt_sha256") != preflight_row["prompt_sha256"]
            or row.get("qwen_context_sha256") != preflight_row["qwen_context_sha256"]
            or row.get("hierarchy_before_sha256")
            != preflight_row["hierarchy_before_sha256"]
            or row.get("hierarchy_after_sha256")
            != preflight_row["hierarchy_after_sha256"]
            or row.get("prompt_tokens") != preflight_row["input_tokens"]
            or sha256_text(str(row.get("response", ""))) != row.get("response_sha256")
            or row.get("task_score") not in (0, 1)
            or row.get("format_score") not in (0, 1)
        ):
            raise GateFailure(f"Invalid GPT-4o Mode {mode} {split}:{index} checkpoint")
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    from cloudgpt_aoai import get_openai_client

    client = get_openai_client(timeout=3600, max_retries=5)

    def request_for(index: int) -> Dict[str, Any]:
        return {
            "index": index,
            "run_fingerprint": run_hash,
            "prompt_sha256": preflight_rows[index]["prompt_sha256"],
            "request_model": REQUEST_MODEL,
            "max_tokens": PLAN_MAX_TOKENS,
            "temperature": 0,
        }

    journal_directory = output.with_suffix(output.suffix + ".pending")
    journal_directory.mkdir(exist_ok=True)
    for journal_path in journal_directory.glob("*.json"):
        try:
            index = int(journal_path.stem)
        except ValueError as error:
            raise GateFailure(
                f"Invalid GPT-4o pending journal: {journal_path}"
            ) from error
        if index not in records:
            continue
        journal = json.loads(journal_path.read_text())
        row = records[index]
        if (
            journal.get("status") != "response_recorded"
            or journal.get("remote_call_consumed") is not True
            or journal.get("request") != request_for(index)
            or journal.get("response_sha256") != row["response_sha256"]
            or journal.get("prompt_tokens") != row["prompt_tokens"]
            or journal.get("completion_tokens") != row["completion_tokens"]
        ):
            raise GateFailure(f"GPT-4o completed journal differs: {journal_path}")
        journal_path.unlink()

    def generate(index: int) -> Tuple[int, Dict[str, Any]]:
        sample = native(dataset.iloc[index].to_dict())
        row = preflight_rows[index]
        messages = add_memory_to_messages(
            bench.get_messages(sample), row["final_prompt"]
        )
        prompt_sha256 = messages_sha256(messages)
        if prompt_sha256 != row["prompt_sha256"]:
            raise GateFailure(f"GPT-4o Mode {mode} {split}:{index} prompt changed")
        request = request_for(index)
        journal_path = pending_path(output, index)
        response: str
        prompt_tokens: Any
        completion_tokens: Any
        if journal_path.is_file():
            journal = json.loads(journal_path.read_text())
            if journal.get("request") != request:
                raise GateFailure(
                    f"GPT-4o pending request changed at {mode}:{split}:{index}"
                )
            if journal.get("status") != "response_recorded":
                raise GateFailure(
                    f"Unknown GPT-4o remote outcome at {mode}:{split}:{index}"
                )
            response = str(journal["response"])
            if sha256_text(response) != journal.get("response_sha256"):
                raise GateFailure(
                    f"GPT-4o pending response changed at {mode}:{split}:{index}"
                )
            prompt_tokens = journal.get("prompt_tokens")
            completion_tokens = journal.get("completion_tokens")
        else:
            atomic_json(
                journal_path, {"status": "request_recorded", "request": request}
            )
            completion = client.chat.completions.create(
                model=REQUEST_MODEL,
                messages=messages,
                max_tokens=PLAN_MAX_TOKENS,
                temperature=0,
            )
            usage = completion.usage
            response = completion.choices[0].message.content or ""
            prompt_tokens = None if usage is None else usage.prompt_tokens
            completion_tokens = None if usage is None else usage.completion_tokens
            atomic_json(
                journal_path,
                {
                    "status": "response_recorded",
                    "request": request,
                    "response": response,
                    "response_sha256": sha256_text(response),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "remote_call_consumed": True,
                },
            )
        if prompt_tokens != int(row["input_tokens"]):
            raise GateFailure(
                f"GPT-4o Mode {mode} {split}:{index} prompt tokens changed"
            )
        metrics = bench.score_response(
            scorer, 2, response, bench.get_ground_truth(sample), index
        )
        return index, {
            "index": index,
            "run_fingerprint": run_hash,
            "backbone": BACKBONE,
            "mode": mode,
            "split": split,
            "replay_semantics": metadata["replay_semantics"],
            "prompt_sha256": prompt_sha256,
            "qwen_context_sha256": row["qwen_context_sha256"],
            "hierarchy_before_sha256": row["hierarchy_before_sha256"],
            "hierarchy_after_sha256": row["hierarchy_after_sha256"],
            "selected_memory_id": row["selected_memory_id"],
            "candidate_memory_ids": row["candidate_memory_ids"],
            "insights": row["insights"],
            "response": response,
            "response_sha256": sha256_text(response),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "injected_tokens": row["injected_tokens"],
            **metrics,
        }

    pending_indices = [index for index in indices if index not in records]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate, index): index for index in pending_indices}
        for future in as_completed(futures):
            index = futures[future]
            try:
                _, record = future.result()
            except Exception:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            records[index] = record
            write_jsonl_snapshot(output, records, indices)
            pending_path(output, index).unlink()
    if set(records) != set(indices):
        raise GateFailure(f"GPT-4o Mode {mode} {split} generation incomplete")
    unresolved = list(output.with_suffix(output.suffix + ".pending").glob("*.json"))
    if unresolved:
        raise GateFailure(f"GPT-4o Mode {mode} {split} has pending journals")
    successes = sum(row["task_score"] == 1 for row in records.values())
    formats = sum(row["format_score"] == 1 for row in records.values())
    summary = {
        "task": "Amendment7_GMemory_GPT4o_plan_replay",
        "status": "PASS",
        "backbone": BACKBONE,
        "mode": mode,
        "split": split,
        "samples": len(records),
        "successes": successes,
        "accuracy": successes / len(records),
        "format_successes": formats,
        "format_compliance": formats / len(records),
        "plan_generation_calls": len(records),
        "memory_control_calls": 0,
        "replay_semantics": metadata["replay_semantics"],
        "native_gpt4o_adaptation_claimed": False,
        "run_fingerprint": run_hash,
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in records.values()),
        "completion_tokens": sum(
            int(row["completion_tokens"]) for row in records.values()
        ),
        "results": str(output),
        "results_sha256": file_sha256(output),
    }
    atomic_json(output.with_suffix(".summary.json"), summary)
    return summary


def comparison(frame: pd.DataFrame, arm: str, baseline: str) -> Dict[str, Any]:
    arm_values = frame[f"{arm}_task_score"].to_numpy(dtype=bool)
    baseline_values = frame[f"{baseline}_task_score"].to_numpy(dtype=bool)
    baseline_only = int((baseline_values & ~arm_values).sum())
    arm_only = int((~baseline_values & arm_values).sum())
    return {
        "baseline": baseline,
        "baseline_successes": int(baseline_values.sum()),
        "gmemory_successes": int(arm_values.sum()),
        "baseline_accuracy": float(baseline_values.mean()),
        "gmemory_accuracy": float(arm_values.mean()),
        "gmemory_minus_baseline": float(arm_values.mean() - baseline_values.mean()),
        "baseline_only": baseline_only,
        "gmemory_only": arm_only,
        "discordant_pairs": baseline_only + arm_only,
        "mcnemar_exact_p": exact_mcnemar_p(baseline_only, arm_only),
    }


def analyze(mode: str, split: str) -> Dict[str, Any]:
    require_frozen()
    dataset, indices, source_path, source_summary_path = source_configuration(split)
    del dataset
    source_records, source_run_hash, _ = validate_source(
        split, indices, source_path, source_summary_path
    )
    result_path = OUTPUT_DIR / f"{BACKBONE}.mode_{mode}.{split}.jsonl"
    result_summary_path = result_path.with_suffix(".summary.json")
    qwen_path = OUTPUT_DIR / f"mode_{mode}.{split}.jsonl"
    qwen_summary_path = qwen_path.with_suffix(".summary.json")
    qwen_run_path = qwen_path.with_suffix(qwen_path.suffix + ".run.json")
    trajectory_path = AMENDMENT6_DIR / f"{BACKBONE}.{split}.trajectory_rag.jsonl"
    trajectory_summary_path = trajectory_path.with_suffix(".summary.json")
    trajectory_run_path = trajectory_path.with_suffix(
        trajectory_path.suffix + ".run.json"
    )
    required = (
        result_path,
        result_summary_path,
        qwen_path,
        qwen_summary_path,
        qwen_run_path,
        trajectory_path,
        trajectory_summary_path,
        trajectory_run_path,
    )
    if any(not path.is_file() for path in required):
        raise GateFailure(f"Missing GPT-4o Mode {mode} {split} analysis input")
    optional_rows = load_jsonl(result_path)
    qwen_rows = load_jsonl(qwen_path)
    trajectory_rows = load_jsonl(trajectory_path)
    result_summary = json.loads(result_summary_path.read_text())
    qwen_summary = json.loads(qwen_summary_path.read_text())
    qwen_run_hash = fingerprint(json.loads(qwen_run_path.read_text()))
    trajectory_summary = json.loads(trajectory_summary_path.read_text())
    trajectory_run_hash = fingerprint(json.loads(trajectory_run_path.read_text()))
    if (
        set(optional_rows) != set(indices)
        or set(qwen_rows) != set(indices)
        or set(trajectory_rows) != set(indices)
        or result_summary.get("status") != "PASS"
        or result_summary.get("results_sha256") != file_sha256(result_path)
        or qwen_summary.get("status") != "PASS"
        or qwen_summary.get("samples") != len(indices)
        or qwen_summary.get("results_sha256") != file_sha256(qwen_path)
        or any(
            row.get("run_fingerprint") != qwen_run_hash for row in qwen_rows.values()
        )
        or trajectory_summary.get("samples") != len(indices)
        or trajectory_summary.get("results_sha256") != file_sha256(trajectory_path)
        or any(
            row.get("run_fingerprint") != trajectory_run_hash
            for row in trajectory_rows.values()
        )
    ):
        raise GateFailure(f"Invalid GPT-4o Mode {mode} {split} analysis input")
    mode_a_rows = (
        load_jsonl(OUTPUT_DIR / f"{BACKBONE}.mode_A.{split}.jsonl")
        if mode == "B"
        else {}
    )
    if mode == "B":
        mode_a_path = OUTPUT_DIR / f"{BACKBONE}.mode_A.{split}.jsonl"
        mode_a_summary_path = mode_a_path.with_suffix(".summary.json")
        mode_a_run_path = mode_a_path.with_suffix(mode_a_path.suffix + ".run.json")
        if not mode_a_summary_path.is_file() or not mode_a_run_path.is_file():
            raise GateFailure(f"Missing GPT-4o Mode A {split} analysis input")
        mode_a_summary = json.loads(mode_a_summary_path.read_text())
        mode_a_run_hash = fingerprint(json.loads(mode_a_run_path.read_text()))
        if (
            set(mode_a_rows) != set(indices)
            or mode_a_summary.get("status") != "PASS"
            or mode_a_summary.get("results_sha256") != file_sha256(mode_a_path)
            or any(
                row.get("run_fingerprint") != mode_a_run_hash
                for row in mode_a_rows.values()
            )
        ):
            raise GateFailure(f"Invalid GPT-4o Mode A {split} analysis input")
    rows = []
    for index in indices:
        row = {
            "index": index,
            "gmemory_task_score": int(optional_rows[index]["task_score"]),
            "gmemory_format_score": int(optional_rows[index]["format_score"]),
            "qwen_gmemory_task_score": int(qwen_rows[index]["task_score"]),
            "qwen_gmemory_format_score": int(qwen_rows[index]["format_score"]),
            "segment_task_score": int(
                source_records[index]["arms"]["segment"]["task_score"]
            ),
            "segment_format_score": int(
                source_records[index]["arms"]["segment"]["format_score"]
            ),
            "trajectory_rag_task_score": int(trajectory_rows[index]["task_score"]),
            "trajectory_rag_format_score": int(trajectory_rows[index]["format_score"]),
        }
        if mode == "B":
            row["mode_A_task_score"] = int(mode_a_rows[index]["task_score"])
            row["mode_A_format_score"] = int(mode_a_rows[index]["format_score"])
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("index")
    row_path = OUTPUT_DIR / f"{BACKBONE}.mode_{mode}.{split}.analysis_rows.parquet"
    frame.to_parquet(row_path, index=False)
    baselines = ["segment", "trajectory_rag", "qwen_gmemory"]
    if mode == "B":
        baselines.insert(0, "mode_A")
    summary = {
        "task": "Amendment7_GMemory_GPT4o_plan_replay_analysis",
        "status": "PASS",
        "backbone": BACKBONE,
        "mode": mode,
        "split": split,
        "samples": len(frame),
        "replay_semantics": "Qwen memory-control context; GPT-4o plan generator",
        "native_gpt4o_adaptation_claimed": False,
        "comparisons": {
            baseline: comparison(frame, "gmemory", baseline) for baseline in baselines
        },
        "format_compliance": {
            arm: float(frame[f"{arm}_format_score"].mean())
            for arm in ["gmemory", *baselines]
        },
        "model_generation_calls": 0,
        "memory_control_calls": 0,
        "source_run_fingerprint": source_run_hash,
        "row_results": str(row_path),
        "row_results_sha256": file_sha256(row_path),
        "mode_results_sha256": file_sha256(result_path),
        "qwen_context_results_sha256": file_sha256(qwen_path),
        "trajectory_rag_results_sha256": file_sha256(trajectory_path),
        "segment_source_results_sha256": file_sha256(source_path),
    }
    output = OUTPUT_DIR / f"{BACKBONE}.mode_{mode}.{split}.analysis.json"
    atomic_json(output, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional GPT-4o VIKI Amendment 7")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("record-protocol")
    for command in ("preflight", "run", "analyze"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--mode", choices=MODES, required=True)
        subparser.add_argument("--split", choices=SPLITS, required=True)
        if command == "run":
            subparser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "record-protocol":
            result = record_protocol()
        elif args.command == "preflight":
            result = preflight(args.mode, args.split)
        elif args.command == "run":
            result = run(args.mode, args.split, args.workers)
        else:
            result = analyze(args.mode, args.split)
    except GateFailure as error:
        print(f"GATE FAILURE: {error}")
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
