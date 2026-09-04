#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from viki_amendment3_f2 import exact_mcnemar_p
from viki_amendment4 import segment_flat_block
from viki_amendment5 import (
    BACKBONES,
    BENCHMARK_ROOT,
    DATA_ROOT,
    EMBEDDING_MODEL,
    H0_PATH,
    M0_INSTANCES_PATH,
    MAX_OUTPUT_TOKENS,
    MIN_CONTEXT_LENGTH,
    SIMILARITY_THRESHOLD,
    atomic_json,
    file_sha256,
    fingerprint,
    load_bench,
    load_jsonl,
    messages_sha256,
    native,
    token_count,
    validate_local_service,
    write_jsonl_snapshot,
)

from habitat_llm.evaluation.viki_branch_conditions import get_instruction
from habitat_llm.evaluation.viki_memory_skill import (
    add_memory_to_messages,
    get_prompt_context,
)
from habitat_llm.evaluation.viki_segment_memory import SegmentInstance

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment6"
PREREGISTRATION_PATH = OUTPUT_DIR / "preregistration.json"
SUPERSESSION_PATH = OUTPUT_DIR / "closure_supersession.json"
ROW_EMBEDDING_CACHE_PATH = OUTPUT_DIR / "trajectory_row_embeddings.npz"
AMENDMENT5_DIR = ROOT / "results/viki_memory_experiments/amendment5"
AMENDMENT5_1_DIR = ROOT / "results/viki_memory_experiments/amendment5_1"
AMENDMENT1_DIR = ROOT / "results/viki_memory_experiments/amendment1"
ID_SLICE_PATH = AMENDMENT1_DIR / "a5_id_safety_manifest.parquet"
from viki_amendment5 import BACKBONE, SERVED_MODEL_FOR_BACKBONE  # noqa: E402

SERVED_MODEL = SERVED_MODEL_FOR_BACKBONE
MODEL_ID = BACKBONES[BACKBONE]["model_id"]
MODEL_REVISION = BACKBONES[BACKBONE]["model_revision"]
SEED = 20260814
TOKEN_TOLERANCE = 0.05
OOD_ROWS = 1218
ID_ROWS = 300

TOM_REASONING_TEMPLATE = """Before producing the executable plan, reason explicitly about coordination.
1. Identify every active robot, its available operations, and the task-relevant scene state visible in the image.
2. For each other active robot, infer what part of the task it can execute, what it is likely to know from the shared scene, and which actions should be assigned to it.
3. Choose complementary assignments that avoid duplicated work and respect operation availability.
4. If only one robot is active, perform the same explicit capability and scene-state reasoning for that robot.
Then return the plan in the original required <think>...</think><answer>...</answer> format. Do not mention this instruction in the answer and do not invent entities or operations."""


class GateFailure(RuntimeError):
    pass


def require_preregistration() -> Dict[str, Any]:
    if not PREREGISTRATION_PATH.is_file():
        raise GateFailure("Freeze Amendment 6 before any deployment operation")
    observed = json.loads(PREREGISTRATION_PATH.read_text())
    if observed != preregistration():
        raise GateFailure("Amendment 6 preregistration or H0 identity changed")
    return observed


def split_configuration(
    split: str,
) -> Tuple[pd.DataFrame, List[int], Path, Path, Path]:
    if split == "ood":
        dataset = pd.read_parquet(DATA_ROOT / "val.parquet")
        indices = list(range(OOD_ROWS))
        manifest_path = AMENDMENT5_DIR / "deployment_manifest.jsonl"
        source_path = AMENDMENT5_DIR / "qwen2_5_vl_72b.jsonl"
        source_summary_path = AMENDMENT5_DIR / "qwen2_5_vl_72b.summary.json"
    elif split == "id":
        dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
        indices = [int(value) for value in pd.read_parquet(ID_SLICE_PATH)["index"]]
        manifest_path = AMENDMENT5_1_DIR / "id_deployment_manifest.jsonl"
        source_path = AMENDMENT5_1_DIR / "qwen2_5_vl_72b.jsonl"
        source_summary_path = AMENDMENT5_1_DIR / "qwen2_5_vl_72b.summary.json"
    else:
        raise GateFailure(f"Unknown split: {split}")
    expected_rows = OOD_ROWS if split == "ood" else ID_ROWS
    if len(indices) != expected_rows or len(set(indices)) != expected_rows:
        raise GateFailure(f"Frozen {split} indices are incomplete")
    return dataset, indices, manifest_path, source_path, source_summary_path


def load_source_rows() -> Tuple[List[int], Dict[int, List[SegmentInstance]], List[str]]:
    raw_instances = [
        json.loads(line) for line in M0_INSTANCES_PATH.read_text().splitlines() if line
    ]
    if len(raw_instances) != 19499:
        raise GateFailure(f"Expected 19,499 M0 segments, observed {len(raw_instances)}")
    grouped: Dict[int, List[Tuple[int, SegmentInstance]]] = defaultdict(list)
    instructions: Dict[int, set[str]] = defaultdict(set)
    for value in raw_instances:
        source = int(value["source_train_index"])
        grouped[source].append(
            (int(value["segment_index"]), SegmentInstance.from_dict(value))
        )
        instructions[source].add(str(value["self_cond"]["instruction"]))
    source_ids = sorted(grouped)
    if len(source_ids) != 6699:
        raise GateFailure(
            f"Expected 6,699 valid M0 source rows, observed {len(source_ids)}"
        )
    rows: Dict[int, List[SegmentInstance]] = {}
    row_instructions = []
    for source in source_ids:
        values = sorted(grouped[source])
        if [position for position, _ in values] != list(range(len(values))):
            raise GateFailure(f"Non-contiguous segment order in source row {source}")
        if len(instructions[source]) != 1:
            raise GateFailure(f"Mixed instructions in source row {source}")
        rows[source] = [instance for _, instance in values]
        row_instructions.append(next(iter(instructions[source])))
    return source_ids, rows, row_instructions


def row_embedding_fingerprint(
    source_ids: Sequence[int], instructions: Sequence[str]
) -> str:
    return fingerprint(
        {
            "embedding_model": EMBEDDING_MODEL,
            "source_ids": list(source_ids),
            "instructions": list(instructions),
        }
    )


def load_row_embeddings(
    source_ids: Sequence[int], instructions: Sequence[str]
) -> np.ndarray:
    cache_key = row_embedding_fingerprint(source_ids, instructions)
    if ROW_EMBEDDING_CACHE_PATH.is_file():
        cache = np.load(ROW_EMBEDDING_CACHE_PATH, allow_pickle=False)
        if str(cache["cache_key"].item()) == cache_key and cache[
            "embeddings"
        ].shape == (6699, 768):
            return np.asarray(cache["embeddings"])
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    embeddings = np.asarray(
        model.encode(
            list(instructions),
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )
    np.savez_compressed(
        ROW_EMBEDDING_CACHE_PATH,
        cache_key=np.asarray(cache_key),
        embeddings=embeddings,
    )
    return embeddings


def format_trajectory_prompt(rows: Sequence[Sequence[SegmentInstance]]) -> str:
    if not rows:
        return ""
    blocks = [segment_flat_block(instance) for row in rows for instance in row]
    blocks.extend(
        [
            "Use the examples only as grounded sub-plan guidance.",
            "Produce one complete plan for the current task using only its available robot APIs.",
        ]
    )
    return "\n".join(blocks)


def add_tom_to_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = copy.deepcopy(messages)
    system = next(message for message in result if message["role"] == "system")
    if not isinstance(system["content"], str):
        raise GateFailure("Official VIKI system message is not text")
    system["content"] = f"{system['content']}\n\n{TOM_REASONING_TEMPLATE}"
    return result


def trim_prompt(
    sample: Dict[str, Any],
    base_url: str,
    tokenizer: Any,
    prompt: str,
    target_tokens: int,
) -> Tuple[str, int]:
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    observed: Dict[int, Tuple[str, int]] = {}

    def evaluate(count: int) -> Tuple[str, int]:
        count = min(len(encoded), max(1, count))
        if count not in observed:
            candidate = tokenizer.decode(encoded[:count], skip_special_tokens=False)
            messages = add_memory_to_messages(
                load_bench().get_messages(sample), candidate
            )
            observed[count] = (
                candidate,
                token_count(base_url, SERVED_MODEL, messages),
            )
        return observed[count]

    lower, upper = 1, len(encoded)
    evaluate(upper)
    while lower <= upper:
        middle = (lower + upper) // 2
        if evaluate(middle)[1] < target_tokens:
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
    return min(observed.values(), key=lambda item: abs(item[1] - target_tokens))


def validate_reused_source(
    split: str,
    indices: Sequence[int],
    source_path: Path,
    source_summary_path: Path,
) -> Tuple[Dict[int, Dict[str, Any]], str, Dict[str, Any]]:
    run_path = source_path.with_suffix(source_path.suffix + ".run.json")
    required = [source_path, source_summary_path, run_path]
    if any(not path.is_file() for path in required):
        raise GateFailure(f"Missing certified source artifact for {split}")
    records = load_jsonl(source_path)
    metadata = json.loads(run_path.read_text())
    summary = json.loads(source_summary_path.read_text())
    run_hash = fingerprint(metadata)
    if (
        set(records) != set(indices)
        or summary.get("samples") != len(indices)
        or (
            # Amendment 5 summaries predate this field; validate it only when the
            # certificate carries it, matching how run_fingerprint is handled.
            summary.get("results_sha256") is not None
            and summary.get("results_sha256") != file_sha256(source_path)
        )
        or (
            summary.get("run_fingerprint") is not None
            and summary.get("run_fingerprint") != run_hash
        )
        or any(record.get("run_fingerprint") != run_hash for record in records.values())
        or any(
            set(record.get("arms", {})) < {"zero_shot", "segment"}
            for record in records.values()
        )
    ):
        raise GateFailure(
            f"Certified source coverage or fingerprint failed for {split}"
        )
    expected = {"ood": (10, 83), "id": (22, 69)}[split]
    observed = tuple(
        sum(record["arms"][arm]["task_score"] == 1 for record in records.values())
        for arm in ("zero_shot", "segment")
    )
    if observed != expected:
        raise GateFailure(
            f"Certified source task outcomes changed for {split}: {observed}"
        )
    return records, run_hash, summary


def preflight(split: str, base_url: str) -> Dict[str, Any]:
    prereg = require_preregistration()
    runtime = validate_local_service(BACKBONE, base_url)
    if int(runtime["models"][0]["max_model_len"]) < MIN_CONTEXT_LENGTH:
        raise GateFailure("Amendment 6 requires at least 16,384 context tokens")
    (
        dataset,
        indices,
        segment_manifest_path,
        source_path,
        source_summary_path,
    ) = split_configuration(split)
    segment_manifest = load_jsonl(segment_manifest_path)
    if set(segment_manifest) != set(indices):
        raise GateFailure(f"Frozen segment manifest coverage changed for {split}")
    source_records, source_run_hash, source_summary = validate_reused_source(
        split, indices, source_path, source_summary_path
    )
    source_ids, source_rows, source_instructions = load_source_rows()
    row_embeddings = load_row_embeddings(source_ids, source_instructions)
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    bench = load_bench()
    sampled_indices = set(
        np.random.default_rng(SEED).choice(indices, size=20, replace=False).tolist()
    )
    records: Dict[int, Dict[str, Any]] = {}
    render_gate_rows = []
    for position, index in enumerate(indices):
        sample = native(dataset.iloc[index].to_dict())
        zero_messages = bench.get_messages(sample)
        segment_messages = add_memory_to_messages(
            bench.get_messages(sample), segment_manifest[index]["segment_prompt"]
        )
        if (
            messages_sha256(zero_messages)
            != segment_manifest[index]["prompt_sha256"]["zero_shot"]
            or messages_sha256(segment_messages)
            != segment_manifest[index]["prompt_sha256"]["segment"]
            or source_records[index]["prompt_sha256"]["zero_shot"]
            != segment_manifest[index]["prompt_sha256"]["zero_shot"]
            or source_records[index]["prompt_sha256"]["segment"]
            != segment_manifest[index]["prompt_sha256"]["segment"]
        ):
            raise GateFailure(f"Frozen reused prompt changed at {split}:{index}")
        zero_tokens = token_count(base_url, SERVED_MODEL, zero_messages)
        segment_tokens = token_count(base_url, SERVED_MODEL, segment_messages)
        similarities = row_embeddings @ query_embeddings[position]
        ranking = sorted(
            range(len(source_ids)),
            key=lambda item: (-float(similarities[item]), source_ids[item]),
        )
        best_similarity = float(similarities[ranking[0]])
        selected_sources: List[int] = []
        selected_rows: List[List[SegmentInstance]] = []
        trajectory_prompt = ""
        trajectory_messages = zero_messages
        trajectory_tokens = zero_tokens
        if best_similarity >= SIMILARITY_THRESHOLD:
            for ranked_position in ranking:
                source = source_ids[ranked_position]
                selected_sources.append(source)
                selected_rows.append(source_rows[source])
                trajectory_prompt = format_trajectory_prompt(selected_rows)
                trajectory_messages = add_memory_to_messages(
                    bench.get_messages(sample), trajectory_prompt
                )
                trajectory_tokens = token_count(
                    base_url, SERVED_MODEL, trajectory_messages
                )
                if trajectory_tokens >= segment_tokens:
                    break
            if trajectory_tokens > segment_tokens:
                trajectory_prompt, trajectory_tokens = trim_prompt(
                    sample, base_url, tokenizer, trajectory_prompt, segment_tokens
                )
                trajectory_messages = add_memory_to_messages(
                    bench.get_messages(sample), trajectory_prompt
                )
        relative_difference = abs(trajectory_tokens - segment_tokens) / segment_tokens
        if relative_difference > TOKEN_TOLERANCE:
            raise GateFailure(
                f"Trajectory RAG outside 5% token band at {split}:{index}: "
                f"trajectory={trajectory_tokens}, segment={segment_tokens}"
            )
        tom_messages = add_tom_to_messages(bench.get_messages(sample))
        tom_tokens = token_count(base_url, SERVED_MODEL, tom_messages)
        if max(
            zero_tokens, segment_tokens, trajectory_tokens, tom_tokens
        ) + MAX_OUTPUT_TOKENS > int(runtime["models"][0]["max_model_len"]):
            raise GateFailure(f"Context gate failed at {split}:{index}")
        if index in sampled_indices:
            for row_rank, source in enumerate(selected_sources, 1):
                expected = "\n".join(
                    segment_flat_block(item) for item in source_rows[source]
                )
                observed = "\n".join(
                    segment_flat_block(item) for item in selected_rows[row_rank - 1]
                )
                render_gate_rows.append(
                    {
                        "index": index,
                        "row_rank": row_rank,
                        "source_train_index": source,
                        "segment_count": len(source_rows[source]),
                        "expected_sha256": hashlib.sha256(
                            expected.encode()
                        ).hexdigest(),
                        "observed_sha256": hashlib.sha256(
                            observed.encode()
                        ).hexdigest(),
                        "byte_identical": observed == expected,
                    }
                )
        _, robots, _ = get_prompt_context(sample)
        ground_truth = bench.get_ground_truth(sample)
        records[index] = {
            "index": index,
            "split": split,
            "instruction": get_instruction(sample),
            "family": str(ground_truth["task_name"]),
            "active_robot_count": len(robots),
            "trajectory_rag": {
                "prompt": trajectory_prompt,
                "best_similarity": best_similarity,
                "fallback_bare": not selected_sources,
                "selected_source_indices_before_token_trim": selected_sources,
                "selected_similarities": [
                    float(similarities[source_ids.index(source)])
                    for source in selected_sources
                ],
                "input_tokens": trajectory_tokens,
                "injected_tokens": trajectory_tokens - zero_tokens,
                "relative_difference_from_segment": relative_difference,
            },
            "tom": {
                "template_sha256": hashlib.sha256(
                    TOM_REASONING_TEMPLATE.encode()
                ).hexdigest(),
                "input_tokens": tom_tokens,
                "injected_tokens": tom_tokens - zero_tokens,
            },
            "token_counts": {"zero_shot": zero_tokens, "segment": segment_tokens},
            "prompt_sha256": {
                "zero_shot": messages_sha256(zero_messages),
                "trajectory_rag": messages_sha256(trajectory_messages),
                "tom": messages_sha256(tom_messages),
                "segment": messages_sha256(segment_messages),
            },
            "reused_response_sha256": {
                arm: hashlib.sha256(
                    source_records[index]["arms"][arm]["response"].encode()
                ).hexdigest()
                for arm in ("zero_shot", "segment")
            },
        }
    output = OUTPUT_DIR / f"qwen2_5_vl_72b.{split}.preflight.jsonl"
    write_jsonl_snapshot(output, records, indices)
    render_frame = pd.DataFrame(render_gate_rows)
    render_path = OUTPUT_DIR / f"qwen2_5_vl_72b.{split}.render_gate.parquet"
    render_frame.to_parquet(render_path, index=False)
    if render_frame.empty or not render_frame["byte_identical"].all():
        raise GateFailure(f"20-row trajectory rendering gate failed for {split}")
    frame = pd.DataFrame(
        [
            {
                "index": index,
                "active_robot_count": records[index]["active_robot_count"],
                "zero_shot_input_tokens": records[index]["token_counts"]["zero_shot"],
                "segment_input_tokens": records[index]["token_counts"]["segment"],
                "trajectory_rag_input_tokens": records[index]["trajectory_rag"][
                    "input_tokens"
                ],
                "tom_input_tokens": records[index]["tom"]["input_tokens"],
                "trajectory_relative_difference": records[index]["trajectory_rag"][
                    "relative_difference_from_segment"
                ],
            }
            for index in indices
        ]
    )
    summary = {
        "task": "Amendment6_preflight",
        "status": "PASS",
        "split": split,
        "rows": len(records),
        "model_generation_calls": 0,
        "runtime": runtime,
        "source_pool_rows": len(source_ids),
        "source_pool_segments": sum(len(row) for row in source_rows.values()),
        "fallback_rows": sum(
            record["trajectory_rag"]["fallback_bare"] for record in records.values()
        ),
        "symmetric_drops": 0,
        "maximum_trajectory_token_difference": float(
            frame["trajectory_relative_difference"].max()
        ),
        "maximum_input_tokens": {
            arm: int(frame[f"{arm}_input_tokens"].max())
            for arm in ("zero_shot", "trajectory_rag", "tom", "segment")
        },
        "robot_count_distribution": {
            str(key): int(value)
            for key, value in frame["active_robot_count"]
            .value_counts()
            .sort_index()
            .items()
        },
        "rendering_gate": {
            "sampled_rows": sorted(sampled_indices),
            "row_exemplars_checked": len(render_frame),
            "byte_identical": int(render_frame["byte_identical"].sum()),
            "results": str(render_path),
            "results_sha256": file_sha256(render_path),
        },
        "reuse_gate": {
            "status": "PASS",
            "arms": ["zero_shot", "segment"],
            "rows": len(records),
            "response_hashes_verified": 2 * len(records),
            "source_run_fingerprint": source_run_hash,
            "source_results": str(source_path),
            "source_results_sha256": file_sha256(source_path),
            "source_summary_sha256": file_sha256(source_summary_path),
            "source_status": source_summary["status"],
        },
        "tom_prompt_structure": prereg["tom"],
        "tom_template_sha256": hashlib.sha256(
            TOM_REASONING_TEMPLATE.encode()
        ).hexdigest(),
        "results": str(output),
        "results_sha256": file_sha256(output),
        "segment_manifest_sha256": file_sha256(segment_manifest_path),
        "h0_sha256": file_sha256(H0_PATH),
    }
    atomic_json(output.with_suffix(".summary.json"), summary)
    return summary


def build_row_cache() -> Dict[str, Any]:
    require_preregistration()
    source_ids, source_rows, source_instructions = load_source_rows()
    embeddings = load_row_embeddings(source_ids, source_instructions)
    if embeddings.shape != (6699, 768):
        raise GateFailure(
            f"Unexpected trajectory row embedding shape: {embeddings.shape}"
        )
    summary = {
        "task": "Amendment6_trajectory_row_embedding_cache",
        "status": "PASS",
        "model_generation_calls": 0,
        "source_rows": len(source_ids),
        "source_segments": sum(len(row) for row in source_rows.values()),
        "source_index_minimum": min(source_ids),
        "source_index_maximum": max(source_ids),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_shape": list(embeddings.shape),
        "cache_key": row_embedding_fingerprint(source_ids, source_instructions),
        "results": str(ROW_EMBEDDING_CACHE_PATH),
        "results_sha256": file_sha256(ROW_EMBEDDING_CACHE_PATH),
        "h0_sha256": file_sha256(H0_PATH),
    }
    atomic_json(OUTPUT_DIR / "trajectory_row_embeddings.summary.json", summary)
    return summary


def required_completed_before(split: str, arm: str) -> List[Path]:
    order = [
        ("ood", "trajectory_rag"),
        ("ood", "tom"),
        ("id", "trajectory_rag"),
        ("id", "tom"),
    ]
    current = order.index((split, arm))
    return [
        OUTPUT_DIR / f"qwen2_5_vl_72b.{prior_split}.{prior_arm}.summary.json"
        for prior_split, prior_arm in order[:current]
    ]


def run_arm(split: str, arm: str, base_url: str, workers: int) -> Dict[str, Any]:
    require_preregistration()
    if arm not in {"trajectory_rag", "tom"}:
        raise GateFailure(f"Unknown Amendment 6 arm: {arm}")
    missing_prior = [
        str(path)
        for path in required_completed_before(split, arm)
        if not path.is_file()
    ]
    if missing_prior:
        raise GateFailure(
            "Required Amendment 6 order is incomplete: " + ", ".join(missing_prior)
        )
    runtime = validate_local_service(BACKBONE, base_url)
    preflight_path = OUTPUT_DIR / f"qwen2_5_vl_72b.{split}.preflight.jsonl"
    preflight_summary_path = preflight_path.with_suffix(".summary.json")
    if not preflight_path.is_file() or not preflight_summary_path.is_file():
        raise GateFailure(f"Run the {split} preflight before generation")
    preflight_summary = json.loads(preflight_summary_path.read_text())
    if (
        preflight_summary.get("status") != "PASS"
        or preflight_summary.get("runtime") != runtime
        or preflight_summary.get("results_sha256") != file_sha256(preflight_path)
        or preflight_summary.get("h0_sha256") != file_sha256(H0_PATH)
    ):
        raise GateFailure(f"Invalid or stale {split} preflight certificate")
    preflight_records = load_jsonl(preflight_path)
    dataset, indices, _, source_path, source_summary_path = split_configuration(split)
    source_records, source_run_hash, _ = validate_reused_source(
        split, indices, source_path, source_summary_path
    )
    if set(preflight_records) != set(indices):
        raise GateFailure(f"Incomplete {split} preflight row coverage")
    usage_surcharges = {
        index: int(source_records[index]["arms"]["zero_shot"]["prompt_tokens"])
        - int(preflight_records[index]["token_counts"]["zero_shot"])
        for index in indices
    }
    if any(value < 0 for value in usage_surcharges.values()):
        raise GateFailure(f"Invalid certified multimodal token surcharge for {split}")
    surcharge_distribution = {
        str(value): sum(observed == value for observed in usage_surcharges.values())
        for value in sorted(set(usage_surcharges.values()))
    }
    metadata = {
        "task": "Amendment6_deployment_baseline",
        "backbone": "qwen2_5_vl_72b",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "served_model": SERVED_MODEL,
        "split": split,
        "arm": arm,
        "indices": indices,
        "runtime": runtime,
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "workers": workers,
        "scorer_seed_rule": "original row index",
        "preflight_sha256": file_sha256(preflight_path),
        "preflight_summary_sha256": file_sha256(preflight_summary_path),
        "source_run_fingerprint": source_run_hash,
        "source_results_sha256": file_sha256(source_path),
        "prompt_token_accounting": {
            "budget_and_preflight": "vLLM /tokenize",
            "generation_usage": "chat completion usage.prompt_tokens",
            "provider_usage_rule": (
                "tokenizer input tokens plus the certified per-row multimodal "
                "surcharge from the reused zero-shot arm"
            ),
            "certified_surcharge_distribution": surcharge_distribution,
        },
        "h0_sha256": file_sha256(H0_PATH),
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
    }
    run_hash = fingerprint(metadata)
    output = OUTPUT_DIR / f"qwen2_5_vl_72b.{split}.{arm}.jsonl"
    run_path = output.with_suffix(output.suffix + ".run.json")
    if run_path.is_file() and json.loads(run_path.read_text()) != metadata:
        raise GateFailure(f"Cannot resume {split} {arm}: run metadata differs")
    atomic_json(run_path, metadata)
    records = load_jsonl(output)
    if any(
        index not in indices or record.get("run_fingerprint") != run_hash
        for index, record in records.items()
    ):
        raise GateFailure(f"Invalid checkpoint record in {split} {arm}")
    pending = [
        index
        for index in indices
        if index not in records or records[index].get("endpoint_error")
    ]
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    client = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=5, timeout=3600)

    def generate(index: int) -> Tuple[int, Dict[str, Any]]:
        sample = native(dataset.iloc[index].to_dict())
        row = preflight_records[index]
        if arm == "trajectory_rag":
            messages = add_memory_to_messages(
                bench.get_messages(sample), row["trajectory_rag"]["prompt"]
            )
            expected_tokenizer_tokens = int(row["trajectory_rag"]["input_tokens"])
            injected_tokens = int(row["trajectory_rag"]["injected_tokens"])
        else:
            messages = add_tom_to_messages(bench.get_messages(sample))
            expected_tokenizer_tokens = int(row["tom"]["input_tokens"])
            injected_tokens = int(row["tom"]["injected_tokens"])
        if messages_sha256(messages) != row["prompt_sha256"][arm]:
            raise GateFailure(f"Prompt hash changed at {split}:{arm}:{index}")
        completion = client.chat.completions.create(
            model=SERVED_MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0,
        )
        usage = completion.usage
        prompt_tokens = None if usage is None else usage.prompt_tokens
        completion_tokens = None if usage is None else usage.completion_tokens
        expected_provider_prompt_tokens = (
            expected_tokenizer_tokens + usage_surcharges[index]
        )
        if prompt_tokens != expected_provider_prompt_tokens:
            raise GateFailure(
                f"Prompt token count changed at {split}:{arm}:{index}: "
                f"expected_provider={expected_provider_prompt_tokens}, "
                f"observed={prompt_tokens}"
            )
        response = completion.choices[0].message.content or ""
        metrics = bench.score_response(
            scorer,
            2,
            response,
            bench.get_ground_truth(sample),
            index,
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
            "tokenizer_input_tokens": expected_tokenizer_tokens,
            "multimodal_usage_surcharge": usage_surcharges[index],
            "injected_tokens": injected_tokens,
            "completion_tokens": completion_tokens,
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
    if set(records) != set(indices):
        raise GateFailure(f"Incomplete generation coverage for {split} {arm}")
    errors = [record for record in records.values() if record.get("endpoint_error")]
    if errors:
        raise GateFailure(f"{split} {arm} halted with {len(errors)} endpoint errors")
    successes = sum(record["task_score"] == 1 for record in records.values())
    formats = sum(record["format_score"] == 1 for record in records.values())
    format_compliance = formats / len(records)
    summary = {
        "task": "Amendment6_deployment_baseline",
        "status": "PASS" if format_compliance >= 0.9 else "FAIL",
        "backbone": "qwen2_5_vl_72b",
        "split": split,
        "arm": arm,
        "samples": len(records),
        "successes": successes,
        "accuracy": successes / len(records),
        "format_successes": formats,
        "format_compliance": format_compliance,
        "format_gate": {
            "status": "PASS" if format_compliance >= 0.9 else "FAIL",
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


ARM_ORDER = ("zero_shot", "trajectory_rag", "tom", "segment")
PAIR_ORDER = (
    ("zero_shot", "trajectory_rag"),
    ("zero_shot", "tom"),
    ("zero_shot", "segment"),
    ("trajectory_rag", "tom"),
    ("trajectory_rag", "segment"),
    ("tom", "segment"),
)


def paired_comparison(
    records: Mapping[int, Mapping[str, Mapping[str, Any]]],
    left: str,
    right: str,
    seed_offset: int,
) -> Dict[str, Any]:
    indices = sorted(records)
    left_success = np.asarray(
        [records[index][left]["task_score"] == 1 for index in indices], dtype=bool
    )
    right_success = np.asarray(
        [records[index][right]["task_score"] == 1 for index in indices], dtype=bool
    )
    left_only = int((left_success & ~right_success).sum())
    right_only = int((~left_success & right_success).sum())
    differences = right_success.astype(int) - left_success.astype(int)
    probabilities = np.asarray(
        [
            float((differences == -1).mean()),
            float((differences == 0).mean()),
            float((differences == 1).mean()),
        ]
    )
    draws = np.random.default_rng(SEED + seed_offset).multinomial(
        len(indices), probabilities, size=100000
    )
    deltas = (draws[:, 2] - draws[:, 0]) / len(indices)
    return {
        "samples": len(indices),
        "left": left,
        "right": right,
        "left_successes": int(left_success.sum()),
        "right_successes": int(right_success.sum()),
        "left_accuracy": float(left_success.mean()),
        "right_accuracy": float(right_success.mean()),
        "right_minus_left": float(right_success.mean() - left_success.mean()),
        "paired_delta_interval": [
            float(value) for value in np.quantile(deltas, [0.025, 0.975])
        ],
        "left_only": left_only,
        "right_only": right_only,
        "discordant_pairs": left_only + right_only,
        "mcnemar_exact_p": exact_mcnemar_p(left_only, right_only),
    }


def summarize_subset(
    arms: Mapping[int, Mapping[str, Mapping[str, Any]]], seed_offset: int
) -> Dict[str, Any]:
    arm_summaries = {}
    for arm in ARM_ORDER:
        successes = sum(record[arm]["task_score"] == 1 for record in arms.values())
        formats = sum(record[arm]["format_score"] == 1 for record in arms.values())
        arm_summaries[arm] = {
            "samples": len(arms),
            "successes": successes,
            "accuracy": successes / len(arms),
            "format_successes": formats,
            "format_compliance": formats / len(arms),
        }
    comparisons = {
        f"{left}_to_{right}": paired_comparison(
            arms, left, right, seed_offset + position
        )
        for position, (left, right) in enumerate(PAIR_ORDER)
    }
    return {"samples": len(arms), "arms": arm_summaries, "comparisons": comparisons}


def validate_generated_arm(
    split: str, arm: str, indices: Sequence[int]
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    output = OUTPUT_DIR / f"qwen2_5_vl_72b.{split}.{arm}.jsonl"
    run_path = output.with_suffix(output.suffix + ".run.json")
    summary_path = output.with_suffix(".summary.json")
    if any(not path.is_file() for path in (output, run_path, summary_path)):
        raise GateFailure(f"Missing completed {split} {arm} artifact")
    records = load_jsonl(output)
    metadata = json.loads(run_path.read_text())
    summary = json.loads(summary_path.read_text())
    run_hash = fingerprint(metadata)
    if (
        set(records) != set(indices)
        or any(record.get("run_fingerprint") != run_hash for record in records.values())
        or any(record.get("endpoint_error") for record in records.values())
        or summary.get("run_fingerprint") != run_hash
        or summary.get("results_sha256") != file_sha256(output)
    ):
        raise GateFailure(f"Invalid completed {split} {arm} artifact")
    return records, summary


def analyze(split: str) -> Dict[str, Any]:
    require_preregistration()
    dataset, indices, _, source_path, source_summary_path = split_configuration(split)
    source_records, source_run_hash, _ = validate_reused_source(
        split, indices, source_path, source_summary_path
    )
    generated = {
        arm: validate_generated_arm(split, arm, indices)[0]
        for arm in ("trajectory_rag", "tom")
    }
    preflight_path = OUTPUT_DIR / f"qwen2_5_vl_72b.{split}.preflight.jsonl"
    preflight = load_jsonl(preflight_path)
    if set(preflight) != set(indices):
        raise GateFailure(f"Incomplete {split} analysis preflight")
    arms: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    token_rows = []
    for index in indices:
        expected_hashes = preflight[index]["reused_response_sha256"]
        for reused_arm in ("zero_shot", "segment"):
            observed_hash = hashlib.sha256(
                source_records[index]["arms"][reused_arm]["response"].encode()
            ).hexdigest()
            if observed_hash != expected_hashes[reused_arm]:
                raise GateFailure(
                    f"Reused response changed at {split}:{index}:{reused_arm}"
                )
        arms[index] = {
            "zero_shot": source_records[index]["arms"]["zero_shot"],
            "trajectory_rag": generated["trajectory_rag"][index],
            "tom": generated["tom"][index],
            "segment": source_records[index]["arms"]["segment"],
        }
        zero_input = int(arms[index]["zero_shot"]["prompt_tokens"])
        row = {
            "index": index,
            "family": preflight[index]["family"],
            "active_robot_count": preflight[index]["active_robot_count"],
        }
        for arm in ARM_ORDER:
            input_tokens = int(arms[index][arm]["prompt_tokens"])
            completion_tokens = arms[index][arm].get("completion_tokens")
            row.update(
                {
                    f"{arm}_task_score": arms[index][arm]["task_score"],
                    f"{arm}_format_score": arms[index][arm]["format_score"],
                    f"{arm}_input_tokens": input_tokens,
                    f"{arm}_injected_tokens": (
                        0 if arm == "zero_shot" else input_tokens - zero_input
                    ),
                    f"{arm}_generated_tokens": completion_tokens,
                }
            )
        token_rows.append(row)
    primary = summarize_subset(arms, 0 if split == "ood" else 100)
    strata_field = "family" if split == "ood" else "active_robot_count"
    strata: Dict[str, Dict[str, Any]] = {}
    for value in sorted({row[strata_field] for row in token_rows}, key=str):
        subset_indices = [
            row["index"] for row in token_rows if row[strata_field] == value
        ]
        strata[str(value)] = summarize_subset(
            {index: arms[index] for index in subset_indices},
            1000 + len(strata) * 10,
        )
    frame = pd.DataFrame(token_rows).sort_values("index")
    token_path = OUTPUT_DIR / f"qwen2_5_vl_72b.{split}.four_arm_rows.parquet"
    frame.to_parquet(token_path, index=False)
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
    arm_table_path = OUTPUT_DIR / f"qwen2_5_vl_72b.{split}.four_arm_table.csv"
    arm_table.to_csv(arm_table_path, index=False)
    pairwise_table = pd.DataFrame(primary["comparisons"].values())
    pairwise_path = OUTPUT_DIR / f"qwen2_5_vl_72b.{split}.pairwise_table.csv"
    pairwise_table.to_csv(pairwise_path, index=False)
    summary = {
        "task": "Amendment6_four_arm_analysis",
        "status": (
            "PASS"
            if all(
                value["format_compliance"] >= 0.9 for value in primary["arms"].values()
            )
            else "COMPLETE_WITH_FORMAT_GATE_FAILURES"
        ),
        "backbone": "qwen2_5_vl_72b",
        "split": split,
        **primary,
        "breakdown_field": strata_field,
        "breakdown": strata,
        "token_accounting": {
            arm: {
                metric: {
                    "minimum": int(frame[f"{arm}_{metric}_tokens"].min()),
                    "mean": float(frame[f"{arm}_{metric}_tokens"].mean()),
                    "maximum": int(frame[f"{arm}_{metric}_tokens"].max()),
                }
                for metric in ("input", "injected", "generated")
            }
            for arm in ARM_ORDER
        },
        "model_generation_calls": 2 * len(indices),
        "reused_generation_calls": 0,
        "reused_arms": ["zero_shot", "segment"],
        "source_run_fingerprint": source_run_hash,
        "source_results_sha256": file_sha256(source_path),
        "row_results": str(token_path),
        "row_results_sha256": file_sha256(token_path),
        "arm_table": str(arm_table_path),
        "arm_table_sha256": file_sha256(arm_table_path),
        "pairwise_table": str(pairwise_path),
        "pairwise_table_sha256": file_sha256(pairwise_path),
        "preflight_sha256": file_sha256(preflight_path),
        "h0_sha256": file_sha256(H0_PATH),
    }
    if split == "ood":
        segment_vs_rag = primary["comparisons"]["trajectory_rag_to_segment"]
        segment_vs_tom = primary["comparisons"]["tom_to_segment"]
        baseline_beats = [
            comparison["left"]
            for comparison in (segment_vs_rag, segment_vs_tom)
            if comparison["right_minus_left"] < 0
            and comparison["mcnemar_exact_p"] < 0.05
        ]
        segment_beats = all(
            comparison["right_minus_left"] > 0 and comparison["mcnemar_exact_p"] < 0.05
            for comparison in (segment_vs_rag, segment_vs_tom)
        )
        ending = (
            "baseline_beats_segment_OOD"
            if baseline_beats
            else "segment_beats_both_OOD"
            if segment_beats
            else "baseline_matches_segment_OOD"
        )
        summary["ending"] = {
            "selected": ending,
            "text": preregistration()["endings"][ending],
            "baseline_beats_segment": baseline_beats,
        }
    atomic_json(OUTPUT_DIR / f"qwen2_5_vl_72b.{split}.analysis.json", summary)
    return summary


def preregistration() -> Dict[str, Any]:
    return {
        "task": "Amendment6_deployment_baselines",
        "status": "PREREGISTERED",
        "seed": SEED,
        "supersedes_closure_once": True,
        "completed_results_modified": False,
        "per_backbone_tuning": False,
        "h0_path": str(H0_PATH),
        "h0_sha256": file_sha256(H0_PATH),
        "backbones": {
            "required": ["qwen2_5_vl_72b"],
            "optional_after_required_completion": ["gpt_4o_optional"],
            "excluded": {
                "gemini_2_5_flash": "not deployed on the available CloudGPT endpoint"
            },
        },
        "order": [
            "R2_OOD_qwen2_5_vl_72b",
            "T2_OOD_qwen2_5_vl_72b",
            "R2_ID_qwen2_5_vl_72b",
            "T2_ID_qwen2_5_vl_72b",
            "optional_GPT4o_R_and_T",
        ],
        "calls": {
            "R2_OOD_qwen2_5_vl_72b": OOD_ROWS,
            "R2_ID_qwen2_5_vl_72b": ID_ROWS,
            "T2_OOD_qwen2_5_vl_72b": OOD_ROWS,
            "T2_ID_qwen2_5_vl_72b": ID_ROWS,
            "required_total": 3036,
            "optional_GPT4o_R_and_T": 3036,
        },
        "trajectory_rag": {
            "retrieval_unit": "whole valid M0 train source row",
            "pool": "all 6,699 valid M0 source rows",
            "leakage_restriction": "none for deployment",
            "query": "test instruction only",
            "embedding_model": EMBEDDING_MODEL,
            "metric": "normalized cosine",
            "ranking": "descending row-instruction cosine, then source index",
            "threshold": SIMILARITY_THRESHOLD,
            "fallback": "bare prompt when best row cosine is below 0.3",
            "rendering": (
                "each row exemplar is the exact concatenation of that source row's "
                "headerless segment renderings in segment_index order"
            ),
            "budget": (
                "append complete ranked rows until injected total tokens first reach "
                "or exceed the frozen segment arm total input tokens; apply the "
                "Amendment 3.1 extend-then-truncate prefix rule"
            ),
            "per_row_token_tolerance": TOKEN_TOLERANCE,
            "expected_symmetric_drops": 0,
            "rendering_gate_sample_rows": 20,
        },
        "tom": {
            "name": "VIKI-native explicit counterpart/state-reasoning port",
            "retrieved_memory": False,
            "reasoning_template": TOM_REASONING_TEMPLATE,
            "insertion": "append to the official VIKI-L2 system message",
            "output_contract": "unchanged official VIKI-L2 tagged executable plan",
            "token_matching": False,
            "format_exemption": False,
            "mapping": {
                "PartNR_definition": (
                    "explicit partner belief modeling through a dedicated reasoning module"
                ),
                "available_local_implementation": False,
                "available_exact_PartNR_prompt": False,
                "VIKI_multi_robot": (
                    "reason over other active robots, shared visual scene state, "
                    "capabilities, and complementary assignments"
                ),
                "VIKI_single_robot": "explicit capability and scene-state reasoning",
                "limitation": (
                    "prompt-level port, not a reproduction of the absent PartNR ToM module"
                ),
            },
        },
        "generation": {
            "temperature": 0,
            "decoding": "greedy",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "prompt": "official VIKI-L2 plus the frozen arm-specific addition",
            "scorer": "official VIKI-L2",
            "scorer_seed": "original row index",
        },
        "predictions": {
            "R_OOD": [
                "segment beats trajectory RAG",
                "trajectory RAG does not beat zero-shot",
            ],
            "R_ID": "no directional prediction",
            "T_ID": "ToM improves over zero-shot on multi-robot rows if anywhere",
            "T_OOD": "no directional prediction",
            "segment_vs_T_OOD": "segment beats ToM",
        },
        "analysis": {
            "arms": ["zero_shot", "trajectory_rag", "tom", "segment"],
            "statistics": ["exact McNemar", "paired bootstrap 95% interval"],
            "OOD_breakdown": "task family",
            "ID_breakdown": "active robot count derived from the visible prompt",
            "report_per_arm": [
                "task success",
                "format compliance",
                "input tokens",
                "injected tokens",
                "generated tokens",
            ],
        },
        "endings": {
            "segment_beats_both_OOD": (
                "deployment claim complete: segment beats trajectory RAG and ToM"
            ),
            "baseline_matches_segment_OOD": (
                "granularity claim narrows to the strict productivity channel"
            ),
            "baseline_beats_segment_OOD": (
                "baseline superiority is the headline for the deployment column"
            ),
        },
        "C4": "blocked on original PartNR memory banks and generation logs",
    }


def freeze() -> Dict[str, Any]:
    if not H0_PATH.is_file():
        raise GateFailure("Amendment 6 requires the frozen Amendment 5 H0")
    value = preregistration()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if PREREGISTRATION_PATH.is_file():
        observed = json.loads(PREREGISTRATION_PATH.read_text())
        if observed != value:
            raise GateFailure("Amendment 6 preregistration already differs")
    atomic_json(PREREGISTRATION_PATH, value)
    prior_closure = (
        ROOT / "results/viki_memory_experiments/amendment4/VIKI_GENERATION_CLOSED.json"
    )
    prior_campaign = AMENDMENT5_1_DIR / "final_summary.json"
    if not prior_closure.is_file() or not prior_campaign.is_file():
        raise GateFailure("Amendment 6 supersession inputs are missing")
    supersession = {
        "task": "Amendment6_closure_supersession",
        "status": "ACTIVE_ONCE",
        "scope_complete": True,
        "completed_results_modified": False,
        "supersedes": str(prior_closure),
        "superseded_sha256": file_sha256(prior_closure),
        "prior_campaign_summary": str(prior_campaign),
        "prior_campaign_summary_sha256": file_sha256(prior_campaign),
        "preregistration": str(PREREGISTRATION_PATH),
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
        "authorized_required_calls": 3036,
        "authorized_optional_calls": 3036,
        "optional_condition": "only after all required Qwen2.5-VL-72B arms complete",
        "authorized_arms": [
            "Qwen2.5-VL-72B trajectory RAG OOD and ID",
            "Qwen2.5-VL-72B ToM OOD and ID",
            "optional GPT-4o trajectory RAG OOD and ID",
            "optional GPT-4o ToM OOD and ID",
        ],
        "excluded": ["Gemini-2.5-Flash"],
        "after_scope": "select preregistered ending and close VIKI generation",
    }
    if (
        SUPERSESSION_PATH.is_file()
        and json.loads(SUPERSESSION_PATH.read_text()) != supersession
    ):
        raise GateFailure("Amendment 6 closure supersession already differs")
    atomic_json(SUPERSESSION_PATH, supersession)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VIKI Amendment 6")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("build-row-cache")
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--split", choices=("ood", "id"), required=True)
    preflight_parser.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--split", choices=("ood", "id"), required=True)
    run_parser.add_argument("--arm", choices=("trajectory_rag", "tom"), required=True)
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    run_parser.add_argument("--workers", type=int, default=8)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--split", choices=("ood", "id"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "freeze":
            result = freeze()
        elif args.command == "build-row-cache":
            result = build_row_cache()
        elif args.command == "run":
            result = run_arm(args.split, args.arm, args.base_url, args.workers)
        elif args.command == "analyze":
            result = analyze(args.split)
        else:
            result = preflight(args.split, args.base_url)
    except GateFailure as error:
        print(f"GATE FAILURE: {error}")
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
