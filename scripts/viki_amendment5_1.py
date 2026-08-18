#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from viki_amendment5 import (
    BOOTSTRAP_DRAWS,
    EMBEDDING_MODEL,
    H0_PATH,
    M0_INSTANCES_PATH,
    M0_SKILLS_PATH,
    M1_CACHE_PATH,
    MAX_OUTPUT_TOKENS,
    MIN_CONTEXT_LENGTH,
    SIMILARITY_THRESHOLD,
    TOP_K,
    GateFailure,
    atomic_json,
    file_sha256,
    fingerprint,
    format_deployment_prompt,
    load_bench,
    load_jsonl,
    messages_sha256,
    native,
    repair_qwen3_response,
    require_h0,
    server_metadata,
    token_count,
    write_jsonl_snapshot,
)

from habitat_llm.evaluation.viki_branch_conditions import get_instruction
from habitat_llm.evaluation.viki_memory_skill import add_memory_to_messages
from habitat_llm.evaluation.viki_segment_memory import SegmentInstance

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT.parent / "VIKI-R"
DATA_ROOT = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2"
AMENDMENT1_DIR = ROOT / "results/viki_memory_experiments/amendment1"
AMENDMENT2_DIR = ROOT / "results/viki_memory_experiments/amendment2"
AMENDMENT3_DIR = ROOT / "results/viki_memory_experiments/amendment3"
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment5_1"
ID_MANIFEST_PATH = AMENDMENT1_DIR / "a5_id_safety_manifest.parquet"
PREREGISTRATION_PATH = OUTPUT_DIR / "preregistration.json"
DEPLOYMENT_MANIFEST_PATH = OUTPUT_DIR / "id_deployment_manifest.jsonl"
DEPLOYMENT_MANIFEST_SUMMARY_PATH = OUTPUT_DIR / "id_deployment_manifest.summary.json"
ROWS = 300
SEED = 20260814

SOURCES = {
    "qwen2_5_vl_72b": AMENDMENT3_DIR / "f2_id.jsonl",
    "qwen2_5_vl_7b_stock": AMENDMENT2_DIR / "b1_id.jsonl",
    "rl_7b": ROOT / "results/viki_official_7b_l2_id.jsonl",
}

BACKBONES: Dict[str, Dict[str, Any]] = {
    "qwen2_5_vl_72b": {
        "label": "Qwen2.5-VL-72B",
        "model_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "served_model": "qwen2.5-vl-72b-amendment3-f2",
        "vllm_version": "0.11.2",
        "source_arm": "zero_shot",
        "source_expected_successes": 22,
        "source_expected_formats": 300,
        "source_token_field": "zero_shot_input_tokens",
        "enforce_source_token_match": True,
        "calls": 300,
    },
    "qwen2_5_vl_7b_stock": {
        "label": "Qwen2.5-VL-7B stock",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "served_model": "qwen2.5-vl-7b-amendment5",
        "vllm_version": "0.11.2",
        "source_arm": "base",
        "source_expected_successes": 1,
        "source_expected_formats": 299,
        "source_token_field": None,
        "enforce_source_token_match": False,
        "calls": 300,
    },
    "rl_7b": {
        "label": "VIKI-R RL 7B",
        "model_id": str(
            (BENCHMARK_ROOT / "models/Qwen2.5VL-7B-Instruct-VIKI-R-2").resolve()
        ),
        "served_model": "viki-r-7b-l2-amendment1",
        "vllm_version": "0.8.4",
        "source_arm": None,
        "source_expected_successes": 278,
        "source_expected_formats": 300,
        "source_token_field": None,
        "enforce_source_token_match": False,
        "calls": 300,
    },
    "qwen3_vl_30b_a3b": {
        "label": "Qwen3-VL-30B-A3B",
        "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "served_model": "qwen3-vl-30b-a3b-amendment5",
        "vllm_version": "0.11.2",
        "source_arm": None,
        "source_expected_successes": None,
        "source_expected_formats": None,
        "source_token_field": None,
        "enforce_source_token_match": False,
        "calls": 600,
        "official_format_gate_exempt": True,
    },
    "gpt_4o_optional": {
        "label": "GPT-4o (optional)",
        "model_id": "gpt-4o-2024-08-06",
        "served_model": "gpt-4o-2024-08-06",
        "request_model": "gpt-4o-20240806",
        "provider": "cloudgpt_azure_cli",
        "context_length": 128000,
        "source_arm": None,
        "source_expected_successes": None,
        "source_expected_formats": None,
        "source_token_field": None,
        "enforce_source_token_match": False,
        "calls": 600,
    },
}


def preregistration() -> Dict[str, Any]:
    return {
        "task": "Amendment5.1_ID_deployment_column",
        "status": "PREREGISTERED",
        "seed": SEED,
        "rows": ROWS,
        "slice": str(ID_MANIFEST_PATH),
        "h0_path": str(H0_PATH),
        "h0_sha256": file_sha256(H0_PATH),
        "h0_configuration_sha256": fingerprint(json.loads(H0_PATH.read_text())),
        "required_order": [
            "qwen2_5_vl_72b",
            "qwen2_5_vl_7b_stock",
            "rl_7b_smoke_then_segment",
        ],
        "optional_order": ["qwen3_vl_30b_a3b"],
        "required_new_calls": 900,
        "optional_new_calls": 600,
        "configuration": json.loads(H0_PATH.read_text())["configuration"],
        "predictions": {
            "qwen2_5_vl_72b": "segment positive; directional bet exceeds composed +3.00pp",
            "qwen2_5_vl_7b_stock": "floor; no meaningful delta",
            "rl_7b": "segment negative; magnitude not bet",
            "qwen3_vl_30b_a3b": "floor and not significant",
        },
        "table_footnote": (
            "ID-with-memory measures deployment effect, not compositional ability, "
            "since the bank contains near-duplicate in-distribution episodes, and "
            "attribution remains with the C-prime controls."
        ),
        "sources": {name: str(path) for name, path in SOURCES.items()},
    }


def freeze() -> Dict[str, Any]:
    require_h0()
    required = [ID_MANIFEST_PATH, *SOURCES.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise GateFailure("Missing Amendment 5.1 inputs: " + ", ".join(missing))
    value = preregistration()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if (
        PREREGISTRATION_PATH.is_file()
        and json.loads(PREREGISTRATION_PATH.read_text()) != value
    ):
        raise GateFailure("Amendment 5.1 preregistration already differs")
    atomic_json(PREREGISTRATION_PATH, value)
    return value


def require_preregistration() -> Dict[str, Any]:
    require_h0()
    if not PREREGISTRATION_PATH.is_file():
        raise GateFailure("Freeze Amendment 5.1 before any deployment operation")
    observed = json.loads(PREREGISTRATION_PATH.read_text())
    if observed != preregistration():
        raise GateFailure("Amendment 5.1 preregistration or H0 identity changed")
    return observed


def load_instances() -> tuple[List[SegmentInstance], np.ndarray]:
    raw_instances = [
        json.loads(line)
        for line in M0_INSTANCES_PATH.read_text().splitlines()
        if line.strip()
    ]
    instances = [SegmentInstance.from_dict(value) for value in raw_instances]
    if (
        len(instances) != 19499
        or len({item.instance_id for item in instances}) != 19499
    ):
        raise GateFailure("H0 full M0 bank is not exactly 19,499 unique segments")
    cache = np.load(M1_CACHE_PATH, allow_pickle=False)
    instance_ids = [item.instance_id for item in instances]
    skill_names = [
        str(value["name"]) for value in json.loads(M0_SKILLS_PATH.read_text())["skills"]
    ]
    digest = hashlib.sha256()
    for value in skill_names + instance_ids:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    if str(cache["cache_key"].item()) != digest.hexdigest():
        raise GateFailure("H0 embedding cache fingerprint mismatch")
    embeddings = np.asarray(cache["context_embeddings"])
    if embeddings.shape != (19499, 768):
        raise GateFailure(f"Unexpected H0 embedding shape: {embeddings.shape}")
    return instances, embeddings


def build_manifest() -> Dict[str, Any]:
    prereg = require_preregistration()
    frozen_slice = pd.read_parquet(ID_MANIFEST_PATH)
    indices = [int(value) for value in frozen_slice["index"]]
    if len(indices) != ROWS or len(set(indices)) != ROWS:
        raise GateFailure("Frozen A5 manifest is not exactly 300 unique indices")
    dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    bench = load_bench()
    instances, context_embeddings = load_instances()
    instance_ids = [item.instance_id for item in instances]
    instructions = [
        get_instruction(native(dataset.iloc[index].to_dict())) for index in indices
    ]
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    query_embeddings = np.asarray(
        embedding_model.encode(
            instructions,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )
    records: Dict[int, Dict[str, Any]] = {}
    for position, index in enumerate(indices):
        similarities = context_embeddings @ query_embeddings[position]
        top_positions = np.argpartition(similarities, -TOP_K)[-TOP_K:]
        ranked = sorted(
            (
                (float(similarities[item]), instance_ids[item], item)
                for item in top_positions
            ),
            key=lambda item: (-item[0], item[1]),
        )
        selected = ranked if ranked[0][0] >= SIMILARITY_THRESHOLD else []
        selected_instances = [instances[item] for _, _, item in selected]
        prompt = format_deployment_prompt(selected_instances)
        sample = native(dataset.iloc[index].to_dict())
        zero_messages = bench.get_messages(sample)
        segment_messages = add_memory_to_messages(bench.get_messages(sample), prompt)
        records[index] = {
            "index": index,
            "instruction": instructions[position],
            "best_similarity": ranked[0][0],
            "fallback_bare": not selected,
            "selected_instance_ids": [item.instance_id for item in selected_instances],
            "selected_similarities": [similarity for similarity, _, _ in selected],
            "segment_prompt": prompt,
            "prompt_sha256": {
                "zero_shot": messages_sha256(zero_messages),
                "segment": messages_sha256(segment_messages),
            },
        }
    write_jsonl_snapshot(DEPLOYMENT_MANIFEST_PATH, records, indices)
    source_hashes = {name: file_sha256(path) for name, path in SOURCES.items()}
    summary = {
        "task": "Amendment5.1_ID_deployment_manifest",
        "status": "PASS",
        "rows": len(records),
        "indices": indices,
        "indices_sha256": fingerprint({"indices": indices}),
        "frozen_slice_sha256": file_sha256(ID_MANIFEST_PATH),
        "bank_instances": len(instances),
        "top_k": TOP_K,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "fallback_rows": sum(record["fallback_bare"] for record in records.values()),
        "source_results_sha256": source_hashes,
        "results": str(DEPLOYMENT_MANIFEST_PATH),
        "results_sha256": file_sha256(DEPLOYMENT_MANIFEST_PATH),
        "h0_sha256": prereg["h0_sha256"],
        "h0_configuration_sha256": prereg["h0_configuration_sha256"],
    }
    atomic_json(DEPLOYMENT_MANIFEST_SUMMARY_PATH, summary)
    return summary


def load_deployment_manifest() -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    prereg = require_preregistration()
    if not DEPLOYMENT_MANIFEST_SUMMARY_PATH.is_file():
        raise GateFailure("Build the Amendment 5.1 ID manifest first")
    summary = json.loads(DEPLOYMENT_MANIFEST_SUMMARY_PATH.read_text())
    indices = [int(value) for value in summary.get("indices", [])]
    if (
        summary.get("status") != "PASS"
        or len(indices) != ROWS
        or len(set(indices)) != ROWS
        or summary.get("frozen_slice_sha256") != file_sha256(ID_MANIFEST_PATH)
        or summary.get("results_sha256") != file_sha256(DEPLOYMENT_MANIFEST_PATH)
        or summary.get("h0_sha256") != prereg["h0_sha256"]
        or summary.get("h0_configuration_sha256") != prereg["h0_configuration_sha256"]
    ):
        raise GateFailure("Amendment 5.1 ID manifest certificate is invalid")
    frozen_indices = [
        int(value) for value in pd.read_parquet(ID_MANIFEST_PATH)["index"]
    ]
    if indices != frozen_indices:
        raise GateFailure("ID deployment indices differ from the frozen A5 manifest")
    records = load_jsonl(DEPLOYMENT_MANIFEST_PATH)
    if set(records) != set(indices):
        raise GateFailure("ID deployment manifest row coverage is incomplete")
    return indices, records


def source_records(
    backbone: str, indices: Sequence[int]
) -> Tuple[Dict[int, Dict[str, Any]], str, str]:
    if backbone in {"qwen3_vl_30b_a3b", "gpt_4o_optional"}:
        return {}, "", ""
    path = SOURCES[backbone]
    all_records = load_jsonl(path)
    if not set(indices).issubset(all_records):
        raise GateFailure(f"Certified source coverage is incomplete: {backbone}")
    selected = {index: all_records[index] for index in indices}
    config = BACKBONES[backbone]
    if backbone == "rl_7b":
        arms = selected
        if any(
            record.get("level") != 2
            or record.get("split") != "test"
            or record.get("error")
            for record in selected.values()
        ):
            raise GateFailure("RL certified baseline metadata is invalid")
        source_run_hash = fingerprint(
            {
                "source_results_sha256": file_sha256(path),
                "selected_indices": list(indices),
            }
        )
    else:
        run_path = path.with_suffix(path.suffix + ".run.json")
        metadata = json.loads(run_path.read_text())
        source_run_hash = fingerprint(metadata)
        if metadata.get("indices") != list(indices) or any(
            record.get("run_fingerprint") != source_run_hash
            for record in selected.values()
        ):
            raise GateFailure(f"Certified source run binding failed: {backbone}")
        source_arm = str(config["source_arm"])
        arms = {index: record["arms"][source_arm] for index, record in selected.items()}
    successes = sum(record["task_score"] == 1 for record in arms.values())
    formats = sum(record["format_score"] == 1 for record in arms.values())
    if (
        successes != config["source_expected_successes"]
        or formats != config["source_expected_formats"]
    ):
        raise GateFailure(
            f"Certified source metrics changed for {backbone}: "
            f"successes={successes}, formats={formats}"
        )
    return arms, source_run_hash, file_sha256(path)


def validate_service(backbone: str, base_url: str) -> Dict[str, Any]:
    config = BACKBONES[backbone]
    if config.get("provider") == "cloudgpt_azure_cli":
        expected_url = "https://cloudgpt-openai.azure-api.net/"
        if os.environ.get("CLOUDGPT_USE_AZURE_CLI") != "1":
            raise GateFailure("Set CLOUDGPT_USE_AZURE_CLI=1 for CloudGPT")
        if base_url.rstrip("/") != expected_url.rstrip("/"):
            raise GateFailure(f"Unexpected CloudGPT endpoint: {base_url}")
        return {
            "base_url": expected_url,
            "provider": "CloudGPT AzureOpenAI",
            "authentication": "Azure CLI cached token",
            "api_version": "2024-06-01",
            "models": [
                {
                    "id": config["request_model"],
                    "root": config["model_id"],
                    "max_model_len": config["context_length"],
                }
            ],
        }
    runtime = server_metadata([base_url])[0]
    if runtime.get("version", {}).get("version") != config["vllm_version"]:
        raise GateFailure(f"Unexpected serving stack for {backbone}: {runtime}")
    models = runtime.get("models", [])
    if len(models) != 1:
        raise GateFailure(f"Unexpected model inventory for {backbone}: {models}")
    model = models[0]
    observed_root = str(model.get("root"))
    expected_root = str(config["model_id"])
    if backbone == "rl_7b":
        observed_root = str(Path(observed_root).resolve())
    if (
        model.get("id") != config["served_model"]
        or observed_root != expected_root
        or int(model.get("max_model_len") or 0) < MIN_CONTEXT_LENGTH
    ):
        raise GateFailure(
            f"H0 service mismatch for {backbone}: observed={model}, "
            f"expected_model={expected_root}, min_context={MIN_CONTEXT_LENGTH}"
        )
    return {"base_url": base_url, **runtime}


def preflight(backbone: str, base_url: str) -> Dict[str, Any]:
    require_preregistration()
    config = BACKBONES[backbone]
    runtime = validate_service(backbone, base_url)
    indices, manifest = load_deployment_manifest()
    sources, source_run_hash, source_results_sha256 = source_records(backbone, indices)
    bench = load_bench()
    dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    rows = []
    for index in indices:
        sample = native(dataset.iloc[index].to_dict())
        zero_messages = bench.get_messages(sample)
        segment_messages = add_memory_to_messages(
            bench.get_messages(sample), manifest[index]["segment_prompt"]
        )
        zero_hash = messages_sha256(zero_messages)
        segment_hash = messages_sha256(segment_messages)
        if (
            zero_hash != manifest[index]["prompt_sha256"]["zero_shot"]
            or segment_hash != manifest[index]["prompt_sha256"]["segment"]
        ):
            raise GateFailure(f"ID deployment prompt changed at index {index}")
        if config.get("provider") == "cloudgpt_azure_cli":
            framing_allowance = 1024
            zero_tokens = (
                len(json.dumps(zero_messages, ensure_ascii=False).encode("utf-8"))
                + framing_allowance
            )
            segment_tokens = (
                len(json.dumps(segment_messages, ensure_ascii=False).encode("utf-8"))
                + framing_allowance
            )
        else:
            zero_tokens = token_count(base_url, config["served_model"], zero_messages)
            segment_tokens = token_count(
                base_url, config["served_model"], segment_messages
            )
        if max(zero_tokens, segment_tokens) + MAX_OUTPUT_TOKENS > int(
            runtime["models"][0]["max_model_len"]
        ):
            raise GateFailure(f"H0 context gate failed at {backbone}:{index}")
        source_tokens = None
        if index in sources:
            source_tokens = (
                sources[index].get("prompt_tokens")
                if config["source_token_field"] is None
                else load_jsonl(SOURCES[backbone])[index].get(
                    config["source_token_field"]
                )
            )
        if config["enforce_source_token_match"] and int(source_tokens) != zero_tokens:
            raise GateFailure(
                f"Reused zero-shot token mismatch at {backbone}:{index}: "
                f"source={source_tokens}, current={zero_tokens}"
            )
        rows.append(
            {
                "index": index,
                "zero_shot_prompt_sha256": zero_hash,
                "segment_prompt_sha256": segment_hash,
                "source_response_sha256": (
                    hashlib.sha256(
                        sources[index]["response"].encode("utf-8")
                    ).hexdigest()
                    if index in sources
                    else None
                ),
                "zero_shot_input_tokens": zero_tokens,
                "segment_input_tokens": segment_tokens,
                "source_input_tokens": source_tokens,
            }
        )
    frame = pd.DataFrame(rows)
    output = OUTPUT_DIR / f"{backbone}.preflight.parquet"
    frame.to_parquet(output, index=False)
    summary = {
        "task": (
            "Amendment5.1_CloudGPT_preflight"
            if config.get("provider") == "cloudgpt_azure_cli"
            else "Amendment5.1_local_preflight"
        ),
        "backbone": backbone,
        "status": "PASS",
        "rows": len(frame),
        "model_generation_calls": 0,
        "runtime": runtime,
        "maximum_zero_shot_input_tokens": int(frame["zero_shot_input_tokens"].max()),
        "maximum_segment_input_tokens": int(frame["segment_input_tokens"].max()),
        "context_limit": int(runtime["models"][0]["max_model_len"]),
        "minimum_headroom": int(
            runtime["models"][0]["max_model_len"]
            - MAX_OUTPUT_TOKENS
            - frame[["zero_shot_input_tokens", "segment_input_tokens"]]
            .max(axis=1)
            .max()
        ),
        "input_token_accounting": (
            "conservative upper bound: serialized UTF-8 bytes plus 1,024 tokens "
            "for provider framing"
            if config.get("provider") == "cloudgpt_azure_cli"
            else "serving-stack tokenizer"
        ),
        "source_run_fingerprint": source_run_hash,
        "source_results_sha256": source_results_sha256,
        "source_prompt_hash_rule": (
            "both arms reconstructed from the frozen ID deployment manifest"
            if config.get("provider") == "cloudgpt_azure_cli"
            else (
                "official test prompt reconstructed from the certified source split "
                "and bound to the frozen source artifact"
            )
        ),
        "source_prompt_provenance": {
            "official_bench_sha256": file_sha256(
                ROOT / "habitat_llm/evaluation/viki_bench.py"
            ),
            "test_parquet_sha256": file_sha256(DATA_ROOT / "test.parquet"),
            "source_runner_sha256": (
                file_sha256(ROOT / "scripts/viki_amendment2_pipeline.py")
                if backbone == "qwen2_5_vl_7b_stock"
                else file_sha256(ROOT / "scripts/viki_amendment3_f2.py")
                if backbone == "qwen2_5_vl_72b"
                else file_sha256(ROOT / "scripts/viki_amendment5.py")
                if backbone in {"qwen3_vl_30b_a3b", "gpt_4o_optional"}
                else file_sha256(ROOT / "habitat_llm/evaluation/viki_bench.py")
            ),
            "legacy_token_accounting_match_required": bool(
                config["enforce_source_token_match"]
            ),
        },
        "results": str(output),
        "results_sha256": file_sha256(output),
        "manifest_sha256": file_sha256(DEPLOYMENT_MANIFEST_PATH),
        "h0_sha256": file_sha256(H0_PATH),
    }
    atomic_json(OUTPUT_DIR / f"{backbone}.preflight.summary.json", summary)
    return summary


def rl_smoke(base_url: str) -> Dict[str, Any]:
    backbone = "rl_7b"
    preflight_path = OUTPUT_DIR / f"{backbone}.preflight.summary.json"
    if not preflight_path.is_file():
        raise GateFailure("Run the RL preflight before its smoke")
    preflight_summary = json.loads(preflight_path.read_text())
    runtime = validate_service(backbone, base_url)
    if preflight_summary.get("status") != "PASS" or runtime != preflight_summary.get(
        "runtime"
    ):
        raise GateFailure("RL runtime changed after preflight")
    indices, manifest = load_deployment_manifest()
    sources, source_run_hash, source_results_sha256 = source_records(backbone, indices)
    smoke_indices = sorted(random.Random(SEED).sample(indices, 20))
    bench = load_bench()
    dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    client = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=5, timeout=3600)
    records: Dict[int, Dict[str, Any]] = {}
    for index in smoke_indices:
        messages = bench.get_messages(native(dataset.iloc[index].to_dict()))
        if messages_sha256(messages) != manifest[index]["prompt_sha256"]["zero_shot"]:
            raise GateFailure(f"RL smoke prompt changed at index {index}")
        completion = client.chat.completions.create(
            model=BACKBONES[backbone]["served_model"],
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0,
        )
        observed = completion.choices[0].message.content or ""
        expected = sources[index]["response"]
        identical = observed == expected
        records[index] = {
            "index": index,
            "prompt_sha256": manifest[index]["prompt_sha256"]["zero_shot"],
            "source_response_sha256": hashlib.sha256(
                expected.encode("utf-8")
            ).hexdigest(),
            "observed_response_sha256": hashlib.sha256(
                observed.encode("utf-8")
            ).hexdigest(),
            "byte_identical": identical,
        }
        if not identical:
            write_jsonl_snapshot(
                OUTPUT_DIR / "rl_7b.smoke.jsonl", records, smoke_indices
            )
            raise GateFailure(f"RL raised-context smoke mismatch at index {index}")
    output = OUTPUT_DIR / "rl_7b.smoke.jsonl"
    write_jsonl_snapshot(output, records, smoke_indices)
    summary = {
        "task": "Amendment5.1_RL_raised_context_smoke",
        "backbone": backbone,
        "status": "PASS",
        "selection_rule": "sorted random.Random(20260814).sample(frozen_indices, 20)",
        "indices": smoke_indices,
        "samples": len(records),
        "byte_identical": sum(record["byte_identical"] for record in records.values()),
        "model_generation_calls": len(records),
        "sanctioned_deviation": "max_model_len raised to 16384",
        "runtime": runtime,
        "source_run_fingerprint": source_run_hash,
        "source_results_sha256": source_results_sha256,
        "results": str(output),
        "results_sha256": file_sha256(output),
    }
    atomic_json(OUTPUT_DIR / "rl_7b.smoke.summary.json", summary)
    return summary


def certify_rl_smoke_failure() -> Dict[str, Any]:
    require_preregistration()
    smoke_path = OUTPUT_DIR / "rl_7b.smoke.jsonl"
    preflight_path = OUTPUT_DIR / "rl_7b.preflight.summary.json"
    if not smoke_path.is_file() or not preflight_path.is_file():
        raise GateFailure("RL smoke failure certificate requires saved smoke/preflight")
    records = load_jsonl(smoke_path)
    indices, _ = load_deployment_manifest()
    expected_indices = sorted(random.Random(SEED).sample(indices, 20))
    tested_indices = [index for index in expected_indices if index in records]
    if not tested_indices:
        raise GateFailure("RL smoke failure certificate has no tested row")
    first_mismatch = next(
        (
            index
            for index in tested_indices
            if records[index].get("byte_identical") is False
        ),
        None,
    )
    if first_mismatch is None:
        raise GateFailure("RL smoke artifact does not contain a mismatch")
    mismatch = records[first_mismatch]
    sources, source_run_hash, source_results_sha256 = source_records("rl_7b", indices)
    expected_source_sha = hashlib.sha256(
        sources[first_mismatch]["response"].encode("utf-8")
    ).hexdigest()
    if mismatch.get("source_response_sha256") != expected_source_sha or mismatch.get(
        "source_response_sha256"
    ) == mismatch.get("observed_response_sha256"):
        raise GateFailure("RL smoke mismatch artifact failed source/hash validation")
    preflight_summary = json.loads(preflight_path.read_text())
    summary = {
        "task": "Amendment5.1_RL_raised_context_smoke",
        "backbone": "rl_7b",
        "status": "FAIL",
        "selection_rule": "sorted random.Random(20260814).sample(frozen_indices, 20)",
        "planned_indices": expected_indices,
        "tested_indices": tested_indices,
        "samples_planned": 20,
        "samples_tested": len(tested_indices),
        "byte_identical": sum(
            record.get("byte_identical") is True for record in records.values()
        ),
        "first_mismatch_index": first_mismatch,
        "source_response_sha256": mismatch["source_response_sha256"],
        "observed_response_sha256": mismatch["observed_response_sha256"],
        "model_generation_calls": len(tested_indices),
        "sanctioned_deviation": "max_model_len raised to 16384",
        "failure_action": "halt rl_7b segment arm; no segment calls authorized",
        "runtime": preflight_summary["runtime"],
        "source_run_fingerprint": source_run_hash,
        "source_results_sha256": source_results_sha256,
        "results": str(smoke_path),
        "results_sha256": file_sha256(smoke_path),
    }
    atomic_json(OUTPUT_DIR / "rl_7b.smoke.summary.json", summary)
    return summary


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    return min(
        1.0,
        2
        * sum(math.comb(discordant, value) for value in range(tail + 1))
        / (2**discordant),
    )


def paired_summary(
    records: Mapping[int, Mapping[str, Any]], seed_offset: int
) -> Dict[str, Any]:
    zero = {
        index: records[index]["arms"]["zero_shot"]["task_score"] == 1
        for index in records
    }
    segment = {
        index: records[index]["arms"]["segment"]["task_score"] == 1 for index in records
    }
    left_only = sum(zero[index] and not segment[index] for index in records)
    right_only = sum(not zero[index] and segment[index] for index in records)
    differences = np.asarray(
        [int(segment[index]) - int(zero[index]) for index in sorted(records)]
    )
    probabilities = np.asarray(
        [
            float((differences == -1).mean()),
            float((differences == 0).mean()),
            float((differences == 1).mean()),
        ]
    )
    draws = np.random.default_rng(SEED + seed_offset).multinomial(
        len(differences), probabilities, size=BOOTSTRAP_DRAWS
    )
    deltas = (draws[:, 2] - draws[:, 0]) / len(differences)
    return {
        "samples": len(records),
        "zero_shot_successes": sum(zero.values()),
        "segment_successes": sum(segment.values()),
        "zero_shot_accuracy": sum(zero.values()) / len(records),
        "segment_accuracy": sum(segment.values()) / len(records),
        "segment_minus_zero_shot": (sum(segment.values()) - sum(zero.values()))
        / len(records),
        "paired_delta_interval": [
            float(value) for value in np.quantile(deltas, [0.025, 0.975])
        ],
        "left_only": left_only,
        "right_only": right_only,
        "discordant_pairs": left_only + right_only,
        "mcnemar_exact_p": exact_mcnemar_p(left_only, right_only),
    }


def analyze(backbone: str, records: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    order = list(BACKBONES).index(backbone) + 1
    comparison = paired_summary(records, order)
    arms = {}
    for arm in ("zero_shot", "segment"):
        successes = sum(
            record["arms"][arm]["task_score"] == 1 for record in records.values()
        )
        formats = sum(
            record["arms"][arm]["format_score"] == 1 for record in records.values()
        )
        arms[arm] = {
            "successes": successes,
            "accuracy": successes / len(records),
            "format_successes": formats,
            "format_compliance": formats / len(records),
        }
    return {
        "task": "Amendment5.1_ID_deployment",
        "backbone": backbone,
        "samples": len(records),
        "arms": arms,
        "comparisons": {"zero_shot_to_segment": comparison},
    }


def run(backbone: str, base_url: str, workers: int) -> Dict[str, Any]:
    config = BACKBONES[backbone]
    preflight_path = OUTPUT_DIR / f"{backbone}.preflight.summary.json"
    if not preflight_path.is_file():
        raise GateFailure(f"Run preflight before generation: {backbone}")
    preflight_summary = json.loads(preflight_path.read_text())
    runtime = validate_service(backbone, base_url)
    if (
        preflight_summary.get("status") != "PASS"
        or preflight_summary.get("manifest_sha256")
        != file_sha256(DEPLOYMENT_MANIFEST_PATH)
        or preflight_summary.get("h0_sha256") != file_sha256(H0_PATH)
        or preflight_summary.get("runtime") != runtime
    ):
        raise GateFailure(f"Preflight certificate is invalid: {backbone}")
    if backbone == "rl_7b":
        smoke_path = OUTPUT_DIR / "rl_7b.smoke.summary.json"
        if not smoke_path.is_file():
            raise GateFailure("RL segment run requires a passing 20-row smoke")
        smoke = json.loads(smoke_path.read_text())
        if (
            smoke.get("status") != "PASS"
            or smoke.get("samples") != 20
            or smoke.get("byte_identical") != 20
            or smoke.get("runtime") != runtime
        ):
            raise GateFailure("RL raised-context smoke certificate is invalid")
    indices, manifest = load_deployment_manifest()
    sources, source_run_hash, source_results_sha256 = source_records(backbone, indices)
    generated_pair = backbone in {"qwen3_vl_30b_a3b", "gpt_4o_optional"}
    metadata = {
        "task": "Amendment5.1_ID_deployment",
        "backbone": backbone,
        "model_id": config["model_id"],
        "served_model": config["served_model"],
        "runtime": runtime,
        "indices": indices,
        "indices_sha256": fingerprint({"indices": indices}),
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "workers": workers,
        "scorer_seed_rule": "original row index",
        "h0_sha256": file_sha256(H0_PATH),
        "manifest_sha256": file_sha256(DEPLOYMENT_MANIFEST_PATH),
        "preflight_sha256": file_sha256(preflight_path),
        "source_run_fingerprint": source_run_hash,
        "source_results_sha256": source_results_sha256,
        "new_arms": ["zero_shot", "segment"] if generated_pair else ["segment"],
        "reused_arms": [] if generated_pair else ["zero_shot"],
        "expected_new_calls": int(config["calls"]),
    }
    if config.get("provider") == "cloudgpt_azure_cli":
        metadata.update(
            {
                "request_model": config["request_model"],
                "transport": "CloudGPT AzureOpenAI with Azure CLI cached authentication",
            }
        )
    run_hash = fingerprint(metadata)
    output = OUTPUT_DIR / f"{backbone}.jsonl"
    run_path = output.with_suffix(output.suffix + ".run.json")
    if run_path.is_file():
        if json.loads(run_path.read_text()) != metadata:
            raise GateFailure(f"Cannot resume {backbone}: run metadata differs")
    else:
        atomic_json(run_path, metadata)
    records = load_jsonl(output)
    if any(
        index not in indices or record.get("run_fingerprint") != run_hash
        for index, record in records.items()
    ):
        raise GateFailure(f"Invalid checkpoint record: {backbone}")
    pending = [
        index
        for index in indices
        if index not in records or records[index].get("endpoint_error")
    ]
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    if config.get("provider") == "cloudgpt_azure_cli":
        from cloudgpt_aoai import get_openai_client

        client = get_openai_client(timeout=3600, max_retries=5)
        request_model = config["request_model"]
    else:
        client = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=5, timeout=3600)
        request_model = config["served_model"]

    def generate(index: int) -> Tuple[int, Dict[str, Any]]:
        sample = native(dataset.iloc[index].to_dict())
        zero_messages = bench.get_messages(sample)
        segment_messages = add_memory_to_messages(
            bench.get_messages(sample), manifest[index]["segment_prompt"]
        )
        generated_arms: Dict[str, Dict[str, Any]] = {}
        arm_messages = {"zero_shot": zero_messages, "segment": segment_messages}
        for arm in ("zero_shot", "segment") if generated_pair else ("segment",):
            completion = client.chat.completions.create(
                model=request_model,
                messages=arm_messages[arm],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0,
            )
            generated = {
                "response": completion.choices[0].message.content or "",
                "prompt_tokens": (
                    None if completion.usage is None else completion.usage.prompt_tokens
                ),
                "completion_tokens": (
                    None
                    if completion.usage is None
                    else completion.usage.completion_tokens
                ),
            }
            metrics = bench.score_response(
                scorer,
                2,
                generated["response"],
                bench.get_ground_truth(sample),
                index,
            )
            generated_arms[arm] = {**generated, **metrics}
        arms = (
            generated_arms
            if generated_pair
            else {
                "zero_shot": {**sources[index], "reused": True},
                "segment": generated_arms["segment"],
            }
        )
        return index, {
            "index": index,
            "run_fingerprint": run_hash,
            "source_run_fingerprint": source_run_hash,
            "prompt_sha256": manifest[index]["prompt_sha256"],
            "source_response_sha256": (
                None
                if generated_pair
                else hashlib.sha256(
                    sources[index]["response"].encode("utf-8")
                ).hexdigest()
            ),
            "retrieval": {
                "best_similarity": manifest[index]["best_similarity"],
                "fallback_bare": manifest[index]["fallback_bare"],
                "selected_instance_ids": manifest[index]["selected_instance_ids"],
            },
            "arms": arms,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                _, record = future.result()
            except Exception as error:
                record = {
                    "index": index,
                    "run_fingerprint": run_hash,
                    "endpoint_error": repr(error),
                }
            records[index] = record
            write_jsonl_snapshot(output, records, indices)
    if set(records) != set(indices):
        raise GateFailure(f"Run is incomplete: {backbone}")
    errors = [record for record in records.values() if record.get("endpoint_error")]
    if errors:
        raise GateFailure(f"{backbone} halted with {len(errors)} endpoint errors")
    summary = analyze(backbone, records)
    raw_format_pass = all(
        summary["arms"][arm]["format_compliance"] >= 0.9
        for arm in ("zero_shot", "segment")
    )
    format_exempt = bool(config.get("official_format_gate_exempt", False))
    format_pass = raw_format_pass or format_exempt
    summary.update(
        {
            "status": "PASS" if format_pass else "FAIL",
            "format_gate": {
                "status": "PASS" if format_pass else "FAIL",
                "minimum_per_arm": 0.9,
                "raw_threshold_pass": raw_format_pass,
                "exempt": format_exempt,
                "policy": (
                    "raw official scores are primary; canonical-null is post-hoc appendix only"
                    if format_exempt
                    else "official format must be at least 90 percent per arm"
                ),
            },
            "model_generation_calls": int(config["calls"]),
            "run_fingerprint": run_hash,
            "source_run_fingerprint": source_run_hash,
            "source_results_sha256": source_results_sha256,
            "runtime": runtime,
            "results": str(output),
            "results_sha256": file_sha256(output),
        }
    )
    atomic_json(output.with_suffix(".summary.json"), summary)
    if not format_pass:
        raise GateFailure(f"{backbone} official format gate failed")
    return summary


def repair_qwen3_offline() -> Dict[str, Any]:
    backbone = "qwen3_vl_30b_a3b"
    source_path = OUTPUT_DIR / f"{backbone}.jsonl"
    source_run_path = source_path.with_suffix(source_path.suffix + ".run.json")
    source_summary_path = source_path.with_suffix(".summary.json")
    if not all(
        path.is_file() for path in (source_path, source_run_path, source_summary_path)
    ):
        raise GateFailure("Qwen3 ID offline repair requires the completed raw run")
    indices, _ = load_deployment_manifest()
    source_records = load_jsonl(source_path)
    source_run = json.loads(source_run_path.read_text())
    source_summary = json.loads(source_summary_path.read_text())
    source_run_hash = fingerprint(source_run)
    if (
        set(source_records) != set(indices)
        or any(record.get("endpoint_error") for record in source_records.values())
        or any(
            set(record.get("arms", {})) != {"zero_shot", "segment"}
            for record in source_records.values()
        )
        or {record.get("run_fingerprint") for record in source_records.values()}
        != {source_run_hash}
        or source_summary.get("run_fingerprint") != source_run_hash
    ):
        raise GateFailure("Qwen3 ID raw run failed offline-repair source validation")
    output = OUTPUT_DIR / f"{backbone}.offline_format_repair.jsonl"
    metadata = {
        "task": "Amendment5.1_Qwen3_ID_offline_format_repair",
        "backbone": backbone,
        "policy": "canonical-null",
        "post_hoc": True,
        "primary_inference_eligible": False,
        "model_generation_calls": 0,
        "source_results": str(source_path),
        "source_results_sha256": file_sha256(source_path),
        "source_summary_sha256": file_sha256(source_summary_path),
        "source_run_fingerprint": source_run_hash,
        "h0_sha256": file_sha256(H0_PATH),
        "scorer": "official VIKI-L2",
        "scorer_seed_rule": "original row index",
        "repair_contract": (
            "preserve each parseable generated answer exactly; use [] only when "
            "no parseable complete answer exists"
        ),
    }
    repair_hash = fingerprint(metadata)
    run_path = output.with_suffix(output.suffix + ".run.json")
    if run_path.is_file() and json.loads(run_path.read_text()) != metadata:
        raise GateFailure("Cannot overwrite differing Qwen3 ID offline repair")
    atomic_json(run_path, metadata)
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    repaired_records: Dict[int, Dict[str, Any]] = {}
    repair_counts: Dict[str, Dict[str, int]] = {
        "zero_shot": {},
        "segment": {},
    }
    task_score_mismatches = {"zero_shot": 0, "segment": 0}
    for index in indices:
        sample = native(dataset.iloc[index].to_dict())
        repaired_arms = {}
        for arm in ("zero_shot", "segment"):
            source_arm = source_records[index]["arms"][arm]
            repaired_response, method = repair_qwen3_response(
                source_arm["response"], "canonical-null"
            )
            metrics = bench.score_response(
                scorer,
                2,
                repaired_response,
                bench.get_ground_truth(sample),
                index,
            )
            if metrics["task_score"] != source_arm["task_score"]:
                task_score_mismatches[arm] += 1
            repair_counts[arm][method] = repair_counts[arm].get(method, 0) + 1
            repaired_arms[arm] = {
                "response": repaired_response,
                "source_response_sha256": hashlib.sha256(
                    source_arm["response"].encode("utf-8")
                ).hexdigest(),
                "repair_method": method,
                **metrics,
            }
        repaired_records[index] = {
            "index": index,
            "run_fingerprint": repair_hash,
            "source_generation_run_fingerprint": source_run_hash,
            "arms": repaired_arms,
        }
    if any(task_score_mismatches.values()):
        raise GateFailure(
            f"Qwen3 ID offline repair changed task outcomes: {task_score_mismatches}"
        )
    write_jsonl_snapshot(output, repaired_records, indices)
    summary = analyze(backbone, repaired_records)
    summary.update(
        {
            "task": "Amendment5.1_Qwen3_ID_offline_format_repair",
            "status": "PASS",
            "artifact_integrity_status": "PASS",
            "post_hoc": True,
            "primary_inference_eligible": False,
            "policy": "canonical-null",
            "model_generation_calls": 0,
            "run_fingerprint": repair_hash,
            "source_run_fingerprint": source_run_hash,
            "source_results_sha256": metadata["source_results_sha256"],
            "repair_counts": repair_counts,
            "task_score_mismatches": task_score_mismatches,
            "results": str(output),
            "results_sha256": file_sha256(output),
        }
    )
    atomic_json(output.with_suffix(".summary.json"), summary)
    return summary


def summarize() -> Dict[str, Any]:
    prereg = require_preregistration()
    gpt_ood_path = AMENDMENT3_DIR.parent / "amendment5/gpt_4o_optional.summary.json"
    route_probe_path = AMENDMENT3_DIR.parent / "amendment5/cloudgpt_route_probe.json"
    paths = {
        "qwen2_5_vl_72b": OUTPUT_DIR / "qwen2_5_vl_72b.summary.json",
        "qwen2_5_vl_7b_stock": OUTPUT_DIR / "qwen2_5_vl_7b_stock.summary.json",
        "rl_7b": OUTPUT_DIR / "rl_7b.smoke.summary.json",
        "qwen3_vl_30b_a3b": OUTPUT_DIR / "qwen3_vl_30b_a3b.summary.json",
        "qwen3_offline_appendix": (
            OUTPUT_DIR / "qwen3_vl_30b_a3b.offline_format_repair.summary.json"
        ),
        "gpt_4o_optional": OUTPUT_DIR / "gpt_4o_optional.summary.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    missing.extend(
        name
        for name, path in {
            "gpt_4o_optional_ood": gpt_ood_path,
            "cloudgpt_route_probe": route_probe_path,
        }.items()
        if not path.is_file()
    )
    if missing:
        raise GateFailure("Amendment 5.1 summary inputs missing: " + ", ".join(missing))
    summaries = {name: json.loads(path.read_text()) for name, path in paths.items()}
    gpt_ood = json.loads(gpt_ood_path.read_text())
    route_probe = json.loads(route_probe_path.read_text())
    if (
        gpt_ood.get("status") != "PASS"
        or gpt_ood.get("samples") != 1218
        or gpt_ood.get("gate", {}).get("status") != "PASS"
        or summaries["gpt_4o_optional"].get("samples") != ROWS
        or summaries["gpt_4o_optional"].get("format_gate", {}).get("status") != "FAIL"
        or route_probe.get("gpt_4o", {}).get("status") != "PASS"
        or route_probe.get("gemini_2_5_flash", {}).get("status") != "UNAVAILABLE"
    ):
        raise GateFailure("CloudGPT closed-model fold-in certificate is invalid")
    comparisons = {
        name: value["comparisons"]["zero_shot_to_segment"]
        for name, value in summaries.items()
        if "comparisons" in value and name != "qwen3_offline_appendix"
    }
    calls = {
        "qwen2_5_vl_72b": int(summaries["qwen2_5_vl_72b"]["model_generation_calls"]),
        "qwen2_5_vl_7b_stock": int(
            summaries["qwen2_5_vl_7b_stock"]["model_generation_calls"]
        ),
        "rl_7b_smoke": int(summaries["rl_7b"]["model_generation_calls"]),
        "rl_7b_segment": 0,
        "qwen3_vl_30b_a3b": int(
            summaries["qwen3_vl_30b_a3b"]["model_generation_calls"]
        ),
        "qwen3_offline_appendix": 0,
        "gpt_4o_optional": int(summaries["gpt_4o_optional"]["model_generation_calls"]),
    }
    prediction_outcomes = {
        "qwen2_5_vl_72b": {
            "prediction": prereg["predictions"]["qwen2_5_vl_72b"],
            "task_direction": "confirmed",
            "exceeds_composed_plus_3pp": (
                comparisons["qwen2_5_vl_72b"]["segment_minus_zero_shot"] > 0.03
            ),
            "gate_outcome": "FAIL_official_format",
        },
        "qwen2_5_vl_7b_stock": {
            "prediction": prereg["predictions"]["qwen2_5_vl_7b_stock"],
            "outcome": "falsified_positive_significant_gain",
            "gate_outcome": summaries["qwen2_5_vl_7b_stock"]["status"],
        },
        "rl_7b": {
            "prediction": prereg["predictions"]["rl_7b"],
            "outcome": "not_measured_smoke_halt",
            "gate_outcome": "FAIL_byte_identical_smoke",
        },
        "qwen3_vl_30b_a3b": {
            "prediction": prereg["predictions"]["qwen3_vl_30b_a3b"],
            "outcome": "falsified_large_significant_gain",
            "gate_outcome": "PASS_with_preregistered_format_exemption",
        },
    }
    required_status = {
        "qwen2_5_vl_72b": summaries["qwen2_5_vl_72b"]["status"],
        "qwen2_5_vl_7b_stock": summaries["qwen2_5_vl_7b_stock"]["status"],
        "rl_7b": summaries["rl_7b"]["status"],
    }
    summary = {
        "task": "Amendment5.1_ID_deployment_column",
        "status": "COMPLETE_WITH_BACKBONE_GATE_FAILURES",
        "campaign_closure_status": "BLOCKED_GEMINI_CLOUDGPT_DEPLOYMENT_UNAVAILABLE",
        "permanent_closure_reinstated": False,
        "chapter_draft_authorized": False,
        "required_backbone_status": required_status,
        "optional_backbone_status": {
            "qwen3_vl_30b_a3b": summaries["qwen3_vl_30b_a3b"]["status"],
            "gpt_4o_optional": summaries["gpt_4o_optional"]["status"],
        },
        "pending_fold_in": ["gemini_2_5_flash_OOD_and_ID_pair"],
        "blocked_fold_in": {
            "gemini_2_5_flash": "CloudGPT DeploymentNotFound for all bounded aliases"
        },
        "calls": calls,
        "total_local_generation_calls": sum(
            value for name, value in calls.items() if name != "gpt_4o_optional"
        ),
        "total_cloudgpt_id_generation_calls": calls["gpt_4o_optional"],
        "total_id_generation_calls": sum(calls.values()),
        "closed_model_fold_in": {
            "authorization": "post-preregistration user-directed optional fold-in",
            "gpt_4o_ood_status": gpt_ood["status"],
            "gpt_4o_ood_calls": gpt_ood["model_generation_calls"],
            "gpt_4o_ood_comparison": gpt_ood["comparisons"]["zero_shot_to_segment"],
            "gpt_4o_id_status": summaries["gpt_4o_optional"]["status"],
            "gpt_4o_id_calls": calls["gpt_4o_optional"],
            "gpt_4o_id_comparison": comparisons["gpt_4o_optional"],
            "gpt_4o_ood_summary": str(gpt_ood_path),
            "gpt_4o_ood_summary_sha256": file_sha256(gpt_ood_path),
            "cloudgpt_route_probe": str(route_probe_path),
            "cloudgpt_route_probe_sha256": file_sha256(route_probe_path),
        },
        "comparisons": comparisons,
        "prediction_outcomes": prediction_outcomes,
        "table_footnote": prereg["table_footnote"],
        "h0_sha256": file_sha256(H0_PATH),
        "manifest_sha256": file_sha256(DEPLOYMENT_MANIFEST_PATH),
        "summaries": {name: str(path) for name, path in paths.items()},
        "summary_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "next_action": (
            "obtain a confirmed CloudGPT Gemini 2.5 Flash deployment alias or access; "
            "then complete its frozen OOD+ID batches"
        ),
    }
    atomic_json(OUTPUT_DIR / "final_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VIKI Amendment 5.1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("build-manifest")
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument(
        "--backbone", choices=sorted(BACKBONES), required=True
    )
    preflight_parser.add_argument("--base-url", required=True)
    smoke_parser = subparsers.add_parser("rl-smoke")
    smoke_parser.add_argument("--base-url", required=True)
    subparsers.add_parser("certify-rl-smoke-failure")
    subparsers.add_parser("repair-qwen3-offline")
    subparsers.add_parser("summarize")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    run_parser.add_argument("--base-url", required=True)
    run_parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "freeze":
            result = freeze()
        elif args.command == "build-manifest":
            result = build_manifest()
        elif args.command == "preflight":
            result = preflight(args.backbone, args.base_url)
        elif args.command == "rl-smoke":
            result = rl_smoke(args.base_url)
        elif args.command == "certify-rl-smoke-failure":
            result = certify_rl_smoke_failure()
        elif args.command == "repair-qwen3-offline":
            result = repair_qwen3_offline()
        elif args.command == "summarize":
            result = summarize()
        else:
            result = run(args.backbone, args.base_url, args.workers)
    except GateFailure as error:
        print(f"GATE FAILURE: {error}")
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
