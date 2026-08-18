#!/usr/bin/env python3

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.stats import beta, binom
from viki_amendment3_f2 import (
    exact_mcnemar_p,
    file_sha256,
    fingerprint,
    load_bench,
    messages_sha256,
    native,
    server_metadata,
    token_count,
)
from viki_amendment4 import segment_flat_block

from habitat_llm.evaluation.viki_branch_conditions import get_instruction
from habitat_llm.evaluation.viki_memory_skill import add_memory_to_messages
from habitat_llm.evaluation.viki_segment_memory import SegmentInstance

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT.parent / "VIKI-R"
DATA_ROOT = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2"
AMENDMENT2_DIR = ROOT / "results/viki_memory_experiments/amendment2"
AMENDMENT3_DIR = ROOT / "results/viki_memory_experiments/amendment3"
AMENDMENT4_DIR = ROOT / "results/viki_memory_experiments/amendment4"
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment5"
M0_INSTANCES_PATH = AMENDMENT2_DIR / "m0_instances.jsonl"
M0_SKILLS_PATH = AMENDMENT2_DIR / "m0_skills.json"
M1_CACHE_PATH = AMENDMENT2_DIR / "m1_embeddings_all_mpnet_base_v2.npz"
H0_PATH = OUTPUT_DIR / "h0_frozen.json"
SUPERSESSION_PATH = OUTPUT_DIR / "closure_supersession.json"
MANIFEST_PATH = OUTPUT_DIR / "deployment_manifest.jsonl"
MANIFEST_SUMMARY_PATH = OUTPUT_DIR / "deployment_manifest.summary.json"
FINAL_CLOSURE_PATH = OUTPUT_DIR / "VIKI_GENERATION_CLOSED_PERMANENTLY.json"

ROWS = 1218
TOP_K = 6
SIMILARITY_THRESHOLD = 0.3
MAX_OUTPUT_TOKENS = 2000
MIN_CONTEXT_LENGTH = 16384
EMBEDDING_MODEL = "all-mpnet-base-v2"
SEED = 20260814
BOOTSTRAP_DRAWS = 100000

BACKBONES: Dict[str, Dict[str, Any]] = {
    "qwen2_5_vl_72b": {
        "label": "Qwen2.5-VL-72B",
        "model_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "model_revision": "89c86200743eec961a297729e7990e8f2ddbc4c5",
        "served_model": "qwen2.5-vl-72b-amendment3-f2",
        "provider": "local_openai_compatible",
        "serving_stack": "vLLM 0.11.2 (frozen F2 service)",
        "arms": ["segment"],
        "reuse": ["zero_shot", "composed"],
        "expected_new_calls": 1218,
    },
    "qwen2_5_vl_7b_stock": {
        "label": "Qwen2.5-VL-7B stock",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "served_model": "qwen2.5-vl-7b-amendment5",
        "provider": "local_openai_compatible",
        "serving_stack": "vLLM 0.11.2 deployment service",
        "arms": ["segment"],
        "reuse": ["zero_shot"],
        "expected_new_calls": 1218,
    },
    "gemini_2_5_flash": {
        "label": "Gemini-2.5-Flash",
        "model_id": "gemini-2.5-flash",
        "model_revision": "provider-managed",
        "served_model": "gemini-2.5-flash",
        "provider": "Gemini REST API",
        "serving_stack": "closed Gemini API",
        "arms": ["zero_shot", "segment"],
        "reuse": [],
        "expected_new_calls": 2436,
    },
    "qwen3_vl_30b_a3b": {
        "label": "Qwen3-VL-30B-A3B",
        "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "model_revision": "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c",
        "served_model": "qwen3-vl-30b-a3b-amendment5",
        "provider": "local_openai_compatible",
        "serving_stack": "upgraded vLLM 0.11.2 from F1",
        "arms": ["zero_shot", "segment"],
        "reuse": [],
        "expected_new_calls": 2436,
        "probe_candidate": "qwen3_vl_30b_a3b",
        "probe_ood_successes": 0,
        "probe_ood_samples": 200,
    },
    "gpt_4o_optional": {
        "label": "GPT-4o (optional)",
        "model_id": "gpt-4o-2024-08-06",
        "model_revision": "gpt-4o-2024-08-06",
        "served_model": "gpt-4o-2024-08-06",
        "provider": "OpenAI API",
        "serving_stack": "closed OpenAI API",
        "arms": ["zero_shot", "segment"],
        "reuse": [],
        "expected_new_calls": 2436,
        "optional": True,
    },
}


class GateFailure(RuntimeError):
    pass


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def repair_qwen3_response(response: str, policy: str) -> Tuple[str, str]:
    if policy == "tag-only":
        return (
            response.replace("<reasoning>", "<think>", 1).replace(
                "</reasoning>", "</think>", 1
            ),
            "reasoning_tags_to_think",
        )
    if policy != "canonical-null":
        raise GateFailure(f"Unknown Qwen3 offline repair policy: {policy}")
    answer_match = ANSWER_PATTERN.search(response)
    if answer_match is not None:
        answer = answer_match.group(1).strip()
        try:
            parsed = ast.literal_eval(answer)
        except (ValueError, SyntaxError, TypeError):
            parsed = None
        if isinstance(parsed, (list, dict)):
            return f"<think></think><answer>{answer}</answer>", "preserved_answer"
    return "<think></think><answer>[]</answer>", "null_unrecoverable_answer"


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


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


def load_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.is_file():
        return {}
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


def require_inputs() -> None:
    required = [
        M0_INSTANCES_PATH,
        M0_SKILLS_PATH,
        M1_CACHE_PATH,
        AMENDMENT3_DIR / "f1_pick.json",
        AMENDMENT3_DIR / "f1_qwen3_vl_30b_a3b.summary.json",
        AMENDMENT3_DIR / "f1_qwen3_vl_32b.summary.json",
        AMENDMENT3_DIR / "f2_ood.jsonl",
        AMENDMENT3_DIR / "f2_ood.jsonl.run.json",
        AMENDMENT3_DIR / "f2_ood.summary.json",
        AMENDMENT4_DIR / "VIKI_GENERATION_CLOSED.json",
        ROOT / "results/viki_official_7b_l2_ood.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise GateFailure("Missing Amendment 5 inputs: " + ", ".join(missing))


def frozen_h0() -> Dict[str, Any]:
    return {
        "task": "Amendment5_H0",
        "status": "FROZEN",
        "scope": ["H1", "H2", "H3.1", "H3.2", "H3.3 optional", "H4"],
        "configuration": {
            "pool": "all 19,499 contiguous M0 segment instances",
            "retrieval_query": "test instruction only",
            "candidate_text": "segment context",
            "embedding_model": EMBEDDING_MODEL,
            "metric": "normalized cosine",
            "top_k": TOP_K,
            "threshold_rule": (
                "if best cosine < 0.3 use bare prompt; otherwise inject top six "
                "segments in descending cosine order"
            ),
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "injection": (
                "headerless singleton skill segment rendering in similarity order"
            ),
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "minimum_context_length": MIN_CONTEXT_LENGTH,
            "temperature": 0,
            "decoding": "greedy",
            "prompt": "official VIKI-L2",
            "scorer_seed": "original row index",
            "per_backbone_tuning": False,
        },
        "backbones": BACKBONES,
        "order": [
            "H1 qwen2_5_vl_72b",
            "H2 qwen2_5_vl_7b_stock (parallel local)",
            "H3.1 gemini_2_5_flash",
            "H3.2 qwen3_vl_30b_a3b",
            "H3.3 gpt_4o_optional",
            "H4",
        ],
        "artifacts": {
            "m0_instances_sha256": file_sha256(M0_INSTANCES_PATH),
            "m0_skills_sha256": file_sha256(M0_SKILLS_PATH),
            "m1_cache_sha256": file_sha256(M1_CACHE_PATH),
            "f1_pick_sha256": file_sha256(AMENDMENT3_DIR / "f1_pick.json"),
            "amendment4_closure_sha256": file_sha256(
                AMENDMENT4_DIR / "VIKI_GENERATION_CLOSED.json"
            ),
        },
    }


def freeze_h0() -> Dict[str, Any]:
    require_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    closure = json.loads((AMENDMENT4_DIR / "VIKI_GENERATION_CLOSED.json").read_text())
    if (
        closure.get("status") != "CLOSED"
        or closure.get("closed_by") != "Amendment 4.1 G2b"
        or closure.get("further_viki_generation_authorized") is not False
    ):
        raise GateFailure("Amendment 5 requires the exact Amendment 4.1 closure")
    value = frozen_h0()
    if H0_PATH.is_file() and json.loads(H0_PATH.read_text()) != value:
        raise GateFailure("H0 is already frozen with different contents")
    atomic_json(H0_PATH, value)
    supersession = {
        "task": "Amendment5_closure_supersession",
        "status": "ACTIVE_ONCE",
        "supersedes": str(AMENDMENT4_DIR / "VIKI_GENERATION_CLOSED.json"),
        "superseded_sha256": file_sha256(
            AMENDMENT4_DIR / "VIKI_GENERATION_CLOSED.json"
        ),
        "authorized_backbones": list(BACKBONES),
        "authorized_calls": {
            key: int(value["expected_new_calls"]) for key, value in BACKBONES.items()
        },
        "configuration_sha256": fingerprint(value),
        "after_h4": "permanent closure; chapter draft",
    }
    if (
        SUPERSESSION_PATH.is_file()
        and json.loads(SUPERSESSION_PATH.read_text()) != supersession
    ):
        raise GateFailure("Amendment 5 supersession already differs")
    atomic_json(SUPERSESSION_PATH, supersession)
    return value


def require_h0(backbone: Optional[str] = None) -> Dict[str, Any]:
    if FINAL_CLOSURE_PATH.is_file():
        raise GateFailure(
            "Amendment 5 is permanently closed: " + FINAL_CLOSURE_PATH.read_text()
        )
    if not H0_PATH.is_file() or not SUPERSESSION_PATH.is_file():
        raise GateFailure("Run h0-freeze before any Amendment 5 operation")
    observed = json.loads(H0_PATH.read_text())
    if observed != frozen_h0():
        raise GateFailure("Observed H0 differs from frozen configuration")
    supersession = json.loads(SUPERSESSION_PATH.read_text())
    if supersession.get("status") != "ACTIVE_ONCE" or supersession.get(
        "configuration_sha256"
    ) != fingerprint(observed):
        raise GateFailure("Amendment 5 supersession is not active")
    if backbone is not None and backbone not in supersession["authorized_backbones"]:
        raise GateFailure(f"Backbone is not authorized by Amendment 5: {backbone}")
    return observed


def format_deployment_prompt(instances: Sequence[SegmentInstance]) -> str:
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


def build_manifest() -> Dict[str, Any]:
    require_h0()
    bench = load_bench()
    dataset = pd.read_parquet(DATA_ROOT / "val.parquet")
    if len(dataset) != ROWS:
        raise GateFailure(f"Expected {ROWS} OOD rows, observed {len(dataset)}")
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
        raise GateFailure(
            f"H0 full M0 bank mismatch: rows={len(instances)}, "
            f"unique={len({item.instance_id for item in instances})}"
        )
    cache = np.load(M1_CACHE_PATH, allow_pickle=False)
    instance_ids = [item.instance_id for item in instances]
    skill_names = [
        str(value["name"]) for value in json.loads(M0_SKILLS_PATH.read_text())["skills"]
    ]
    cache_digest = hashlib.sha256()
    for value in skill_names + instance_ids:
        cache_digest.update(value.encode("utf-8"))
        cache_digest.update(b"\0")
    expected_cache_key = cache_digest.hexdigest()
    if str(cache["cache_key"].item()) != expected_cache_key:
        raise GateFailure("H0 embedding cache fingerprint mismatch")
    context_embeddings = np.asarray(cache["context_embeddings"])
    if context_embeddings.shape != (19499, 768):
        raise GateFailure(
            f"H0 context embedding shape mismatch: {context_embeddings.shape}"
        )
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    instructions = [
        get_instruction(native(dataset.iloc[index].to_dict())) for index in range(ROWS)
    ]
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
    for index in range(ROWS):
        similarities = context_embeddings @ query_embeddings[index]
        top_positions = np.argpartition(similarities, -TOP_K)[-TOP_K:]
        ranked = sorted(
            (
                (float(similarities[position]), instance_ids[position], position)
                for position in top_positions
            ),
            key=lambda item: (-item[0], item[1]),
        )
        selected = ranked if ranked[0][0] >= SIMILARITY_THRESHOLD else []
        selected_instances = [instances[position] for _, _, position in selected]
        prompt = format_deployment_prompt(selected_instances)
        sample = native(dataset.iloc[index].to_dict())
        zero_messages = bench.get_messages(sample)
        segment_messages = add_memory_to_messages(bench.get_messages(sample), prompt)
        records[index] = {
            "index": index,
            "instruction": instructions[index],
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
    write_jsonl_snapshot(MANIFEST_PATH, records, list(range(ROWS)))
    summary = {
        "task": "Amendment5_deployment_manifest",
        "status": "PASS",
        "rows": len(records),
        "bank_instances": len(instances),
        "top_k": TOP_K,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "fallback_rows": sum(record["fallback_bare"] for record in records.values()),
        "selected_count_distribution": dict(
            sorted(
                pd.Series(
                    [
                        len(record["selected_instance_ids"])
                        for record in records.values()
                    ]
                )
                .value_counts()
                .astype(int)
                .to_dict()
                .items()
            )
        ),
        "minimum_best_similarity": min(
            record["best_similarity"] for record in records.values()
        ),
        "maximum_best_similarity": max(
            record["best_similarity"] for record in records.values()
        ),
        "results": str(MANIFEST_PATH),
        "results_sha256": file_sha256(MANIFEST_PATH),
        "h0_sha256": file_sha256(H0_PATH),
    }
    atomic_json(MANIFEST_SUMMARY_PATH, summary)
    return summary


def load_manifest() -> Dict[int, Dict[str, Any]]:
    require_h0()
    if not MANIFEST_SUMMARY_PATH.is_file():
        raise GateFailure("Build the deployment manifest before preflight")
    summary = json.loads(MANIFEST_SUMMARY_PATH.read_text())
    if (
        summary.get("status") != "PASS"
        or summary.get("rows") != ROWS
        or summary.get("bank_instances") != 19499
        or summary.get("top_k") != TOP_K
        or summary.get("similarity_threshold") != SIMILARITY_THRESHOLD
        or summary.get("h0_sha256") != file_sha256(H0_PATH)
        or summary.get("results_sha256") != file_sha256(MANIFEST_PATH)
    ):
        raise GateFailure("Deployment manifest certificate is invalid")
    records = load_jsonl(MANIFEST_PATH)
    if set(records) != set(range(ROWS)):
        raise GateFailure("Deployment manifest row coverage is incomplete")
    return records


def validate_local_service(backbone: str, base_url: str) -> Dict[str, Any]:
    expected = BACKBONES[backbone]
    runtime = server_metadata([base_url])
    endpoint = runtime[0]
    models = endpoint.get("models", [])
    if len(models) != 1:
        raise GateFailure(f"Unexpected model inventory for {backbone}: {models}")
    model = models[0]
    if (
        model.get("id") != expected["served_model"]
        or model.get("root") != expected["model_id"]
        or model.get("max_model_len", 0) < MIN_CONTEXT_LENGTH
    ):
        raise GateFailure(
            f"H0 service mismatch for {backbone}: observed={model}, "
            f"expected_served={expected['served_model']}, "
            f"min_context={MIN_CONTEXT_LENGTH}"
        )
    return {"base_url": base_url, **endpoint}


def preflight_local(backbone: str, base_url: str) -> Dict[str, Any]:
    require_h0(backbone)
    if BACKBONES[backbone]["provider"] != "local_openai_compatible":
        raise GateFailure(f"Backbone is not local: {backbone}")
    runtime = validate_local_service(backbone, base_url)
    bench = load_bench()
    dataset = pd.read_parquet(DATA_ROOT / "val.parquet")
    manifest = load_manifest()
    records = []
    for index in range(ROWS):
        sample = native(dataset.iloc[index].to_dict())
        zero_messages = bench.get_messages(sample)
        segment_messages = add_memory_to_messages(
            bench.get_messages(sample), manifest[index]["segment_prompt"]
        )
        if (
            messages_sha256(zero_messages)
            != manifest[index]["prompt_sha256"]["zero_shot"]
            or messages_sha256(segment_messages)
            != manifest[index]["prompt_sha256"]["segment"]
        ):
            raise GateFailure(f"Deployment prompt changed at index {index}")
        zero_tokens = token_count(
            base_url, BACKBONES[backbone]["served_model"], zero_messages
        )
        segment_tokens = token_count(
            base_url, BACKBONES[backbone]["served_model"], segment_messages
        )
        if max(zero_tokens, segment_tokens) + MAX_OUTPUT_TOKENS > int(
            runtime["models"][0]["max_model_len"]
        ):
            raise GateFailure(f"H0 context gate failed at {backbone}:{index}")
        records.append(
            {
                "index": index,
                "zero_shot_input_tokens": zero_tokens,
                "segment_input_tokens": segment_tokens,
            }
        )
    frame = pd.DataFrame(records).sort_values("index")
    parquet_path = OUTPUT_DIR / f"{backbone}.preflight.parquet"
    frame.to_parquet(parquet_path, index=False)
    summary = {
        "task": "Amendment5_local_preflight",
        "backbone": backbone,
        "status": "PASS",
        "rows": len(frame),
        "model_generation_calls": 0,
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
        "runtime": runtime,
        "results": str(parquet_path),
        "results_sha256": file_sha256(parquet_path),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "h0_sha256": file_sha256(H0_PATH),
    }
    atomic_json(OUTPUT_DIR / f"{backbone}.preflight.summary.json", summary)
    return summary


def source_records(backbone: str) -> Tuple[Dict[int, Dict[str, Any]], str]:
    if backbone == "qwen2_5_vl_72b":
        path = AMENDMENT3_DIR / "f2_ood.jsonl"
        run_path = path.with_suffix(path.suffix + ".run.json")
        records = load_jsonl(path)
        metadata = json.loads(run_path.read_text())
        run_hash = fingerprint(metadata)
        if len(records) != ROWS or any(
            record.get("run_fingerprint") != run_hash for record in records.values()
        ):
            raise GateFailure("H1 source records are incomplete or mismatched")
        return records, run_hash
    if backbone == "qwen2_5_vl_7b_stock":
        path = AMENDMENT2_DIR / "b1_ood.jsonl"
        records = load_jsonl(path)
        hashes = {record.get("run_fingerprint") for record in records.values()}
        if (
            len(records) != ROWS
            or len(hashes) != 1
            or sum(
                record["arms"]["base"]["task_score"] == 1 for record in records.values()
            )
            != 0
        ):
            raise GateFailure("H2 baseline source is incomplete or mixed")
        return records, str(next(iter(hashes)))
    return {}, ""


def run_metadata_local(
    backbone: str, base_url: str, source_hash: str
) -> Dict[str, Any]:
    return {
        "task": "Amendment5_deployment",
        "backbone": backbone,
        "config": BACKBONES[backbone],
        "h0_sha256": file_sha256(H0_PATH),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "preflight_sha256": file_sha256(
            OUTPUT_DIR / f"{backbone}.preflight.summary.json"
        ),
        "source_run_fingerprint": source_hash,
        "source_results_sha256": (
            file_sha256(AMENDMENT3_DIR / "f2_ood.jsonl")
            if backbone == "qwen2_5_vl_72b"
            else file_sha256(AMENDMENT2_DIR / "b1_ood.jsonl")
            if backbone == "qwen2_5_vl_7b_stock"
            else None
        ),
        "base_url": base_url,
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "scorer_seed_rule": "original row index",
    }


def local_completion_gate(
    backbone: str, records: Mapping[int, Mapping[str, Any]]
) -> Dict[str, Any]:
    if backbone != "qwen3_vl_30b_a3b":
        return {"status": "NOT_APPLICABLE"}
    probe = BACKBONES[backbone]
    lower, upper = probe_interval(
        int(probe["probe_ood_successes"]),
        int(probe["probe_ood_samples"]),
    )
    zero_successes = sum(
        record["arms"]["zero_shot"]["task_score"] == 1 for record in records.values()
    )
    zero_accuracy = zero_successes / len(records)
    zero_format = sum(
        record["arms"]["zero_shot"]["format_score"] == 1 for record in records.values()
    ) / len(records)
    consistent = lower <= zero_accuracy <= upper
    format_pass = zero_format >= 0.9
    return {
        "status": "PASS" if consistent and format_pass else "FAIL",
        "probe_successes": int(probe["probe_ood_successes"]),
        "probe_samples": int(probe["probe_ood_samples"]),
        "probe_clopper_pearson_95": [lower, upper],
        "full_zero_shot_successes": zero_successes,
        "full_zero_shot_accuracy": zero_accuracy,
        "full_zero_shot_format_compliance": zero_format,
        "minimum_format_compliance": 0.9,
        "consistent_with_probe": consistent,
        "format_pass": format_pass,
    }


def run_local(backbone: str, base_url: str, workers: int) -> Dict[str, Any]:
    require_h0(backbone)
    config = BACKBONES[backbone]
    if config["provider"] != "local_openai_compatible":
        raise GateFailure(f"Backbone is not a local run: {backbone}")
    preflight_path = OUTPUT_DIR / f"{backbone}.preflight.summary.json"
    if not preflight_path.is_file():
        raise GateFailure(f"Run local preflight first: {backbone}")
    preflight = json.loads(preflight_path.read_text())
    if (
        preflight.get("status") != "PASS"
        or preflight.get("rows") != ROWS
        or preflight.get("h0_sha256") != file_sha256(H0_PATH)
        or preflight.get("manifest_sha256") != file_sha256(MANIFEST_PATH)
    ):
        raise GateFailure(f"Local preflight certificate is invalid: {backbone}")
    runtime = validate_local_service(backbone, base_url)
    if runtime != preflight["runtime"]:
        raise GateFailure(f"Runtime changed after preflight: {backbone}")
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    dataset = pd.read_parquet(DATA_ROOT / "val.parquet")
    manifest = load_manifest()
    source, source_hash = source_records(backbone)
    metadata = run_metadata_local(backbone, base_url, source_hash)
    run_hash = fingerprint(metadata)
    output = OUTPUT_DIR / f"{backbone}.jsonl"
    run_path = output.with_suffix(output.suffix + ".run.json")
    if run_path.is_file():
        if json.loads(run_path.read_text()) != metadata:
            raise GateFailure(f"Cannot resume {backbone}: metadata differs")
    else:
        atomic_json(run_path, metadata)
    records = load_jsonl(output)
    for index, record in records.items():
        if index not in range(ROWS) or record.get("run_fingerprint") != run_hash:
            raise GateFailure(f"Invalid resume record at {backbone}:{index}")
    pending = [
        index
        for index in range(ROWS)
        if index not in records or records[index].get("endpoint_error")
    ]
    client = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=5, timeout=3600)

    def generate(index: int) -> Tuple[int, Dict[str, Any]]:
        sample = native(dataset.iloc[index].to_dict())
        zero_messages = bench.get_messages(sample)
        segment_messages = add_memory_to_messages(
            bench.get_messages(sample), manifest[index]["segment_prompt"]
        )
        reused: Dict[str, Any] = {}
        reused_prompt_hashes: Dict[str, str] = {}
        if backbone == "qwen2_5_vl_72b":
            reused = {
                "zero_shot": source[index]["arms"]["zero_shot"],
                "composed": source[index]["arms"]["skill_memory"],
            }
            composed_messages = add_memory_to_messages(
                bench.get_messages(sample), source[index]["route"]["memory_prompt"]
            )
            reused_prompt_hashes = {
                "zero_shot": messages_sha256(zero_messages),
                "composed": messages_sha256(composed_messages),
            }
            if int(source[index]["zero_shot_input_tokens"]) != token_count(
                base_url, config["served_model"], zero_messages
            ) or int(source[index]["route"]["memory_input_tokens"]) != token_count(
                base_url, config["served_model"], composed_messages
            ):
                raise GateFailure(f"H1 reused prompt token mismatch at {index}")
        elif backbone == "qwen2_5_vl_7b_stock":
            reused = {"zero_shot": source[index]["arms"]["base"]}
            reused_prompt_hashes = {"zero_shot": messages_sha256(zero_messages)}
        generated_arms: Dict[str, Any] = {}
        arm_messages = {
            "zero_shot": zero_messages,
            "segment": segment_messages,
        }
        for arm in config["arms"]:
            completion = client.chat.completions.create(
                model=config["served_model"],
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
        return index, {
            "index": index,
            "run_fingerprint": run_hash,
            "source_run_fingerprint": source_hash,
            "prompt_sha256": {
                **manifest[index]["prompt_sha256"],
                **reused_prompt_hashes,
            },
            "retrieval": {
                "best_similarity": manifest[index]["best_similarity"],
                "fallback_bare": manifest[index]["fallback_bare"],
                "selected_instance_ids": manifest[index]["selected_instance_ids"],
            },
            "arms": {**reused, **generated_arms},
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
            write_jsonl_snapshot(output, records, list(range(ROWS)))
    if set(records) != set(range(ROWS)):
        raise GateFailure(f"Run is incomplete: {backbone}")
    errors = [record for record in records.values() if record.get("endpoint_error")]
    if errors:
        raise GateFailure(f"{backbone} halted with {len(errors)} endpoint errors")
    gate = local_completion_gate(backbone, records)
    summary = analyze_records(backbone, records)
    summary.update(
        {
            "status": "PASS" if gate["status"] != "FAIL" else "FAIL",
            "model_generation_calls": int(config["expected_new_calls"]),
            "new_run_fingerprint": run_hash,
            "source_run_fingerprint": source_hash,
            "runtime": runtime,
            "gate": gate,
            "results": str(output),
        }
    )
    atomic_json(output.with_suffix(".summary.json"), summary)
    if gate["status"] == "FAIL":
        raise GateFailure(f"{backbone} completion gate failed: {gate}")
    return summary


def repair_qwen3_offline(policy: str) -> Dict[str, Any]:
    backbone = "qwen3_vl_30b_a3b"
    require_h0(backbone)
    source_path = OUTPUT_DIR / f"{backbone}.jsonl"
    source_run_path = source_path.with_suffix(source_path.suffix + ".run.json")
    source_summary_path = source_path.with_suffix(".summary.json")
    if not all(
        path.is_file() for path in (source_path, source_run_path, source_summary_path)
    ):
        raise GateFailure("Qwen3 offline repair requires the completed raw run")
    source_records = load_jsonl(source_path)
    source_run = json.loads(source_run_path.read_text())
    source_summary = json.loads(source_summary_path.read_text())
    source_run_hash = fingerprint(source_run)
    if (
        set(source_records) != set(range(ROWS))
        or any(record.get("endpoint_error") for record in source_records.values())
        or any(
            set(record.get("arms", {})) != {"zero_shot", "segment"}
            for record in source_records.values()
        )
        or {record.get("run_fingerprint") for record in source_records.values()}
        != {source_run_hash}
        or source_summary.get("new_run_fingerprint") != source_run_hash
    ):
        raise GateFailure("Qwen3 raw run failed offline-repair source validation")
    suffix = "offline_tag_only" if policy == "tag-only" else "offline_format_repair"
    output = OUTPUT_DIR / f"{backbone}.{suffix}.jsonl"
    metadata = {
        "task": "Amendment5_qwen3_offline_format_repair",
        "backbone": backbone,
        "policy": policy,
        "post_hoc": True,
        "primary_inference_eligible": False,
        "model_generation_calls": 0,
        "source_results": str(source_path),
        "source_results_sha256": file_sha256(source_path),
        "source_summary_sha256": file_sha256(source_summary_path),
        "source_run_metadata_sha256": file_sha256(source_run_path),
        "source_run_fingerprint": source_run_hash,
        "h0_sha256": file_sha256(H0_PATH),
        "scorer": "official VIKI-L2",
        "scorer_seed_rule": "original row index",
        "repair_contract": (
            "replace the first reasoning tag pair only"
            if policy == "tag-only"
            else (
                "preserve each parseable generated answer exactly; use [] only "
                "when no parseable complete answer exists"
            )
        ),
    }
    repair_hash = fingerprint(metadata)
    run_path = output.with_suffix(output.suffix + ".run.json")
    if run_path.is_file() and json.loads(run_path.read_text()) != metadata:
        raise GateFailure(f"Cannot overwrite differing offline repair: {run_path}")
    atomic_json(run_path, metadata)
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    dataset = pd.read_parquet(DATA_ROOT / "val.parquet")
    repaired_records: Dict[int, Dict[str, Any]] = {}
    repair_counts: Dict[str, Dict[str, int]] = {
        "zero_shot": {},
        "segment": {},
    }
    task_score_mismatches = {"zero_shot": 0, "segment": 0}
    for index in range(ROWS):
        source_record = source_records[index]
        sample = native(dataset.iloc[index].to_dict())
        repaired_arms = {}
        for arm in ("zero_shot", "segment"):
            source_arm = source_record["arms"][arm]
            repaired_response, method = repair_qwen3_response(
                source_arm["response"], policy
            )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
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
                "source_prompt_tokens": source_arm.get("prompt_tokens"),
                "source_completion_tokens": source_arm.get("completion_tokens"),
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
            f"Offline repair changed task outcomes: {task_score_mismatches}"
        )
    write_jsonl_snapshot(output, repaired_records, list(range(ROWS)))
    gate = local_completion_gate(backbone, repaired_records)
    summary = analyze_records(backbone, repaired_records)
    summary.update(
        {
            "task": "Amendment5_qwen3_offline_format_repair",
            "status": gate["status"],
            "artifact_integrity_status": "PASS",
            "post_hoc": True,
            "primary_inference_eligible": False,
            "policy": policy,
            "model_generation_calls": 0,
            "run_fingerprint": repair_hash,
            "source_run_fingerprint": source_run_hash,
            "source_results_sha256": metadata["source_results_sha256"],
            "repair_counts": repair_counts,
            "task_score_mismatches": task_score_mismatches,
            "gate": gate,
            "results": str(output),
            "results_sha256": file_sha256(output),
        }
    )
    atomic_json(output.with_suffix(".summary.json"), summary)
    return summary


def migrate_h1_metadata(base_url: str) -> Dict[str, Any]:
    require_h0("qwen2_5_vl_72b")
    backbone = "qwen2_5_vl_72b"
    output = OUTPUT_DIR / f"{backbone}.jsonl"
    run_path = output.with_suffix(output.suffix + ".run.json")
    summary_path = output.with_suffix(".summary.json")
    if not output.is_file() or not run_path.is_file():
        raise GateFailure("H1 migration requires completed run artifacts")
    records = load_jsonl(output)
    if set(records) != set(range(ROWS)):
        raise GateFailure("H1 migration requires all 1,218 records")
    source, source_hash = source_records(backbone)
    expected_metadata = run_metadata_local(backbone, base_url, source_hash)
    observed_metadata = json.loads(run_path.read_text())
    if observed_metadata == expected_metadata:
        return {
            "task": "Amendment5_H1_metadata_migration",
            "status": "NOT_NEEDED",
            "rows": len(records),
            "run_fingerprint": fingerprint(expected_metadata),
        }
    comparable = native(expected_metadata)
    comparable.pop("source_results_sha256", None)
    if observed_metadata != comparable:
        raise GateFailure("H1 prior metadata differs beyond the approved migration")
    prior_hash = fingerprint(observed_metadata)
    if any(
        record.get("run_fingerprint") != prior_hash
        or set(record.get("arms", {})) != {"zero_shot", "composed", "segment"}
        for record in records.values()
    ):
        raise GateFailure("H1 records do not match the prior run fingerprint")
    runtime = validate_local_service(backbone, base_url)
    bench = load_bench()
    dataset = pd.read_parquet(DATA_ROOT / "val.parquet")
    zero_verified = 0
    composed_verified = 0
    for index in range(ROWS):
        sample = native(dataset.iloc[index].to_dict())
        zero_messages = bench.get_messages(sample)
        composed_messages = add_memory_to_messages(
            bench.get_messages(sample), source[index]["route"]["memory_prompt"]
        )
        if int(source[index]["zero_shot_input_tokens"]) != token_count(
            base_url, BACKBONES[backbone]["served_model"], zero_messages
        ) or int(source[index]["route"]["memory_input_tokens"]) != token_count(
            base_url,
            BACKBONES[backbone]["served_model"],
            composed_messages,
        ):
            raise GateFailure(f"H1 migration prompt mismatch at index {index}")
        if (
            records[index]["arms"]["zero_shot"] != source[index]["arms"]["zero_shot"]
            or records[index]["arms"]["composed"]
            != source[index]["arms"]["skill_memory"]
        ):
            raise GateFailure(f"H1 migration reused output mismatch at {index}")
        zero_verified += 1
        composed_verified += 1
    new_hash = fingerprint(expected_metadata)
    for record in records.values():
        record["run_fingerprint"] = new_hash
    write_jsonl_snapshot(output, records, list(range(ROWS)))
    atomic_json(run_path, expected_metadata)
    summary = analyze_records(backbone, records)
    summary.update(
        {
            "status": "PASS",
            "model_generation_calls": ROWS,
            "new_run_fingerprint": new_hash,
            "source_run_fingerprint": source_hash,
            "runtime": runtime,
            "gate": {"status": "NOT_APPLICABLE"},
            "results": str(output),
        }
    )
    atomic_json(summary_path, summary)
    migration = {
        "task": "Amendment5_H1_metadata_migration",
        "status": "PASS",
        "rows": len(records),
        "prior_run_fingerprint": prior_hash,
        "new_run_fingerprint": new_hash,
        "zero_shot_prompts_verified": zero_verified,
        "composed_prompts_verified": composed_verified,
        "source_results_sha256": expected_metadata["source_results_sha256"],
        "model_generation_calls_repeated": 0,
    }
    atomic_json(OUTPUT_DIR / "qwen2_5_vl_72b.migration.json", migration)
    return migration


def api_configuration(backbone: str) -> Tuple[str, str]:
    if backbone == "gemini_2_5_flash":
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "GEMINI_API_KEY",
        )
    if backbone == "gpt_4o_optional":
        return "https://api.openai.com/v1", "OPENAI_API_KEY"
    raise GateFailure(f"Backbone is not a closed API model: {backbone}")


def cloudgpt_deployment(backbone: str) -> str:
    if backbone == "gpt_4o_optional":
        return "gpt-4o-20240806"
    raise GateFailure(f"CloudGPT deployment is unavailable for {backbone}")


def api_gate(backbone: str, records: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    zero_successes = sum(
        record["arms"]["zero_shot"]["task_score"] == 1 for record in records.values()
    )
    zero_format = sum(
        record["arms"]["zero_shot"]["format_score"] == 1 for record in records.values()
    ) / len(records)
    if backbone == "gemini_2_5_flash":
        expected_probability = 0.1051
        lower, upper = binomial_acceptance(ROWS, expected_probability)
        status = "PASS" if lower <= zero_successes <= upper else "FAIL"
        return {
            "status": status,
            "published_accuracy": expected_probability,
            "acceptance_successes": [lower, upper],
            "observed_successes": zero_successes,
            "observed_accuracy": zero_successes / ROWS,
            "format_compliance": zero_format,
            "method": "central 95% Binomial(n=1218,p=0.1051) acceptance interval",
        }
    if backbone == "gpt_4o_optional":
        expected_probability = 0.1002
        lower, upper = binomial_acceptance(ROWS, expected_probability)
        status = "PASS" if lower <= zero_successes <= upper else "FAIL"
        return {
            "status": status,
            "published_accuracy": expected_probability,
            "acceptance_successes": [lower, upper],
            "observed_successes": zero_successes,
            "observed_accuracy": zero_successes / ROWS,
            "format_compliance": zero_format,
            "method": "central 95% Binomial(n=1218,p=0.1002) acceptance interval",
        }
    raise GateFailure(f"No API gate for backbone: {backbone}")


def run_api(backbone: str, workers: int) -> Dict[str, Any]:
    require_h0(backbone)
    config = BACKBONES[backbone]
    if config["provider"] not in {"Gemini REST API", "OpenAI API"}:
        raise GateFailure(f"Backbone is not an API run: {backbone}")
    use_cloudgpt = os.environ.get("CLOUDGPT_USE_AZURE_CLI") == "1"
    if use_cloudgpt:
        from cloudgpt_aoai import get_openai_client

        base_url = "https://cloudgpt-openai.azure-api.net/"
        api_key_env = "CLOUDGPT_USE_AZURE_CLI"
        request_model = cloudgpt_deployment(backbone)
        client = get_openai_client(timeout=3600, max_retries=5)
        transport = "CloudGPT AzureOpenAI with Azure CLI cached authentication"
    else:
        base_url, api_key_env = api_configuration(backbone)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise GateFailure(
                f"Missing {api_key_env}; set it directly in the terminal before {backbone}"
            )
        request_model = config["served_model"]
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=5,
            timeout=3600,
        )
        transport = config["provider"]
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    dataset = pd.read_parquet(DATA_ROOT / "val.parquet")
    manifest = load_manifest()
    metadata = {
        "task": "Amendment5_deployment",
        "backbone": backbone,
        "config": config,
        "h0_sha256": file_sha256(H0_PATH),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "base_url": base_url,
        "api_key_env": api_key_env,
        "request_model": request_model,
        "transport": transport,
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "scorer_seed_rule": "original row index",
    }
    run_hash = fingerprint(metadata)
    output = OUTPUT_DIR / f"{backbone}.jsonl"
    run_path = output.with_suffix(output.suffix + ".run.json")
    if run_path.is_file():
        if json.loads(run_path.read_text()) != metadata:
            raise GateFailure(f"Cannot resume {backbone}: metadata differs")
    else:
        atomic_json(run_path, metadata)
    records = load_jsonl(output)
    for index, record in records.items():
        if index not in range(ROWS) or record.get("run_fingerprint") != run_hash:
            raise GateFailure(f"Invalid API resume record at {backbone}:{index}")
    pending = [
        index
        for index in range(ROWS)
        if index not in records or records[index].get("endpoint_error")
    ]

    def generate(index: int) -> Tuple[int, Dict[str, Any]]:
        sample = native(dataset.iloc[index].to_dict())
        zero_messages = bench.get_messages(sample)
        segment_messages = add_memory_to_messages(
            bench.get_messages(sample), manifest[index]["segment_prompt"]
        )
        messages = {"zero_shot": zero_messages, "segment": segment_messages}
        arms = {}
        for arm in ("zero_shot", "segment"):
            completion = client.chat.completions.create(
                model=request_model,
                messages=messages[arm],
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
            arms[arm] = {**generated, **metrics}
        return index, {
            "index": index,
            "run_fingerprint": run_hash,
            "prompt_sha256": manifest[index]["prompt_sha256"],
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
            write_jsonl_snapshot(output, records, list(range(ROWS)))
    if set(records) != set(range(ROWS)):
        raise GateFailure(f"API run is incomplete: {backbone}")
    errors = [record for record in records.values() if record.get("endpoint_error")]
    if errors:
        raise GateFailure(f"{backbone} halted with {len(errors)} endpoint errors")
    gate = api_gate(backbone, records)
    summary = analyze_records(backbone, records)
    summary.update(
        {
            "status": "PASS" if gate["status"] == "PASS" else "FAIL",
            "model_generation_calls": int(config["expected_new_calls"]),
            "new_run_fingerprint": run_hash,
            "gate": gate,
            "serving_stack": transport,
            "results": str(output),
        }
    )
    atomic_json(output.with_suffix(".summary.json"), summary)
    if gate["status"] != "PASS":
        raise GateFailure(f"{backbone} zero-shot replication gate failed: {gate}")
    return summary


def paired_interval(
    records: Mapping[int, Mapping[str, Any]],
    left: str,
    right: str,
    seed_offset: int,
) -> List[float]:
    differences = np.array(
        [
            int(records[index]["arms"][right]["task_score"] == 1)
            - int(records[index]["arms"][left]["task_score"] == 1)
            for index in sorted(records)
        ]
    )
    probabilities = np.array(
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
    return [float(value) for value in np.quantile(deltas, [0.025, 0.975])]


def pair_summary(
    records: Mapping[int, Mapping[str, Any]],
    left: str,
    right: str,
    seed_offset: int,
) -> Dict[str, Any]:
    left_success = {
        index: records[index]["arms"][left]["task_score"] == 1 for index in records
    }
    right_success = {
        index: records[index]["arms"][right]["task_score"] == 1 for index in records
    }
    left_only = sum(
        left_success[index] and not right_success[index] for index in records
    )
    right_only = sum(
        not left_success[index] and right_success[index] for index in records
    )
    return {
        "samples": len(records),
        f"{left}_successes": sum(left_success.values()),
        f"{right}_successes": sum(right_success.values()),
        f"{left}_accuracy": sum(left_success.values()) / len(records),
        f"{right}_accuracy": sum(right_success.values()) / len(records),
        f"{right}_minus_{left}": (
            sum(right_success.values()) - sum(left_success.values())
        )
        / len(records),
        "paired_delta_interval": paired_interval(records, left, right, seed_offset),
        "left_only": left_only,
        "right_only": right_only,
        "discordant_pairs": left_only + right_only,
        "mcnemar_exact_p": exact_mcnemar_p(right_only, left_only),
    }


def analyze_records(
    backbone: str, records: Mapping[int, Mapping[str, Any]]
) -> Dict[str, Any]:
    arm_names = list(next(iter(records.values()))["arms"])
    comparisons = {}
    offset = 0
    for left_position, left in enumerate(arm_names):
        for right in arm_names[left_position + 1 :]:
            offset += 1
            comparisons[f"{left}_to_{right}"] = pair_summary(
                records, left, right, offset
            )
    dataset = pd.read_parquet(DATA_ROOT / "val.parquet")
    family_by_index = {
        index: str(
            native(dataset.iloc[index]["reward_model"])["ground_truth"]["task_name"]
        )
        for index in range(ROWS)
    }
    families: Dict[str, Any] = {}
    for family in sorted(set(family_by_index.values())):
        subset = {
            index: record
            for index, record in records.items()
            if family_by_index[index] == family
        }
        families[family] = {
            "samples": len(subset),
            "arms": {
                arm: {
                    "successes": sum(
                        record["arms"][arm]["task_score"] == 1
                        for record in subset.values()
                    ),
                    "format_compliance": sum(
                        record["arms"][arm]["format_score"] == 1
                        for record in subset.values()
                    )
                    / len(subset),
                }
                for arm in arm_names
            },
            "segment_vs_zero": (
                pair_summary(subset, "zero_shot", "segment", 100 + len(families))
                if "zero_shot" in arm_names
                else None
            ),
        }
    return {
        "task": "Amendment5_deployment",
        "backbone": backbone,
        "samples": len(records),
        "arms": {
            arm: {
                "successes": sum(
                    record["arms"][arm]["task_score"] == 1
                    for record in records.values()
                ),
                "accuracy": sum(
                    record["arms"][arm]["task_score"] == 1
                    for record in records.values()
                )
                / len(records),
                "format_compliance": sum(
                    record["arms"][arm]["format_score"] == 1
                    for record in records.values()
                )
                / len(records),
            }
            for arm in arm_names
        },
        "comparisons": comparisons,
        "by_family": families,
    }


def binomial_acceptance(n: int, probability: float) -> Tuple[int, int]:
    return (
        int(binom.ppf(0.025, n, probability)),
        int(binom.ppf(0.975, n, probability)),
    )


def probe_interval(successes: int, samples: int) -> Tuple[float, float]:
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(0.025, successes, samples - successes + 1))
    )
    upper = (
        1.0
        if successes == samples
        else float(beta.ppf(0.975, successes + 1, samples - successes))
    )
    return lower, upper


def finalize() -> Dict[str, Any]:
    require_h0()
    required = [
        "qwen2_5_vl_72b",
        "qwen2_5_vl_7b_stock",
        "gemini_2_5_flash",
        "qwen3_vl_30b_a3b",
    ]
    optional = ["gpt_4o_optional"]
    summaries = {}
    missing = []
    for backbone in required + optional:
        path = OUTPUT_DIR / f"{backbone}.summary.json"
        if not path.is_file():
            if backbone in required:
                missing.append(backbone)
            continue
        value = json.loads(path.read_text())
        if value.get("status") != "PASS":
            raise GateFailure(f"H4 input did not pass: {backbone}")
        summaries[backbone] = value
    if missing:
        raise GateFailure("H4 required backbones are incomplete: " + ", ".join(missing))
    points = []
    for backbone, summary in summaries.items():
        comparison = summary["comparisons"]["zero_shot_to_segment"]
        points.append(
            {
                "backbone": backbone,
                "label": BACKBONES[backbone]["label"],
                "memory_variant": "full-bank segment-flat deployment",
                "zero_shot_accuracy": comparison["zero_shot_accuracy"],
                "memory_delta": comparison["segment_minus_zero_shot"],
                "memory_delta_lower": comparison["paired_delta_interval"][0],
                "memory_delta_upper": comparison["paired_delta_interval"][1],
                "mcnemar_exact_p": comparison["mcnemar_exact_p"],
            }
        )
    rl_summary = json.loads(
        (ROOT / "results/viki_memory_skill_7b_l2_ood.summary.json").read_text()
    )
    points.append(
        {
            "backbone": "rl_7b_frozen_negative",
            "label": "RL 7B",
            "memory_variant": "legacy skill memory",
            "zero_shot_accuracy": 403 / 1218,
            "memory_delta": (373 - 403) / 1218,
            "memory_delta_lower": None,
            "memory_delta_upper": None,
            "mcnemar_exact_p": rl_summary.get("mcnemar_exact_p"),
        }
    )
    frame = pd.DataFrame(points)
    table_path = OUTPUT_DIR / "h4_cross_model_points.csv"
    frame.to_csv(table_path, index=False)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for row in points:
        axis.scatter(row["zero_shot_accuracy"], row["memory_delta"], s=70)
        axis.annotate(
            f"{row['label']}\n{row['memory_variant']}",
            (row["zero_shot_accuracy"], row["memory_delta"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel("Zero-shot OOD accuracy")
    axis.set_ylabel("Paired memory delta")
    axis.set_title("VIKI deployment competence window")
    figure.tight_layout()
    figure_path = OUTPUT_DIR / "h4_cross_model_window.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    window_backbones = [
        "qwen2_5_vl_72b",
        "gemini_2_5_flash",
        "qwen3_vl_30b_a3b",
    ]
    if "gpt_4o_optional" in summaries:
        window_backbones.append("gpt_4o_optional")
    positive = [
        backbone
        for backbone in window_backbones
        if summaries[backbone]["comparisons"]["zero_shot_to_segment"][
            "segment_minus_zero_shot"
        ]
        > 0
        and summaries[backbone]["comparisons"]["zero_shot_to_segment"][
            "mcnemar_exact_p"
        ]
        < 0.05
    ]
    floor_comparison = summaries["qwen2_5_vl_7b_stock"]["comparisons"][
        "zero_shot_to_segment"
    ]
    floor_positive = (
        floor_comparison["segment_minus_zero_shot"] > 0
        and floor_comparison["mcnemar_exact_p"] < 0.05
    )
    if len(positive) == len(window_backbones) and not floor_positive:
        ending = "gain_on_every_window_and_none_at_floor"
    elif positive == ["qwen2_5_vl_72b"]:
        ending = "gain_only_on_72b"
    else:
        ending = "heterogeneous_window_gain"
    final = {
        "task": "Amendment5_H4",
        "status": "PASS",
        "campaign_status": "VIKI_GENERATION_CLOSED_PERMANENTLY",
        "summaries": summaries,
        "points": points,
        "window_backbones": window_backbones,
        "positive_window_backbones": positive,
        "positive_window_count": len(positive),
        "window_backbone_count": len(window_backbones),
        "floor_positive": floor_positive,
        "selected_ending": ending,
        "optional_gpt4o_run": "gpt_4o_optional" in summaries,
        "figure": str(figure_path),
        "table": str(table_path),
        "c4_status": (
            "blocked on original PartNR memory banks and generation logs; "
            "user-side handoff unchanged"
        ),
        "next_deliverable": "chapter draft",
        "further_viki_generation_authorized": False,
    }
    final_path = OUTPUT_DIR / "h4_final.json"
    atomic_json(final_path, final)
    atomic_json(
        FINAL_CLOSURE_PATH,
        {
            "status": "CLOSED_PERMANENTLY",
            "closed_by": "Amendment 5 H4",
            "final": str(final_path),
            "final_sha256": file_sha256(final_path),
            "further_viki_generation_authorized": False,
            "next_deliverable": "chapter draft",
        },
    )
    supersession = json.loads(SUPERSESSION_PATH.read_text())
    supersession["status"] = "CONSUMED_AND_CLOSED"
    supersession["final_closure"] = str(FINAL_CLOSURE_PATH)
    atomic_json(SUPERSESSION_PATH, supersession)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VIKI Amendment 5")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("h0-freeze")
    subparsers.add_parser("build-manifest")
    preflight = subparsers.add_parser("preflight-local")
    run = subparsers.add_parser("run-local")
    for subparser in (preflight, run):
        subparser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
        subparser.add_argument("--base-url", required=True)
    run.add_argument("--workers", type=int, default=8)
    api = subparsers.add_parser("run-api")
    api.add_argument(
        "--backbone",
        choices=("gemini_2_5_flash", "gpt_4o_optional"),
        required=True,
    )
    api.add_argument("--workers", type=int, default=8)
    migrate = subparsers.add_parser("migrate-h1")
    migrate.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    repair = subparsers.add_parser("repair-qwen3-offline")
    repair.add_argument(
        "--policy", choices=("tag-only", "canonical-null"), required=True
    )
    subparsers.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "h0-freeze":
            result = freeze_h0()
        elif args.command == "build-manifest":
            result = build_manifest()
        elif args.command == "preflight-local":
            result = preflight_local(args.backbone, args.base_url)
        elif args.command == "run-api":
            result = run_api(args.backbone, args.workers)
        elif args.command == "migrate-h1":
            result = migrate_h1_metadata(args.base_url)
        elif args.command == "repair-qwen3-offline":
            result = repair_qwen3_offline(args.policy)
        elif args.command == "finalize":
            result = finalize()
        else:
            result = run_local(args.backbone, args.base_url, args.workers)
    except GateFailure as error:
        print(f"GATE FAILURE: {error}")
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
