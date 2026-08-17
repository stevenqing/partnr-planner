#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from transformers import AutoTokenizer

from habitat_llm.evaluation.viki_memory_skill import add_memory_to_messages
from habitat_llm.evaluation.viki_segment_memory import (
    RetrievedGroup,
    SegmentInstance,
    SegmentMemoryBank,
    build_retrieval_context,
    build_subgoal_messages,
    format_grouped_memory,
    parse_subgoal_prediction,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT.parent / "VIKI-R"
DATA_ROOT = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2"
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment3"
AMENDMENT2_DIR = ROOT / "results/viki_memory_experiments/amendment2"
OVERRIDE_PATH = OUTPUT_DIR / "f2_local_override.json"
ID_MANIFEST_PATH = (
    ROOT / "results/viki_memory_experiments/amendment1/a5_id_safety_manifest.parquet"
)
CENSUS_PATH = (
    ROOT / "results/viki_memory_experiments/amendment1/c0_composition_census.parquet"
)
C_PRIME_MANIFEST_PATH = OUTPUT_DIR / "f2_cprime_manifest.parquet"
C_PRIME_AUDIT_PATH = OUTPUT_DIR / "f2_cprime_leakage_audit.parquet"
M0_SUMMARY_PATH = AMENDMENT2_DIR / "m0.summary.json"
M0_INSTANCES_PATH = AMENDMENT2_DIR / "m0_instances.jsonl"
M0_SKILLS_PATH = AMENDMENT2_DIR / "m0_skills.json"
M1_CACHE_PATH = AMENDMENT2_DIR / "m1_embeddings_all_mpnet_base_v2.npz"

MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
MODEL_REVISION = "89c86200743eec961a297729e7990e8f2ddbc4c5"
SERVED_MODEL = "qwen2.5-vl-72b-amendment3-f2"
SEED = 20260814
CONTEXT_LENGTH = 16384
MAX_OUTPUT_TOKENS = 2000
SUBGOAL_MAX_TOKENS = 128
SIMILARITY_THRESHOLD = 0.3
INSTANCES_PER_SKILL = 2
MAX_INSTANCES = 6
TOKEN_TOLERANCE = 0.05
C_PRIME_ROWS = 400
UNIT_MIN_INSTANCES = 30
C_PRIME_CHANNELS = ("instance", "productivity")
CLARIFICATION = "Amendment 3.1"
FLAT_BUILDER = "next-nearest-whole-train-rows-then-token-trim"
C_PRIME_PREFLIGHT_SUMMARY_PATH = OUTPUT_DIR / "f2_cprime_preflight.summary.json"


class GateFailure(RuntimeError):
    pass


class SymmetricDrop(GateFailure):
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


def fingerprint(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def messages_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    serialized = json.dumps(
        list(messages), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def cprime_preflight_path(channel: str) -> Path:
    return OUTPUT_DIR / f"f2_cprime_{channel}.preflight.jsonl"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bench() -> Any:
    path = ROOT / "habitat_llm/evaluation/viki_bench.py"
    spec = importlib.util.spec_from_file_location("_viki_bench_amendment3_f2", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_inputs() -> Dict[str, Any]:
    required = (
        OVERRIDE_PATH,
        ID_MANIFEST_PATH,
        CENSUS_PATH,
        M0_SUMMARY_PATH,
        M0_INSTANCES_PATH,
        M0_SKILLS_PATH,
        M1_CACHE_PATH,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise GateFailure("Missing F2 inputs: " + ", ".join(missing))
    override = json.loads(OVERRIDE_PATH.read_text())
    selection = override.get("local_selection", {})
    if (
        override.get("status") != "COMMITTED"
        or selection.get("model_id") != MODEL_ID
        or selection.get("model_revision") != MODEL_REVISION
        or selection.get("served_model") != SERVED_MODEL
    ):
        raise GateFailure("F2 local override does not match the frozen runner")
    m0_summary = json.loads(M0_SUMMARY_PATH.read_text())
    if m0_summary.get("gate", {}).get("status") != "PASS":
        raise GateFailure("F2 requires a passing frozen M0 bank")
    return override


def require_frozen_args(args: argparse.Namespace) -> None:
    observed = {
        "model": args.model,
        "base_urls": list(args.base_urls),
        "max_tokens": args.max_tokens,
        "subgoal_max_tokens": args.subgoal_max_tokens,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
    }
    expected = {
        "model": SERVED_MODEL,
        "base_urls": ["http://127.0.0.1:8050/v1"],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "subgoal_max_tokens": SUBGOAL_MAX_TOKENS,
        "embedding_model": "all-mpnet-base-v2",
        "embedding_device": "cpu",
    }
    if observed != expected:
        raise GateFailure(
            "F2 command differs from the frozen local override: "
            f"observed={observed}, expected={expected}"
        )


def exact_mcnemar_p(fail_to_success: int, success_to_fail: int) -> float:
    discordant = fail_to_success + success_to_fail
    if discordant == 0:
        return 1.0
    tail = min(fail_to_success, success_to_fail)
    return min(
        1.0,
        2
        * sum(math.comb(discordant, value) for value in range(tail + 1))
        / (2**discordant),
    )


def tokenize_url(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1") + "/tokenize"


def token_count(
    base_url: str, model: str, messages: Sequence[Mapping[str, Any]]
) -> int:
    response = requests.post(
        tokenize_url(base_url),
        json={
            "model": model,
            "messages": list(messages),
            "add_generation_prompt": True,
        },
        timeout=120,
    )
    response.raise_for_status()
    return int(response.json()["count"])


def server_metadata(base_urls: Sequence[str]) -> List[Dict[str, Any]]:
    metadata = []
    for base_url in base_urls:
        root_url = base_url.rstrip("/").removesuffix("/v1")
        version_response = requests.get(f"{root_url}/version", timeout=30)
        version_response.raise_for_status()
        models_response = requests.get(f"{base_url.rstrip('/')}/models", timeout=30)
        models_response.raise_for_status()
        models = models_response.json().get("data", [])
        metadata.append(
            {
                "base_url": base_url,
                "version": version_response.json(),
                "models": [
                    {
                        key: model.get(key)
                        for key in ("id", "root", "parent", "max_model_len")
                    }
                    for model in models
                ],
            }
        )
    return metadata


def validate_server_metadata(metadata: Sequence[Mapping[str, Any]]) -> None:
    if len(metadata) != 1:
        raise GateFailure("F2 requires exactly one frozen local endpoint")
    endpoint = metadata[0]
    if endpoint.get("version", {}).get("version") != "0.11.2":
        raise GateFailure(f"Unexpected vLLM runtime: {endpoint.get('version')}")
    models = endpoint.get("models", [])
    if len(models) != 1:
        raise GateFailure(f"Unexpected served model inventory: {models}")
    model = models[0]
    expected = {
        "id": SERVED_MODEL,
        "root": MODEL_ID,
        "parent": None,
        "max_model_len": CONTEXT_LENGTH,
    }
    if model != expected:
        raise GateFailure(f"F2 server differs from the frozen local override: {model}")


def load_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.is_file():
        return {}
    lines = path.read_text().splitlines(keepends=True)
    records = {}
    for position, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if position == len(lines) - 1 and not line.endswith("\n"):
                break
            raise GateFailure(f"Malformed JSONL record in {path}:{position + 1}")
        if "index" not in record:
            raise GateFailure(f"JSONL record has no index in {path}:{position + 1}")
        index = int(record["index"])
        if index in records:
            raise GateFailure(f"Duplicate JSONL index in {path}: {index}")
        records[index] = record
    return records


def load_cprime_checkpoint(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.is_file():
        return {}
    records: Dict[int, Dict[str, Any]] = {}
    lines = path.read_text().splitlines(keepends=True)
    for position, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if position == len(lines) - 1 and not line.endswith("\n"):
                break
            raise GateFailure(f"Malformed C-prime JSONL in {path}:{position + 1}")
        if "index" not in record:
            raise GateFailure(
                f"C-prime JSONL record has no index in {path}:{position + 1}"
            )
        index = int(record["index"])
        prior = records.get(index)
        if prior is not None:
            if record.get("run_fingerprint") != prior.get("run_fingerprint"):
                raise GateFailure(
                    f"C-prime journal fingerprint changed at index {index}"
                )
            prior_arms = set(prior.get("arms", {}))
            current_arms = set(record.get("arms", {}))
            if not prior_arms < current_arms:
                raise GateFailure(
                    f"C-prime journal revision is not an arm superset at {index}"
                )
        records[index] = record
    return records


def validate_resume_records(
    records: Mapping[int, Mapping[str, Any]],
    indices: Sequence[int],
    run_hash: str,
    required_arms: Sequence[str],
) -> None:
    expected = set(indices)
    unexpected = sorted(set(records) - expected)
    if unexpected:
        raise GateFailure(f"Resume contains unexpected indices: {unexpected[:10]}")
    for index, record in records.items():
        if record.get("run_fingerprint") != run_hash:
            raise GateFailure(f"Resume fingerprint mismatch at index {index}")
        if record.get("endpoint_error"):
            continue
        route = record.get("route")
        arms = record.get("arms")
        if not isinstance(route, dict) or not isinstance(arms, dict):
            raise GateFailure(f"Malformed successful resume record at index {index}")
        if set(required_arms) - set(arms):
            raise GateFailure(f"Resume record is missing arms at index {index}")
        for arm in required_arms:
            if not {
                "response",
                "prompt_tokens",
                "completion_tokens",
                "task_score",
                "format_score",
            } <= set(arms[arm]):
                raise GateFailure(f"Resume arm is malformed at index {index}:{arm}")


def pair_summary(frame: pd.DataFrame, left: str, right: str) -> Dict[str, Any]:
    left_success = frame[f"{left}_task_score"] == 1
    right_success = frame[f"{right}_task_score"] == 1
    fail_to_success = int(((~left_success) & right_success).sum())
    success_to_fail = int((left_success & (~right_success)).sum())
    return {
        "samples": len(frame),
        f"{left}_successes": int(left_success.sum()),
        f"{right}_successes": int(right_success.sum()),
        f"{left}_mean_task_score": float(frame[f"{left}_task_score"].mean()),
        f"{right}_mean_task_score": float(frame[f"{right}_task_score"].mean()),
        "absolute_delta": float(
            frame[f"{right}_task_score"].mean() - frame[f"{left}_task_score"].mean()
        ),
        f"{left}_format_compliance": float((frame[f"{left}_format_score"] == 1).mean()),
        f"{right}_format_compliance": float(
            (frame[f"{right}_format_score"] == 1).mean()
        ),
        "fail_to_success": fail_to_success,
        "success_to_fail": success_to_fail,
        "discordant_pairs": fail_to_success + success_to_fail,
        "mcnemar_exact_p": exact_mcnemar_p(fail_to_success, success_to_fail),
    }


def arm_summary(frame: pd.DataFrame, arm: str) -> Dict[str, Any]:
    return {
        "samples": len(frame),
        "successes": int((frame[f"{arm}_task_score"] == 1).sum()),
        "mean_task_score": float(frame[f"{arm}_task_score"].mean()),
        "format_successes": int((frame[f"{arm}_format_score"] == 1).sum()),
        "format_compliance": float((frame[f"{arm}_format_score"] == 1).mean()),
    }


def retained_diagnostics(frame: pd.DataFrame) -> Dict[str, Any]:
    counts = frame["retained_instances"].astype(int)
    return {
        "mean": float(counts.mean()),
        "median": float(counts.median()),
        "minimum": int(counts.min()),
        "maximum": int(counts.max()),
        "zero_instance_rows": int((counts == 0).sum()),
        "distribution": {
            str(key): int(value)
            for key, value in sorted(Counter(counts.tolist()).items())
        },
        "route_parse_errors": int(frame["route_parse_error"].sum()),
        "skill_below_threshold_groups": int(
            frame["skill_below_threshold_groups"].sum()
        ),
        "context_below_threshold_groups": int(
            frame["context_below_threshold_groups"].sum()
        ),
        "total_limit_groups": int(frame["total_limit_groups"].sum()),
    }


class F2Provider:
    def __init__(self, args: argparse.Namespace, bench: Any) -> None:
        self.args = args
        self.bench = bench
        self.clients = [
            OpenAI(
                api_key=os.environ.get(args.api_key_env, "EMPTY"),
                base_url=base_url,
                max_retries=args.max_retries,
                timeout=args.timeout,
            )
            for base_url in args.base_urls
        ]
        self.memory = SegmentMemoryBank(
            M0_INSTANCES_PATH,
            M0_SKILLS_PATH,
            embedding_model=args.embedding_model,
            embedding_device=args.embedding_device,
            cache_path=M1_CACHE_PATH,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )

    def endpoint(self, index: int) -> str:
        return self.args.base_urls[index % len(self.args.base_urls)]

    def call(
        self,
        index: int,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
    ) -> Dict[str, Any]:
        completion = self.clients[index % len(self.clients)].chat.completions.create(
            model=self.args.model,
            messages=list(messages),
            max_tokens=max_tokens,
            temperature=0,
        )
        usage = completion.usage
        return {
            "response": completion.choices[0].message.content or "",
            "prompt_tokens": None if usage is None else usage.prompt_tokens,
            "completion_tokens": None if usage is None else usage.completion_tokens,
        }

    def route(
        self,
        sample: Dict[str, Any],
        index: int,
        allowed_instance_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, Any], List[RetrievedGroup], str, List[Dict[str, Any]], int]:
        route_messages = build_subgoal_messages(sample)
        prediction = self.call(index, route_messages, self.args.subgoal_max_tokens)
        prediction_error = None
        try:
            subgoals = parse_subgoal_prediction(prediction["response"])
        except (json.JSONDecodeError, ValueError) as error:
            subgoals = []
            prediction_error = repr(error)
        groups = self.memory.retrieve(
            subgoals,
            build_retrieval_context(sample),
            threshold=SIMILARITY_THRESHOLD,
            per_skill=INSTANCES_PER_SKILL,
            total=MAX_INSTANCES,
            allowed_instance_ids=allowed_instance_ids,
        )
        memory_prompt = format_grouped_memory(groups)
        memory_messages = add_memory_to_messages(
            self.bench.get_messages(sample), memory_prompt
        )
        memory_tokens = token_count(
            self.endpoint(index), self.args.model, memory_messages
        )
        if memory_tokens + self.args.max_tokens > CONTEXT_LENGTH:
            raise GateFailure(
                "F2 memory prompt exceeds frozen context without trimming: "
                f"{memory_tokens}+{self.args.max_tokens}>{CONTEXT_LENGTH}"
            )
        route = {
            **prediction,
            "messages": route_messages,
            "parsed_subgoals": subgoals,
            "parse_error": prediction_error,
            "groups": [group.to_dict() for group in groups],
            "memory_prompt": memory_prompt,
            "memory_input_tokens": memory_tokens,
            "token_headroom": CONTEXT_LENGTH - (memory_tokens + self.args.max_tokens),
            "retained_instance_count": sum(len(group.instances) for group in groups),
        }
        return route, groups, memory_prompt, memory_messages, memory_tokens

    def generate_pair(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        route, _, _, memory_messages, _ = self.route(sample, index)
        zero_messages = self.bench.get_messages(sample)
        zero_tokens = token_count(self.endpoint(index), self.args.model, zero_messages)
        if zero_tokens + self.args.max_tokens > CONTEXT_LENGTH:
            raise GateFailure(
                "F2 zero-shot prompt exceeds frozen context: "
                f"{zero_tokens}+{self.args.max_tokens}>{CONTEXT_LENGTH}"
            )
        return {
            "route": route,
            "arms": {
                "zero_shot": self.call(index, zero_messages, self.args.max_tokens),
                "skill_memory": self.call(index, memory_messages, self.args.max_tokens),
            },
            "zero_shot_input_tokens": zero_tokens,
        }

    def flat_candidate_rows(
        self,
        sample: Dict[str, Any],
        allowed_instance_ids: Sequence[str],
    ) -> List[List[SegmentInstance]]:
        allowed = set(str(value) for value in allowed_instance_ids)
        query_context = build_retrieval_context(sample)
        with self.memory.embedding_lock:
            query_embedding = np.asarray(
                self.memory.embedding_model.encode(
                    [query_context],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]
            )
        grouped: Dict[int, List[Tuple[float, SegmentInstance]]] = {}
        for position, instance in enumerate(self.memory.instances):
            if instance.instance_id not in allowed:
                continue
            similarity = float(
                np.dot(query_embedding, self.memory.context_embeddings[position])
            )
            grouped.setdefault(instance.source_train_index, []).append(
                (similarity, instance)
            )
        ranked_rows = sorted(
            grouped.items(),
            key=lambda item: (
                -max(similarity for similarity, _ in item[1]),
                item[0],
            ),
        )
        return [
            [
                instance
                for _, instance in sorted(
                    row,
                    key=lambda item: (-item[0], item[1].instance_id),
                )
            ]
            for _, row in ranked_rows
        ]

    def trim_flat_prompt(
        self,
        sample: Dict[str, Any],
        index: int,
        prompt: str,
        target_fragment_tokens: int,
        target_total_tokens: int,
    ) -> Tuple[str, int]:
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        observed: Dict[int, Tuple[str, int]] = {}

        def evaluate(count: int) -> Tuple[str, int]:
            count = min(len(encoded), max(1, count))
            if count not in observed:
                candidate = self.tokenizer.decode(
                    encoded[:count], skip_special_tokens=False
                )
                messages = add_memory_to_messages(
                    self.bench.get_messages(sample), candidate
                )
                observed[count] = (
                    candidate,
                    token_count(self.endpoint(index), self.args.model, messages),
                )
            return observed[count]

        evaluate(len(encoded))
        lower = 1
        upper = len(encoded)
        while lower <= upper:
            middle = (lower + upper) // 2
            _, total = evaluate(middle)
            if total < target_total_tokens:
                lower = middle + 1
            else:
                upper = middle - 1
        centers = {
            lower,
            upper,
            min(len(encoded), max(1, target_fragment_tokens)),
        }
        counts = {
            count
            for center in centers
            for count in range(max(1, center - 16), min(len(encoded), center + 16) + 1)
        }
        for count in sorted(counts):
            evaluate(count)
        best = min(
            observed.values(),
            key=lambda item: abs(item[1] - target_total_tokens),
        )
        for count in range(max(1, lower - 2), min(len(encoded), lower + 2) + 1):
            candidate = self.tokenizer.decode(
                encoded[:count], skip_special_tokens=False
            )
            messages = add_memory_to_messages(
                self.bench.get_messages(sample), candidate
            )
            total = token_count(self.endpoint(index), self.args.model, messages)
            if abs(total - target_total_tokens) < abs(best[1] - target_total_tokens):
                best = (candidate, total)
        return best

    def build_flat_control(
        self,
        sample: Dict[str, Any],
        index: int,
        allowed_instance_ids: Sequence[str],
        skill_total_tokens: int,
        has_skill_memory: bool,
    ) -> Dict[str, Any]:
        base_messages = self.bench.get_messages(sample)
        base_tokens = token_count(self.endpoint(index), self.args.model, base_messages)
        if not has_skill_memory:
            return {
                "prompt": "",
                "messages": base_messages,
                "input_tokens": base_tokens,
                "candidate_instance_ids_before_prefix_trim": [],
                "flat_k": 0,
                "flat_row_k": 0,
                "truncated": False,
                "relative_difference": 0.0,
            }
        candidate_rows = self.flat_candidate_rows(sample, allowed_instance_ids)
        selected: List[SegmentInstance] = []
        prompt = ""
        messages = base_messages
        total = base_tokens
        flat_row_k = 0
        for row in candidate_rows:
            selected.extend(row)
            flat_row_k += 1
            prompt = format_flat_prompt(selected)
            messages = add_memory_to_messages(self.bench.get_messages(sample), prompt)
            total = token_count(self.endpoint(index), self.args.model, messages)
            if total >= skill_total_tokens:
                break
        relative_difference = abs(total - skill_total_tokens) / skill_total_tokens
        if total < skill_total_tokens * (1 - TOKEN_TOLERANCE):
            raise SymmetricDrop(
                "Flat pool exhausted below the frozen 5% lower bound: "
                f"skill={skill_total_tokens}, flat={total}, "
                f"rows={len(candidate_rows)}, difference={relative_difference:.6f}"
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
            relative_difference = abs(total - skill_total_tokens) / skill_total_tokens
        if relative_difference > TOKEN_TOLERANCE:
            raise GateFailure(
                "Flat control is outside the frozen 5% token tolerance: "
                f"skill={skill_total_tokens}, flat={total}, "
                f"difference={relative_difference:.6f}"
            )
        if total + self.args.max_tokens > CONTEXT_LENGTH:
            raise GateFailure(
                "F2 flat prompt exceeds frozen context: "
                f"{total}+{self.args.max_tokens}>{CONTEXT_LENGTH}"
            )
        return {
            "prompt": prompt,
            "messages": messages,
            "input_tokens": total,
            "candidate_instance_ids_before_prefix_trim": [
                item.instance_id for item in selected
            ],
            "flat_k": len(selected),
            "flat_row_k": flat_row_k,
            "truncated": truncated,
            "relative_difference": relative_difference,
        }

    def generate_cprime(
        self,
        sample: Dict[str, Any],
        index: int,
        allowed_instance_ids: Sequence[str],
    ) -> Dict[str, Any]:
        route, groups, _, skill_messages, skill_tokens = self.route(
            sample, index, allowed_instance_ids=allowed_instance_ids
        )
        has_skill_memory = any(group.instances for group in groups)
        flat = self.build_flat_control(
            sample,
            index,
            allowed_instance_ids,
            skill_tokens,
            has_skill_memory,
        )
        zero_messages = self.bench.get_messages(sample)
        zero_tokens = token_count(self.endpoint(index), self.args.model, zero_messages)
        if zero_tokens + self.args.max_tokens > CONTEXT_LENGTH:
            raise GateFailure(
                "F2 C-prime zero-shot prompt exceeds frozen context: "
                f"{zero_tokens}+{self.args.max_tokens}>{CONTEXT_LENGTH}"
            )
        return {
            "route": route,
            "flat": {key: value for key, value in flat.items() if key != "messages"},
            "zero_shot_input_tokens": zero_tokens,
            "arms": {
                "zero_shot": self.call(index, zero_messages, self.args.max_tokens),
                "skill_memory": self.call(index, skill_messages, self.args.max_tokens),
                "flat_memory": self.call(index, flat["messages"], self.args.max_tokens),
            },
        }

    def preflight_cprime(
        self,
        sample: Dict[str, Any],
        index: int,
        allowed_instance_ids: Sequence[str],
        reused_route: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if reused_route is None:
            route, groups, _, skill_messages, skill_tokens = self.route(
                sample, index, allowed_instance_ids=allowed_instance_ids
            )
            has_skill_memory = any(group.instances for group in groups)
        else:
            route = native(dict(reused_route))
            memory_prompt = str(route.get("memory_prompt", ""))
            skill_messages = add_memory_to_messages(
                self.bench.get_messages(sample), memory_prompt
            )
            skill_tokens = token_count(
                self.endpoint(index), self.args.model, skill_messages
            )
            if skill_tokens != int(route.get("memory_input_tokens", -1)):
                raise GateFailure(
                    "Reused route changed its skill prompt token count at "
                    f"index {index}: stored={route.get('memory_input_tokens')}, "
                    f"observed={skill_tokens}"
                )
            serialized_groups = route.get("groups")
            if not isinstance(serialized_groups, list):
                raise GateFailure(f"Reused route is malformed at index {index}")
            has_skill_memory = any(
                group.get("instances") for group in serialized_groups
            )
        zero_messages = self.bench.get_messages(sample)
        zero_tokens = token_count(self.endpoint(index), self.args.model, zero_messages)
        try:
            flat = self.build_flat_control(
                sample,
                index,
                allowed_instance_ids,
                skill_tokens,
                has_skill_memory,
            )
        except SymmetricDrop as error:
            return {
                "route": route,
                "zero_shot_input_tokens": zero_tokens,
                "prompt_sha256": {
                    "zero_shot": messages_sha256(zero_messages),
                    "skill_memory": messages_sha256(skill_messages),
                },
                "dropped": True,
                "drop_reason": str(error),
                "legacy_route_reused": reused_route is not None,
            }
        return {
            "route": route,
            "flat": {key: value for key, value in flat.items() if key != "messages"},
            "zero_shot_input_tokens": zero_tokens,
            "prompt_sha256": {
                "zero_shot": messages_sha256(zero_messages),
                "skill_memory": messages_sha256(skill_messages),
                "flat_memory": messages_sha256(flat["messages"]),
            },
            "dropped": False,
            "legacy_route_reused": reused_route is not None,
        }

    def generate_cprime_from_preflight(
        self,
        sample: Dict[str, Any],
        index: int,
        preflight: Mapping[str, Any],
        arms: Sequence[str],
    ) -> Dict[str, Dict[str, Any]]:
        zero_messages = self.bench.get_messages(sample)
        skill_messages = add_memory_to_messages(
            self.bench.get_messages(sample), preflight["route"]["memory_prompt"]
        )
        flat_messages = add_memory_to_messages(
            self.bench.get_messages(sample), preflight["flat"]["prompt"]
        )
        messages = {
            "zero_shot": zero_messages,
            "skill_memory": skill_messages,
            "flat_memory": flat_messages,
        }
        expected_hashes = preflight["prompt_sha256"]
        for arm, arm_messages in messages.items():
            observed = messages_sha256(arm_messages)
            if observed != expected_hashes.get(arm):
                raise GateFailure(f"Preflight prompt changed at index {index}:{arm}")
        return {
            arm: self.call(index, messages[arm], self.args.max_tokens) for arm in arms
        }


def format_flat_prompt(instances: Sequence[SegmentInstance]) -> str:
    if not instances:
        return ""
    lines = [
        "Flat memory guidance from successful training trajectory segments:",
        "Retrieved examples:",
    ]
    for number, instance in enumerate(instances, 1):
        lines.extend(
            [
                f"{number}. Context: {instance.context}",
                "   Availability: "
                + json.dumps(instance.self_cond, sort_keys=True, ensure_ascii=True),
                "   Demonstration: "
                + json.dumps(instance.demo, separators=(",", ":"), ensure_ascii=True),
            ]
        )
    lines.extend(
        [
            "Use these examples only as generic in-context evidence.",
            "Ground every concrete choice in the current task, image, and robot APIs.",
        ]
    )
    return "\n".join(lines)


def run_metadata(
    args: argparse.Namespace,
    task: str,
    channel: str,
    split: str,
    indices: Sequence[int],
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    artifacts = {
        "override_sha256": file_sha256(OVERRIDE_PATH),
        "m0_summary_sha256": file_sha256(M0_SUMMARY_PATH),
        "m0_instances_sha256": file_sha256(M0_INSTANCES_PATH),
        "m0_skills_sha256": file_sha256(M0_SKILLS_PATH),
        "m1_cache_sha256": file_sha256(M1_CACHE_PATH),
    }
    if manifest_path is not None:
        artifacts["manifest_sha256"] = file_sha256(manifest_path)
    runtime = server_metadata(args.base_urls)
    validate_server_metadata(runtime)
    return {
        "task": task,
        "channel": channel,
        "split": split,
        "indices": list(indices),
        "indices_sha256": fingerprint({"indices": list(indices)}),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "served_model": args.model,
        "base_urls": list(args.base_urls),
        "server_metadata": runtime,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "subgoal_max_tokens": args.subgoal_max_tokens,
        "context_length": CONTEXT_LENGTH,
        "embedding_model": args.embedding_model,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "instances_per_skill": INSTANCES_PER_SKILL,
        "max_instances": MAX_INSTANCES,
        "token_budget_rule": "halt; never remove retrieved instances",
        "scorer_seed_rule": "original dataset index, paired across arms",
        "artifacts": artifacts,
    }


def write_record(
    destination: Any, records: Dict[int, Dict[str, Any]], record: Dict[str, Any]
) -> None:
    destination.write(json.dumps(record, ensure_ascii=True) + "\n")
    destination.flush()
    records[int(record["index"])] = record


def compact_jsonl(
    output: Path, records: Mapping[int, Dict[str, Any]], indices: Sequence[int]
) -> None:
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w") as destination:
        for index in indices:
            destination.write(json.dumps(records[index], ensure_ascii=True) + "\n")
    temporary.replace(output)


def tidy_route(record: Mapping[str, Any]) -> Dict[str, Any]:
    groups = record["route"]["groups"]
    reasons = Counter(
        group["dropped_reason"] for group in groups if group["dropped_reason"]
    )
    return {
        "route_parse_error": record["route"]["parse_error"] is not None,
        "retained_instances": int(record["route"]["retained_instance_count"]),
        "skill_below_threshold_groups": int(reasons["skill_below_threshold"]),
        "context_below_threshold_groups": int(reasons["context_below_threshold"]),
        "total_limit_groups": int(reasons["total_limit"]),
        "memory_input_tokens": int(record["route"]["memory_input_tokens"]),
        "token_headroom": int(record["route"]["token_headroom"]),
    }


def paired_indices(channel: str) -> Tuple[str, List[int], Optional[Path]]:
    if channel == "ood":
        rows = len(pd.read_parquet(DATA_ROOT / "val.parquet", columns=["prompt"]))
        if rows != 1218:
            raise GateFailure(f"Expected 1218 OOD rows, observed {rows}")
        return "val", list(range(rows)), None
    if channel == "id":
        manifest = pd.read_parquet(ID_MANIFEST_PATH)
        indices = sorted(int(value) for value in manifest["index"])
        if len(indices) != 300 or len(set(indices)) != 300:
            raise GateFailure("Frozen F2 ID manifest must contain 300 unique rows")
        return "test", indices, ID_MANIFEST_PATH
    raise ValueError(channel)


def run_paired(args: argparse.Namespace) -> Dict[str, Any]:
    require_inputs()
    require_frozen_args(args)
    split, indices, manifest_path = paired_indices(args.channel)
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    dataset = pd.read_parquet(DATA_ROOT / f"{split}.parquet")
    output = OUTPUT_DIR / f"f2_{args.channel}.jsonl"
    summary_path = output.with_suffix(".summary.json")
    run_path = output.with_suffix(output.suffix + ".run.json")
    metadata = run_metadata(
        args, "F2_paired", args.channel, split, indices, manifest_path
    )
    run_hash = fingerprint(metadata)
    if run_path.is_file():
        if json.loads(run_path.read_text()) != metadata:
            raise GateFailure(f"Cannot resume {output}: run metadata differs")
    else:
        atomic_json(run_path, metadata)
    records = load_jsonl(output)
    validate_resume_records(records, indices, run_hash, ("zero_shot", "skill_memory"))
    pending = [
        index
        for index in indices
        if index not in records or records[index].get("endpoint_error")
    ]
    provider = F2Provider(args, bench) if pending else None

    def generate(index: int) -> Tuple[int, Dict[str, Any]]:
        sample = native(dataset.iloc[index].to_dict())
        return index, provider.generate_pair(sample, index)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with output.open("a") as destination, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {executor.submit(generate, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            sample = native(dataset.iloc[index].to_dict())
            try:
                _, generated = future.result()
                arms = {}
                for arm in ("zero_shot", "skill_memory"):
                    metrics = bench.score_response(
                        scorer,
                        2,
                        generated["arms"][arm]["response"],
                        bench.get_ground_truth(sample),
                        index,
                    )
                    arms[arm] = {**generated["arms"][arm], **metrics}
                record = {
                    "index": index,
                    "run_fingerprint": run_hash,
                    "route": generated["route"],
                    "zero_shot_input_tokens": generated["zero_shot_input_tokens"],
                    "arms": arms,
                }
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
            write_record(destination, records, record)
    compact_jsonl(output, records, indices)
    selected = [records[index] for index in indices]
    errors = [record for record in selected if record.get("endpoint_error")]
    if errors:
        failure = {
            "task": "F2_paired",
            "channel": args.channel,
            "samples": len(indices),
            "endpoint_errors": len(errors),
            "status": "FAIL",
            "results": str(output),
        }
        atomic_json(summary_path, failure)
        raise GateFailure(
            f"F2 {args.channel} halted with {len(errors)} generation errors"
        )
    tidy_rows = []
    for record in selected:
        index = int(record["index"])
        ground_truth = native(dataset.iloc[index]["reward_model"])["ground_truth"]
        row = {
            "index": index,
            "task_name": str(ground_truth["task_name"]),
            "robot_count": sum(
                robot is not None for robot in ground_truth["robots"].values()
            ),
            "zero_shot_input_tokens": int(record["zero_shot_input_tokens"]),
            **tidy_route(record),
        }
        for arm in ("zero_shot", "skill_memory"):
            row[f"{arm}_task_score"] = float(record["arms"][arm]["task_score"])
            row[f"{arm}_format_score"] = float(record["arms"][arm]["format_score"])
        tidy_rows.append(row)
    tidy = pd.DataFrame(tidy_rows).sort_values("index")
    parquet_path = output.with_suffix(".parquet")
    tidy.to_parquet(parquet_path, index=False)
    summary = {
        "task": "F2_paired",
        "channel": args.channel,
        "samples": len(tidy),
        "overall": pair_summary(tidy, "zero_shot", "skill_memory"),
        "by_task": {
            str(task): pair_summary(group, "zero_shot", "skill_memory")
            for task, group in tidy.groupby("task_name", sort=True)
        },
        "by_robot_count": {
            str(count): pair_summary(group, "zero_shot", "skill_memory")
            for count, group in tidy.groupby("robot_count", sort=True)
        },
        "retained_instances": retained_diagnostics(tidy),
        "token_budget": {
            "max_model_len": CONTEXT_LENGTH,
            "max_output_tokens": args.max_tokens,
            "maximum_zero_shot_input_tokens": int(tidy["zero_shot_input_tokens"].max()),
            "maximum_memory_input_tokens": int(tidy["memory_input_tokens"].max()),
            "minimum_memory_headroom": int(tidy["token_headroom"].min()),
            "truncated_rows": 0,
            "removed_instances": 0,
        },
        "endpoint_errors": 0,
        "status": "PASS",
        "results": str(output),
        "parquet": str(parquet_path),
    }
    atomic_json(summary_path, summary)
    return summary


def parse_json_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return list(json.loads(value))
    return list(value)


def seeded_cap(indices: Sequence[int], offset: int) -> List[int]:
    values = sorted(set(int(index) for index in indices))
    if len(values) <= C_PRIME_ROWS:
        return values
    return sorted(random.Random(SEED + offset).sample(values, C_PRIME_ROWS))


def covers_full_signature(instance_units: Sequence[str], test_signature: str) -> bool:
    test_units = list(json.loads(test_signature)["ordered_units"])
    if not test_units or len(instance_units) < len(test_units):
        return False
    width = len(test_units)
    return any(
        list(instance_units[position : position + width]) == test_units
        for position in range(len(instance_units) - width + 1)
    )


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


def pool_audit(
    channel: str,
    index: int,
    full_signature: str,
    instances: Sequence[Mapping[str, Any]],
    test_tokens: Sequence[str],
) -> Dict[str, Any]:
    full_cover_ids = [
        str(instance["instance_id"])
        for instance in instances
        if covers_full_signature(instance.get("ordered_units", []), full_signature)
    ]
    distances = [
        (
            levenshtein(demo_tokens(instance["demo"]), test_tokens),
            str(instance["instance_id"]),
        )
        for instance in instances
    ]
    overlaps = [
        (
            ngram_jaccard(demo_tokens(instance["demo"]), test_tokens),
            str(instance["instance_id"]),
        )
        for instance in instances
    ]
    nearest_distance = min(distances) if distances else (None, None)
    maximum_overlap = max(overlaps) if overlaps else (None, None)
    return {
        "channel": channel,
        "index": index,
        "candidate_instances": len(instances),
        "full_signature_cover_count": len(full_cover_ids),
        "full_signature_cover_ids": json.dumps(full_cover_ids, separators=(",", ":")),
        "min_plan_edit_distance": nearest_distance[0],
        "nearest_edit_instance_id": nearest_distance[1],
        "max_plan_trigram_jaccard": maximum_overlap[0],
        "nearest_trigram_instance_id": maximum_overlap[1],
    }


def prepare_cprime() -> Dict[str, Any]:
    require_inputs()
    with M0_INSTANCES_PATH.open() as source:
        instances = [json.loads(line) for line in source if line.strip()]
    census = pd.read_parquet(CENSUS_PATH)
    train = census[census["split"] == "train"].copy()
    test = census[census["split"] == "test"].copy()
    if len(train) != 7196 or len(test) != 1800:
        raise GateFailure(
            f"C-prime census mismatch: train={len(train)}, test={len(test)}"
        )
    train["unit_kinds_value"] = train["unit_kinds"].map(parse_json_list)
    test["unit_kinds_value"] = test["unit_kinds"].map(parse_json_list)
    test["unit_count"] = test["unit_kinds_value"].map(len)
    train_by_index = train.set_index("index")
    test_by_index = test.set_index("index")
    single_instances = [
        instance
        for instance in instances
        if len(instance.get("ordered_units", [])) == 1
        and len(instance.get("unit_kinds", [])) == 1
    ]
    unit_counts = Counter(
        str(instance["unit_kinds"][0]) for instance in single_instances
    )
    covered_units = {
        kind for kind, count in unit_counts.items() if count >= UNIT_MIN_INSTANCES
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
    productivity_candidates = [
        int(row["index"])
        for _, row in multi_unit.iterrows()
        if set(row["unit_kinds_value"]) <= covered_units
    ]
    selected = {
        "instance": seeded_cap(instance_candidates, 1),
        "productivity": seeded_cap(productivity_candidates, 2),
    }
    short = {
        channel: len(indices)
        for channel, indices in selected.items()
        if len(indices) != C_PRIME_ROWS
    }
    if short:
        raise GateFailure(f"F2 C-prime cannot form frozen channels: {short}")
    dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    manifest_rows = []
    audit_rows = []
    for channel, indices in selected.items():
        for index in indices:
            row = test_by_index.loc[index]
            pool = (
                task_instances[str(row["task_name"])]
                if channel == "instance"
                else single_instances
            )
            instance_ids = sorted(str(item["instance_id"]) for item in pool)
            source_indices = sorted({int(item["source_train_index"]) for item in pool})
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
            if channel == "productivity":
                ground_truth = native(dataset.iloc[index]["reward_model"])[
                    "ground_truth"
                ]
                audit_rows.append(
                    pool_audit(
                        channel,
                        index,
                        str(row["full_signature"]),
                        pool,
                        grounded_plan_tokens(ground_truth),
                    )
                )
    manifest = pd.DataFrame(manifest_rows).sort_values(["channel", "index"])
    audit = pd.DataFrame(audit_rows).sort_values(["channel", "index"])
    strict_cover_rows = int((audit["full_signature_cover_count"] > 0).sum())
    exact_plan_match_rows = int((audit["min_plan_edit_distance"] == 0).sum())
    if strict_cover_rows or exact_plan_match_rows:
        raise GateFailure(
            "F2 C-prime productivity leakage: "
            f"signature_rows={strict_cover_rows}, "
            f"exact_plan_rows={exact_plan_match_rows}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(C_PRIME_MANIFEST_PATH, index=False)
    manifest.to_csv(C_PRIME_MANIFEST_PATH.with_suffix(".csv"), index=False)
    audit.to_parquet(C_PRIME_AUDIT_PATH, index=False)
    audit.to_csv(C_PRIME_AUDIT_PATH.with_suffix(".csv"), index=False)
    summary = {
        "task": "F2_C-prime_manifest",
        "seed": SEED,
        "channels": list(C_PRIME_CHANNELS),
        "systematicity": "permanently excluded as grammar-impossible",
        "channel_rows": C_PRIME_ROWS,
        "unit_min_instances": UNIT_MIN_INSTANCES,
        "coverage": {
            "single_unit_instances_by_kind": dict(unit_counts),
            "covered_unit_kinds": sorted(covered_units),
            "productivity_candidates_before_cap": len(productivity_candidates),
        },
        "by_channel": {
            channel: {
                "samples": len(group),
                "allowed_instances_min": int(group["allowed_instance_count"].min()),
                "allowed_instances_max": int(group["allowed_instance_count"].max()),
            }
            for channel, group in manifest.groupby("channel", sort=True)
        },
        "leakage_gate": {
            "status": "PASS",
            "productivity_full_signature_cover_rows": strict_cover_rows,
            "productivity_exact_plan_match_rows": exact_plan_match_rows,
            "productivity_min_plan_edit_distance": int(
                audit["min_plan_edit_distance"].min()
            ),
            "productivity_max_plan_trigram_jaccard": float(
                audit["max_plan_trigram_jaccard"].max()
            ),
        },
        "manifest": str(C_PRIME_MANIFEST_PATH),
        "audit": str(C_PRIME_AUDIT_PATH),
    }
    atomic_json(C_PRIME_MANIFEST_PATH.with_suffix(".summary.json"), summary)
    return summary


def cprime_preflight_metadata(
    args: argparse.Namespace, channel: str, indices: Sequence[int]
) -> Dict[str, Any]:
    metadata = run_metadata(
        args,
        "F2_C-prime_preflight",
        channel,
        "test",
        indices,
        C_PRIME_MANIFEST_PATH,
    )
    metadata.update(
        {
            "clarification": CLARIFICATION,
            "flat_builder": FLAT_BUILDER,
            "token_match_criterion": "per-row",
            "token_tolerance": TOKEN_TOLERANCE,
            "flat_candidate_limit": "entire allowed pool",
            "arm_generation": "none; router output and token accounting only",
        }
    )
    return metadata


def validate_preflight_records(
    records: Mapping[int, Mapping[str, Any]],
    indices: Sequence[int],
    run_hash: str,
    channel: str,
) -> None:
    expected = set(indices)
    unexpected = sorted(set(records) - expected)
    if unexpected:
        raise GateFailure(
            f"C-prime preflight contains unexpected indices: {unexpected[:10]}"
        )
    for index, record in records.items():
        if record.get("run_fingerprint") != run_hash:
            raise GateFailure(f"Preflight fingerprint mismatch at index {index}")
        if record.get("channel") != channel:
            raise GateFailure(f"Preflight channel mismatch at index {index}")
        if not isinstance(record.get("route"), dict):
            raise GateFailure(f"Preflight route is malformed at index {index}")
        hashes = record.get("prompt_sha256")
        if not isinstance(hashes, dict) or not {
            "zero_shot",
            "skill_memory",
        } <= set(hashes):
            raise GateFailure(f"Preflight prompt hashes are malformed at index {index}")
        if record.get("dropped"):
            if not record.get("drop_reason"):
                raise GateFailure(f"Preflight drop has no reason at index {index}")
            continue
        flat = record.get("flat")
        if not isinstance(flat, dict) or "flat_memory" not in hashes:
            raise GateFailure(f"Preflight flat control is malformed at index {index}")
        difference = float(flat.get("relative_difference", math.inf))
        if difference > TOKEN_TOLERANCE:
            raise GateFailure(
                f"Preflight row is outside tolerance at index {index}: {difference}"
            )


def reusable_cprime_routes(
    channel: str, indices: Sequence[int]
) -> Dict[int, Dict[str, Any]]:
    output = OUTPUT_DIR / f"f2_cprime_{channel}.jsonl"
    run_path = output.with_suffix(output.suffix + ".run.json")
    if not output.is_file() and not run_path.is_file():
        return {}
    if not output.is_file() or not run_path.is_file():
        raise GateFailure(f"Incomplete legacy C-prime checkpoint at {output}")
    metadata = json.loads(run_path.read_text())
    if metadata.get("task") != "F2_C-prime" or metadata.get("channel") != channel:
        raise GateFailure(f"Unexpected C-prime checkpoint metadata at {run_path}")
    run_hash = fingerprint(metadata)
    records = load_jsonl(output)
    unexpected = sorted(set(records) - set(indices))
    if unexpected:
        raise GateFailure(
            f"Legacy C-prime checkpoint has unexpected indices: {unexpected[:10]}"
        )
    routes = {}
    for index, record in records.items():
        if record.get("run_fingerprint") != run_hash:
            raise GateFailure(f"Legacy checkpoint fingerprint mismatch at {index}")
        if record.get("endpoint_error"):
            continue
        if not isinstance(record.get("route"), dict):
            raise GateFailure(f"Legacy checkpoint route is malformed at {index}")
        routes[index] = record["route"]
    return routes


def preflight_cprime_channel(
    args: argparse.Namespace,
    channel: str,
    manifest: pd.DataFrame,
    dataset: pd.DataFrame,
    bench: Any,
) -> Dict[str, Any]:
    channel_manifest = manifest[manifest["channel"] == channel].sort_values("index")
    indices = [int(value) for value in channel_manifest["index"]]
    if len(indices) != C_PRIME_ROWS or len(set(indices)) != C_PRIME_ROWS:
        raise GateFailure(
            f"F2 C-prime {channel} preflight requires {C_PRIME_ROWS} unique rows"
        )
    rows_by_index = {
        int(row["index"]): row for row in channel_manifest.to_dict("records")
    }
    output = cprime_preflight_path(channel)
    summary_path = output.with_suffix(".summary.json")
    run_path = output.with_suffix(output.suffix + ".run.json")
    metadata = cprime_preflight_metadata(args, channel, indices)
    run_hash = fingerprint(metadata)
    if run_path.is_file():
        if json.loads(run_path.read_text()) != metadata:
            raise GateFailure(f"Cannot resume {output}: preflight metadata differs")
    else:
        atomic_json(run_path, metadata)
    records = load_jsonl(output)
    validate_preflight_records(records, indices, run_hash, channel)
    pending = [index for index in indices if index not in records]
    reusable_routes = reusable_cprime_routes(channel, indices)
    provider = F2Provider(args, bench) if pending else None

    def inspect(index: int) -> Tuple[int, Dict[str, Any]]:
        sample = native(dataset.iloc[index].to_dict())
        allowed = json.loads(rows_by_index[index]["allowed_instance_ids"])
        result = provider.preflight_cprime(
            sample,
            index,
            allowed,
            reused_route=reusable_routes.get(index),
        )
        return index, result

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with output.open("a") as destination, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {executor.submit(inspect, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                _, inspected = future.result()
            except Exception:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            record = {
                "index": index,
                "channel": channel,
                "run_fingerprint": run_hash,
                **inspected,
            }
            write_record(destination, records, record)
    compact_jsonl(output, records, indices)
    selected = [records[index] for index in indices]
    dropped = [record for record in selected if record.get("dropped")]
    retained = [record for record in selected if not record.get("dropped")]
    violations = [
        record
        for record in retained
        if float(record["flat"]["relative_difference"]) > TOKEN_TOLERANCE
    ]
    if violations:
        raise GateFailure(
            f"C-prime {channel} preflight has {len(violations)} token violations"
        )
    summary = {
        "task": "F2_C-prime_preflight",
        "channel": channel,
        "clarification": CLARIFICATION,
        "flat_builder": FLAT_BUILDER,
        "token_match_criterion": "per-row",
        "token_tolerance": TOKEN_TOLERANCE,
        "planned_rows": len(indices),
        "retained_rows": len(retained),
        "symmetric_drop_count": len(dropped),
        "symmetric_drop_indices": [int(record["index"]) for record in dropped],
        "in_band_rows": len(retained),
        "violations": 0,
        "maximum_relative_difference": max(
            (float(record["flat"]["relative_difference"]) for record in retained),
            default=0.0,
        ),
        "legacy_routes_reused": sum(
            bool(record.get("legacy_route_reused")) for record in selected
        ),
        "arm_generations": 0,
        "status": "PASS",
        "results": str(output),
    }
    atomic_json(summary_path, summary)
    return summary


def run_cprime_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    require_inputs()
    require_frozen_args(args)
    if not C_PRIME_MANIFEST_PATH.is_file():
        prepare_cprime()
    manifest = pd.read_parquet(C_PRIME_MANIFEST_PATH)
    dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    bench = load_bench()
    summaries = {
        channel: preflight_cprime_channel(args, channel, manifest, dataset, bench)
        for channel in C_PRIME_CHANNELS
    }
    result = {
        "task": "F2_C-prime_preflight",
        "clarification": CLARIFICATION,
        "flat_builder": FLAT_BUILDER,
        "token_match_criterion": "per-row",
        "token_tolerance": TOKEN_TOLERANCE,
        "channels": summaries,
        "artifacts": {
            channel: {
                "results": str(cprime_preflight_path(channel)),
                "results_sha256": file_sha256(cprime_preflight_path(channel)),
                "summary": str(
                    cprime_preflight_path(channel).with_suffix(".summary.json")
                ),
                "summary_sha256": file_sha256(
                    cprime_preflight_path(channel).with_suffix(".summary.json")
                ),
            }
            for channel in C_PRIME_CHANNELS
        },
        "status": "PASS",
    }
    atomic_json(C_PRIME_PREFLIGHT_SUMMARY_PATH, result)
    return result


def load_certified_cprime_preflight(
    args: argparse.Namespace,
    channel: str,
    manifest: pd.DataFrame,
) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    if not C_PRIME_PREFLIGHT_SUMMARY_PATH.is_file():
        raise GateFailure(
            "C-prime generation requires the completed Amendment 3.1 preflight"
        )
    certificate = json.loads(C_PRIME_PREFLIGHT_SUMMARY_PATH.read_text())
    if (
        certificate.get("status") != "PASS"
        or certificate.get("clarification") != CLARIFICATION
        or certificate.get("flat_builder") != FLAT_BUILDER
        or certificate.get("token_match_criterion") != "per-row"
        or certificate.get("token_tolerance") != TOKEN_TOLERANCE
    ):
        raise GateFailure("C-prime preflight certificate is not binding")
    requested_records = None
    requested_drops = None
    for candidate_channel in C_PRIME_CHANNELS:
        indices = sorted(
            int(value)
            for value in manifest.loc[manifest["channel"] == candidate_channel, "index"]
        )
        if len(indices) != C_PRIME_ROWS or len(set(indices)) != C_PRIME_ROWS:
            raise GateFailure(
                f"C-prime {candidate_channel} manifest changed after preflight"
            )
        artifact = certificate.get("artifacts", {}).get(candidate_channel, {})
        path = cprime_preflight_path(candidate_channel)
        summary_path = path.with_suffix(".summary.json")
        if (
            not path.is_file()
            or not summary_path.is_file()
            or file_sha256(path) != artifact.get("results_sha256")
            or file_sha256(summary_path) != artifact.get("summary_sha256")
        ):
            raise GateFailure(
                f"C-prime {candidate_channel} preflight artifact hash mismatch"
            )
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "PASS" or summary.get("violations") != 0:
            raise GateFailure(f"C-prime {candidate_channel} preflight did not pass")
        run_path = path.with_suffix(path.suffix + ".run.json")
        if not run_path.is_file():
            raise GateFailure(
                f"C-prime {candidate_channel} preflight metadata is missing"
            )
        observed_metadata = json.loads(run_path.read_text())
        expected_metadata = cprime_preflight_metadata(args, candidate_channel, indices)
        if observed_metadata != expected_metadata:
            raise GateFailure(
                f"C-prime {candidate_channel} preflight metadata is stale"
            )
        run_hash = fingerprint(observed_metadata)
        records = load_jsonl(path)
        validate_preflight_records(records, indices, run_hash, candidate_channel)
        if set(records) != set(indices):
            raise GateFailure(f"C-prime {candidate_channel} preflight is incomplete")
        dropped = sorted(
            index for index, record in records.items() if record.get("dropped")
        )
        if dropped != sorted(int(value) for value in summary["symmetric_drop_indices"]):
            raise GateFailure(
                f"C-prime {candidate_channel} symmetric-drop audit mismatch"
            )
        if candidate_channel == channel:
            requested_records = records
            requested_drops = dropped
    if requested_records is None or requested_drops is None:
        raise GateFailure(f"Unknown C-prime channel: {channel}")
    return requested_records, requested_drops


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


def migrate_cprime_checkpoint(
    channel: str,
    output: Path,
    run_path: Path,
    metadata: Mapping[str, Any],
    active_indices: Sequence[int],
    preflight: Mapping[int, Mapping[str, Any]],
    dataset: pd.DataFrame,
    bench: Any,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    run_hash = fingerprint(metadata)
    migration_path = OUTPUT_DIR / f"f2_cprime_{channel}.migration.json"
    if not run_path.is_file() and not output.is_file():
        atomic_json(run_path, metadata)
        migration = {
            "clarification": CLARIFICATION,
            "legacy_records": 0,
            "rows_reusing_zero_shot_and_skill": 0,
            "flat_outputs_discarded": 0,
            "new_run_fingerprint": run_hash,
            "status": "PASS",
        }
        atomic_json(migration_path, migration)
        return {}, migration
    if not run_path.is_file() or not output.is_file():
        raise GateFailure(f"Incomplete C-prime checkpoint at {output}")
    observed_metadata = json.loads(run_path.read_text())
    if observed_metadata == metadata:
        migration = (
            json.loads(migration_path.read_text())
            if migration_path.is_file()
            else {
                "clarification": CLARIFICATION,
                "new_run_fingerprint": run_hash,
                "status": "PASS",
            }
        )
        return load_cprime_checkpoint(output), migration
    interrupted_records = load_cprime_checkpoint(output)
    if interrupted_records and {
        record.get("run_fingerprint") for record in interrupted_records.values()
    } == {run_hash}:
        atomic_json(run_path, metadata)
        migration = {
            "clarification": CLARIFICATION,
            "legacy_records": len(interrupted_records),
            "rows_reusing_zero_shot_and_skill": sum(
                "checkpoint_reuse" in record for record in interrupted_records.values()
            ),
            "flat_outputs_discarded": sum(
                "checkpoint_reuse" in record for record in interrupted_records.values()
            ),
            "new_run_fingerprint": run_hash,
            "recovered_interrupted_commit": True,
            "status": "PASS",
        }
        atomic_json(migration_path, migration)
        return interrupted_records, migration
    if (
        observed_metadata.get("task") != "F2_C-prime"
        or observed_metadata.get("channel") != channel
        or observed_metadata.get("token_tolerance") != TOKEN_TOLERANCE
        or observed_metadata.get("clarification") is not None
    ):
        raise GateFailure(f"Cannot migrate unexpected checkpoint at {output}")
    legacy_hash = fingerprint(observed_metadata)
    legacy_records = load_jsonl(output)
    migrated: Dict[int, Dict[str, Any]] = {}
    discarded_flat = 0
    reused_rows = 0
    for index, record in legacy_records.items():
        if record.get("run_fingerprint") != legacy_hash:
            raise GateFailure(f"Legacy fingerprint mismatch at index {index}")
        if index not in active_indices or record.get("endpoint_error"):
            continue
        arms = record.get("arms")
        route = record.get("route")
        if not isinstance(arms, dict) or not isinstance(route, dict):
            raise GateFailure(f"Malformed legacy checkpoint at index {index}")
        if not {"zero_shot", "skill_memory"} <= set(arms):
            continue
        sample = native(dataset.iloc[index].to_dict())
        zero_messages = bench.get_messages(sample)
        skill_messages = add_memory_to_messages(
            bench.get_messages(sample), route.get("memory_prompt", "")
        )
        expected_hashes = preflight[index]["prompt_sha256"]
        if (
            messages_sha256(zero_messages) != expected_hashes["zero_shot"]
            or messages_sha256(skill_messages) != expected_hashes["skill_memory"]
        ):
            continue
        migrated[index] = {
            "index": index,
            "run_fingerprint": run_hash,
            "route": preflight[index]["route"],
            "flat": preflight[index]["flat"],
            "zero_shot_input_tokens": preflight[index]["zero_shot_input_tokens"],
            "arms": {
                "zero_shot": arms["zero_shot"],
                "skill_memory": arms["skill_memory"],
            },
            "checkpoint_reuse": {
                "prior_run_fingerprint": legacy_hash,
                "prompt_sha256": {
                    key: expected_hashes[key] for key in ("zero_shot", "skill_memory")
                },
                "reused_arms": ["zero_shot", "skill_memory"],
                "discarded_arms": ["flat_memory"],
            },
        }
        reused_rows += 1
        discarded_flat += int("flat_memory" in arms)
    write_jsonl_snapshot(output, migrated, active_indices)
    atomic_json(run_path, metadata)
    migration = {
        "clarification": CLARIFICATION,
        "legacy_run_fingerprint": legacy_hash,
        "legacy_records": len(legacy_records),
        "rows_reusing_zero_shot_and_skill": reused_rows,
        "flat_outputs_discarded": discarded_flat,
        "new_run_fingerprint": run_hash,
        "status": "PASS",
    }
    atomic_json(migration_path, migration)
    return migrated, migration


def validate_cprime_resume_records(
    records: Mapping[int, Mapping[str, Any]],
    indices: Sequence[int],
    run_hash: str,
    preflight: Mapping[int, Mapping[str, Any]],
) -> None:
    expected = set(indices)
    unexpected = sorted(set(records) - expected)
    if unexpected:
        raise GateFailure(f"C-prime resume has unexpected indices: {unexpected[:10]}")
    valid_arms = {"zero_shot", "skill_memory", "flat_memory"}
    for index, record in records.items():
        if record.get("run_fingerprint") != run_hash:
            raise GateFailure(f"C-prime resume fingerprint mismatch at {index}")
        if record.get("endpoint_error"):
            continue
        if record.get("route") != preflight[index]["route"]:
            raise GateFailure(f"C-prime route differs from preflight at {index}")
        if record.get("flat") != preflight[index]["flat"]:
            raise GateFailure(f"C-prime flat prompt differs from preflight at {index}")
        arms = record.get("arms")
        if not isinstance(arms, dict) or set(arms) - valid_arms:
            raise GateFailure(f"C-prime arms are malformed at {index}")
        for arm, value in arms.items():
            if not {
                "response",
                "prompt_tokens",
                "completion_tokens",
                "task_score",
                "format_score",
            } <= set(value):
                raise GateFailure(f"C-prime arm is malformed at {index}:{arm}")


def run_cprime(args: argparse.Namespace) -> Dict[str, Any]:
    require_inputs()
    require_frozen_args(args)
    if not C_PRIME_MANIFEST_PATH.is_file():
        prepare_cprime()
    full_manifest = pd.read_parquet(C_PRIME_MANIFEST_PATH)
    manifest = full_manifest[full_manifest["channel"] == args.channel].sort_values(
        "index"
    )
    planned_indices = [int(value) for value in manifest["index"]]
    if (
        len(planned_indices) != C_PRIME_ROWS
        or len(set(planned_indices)) != C_PRIME_ROWS
    ):
        raise GateFailure(f"F2 C-prime {args.channel} must contain {C_PRIME_ROWS} rows")
    preflight, dropped_indices = load_certified_cprime_preflight(
        args, args.channel, full_manifest
    )
    indices = [index for index in planned_indices if index not in dropped_indices]
    if not indices:
        raise GateFailure(f"F2 C-prime {args.channel} has no rows after preflight")
    rows_by_index = {int(row["index"]): row for row in manifest.to_dict("records")}
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    dataset = pd.read_parquet(DATA_ROOT / "test.parquet")
    output = OUTPUT_DIR / f"f2_cprime_{args.channel}.jsonl"
    summary_path = output.with_suffix(".summary.json")
    run_path = output.with_suffix(output.suffix + ".run.json")
    metadata = run_metadata(
        args,
        "F2_C-prime",
        args.channel,
        "test",
        indices,
        C_PRIME_MANIFEST_PATH,
    )
    metadata.update(
        {
            "clarification": CLARIFICATION,
            "flat_builder": FLAT_BUILDER,
            "token_match_criterion": "per-row",
            "token_tolerance": TOKEN_TOLERANCE,
            "preflight_certificate_sha256": file_sha256(C_PRIME_PREFLIGHT_SUMMARY_PATH),
            "channel_preflight_sha256": file_sha256(
                cprime_preflight_path(args.channel)
            ),
            "planned_indices": planned_indices,
            "symmetric_drop_indices": dropped_indices,
        }
    )
    run_hash = fingerprint(metadata)
    records, migration = migrate_cprime_checkpoint(
        args.channel,
        output,
        run_path,
        metadata,
        indices,
        preflight,
        dataset,
        bench,
    )
    validate_cprime_resume_records(
        records,
        indices,
        run_hash,
        preflight,
    )
    required_arms = ("zero_shot", "skill_memory", "flat_memory")
    pending = {}
    for index in indices:
        record = records.get(index, {})
        completed_arms = record.get("arms", {})
        missing_arms = [arm for arm in required_arms if arm not in completed_arms]
        if record.get("endpoint_error") or missing_arms:
            pending[index] = missing_arms or list(required_arms)
    provider = F2Provider(args, bench) if pending else None

    def generate(index: int, arms: Sequence[str]) -> Tuple[int, Dict[str, Any]]:
        sample = native(dataset.iloc[index].to_dict())
        return index, provider.generate_cprime_from_preflight(
            sample, index, preflight[index], arms
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate, index, arms): index
            for index, arms in pending.items()
        }
        for future in as_completed(futures):
            index = futures[future]
            sample = native(dataset.iloc[index].to_dict())
            try:
                _, generated = future.result()
                arms = dict(records.get(index, {}).get("arms", {}))
                for arm, generated_arm in generated.items():
                    metrics = bench.score_response(
                        scorer,
                        2,
                        generated_arm["response"],
                        bench.get_ground_truth(sample),
                        index,
                    )
                    arms[arm] = {**generated_arm, **metrics}
                record = {
                    "index": index,
                    "run_fingerprint": run_hash,
                    "route": preflight[index]["route"],
                    "flat": preflight[index]["flat"],
                    "zero_shot_input_tokens": preflight[index][
                        "zero_shot_input_tokens"
                    ],
                    "arms": arms,
                }
                if "checkpoint_reuse" in records.get(index, {}):
                    record["checkpoint_reuse"] = records[index]["checkpoint_reuse"]
            except GateFailure:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            except Exception as error:
                record = dict(records.get(index, {}))
                record.update(
                    {
                        "index": index,
                        "run_fingerprint": run_hash,
                        "endpoint_error": repr(error),
                    }
                )
            records[index] = record
            write_jsonl_snapshot(output, records, indices)
    compact_jsonl(output, records, indices)
    selected = [records[index] for index in indices]
    errors = [record for record in selected if record.get("endpoint_error")]
    if errors:
        failure = {
            "task": "F2_C-prime",
            "channel": args.channel,
            "samples": len(indices),
            "planned_samples": len(planned_indices),
            "symmetric_drop_count": len(dropped_indices),
            "generation_errors": len(errors),
            "status": "FAIL",
            "results": str(output),
        }
        atomic_json(summary_path, failure)
        raise GateFailure(
            f"F2 C-prime {args.channel} halted with {len(errors)} generation errors"
        )
    tidy_rows = []
    for record in selected:
        index = int(record["index"])
        manifest_row = rows_by_index[index]
        row = {
            "index": index,
            "task_name": str(manifest_row["task_name"]),
            "unit_count": int(manifest_row["unit_count"]),
            "allowed_instance_count": int(manifest_row["allowed_instance_count"]),
            "zero_shot_input_tokens": int(record["zero_shot_input_tokens"]),
            "flat_input_tokens": int(record["flat"]["input_tokens"]),
            "flat_k": int(record["flat"]["flat_k"]),
            "flat_row_k": int(record["flat"].get("flat_row_k", 0)),
            "flat_truncated": bool(record["flat"]["truncated"]),
            "flat_token_relative_difference": float(
                record["flat"]["relative_difference"]
            ),
            **tidy_route(record),
        }
        for arm in ("zero_shot", "skill_memory", "flat_memory"):
            row[f"{arm}_task_score"] = float(record["arms"][arm]["task_score"])
            row[f"{arm}_format_score"] = float(record["arms"][arm]["format_score"])
        tidy_rows.append(row)
    tidy = pd.DataFrame(tidy_rows).sort_values("index")
    parquet_path = output.with_suffix(".parquet")
    tidy.to_parquet(parquet_path, index=False)
    summary = {
        "task": "F2_C-prime",
        "channel": args.channel,
        "clarification": CLARIFICATION,
        "flat_builder": FLAT_BUILDER,
        "token_match_criterion": "per-row",
        "planned_samples": len(planned_indices),
        "samples": len(tidy),
        "symmetric_drops": {
            "count": len(dropped_indices),
            "indices": dropped_indices,
        },
        "checkpoint_reuse": migration,
        "arms": {
            arm: arm_summary(tidy, arm)
            for arm in ("zero_shot", "skill_memory", "flat_memory")
        },
        "comparisons": {
            "zero_shot_to_skill": pair_summary(tidy, "zero_shot", "skill_memory"),
            "zero_shot_to_flat": pair_summary(tidy, "zero_shot", "flat_memory"),
            "flat_to_skill": pair_summary(tidy, "flat_memory", "skill_memory"),
        },
        "by_task": {
            str(task): {
                "zero_shot_to_skill": pair_summary(group, "zero_shot", "skill_memory"),
                "flat_to_skill": pair_summary(group, "flat_memory", "skill_memory"),
            }
            for task, group in tidy.groupby("task_name", sort=True)
        },
        "retained_instances": retained_diagnostics(tidy),
        "flat_token_match": {
            "criterion": "per-row",
            "tolerance": TOKEN_TOLERANCE,
            "maximum_relative_difference": float(
                tidy["flat_token_relative_difference"].max()
            ),
            "mean_relative_difference": float(
                tidy["flat_token_relative_difference"].mean()
            ),
            "truncated_prompts": int(tidy["flat_truncated"].sum()),
            "violations": int(
                (tidy["flat_token_relative_difference"] > TOKEN_TOLERANCE).sum()
            ),
        },
        "token_budget": {
            "max_model_len": CONTEXT_LENGTH,
            "max_output_tokens": args.max_tokens,
            "maximum_zero_shot_input_tokens": int(tidy["zero_shot_input_tokens"].max()),
            "maximum_skill_input_tokens": int(tidy["memory_input_tokens"].max()),
            "maximum_flat_input_tokens": int(tidy["flat_input_tokens"].max()),
            "minimum_skill_headroom": int(tidy["token_headroom"].min()),
            "truncated_skill_rows": 0,
            "removed_instances": 0,
        },
        "endpoint_errors": 0,
        "status": "PASS",
        "results": str(output),
        "parquet": str(parquet_path),
    }
    atomic_json(summary_path, summary)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VIKI Amendment 3 F2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-cprime")
    paired = subparsers.add_parser("paired")
    paired.add_argument("--channel", choices=("ood", "id"), required=True)
    preflight = subparsers.add_parser("preflight-cprime")
    cprime = subparsers.add_parser("cprime")
    cprime.add_argument("--channel", choices=C_PRIME_CHANNELS, required=True)
    for subparser in (paired, preflight, cprime):
        subparser.add_argument("--base-url", dest="base_urls", action="append")
        subparser.add_argument("--api-key-env", default="OPENAI_API_KEY")
        subparser.add_argument("--model", default=SERVED_MODEL)
        subparser.add_argument("--max-tokens", type=int, default=MAX_OUTPUT_TOKENS)
        subparser.add_argument(
            "--subgoal-max-tokens", type=int, default=SUBGOAL_MAX_TOKENS
        )
        subparser.add_argument("--workers", type=int, default=8)
        subparser.add_argument("--max-retries", type=int, default=5)
        subparser.add_argument("--timeout", type=float, default=3600)
        subparser.add_argument("--embedding-model", default="all-mpnet-base-v2")
        subparser.add_argument("--embedding-device", default="cpu")
    args = parser.parse_args(argv)
    if args.command != "prepare-cprime" and args.base_urls is None:
        args.base_urls = ["http://127.0.0.1:8050/v1"]
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.command == "prepare-cprime":
            result = prepare_cprime()
        elif args.command == "paired":
            result = run_paired(args)
        elif args.command == "preflight-cprime":
            result = run_cprime_preflight(args)
        else:
            result = run_cprime(args)
    except GateFailure as error:
        print(f"GATE FAILURE: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
