#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

from habitat_llm.evaluation.viki_branch_conditions import (
    AvailabilityPredicate,
    derive_train_portable_assets,
    derive_train_vocabularies,
    discover_instruction_regions,
    get_instruction,
)
from habitat_llm.evaluation.viki_stage1_memory import (
    EXTRACTION_SYSTEM_PROMPT,
    Stage1ExtractionError,
    Stage1Instance,
    build_extraction_messages,
    build_instances,
    parse_extraction_response,
    to_native,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT.parent / "VIKI-R"
TRAIN_PATH = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/train.parquet"
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment2"
RAW_PATH = OUTPUT_DIR / "m0_extractions.jsonl"
CLUSTER_THRESHOLD = 0.90
SMOKE_SEED = 20260814
SMOKE_ROWS = 100
RETRIEVAL_THRESHOLD = 0.30


class M0GateFailure(RuntimeError):
    pass


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_train() -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    frame = pd.read_parquet(TRAIN_PATH, columns=["prompt", "reward_model"])
    if len(frame) != 7196:
        raise M0GateFailure(f"Expected 7,196 train rows, found {len(frame)}")
    rows = [to_native(row) for row in frame.to_dict("records")]
    return frame, rows


def build_predicate(rows: Sequence[Mapping[str, Any]]) -> AvailabilityPredicate:
    assets, locations = derive_train_vocabularies(rows)
    portable_assets = derive_train_portable_assets(rows)
    discovered_regions = discover_instruction_regions(
        (get_instruction(row) for row in rows), assets, locations
    )
    return AvailabilityPredicate(
        assets, locations | discovered_regions, portable_assets
    )


def train_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        digest.update(
            json.dumps(
                {
                    "index": index,
                    "prompt": to_native(row["prompt"]),
                    "reward_model": to_native(row["reward_model"]),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return digest.hexdigest()


def run_metadata(
    args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    return {
        "task": "M0_extraction",
        "train_path": str(TRAIN_PATH.resolve()),
        "train_rows": 7196,
        "train_prompt_reward_sha256": train_fingerprint(rows),
        "model": args.model,
        "model_revision": args.model_revision,
        "base_urls": list(args.base_urls),
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "workers": args.workers,
        "serving": {
            "vllm_version": "0.8.4",
            "transformers_version": "4.51.3",
            "dtype": "bfloat16",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.60,
            "max_num_seqs_per_server": 16,
            "generation_config": "vllm",
            "seed": 0,
        },
        "system_prompt": EXTRACTION_SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(
            EXTRACTION_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "scope": "individual skills only; no cooperation skills",
    }


def load_records(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.is_file():
        return {}
    records = {}
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record["train_index"])
            if index in records:
                raise M0GateFailure(f"Duplicate extraction row {index}")
            records[index] = record
    return records


def extract(args: argparse.Namespace) -> None:
    _, rows = load_train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / RAW_PATH.name
    run_path = output.with_suffix(output.suffix + ".run.json")
    metadata = run_metadata(args, rows)
    if args.resume:
        if not run_path.is_file() or json.loads(run_path.read_text()) != metadata:
            raise M0GateFailure("Cannot resume M0: run metadata differs")
        records = load_records(output)
    else:
        if output.exists() or run_path.exists():
            raise M0GateFailure("M0 output already exists; use --resume")
        atomic_json(run_path, metadata)
        records = {}

    clients = [
        OpenAI(
            api_key=os.environ.get(args.api_key_env, "EMPTY"),
            base_url=url,
            max_retries=args.max_retries,
        )
        for url in args.base_urls
    ]
    pending = [
        index
        for index in range(len(rows))
        if index not in records or records[index].get("endpoint_error")
    ]

    def generate(index: int) -> Tuple[int, Dict[str, Any]]:
        messages = build_extraction_messages(rows[index])
        completion = clients[index % len(clients)].chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0,
            max_tokens=args.max_tokens,
        )
        response = completion.choices[0].message.content or ""
        plan_length = len(rows[index]["reward_model"]["ground_truth"]["time_steps"])
        parse_error = None
        parsed_segments = None
        try:
            parsed_segments = [
                segment.to_dict()
                for segment in parse_extraction_response(response, plan_length)
            ]
        except Stage1ExtractionError as error:
            parse_error = str(error)
        usage = completion.usage
        return index, {
            "train_index": index,
            "messages": messages,
            "response": response,
            "segments": parsed_segments,
            "parse_error": parse_error,
            "prompt_tokens": None if usage is None else usage.prompt_tokens,
            "completion_tokens": None if usage is None else usage.completion_tokens,
        }

    mode = "a" if args.resume else "w"
    with output.open(mode) as destination, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {executor.submit(generate, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                _, record = future.result()
            except Exception as error:
                record = {
                    "train_index": index,
                    "messages": build_extraction_messages(rows[index]),
                    "endpoint_error": repr(error),
                }
            destination.write(json.dumps(record, ensure_ascii=True) + "\n")
            destination.flush()
            records[index] = record

    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w") as destination:
        for index in sorted(records):
            destination.write(json.dumps(records[index], ensure_ascii=True) + "\n")
    temporary.replace(output)
    errors = sum("endpoint_error" in record for record in records.values())
    invalid = sum(bool(record.get("parse_error")) for record in records.values())
    print(
        json.dumps(
            {
                "rows": len(records),
                "endpoint_errors": errors,
                "parse_exclusions": invalid,
                "output": str(output.resolve()),
            },
            indent=2,
        )
    )
    if errors:
        raise M0GateFailure(f"M0 extraction has {errors} endpoint errors")


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def cluster_names(
    instances: Sequence[Stage1Instance], embedding_model: Any
) -> Tuple[Dict[str, str], np.ndarray, Sequence[str]]:
    names = sorted({instance.raw_skill_name for instance in instances})
    embeddings = np.asarray(
        embedding_model.encode(
            names,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )
    union_find = UnionFind(len(names))
    block_size = 512
    for start in range(0, len(names), block_size):
        similarities = embeddings[start : start + block_size] @ embeddings.T
        for local_index, row in enumerate(similarities):
            left = start + local_index
            for right in np.flatnonzero(row >= CLUSTER_THRESHOLD):
                if int(right) > left:
                    union_find.union(left, int(right))
    components = defaultdict(list)
    for index in range(len(names)):
        components[union_find.find(index)].append(index)
    merge_map = {}
    for indices in components.values():
        component_embeddings = embeddings[indices]
        centrality = component_embeddings @ component_embeddings.T
        best_score = float(centrality.mean(axis=1).max())
        candidates = [
            indices[position]
            for position, score in enumerate(centrality.mean(axis=1))
            if abs(float(score) - best_score) <= 1e-12
        ]
        medoid = min(candidates, key=lambda index: names[index])
        for index in indices:
            merge_map[names[index]] = names[medoid]
    return merge_map, embeddings, names


def exact_quantiles(values: Sequence[int]) -> Dict[str, float]:
    array = np.asarray(values)
    return {
        "min": int(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


def retrieval_smoke(
    instances: Sequence[Stage1Instance], embedding_model: Any
) -> Dict[str, Any]:
    by_source = defaultdict(list)
    by_skill = defaultdict(list)
    for position, instance in enumerate(instances):
        by_source[instance.source_train_index].append(position)
        by_skill[instance.skill_name].append(position)
    eligible_sources = sorted(by_source)
    if len(eligible_sources) < SMOKE_ROWS:
        raise M0GateFailure("Fewer than 100 extracted source rows for smoke test")
    rng = random.Random(SMOKE_SEED)
    heldout_sources = set(rng.sample(eligible_sources, SMOKE_ROWS))
    skills = sorted(by_skill)
    skill_embeddings = np.asarray(
        embedding_model.encode(
            skills,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )
    raw_queries = [
        sorted(
            by_source[source], key=lambda position: instances[position].segment_index
        )[0]
        for source in sorted(heldout_sources)
    ]
    query_names = [instances[position].raw_skill_name for position in raw_queries]
    query_embeddings = np.asarray(
        embedding_model.encode(
            query_names,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )
    query_context_embeddings = np.asarray(
        embedding_model.encode(
            [instances[position].context for position in raw_queries],
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )
    context_candidates = [
        position
        for position, instance in enumerate(instances)
        if instance.source_train_index not in heldout_sources
    ]
    context_embeddings = np.asarray(
        embedding_model.encode(
            [instances[position].context for position in context_candidates],
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )
    candidate_embedding = dict(zip(context_candidates, context_embeddings))
    rows = []
    passes = 0
    for source, query_position, query_embedding, context_query in zip(
        sorted(heldout_sources),
        raw_queries,
        query_embeddings,
        query_context_embeddings,
    ):
        target = instances[query_position]
        similarities = skill_embeddings @ query_embedding
        best_position = int(np.argmax(similarities))
        selected_skill = skills[best_position]
        skill_similarity = float(similarities[best_position])
        candidates = [
            position
            for position in by_skill[selected_skill]
            if position in candidate_embedding
        ]
        selected_instance = None
        context_similarity = None
        if skill_similarity >= RETRIEVAL_THRESHOLD and candidates:
            selected_instance = max(
                candidates,
                key=lambda position: (
                    float(np.dot(context_query, candidate_embedding[position])),
                    -instances[position].source_train_index,
                    -instances[position].segment_index,
                ),
            )
            context_similarity = float(
                np.dot(context_query, candidate_embedding[selected_instance])
            )
        passed = (
            selected_instance is not None
            and selected_skill == target.skill_name
            and instances[selected_instance].skill_name == target.skill_name
        )
        passes += passed
        rows.append(
            {
                "source_train_index": source,
                "query_raw_skill": target.raw_skill_name,
                "target_skill": target.skill_name,
                "selected_skill": selected_skill,
                "skill_similarity": skill_similarity,
                "retrieved_instance_id": (
                    None
                    if selected_instance is None
                    else instances[selected_instance].instance_id
                ),
                "context_similarity": context_similarity,
                "passed": passed,
            }
        )
    return {
        "seed": SMOKE_SEED,
        "rows": SMOKE_ROWS,
        "passes": passes,
        "required": 90,
        "status": "PASS" if passes >= 90 else "FAIL",
        "details": rows,
    }


def finalize(args: argparse.Namespace) -> Dict[str, Any]:
    _, rows = load_train()
    predicate = build_predicate(rows)
    raw_path = args.output_dir / RAW_PATH.name
    records = load_records(raw_path)
    if len(records) != len(rows):
        raise M0GateFailure(
            f"M0 finalization requires 7,196 records, found {len(records)}"
        )
    instances = []
    exceptions = []
    for index, sample in enumerate(rows):
        record = records[index]
        expected_messages = build_extraction_messages(sample)
        if record.get("messages") != expected_messages:
            raise M0GateFailure(f"Logged prompt differs at train row {index}")
        if record.get("endpoint_error"):
            exceptions.append(
                {
                    "train_index": index,
                    "kind": "endpoint_error",
                    "detail": record["endpoint_error"],
                }
            )
            continue
        plan = sample["reward_model"]["ground_truth"]["time_steps"]
        try:
            segments = parse_extraction_response(record["response"], len(plan))
            row_instances = build_instances(index, sample, segments, predicate)
        except Stage1ExtractionError as error:
            exceptions.append(
                {
                    "train_index": index,
                    "kind": "invalid_extraction",
                    "detail": str(error),
                }
            )
            continue
        instances.extend(row_instances)
    if not instances:
        raise M0GateFailure("M0 produced no valid instances")

    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(
        args.embedding_model, device=args.embedding_device
    )
    merge_map, _, raw_names = cluster_names(instances, embedding_model)
    merged_instances = [
        replace(instance, skill_name=merge_map[instance.raw_skill_name])
        for instance in instances
    ]
    smoke = retrieval_smoke(merged_instances, embedding_model)

    descriptions: Dict[str, Counter[str]] = defaultdict(Counter)
    raw_names_by_skill: Dict[str, Set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for instance in merged_instances:
        descriptions[instance.skill_name][instance.description] += 1
        raw_names_by_skill[instance.skill_name].add(instance.raw_skill_name)
        counts[instance.skill_name] += 1
    skills: List[Dict[str, Any]] = []
    for skill_name in sorted(counts):
        description = min(
            descriptions[skill_name],
            key=lambda value: (-descriptions[skill_name][value], len(value), value),
        )
        skills.append(
            {
                "name": skill_name,
                "description": description,
                "raw_names": sorted(raw_names_by_skill[skill_name]),
                "instance_count": counts[skill_name],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    instance_jsonl = args.output_dir / "m0_instances.jsonl"
    with instance_jsonl.open("w") as destination:
        for instance in merged_instances:
            destination.write(json.dumps(instance.to_dict(), ensure_ascii=True) + "\n")
    parquet_rows = []
    for instance in merged_instances:
        value = instance.to_dict()
        for field in ("demo", "self_cond", "ordered_units", "unit_kinds"):
            value[field] = json.dumps(
                value[field], sort_keys=True, separators=(",", ":")
            )
        parquet_rows.append(value)
    pd.DataFrame(parquet_rows).to_parquet(
        args.output_dir / "m0_instances.parquet", index=False
    )
    pd.DataFrame(exceptions).to_csv(args.output_dir / "m0_exceptions.csv", index=False)
    atomic_json(args.output_dir / "m0_skills.json", {"skills": skills})
    atomic_json(
        args.output_dir / "m0_merge_map.json",
        {
            "embedding_model": args.embedding_model,
            "method": "single-link connected components over normalized name embeddings; canonical name is component medoid",
            "cosine_threshold": CLUSTER_THRESHOLD,
            "raw_name_count": len(raw_names),
            "abstract_skill_count": len(skills),
            "merge_map": merge_map,
        },
    )
    pd.DataFrame(smoke["details"]).to_csv(
        args.output_dir / "m0_retrieval_smoke.csv", index=False
    )
    valid_sources = len({instance.source_train_index for instance in merged_instances})
    unit_counts = [len(instance.ordered_units) for instance in merged_instances]
    single_unit_kinds = Counter(
        instance.unit_kinds[0]
        for instance in merged_instances
        if len(instance.ordered_units) == 1
    )
    summary: Dict[str, Any] = {
        "task": "M0",
        "gate": {
            "status": "PASS" if smoke["status"] == "PASS" else "FAIL",
            "lossless_valid_rows": valid_sources,
            "excluded_rows": len(exceptions),
            "accounted_rows": valid_sources + len(exceptions),
            "contiguous_instances": len(merged_instances),
            "smoke_status": smoke["status"],
            "smoke_passes": smoke["passes"],
            "smoke_required": smoke["required"],
        },
        "extraction": {
            "total_train_rows": len(rows),
            "valid_source_rows": valid_sources,
            "excluded_source_rows": len(exceptions),
            "exception_kinds": dict(Counter(item["kind"] for item in exceptions)),
            "instances": len(merged_instances),
            "units_per_instance": exact_quantiles(unit_counts),
            "zero_unit_instances": sum(value == 0 for value in unit_counts),
            "single_unit_instances": sum(value == 1 for value in unit_counts),
            "single_unit_instances_by_kind": dict(sorted(single_unit_kinds.items())),
        },
        "library": {
            "abstract_skills": len(skills),
            "raw_skill_names": len(raw_names),
            "instances_per_skill": exact_quantiles(list(counts.values())),
            "instances_by_skill": dict(sorted(counts.items())),
        },
        "clustering": {
            "embedding_model": args.embedding_model,
            "cosine_threshold": CLUSTER_THRESHOLD,
            "method": "single-link connected components; medoid canonical name",
        },
        "smoke": {key: value for key, value in smoke.items() if key != "details"},
        "scope": "individual skills only; no cooperation skills",
        "raw_extractions": str(raw_path.resolve()),
        "instances": str(instance_jsonl.resolve()),
    }
    if summary["gate"]["accounted_rows"] != len(rows):
        raise M0GateFailure("M0 row accounting does not cover all train rows")
    atomic_json(args.output_dir / "m0.summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["gate"]["status"] != "PASS":
        raise M0GateFailure(
            f"GATE M0 failed: retrieval smoke {smoke['passes']}/{SMOKE_ROWS}"
        )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VIKI Amendment 2 M0")
    parser.add_argument("command", choices=("extract", "finalize"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--base-url", dest="base_urls", action="append", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model-revision", required=False, default="UNRECORDED")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--embedding-model", default="all-mpnet-base-v2")
    parser.add_argument("--embedding-device", default="cpu")
    args = parser.parse_args(argv)
    if args.base_urls is None:
        args.base_urls = ["http://127.0.0.1:8040/v1"]
    if args.command == "extract" and args.model_revision == "UNRECORDED":
        parser.error("extract requires --model-revision")
    return args


if __name__ == "__main__":
    try:
        arguments = parse_args()
        extract(arguments) if arguments.command == "extract" else finalize(arguments)
    except M0GateFailure as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
