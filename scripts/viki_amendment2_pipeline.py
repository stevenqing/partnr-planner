#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import requests
from openai import OpenAI

from habitat_llm.evaluation.viki_memory_skill import add_memory_to_messages
from habitat_llm.evaluation.viki_segment_memory import (
    RetrievedGroup,
    SegmentMemoryBank,
    build_retrieval_context,
    build_subgoal_messages,
    format_grouped_memory,
    parse_subgoal_prediction,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT.parent / "VIKI-R"
DATA_ROOT = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2"
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment2"
M0_SUMMARY = OUTPUT_DIR / "m0.summary.json"
M0_INSTANCES = OUTPUT_DIR / "m0_instances.jsonl"
M0_SKILLS = OUTPUT_DIR / "m0_skills.json"
M1_CACHE = OUTPUT_DIR / "m1_embeddings_all_mpnet_base_v2.npz"
A5_ID_MANIFEST = (
    ROOT / "results/viki_memory_experiments/amendment1/a5_id_safety_manifest.parquet"
)
C1_MANIFEST = (
    ROOT / "results/viki_memory_experiments/amendment1/c1_split_manifest.parquet"
)
B0_MANIFEST = OUTPUT_DIR / "b0_id_smoke_manifest.parquet"
SEED = 20260814
B0_ROWS = 100
SIMILARITY_THRESHOLD = 0.3
INSTANCES_PER_SKILL = 2
MAX_INSTANCES = 6
CONTEXT_LENGTH = 4096


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


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def fingerprint(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_bench() -> Any:
    path = ROOT / "habitat_llm/evaluation/viki_bench.py"
    spec = importlib.util.spec_from_file_location("_viki_bench_amendment2", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def load_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open() as source:
        return {
            int(record["index"]): record
            for record in (json.loads(line) for line in source if line.strip())
        }


def prepare_b0_manifest() -> Dict[str, Any]:
    if not A5_ID_MANIFEST.is_file():
        raise GateFailure(f"Missing frozen A5 ID manifest: {A5_ID_MANIFEST}")
    id_manifest = pd.read_parquet(A5_ID_MANIFEST)
    eligible = sorted(int(value) for value in id_manifest["index"])
    selected = sorted(random.Random(SEED).sample(eligible, B0_ROWS))
    manifest = id_manifest[id_manifest["index"].isin(selected)].sort_values("index")
    if len(manifest) != B0_ROWS:
        raise GateFailure("B0 manifest does not contain 100 unique A5 ID rows")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(B0_MANIFEST, index=False)
    manifest.to_csv(B0_MANIFEST.with_suffix(".csv"), index=False)
    summary = {
        "task": "B0_manifest",
        "seed": SEED,
        "source": str(A5_ID_MANIFEST),
        "eligible_rows": len(eligible),
        "samples": len(manifest),
        "indices_sha256": fingerprint({"indices": selected}),
    }
    atomic_json(B0_MANIFEST.with_suffix(".summary.json"), summary)
    return summary


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


def fit_memory_prompt(
    bench: Any,
    sample: Dict[str, Any],
    groups: Sequence[RetrievedGroup],
    base_url: str,
    model: str,
    max_tokens: int,
) -> Tuple[List[RetrievedGroup], str, int, int]:
    fitted = list(groups)
    removed = 0
    while True:
        prompt = format_grouped_memory(fitted)
        messages = add_memory_to_messages(bench.get_messages(sample), prompt)
        input_tokens = token_count(base_url, model, messages)
        if input_tokens + max_tokens <= CONTEXT_LENGTH:
            return fitted, prompt, input_tokens, removed
        position = next(
            (
                index
                for index in range(len(fitted) - 1, -1, -1)
                if fitted[index].instances
            ),
            None,
        )
        if position is None:
            raise GateFailure(
                f"Base prompt exceeds context budget: {input_tokens}+{max_tokens}>{CONTEXT_LENGTH}"
            )
        group = fitted[position]
        remaining = list(group.instances[:-1])
        fitted[position] = replace(
            group,
            instances=remaining,
            dropped_reason=None if remaining else "token_budget",
        )
        removed += 1


class PairedProvider:
    def __init__(self, args: argparse.Namespace, bench: Any) -> None:
        self.args = args
        self.bench = bench
        self.clients = [
            OpenAI(
                api_key=os.environ.get(args.api_key_env, "EMPTY"),
                base_url=base_url,
                max_retries=args.max_retries,
            )
            for base_url in args.base_urls
        ]
        self.memory = SegmentMemoryBank(
            M0_INSTANCES,
            M0_SKILLS,
            embedding_model=args.embedding_model,
            embedding_device=args.embedding_device,
            cache_path=M1_CACHE,
        )

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

    def generate(
        self,
        sample: Dict[str, Any],
        index: int,
        allowed_source_indices: Optional[Sequence[int]],
    ) -> Dict[str, Any]:
        endpoint = self.args.base_urls[index % len(self.args.base_urls)]
        base_messages = self.bench.get_messages(sample)
        base = self.call(index, base_messages, self.args.max_tokens)
        subgoal_messages = build_subgoal_messages(sample)
        prediction = self.call(index, subgoal_messages, self.args.subgoal_max_tokens)
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
            allowed_source_indices=allowed_source_indices,
        )
        fitted, memory_prompt, input_tokens, budget_removed = fit_memory_prompt(
            self.bench,
            sample,
            groups,
            endpoint,
            self.args.model,
            self.args.max_tokens,
        )
        memory_messages = add_memory_to_messages(
            self.bench.get_messages(sample), memory_prompt
        )
        memory = self.call(index, memory_messages, self.args.max_tokens)
        return {
            "base": base,
            "memory": memory,
            "route": {
                **prediction,
                "messages": subgoal_messages,
                "parsed_subgoals": subgoals,
                "parse_error": prediction_error,
                "groups_before_budget": [group.to_dict() for group in groups],
                "groups": [group.to_dict() for group in fitted],
                "memory_prompt": memory_prompt,
                "memory_input_tokens": input_tokens,
                "budget_removed_instances": budget_removed,
            },
        }


def pair_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    base_success = frame["base_task_score"] == 1
    memory_success = frame["memory_task_score"] == 1
    fail_to_success = int(((~base_success) & memory_success).sum())
    success_to_fail = int((base_success & (~memory_success)).sum())
    return {
        "samples": len(frame),
        "base_mean_task_score": float(frame["base_task_score"].mean()),
        "memory_mean_task_score": float(frame["memory_task_score"].mean()),
        "absolute_delta": float(
            frame["memory_task_score"].mean() - frame["base_task_score"].mean()
        ),
        "base_format_compliance": float((frame["base_format_score"] == 1).mean()),
        "memory_format_compliance": float((frame["memory_format_score"] == 1).mean()),
        "fail_to_success": fail_to_success,
        "success_to_fail": success_to_fail,
        "discordant_pairs": fail_to_success + success_to_fail,
        "mcnemar_exact_p": exact_mcnemar_p(fail_to_success, success_to_fail),
    }


def channel_definition(channel: str) -> Tuple[str, List[int], Dict[int, List[int]]]:
    if channel == "b0":
        if not B0_MANIFEST.is_file():
            prepare_b0_manifest()
        manifest = pd.read_parquet(B0_MANIFEST)
        return "test", sorted(int(value) for value in manifest["index"]), {}
    if channel == "id":
        manifest = pd.read_parquet(A5_ID_MANIFEST)
        return "test", sorted(int(value) for value in manifest["index"]), {}
    if channel == "episode":
        manifest = pd.read_parquet(C1_MANIFEST)
        manifest = manifest[manifest["channel"] == "episode_heldout"]
        allowed = {
            int(row["test_index"]): [
                int(value) for value in json.loads(row["allowed_memory_indices"])
            ]
            for _, row in manifest.iterrows()
        }
        return "test", sorted(allowed), allowed
    if channel == "ood":
        rows = len(pd.read_parquet(DATA_ROOT / "val.parquet", columns=["prompt"]))
        return "val", list(range(rows)), {}
    raise ValueError(channel)


def oracle_check(
    bench: Any, scorer: Any, dataset: pd.DataFrame, indices: Sequence[int]
) -> Dict[str, Any]:
    failures = []
    for index in indices:
        sample = native(dataset.iloc[index].to_dict())
        metrics = bench.score_response(
            scorer,
            2,
            bench.oracle_response(2, sample),
            bench.get_ground_truth(sample),
            index,
        )
        if metrics["task_score"] != 1 or metrics["format_score"] != 1:
            failures.append({"index": index, **metrics})
    return {
        "status": "PASS" if not failures else "FAIL",
        "rows": len(indices),
        "failures": failures,
    }


def require_m0() -> None:
    if not M0_SUMMARY.is_file():
        raise GateFailure("M0 summary is missing")
    summary = json.loads(M0_SUMMARY.read_text())
    if summary.get("gate", {}).get("status") != "PASS":
        raise GateFailure("GATE M0 did not pass")


def require_b0() -> None:
    summary_path = OUTPUT_DIR / "b0_smoke.summary.json"
    if not summary_path.is_file():
        raise GateFailure("B1 requires completed GATE B0")
    summary = json.loads(summary_path.read_text())
    if summary.get("gate", {}).get("status") != "PASS":
        raise GateFailure("B1 blocked by failed GATE B0")


def run_channel(args: argparse.Namespace, channel: str) -> Dict[str, Any]:
    require_m0()
    if channel != "b0":
        require_b0()
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    split, indices, allowed_by_index = channel_definition(channel)
    dataset = pd.read_parquet(DATA_ROOT / f"{split}.parquet")
    oracle = oracle_check(bench, scorer, dataset, indices) if channel == "b0" else None
    if oracle is not None and oracle["status"] != "PASS":
        raise GateFailure("GATE B0 oracle wiring check failed")
    output = OUTPUT_DIR / (
        "b0_smoke.jsonl" if channel == "b0" else f"b1_{channel}.jsonl"
    )
    summary_path = output.with_suffix(".summary.json")
    run_path = output.with_suffix(output.suffix + ".run.json")
    run_metadata = {
        "task": "B0" if channel == "b0" else "B1",
        "channel": channel,
        "split": split,
        "indices": indices,
        "model": args.model,
        "model_revision": args.model_revision,
        "base_urls": args.base_urls,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "subgoal_max_tokens": args.subgoal_max_tokens,
        "context_length": CONTEXT_LENGTH,
        "embedding_model": args.embedding_model,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "instances_per_skill": INSTANCES_PER_SKILL,
        "max_instances": MAX_INSTANCES,
        "scorer_seed_rule": "dataset index, paired across arms",
        "token_budget_rule": "remove instances from final active group until input+output<=4096",
    }
    run_hash = fingerprint(run_metadata)
    if run_path.is_file():
        if json.loads(run_path.read_text()) != run_metadata:
            raise GateFailure(f"Cannot resume {channel}: run metadata differs")
    else:
        atomic_json(run_path, run_metadata)
    records = load_jsonl(output)
    pending = [index for index in indices if index not in records]
    provider = PairedProvider(args, bench) if pending else None

    def generate(index: int) -> Tuple[int, Dict[str, Any]]:
        sample = native(dataset.iloc[index].to_dict())
        return index, provider.generate(sample, index, allowed_by_index.get(index))

    mode = "a" if output.is_file() else "w"
    with output.open(mode) as destination, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {executor.submit(generate, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            sample = native(dataset.iloc[index].to_dict())
            try:
                _, generated = future.result()
                arms = {}
                for arm in ("base", "memory"):
                    metrics = bench.score_response(
                        scorer,
                        2,
                        generated[arm]["response"],
                        bench.get_ground_truth(sample),
                        index,
                    )
                    arms[arm] = {**generated[arm], **metrics}
                record = {
                    "index": index,
                    "run_fingerprint": run_hash,
                    "route": generated["route"],
                    "arms": arms,
                }
            except Exception as error:
                record = {
                    "index": index,
                    "run_fingerprint": run_hash,
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
    selected = [records[index] for index in indices]
    errors = [record for record in selected if record.get("endpoint_error")]
    if errors:
        failure_summary = {
            "task": run_metadata["task"],
            "channel": channel,
            "samples": len(indices),
            "completed": len(selected),
            "endpoint_errors": len(errors),
            "gate": {"status": "FAIL" if channel == "b0" else "NOT_APPLICABLE"},
            "oracle": oracle,
            "results": str(output),
        }
        atomic_json(summary_path, failure_summary)
        raise GateFailure(f"{channel} halted with {len(errors)} endpoint errors")
    tidy_rows = []
    for record in selected:
        index = int(record["index"])
        ground_truth = native(dataset.iloc[index]["reward_model"])["ground_truth"]
        row = {
            "index": index,
            "task_name": ground_truth["task_name"],
            "robot_count": sum(
                robot is not None for robot in ground_truth["robots"].values()
            ),
            "route_parse_error": record["route"]["parse_error"] is not None,
            "routed_subgoals": json.dumps(record["route"]["parsed_subgoals"]),
            "retrieved_instances": sum(
                len(group["instances"]) for group in record["route"]["groups"]
            ),
            "budget_removed_instances": record["route"]["budget_removed_instances"],
        }
        for arm in ("base", "memory"):
            row[f"{arm}_task_score"] = float(record["arms"][arm]["task_score"])
            row[f"{arm}_format_score"] = float(record["arms"][arm]["format_score"])
        tidy_rows.append(row)
    tidy = pd.DataFrame(tidy_rows).sort_values("index")
    tidy_path = output.with_suffix(".parquet")
    tidy.to_parquet(tidy_path, index=False)
    overall = pair_summary(tidy)
    by_task = {
        str(key): pair_summary(group) for key, group in tidy.groupby("task_name")
    }
    by_robot = {
        str(key): pair_summary(group) for key, group in tidy.groupby("robot_count")
    }
    gate: Dict[str, Any] = {"status": "NOT_APPLICABLE"}
    if channel == "b0":
        failures = {
            arm: overall[f"{arm}_format_compliance"]
            for arm in ("base", "memory")
            if overall[f"{arm}_format_compliance"] < 0.9
        }
        gate = {
            "status": "PASS" if not failures else "FAIL",
            "minimum_format_compliance": 0.9,
            "failures": failures,
            "endpoint_errors": 0,
        }
    summary: Dict[str, Any] = {
        "task": run_metadata["task"],
        "channel": channel,
        "samples": len(tidy),
        "gate": gate,
        "oracle": oracle,
        "overall": overall,
        "by_task": by_task,
        "by_robot_count": by_robot,
        "route_parse_errors": int(tidy["route_parse_error"].sum()),
        "mean_retrieved_instances": float(tidy["retrieved_instances"].mean()),
        "budget_removed_instances": int(tidy["budget_removed_instances"].sum()),
        "endpoint_errors": 0,
        "results": str(output),
        "parquet": str(tidy_path),
    }
    atomic_json(summary_path, summary)
    if gate["status"] == "FAIL":
        raise GateFailure(
            "GATE B0 failed format compliance: "
            + ", ".join(f"{arm}={value:.3f}" for arm, value in failures.items())
        )
    return summary


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    require_m0()
    results = {
        "b0": run_channel(args, "b0"),
        "b1_id": run_channel(args, "id"),
        "b1_episode": run_channel(args, "episode"),
        "b1_ood": run_channel(args, "ood"),
    }
    return results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VIKI Amendment 2 B0/B1")
    parser.add_argument(
        "command",
        choices=("prepare-b0", "b0", "b1-id", "b1-episode", "b1-ood", "run-all"),
    )
    parser.add_argument("--base-url", dest="base_urls", action="append", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="qwen2.5-vl-7b-instruct-amendment2")
    parser.add_argument(
        "--model-revision",
        default="cc594898137f460bfe9f0759e9844b3ce807cfb5",
    )
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--subgoal-max-tokens", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--embedding-model", default="all-mpnet-base-v2")
    parser.add_argument("--embedding-device", default="cpu")
    args = parser.parse_args(argv)
    if args.base_urls is None:
        args.base_urls = [
            "http://127.0.0.1:8040/v1",
            "http://127.0.0.1:8041/v1",
            "http://127.0.0.1:8042/v1",
            "http://127.0.0.1:8043/v1",
        ]
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.command == "prepare-b0":
            result = prepare_b0_manifest()
        elif args.command == "b0":
            result = run_channel(args, "b0")
        elif args.command == "b1-id":
            result = run_channel(args, "id")
        elif args.command == "b1-episode":
            result = run_channel(args, "episode")
        elif args.command == "b1-ood":
            result = run_channel(args, "ood")
        else:
            result = run_all(args)
    except GateFailure as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
