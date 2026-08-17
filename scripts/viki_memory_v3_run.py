#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import viki_memory_experiments as exp
from openai import OpenAI


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_fingerprint(metadata: Dict[str, Any]) -> str:
    serialized = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class FilteredPlanProvider:
    def __init__(self, args: argparse.Namespace, bench: Any, memory_module: Any):
        from habitat_llm.evaluation.viki_memory_replay import FrozenOODMemoryReplay

        self.bench = bench
        self.memory_module = memory_module
        self.replay = FrozenOODMemoryReplay(
            args.benchmark_root,
            args.v0_table,
            args.frozen_memory_log,
        )
        self.client = OpenAI(
            api_key=os.environ.get(args.api_key_env, "EMPTY"),
            base_url=args.base_url,
            max_retries=args.max_retries,
        )
        self.model = args.model
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature

    def generate(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        filtered = self.replay.retrieve(index, sample, filter_objects=True)
        memory_prompt = self.memory_module.format_memory_prompt(filtered.retrieval)
        messages = self.memory_module.add_memory_to_messages(
            self.bench.get_messages(sample), memory_prompt
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        usage = completion.usage
        return {
            "response": completion.choices[0].message.content or "",
            "provider_metadata": filtered.to_metadata(),
            "prompt_tokens": None if usage is None else usage.prompt_tokens,
            "completion_tokens": (None if usage is None else usage.completion_tokens),
        }


def aggregate(frame: pd.DataFrame) -> Dict[str, Any]:
    baseline_successes = int(frame["baseline_success"].sum())
    filtered_successes = int(frame["filtered_success"].sum())
    fail_to_success = int(
        ((~frame["baseline_success"]) & frame["filtered_success"]).sum()
    )
    success_to_fail = int(
        (frame["baseline_success"] & (~frame["filtered_success"])).sum()
    )
    return {
        "samples": len(frame),
        "baseline_successes": baseline_successes,
        "filtered_successes": filtered_successes,
        "baseline_accuracy": baseline_successes / len(frame),
        "filtered_accuracy": filtered_successes / len(frame),
        "absolute_delta": (filtered_successes - baseline_successes) / len(frame),
        "fail_to_success": fail_to_success,
        "success_to_fail": success_to_fail,
        "mcnemar_exact_p": exp.exact_mcnemar_p(fail_to_success, success_to_fail),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    bench = exp._load_source(
        "_viki_bench_v3", exp.ROOT / "habitat_llm/evaluation/viki_bench.py"
    )
    memory_module = exp._load_source(
        "_viki_memory_v3",
        exp.ROOT / "habitat_llm/evaluation/viki_memory_skill.py",
    )
    dataset_path = args.benchmark_root / "data/VIKI-R/viki/VIKI-L2/val.parquet"
    dataset = pd.read_parquet(dataset_path)
    v0 = pd.read_parquet(args.v0_table).set_index("index")
    baseline = exp._index_records(exp._load_jsonl(exp.BASELINE_LOG), exp.BASELINE_LOG)
    scorer = bench.load_official_scorer(2, args.benchmark_root)
    provider = FilteredPlanProvider(args, bench, memory_module)

    stop = (
        len(dataset)
        if args.limit is None
        else min(len(dataset), args.start + args.limit)
    )
    indices = list(range(args.start, stop))
    metadata = {
        "task": "V3",
        "split": "val",
        "provider": "frozen-routing-object-filter",
        "benchmark_root": str(args.benchmark_root.resolve()),
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "start": args.start,
        "limit": args.limit,
        "v0_table": str(args.v0_table.resolve()),
        "frozen_memory_log": str(args.frozen_memory_log.resolve()),
        "filter_rule": (
            "Drop each frozen top-5 demo if a Move/Reach/Grasp/Place/Open/Close/"
            "Interact target is a train asset type absent from the current non-null "
            "init_pos asset types, or is neither a train asset nor train location."
        ),
    }
    fingerprint = run_fingerprint(metadata)
    run_path = args.output.with_suffix(args.output.suffix + ".run.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.is_file():
        if not run_path.is_file() or json.loads(run_path.read_text()) != metadata:
            raise ValueError("Cannot resume V3: run metadata does not match")
    else:
        atomic_json(run_path, metadata)

    records = {}
    if args.resume and args.output.is_file():
        records = exp._index_records(exp._load_jsonl(args.output), args.output)
        for index, record in records.items():
            if record.get("run_fingerprint") != fingerprint:
                raise ValueError(f"V3 fingerprint mismatch at index {index}")
    pending = [
        index
        for index in indices
        if index not in records or records[index].get("error")
    ]

    def generate(index: int) -> tuple[int, Dict[str, Any]]:
        sample = exp._native(dataset.iloc[index].to_dict())
        return index, provider.generate(sample, index)

    mode = "a" if args.resume else "w"
    with args.output.open(mode) as output, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {executor.submit(generate, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            sample = exp._native(dataset.iloc[index].to_dict())
            try:
                _, generated = future.result()
                response = generated["response"]
                metrics = bench.score_response(
                    scorer,
                    2,
                    response,
                    bench.get_ground_truth(sample),
                    args.seed + index,
                )
                record = {
                    "index": index,
                    "level": 2,
                    "split": "val",
                    "response": response,
                    "run_fingerprint": fingerprint,
                    "attempts": 1,
                    **metrics,
                    **generated,
                }
            except Exception as error:
                record = {
                    "index": index,
                    "level": 2,
                    "split": "val",
                    "response": "",
                    "run_fingerprint": fingerprint,
                    "attempts": 1,
                    "score": 0.0,
                    "format_score": 0.0,
                    "task_score": 0.0,
                    "error": repr(error),
                }
            output.write(json.dumps(record, ensure_ascii=True) + "\n")
            output.flush()
            records[index] = record

    selected = [records[index] for index in indices if index in records]
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w") as output:
        for record in sorted(records.values(), key=lambda item: item["index"]):
            output.write(json.dumps(record, ensure_ascii=True) + "\n")
    temporary.replace(args.output)

    errors = [record for record in selected if record.get("error")]
    if errors:
        raise RuntimeError(f"V3 has {len(errors)} endpoint errors")
    if len(selected) != len(indices):
        raise RuntimeError("V3 result coverage is incomplete")

    tidy_rows: List[Dict[str, Any]] = []
    for record in selected:
        index = int(record["index"])
        filter_metadata = record["provider_metadata"]
        kept_ids = filter_metadata["replay_kept_demo_ids"]
        original_ids = filter_metadata["replay_original_demo_ids"]
        baseline_input_tokens = int(
            v0.loc[index, "input_tokens_total"] - v0.loc[index, "injected_tokens"]
        )
        prompt_tokens = int(record["prompt_tokens"])
        tidy_rows.append(
            {
                "index": index,
                "ood_subset": v0.loc[index, "ood_subset"],
                "baseline_success": baseline[index]["task_score"] == 1,
                "filtered_success": record["task_score"] == 1,
                "routed_skill": filter_metadata["predicted_skill"],
                "original_demo_ids": original_ids,
                "injected_demo_ids": kept_ids,
                "dropped_demo_ids": filter_metadata["replay_dropped_demo_ids"],
                "n_demos_original": len(original_ids),
                "n_demos_injected": len(kept_ids),
                "n_demos_dropped": len(original_ids) - len(kept_ids),
                "fallback_fired": bool(filter_metadata["object_filter_fallback"]),
                "input_tokens_total": prompt_tokens,
                "injected_tokens": prompt_tokens - baseline_input_tokens,
                "truncation_flag": prompt_tokens + args.max_tokens > 4096,
                "plan_len_baseline": int(v0.loc[index, "plan_len_baseline"]),
                "plan_len_filtered": len(exp.parse_plan(record["response"])),
                "scorer_seed": args.seed + index,
                "format_score": float(record["format_score"]),
            }
        )
    tidy = pd.DataFrame(tidy_rows).sort_values("index")
    parquet_path = args.output.with_suffix(".parquet")
    tidy.to_parquet(parquet_path, index=False)

    overall = aggregate(tidy)
    by_subset = {
        subset: aggregate(tidy[tidy["ood_subset"] == subset])
        for subset in ("bowl_missing", "plate_missing", "both_missing")
    }
    fallback_rows = tidy[tidy["fallback_fired"]]
    fallback_response_mismatches = sum(
        records[int(index)]["response"] != baseline[int(index)]["response"]
        for index in fallback_rows["index"]
    )
    summary = {
        "task": "V3",
        "samples": len(tidy),
        "mean_task_score": float(tidy["filtered_success"].mean()),
        "mean_format_score": float(tidy["format_score"].mean()),
        "errors": 0,
        "overall": overall,
        "by_subset": by_subset,
        "filter": {
            "dropped_demo_fraction": float(
                tidy["n_demos_dropped"].sum() / tidy["n_demos_original"].sum()
            ),
            "mean_surviving_k": float(tidy["n_demos_injected"].mean()),
            "fallback_count": int(tidy["fallback_fired"].sum()),
            "fallback_rate": float(tidy["fallback_fired"].mean()),
        },
        "fallback_response_mismatches_vs_frozen_baseline": int(
            fallback_response_mismatches
        ),
        "results": str(args.output),
        "parquet": str(parquet_path),
    }
    summary_path = args.output.with_suffix(".summary.json")
    atomic_json(summary_path, summary)
    return summary


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VIKI OOD V3 filtered plans")
    parser.add_argument(
        "--benchmark-root", type=Path, default=exp.DEFAULT_BENCHMARK_ROOT
    )
    parser.add_argument(
        "--v0-table",
        type=Path,
        default=exp.DEFAULT_OUTPUT_DIR / "viki_ood_samples.parquet",
    )
    parser.add_argument("--frozen-memory-log", type=Path, default=exp.MEMORY_LOG)
    parser.add_argument(
        "--output",
        type=Path,
        default=exp.DEFAULT_OUTPUT_DIR / "viki_ood_filtered.jsonl",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8022/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="viki-r-7b-l2-memory")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2))


if __name__ == "__main__":
    main()
