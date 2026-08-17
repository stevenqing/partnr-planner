#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd
from openai import OpenAI

from habitat_llm.evaluation.viki_branch_conditions import (
    AvailabilityPredicate,
    derive_train_portable_assets,
    derive_train_vocabularies,
    discover_instruction_regions,
    get_instruction,
)
from habitat_llm.evaluation.viki_branch_memory import BranchIndexedMemory
from habitat_llm.evaluation.viki_memory_skill import (
    VikiMemorySkillLibrary,
    add_memory_to_messages,
    get_skill_prediction_messages,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT.parent / "VIKI-R"
DATA_ROOT = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2"
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment1"
TRAIN_CACHE = ROOT / "results/viki_l2_memory_skill_all_mpnet_base_v2.npz"
A0_CENSUS = OUTPUT_DIR / "a0_branch_census.parquet"
C0_LABELS = ROOT / "docs/viki_amendment1_c0_adjudication.tsv"
ID_BASELINE = ROOT / "results/viki_official_7b_l2_id.jsonl"
OOD_BASELINE = ROOT / "results/viki_official_7b_l2_ood.jsonl"
OOD_MEMORY = ROOT / "results/viki_memory_skill_7b_l2_ood.jsonl"
ID_MANIFEST = OUTPUT_DIR / "a5_id_safety_manifest.parquet"
ID_SEED = 20260814
ID_SAMPLE_SIZE = 300
ID_GATE_MIN_DELTA = -0.01


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


def load_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    with path.open() as source:
        records = {
            int(record["index"]): record
            for record in (json.loads(line) for line in source if line.strip())
        }
    return records


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def fingerprint(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def exact_mcnemar_p(fail_to_success: int, success_to_fail: int) -> float:
    import math

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


def prepare_id_manifest() -> Dict[str, Any]:
    test = pd.read_parquet(DATA_ROOT / "test.parquet", columns=["reward_model"])
    labels = pd.read_csv(C0_LABELS, sep="\t")
    excluded = sorted(
        int(value) for value in labels.loc[labels["split"] == "test", "index"]
    )
    eligible = sorted(set(range(len(test))) - set(excluded))
    selected = sorted(random.Random(ID_SEED).sample(eligible, ID_SAMPLE_SIZE))
    rows = []
    for index in selected:
        ground_truth = test.iloc[index]["reward_model"]["ground_truth"]
        rows.append(
            {
                "index": index,
                "task_name": ground_truth["task_name"],
                "robot_count": sum(
                    robot is not None for robot in ground_truth["robots"].values()
                ),
            }
        )
    manifest = pd.DataFrame(rows)
    if set(selected) & set(excluded):
        raise GateFailure("A5 ID sample overlaps C0 calibration rows")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(ID_MANIFEST, index=False)
    manifest.to_csv(ID_MANIFEST.with_suffix(".csv"), index=False)
    summary = {
        "task": "A5_ID_manifest",
        "seed": ID_SEED,
        "samples": len(manifest),
        "eligible_rows": len(eligible),
        "excluded_c0_calibration_rows": excluded,
        "overlap_with_calibration": 0,
        "robot_count": {
            str(key): int(value)
            for key, value in manifest["robot_count"].value_counts().items()
        },
        "task_count": {
            str(key): int(value)
            for key, value in manifest["task_name"].value_counts().items()
        },
    }
    atomic_json(ID_MANIFEST.with_suffix(".summary.json"), summary)
    return summary


def build_memory() -> BranchIndexedMemory:
    train = pd.read_parquet(DATA_ROOT / "train.parquet")
    samples = [row.to_dict() for _, row in train.iterrows()]
    assets, locations = derive_train_vocabularies(samples)
    portable = derive_train_portable_assets(samples)
    regions = discover_instruction_regions(
        (get_instruction(sample) for sample in samples), assets, locations
    )
    predicate = AvailabilityPredicate(assets, locations | regions, portable)
    library = VikiMemorySkillLibrary(
        BENCHMARK_ROOT,
        "all-mpnet-base-v2",
        "cpu",
        cache_path=TRAIN_CACHE,
    )
    census = pd.read_parquet(A0_CENSUS)
    labels = {
        int(row["index"]): str(row["branch"])
        for _, row in census[census["split"] == "train"].iterrows()
    }
    return BranchIndexedMemory(library, predicate, labels, sorted(assets))


class DualArmProvider:
    def __init__(self, args: argparse.Namespace, bench: Any) -> None:
        self.args = args
        self.bench = bench
        self.memory = build_memory()
        self.clients = [
            OpenAI(
                api_key=os.environ.get(args.api_key_env, "EMPTY"),
                base_url=base_url,
                max_retries=args.max_retries,
            )
            for base_url in args.base_urls
        ]
        self.route_cache: Dict[str, str] = {}
        self.route_lock = Lock()
        self.ood_routes = load_jsonl(OOD_MEMORY) if args.split == "val" else None

    def generate_text(
        self,
        index: int,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
    ) -> Dict[str, Any]:
        client = self.clients[index % len(self.clients)]
        completion = client.chat.completions.create(
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

    def route(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        if self.ood_routes is not None:
            raw = self.ood_routes[index]["provider_metadata"]["raw_skill_prediction"]
            return {"raw": raw, "cached": True, "replayed": True}
        descriptions = self.memory.library.executable_skill_descriptions(sample)
        messages = get_skill_prediction_messages(sample, descriptions)
        key = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        with self.route_lock:
            cached = self.route_cache.get(key)
        if cached is not None:
            return {"raw": cached, "cached": True, "replayed": False}
        generated = self.generate_text(index, messages, self.args.routing_max_tokens)
        raw = generated["response"]
        with self.route_lock:
            self.route_cache[key] = raw
        return {
            "raw": raw,
            "cached": False,
            "replayed": False,
            "prompt_tokens": generated["prompt_tokens"],
            "completion_tokens": generated["completion_tokens"],
        }

    def generate(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        route = self.route(sample, index)
        grounded = self.memory.retrieve(
            sample,
            top_k=5,
            predicted_skill=route["raw"],
            branch_indexing=True,
            graded_injection=False,
        )
        graded = self.memory.retrieve(
            sample,
            top_k=5,
            predicted_skill=route["raw"],
            branch_indexing=True,
            graded_injection=True,
        )
        prompts = {
            "grounded": add_memory_to_messages(
                self.bench.get_messages(sample), self.memory.format_prompt(grounded)
            ),
            "graded": add_memory_to_messages(
                self.bench.get_messages(sample), self.memory.format_prompt(graded)
            ),
        }
        generated: Dict[str, Dict[str, Any]] = {}
        prompt_keys: Dict[str, str] = {}
        for arm, messages in prompts.items():
            prompt_key = json.dumps(messages, sort_keys=True, separators=(",", ":"))
            prompt_keys[arm] = prompt_key
            reused_arm = next(
                (
                    prior_arm
                    for prior_arm, prior_key in prompt_keys.items()
                    if prior_arm != arm and prior_key == prompt_key
                ),
                None,
            )
            if reused_arm is not None:
                generated[arm] = {**generated[reused_arm], "reused_from": reused_arm}
            else:
                generated[arm] = self.generate_text(
                    index, messages, self.args.max_tokens
                )
        return {
            "route": route,
            "arms": {
                "grounded": {
                    **generated["grounded"],
                    "provider_metadata": grounded.to_metadata(),
                },
                "graded": {
                    **generated["graded"],
                    "provider_metadata": graded.to_metadata(),
                },
            },
        }


def aggregate(frame: pd.DataFrame, arm: str) -> Dict[str, Any]:
    success_column = f"{arm}_success"
    baseline_successes = int(frame["baseline_success"].sum())
    arm_successes = int(frame[success_column].sum())
    fixes = int(((~frame["baseline_success"]) & frame[success_column]).sum())
    regressions = int((frame["baseline_success"] & (~frame[success_column])).sum())
    return {
        "samples": len(frame),
        "baseline_successes": baseline_successes,
        "arm_successes": arm_successes,
        "baseline_accuracy": baseline_successes / len(frame),
        "arm_accuracy": arm_successes / len(frame),
        "absolute_delta": (arm_successes - baseline_successes) / len(frame),
        "fail_to_success": fixes,
        "success_to_fail": regressions,
        "mcnemar_exact_p": exact_mcnemar_p(fixes, regressions),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    import importlib.util

    bench_path = ROOT / "habitat_llm/evaluation/viki_bench.py"
    spec = importlib.util.spec_from_file_location("_viki_bench_a5", bench_path)
    if spec is None or spec.loader is None:
        raise ImportError(bench_path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    if args.split == "test":
        if not ID_MANIFEST.is_file():
            prepare_id_manifest()
        indices = [int(value) for value in pd.read_parquet(ID_MANIFEST)["index"]]
        baseline_path = ID_BASELINE
    else:
        id_summary_path = args.output_dir / "a5_id_safety.summary.json"
        if not id_summary_path.is_file():
            raise GateFailure("A5 OOD requires a passing ID safety summary")
        id_summary = json.loads(id_summary_path.read_text())
        if id_summary.get("gate", {}).get("status") != "PASS":
            raise GateFailure("A5 OOD blocked by failed ID safety gate")
        indices = list(range(1218))
        baseline_path = OOD_BASELINE
    dataset = pd.read_parquet(DATA_ROOT / f"{args.split}.parquet")
    baseline = load_jsonl(baseline_path)
    if sorted(baseline) != list(range(len(dataset))):
        raise GateFailure("Frozen baseline coverage mismatch")
    if args.split == "test":
        baseline_successes = sum(
            record["task_score"] == 1 for record in baseline.values()
        )
        format_mean = sum(record["format_score"] for record in baseline.values()) / len(
            baseline
        )
        baseline_errors = sum(bool(record.get("error")) for record in baseline.values())
        if baseline_successes != 1695 or format_mean != 1.0 or baseline_errors != 0:
            raise GateFailure(
                "GATE A5 baseline failed: "
                f"successes observed={baseline_successes}, expected=1695, "
                f"miss={baseline_successes - 1695}; format observed={format_mean}, "
                f"expected=1.0, miss={format_mean - 1.0}; "
                f"errors observed={baseline_errors}, "
                "expected=0"
            )
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    provider = DualArmProvider(args, bench)

    run_metadata = {
        "task": "A5",
        "split": args.split,
        "indices": indices,
        "base_urls": args.base_urls,
        "model": args.model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "routing_max_tokens": args.routing_max_tokens,
        "workers": args.workers,
        "scorer_seed_rule": "0 + index",
        "train_similarity_bar": provider.memory.similarity_bar,
        "grounded_arm": "branch-indexed; abstract only if no branch instances",
        "graded_arm": "branch-indexed; grounded iff top similarity clears train-only bar",
    }
    run_hash = fingerprint(run_metadata)
    output = args.output or args.output_dir / (
        "a5_id_safety.jsonl" if args.split == "test" else "a5_ood_dual_arm.jsonl"
    )
    run_path = output.with_suffix(output.suffix + ".run.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and output.is_file():
        if not run_path.is_file() or json.loads(run_path.read_text()) != run_metadata:
            raise ValueError("Cannot resume A5: run metadata differs")
        records = load_jsonl(output)
    else:
        atomic_json(run_path, run_metadata)
        records = {}

    pending = [
        index
        for index in indices
        if index not in records or records[index].get("error")
    ]

    def generate(index: int) -> tuple[int, Dict[str, Any]]:
        sample = native(dataset.iloc[index].to_dict())
        return index, provider.generate(sample, index)

    mode = "a" if args.resume else "w"
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
                for arm, arm_generated in generated["arms"].items():
                    metrics = bench.score_response(
                        scorer,
                        2,
                        arm_generated["response"],
                        bench.get_ground_truth(sample),
                        index,
                    )
                    arms[arm] = {**arm_generated, **metrics}
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
                    "error": repr(error),
                }
            destination.write(json.dumps(record, ensure_ascii=True) + "\n")
            destination.flush()
            records[index] = record

    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w") as destination:
        for index in sorted(records):
            destination.write(json.dumps(records[index], ensure_ascii=True) + "\n")
    temporary.replace(output)

    selected_records: List[Dict[str, Any]] = [records[index] for index in indices]
    endpoint_error_records = [
        record for record in selected_records if record.get("error")
    ]
    if endpoint_error_records:
        raise RuntimeError(f"A5 has {len(endpoint_error_records)} endpoint errors")
    tidy_rows = []
    for record in selected_records:
        index = int(record["index"])
        ground_truth = dataset.iloc[index]["reward_model"]["ground_truth"]
        robot_count = sum(
            robot is not None for robot in ground_truth["robots"].values()
        )
        row = {
            "index": index,
            "task_name": ground_truth["task_name"],
            "robot_count": robot_count,
            "baseline_success": baseline[index]["task_score"] == 1,
            "scorer_seed": index,
        }
        for arm in ("grounded", "graded"):
            value = record["arms"][arm]
            metadata = value["provider_metadata"]
            row.update(
                {
                    f"{arm}_success": value["task_score"] == 1,
                    f"{arm}_format_score": float(value["format_score"]),
                    f"{arm}_tier": metadata["injection_tier"],
                    f"{arm}_routed_skill": metadata["predicted_skill"],
                    f"{arm}_branch": metadata["current_branch"],
                    f"{arm}_absent_assets": json.dumps(
                        metadata["current_absent_assets"], separators=(",", ":")
                    ),
                    f"{arm}_demo_ids": json.dumps(
                        [instance["train_index"] for instance in metadata["instances"]],
                        separators=(",", ":"),
                    ),
                    f"{arm}_prompt_tokens": value["prompt_tokens"],
                }
            )
        tidy_rows.append(row)
    tidy = pd.DataFrame(tidy_rows).sort_values("index")
    tidy_path = output.with_suffix(".parquet")
    tidy.to_parquet(tidy_path, index=False)

    arms = {arm: aggregate(tidy, arm) for arm in ("grounded", "graded")}
    by_robot = {
        str(count): {arm: aggregate(group, arm) for arm in ("grounded", "graded")}
        for count, group in tidy.groupby("robot_count")
    }
    by_task = {
        str(task): {arm: aggregate(group, arm) for arm in ("grounded", "graded")}
        for task, group in tidy.groupby("task_name")
    }
    gate: Dict[str, Any] = {"status": "NOT_APPLICABLE"}
    if args.split == "test":
        failures = {
            arm: values["absolute_delta"]
            for arm, values in arms.items()
            if values["absolute_delta"] < ID_GATE_MIN_DELTA
        }
        gate = {
            "status": "PASS" if not failures else "FAIL",
            "minimum_delta": ID_GATE_MIN_DELTA,
            "failures": failures,
        }
    summary: Dict[str, Any] = {
        "task": "A5",
        "split": args.split,
        "samples": len(tidy),
        "gate": gate,
        "arms": arms,
        "by_robot_count": by_robot,
        "by_task": by_task,
        "tier_counts": {
            arm: {
                str(key): int(value)
                for key, value in tidy[f"{arm}_tier"].value_counts().items()
            }
            for arm in ("grounded", "graded")
        },
        "mean_format_score": {
            arm: float(tidy[f"{arm}_format_score"].mean())
            for arm in ("grounded", "graded")
        },
        "errors": 0,
        "results": str(output),
        "parquet": str(tidy_path),
    }
    summary_path = output.with_suffix(".summary.json")
    atomic_json(summary_path, summary)
    if gate["status"] == "FAIL":
        raise GateFailure(
            "GATE A5 failed: "
            + ", ".join(
                f"{arm} observed={delta:.6f}, expected>={ID_GATE_MIN_DELTA:.6f}, "
                f"miss={delta - ID_GATE_MIN_DELTA:.6f}"
                for arm, delta in gate["failures"].items()
            )
        )
    return summary


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Amendment 1 A5 arms")
    parser.add_argument("command", choices=("prepare-id", "run"))
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--base-url",
        dest="base_urls",
        action="append",
        default=None,
        help="Repeat for deterministic load distribution across identical endpoints.",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="viki-r-7b-l2-memory")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--routing-max-tokens", type=int, default=768)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.base_urls is None:
        args.base_urls = ["http://127.0.0.1:8022/v1"]
    return args


def main() -> None:
    args = parse_args()
    try:
        result = (
            prepare_id_manifest() if args.command == "prepare-id" else evaluate(args)
        )
    except GateFailure as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
