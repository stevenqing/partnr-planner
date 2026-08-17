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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import requests
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT.parent / "VIKI-R"
DATA_ROOT = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2"
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment3"
MANIFEST_PATH = OUTPUT_DIR / "f1_probe_manifest.parquet"
PICK_PATH = OUTPUT_DIR / "f1_pick.json"
SEED = 20260814
ROWS_PER_SPLIT = 200
ANCHOR_RATES = {"id": 0.084, "ood": 0.012}
MIN_OOD = 0.02
MIN_ID = 0.05
MIN_FORMAT = 0.90
CONTEXT_LENGTH = 16384
CANDIDATES = {
    "qwen2_5_vl_72b": {
        "model_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "published_anchor": True,
    },
    "qwen3_vl_32b": {
        "model_id": "Qwen/Qwen3-VL-32B-Instruct",
        "published_anchor": False,
    },
    "qwen3_vl_30b_a3b": {
        "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "published_anchor": False,
    },
    "qwen3_vl_235b_a22b": {
        "model_id": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "published_anchor": False,
    },
}
REQUIRED_CANDIDATES = (
    "qwen2_5_vl_72b",
    "qwen3_vl_32b",
    "qwen3_vl_30b_a3b",
)
OPTIONAL_CANDIDATES = ("qwen3_vl_235b_a22b",)


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
    spec = importlib.util.spec_from_file_location("_viki_bench_amendment3", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binomial_cdf(count: int, samples: int, probability: float) -> float:
    return sum(
        math.comb(samples, value)
        * probability**value
        * (1 - probability) ** (samples - value)
        for value in range(count + 1)
    )


def exact_predictive_interval(
    samples: int, probability: float, alpha: float = 0.05
) -> Tuple[int, int]:
    lower = next(
        count
        for count in range(samples + 1)
        if binomial_cdf(count, samples, probability) >= alpha / 2
    )
    upper = next(
        count
        for count in range(samples + 1)
        if binomial_cdf(count, samples, probability) >= 1 - alpha / 2
    )
    return lower, upper


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


def prepare_manifest() -> Dict[str, Any]:
    rows = []
    for split_name, dataset_name, offset in (
        ("id", "test", 0),
        ("ood", "val", 1),
    ):
        dataset = pd.read_parquet(DATA_ROOT / f"{dataset_name}.parquet")
        selected = sorted(
            random.Random(SEED + offset).sample(range(len(dataset)), ROWS_PER_SPLIT)
        )
        for index in selected:
            ground_truth = native(dataset.iloc[index]["reward_model"])["ground_truth"]
            rows.append(
                {
                    "probe_split": split_name,
                    "dataset_split": dataset_name,
                    "index": index,
                    "task_name": str(ground_truth["task_name"]),
                    "robot_count": sum(
                        robot is not None for robot in ground_truth["robots"].values()
                    ),
                    "scorer_seed": index,
                }
            )
    manifest = pd.DataFrame(rows).sort_values(["probe_split", "index"])
    if len(manifest) != 2 * ROWS_PER_SPLIT:
        raise GateFailure("F1 manifest does not contain 400 rows")
    if manifest.duplicated(["probe_split", "index"]).any():
        raise GateFailure("F1 manifest contains duplicate split/index rows")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(MANIFEST_PATH, index=False)
    manifest.to_csv(MANIFEST_PATH.with_suffix(".csv"), index=False)
    indices = {
        split: [int(value) for value in group["index"]]
        for split, group in manifest.groupby("probe_split", sort=True)
    }
    intervals = {
        split: {
            "published_rate": rate,
            "samples": ROWS_PER_SPLIT,
            "minimum_successes": exact_predictive_interval(ROWS_PER_SPLIT, rate)[0],
            "maximum_successes": exact_predictive_interval(ROWS_PER_SPLIT, rate)[1],
        }
        for split, rate in ANCHOR_RATES.items()
    }
    summary = {
        "task": "F1_manifest",
        "seed": SEED,
        "sampling": "uniform without replacement, independently by split",
        "samples_per_split": ROWS_PER_SPLIT,
        "indices_sha256": fingerprint(indices),
        "anchor_gate": {
            "method": "two-sided 95% exact binomial predictive interval",
            "alpha": 0.05,
            "intervals": intervals,
        },
        "pick_rule": {
            "primary": "highest OOD mean_task_score",
            "minimum_ood": MIN_OOD,
            "minimum_id": MIN_ID,
            "minimum_format_each_split": MIN_FORMAT,
            "tie_break": "higher ID, then candidate order",
            "fallback_order": ["Gemini-2.5-Flash", "GPT-4o"],
        },
        "candidate_plan": {
            "included": list(REQUIRED_CANDIDATES),
            "excluded": {
                "qwen3_vl_235b_a22b": (
                    "requires all 8x80GB GPUs; only GPU 0,1,3,5 are available "
                    "while the retained Amendment 2 servers occupy GPU 2,4,6,7"
                )
            },
        },
        "manifest": str(MANIFEST_PATH),
    }
    atomic_json(MANIFEST_PATH.with_suffix(".summary.json"), summary)
    return summary


def load_jsonl(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open() as source:
        return {
            (str(record["probe_split"]), int(record["index"])): record
            for record in (json.loads(line) for line in source if line.strip())
        }


def server_metadata(base_urls: Sequence[str]) -> List[Dict[str, Any]]:
    metadata = []
    for base_url in base_urls:
        root_url = base_url.rstrip("/").removesuffix("/v1")
        version_response = requests.get(f"{root_url}/version", timeout=30)
        version_response.raise_for_status()
        models_response = requests.get(f"{base_url.rstrip('/')}/models", timeout=30)
        models_response.raise_for_status()
        models = models_response.json()
        metadata.append(
            {
                "base_url": base_url,
                "version": version_response.json(),
                "models": [
                    {
                        key: model.get(key)
                        for key in ("id", "root", "parent", "max_model_len")
                    }
                    for model in models.get("data", [])
                ],
            }
        )
    return metadata


def generate_one(
    client: OpenAI,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    max_tokens: int,
) -> Dict[str, Any]:
    completion = client.chat.completions.create(
        model=model,
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


def pair_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    return {
        "samples": len(frame),
        "successes": int((frame["task_score"] == 1).sum()),
        "mean_task_score": float(frame["task_score"].mean()),
        "format_successes": int((frame["format_score"] == 1).sum()),
        "format_compliance": float((frame["format_score"] == 1).mean()),
        "mean_score": float(frame["score"].mean()),
        "endpoint_errors": int(frame["endpoint_error"].sum()),
    }


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    if args.candidate not in CANDIDATES:
        raise ValueError(args.candidate)
    if not MANIFEST_PATH.is_file():
        prepare_manifest()
    candidate = CANDIDATES[args.candidate]
    manifest = pd.read_parquet(MANIFEST_PATH)
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    datasets = {
        "test": pd.read_parquet(DATA_ROOT / "test.parquet"),
        "val": pd.read_parquet(DATA_ROOT / "val.parquet"),
    }
    output = OUTPUT_DIR / f"f1_{args.candidate}.jsonl"
    summary_path = output.with_suffix(".summary.json")
    metadata_path = output.with_suffix(output.suffix + ".run.json")
    run_metadata = {
        "task": "F1_probe",
        "candidate": args.candidate,
        "model_id": candidate["model_id"],
        "served_model": args.model,
        "model_revision": args.model_revision,
        "base_urls": list(args.base_urls),
        "server_metadata": server_metadata(args.base_urls),
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "max_model_len": CONTEXT_LENGTH,
        "image_processor": {"use_fast": False},
        "workers": args.workers,
        "manifest_sha256": json.loads(
            MANIFEST_PATH.with_suffix(".summary.json").read_text()
        )["indices_sha256"],
        "scorer_seed_rule": "dataset index",
    }
    run_hash = fingerprint(run_metadata)
    if metadata_path.is_file():
        if json.loads(metadata_path.read_text()) != run_metadata:
            raise GateFailure(
                f"Cannot resume {output}: run configuration does not match"
            )
    else:
        atomic_json(metadata_path, run_metadata)
    records = load_jsonl(output)
    clients = [
        OpenAI(
            api_key=os.environ.get(args.api_key_env, "EMPTY"),
            base_url=base_url,
            max_retries=args.max_retries,
            timeout=args.timeout,
        )
        for base_url in args.base_urls
    ]
    pending = []
    for row in manifest.to_dict("records"):
        key = (str(row["probe_split"]), int(row["index"]))
        if key not in records or records[key].get("endpoint_error"):
            pending.append(row)

    def work(row: Mapping[str, Any]) -> Dict[str, Any]:
        probe_split = str(row["probe_split"])
        dataset_split = str(row["dataset_split"])
        index = int(row["index"])
        sample = native(datasets[dataset_split].iloc[index].to_dict())
        client = clients[index % len(clients)]
        generated = generate_one(
            client, args.model, bench.get_messages(sample), args.max_tokens
        )
        metrics = bench.score_response(
            scorer,
            2,
            generated["response"],
            bench.get_ground_truth(sample),
            index,
        )
        return {
            "probe_split": probe_split,
            "dataset_split": dataset_split,
            "index": index,
            "task_name": str(row["task_name"]),
            "robot_count": int(row["robot_count"]),
            "run_fingerprint": run_hash,
            **generated,
            **metrics,
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with output.open("a") as destination, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {executor.submit(work, row): row for row in pending}
        for future in as_completed(futures):
            row = futures[future]
            key = (str(row["probe_split"]), int(row["index"]))
            try:
                record = future.result()
            except Exception as error:
                record = {
                    "probe_split": key[0],
                    "dataset_split": str(row["dataset_split"]),
                    "index": key[1],
                    "task_name": str(row["task_name"]),
                    "robot_count": int(row["robot_count"]),
                    "run_fingerprint": run_hash,
                    "endpoint_error": repr(error),
                }
            destination.write(json.dumps(record, ensure_ascii=True) + "\n")
            destination.flush()
            records[key] = record
    ordered_keys = [
        (str(row["probe_split"]), int(row["index"]))
        for row in manifest.to_dict("records")
    ]
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w") as destination:
        for key in ordered_keys:
            destination.write(json.dumps(records[key], ensure_ascii=True) + "\n")
    temporary.replace(output)
    errors = [
        records[key] for key in ordered_keys if records[key].get("endpoint_error")
    ]
    if errors:
        summary = {
            "task": "F1_probe",
            "candidate": args.candidate,
            "samples": len(ordered_keys),
            "endpoint_errors": len(errors),
            "status": "FAIL",
        }
        atomic_json(summary_path, summary)
        raise GateFailure(f"F1 {args.candidate} has {len(errors)} endpoint errors")
    tidy = pd.DataFrame(
        [
            {
                "probe_split": record["probe_split"],
                "dataset_split": record["dataset_split"],
                "index": int(record["index"]),
                "task_name": record["task_name"],
                "robot_count": int(record["robot_count"]),
                "task_score": float(record["task_score"]),
                "format_score": float(record["format_score"]),
                "score": float(record["score"]),
                "prompt_tokens": record["prompt_tokens"],
                "completion_tokens": record["completion_tokens"],
                "endpoint_error": False,
            }
            for record in (records[key] for key in ordered_keys)
        ]
    ).sort_values(["probe_split", "index"])
    maximum_prompt_tokens = int(tidy["prompt_tokens"].max())
    minimum_token_headroom = CONTEXT_LENGTH - (maximum_prompt_tokens + args.max_tokens)
    if minimum_token_headroom < 0:
        raise GateFailure(
            "F1 prompt budget exceeded: "
            f"{maximum_prompt_tokens}+{args.max_tokens}>{CONTEXT_LENGTH}"
        )
    tidy.to_parquet(output.with_suffix(".parquet"), index=False)
    by_split = {
        split: pair_summary(group)
        for split, group in tidy.groupby("probe_split", sort=True)
    }
    by_task = {
        split: {
            str(task): pair_summary(group)
            for task, group in split_frame.groupby("task_name", sort=True)
        }
        for split, split_frame in tidy.groupby("probe_split", sort=True)
    }
    summary = {
        "task": "F1_probe",
        "candidate": args.candidate,
        "model_id": candidate["model_id"],
        "model_revision": args.model_revision,
        "samples": len(tidy),
        "by_split": by_split,
        "by_task": by_task,
        "endpoint_errors": 0,
        "token_budget": {
            "max_model_len": CONTEXT_LENGTH,
            "max_output_tokens": args.max_tokens,
            "maximum_prompt_tokens": maximum_prompt_tokens,
            "minimum_token_headroom": minimum_token_headroom,
            "truncated_rows": 0,
        },
        "status": "PASS",
        "results": str(output),
        "parquet": str(output.with_suffix(".parquet")),
    }
    if candidate["published_anchor"]:
        intervals = json.loads(MANIFEST_PATH.with_suffix(".summary.json").read_text())[
            "anchor_gate"
        ]["intervals"]
        failures = {
            split: {
                "successes": by_split[split]["successes"],
                **intervals[split],
            }
            for split in ("id", "ood")
            if not intervals[split]["minimum_successes"]
            <= by_split[split]["successes"]
            <= intervals[split]["maximum_successes"]
        }
        summary["anchor_gate"] = {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
            "intervals": intervals,
        }
        if failures:
            summary["status"] = "FAIL"
    atomic_json(summary_path, summary)
    if summary["status"] == "FAIL":
        raise GateFailure("GATE F1-anchor failed")
    return summary


def pick_model(args: argparse.Namespace) -> Dict[str, Any]:
    if PICK_PATH.exists() and not args.verify:
        raise GateFailure(f"F1 pick already committed: {PICK_PATH}")
    anchor_path = OUTPUT_DIR / "f1_qwen2_5_vl_72b.summary.json"
    if not anchor_path.is_file():
        raise GateFailure("F1 pick requires the 72B anchor probe")
    anchor = json.loads(anchor_path.read_text())
    if anchor.get("anchor_gate", {}).get("status") != "PASS":
        raise GateFailure("F1 halted by failed 72B anchor gate")
    available = []
    missing = []
    for order, candidate in enumerate(CANDIDATES):
        path = OUTPUT_DIR / f"f1_{candidate}.summary.json"
        if not path.is_file():
            missing.append(candidate)
            continue
        summary = json.loads(path.read_text())
        if summary.get("status") != "PASS":
            raise GateFailure(f"Candidate probe did not pass: {candidate}")
        id_result = summary["by_split"]["id"]
        ood_result = summary["by_split"]["ood"]
        eligible = (
            ood_result["mean_task_score"] >= MIN_OOD
            and id_result["mean_task_score"] >= MIN_ID
            and ood_result["format_compliance"] >= MIN_FORMAT
            and id_result["format_compliance"] >= MIN_FORMAT
        )
        available.append(
            {
                "candidate": candidate,
                "model_id": CANDIDATES[candidate]["model_id"],
                "order": order,
                "id": id_result,
                "ood": ood_result,
                "eligible": eligible,
            }
        )
    missing_required = [
        candidate for candidate in missing if candidate in REQUIRED_CANDIDATES
    ]
    missing_optional = [
        candidate for candidate in missing if candidate in OPTIONAL_CANDIDATES
    ]
    if missing_required:
        raise GateFailure(
            "F1 pick missing required candidate probes: " + ", ".join(missing_required)
        )
    if missing_optional and not args.allow_missing:
        raise GateFailure(
            "F1 pick missing optional candidate probes: " + ", ".join(missing_optional)
        )
    eligible = [item for item in available if item["eligible"]]
    if eligible:
        winner = sorted(
            eligible,
            key=lambda item: (
                -item["ood"]["mean_task_score"],
                -item["id"]["mean_task_score"],
                item["order"],
            ),
        )[0]
        selection = {
            "route": "open",
            "candidate": winner["candidate"],
            "model_id": winner["model_id"],
        }
    else:
        selection = {
            "route": "closed_fallback",
            "candidate": "Gemini-2.5-Flash",
            "model_id": "Gemini-2.5-Flash",
            "reason": "no open candidate cleared the preregistered thresholds",
        }
    result = {
        "task": "F1_pick",
        "status": "PASS",
        "selection": selection,
        "candidates": available,
        "missing_optional_candidates": missing_optional,
        "pick_rule": json.loads(MANIFEST_PATH.with_suffix(".summary.json").read_text())[
            "pick_rule"
        ],
    }
    if args.verify:
        if not PICK_PATH.is_file() or json.loads(PICK_PATH.read_text()) != result:
            raise GateFailure("Existing F1 pick does not match recomputation")
    else:
        atomic_json(PICK_PATH, result)
    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VIKI Amendment 3 F1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    probe = subparsers.add_parser("probe")
    probe.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    probe.add_argument("--model", required=True)
    probe.add_argument("--model-revision", required=True)
    probe.add_argument("--base-url", dest="base_urls", action="append", required=True)
    probe.add_argument("--api-key-env", default="OPENAI_API_KEY")
    probe.add_argument("--max-tokens", type=int, default=2000)
    probe.add_argument("--workers", type=int, default=8)
    probe.add_argument("--max-retries", type=int, default=5)
    probe.add_argument("--timeout", type=float, default=3600)
    pick = subparsers.add_parser("pick")
    pick.add_argument("--allow-missing", action="store_true")
    pick.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        if args.command == "prepare":
            result = prepare_manifest()
        elif args.command == "probe":
            result = run_probe(args)
        else:
            result = pick_model(args)
    except GateFailure as error:
        print(f"GATE FAILURE: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
