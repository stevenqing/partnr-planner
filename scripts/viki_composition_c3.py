#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from transformers import AutoTokenizer

from habitat_llm.evaluation.viki_branch_conditions import (
    AvailabilityPredicate,
    derive_train_portable_assets,
    derive_train_vocabularies,
    discover_instruction_regions,
    get_instruction,
)
from habitat_llm.evaluation.viki_branch_memory import BranchIndexedMemory
from habitat_llm.evaluation.viki_memory_skill import (
    RetrievedInstance,
    VikiMemorySkillLibrary,
    add_memory_to_messages,
    get_prompt_context,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT.parent / "VIKI-R"
DATA_ROOT = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2"
MODEL_PATH = BENCHMARK_ROOT / "models/Qwen2.5VL-7B-Instruct-VIKI-R-2"
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment1"
MANIFEST_PATH = OUTPUT_DIR / "c1_split_manifest.parquet"
PREFLIGHT_PATH = OUTPUT_DIR / "c3_prompt_preflight.parquet"
BASELINE_PATH = ROOT / "results/viki_official_7b_l2_id.jsonl"
TRAIN_CACHE = ROOT / "results/viki_l2_memory_skill_all_mpnet_base_v2.npz"
A0_CENSUS = OUTPUT_DIR / "a0_branch_census.parquet"
TOKEN_TOLERANCE = 0.05


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


def load_bench() -> Any:
    path = ROOT / "habitat_llm/evaluation/viki_bench.py"
    spec = importlib.util.spec_from_file_location("_viki_bench_c3", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    with path.open() as source:
        return {
            int(record["index"]): record
            for record in (json.loads(line) for line in source if line.strip())
        }


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


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


def format_flat_prompt(instances: Sequence[RetrievedInstance]) -> str:
    if not instances:
        return ""
    lines = [
        "Flat memory guidance from successful training trajectories:",
        "Retrieved examples:",
    ]
    for number, item in enumerate(instances, 1):
        lines.extend(
            [
                f"{number}. Context: {item.instance.context}",
                f"   Demonstration: {item.instance.demonstration}",
            ]
        )
    lines.extend(
        [
            "Use these examples only as generic in-context evidence.",
            "Ground every concrete choice in the current task, image, and robot APIs.",
        ]
    )
    return "\n".join(lines)


def capability_indices(
    memory: BranchIndexedMemory, sample: Dict[str, Any]
) -> List[int]:
    instruction, robots, available_actions = get_prompt_context(sample)
    del instruction
    key = (
        tuple(sorted(robots.items())),
        tuple(
            (robot, tuple(actions))
            for robot, actions in sorted(available_actions.items())
        ),
    )
    indices = memory.executable_indices_cache.get(key)
    if indices is None:
        indices = memory.library._executable_indices(sample)
        memory.executable_indices_cache[key] = indices
    return indices


def flat_candidates(
    memory: BranchIndexedMemory,
    sample: Dict[str, Any],
    allowed_train_indices: Sequence[int],
    top_k: int = 5,
) -> List[RetrievedInstance]:
    allowed = set(int(index) for index in allowed_train_indices)
    executable = [
        index
        for index in capability_indices(memory, sample)
        if memory.library.instances[index].train_index in allowed
    ]
    instruction = get_prompt_context(sample)[0]
    query = memory.query_embedding_cache.get(instruction)
    if query is None:
        query = np.asarray(
            memory.library.embedding_model.encode(
                [instruction],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        )
        memory.query_embedding_cache[instruction] = query
    ranked = sorted(
        (
            RetrievedInstance(
                memory.library.instances[index],
                float(np.dot(query, memory.library.embeddings[index])),
            )
            for index in executable
        ),
        key=lambda item: (-item.similarity, item.instance.train_index),
    )
    result = []
    seen = set()
    for item in ranked:
        key = (item.instance.context, item.instance.demonstration)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) == top_k:
            break
    return result


def trim_flat_prompt(
    tokenizer: Any,
    prompt: str,
    target_fragment_tokens: int,
    bench: Any,
    sample: Dict[str, Any],
    base_url: str,
    model: str,
    target_total: int,
) -> Tuple[str, int]:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    best = (
        prompt,
        token_count(
            base_url, model, add_memory_to_messages(bench.get_messages(sample), prompt)
        ),
    )
    center = min(len(prompt_tokens), max(1, target_fragment_tokens))
    for count in range(max(1, center - 24), min(len(prompt_tokens), center + 24) + 1):
        candidate = tokenizer.decode(prompt_tokens[:count], skip_special_tokens=False)
        messages = add_memory_to_messages(bench.get_messages(sample), candidate)
        total = token_count(base_url, model, messages)
        if abs(total - target_total) < abs(best[1] - target_total):
            best = (candidate, total)
    return best


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    bench = load_bench()
    memory = build_memory()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    pd.read_parquet(DATA_ROOT / "train.parquet", columns=["reward_model"])
    test = pd.read_parquet(DATA_ROOT / "test.parquet")
    manifest = pd.read_parquet(args.manifest)
    train_signatures = {
        int(row["index"]): row["full_signature"]
        for _, row in pd.read_parquet(OUTPUT_DIR / "c0_composition_census.parquet")
        .query("split == 'train'")
        .iterrows()
    }
    rows = []
    violations = []
    for _, item in manifest.iterrows():
        index = int(item["test_index"])
        sample = native(test.iloc[index].to_dict())
        allowed = [int(value) for value in json.loads(item["allowed_memory_indices"])]
        skills = sorted(
            {
                memory.library.instances[train_index].skill_name
                for train_index in allowed
            }
        )
        if len(skills) != 1:
            raise GateFailure(
                f"C3 preflight requires one skill per pool at {item['channel']}:{index}; "
                f"observed={skills}"
            )
        if item["channel"].startswith("task_heldout"):
            signatures = {train_signatures[value] for value in allowed}
            if item["full_signature"] in signatures:
                raise GateFailure(f"C3 strict leakage at {item['channel']}:{index}")
        predicted_skill = f"<answer>{skills[0]}</answer>"
        skill = memory.retrieve(
            sample,
            5,
            predicted_skill,
            branch_indexing=True,
            graded_injection=True,
            allowed_train_indices=allowed,
        )
        skill_prompt = memory.format_prompt(skill)
        skill_messages = add_memory_to_messages(
            bench.get_messages(sample), skill_prompt
        )
        endpoint = args.base_urls[index % len(args.base_urls)]
        base_count = token_count(endpoint, args.model, bench.get_messages(sample))
        skill_count = token_count(endpoint, args.model, skill_messages)

        candidates = flat_candidates(memory, sample, allowed, 5)
        flat_options = []
        if candidates:
            for count in range(1, len(candidates) + 1):
                prompt = format_flat_prompt(candidates[:count])
                total = token_count(
                    endpoint,
                    args.model,
                    add_memory_to_messages(bench.get_messages(sample), prompt),
                )
                flat_options.append((prompt, total, count, False))
        else:
            flat_options.append(("", base_count, 0, False))
        flat_prompt, flat_count, flat_k, flat_truncated = min(
            flat_options,
            key=lambda option: (abs(option[1] - skill_count), option[2]),
        )
        relative_difference = abs(flat_count - skill_count) / skill_count
        if relative_difference > TOKEN_TOLERANCE and flat_prompt:
            target_fragment = max(1, skill_count - base_count)
            trimmed_prompt, trimmed_count = trim_flat_prompt(
                tokenizer,
                flat_prompt,
                target_fragment,
                bench,
                sample,
                endpoint,
                args.model,
                skill_count,
            )
            if abs(trimmed_count - skill_count) < abs(flat_count - skill_count):
                flat_prompt = trimmed_prompt
                flat_count = trimmed_count
                flat_truncated = True
                relative_difference = abs(flat_count - skill_count) / skill_count
        if relative_difference > TOKEN_TOLERANCE:
            violations.append(
                {
                    "channel": item["channel"],
                    "index": index,
                    "skill_tokens": skill_count,
                    "flat_tokens": flat_count,
                    "relative_difference": relative_difference,
                }
            )
        rows.append(
            {
                "channel": item["channel"],
                "index": index,
                "task_name": item["task_name"],
                "allowed_memory_count": len(allowed),
                "selected_skill": skills[0],
                "current_branch": skill.current_branch,
                "skill_tier": skill.tier,
                "skill_demo_ids": json.dumps(
                    [entry.instance.train_index for entry in skill.retrieval.instances],
                    separators=(",", ":"),
                ),
                "skill_prompt": skill_prompt,
                "skill_prompt_tokens": skill_count,
                "flat_k": flat_k,
                "flat_demo_ids": json.dumps(
                    [entry.instance.train_index for entry in candidates[:flat_k]],
                    separators=(",", ":"),
                ),
                "flat_prompt": flat_prompt,
                "flat_prompt_tokens": flat_count,
                "flat_truncated": flat_truncated,
                "token_relative_difference": relative_difference,
                "scorer_seed": index,
            }
        )
    if violations:
        raise GateFailure(
            f"C3 token-match preflight failed: observed={len(violations)} violations, "
            f"expected=0, miss={len(violations)}, first={violations[:5]}"
        )
    preflight = pd.DataFrame(rows)
    preflight.to_parquet(PREFLIGHT_PATH, index=False)
    preflight.to_csv(PREFLIGHT_PATH.with_suffix(".csv"), index=False)
    summary = {
        "task": "C3_preflight",
        "status": "PASS",
        "samples": len(preflight),
        "token_tolerance": TOKEN_TOLERANCE,
        "max_relative_difference": float(preflight["token_relative_difference"].max()),
        "flat_truncated_rows": int(preflight["flat_truncated"].sum()),
        "channels": {
            channel: {
                "samples": len(group),
                "skill_tiers": {
                    str(key): int(value)
                    for key, value in group["skill_tier"].value_counts().items()
                },
                "flat_k": {
                    str(key): int(value)
                    for key, value in group["flat_k"].value_counts().items()
                },
                "max_token_relative_difference": float(
                    group["token_relative_difference"].max()
                ),
            }
            for channel, group in preflight.groupby("channel")
        },
    }
    atomic_json(PREFLIGHT_PATH.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))
    return summary


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    import math

    total = left_only + right_only
    if total == 0:
        return 1.0
    tail = min(left_only, right_only)
    return min(
        1.0,
        2 * sum(math.comb(total, value) for value in range(tail + 1)) / (2**total),
    )


def compare(frame: pd.DataFrame, left: str, right: str) -> Dict[str, Any]:
    left_values = frame[f"{left}_success"]
    right_values = frame[f"{right}_success"]
    left_only = int((left_values & ~right_values).sum())
    right_only = int((~left_values & right_values).sum())
    return {
        "left": left,
        "right": right,
        "left_accuracy": float(left_values.mean()),
        "right_accuracy": float(right_values.mean()),
        "right_minus_left": float(right_values.mean() - left_values.mean()),
        "left_only": left_only,
        "right_only": right_only,
        "mcnemar_exact_p": exact_mcnemar_p(left_only, right_only),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not PREFLIGHT_PATH.is_file():
        raise GateFailure("C3 run requires passing prompt preflight")
    preflight = pd.read_parquet(PREFLIGHT_PATH)
    summary_path = PREFLIGHT_PATH.with_suffix(".summary.json")
    if json.loads(summary_path.read_text()).get("status") != "PASS":
        raise GateFailure("C3 run blocked by failed prompt preflight")
    bench = load_bench()
    test = pd.read_parquet(DATA_ROOT / "test.parquet")
    baseline = load_jsonl(BASELINE_PATH)
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    clients = [
        OpenAI(
            api_key=os.environ.get(args.api_key_env, "EMPTY"),
            base_url=url,
            max_retries=args.max_retries,
        )
        for url in args.base_urls
    ]
    metadata = {
        "task": "C3",
        "preflight": str(PREFLIGHT_PATH.resolve()),
        "preflight_sha256": hashlib.sha256(PREFLIGHT_PATH.read_bytes()).hexdigest(),
        "base_urls": args.base_urls,
        "model": args.model,
        "max_tokens": 2000,
        "temperature": 0,
        "workers": args.workers,
        "scorer_seed_rule": "0 + original test index",
    }
    run_hash = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = args.output_dir / "c3_results.jsonl"
    run_path = output.with_suffix(output.suffix + ".run.json")
    if args.resume and output.is_file():
        if not run_path.is_file() or json.loads(run_path.read_text()) != metadata:
            raise ValueError("Cannot resume C3: run metadata differs")
        with output.open() as source:
            records = {
                (record["channel"], int(record["index"])): record
                for record in (json.loads(line) for line in source if line.strip())
            }
    else:
        atomic_json(run_path, metadata)
        records = {}

    keys = [(row["channel"], int(row["index"])) for _, row in preflight.iterrows()]
    pending = [key for key in keys if key not in records or records[key].get("error")]
    rows_by_key = {
        (row["channel"], int(row["index"])): row for _, row in preflight.iterrows()
    }

    def generate(key: Tuple[str, int]) -> Tuple[Tuple[str, int], Dict[str, Any]]:
        channel, index = key
        row = rows_by_key[key]
        sample = native(test.iloc[index].to_dict())
        client = clients[index % len(clients)]
        prompts = {"skill": row["skill_prompt"], "flat": row["flat_prompt"]}
        generated = {}
        prompt_to_arm: Dict[str, str] = {}
        for arm, prompt in prompts.items():
            if not prompt:
                generated[arm] = {
                    "response": baseline[index]["response"],
                    "prompt_tokens": None,
                    "replayed_baseline": True,
                }
                continue
            if prompt in prompt_to_arm:
                prior = prompt_to_arm[prompt]
                generated[arm] = {**generated[prior], "reused_from": prior}
                continue
            messages = add_memory_to_messages(bench.get_messages(sample), prompt)
            completion = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=2000,
                temperature=0,
            )
            generated[arm] = {
                "response": completion.choices[0].message.content or "",
                "prompt_tokens": (
                    None if completion.usage is None else completion.usage.prompt_tokens
                ),
                "replayed_baseline": False,
            }
            prompt_to_arm[prompt] = arm
        return key, generated

    mode = "a" if args.resume else "w"
    with output.open(mode) as destination, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {executor.submit(generate, key): key for key in pending}
        for future in as_completed(futures):
            key = futures[future]
            channel, index = key
            sample = native(test.iloc[index].to_dict())
            try:
                _, generated = future.result()
                arms = {}
                for arm, value in generated.items():
                    metrics = bench.score_response(
                        scorer,
                        2,
                        value["response"],
                        bench.get_ground_truth(sample),
                        index,
                    )
                    arms[arm] = {**value, **metrics}
                record = {
                    "channel": channel,
                    "index": index,
                    "run_fingerprint": run_hash,
                    "arms": arms,
                }
            except Exception as error:
                record = {
                    "channel": channel,
                    "index": index,
                    "run_fingerprint": run_hash,
                    "error": repr(error),
                }
            destination.write(json.dumps(record, ensure_ascii=True) + "\n")
            destination.flush()
            records[key] = record
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w") as destination:
        for key in sorted(records):
            destination.write(json.dumps(records[key], ensure_ascii=True) + "\n")
    temporary.replace(output)

    selected = [records[key] for key in keys]
    errors = [record for record in selected if record.get("error")]
    if errors:
        raise RuntimeError(f"C3 has {len(errors)} endpoint errors")
    tidy_rows = []
    for record in selected:
        channel = record["channel"]
        index = int(record["index"])
        row = rows_by_key[(channel, index)]
        tidy_rows.append(
            {
                "channel": channel,
                "index": index,
                "task_name": row["task_name"],
                "baseline_success": baseline[index]["task_score"] == 1,
                "skill_success": record["arms"]["skill"]["task_score"] == 1,
                "flat_success": record["arms"]["flat"]["task_score"] == 1,
                "skill_format": float(record["arms"]["skill"]["format_score"]),
                "flat_format": float(record["arms"]["flat"]["format_score"]),
                "skill_tier": row["skill_tier"],
                "skill_demo_ids": row["skill_demo_ids"],
                "flat_demo_ids": row["flat_demo_ids"],
                "skill_prompt_tokens": row["skill_prompt_tokens"],
                "flat_prompt_tokens": row["flat_prompt_tokens"],
                "token_relative_difference": row["token_relative_difference"],
                "scorer_seed": index,
            }
        )
    tidy = pd.DataFrame(tidy_rows)
    tidy_path = args.output_dir / "c3_results.parquet"
    tidy.to_parquet(tidy_path, index=False)
    channel_summaries: Dict[str, Any] = {}
    for channel, group in tidy.groupby("channel"):
        channel_summaries[channel] = {
            "samples": len(group),
            "accuracies": {
                arm: float(group[f"{arm}_success"].mean())
                for arm in ("baseline", "skill", "flat")
            },
            "baseline_vs_skill": compare(group, "baseline", "skill"),
            "baseline_vs_flat": compare(group, "baseline", "flat"),
            "flat_vs_skill": compare(group, "flat", "skill"),
            "mean_format": {
                "skill": float(group["skill_format"].mean()),
                "flat": float(group["flat_format"].mean()),
            },
        }
    episode_advantage = channel_summaries["episode_heldout"]["flat_vs_skill"][
        "right_minus_left"
    ]
    strict_advantages = {
        channel: values["flat_vs_skill"]["right_minus_left"]
        for channel, values in channel_summaries.items()
        if channel.startswith("task_heldout")
    }
    summary: Dict[str, Any] = {
        "task": "C3",
        "samples": len(tidy),
        "errors": 0,
        "channels": channel_summaries,
        "predictions": {
            "all_methods_drop_episode_to_strict": {
                arm: all(
                    channel_summaries[channel]["accuracies"][arm]
                    < channel_summaries["episode_heldout"]["accuracies"][arm]
                    for channel in strict_advantages
                )
                for arm in ("baseline", "skill", "flat")
            },
            "skill_advantage_larger_in_strict": {
                channel: advantage > episode_advantage
                for channel, advantage in strict_advantages.items()
            },
        },
        "results": str(output),
        "parquet": str(tidy_path),
    }
    atomic_json(args.output_dir / "c3_results.summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and run VIKI C3 controls")
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--base-url", dest="base_urls", action="append", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="viki-r-7b-l2-amendment1")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.base_urls is None:
        args.base_urls = ["http://127.0.0.1:8030/v1"]
    return args


if __name__ == "__main__":
    arguments = parse_args()
    try:
        prepare(arguments) if arguments.command == "prepare" else run(arguments)
    except GateFailure as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
