#!/usr/bin/env python3
"""VIKI-L2-Interactive: a partner-observation extension of VIKI-L2.

The benchmark data, the answer format and the official scorer are untouched.
The only change is on the prompt side: the ground-truth actions that the partner
robots perform at step 1 are revealed, and the model is asked for the complete
plan starting from step 1. Every arm sees the identical extension, so the
comparison between arms stays internal to this protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from openai import OpenAI

from habitat_llm.evaluation import viki_bench as bench
from habitat_llm.evaluation.viki_memory_skill import add_memory_to_messages
from viki_amendment5 import (
    BENCHMARK_ROOT,
    DATA_ROOT,
    atomic_json,
    file_sha256,
    fingerprint,
    load_jsonl,
    messages_sha256,
    native,
    validate_local_service,
)
from viki_amendment6 import ROOT, GateFailure

OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment8"
PREREGISTRATION_PATH = OUTPUT_DIR / "preregistration.json"
MANIFEST_PATH = OUTPUT_DIR / "interactive_manifest.jsonl"
MANIFEST_SUMMARY_PATH = OUTPUT_DIR / "interactive_manifest.summary.json"
TOKEN_PROTOCOL_PATH = OUTPUT_DIR / "token_alignment_protocol.json"

SOURCE_PARQUET = DATA_ROOT / "test.parquet"
MEMORY_PARQUET = DATA_ROOT / "train.parquet"
OFFICIAL_SCORER = BENCHMARK_ROOT / "verl/verl/utils/reward_score/viki_2.py"

EXPOSED_STEPS = 1
SELF_ROBOT = "R1"
ARMS = ("zero_shot", "skill_memory", "gmemory", "trajectory_rag")
SEED = 20260829

SERVED_MODEL = "qwen2.5-vl-72b-amendment3-f2"
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
MODEL_REVISION = "89c86200743eec961a297729e7990e8f2ddbc4c5"
PLAN_MAX_TOKENS = 2000
PLAN_TEMPERATURE = 0

PARTNER_PREFIX_TEMPLATE = (
    "The partner robots have already executed step {step} of this plan:\n"
    "{actions}\n"
    "Continue from this shared starting point and output the complete plan, "
    "beginning at step {step} and including the partner actions shown above."
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def step_actions(step: Dict[str, Any]) -> Dict[str, str]:
    """Actions actually issued at one time step, dropping empty robot slots."""
    issued: Dict[str, str] = {}
    for robot_id, value in step["actions"].items():
        if value is None:
            continue
        text = str(value)
        if text in ("None", "nan", ""):
            continue
        issued[robot_id] = text
    return issued


def ordered_steps(ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sorted(ground_truth["time_steps"], key=lambda item: int(item["step"]))


def qualifies(ground_truth: Dict[str, Any]) -> bool:
    """A row is usable when a partner acts inside the exposed prefix and the
    self robot still has work left afterwards."""
    steps = ordered_steps(ground_truth)
    if len(steps) <= EXPOSED_STEPS:
        return False
    exposed = [step_actions(step) for step in steps[:EXPOSED_STEPS]]
    remaining = [step_actions(step) for step in steps[EXPOSED_STEPS:]]
    partner_acted = any(
        any(robot_id != SELF_ROBOT for robot_id in issued) for issued in exposed
    )
    self_has_work = any(SELF_ROBOT in issued for issued in remaining)
    return partner_acted and self_has_work


def partner_prefix(ground_truth: Dict[str, Any]) -> str:
    steps = ordered_steps(ground_truth)
    lines = []
    for step in steps[:EXPOSED_STEPS]:
        issued = step_actions(step)
        partner = {
            robot_id: value
            for robot_id, value in sorted(issued.items())
            if robot_id != SELF_ROBOT
        }
        rendered = ", ".join(f"{rid}: {value}" for rid, value in partner.items())
        lines.append(f"  step {int(step['step'])} -> {rendered}")
    return PARTNER_PREFIX_TEMPLATE.format(
        step=int(steps[0]["step"]), actions="\n".join(lines)
    )


def preregistration() -> Dict[str, Any]:
    return {
        "task": "Amendment8_VIKI_L2_Interactive",
        "status": "PREREGISTERED",
        "filed_before_generation": True,
        "training": False,
        "gradient_updates": False,
        "seed": SEED,
        "extension": {
            "name": "VIKI-L2-Interactive",
            "claim": "A partner-observation extension of VIKI-L2, not VIKI-L2 itself.",
            "exposed_steps": EXPOSED_STEPS,
            "self_robot": SELF_ROBOT,
            "output_contract": "complete plan from step 1, official answer format",
            "prompt_template_sha256": sha256_text(PARTNER_PREFIX_TEMPLATE),
        },
        "scoring": {
            "scorer": "verl.utils.reward_score.viki_2.compute_score",
            "scorer_sha256": file_sha256(OFFICIAL_SCORER),
            "modified": False,
            "note": (
                "The official scorer is used unmodified. Because the model returns "
                "the complete plan, the step-count ratio gate keeps its original "
                "meaning."
            ),
        },
        "data": {
            "source_parquet": str(SOURCE_PARQUET.relative_to(BENCHMARK_ROOT)),
            "source_parquet_sha256": file_sha256(SOURCE_PARQUET),
            "memory_parquet": str(MEMORY_PARQUET.relative_to(BENCHMARK_ROOT)),
            "memory_parquet_sha256": file_sha256(MEMORY_PARQUET),
            "selection_rule": (
                "rows of test.parquet where a non-self robot acts within the "
                "exposed prefix and the self robot still acts afterwards, in "
                "ascending dataset index order"
            ),
        },
        "arms": list(ARMS),
        "shared_memory_source": "train.parquet, identical rows for every memory arm",
        "comparisons": "paired exact McNemar between every memory arm and zero_shot",
    }


def build_manifest() -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(SOURCE_PARQUET)
    selected: List[Dict[str, Any]] = []
    for index in range(len(dataset)):
        row = native(dataset.iloc[index].to_dict())
        ground_truth = row["reward_model"]["ground_truth"]
        if not qualifies(ground_truth):
            continue
        steps = ordered_steps(ground_truth)
        remaining = [step_actions(step) for step in steps[EXPOSED_STEPS:]]
        selected.append(
            {
                "index": index,
                "task_id": ground_truth.get("task_id"),
                "task_name": ground_truth.get("task_name"),
                "robots": sorted(ground_truth["robots"].keys()),
                "plan_steps": len(steps),
                "self_actions_remaining": sum(
                    1 for issued in remaining if SELF_ROBOT in issued
                ),
                "partner_prefix": partner_prefix(ground_truth),
                "partner_prefix_sha256": sha256_text(partner_prefix(ground_truth)),
            }
        )

    with MANIFEST_PATH.open("w") as sink:
        for record in selected:
            sink.write(canonical_json(record) + "\n")

    summary = {
        "task": "Amendment8_interactive_manifest",
        "status": "PASS",
        "rows": len(selected),
        "source_rows": len(dataset),
        "exposed_steps": EXPOSED_STEPS,
        "mean_self_actions_remaining": float(
            np.mean([item["self_actions_remaining"] for item in selected])
        ),
        "mean_plan_steps": float(np.mean([item["plan_steps"] for item in selected])),
        "manifest": str(MANIFEST_PATH),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
    }
    atomic_json(MANIFEST_SUMMARY_PATH, summary)
    return summary


def freeze() -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observed = preregistration()
    if PREREGISTRATION_PATH.is_file():
        existing = json.loads(PREREGISTRATION_PATH.read_text())
        if existing != observed:
            raise GateFailure("Amendment 8 preregistration changed after filing")
        return existing
    atomic_json(PREREGISTRATION_PATH, observed)
    return observed


def token_alignment_protocol() -> Dict[str, Any]:
    return {
        "task": "Amendment8_token_alignment",
        "status": "FILED_BEFORE_MEMORY_ARM_GENERATION",
        "why": (
            "A memory arm could win merely by placing more context in the prompt. "
            "Aligning the memory budget removes that explanation."
        ),
        "stages": {
            "stage_1_natural": (
                "Every memory arm first runs at its own natural memory-prompt "
                "length. These are the primary results."
            ),
            "stage_2_aligned": (
                "Every memory arm then reruns with its memory prompt trimmed to "
                "the per-row budget below. These are the controlled results."
            ),
        },
        "per_row_budget": (
            "the minimum natural memory-prompt token count across all memory arms "
            "for that row"
        ),
        "budget_rationale": (
            "The minimum is fixed by the data rather than by choosing a reference "
            "arm, so no arm is privileged and no arm can hold a context-length "
            "advantage over another."
        ),
        "trimming": (
            "reuse viki_amendment6.trim_prompt, the same routine Amendment 7 used, "
            "applied identically to every arm"
        ),
        "reporting": (
            "Both stages are reported. A win that survives stage 2 is attributed to "
            "retrieval quality; a win that disappears is attributed to context length."
        ),
        "zero_shot_unaffected": (
            "The zero_shot arm carries no memory prompt, so alignment does not "
            "change it and its stage-1 run stands for both stages."
        ),
        "arms_covered": [arm for arm in ARMS if arm != "zero_shot"],
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
    }


def freeze_token_protocol() -> Dict[str, Any]:
    observed = token_alignment_protocol()
    for arm in ARMS:
        if arm == "zero_shot":
            continue
        if arm_paths(arm)["results"].is_file():
            raise GateFailure(
                f"Cannot file the token protocol after {arm} generation started"
            )
    if TOKEN_PROTOCOL_PATH.is_file():
        existing = json.loads(TOKEN_PROTOCOL_PATH.read_text())
        if existing != observed:
            raise GateFailure("Amendment 8 token alignment protocol changed")
        return existing
    atomic_json(TOKEN_PROTOCOL_PATH, observed)
    return observed


def load_manifest() -> Dict[int, Dict[str, Any]]:
    if not MANIFEST_PATH.is_file() or not MANIFEST_SUMMARY_PATH.is_file():
        raise GateFailure("Build the Amendment 8 interactive manifest first")
    summary = json.loads(MANIFEST_SUMMARY_PATH.read_text())
    if summary.get("manifest_sha256") != file_sha256(MANIFEST_PATH):
        raise GateFailure("Amendment 8 manifest certificate is stale")
    records = {}
    with MANIFEST_PATH.open() as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record["index"])] = record
    return records


def interactive_messages(
    sample: Dict[str, Any], partner_text: str, memory_prompt: str
) -> List[Dict[str, Any]]:
    """Official messages, plus the partner prefix as part of the task, plus any
    arm-specific memory. The partner prefix is identical for every arm; only the
    memory prompt distinguishes them."""
    messages = bench.get_messages(sample)
    user_message = next(
        message for message in reversed(messages) if message["role"] == "user"
    )
    content = user_message["content"]
    if isinstance(content, list):
        text_item = next(item for item in content if item["type"] == "text")
        text_item["text"] = f"{text_item['text']}\n\n{partner_text}"
    else:
        user_message["content"] = f"{content}\n\n{partner_text}"
    return add_memory_to_messages(messages, memory_prompt)


def arm_paths(arm: str) -> Dict[str, Path]:
    return {
        "results": OUTPUT_DIR / f"{arm}.jsonl",
        "run": OUTPUT_DIR / f"{arm}.jsonl.run.json",
        "summary": OUTPUT_DIR / f"{arm}.summary.json",
    }


def make_memory(arm: str, client) -> Any:
    """One provider per arm. zero_shot has none; the rest come from
    viki_amendment8_memory, which keeps every arm's retrieval in one place."""
    if arm == "zero_shot":
        return None
    import viki_amendment8_memory as memories

    if arm == "trajectory_rag":
        return memories.TrajectoryRag()
    if arm == "gmemory":
        return memories.GMemory(client)
    if arm == "skill_memory":
        return memories.SkillMemory(
            OUTPUT_DIR / "skill_memory_bank", EXPOSED_STEPS, SELF_ROBOT
        )
    raise GateFailure(f"Unknown arm: {arm}")


def run_arm(arm: str, base_url: str, workers: int) -> Dict[str, Any]:
    if arm not in ARMS:
        raise GateFailure(f"Unknown arm: {arm}")
    freeze()
    manifest = load_manifest()
    runtime = validate_local_service("qwen2_5_vl_72b", base_url)
    if runtime["models"][0]["id"] != SERVED_MODEL:
        raise GateFailure("Unexpected served model for Amendment 8")

    dataset = pd.read_parquet(SOURCE_PARQUET)
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    paths = arm_paths(arm)
    metadata = {
        "task": "Amendment8_VIKI_L2_Interactive_run",
        "arm": arm,
        "served_model": SERVED_MODEL,
        "model_revision": MODEL_REVISION,
        "temperature": PLAN_TEMPERATURE,
        "max_tokens": PLAN_MAX_TOKENS,
        "seed": SEED,
        "rows": len(manifest),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
        "scorer_sha256": file_sha256(OFFICIAL_SCORER),
    }
    if paths["run"].is_file() and json.loads(paths["run"].read_text()) != metadata:
        raise GateFailure(f"Cannot resume {arm}: metadata differs")
    atomic_json(paths["run"], metadata)
    run_hash = fingerprint(metadata)

    done = load_jsonl(paths["results"]) if paths["results"].is_file() else {}
    pending = [index for index in sorted(manifest) if index not in done]
    client = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=5, timeout=3600)
    provider = make_memory(arm, client)

    def one(index: int) -> Dict[str, Any]:
        sample = native(dataset.iloc[index].to_dict())
        record = manifest[index]
        memory_prompt = "" if provider is None else provider.prompt(index, sample)
        messages = interactive_messages(
            sample, record["partner_prefix"], memory_prompt
        )
        completion = client.chat.completions.create(
            model=SERVED_MODEL,
            messages=messages,
            temperature=PLAN_TEMPERATURE,
            max_tokens=PLAN_MAX_TOKENS,
            seed=SEED,
        )
        response = completion.choices[0].message.content or ""
        metrics = bench.score_response(
            scorer, 2, response, sample["reward_model"]["ground_truth"], SEED
        )
        return {
            "index": index,
            "arm": arm,
            "run_fingerprint": run_hash,
            "prompt_sha256": messages_sha256(messages),
            "memory_prompt_sha256": sha256_text(memory_prompt),
            "memory_prompt_chars": len(memory_prompt),
            "partner_prefix_sha256": record["partner_prefix_sha256"],
            "response": response,
            "response_sha256": sha256_text(response),
            "score": metrics["score"],
            "task_score": int(metrics["task_score"] == 1.0),
            "format_score": int(metrics["format_score"] == 1.0),
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
        }

    if pending:
        with paths["results"].open("a") as sink, ThreadPoolExecutor(
            max_workers=workers
        ) as pool:
            futures = {pool.submit(one, index): index for index in pending}
            for future in as_completed(futures):
                sink.write(canonical_json(future.result()) + "\n")
                sink.flush()

    rows = load_jsonl(paths["results"])
    if set(rows) != set(manifest):
        raise GateFailure(f"{arm} coverage incomplete: {len(rows)}/{len(manifest)}")
    successes = sum(row["task_score"] for row in rows.values())
    formats = sum(row["format_score"] for row in rows.values())
    summary = {
        "task": "Amendment8_VIKI_L2_Interactive",
        "status": "PASS",
        "arm": arm,
        "samples": len(rows),
        "successes": successes,
        "accuracy": successes / len(rows),
        "format_successes": formats,
        "format_compliance": formats / len(rows),
        "plan_generation_calls": len(manifest),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows.values()),
        "completion_tokens": sum(row["completion_tokens"] for row in rows.values()),
        "runtime": runtime,
        "results": str(paths["results"]),
        "results_sha256": file_sha256(paths["results"]),
    }
    atomic_json(paths["summary"], summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VIKI Amendment 8")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("build-manifest")
    subparsers.add_parser("freeze-token-protocol")
    run_parser = subparsers.add_parser("run-arm")
    run_parser.add_argument("--arm", choices=ARMS, required=True)
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    run_parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "freeze":
            result = freeze()
        elif args.command == "build-manifest":
            result = build_manifest()
        elif args.command == "freeze-token-protocol":
            result = freeze_token_protocol()
        else:
            result = run_arm(args.arm, args.base_url, args.workers)
    except GateFailure as error:
        print(f"GATE FAILURE: {error}")
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
