#!/usr/bin/env python3
"""Run the arms over the recombination split, in both its imaged and text forms.

This is a separate runner rather than a flag on the Amendment 8b one. That script
reads its parquet, manifest and output paths from module constants and checks a
freeze certificate over them, so pointing it at a new dataset would mean either
editing a filed script or overriding its constants at runtime -- and its artefacts
are what every earlier result is certified against. The loop is small enough to
restate; the parts that must not drift (message assembly, partner prefix, scoring,
the memory providers) are imported from it rather than copied.

Both split variants carry identical ground truth and differ only in what the model
is shown, so the pair doubles as the control for the one question the metadata could
not answer: whether an asset restored to init_pos is actually in the row's picture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd
from openai import OpenAI

import viki_fork_guard

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT, atomic_json, file_sha256, messages_sha256
from viki_amendment6 import GateFailure
from viki_amendment8b import (
    EXPOSED_STEPS,
    PLAN_MAX_TOKENS,
    PLAN_TEMPERATURE,
    SEED,
    SELF_ROBOT,
    SERVED_MODEL,
    interactive_messages,
    make_memory,
    native,
    partner_prefix,
    qualifies,
    sha256_text,
)

SPLIT_DIR = Path(
    "/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner/results/viki_memory_experiments/amendment10"
)
ARMS = ("zero_shot", "trajectory_rag", "gmemory", "skill_memory")


def build_messages(
    sample: Dict[str, Any], partner_text: str, memory_prompt: str, imaged: bool
) -> List[Dict[str, Any]]:
    """The imaged split goes through the benchmark's own assembly. The text split
    cannot: get_messages fetches the image unconditionally and the row has none.
    Everything else is kept identical -- the same prompt messages, the same partner
    prefix appended to the last user turn, the same memory insertion -- so the two
    variants differ in the image and nothing else."""
    if imaged:
        return interactive_messages(sample, partner_text, memory_prompt)

    from habitat_llm.evaluation.viki_memory_skill import add_memory_to_messages
    from habitat_llm.evaluation.viki_bench import to_native

    messages = [
        {"role": message["role"], "content": message["content"]}
        for message in to_native(sample["prompt"])
    ]
    if any("<image>" in str(m["content"]) for m in messages):
        raise GateFailure("text split still carries an <image> marker")
    user = next(m for m in reversed(messages) if m["role"] == "user")
    user["content"] = f"{user['content']}\n\n{partner_text}"
    return add_memory_to_messages(messages, memory_prompt)


def split_path(variant: str) -> Path:
    path = SPLIT_DIR / f"recombination.{variant}.parquet"
    if not path.is_file():
        raise GateFailure(f"Split not built: {path}")
    return path


def manifest_for(variant: str) -> Dict[int, Dict[str, Any]]:
    """Rows usable as interactive: a partner acts inside the exposed prefix and the
    evaluated robot still has work after it. Recomputed from the recombined plans,
    because the reference plan changed and a stale prefix would describe a partner
    doing something this task never asks for."""
    frame = pd.read_parquet(split_path(variant))
    records: Dict[int, Dict[str, Any]] = {}
    for index in range(len(frame)):
        truth = native(frame.iloc[index].to_dict())["reward_model"]["ground_truth"]
        if not qualifies(truth):
            continue
        prefix = partner_prefix(truth)
        records[index] = {
            "index": index,
            "task_id": truth.get("task_id"),
            "source_row": truth.get("source_row"),
            "partner_prefix": prefix,
            "partner_prefix_sha256": sha256_text(prefix),
        }
    return records


def run(variant: str, arm: str, base_url: str, workers: int, tag: str) -> Dict[str, Any]:
    frame = pd.read_parquet(split_path(variant))
    manifest = manifest_for(variant)
    if not manifest:
        raise GateFailure(f"No usable rows in the {variant} split")
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    # See viki_fork_guard: one fork per request, from a pool worker, in a process with
    # torch resident -- the hang that has twice stopped this runner with an idle
    # endpoint and a zero-byte output.
    viki_fork_guard.install()
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    name = f"{arm}.{tag}" if tag else arm
    out = SPLIT_DIR / variant
    out.mkdir(parents=True, exist_ok=True)
    results = out / f"{name}.jsonl"
    run_json = out / f"{name}.jsonl.run.json"

    metadata = {
        "task": "Amendment10_recombination_run",
        "split": variant,
        "arm": arm,
        "tag": tag,
        "served_model": SERVED_MODEL,
        "seed": SEED,
        "rows": len(manifest),
        "split_sha256": file_sha256(split_path(variant)),
        "env": {
            key: os.environ.get(key, "")
            for key in (
                "A8B_SKILL_TOPK",
                "A9_ACTION_CAP",
                "A9_STEP_ALIGNED",
                "A9_ROLE_AWARE",
                "A9_PATTERN_SLOTS",
                "A9_MODE",
            )
        },
    }
    if run_json.is_file() and json.loads(run_json.read_text()) != metadata:
        raise GateFailure(f"Cannot resume {name} on {variant}: metadata differs")
    atomic_json(run_json, metadata)

    done = set()
    if results.is_file():
        with results.open() as handle:
            for line in handle:
                if line.strip():
                    done.add(int(json.loads(line)["index"]))
    pending = [index for index in sorted(manifest) if index not in done]
    print(f"{variant}/{name}: {len(done)} done, {len(pending)} to run")

    provider = make_memory(arm, client)
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()

    def one(index: int) -> Dict[str, Any]:
        sample = native(frame.iloc[index].to_dict())
        record = manifest[index]
        memory_prompt = "" if provider is None else provider.prompt(index, sample)
        messages = build_messages(
            sample, record["partner_prefix"], memory_prompt, variant == "imaged"
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
            "split": variant,
            "run_fingerprint": fingerprint,
            "prompt_sha256": messages_sha256(messages),
            "memory_prompt_sha256": sha256_text(memory_prompt),
            "memory_prompt_chars": len(memory_prompt),
            "response": response,
            "response_sha256": sha256_text(response),
            "score": metrics["score"],
            "task_score": int(metrics["task_score"] == 1.0),
            "format_score": int(metrics["format_score"] == 1.0),
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
        }

    if pending:
        with results.open("a") as sink, ThreadPoolExecutor(
            max_workers=workers
        ) as pool:
            futures = {pool.submit(one, index): index for index in pending}
            for future in as_completed(futures):
                sink.write(json.dumps(future.result(), sort_keys=True) + "\n")
                sink.flush()

    rows = {}
    with results.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows[record["index"]] = record
    if set(rows) != set(manifest):
        raise GateFailure(
            f"{name} on {variant}: {len(rows)}/{len(manifest)} rows"
        )
    summary = {
        "task": "Amendment10_recombination",
        "split": variant,
        "arm": name,
        "rows": len(rows),
        "successes": sum(r["task_score"] for r in rows.values()),
        "format_successes": sum(r["format_score"] for r in rows.values()),
        "accuracy": sum(r["task_score"] for r in rows.values()) / len(rows),
        "results_sha256": file_sha256(results),
    }
    atomic_json(out / f"{name}.summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def preview(variant: str) -> None:
    """Show what the model will be given, before any inference is paid for."""
    frame = pd.read_parquet(split_path(variant))
    manifest = manifest_for(variant)
    index = sorted(manifest)[0]
    sample = native(frame.iloc[index].to_dict())
    messages = build_messages(
        sample, manifest[index]["partner_prefix"], "", variant == "imaged"
    )
    print(f"split {variant}: {len(manifest)}/{len(frame)} rows usable")
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            rendered = " | ".join(
                item.get("text", f"<{item.get('type')}>")[:400] for item in content
            )
        else:
            rendered = str(content)[:700]
        print(f"--- {message['role']} ---")
        print(rendered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("imaged", "text"), required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--base-url", default="http://192.168.32.40:8050/v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tag", default="")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.preview:
        preview(args.split)
        return
    if not args.arm:
        raise SystemExit("--arm is required unless --preview")
    run(args.split, args.arm, args.base_url, args.workers, args.tag)


if __name__ == "__main__":
    main()
