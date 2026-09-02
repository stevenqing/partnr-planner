#!/usr/bin/env python3
"""Does v2 make better memory for the model itself, and not only for a planner?

The obvious objection to skill memory v2 is that the model barely appears in it: it says
what must become true and a program does the rest, so the gain could be the planner
rather than the memory. This settles that by putting v2 back into the job v1 and
G-Memory were measured on -- the model reads the memory and writes the plan itself, with
the benchmark's own system prompt, the same partner prefix the archived arms were given,
and the same JSON-tolerant scoring. Only the memory block differs.

v2 is handed over whole. Nineteen operators, three ordering rules and eight place names
come to about nine hundred tokens, less than the retrieved trajectories the G-Memory arm
was given, so there is no retrieval step and nothing is selected for the row. Whatever
this arm scores is the representation, not a lucky neighbour -- which is exactly the
comparison v1's 16.77% cannot make, because its eight hundred skills only fit through a
retriever.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
from openai import OpenAI

from habitat_llm.evaluation import viki_bench as bench
from habitat_llm.evaluation.viki_memory_skill import add_memory_to_messages
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment9_diag102 import parse_plan
from our_method.skill_memory_v2 import SEED, SkillMemoryV2, Simulator

OUT = ROOT / "results/viki_memory_experiments/amendment11"
MANIFEST = ROOT / "results/viki_memory_experiments/amendment8b/interactive_manifest.jsonl"
SERVED_MODEL = "qwen2.5-vl-72b-amendment3-f2"
BASE_URL = "http://192.168.32.40:8050/v1"
PLAN_MAX_TOKENS = 2000
PLAN_TEMPERATURE = 0


def tolerant(sim: Simulator, response: str, truth: Dict[str, Any], seed: int) -> int:
    """The archived口径: read the answer as JSON, drop idle robots, then score officially.

    The official reader is ast.literal_eval, which chokes on the `null` a model writes
    for a robot that stands still and fails four rows in five for reasons that have
    nothing to do with planning. Every number this line reports is taken this way.
    """
    parsed = parse_plan(response)
    if parsed is None:
        return 0
    if isinstance(parsed, list):
        for step in parsed:
            if isinstance(step, dict) and isinstance(step.get("actions"), dict):
                step["actions"] = {k: v for k, v in step["actions"].items() if v is not None}
    transformed = sim.scorer.transform_actions(parsed)
    if not transformed:
        return 0
    globals_ = sim.scorer.eval_single.__globals__
    original = globals_["random"]
    try:
        globals_["random"] = random.Random(seed)
        ok = sim.scorer.eval_single(transformed, truth)
    except Exception:
        return 0
    finally:
        globals_["random"] = original
    if not ok:
        return 0
    return int(len(truth["time_steps"]) / len(transformed) >= 0.99)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["v2", "none"], default="v2")
    parser.add_argument("--memory", type=Path, default=OUT / "skill_memory_v2.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--tag", default=None)
    arguments = parser.parse_args()

    sim = Simulator(BENCHMARK_ROOT)
    block = SkillMemoryV2.load(arguments.memory).render() if arguments.arm == "v2" else ""
    manifest = {
        int(json.loads(line)["index"]): json.loads(line)
        for line in MANIFEST.read_text().splitlines() if line.strip()
    }
    indices = sorted(manifest)
    if arguments.limit:
        indices = indices[: arguments.limit]
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    client = OpenAI(api_key="EMPTY", base_url=BASE_URL, max_retries=5, timeout=3600)
    lock, done = threading.Lock(), [0]

    def work(index: int) -> Dict[str, Any]:
        sample = bench.to_native(frame.iloc[index].to_dict())
        truth = bench.get_ground_truth(sample)
        messages = bench.get_messages(sample)
        user = next(m for m in reversed(messages) if m["role"] == "user")
        partner = manifest[index]["partner_prefix"]
        content = user["content"]
        if isinstance(content, list):
            item = next(i for i in content if i["type"] == "text")
            item["text"] = f"{item['text']}\n\n{partner}"
        else:
            user["content"] = f"{content}\n\n{partner}"
        messages = add_memory_to_messages(messages, block)
        record = {"index": index, "task_name": truth["task_name"], "score": 0}
        try:
            completion = client.chat.completions.create(
                model=SERVED_MODEL, messages=messages,
                temperature=PLAN_TEMPERATURE, max_tokens=PLAN_MAX_TOKENS,
            )
            text = completion.choices[0].message.content or ""
        except Exception as error:
            record["reason"] = f"REQUEST_FAILED:{type(error).__name__}"
            return record
        record["raw"] = text[-6000:]
        record["score"] = tolerant(sim, text, truth, SEED)
        return record

    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        futures = [pool.submit(work, index) for index in indices]
        for future in as_completed(futures):
            record = future.result()
            with lock:
                records.append(record)
                done[0] += 1
                if done[0] % 100 == 0:
                    hit = sum(r["score"] for r in records)
                    print(f"  {done[0]}/{len(indices)}  solved={hit} "
                          f"({hit / len(records) * 100:.1f}%)", flush=True)

    records.sort(key=lambda r: r["index"])
    tag = arguments.tag or f"inprompt_{arguments.arm}"
    with (OUT / f"{tag}.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    hit = sum(r["score"] for r in records)
    print(f"\n=== {tag}: the model writes the plan, {len(records)} rows (JSON-tolerant) ===")
    print(f"memory block   {len(block)} chars (~{len(block)//4} tokens)")
    print(f"accuracy       {hit}/{len(records)} = {hit / len(records) * 100:.2f}%")
    families = defaultdict(lambda: [0, 0])
    for record in records:
        families[record["task_name"]][0] += record["score"]
        families[record["task_name"]][1] += 1
    print("\nby family:")
    for family, (won, total) in sorted(families.items(), key=lambda kv: -kv[1][1]):
        print(f"  {family:<48} {won:>4}/{total:<4} {won / total * 100:5.1f}%")
    print(f"\nwrote {OUT / (tag + '.jsonl')}")


if __name__ == "__main__":
    main()
