#!/usr/bin/env python3
"""Run the ported MEMENTO memory as a baseline arm on VIKI-L2.

The harness is the one the archived arms were measured in and nothing about it moves:
the benchmark's own system prompt and image, the same partner prefix, the memory block
inserted the same way, the model writing the plan itself, and the same JSON-tolerant
reading of the answer. Only the memory differs, which is the only way the number means
anything.

Two calls per row, as the method specifies: one for the model to name what the
instruction relies on, one for the plan. The retrieval between them is type-separated,
which is MEMENTO's contribution and is kept.

See `viki_memento_rag.py` for what this port can and cannot be said to show.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
from openai import OpenAI

from habitat_llm.evaluation import viki_bench as bench
from habitat_llm.evaluation.viki_memory_skill import add_memory_to_messages
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment9_diag102 import parse_plan
from viki_amendment11_goalparse import extract_json
from viki_memento_rag import EXTRACTION_PROMPT, MementoRAG, TOP_K, build
from our_method.skill_memory_v2 import SEED, Simulator
from our_method.skill_memory_v2.build import load_episodes

OUT = ROOT / "results/viki_memory_experiments/amendment11"
MANIFEST = ROOT / "results/viki_memory_experiments/amendment8b/interactive_manifest.jsonl"
SPLITS = {"id": None,
          "recombination-imaged": "recombination.imaged.parquet",
          "recombination-text": "recombination.text.parquet"}


def tolerant(sim: Simulator, response: str, truth: Dict[str, Any], seed: int) -> int:
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
    return int(bool(ok) and len(truth["time_steps"]) / len(transformed) >= 0.99)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-vl-72b-amendment3-f2")
    parser.add_argument("--base-url", default="http://192.168.32.40:8050/v1")
    parser.add_argument("--split", choices=sorted(SPLITS), default="id")
    parser.add_argument("--exclude-family", default=None, help="held-out fold")
    parser.add_argument("--only-family", default=None,
                        help="score only this family's rows, for a held-out fold")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--aggregate", action="store_true",
                        help="one node per family instead of one per episode")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--tag", default="memento")
    arguments = parser.parse_args()

    sim = Simulator(BENCHMARK_ROOT)
    episodes = load_episodes(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/train.parquet")
    graph = build(episodes[::2], set(sim.viki2.CONTAINER_ASSETS), arguments.exclude_family,
                  per_episode=not arguments.aggregate)
    print(f"ported graph: {graph['counts']}", flush=True)
    rag = MementoRAG(graph)

    if arguments.split == "id":
        manifest = {int(json.loads(l)["index"]): json.loads(l)
                    for l in MANIFEST.read_text().splitlines() if l.strip()}
        frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
        indices = sorted(manifest)
    else:
        manifest = {}
        frame = pd.read_parquet(
            ROOT / "results/viki_memory_experiments/amendment10" / SPLITS[arguments.split])
        indices = list(range(len(frame)))
    if arguments.only_family:
        keep = []
        for index in indices:
            truth = bench.get_ground_truth(bench.to_native(frame.iloc[index].to_dict()))
            if truth.get("task_name") == arguments.only_family:
                keep.append(index)
        indices = keep
    imaged = arguments.split != "recombination-text"
    if arguments.limit:
        indices = indices[: arguments.limit]
    client = OpenAI(api_key="EMPTY", base_url=arguments.base_url, max_retries=5, timeout=3600)
    lock, done = threading.Lock(), [0]

    def work(index: int) -> Dict[str, Any]:
        sample = bench.to_native(frame.iloc[index].to_dict())
        truth = bench.get_ground_truth(sample)
        record = {"index": index, "task_name": truth.get("task_name", "?"), "score": 0}
        instruction = truth.get("description") or ""

        # Stage one: the model names what the instruction relies on, by type.
        extracted = {"object_semantics": [], "user_pattern": []}
        try:
            reply = client.chat.completions.create(
                model=arguments.model, temperature=0, max_tokens=300,
                messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(instruction=instruction)}],
            ).choices[0].message.content or ""
            parsed = extract_json(reply)
            if isinstance(parsed, dict):
                extracted = {k: parsed.get(k) or [] for k in extracted}
        except Exception as error:
            record["extract_error"] = type(error).__name__
        record["extracted"] = extracted

        retrieved = rag.retrieve(extracted, arguments.top_k)
        block = rag.to_natural_language(retrieved)
        record["memory_chars"] = len(block)

        # Stage two: the plan, in the harness the archived arms were measured in.
        if imaged:
            messages = bench.get_messages(sample)
        else:
            messages = [{"role": m["role"], "content": m["content"]}
                        for m in bench.to_native(sample["prompt"])]
        user = next(m for m in reversed(messages) if m["role"] == "user")
        partner = manifest.get(index, {}).get("partner_prefix")
        if partner:
            content = user["content"]
            if isinstance(content, list):
                item = next(i for i in content if i["type"] == "text")
                item["text"] = f"{item['text']}\n\n{partner}"
            else:
                user["content"] = f"{content}\n\n{partner}"
        messages = add_memory_to_messages(messages, block)
        try:
            text = client.chat.completions.create(
                model=arguments.model, messages=messages, temperature=0, max_tokens=2000,
            ).choices[0].message.content or ""
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
    with (OUT / f"{arguments.tag}.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    hit = sum(r["score"] for r in records)
    print(f"\n=== {arguments.tag}: MEMENTO-style memory, model writes the plan, "
          f"{len(records)} rows (JSON-tolerant) ===")
    print(f"model          {arguments.model}   split {arguments.split}   top-k {arguments.top_k}")
    print(f"graph          {graph['counts']}   held out: {arguments.exclude_family}")
    print(f"memory block   {sum(r.get('memory_chars', 0) for r in records) / max(1, len(records)):.0f}"
          f" chars mean")
    print(f"accuracy       {hit}/{len(records)} = {hit / len(records) * 100:.2f}%")
    families = defaultdict(lambda: [0, 0])
    for record in records:
        families[record["task_name"]][0] += record["score"]
        families[record["task_name"]][1] += 1
    print("\nby family:")
    for family, (won, total) in sorted(families.items(), key=lambda kv: -kv[1][1]):
        print(f"  {family:<48} {won:>4}/{total:<4} {won / total * 100:5.1f}%")
    print(f"\nwrote {OUT / (arguments.tag + '.jsonl')}")


if __name__ == "__main__":
    main()
