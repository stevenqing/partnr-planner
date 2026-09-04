#!/usr/bin/env python3
"""Build the prompt-shaped memories the compositional comparison needs, from one pile.

Three arms, one training pile, one consumption path. What differs between them is the
representation, which is the variable the paper is about:

  react_rag_R   PARTNR's own trajectory retrieval, pool restricted to the same
                rearrange-only episodes (built by scripts/partnr_make_rag_source.py)
  gmemory       G-Memory's hierarchical trajectory graph plus distilled insights
  memento       MEMENTO's type-separated user-profile knowledge graph

Restricting every pool to the rearrange-only half of `train_mini` is what makes the
comparison the paper's: skill memory v2's operators are induced from exactly those 161
rollouts, so an unrestricted baseline would be one that has seen spatial and temporal
work the operators never did. The unrestricted arm answers "is retrieval enough"; only
the restricted one answers "does this memory compose".

MEMENTO's query-time entity extraction is precomputed here for the evaluation split's
instructions rather than called inside the planner loop. The extraction is a pure
function of the instruction and the instruction does not change within an episode, so
precomputing it changes the arm's behaviour not at all, keeps the eval deterministic, and
stops one answer costing fifty identical model calls.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT))

from our_method.partnr_baselines import gmemory as gmemory_port  # noqa: E402
from our_method.partnr_baselines import memento as memento_port  # noqa: E402
from our_method.partnr_baselines import substrate  # noqa: E402


def make_client(base_url: str, model: str, temperature: float, max_tokens: int):
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    lock = threading.Lock()
    calls = {"n": 0}

    def complete(messages: List[Dict[str, str]]) -> str:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, seed=gmemory_port.SEED)
        with lock:
            calls["n"] += 1
        return response.choices[0].message.content or ""

    return complete, calls


def build_memento(arguments) -> None:
    records = substrate.load(arguments.split, arguments.types or None)
    built = memento_port.build(records)
    arguments.memento_out.write_text(json.dumps(built, indent=1) + "\n")
    print(f"memento: {built['counts']} nodes from {len(records)} episodes "
          f"-> {arguments.memento_out}")


def build_extractions(arguments, complete, calls) -> None:
    """MEMENTO's first step, run once per distinct instruction in the eval split."""
    with gzip.open(substrate.DATASET / f"{arguments.eval_split}.json.gz") as handle:
        episodes = json.load(handle)["episodes"]
    instructions = sorted({(e.get("instruction") or "").strip() for e in episodes} - {""})
    print(f"extractions: {len(instructions)} distinct instructions in {arguments.eval_split}")

    def one(instruction: str):
        prompt = memento_port.EXTRACTION_PROMPT.format(instruction=instruction)
        return instruction, memento_port.parse_extraction(
            complete([{"role": "user", "content": prompt}]))

    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        table = dict(pool.map(one, instructions))
    empty = sum(1 for value in table.values()
                if not value["object_semantics"] and not value["user_pattern"])
    arguments.extractions_out.write_text(json.dumps(table, indent=1) + "\n")
    print(f"extractions: wrote {len(table)} ({empty} came back empty), "
          f"{calls['n']} calls -> {arguments.extractions_out}")


def build_gmemory(arguments, complete, calls) -> None:
    records = substrate.load(arguments.split, arguments.types or None)
    embed = None

    def embed_texts(texts):
        nonlocal embed
        if embed is None:
            from our_method.partnr_baselines.prompt_memory import _sentence_embedder

            embed = _sentence_embedder(arguments.device)
        return embed(texts)

    state, embeddings = gmemory_port.build(records, complete, embed_texts, progress=print,
                                           workers=arguments.workers)
    gmemory_port.save(state, embeddings, arguments.gmemory_out)
    # Loading it back here is the cheap proof the artifact is usable: the restore path
    # verifies the embedding hashes, so a silent precision loss fails now rather than
    # eight hours into the sweep.
    gmemory_port.restore(arguments.gmemory_out)
    print(f"gmemory: {len(state.records)} records, {len(state.insights)} insights, "
          f"{calls['n']} calls -> {arguments.gmemory_out}")
    for insight in state.insights[:5]:
        print(f"  - {insight['rule'][:160]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", nargs="+",
                        choices=["memento", "extractions", "gmemory", "all"],
                        default=["all"])
    parser.add_argument("--split", default="train_mini", help="where the memory is built from")
    parser.add_argument("--types", nargs="*", default=["R"],
                        help="task types to build from; the paper's cell is R")
    parser.add_argument("--eval-split", default="val_mini",
                        help="where the memory will be used, for the extraction table")
    parser.add_argument("--base-url", default="http://127.0.0.1:8062/v1")
    parser.add_argument("--model", default="qwen3-vl-30b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--memento-out", type=Path,
                        default=ROOT / "results/partnr_memento_train_mini_R.json")
    parser.add_argument("--extractions-out", type=Path,
                        default=ROOT / "results/partnr_memento_extractions_val_mini.json")
    parser.add_argument("--gmemory-out", type=Path,
                        default=ROOT / "results/partnr_gmemory_train_mini_R.json")
    arguments = parser.parse_args()

    wanted = set(arguments.what)
    if "all" in wanted:
        wanted = {"memento", "extractions", "gmemory"}

    complete = calls = None
    if wanted & {"extractions", "gmemory"}:
        complete, calls = make_client(arguments.base_url, arguments.model,
                                      arguments.temperature, arguments.max_tokens)

    if "memento" in wanted:
        build_memento(arguments)
    if "extractions" in wanted:
        build_extractions(arguments, complete, calls)
    if "gmemory" in wanted:
        build_gmemory(arguments, complete, calls)


if __name__ == "__main__":
    main()
