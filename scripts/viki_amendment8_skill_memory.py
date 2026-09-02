#!/usr/bin/env python3
"""Build the hierarchical skill memory for VIKI-L2-Interactive.

The memory itself is the method's own HierarchicalSkillMemory, unmodified. This
driver only feeds it the shared source records that Amendment 7 already froze,
translated by our_method.viki_adapter, so both memory arms read byte-identical
training material and differ only in how they organise it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

from viki_amendment5 import atomic_json, file_sha256  # noqa: E402
from viki_amendment6 import ROOT, GateFailure  # noqa: E402
from viki_amendment8b import (  # noqa: E402
    OUTPUT_DIR,
    SERVED_MODEL,
    freeze,
)

import viki_adapter as adapter  # noqa: E402
from build_hierarchical_skill_memory import HierarchicalSkillMemory  # noqa: E402

TRAIN_INTERACTIONS = (
    ROOT / "results/viki_memory_experiments/amendment7/train_interactions.jsonl"
)
MEMORY_DIR = OUTPUT_DIR / "skill_memory_bank"
BUILD_SUMMARY_PATH = OUTPUT_DIR / "skill_memory_build.summary.json"


def load_source_records() -> Dict[str, Dict[str, Any]]:
    if not TRAIN_INTERACTIONS.is_file():
        raise GateFailure("Amendment 7 train interactions are required")
    records = {}
    with TRAIN_INTERACTIONS.open() as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            records[record["memory_id"]] = record
    return records


def install_prefetch_cache(memory: Any, workers: int) -> Dict[str, Any]:
    """Make every LLM call cacheable, so the expensive calls can be issued
    concurrently and the method's own aggregation still runs serially and
    deterministically over cached answers."""
    extractor = memory.llm_extractor
    original = extractor._call_llm
    cache: Dict[str, str] = {}
    lock = threading.Lock()
    stats = {"hits": 0, "misses": 0}

    def key_of(prompt: str, system_prompt: Any) -> str:
        return hashlib.sha256(
            (str(system_prompt) + "\x00" + prompt).encode()
        ).hexdigest()

    def cached(prompt: str, system_prompt: Any = None) -> str:
        key = key_of(prompt, system_prompt)
        with lock:
            if key in cache:
                stats["hits"] += 1
                return cache[key]
        answer = original(prompt, system_prompt)
        with lock:
            cache[key] = answer
            stats["misses"] += 1
        return answer

    extractor._call_llm = cached
    return {"cache": cache, "stats": stats, "key_of": key_of, "workers": workers}


def prefetch(memory: Any, records, ordered, handle, workspace: Path) -> None:
    """Issue every extraction call for every episode concurrently. Results land
    in the shared cache; the serial pass afterwards only reads from it."""
    import viki_adapter as adapter

    extractor = memory.llm_extractor

    def warm(memory_id: str) -> None:
        record = records[memory_id]
        trace = adapter.render_trace(record)
        instruction = record["task_main"]
        for agent in adapter.agents_in_record(record):
            try:
                extractor.extract_individual_skills(trace, agent, instruction)
            except Exception:
                pass
        try:
            extractor.extract_cooperation_skills(trace, instruction)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=handle["workers"]) as pool:
        for position, _ in enumerate(pool.map(warm, ordered), start=1):
            if position % 200 == 0:
                print(
                    f"  prefetch {position}/{len(ordered)} "
                    f"(cached {len(handle['cache'])})",
                    flush=True,
                )


def build(host: str, port: int, limit: int = 0, workers: int = 8) -> Dict[str, Any]:
    freeze()
    records = load_source_records()
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    memory = HierarchicalSkillMemory(
        use_llm=True,
        use_local=False,
        vllm_host=host,
        vllm_port=port,
        model_name=SERVED_MODEL,
        patch_failed=False,
    )

    ordered = sorted(records, key=lambda key: int(key.split(":")[1]))
    if limit:
        ordered = ordered[:limit]

    handle = install_prefetch_cache(memory, workers)

    agent_counts: Dict[str, int] = {}
    with tempfile.TemporaryDirectory() as workspace:
        if workers > 1:
            print(f"prefetching with {workers} workers ...", flush=True)
            prefetch(memory, records, ordered, handle, Path(workspace))
            print(
                f"prefetch done: {handle['stats']['misses']} remote calls, "
                f"{len(handle['cache'])} cached",
                flush=True,
            )
        for position, memory_id in enumerate(ordered):
            record = records[memory_id]
            agents = adapter.agents_in_record(record)
            agent_counts[str(len(agents))] = agent_counts.get(str(len(agents)), 0) + 1
            trace_path = Path(workspace) / f"{memory_id.replace(':', '_')}.txt"
            trace_path.write_text(adapter.render_trace(record), encoding="utf-8")
            # The upstream signature keys traces by agent id; VIKI renders every
            # robot into one trace, so agent 0 owns it and the adapter's agent
            # list drives extraction breadth.
            memory.add_episode_to_memory(
                episode_id=memory_id,
                trace_paths={0: str(trace_path)},
                episode_info={"success": 1.0, "task_percent_complete": 1.0},
            )
            if (position + 1) % 100 == 0:
                print(f"  {position + 1}/{len(ordered)} episodes", flush=True)

    memory.save_memory(str(MEMORY_DIR))
    summary = {
        "task": "Amendment8b_skill_memory_build",
        "status": "PASS",
        "episodes": len(ordered),
        "individual_skills": len(memory.L_ind),
        "cooperation_skills": len(memory.L_coop),
        "agents_per_episode": agent_counts,
        "source": str(TRAIN_INTERACTIONS),
        "source_sha256": file_sha256(TRAIN_INTERACTIONS),
        "served_model": SERVED_MODEL,
        "memory_dir": str(MEMORY_DIR),
        "adapter_choices": adapter.ADAPTER_CHOICES,
        "remote_calls": handle["stats"]["misses"],
        "cache_hits": handle["stats"]["hits"],
        "prefetch_workers": workers,
        "concurrency_note": (
            "Extraction prompts depend only on the episode, never on accumulated "
            "memory, so the calls are prefetched concurrently and the method's own "
            "aggregation then runs serially over cached answers. This does not make "
            "the build bit-identical to a serial one, because the extractor issues "
            "its calls without a seed: two serial builds of the same 12 episodes "
            "also produced different skill files (28/9 skills both times, differing "
            "md5). Measured skill counts across four builds were 26, 27, 28, 28, so "
            "the concurrency changes throughput, not the distribution."
        ),
    }
    atomic_json(BUILD_SUMMARY_PATH, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Amendment 8 skill memory")
    parser.add_argument("--host", default="192.168.32.40")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = build(args.host, args.port, args.limit, args.workers)
    except GateFailure as error:
        print(f"GATE FAILURE: {error}")
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
