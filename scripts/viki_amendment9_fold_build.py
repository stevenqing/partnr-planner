#!/usr/bin/env python3
"""Build one skill-memory bank per held-out family, reusing a single warm cache.

The extraction prompts depend only on the episode, so every fold's calls are a
subset of the full build's calls. One prefetch over all 6699 episodes therefore
warms every fold: the folds cost aggregation only, not 8 x 2h45m of LLM traffic.
The cache is also written to disk so an interrupted run resumes without repaying
for the prefetch.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

from viki_amendment5 import atomic_json, file_sha256
from viki_amendment6 import GateFailure
from viki_amendment8b import OUTPUT_DIR, SERVED_MODEL, freeze
from viki_amendment8_skill_memory import (
    TRAIN_INTERACTIONS,
    install_prefetch_cache,
    load_source_records,
    prefetch,
)
from viki_amendment9_folds import episode_families, folds, held_out_ids

import viki_adapter as adapter
from build_hierarchical_skill_memory import HierarchicalSkillMemory

CACHE_PATH = OUTPUT_DIR / "extraction_cache.json.gz"
FOLD_ROOT = OUTPUT_DIR / "folds"
FOLD_SUMMARY = OUTPUT_DIR / "fold_build.summary.json"


def load_cache() -> Dict[str, str]:
    if not CACHE_PATH.is_file():
        return {}
    with gzip.open(CACHE_PATH, "rt") as handle:
        return json.load(handle)


def save_cache(cache: Dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_suffix(".tmp")
    with gzip.open(temporary, "wt") as handle:
        json.dump(cache, handle)
    temporary.replace(CACHE_PATH)


def new_memory(host: str, port: int) -> Any:
    return HierarchicalSkillMemory(
        use_llm=True,
        use_local=False,
        vllm_host=host,
        vllm_port=port,
        model_name=SERVED_MODEL,
        patch_failed=False,
    )


def build_one(
    family: str, records, ordered, host: str, port: int, cache: Dict[str, str]
) -> Dict[str, Any]:
    removed = held_out_ids(family)
    kept = [mid for mid in ordered if mid not in removed]
    if not kept:
        raise GateFailure(f"fold {family} kept no episodes")
    target = FOLD_ROOT / family
    target.mkdir(parents=True, exist_ok=True)

    memory = new_memory(host, port)
    handle = install_prefetch_cache(memory, workers=1)
    handle["cache"].update(cache)

    started = time.time()
    with tempfile.TemporaryDirectory() as workspace:
        for position, memory_id in enumerate(kept):
            record = records[memory_id]
            trace_path = Path(workspace) / f"{memory_id.replace(':', '_')}.txt"
            trace_path.write_text(adapter.render_trace(record), encoding="utf-8")
            memory.add_episode_to_memory(
                episode_id=memory_id,
                trace_paths={0: str(trace_path)},
                episode_info={"success": 1.0, "task_percent_complete": 1.0},
            )
            if (position + 1) % 1000 == 0:
                print(f"    {position + 1}/{len(kept)}", flush=True)
    memory.save_memory(str(target))
    cache.update(handle["cache"])

    return {
        "family": family,
        "episodes_kept": len(kept),
        "episodes_removed": len(ordered) - len(kept),
        "individual_skills": len(memory.L_ind),
        "cooperation_skills": len(memory.L_coop),
        "remote_calls": handle["stats"]["misses"],
        "cache_hits": handle["stats"]["hits"],
        "bank": str(target),
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build held-out-family banks")
    parser.add_argument("--host", default="192.168.32.40")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-warm", action="store_true")
    arguments = parser.parse_args()

    freeze()
    records = load_source_records()
    ordered = sorted(records, key=lambda key: int(key.split(":")[1]))
    families = episode_families()
    if set(families) != set(records):
        raise GateFailure("family labels and source records disagree")

    cache = load_cache()
    print(f"cache loaded: {len(cache)} entries", flush=True)

    if not arguments.skip_warm:
        # One prefetch over every episode. Each fold is a subset, so this is the
        # only time the model is called.
        warm = new_memory(arguments.host, arguments.port)
        handle = install_prefetch_cache(warm, arguments.workers)
        handle["cache"].update(cache)
        print(f"warming over {len(ordered)} episodes ...", flush=True)
        prefetch(warm, records, ordered, handle, Path("."))
        cache = handle["cache"]
        save_cache(cache)
        print(
            f"warm done: {handle['stats']['misses']} remote calls, "
            f"{len(cache)} cached",
            flush=True,
        )

    results = []
    for family in folds():
        print(f"=== fold {family}", flush=True)
        outcome = build_one(family, records, ordered, arguments.host, arguments.port, cache)
        print(json.dumps(outcome), flush=True)
        results.append(outcome)
        save_cache(cache)

    summary = {
        "task": "Amendment9_fold_banks",
        "status": "PASS",
        "source": str(TRAIN_INTERACTIONS),
        "source_sha256": file_sha256(TRAIN_INTERACTIONS),
        "served_model": SERVED_MODEL,
        "folds": results,
        "cache_entries": len(cache),
    }
    atomic_json(FOLD_SUMMARY, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
