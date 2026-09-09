#!/usr/bin/env python3
"""Score the baseline cells that exist under a *different condition* than the main table.

The main table's 30B and 7B baseline columns are the think condition (tags `m30`, `m7`).
A no-think family (`m30nt*`) also exists and roughly doubles G-Memory, so any 30B claim has
to be labelled with its condition. Our own arm has no no-think cell at all --
`viki_eval_v2_intent_choice.py` has no such switch -- so the two conditions cannot be
compared for us, and that is the gap this file records rather than hides.

Also records the repeats, which exist for the 30B baselines and for nothing else.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "our_method"))
sys.path.insert(0, str(ROOT))

import pandas as pd                                                    # noqa: E402
import viki_p0_report as p0                                            # noqa: E402
from habitat_llm.evaluation import viki_bench as bench                 # noqa: E402
from our_method.skill_memory_v2.simulator import Simulator             # noqa: E402

A8B = ROOT / "results/viki_memory_experiments/amendment8b"
OUT = ROOT / "results/condition_variants.json"

# tag -> (model, condition, repeat)
TAGS = {"m30": ("30B", "think", 1), "m30r2": ("30B", "think", 2), "m30r3": ("30B", "think", 3),
        "m30nt": ("30B", "no-think", 1), "m30ntr2": ("30B", "no-think", 2),
        "m30ntr3": ("30B", "no-think", 3), "m7": ("7B", "think", 1),
        "m7nt": ("7B", "no-think", 1)}
ARMS = {"G-Memory": "gmemory", "trajectory RAG": "trajectory_rag", "zero-shot": "zero_shot"}


def main() -> int:
    sim = Simulator(p0.BENCHMARK_ROOT)
    frame = pd.read_parquet(p0.BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    cache = {}

    def truth(index):
        if index not in cache:
            cache[index] = bench.get_ground_truth(bench.to_native(frame.iloc[index].to_dict()))
        return cache[index]

    report = {"note": ("ID split, JSON-tolerant. The main table uses the think tags (m30, m7); "
                       "our own arm has no no-think cell, so the conditions cannot be compared "
                       "for it."),
              "arms": defaultdict(dict), "missing": []}
    for arm, stem in ARMS.items():
        for tag, (model, condition, repeat) in TAGS.items():
            path = A8B / ("%s.%s.jsonl" % (stem, tag))
            if not path.is_file():
                report["missing"].append("%s %s" % (stem, tag))
                continue
            solved = total = 0
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                index = int(record["index"])
                text = record.get("response") or record.get("raw") or ""
                solved += p0.tolerant(sim, text, truth(index))
                total += 1
            report["arms"][arm]["%s/%s/rep%d" % (model, condition, repeat)] = {
                "tag": tag, "solved": solved, "n": total, "rate": round(solved / total, 4)}
    report["arms"] = dict(report["arms"])
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("wrote %s" % OUT)
    for arm, entries in report["arms"].items():
        for key, value in sorted(entries.items()):
            print("  %-16s %-22s %s = %.4f" % (arm, key, value["tag"], value["rate"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
