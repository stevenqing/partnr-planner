"""The 30B baseline table with a spread, and the no-think condition kept apart from it.

Three runs of the same twelve cells under identical conditions, re-scored here from raw
responses with the same JSON-tolerant reading `viki_p0_report.py` uses -- nothing is taken
from a summary and no model is called. These arms are not greedy, so a single run is a
single draw; this is what the repeats were for.

`NOTHINK=1` drops the thinking instruction from the benchmark's own system prompt. It is a
NEW CONDITION, not a fix, so its three runs are reported as their own table and are never
averaged with the think ones.
"""
import argparse, json, statistics, sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from habitat_llm.evaluation import viki_bench as bench  # noqa: E402
from viki_amendment5 import BENCHMARK_ROOT  # noqa: E402
from viki_amendment9_diag102 import parse_plan  # noqa: E402
from viki_report_matrix import tolerant  # noqa: E402
from our_method.skill_memory_v2 import Simulator  # noqa: E402

A8B = ROOT / "results/viki_memory_experiments/amendment8b"
A10 = ROOT / "results/viki_memory_experiments/amendment10"
METHODS = [("zero-shot", "zero_shot"), ("trajectory RAG", "trajectory_rag"),
           ("skill memory v1", "skill_memory.fullactions_k8"), ("G-Memory", "gmemory")]
SPLITS = ["id", "imaged", "text"]
THINK = ["m30", "m30r2", "m30r3"]
NOTHINK = ["m30nt", "m30ntr2", "m30ntr3"]


def cell_path(stem: str, split: str, variant: str) -> Path:
    if split == "id":
        name = f"{stem}_{variant}" if stem.endswith("k8") else f"{stem}.{variant}"
        return A8B / f"{name}.jsonl"
    short = "skill_memory" if stem.startswith("skill_memory") else stem
    return A10 / split / f"{short}.{variant}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=ROOT / "outputs/p0_30b_repeats.json")
    arguments = parser.parse_args()

    sim = Simulator(BENCHMARK_ROOT)
    frames = {"id": pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet"),
              "imaged": pd.read_parquet(A10 / "recombination.imaged.parquet"),
              "text": pd.read_parquet(A10 / "recombination.text.parquet")}
    truths: Dict[str, Dict[int, object]] = {}

    def truth_of(split: str, index: int):
        table = truths.setdefault(split, {})
        if index not in table:
            table[index] = bench.get_ground_truth(bench.to_native(frames[split].iloc[index].to_dict()))
        return table[index]

    cells: Dict[str, Dict] = {}
    per_index_all: Dict[str, Dict[int, int]] = {}
    for label, stem in METHODS:
        for split in SPLITS:
            for variant in THINK + NOTHINK:
                path = cell_path(stem, split, variant)
                key = f"{label}|{split}|{variant}"
                if not path.is_file():
                    cells[key] = {"status": "absent", "path": str(path)}
                    continue
                per_index: Dict[int, int] = {}
                fmt = parses = 0
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    index = int(record["index"])
                    text = record.get("response") or record.get("raw") or ""
                    per_index[index] = tolerant(sim, text, truth_of(split, index))
                    fmt += int(round(float(record.get("format_score", 0) or 0)))
                    parses += int(parse_plan(text) is not None)
                rows = len(per_index)
                per_index_all[key] = per_index
                cells[key] = {"status": "scored", "numerator": sum(per_index.values()),
                              "denominator": rows,
                              "accuracy": sum(per_index.values()) / rows if rows else None,
                              "official_format": fmt / rows if rows else None,
                              "parseable": parses / rows if rows else None,
                              "path": str(path.relative_to(ROOT))}
                print(f"scored {key}: {cells[key]['numerator']}/{rows}", flush=True)

    def table(name: str, variants: List[str]):
        print()
        print(f"== Qwen3-VL-30B, {name} (three runs, identical conditions) ==")
        header = "  " + "method".ljust(18) + "".join(
            {"id": "ID (924)", "imaged": "recomb imaged (297)", "text": "recomb text (297)"}[s].rjust(24)
            for s in SPLITS)
        print(header)
        out = {}
        for label, _ in METHODS:
            row = "  " + label.ljust(18)
            for split in SPLITS:
                values = [cells[f"{label}|{split}|{v}"].get("accuracy") for v in variants]
                values = [v for v in values if v is not None]
                if len(values) < 2:
                    row += "n/a".rjust(24)
                    continue
                mean = statistics.mean(values)
                sd = statistics.stdev(values)
                out[f"{label}|{split}"] = {"mean": mean, "sd": sd, "runs": values}
                row += f"{mean * 100:6.2f}% +/- {sd * 100:4.2f}".rjust(24)
            print(row)
        print("  per-run values:")
        for label, _ in METHODS:
            for split in SPLITS:
                entry = out.get(f"{label}|{split}")
                if entry:
                    print("    %-18s %-7s %s" % (label, split,
                          "  ".join(f"{v * 100:6.2f}%" for v in entry["runs"])))
        return out

    think = table("think", THINK)
    nothink = table("NO-THINK -- a separate condition, never merged with the table above", NOTHINK)

    print()
    print("== think vs no-think, same cells, reported side by side and not pooled ==")
    for label, _ in METHODS:
        for split in SPLITS:
            a, b = think.get(f"{label}|{split}"), nothink.get(f"{label}|{split}")
            if a and b:
                print("    %-18s %-7s think %6.2f%% +/-%4.2f   no-think %6.2f%% +/-%4.2f   delta %+6.2f"
                      % (label, split, a["mean"] * 100, a["sd"] * 100,
                         b["mean"] * 100, b["sd"] * 100, (b["mean"] - a["mean"]) * 100))

    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(json.dumps(
        {"cells": cells, "think": think, "nothink": nothink,
         "note": "no-think is a separate condition; the two tables are never pooled"},
        indent=1))
    print()
    print("->", arguments.json)


if __name__ == "__main__":
    main()
