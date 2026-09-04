#!/usr/bin/env python3
"""The 30B baseline table, scored the way the 72B one was, with the metadata to check it.

Every cell is re-scored here from its own raw responses under the JSON-tolerant reading --
`json.loads` on the answer with null actions dropped, then the official simulator check
and the `len(gt)/len(pred) >= 0.99` length bound. Nothing is inherited from a summary, and
nothing here calls a model.

Two things are reported that the 72B table does not carry, because the second model made
them necessary:

  parseable   the fraction of responses an answer can be read out of. `format_score` is
              the official check and it requires `<think>`; Qwen3-VL-30B writes
              `<reasoning>`, so its official format is 0.0000 on every cell while 92% of
              its answers are perfectly well formed. Reporting only the official number
              would say the model failed when the harness simply does not recognise its
              tag, so both are printed and the official one is never used as a gate.
  tokens      served-side prompt and completion means, which the MEMENTO cells lack and
              new cells should not.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
A11 = ROOT / "results/viki_memory_experiments/amendment11"

METHODS = [
    ("zero-shot", "zero_shot", "zero_shot"),
    ("trajectory RAG", "trajectory_rag", "trajectory_rag"),
    # The 72B recombination archive keeps the configuration in the filename too.
    ("skill memory v1", "skill_memory.fullactions_k8", "skill_memory.fullactions_k8"),
    ("G-Memory", "gmemory", "gmemory"),
]
SPLITS = ["id", "imaged", "text"]


def cell_path(model: str, method: Tuple[str, str, str], split: str) -> Optional[Path]:
    _, id_stem, recomb_stem = method
    if model == "72B":
        return (A8B / f"{id_stem}.jsonl" if split == "id"
                else A10 / split / f"{recomb_stem}.jsonl")
    if split == "id":
        stem = f"{id_stem}_m30" if id_stem.endswith("k8") else f"{id_stem}.m30"
        return A8B / f"{stem}.jsonl"
    stem = "skill_memory" if recomb_stem.startswith("skill_memory") else recomb_stem
    return A10 / split / f"{stem}.m30.jsonl"


def mcnemar(a: Dict[int, int], b: Dict[int, int]) -> Tuple[int, int, float]:
    """Exact two-sided test on the rows where the two arms disagree."""
    shared = sorted(set(a) & set(b))
    n01 = sum(1 for i in shared if not a[i] and b[i])
    n10 = sum(1 for i in shared if a[i] and not b[i])
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    tail = sum(comb(n, k) for k in range(0, min(n01, n10) + 1)) / (2 ** n)
    return n10, n01, min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["72B", "30B"])
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args()

    sim = Simulator(BENCHMARK_ROOT)
    frames = {
        "id": pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet"),
        "imaged": pd.read_parquet(A10 / "recombination.imaged.parquet"),
        "text": pd.read_parquet(A10 / "recombination.text.parquet"),
    }
    truths: Dict[str, Dict[int, object]] = {}

    def truth_of(split: str, index: int):
        table = truths.setdefault(split, {})
        if index not in table:
            table[index] = bench.get_ground_truth(
                bench.to_native(frames[split].iloc[index].to_dict()))
        return table[index]

    scored: Dict[Tuple[str, str, str], Dict[int, int]] = {}
    meta: List[Dict[str, object]] = []

    for model in arguments.models:
        for method in METHODS:
            for split in SPLITS:
                path = cell_path(model, method, split)
                if path is None or not path.is_file():
                    meta.append({"model": model, "method": method[0], "split": split,
                                 "status": "absent", "path": str(path)})
                    continue
                per_index: Dict[int, int] = {}
                fmt = parses = 0
                prompt_tokens: List[float] = []
                completion_tokens: List[float] = []
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    index = int(record["index"])
                    text = record.get("response") or record.get("raw") or ""
                    per_index[index] = tolerant(sim, text, truth_of(split, index))
                    fmt += int(round(float(record.get("format_score", 0) or 0)))
                    parses += int(parse_plan(text) is not None)
                    for key, sink in (("prompt_tokens", prompt_tokens),
                                      ("completion_tokens", completion_tokens)):
                        value = record.get(key)
                        if isinstance(value, (int, float)):
                            sink.append(float(value))
                rows = len(per_index)
                scored[(model, method[0], split)] = per_index
                meta.append({
                    "model": model, "method": method[0], "split": split, "status": "scored",
                    "numerator": sum(per_index.values()), "denominator": rows,
                    "accuracy": sum(per_index.values()) / rows if rows else None,
                    "official_format": fmt / rows if rows else None,
                    "parseable": parses / rows if rows else None,
                    "prompt_tokens_mean": (sum(prompt_tokens) / len(prompt_tokens)
                                           if prompt_tokens else None),
                    "completion_tokens_mean": (sum(completion_tokens) / len(completion_tokens)
                                               if completion_tokens else None),
                    "path": str(path.relative_to(ROOT)),
                })

    label = {"id": "ID (924)", "imaged": "recomb imaged (297)", "text": "recomb text (297)"}
    for model in arguments.models:
        print(f"\n== {model} ==")
        print("  " + "method".ljust(18) + "".join(label[s].rjust(22) for s in SPLITS))
        for method in METHODS:
            cells = []
            for split in SPLITS:
                key = (model, method[0], split)
                if key not in scored:
                    cells.append("-".rjust(22))
                    continue
                got = scored[key]
                n, d = sum(got.values()), len(got)
                cells.append(f"{n}/{d} = {n/d:6.2%}".rjust(22))
            print("  " + method[0].ljust(18) + "".join(cells))

    print("\n== per-cell metadata ==")
    print("  " + "model/method/split".ljust(46) + "num/den".rjust(11)
          + "acc".rjust(9) + "offFmt".rjust(8) + "parse".rjust(8)
          + "ptok".rjust(7) + "ctok".rjust(7))
    for row in meta:
        name = f"{row['model']}/{row['method']}/{row['split']}"
        if row["status"] != "scored":
            print("  " + name.ljust(46) + "   ABSENT")
            continue
        def num(value, spec):
            return format(value, spec) if value is not None else "-"
        print("  " + name.ljust(46)
              + f"{row['numerator']}/{row['denominator']}".rjust(11)
              + num(row["accuracy"], "9.2%") + num(row["official_format"], "8.2%")
              + num(row["parseable"], "8.2%")
              + num(row["prompt_tokens_mean"], "7.0f") + num(row["completion_tokens_mean"], "7.0f"))

    # ---------------------------------------------------------------- paired tests
    #
    # Against the v2 arm that is actually archived for this model. On the 30B that is the
    # memory-dispatch arm; the LLM-orchestration variant exists only for the 72B, so it is
    # named as a gap rather than quietly replaced by a different arm.
    v2_files = {
        ("30B", "id"): A11 / "m30_id.jsonl",
        ("30B", "imaged"): A11 / "m30_recomb_imaged.jsonl",
        ("30B", "text"): A11 / "m30_recomb_text.jsonl",
        ("72B", "id"): A11 / "intent_clean.jsonl",
        ("72B", "imaged"): A11 / "m72_recomb_imaged.jsonl",
        ("72B", "text"): A11 / "m72_recomb_text.jsonl",
    }
    print("\n== McNemar exact, v2 (memory dispatch) against each baseline ==")
    print("  the 30B has no archived LLM-orchestration arm; that cell is a gap, not a"
          " substitution")
    print("  " + "model/split".ljust(16) + "baseline".ljust(18)
          + "v2 wins".rjust(9) + "base wins".rjust(11) + "p".rjust(12))
    for model in arguments.models:
        for split in SPLITS:
            path = v2_files.get((model, split))
            if path is None or not path.is_file():
                print("  " + f"{model}/{split}".ljust(16) + "v2 arm absent")
                continue
            v2 = {}
            for line in path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    value = record.get("accuracy", record.get("score"))
                    v2[int(record["index"])] = int(round(float(value or 0)))
            for method in METHODS:
                key = (model, method[0], split)
                if key not in scored:
                    continue
                wins, losses, p = mcnemar(scored[key], v2)
                print("  " + f"{model}/{split}".ljust(16) + method[0].ljust(18)
                      + f"{wins}".rjust(9) + f"{losses}".rjust(11) + f"{p:12.3g}")

    if arguments.json:
        arguments.json.write_text(json.dumps({
            "meta": meta,
            "per_index": {f"{m}|{a}|{s}": v for (m, a, s), v in scored.items()},
        }, indent=1) + "\n")
        print(f"\nwrote {arguments.json}")


if __name__ == "__main__":
    main()
