#!/usr/bin/env python3
"""Re-score the MEMENTO-style arms from their raw responses, and check the archive.

`viki_report_matrix.py` treats the MEMENTO arms differently from the four prompt-shaped
baselines: the baselines are re-scored from `response` under the JSON-tolerant reading,
while MEMENTO is read straight out of its archived `score` field. That is an inherited
number, which is the one thing the reporting convention says not to do -- and it matters
here because the archived `score` of the *other* arms is the official scorer's, not the
tolerant one (G-Memory's archive says 91/924 = 9.85%, the tolerant reading is 471/924).

So before any MEMENTO number goes into a cross-model table, this re-scores it the way
everything else is scored and prints both. If they agree, the archived numbers were
already tolerant and can be quoted; if they do not, the table has been mixing two
conventions and the re-scored column is the one to use.

Format compliance is recomputed too, because the MEMENTO records carry no `format_score`
field -- a cell reported as 0% format would otherwise be an artefact of the field being
absent rather than a measurement.

Nothing here calls a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from habitat_llm.evaluation import viki_bench as bench  # noqa: E402
from viki_amendment5 import BENCHMARK_ROOT  # noqa: E402
from viki_amendment9_diag102 import parse_plan  # noqa: E402
from viki_report_matrix import tolerant  # noqa: E402
from our_method.skill_memory_v2 import Simulator  # noqa: E402

A10 = ROOT / "results/viki_memory_experiments/amendment10"
A11 = ROOT / "results/viki_memory_experiments/amendment11"

CELLS = [
    ("72B", "ID", A11 / "memento_id.jsonl", "id"),
    ("30B", "ID", A11 / "memento_id_m30.jsonl", "id"),
    ("7B", "ID", A11 / "memento_id_m7.jsonl", "id"),
    ("72B", "recomb imaged", A11 / "memento_recomb_imaged.jsonl", "imaged"),
    ("72B", "recomb text", A11 / "memento_recomb_text.jsonl", "text"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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

    print(f"{'model':7s} {'split':15s} {'rows':>5s} {'archived':>10s} {'tolerant':>10s} "
          f"{'format':>8s}  agreement")
    out = []
    for model, split_name, path, split in CELLS:
        if not path.is_file():
            print(f"{model:7s} {split_name:15s}  MISSING {path}")
            continue
        archived = rescored = parsed = rows = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            rows += 1
            archived += int(round(float(record.get("score", 0) or 0)))
            text = record.get("response") or record.get("raw") or ""
            parsed += int(parse_plan(text) is not None)
            rescored += tolerant(sim, text, truth_of(split, int(record["index"])))
        same = "same" if archived == rescored else f"DIFFER by {rescored - archived:+d}"
        print(f"{model:7s} {split_name:15s} {rows:5d} "
              f"{archived:5d}/{rows} {rescored:5d}/{rows} {parsed / rows:8.4f}  {same}")
        out.append({"model": model, "split": split_name, "rows": rows,
                    "archived": archived, "tolerant": rescored,
                    "format": parsed / rows, "path": str(path.relative_to(ROOT))})

    if arguments.json:
        arguments.json.write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {arguments.json}")


if __name__ == "__main__":
    main()
