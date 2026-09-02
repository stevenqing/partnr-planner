#!/usr/bin/env python3
"""What work the model listed, beside the work the task actually required.

The three decisions the operator-choice arm asks for are not equally hard, and handing
two of them back to the memory changed nothing, so what is left is the list itself. This
prints it against the ground truth for the families that fail, which is the only way to
see whether the model is missing requirements, inventing them, or naming them wrongly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment11_goalparse import extract_json

OUT = ROOT / "results/viki_memory_experiments/amendment11"
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "opchoice_smoke"
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def brief(constraints):
    out = []
    stack = list(constraints or [])
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            status = {k: v for k, v in (node.get("status") or {}).items() if v is not None}
            for key, value in status.items():
                out.append(f"{node['name']}{'@' if key == 'pos.name' else '!'}{value}")
    return sorted(set(out))


def main() -> None:
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    records = [json.loads(line) for line in (OUT / f"{SOURCE}.jsonl").read_text().splitlines() if line.strip()]
    shown = {}
    for record in records:
        if record.get("reason") == "SOLVED":
            continue
        family = record["task_name"]
        if shown.get(family, 0) >= PER:
            continue
        shown[family] = shown.get(family, 0) + 1
        truth = bench.get_ground_truth(bench.to_native(frame.iloc[record["index"]].to_dict()))
        parsed = extract_json(record.get("raw") or "") or {}
        print("=" * 78)
        print(f"[{family}] index={record['index']}  {record.get('reason')}  "
              f"budget {len(truth['time_steps'])}  planned {record.get('plan_len', 0)}")
        print(f"  desc     : {(truth.get('description') or '')[:130]}")
        print(f"  robots   : {[r for r, t in truth['robots'].items() if t]}")
        print(f"  GT goals : {brief(truth['goal_constraints'])}")
        print(f"  GT order : {brief(truth['temporal_constraints'])}")
        print(f"  work     : {json.dumps(parsed.get('work'))[:560]}")


if __name__ == "__main__":
    main()
