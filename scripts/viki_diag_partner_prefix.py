#!/usr/bin/env python3
"""How much was the partner prefix worth to the archived plan-writing arms?

Those arms were handed a line of the reference plan -- what a partner robot does at step
one -- and any arm compared against them without it is being compared from behind. The
responses are on disk, so the hint's value can be read off them rather than guessed at:
how often the model puts exactly that action in exactly that step, and whether the rows
where it did are the rows it got right.

This does not remove the hint from their arm, which would need the retrieval code that
is no longer in the tree. It bounds what removing it could cost them.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from viki_amendment9_diag102 import parse_plan

BASE = ROOT / "results/viki_memory_experiments/amendment8b"
MANIFEST = BASE / "interactive_manifest.jsonl"
STEP = re.compile(r"step\s+1\s*->\s*(R\d+):\s*(\[[^\]]*\])")


def main() -> None:
    manifest = {int(json.loads(l)["index"]): json.loads(l)
                for l in MANIFEST.read_text().splitlines() if l.strip()}
    for arm in ("gmemory", "skill_memory.fullactions_k8", "zero_shot"):
        path = BASE / f"{arm}.jsonl"
        if not path.is_file():
            continue
        told, copied, counts = 0, 0, Counter()
        scored = {"copied": [0, 0], "not copied": [0, 0]}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record["index"])
            entry = manifest.get(index)
            if not entry:
                continue
            hints = STEP.findall(entry.get("partner_prefix", ""))
            if not hints:
                continue
            told += 1
            plan = parse_plan(record.get("response") or "")
            first = {}
            if isinstance(plan, list) and plan:
                step = plan[0]
                if isinstance(step, dict) and isinstance(step.get("actions"), dict):
                    first = {k: v for k, v in step["actions"].items() if v is not None}
            match = False
            for robot, text in hints:
                try:
                    action = ast.literal_eval(text)
                except Exception:
                    continue
                if [str(x) for x in (first.get(robot) or [])] == [str(x) for x in action]:
                    match = True
            copied += int(match)
            bucket = "copied" if match else "not copied"
            # task_score is the official one; the tolerant reading is what this line
            # reports elsewhere, but the split is what matters here, not the level.
            scored[bucket][0] += int(record.get("task_score", 0) == 1)
            scored[bucket][1] += 1
            counts[bucket] += 1
        if not told:
            continue
        print(f"\n{arm}")
        print(f"  rows given a partner hint            {told}")
        print(f"  reproduced the hinted action at step 1  {copied}  ({copied / told * 100:.1f}%)")
        for bucket, (won, total) in scored.items():
            if total:
                print(f"  official accuracy when {bucket:<11} {won}/{total} = {won / total * 100:.1f}%")


if __name__ == "__main__":
    main()
