#!/usr/bin/env python3
"""Paired comparison across every Amendment 9 rung and the Amendment 8b arms.

Reports the official score and the JSON-tolerant score side by side, because the
official parser scores an arm zero for emitting JSON `null` and that penalty is not
a planning difference. Neither number replaces the other and the scorer is not
modified.
"""

from __future__ import annotations

import ast
import itertools
import json
import re
import random
import sys
from math import comb
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment8b import (
    BENCHMARK_ROOT,
    OUTPUT_DIR,
    SEED,
    SOURCE_PARQUET,
    load_manifest,
    native,
)

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def load(path: Path) -> Dict[int, Dict[str, Any]]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows[record["index"]] = record
    return rows


def mcnemar(a: Dict[int, Any], b: Dict[int, Any], key: str) -> Dict[str, Any]:
    shared = sorted(set(a) & set(b))
    n01 = sum(1 for i in shared if not a[i][key] and b[i][key])
    n10 = sum(1 for i in shared if a[i][key] and not b[i][key])
    total = n01 + n10
    if total == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    smaller = min(n01, n10)
    p = min(1.0, 2 * sum(comb(total, j) for j in range(smaller + 1)) / 2 ** total)
    return {"a_only": n10, "b_only": n01, "p": p}


def json_tolerant(scorer, response: str, ground_truth: Any) -> int:
    """acc_reward widened to accept JSON `null`.

    The official parser is ast.literal_eval, which reads the single-quoted Python
    literals the prompt asks for but rejects `null`. json.loads is the mirror image:
    it accepts `null` and rejects single quotes. Neither alone is a superset, so a
    row counts if either parser yields a passing plan. Scoring stays the official
    eval_single plus the official step-count ratio.
    """
    match = ANSWER.search(response)
    if not match:
        return 0
    body = match.group(1).strip()
    if re.search(r"\{\{.*\}\}", body, re.DOTALL):
        body = body.replace("{{", "{").replace("}}", "}")
    parsed = None
    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(body)
            break
        except Exception:
            continue
    if parsed is None:
        return 0
    if isinstance(parsed, list):
        for step in parsed:
            if isinstance(step, dict) and isinstance(step.get("actions"), dict):
                step["actions"] = {
                    k: v for k, v in step["actions"].items() if v is not None
                }
    transformed = scorer.transform_actions(parsed)
    if not transformed:
        return 0
    globals_ = scorer.eval_single.__globals__
    original = globals_["random"]
    try:
        globals_["random"] = random.Random(SEED)
        ok = scorer.eval_single(transformed, ground_truth)
    except Exception:
        return 0
    finally:
        globals_["random"] = original
    if not ok:
        return 0
    return int(len(ground_truth["time_steps"]) / len(transformed) >= 0.99)


def main() -> None:
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    dataset = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()
    truth = {
        i: native(dataset.iloc[i].to_dict())["reward_model"]["ground_truth"]
        for i in sorted(manifest)
    }

    arms: Dict[str, Dict[int, Any]] = {}
    for path in sorted(OUTPUT_DIR.glob("*.jsonl")):
        if path.name.endswith(".aligned.jsonl") or "manifest" in path.name:
            continue
        rows = load(path)
        if len(rows) == len(manifest):
            arms[path.name[: -len(".jsonl")]] = rows

    print(f"{'arm':28s} {'official':>18s} {'json-tolerant':>16s} {'format':>9s}")
    for name, rows in sorted(arms.items()):
        official = sum(r["task_score"] for r in rows.values())
        tolerant = sum(
            json_tolerant(scorer, r["response"], truth[i]) for i, r in rows.items()
        )
        fmt = sum(r["format_score"] for r in rows.values())
        n = len(rows)
        for record, value in ((rows, tolerant),):
            pass
        print(
            f"{name:28s} {official:4d}/{n} ({100*official/n:5.2f}%) "
            f"{tolerant:4d} ({100*tolerant/n:5.2f}%) {100*fmt/n:8.1f}%"
        )

    print()
    print("McNemar exact, official task_score")
    for a, b in itertools.combinations(sorted(arms), 2):
        result = mcnemar(arms[a], arms[b], "task_score")
        if result["a_only"] == result["b_only"] == 0:
            continue
        star = (
            "***" if result["p"] < 0.001 else "**" if result["p"] < 0.01
            else "*" if result["p"] < 0.05 else ""
        )
        print(
            f"  {a:26s} vs {b:26s} {result['a_only']:4d}/{result['b_only']:<4d} "
            f"p={result['p']:.3g} {star}"
        )


if __name__ == "__main__":
    main()
