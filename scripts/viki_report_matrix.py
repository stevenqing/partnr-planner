#!/usr/bin/env python3
"""The whole comparison in one place, with paired tests, computed from the raw files.

Every arm is re-scored here from its own saved responses under one reading of the answer
-- the JSON-tolerant one -- so no number in the report is inherited from a summary
written under a different convention. Arms are compared with McNemar's exact test on the
rows they share, because these are paired binary outcomes on identical instances and a
difference of means says nothing about whether the same rows moved.

Nothing here calls a model.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment9_diag102 import parse_plan
from our_method.skill_memory_v2 import SEED, Simulator

OUT = ROOT / "results/viki_memory_experiments/amendment11"
A8B = ROOT / "results/viki_memory_experiments/amendment8b"
A10 = ROOT / "results/viki_memory_experiments/amendment10"


def tolerant(sim, response: str, truth: Dict[str, Any]) -> int:
    parsed = parse_plan(response or "")
    if parsed is None:
        return 0
    if isinstance(parsed, list):
        for step in parsed:
            if isinstance(step, dict) and isinstance(step.get("actions"), dict):
                step["actions"] = {k: v for k, v in step["actions"].items() if v is not None}
    transformed = sim.scorer.transform_actions(parsed)
    if not transformed:
        return 0
    globals_ = sim.scorer.eval_single.__globals__
    original = globals_["random"]
    try:
        globals_["random"] = random.Random(SEED)
        ok = sim.scorer.eval_single(transformed, truth)
    except Exception:
        return 0
    finally:
        globals_["random"] = original
    return int(bool(ok) and len(truth["time_steps"]) / len(transformed) >= 0.99)


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


def load_responses(path: Path, field: str = "response") -> Dict[int, str]:
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            out[int(record["index"])] = record.get(field) or record.get("raw") or ""
    return out


def load_scored(path: Path, field: str) -> Dict[int, int]:
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            out[int(record["index"])] = int(round(float(record.get(field, 0))))
    return out


def main() -> None:
    sim = Simulator(BENCHMARK_ROOT)
    truths: Dict[str, Dict[int, Any]] = {}
    frames = {
        "id": pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet"),
        "imaged": pd.read_parquet(A10 / "recombination.imaged.parquet"),
        "text": pd.read_parquet(A10 / "recombination.text.parquet"),
    }

    def truth_of(split: str, index: int):
        table = truths.setdefault(split, {})
        if index not in table:
            table[index] = bench.get_ground_truth(bench.to_native(frames[split].iloc[index].to_dict()))
        return table[index]

    scores: Dict[Tuple[str, str], Dict[int, int]] = {}

    def add_from_responses(split: str, arm: str, path: Path, field: str = "response"):
        raw = load_responses(path, field)
        if raw:
            scores[(split, arm)] = {i: tolerant(sim, text, truth_of(split, i)) for i, text in raw.items()}

    def add_scored(split: str, arm: str, path: Path, field: str):
        got = load_scored(path, field)
        if got:
            scores[(split, arm)] = got

    # archived prompt-writing arms, re-read under the tolerant convention
    for arm, name in (("zero_shot", "zero_shot"), ("trajectory_rag", "trajectory_rag"),
                      ("skill_memory v1", "skill_memory.fullactions_k8"), ("G-Memory", "gmemory")):
        add_from_responses("id", arm, A8B / f"{name}.jsonl")
        for split, folder in (("imaged", "imaged"), ("text", "text")):
            add_from_responses(split, arm, A10 / folder / f"{name}.jsonl")

    add_scored("id", "MEMENTO-style", OUT / "memento_id.jsonl", "score")
    add_scored("imaged", "MEMENTO-style", OUT / "memento_recomb_imaged.jsonl", "score")
    add_scored("text", "MEMENTO-style", OUT / "memento_recomb_text.jsonl", "score")

    add_scored("id", "v2 72B", OUT / "intent_clean.jsonl", "accuracy")
    add_scored("id", "v2 30B", OUT / "m30_id.jsonl", "accuracy")
    add_scored("id", "v2 7B", OUT / "m7_id.jsonl", "accuracy")
    add_scored("imaged", "v2 72B", OUT / "m72_recomb_imaged.jsonl", "accuracy")
    add_scored("text", "v2 72B", OUT / "m72_recomb_text.jsonl", "accuracy")
    add_scored("imaged", "v2 30B", OUT / "m30_recomb_imaged.jsonl", "accuracy")
    add_scored("text", "v2 30B", OUT / "m30_recomb_text.jsonl", "accuracy")
    add_scored("imaged", "v2 7B", OUT / "m7_recomb_imaged.jsonl", "accuracy")
    add_scored("text", "v2 7B", OUT / "m7_recomb_text.jsonl", "accuracy")

    # delegation rungs, whichever have been run at full scale
    add_scored("id", "v2 as prose", OUT / "inprompt_v2_full.jsonl", "score")
    add_scored("id", "v2 body menu", OUT / "opchoice_full.jsonl", "accuracy")
    add_scored("id", "v2 intent, model crew", OUT / "intent_crew_clean.jsonl", "accuracy")
    add_scored("id", "MEMENTO-style 30B", OUT / "memento_id_m30.jsonl", "score")
    add_scored("id", "MEMENTO-style 7B", OUT / "memento_id_m7.jsonl", "score")

    print("=" * 92)
    print("VIKI-L2 interactive: every arm, one scoring convention (JSON-tolerant, seed 20260829)")
    print("=" * 92)
    for split, label, size in (("id", "in-domain (924 rows)", 924),
                               ("imaged", "recombination, imaged (297)", 297),
                               ("text", "recombination, text (297)", 297)):
        print(f"\n--- {label} ---")
        rows = [(arm, values) for (s, arm), values in scores.items() if s == split]
        for arm, values in sorted(rows, key=lambda kv: -sum(kv[1].values()) / max(1, len(kv[1]))):
            hit, total = sum(values.values()), len(values)
            print(f"  {arm:<26} {hit:>4}/{total:<5} {hit / total * 100:6.2f}%")

    # held-out family, from the fold runs
    print("\n--- held-out family (8 folds over the same 924 rows) ---")
    for tag, arm in (("m72_id_folds", "v2 72B"), ("m30_id_folds", "v2 30B"), ("m7_id_folds", "v2 7B")):
        path = OUT / f"{tag}.csv"
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        fold = table[table["arm"] == "fold"]
        print(f"  {arm:<26} {int(fold['accuracy'].sum()):>4}/{len(fold):<5} "
              f"{fold['accuracy'].mean() * 100:6.2f}%")
    memento_folds = {}
    for path in sorted(OUT.glob("memento_fold_*.jsonl")):
        memento_folds.update(load_scored(path, "score"))
    if memento_folds:
        print(f"  {'MEMENTO-style':<26} {sum(memento_folds.values()):>4}/{len(memento_folds):<5} "
              f"{sum(memento_folds.values()) / len(memento_folds) * 100:6.2f}%")

    print("\n" + "=" * 92)
    print("Paired tests (McNemar exact, on the rows both arms attempted)")
    print("=" * 92)
    pairs = [
        ("id", "v2 72B", "G-Memory"), ("id", "v2 72B", "MEMENTO-style"),
        ("id", "v2 72B", "skill_memory v1"), ("id", "MEMENTO-style", "G-Memory"),
        ("id", "MEMENTO-style", "skill_memory v1"),
        ("id", "v2 72B", "v2 30B"), ("id", "v2 30B", "v2 7B"),
        ("id", "v2 72B", "v2 intent, model crew"), ("id", "v2 72B", "v2 body menu"),
        ("id", "v2 body menu", "G-Memory"), ("id", "v2 as prose", "skill_memory v1"),
        ("imaged", "v2 72B", "G-Memory"), ("text", "v2 72B", "G-Memory"),
        ("imaged", "v2 72B", "MEMENTO-style"),
    ]
    for split, left, right in pairs:
        a, b = scores.get((split, left)), scores.get((split, right))
        if not a or not b:
            continue
        n10, n01, p = mcnemar(a, b)
        shared = len(set(a) & set(b))
        stars = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""
        print(f"  [{split:<6}] {left:<24} vs {right:<24} "
              f"{n10:>4} / {n01:<4} on {shared:>4} shared   p={p:.3g} {stars}")

    for tag in ("intent_clean_ablation", "m30_id_ablation", "m7_id_ablation"):
        path = OUT / f"{tag}.csv"
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        print(f"\n--- layer ablation: {tag} ({len(table)} rows, identical answers) ---")
        for column in ("full", "no order", "no grounding", "no order+grounding"):
            if column in table:
                print(f"  {column:<20} {int(table[column].sum()):>4}/{len(table)} "
                      f"= {table[column].mean() * 100:6.2f}%")


if __name__ == "__main__":
    main()
