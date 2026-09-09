#!/usr/bin/env python3
"""Where the held-out column is lost: per family, per arm, plus the fold's library.

The OOD cell is a single number over 924 rows built from eight fold runs, and a single
number cannot say whether we lose because the retriever ranks badly, because the operators
are wrong, or because the fold removed the only operator that family ever needed. This
splits the column by family and puts three things beside each other for each one:

  * what ours scored on that family with the WHOLE library (the ID column), so the fold's
    cost is a difference rather than a level;
  * what ours scored on it in the fold, with the reasons the rows failed;
  * what every archived baseline scored on the same rows of the same fold;
  * how many operators that fold's library actually held, and which one it lost.

Read-only, zero model calls: ours is read by `reason == SOLVED` the way the results
document reads it, baselines are re-scored from their archived responses under the
JSON-tolerant convention. Written to disk before anything is printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import viki_p0_report as p0  # noqa: E402
from habitat_llm.evaluation import viki_bench as bench  # noqa: E402

A11 = p0.A11
FOLDS = p0.A8B / "folds"
MEMORIES = ROOT / "outputs/v2_memories"

ARM_STEM = {"zero-shot": "zero_shot", "trajectory RAG": "trajectory_rag",
            "skill memory v1": "skill_memory", "G-Memory": "gmemory",
            "G-Memory (shuffled control)": "gmemory.shuffled"}
FOLD_STEM = {"zero-shot": "zero_shot", "trajectory RAG": "trajectory_rag",
             "skill memory v1": "skill_memory.fullactions_k8", "G-Memory": "gmemory",
             "G-Memory (shuffled control)": "gmemory.shuffled"}


def ours_rows(tag: str) -> Optional[Dict[int, dict]]:
    path = A11 / ("%s.jsonl" % tag)
    if not path.is_file():
        return None
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            out[int(record["index"])] = record
    return out or None


def operator_id(operator: dict) -> str:
    effect = operator.get("effect")
    key = effect.get("key") if isinstance(effect, dict) else effect
    body = operator.get("body") or operator.get("actions") or []
    return "%s :: %s" % (key, " ".join(str(step[0] if isinstance(step, (list, tuple))
                                            else step) for step in body))


def library(path: Path) -> Optional[list]:
    if not path.is_file():
        return None
    return [operator_id(o) for o in json.loads(path.read_text())["layer1"]["operators"]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="72B")
    parser.add_argument("--json", type=Path,
                        default=ROOT / ("results/viki_ood_gap_%s.json" % date.today().isoformat()))
    arguments = parser.parse_args(argv)
    model = arguments.model

    sim = p0.Simulator(p0.BENCHMARK_ROOT)
    frame = pd.read_parquet(p0.BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    truths: Dict[int, object] = {}

    def truth_of(index: int):
        if index not in truths:
            truths[index] = bench.get_ground_truth(bench.to_native(frame.iloc[index].to_dict()))
        return truths[index]

    ours_id = ours_rows("v2_ours_%s_id" % model)
    ours_ho = ours_rows("v2_ours_%s_heldout" % model)
    ours_grp = ours_rows("v2_ours_%s_heldout_sibgrp" % model)
    if not ours_id or not ours_ho:
        print("missing ours cells for %s" % model)
        return 1
    family_of = {i: r.get("task_name") for i, r in ours_id.items()}
    families = sorted({f for f in family_of.values() if f})

    whole = library(MEMORIES / "memory_all.json") or []

    report = {"generated": date.today().isoformat(), "model": model,
              "whole_library": whole, "families": [], "arm_totals": {}}

    sibgrp = json.loads((ROOT / "results/sibling_folds_preregistration.json").read_text())
    SIBGRP = sibgrp["affected_eval_folds"]

    arm_rows: Dict[str, Dict[int, int]] = {}
    grp_rows: Dict[str, Dict[int, int]] = {}
    for arm, stem in FOLD_STEM.items():
        rows: Dict[int, int] = {}
        for family in families:
            path = FOLDS / family / ("%s.jsonl" % stem)
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                index = int(record["index"])
                text = record.get("response") or record.get("raw") or ""
                rows[index] = p0.tolerant(sim, text, truth_of(index))
        arm_rows[arm] = rows
        # The sibling-grouped column: the grouped run for the three folds that have a
        # sibling, the single-family run for the five that do not -- for those two the
        # experiment is the same one, so it is reused rather than rerun.
        grouped: Dict[int, int] = {}
        for family in families:
            stem_here = (ARM_STEM.get(arm, stem) + ".sibgrp") if family in SIBGRP else stem
            path = FOLDS / family / ("%s.jsonl" % stem_here)
            if not path.is_file():
                # An arm that holds no memory cannot be changed by hiding a sibling, so it
                # has no grouped run and reuses its single-family rows -- the same reuse the
                # published column makes. A missing file for a memory arm is a real gap and
                # would show up as a short column, so it is not silently filled.
                path = FOLDS / family / ("%s.jsonl" % stem)
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                index = int(record["index"])
                text = record.get("response") or record.get("raw") or ""
                grouped[index] = p0.tolerant(sim, text, truth_of(index))
        grp_rows[arm] = grouped
    # MEMENTO's folds live beside ours, one file per family, all 924 rows in each.
    memento: Dict[int, int] = {}
    for family in families:
        path = A11 / ("memento_fold_%s.jsonl" % family)
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record["index"])
            if family_of.get(index) != family:
                continue
            text = record.get("response") or record.get("raw") or ""
            memento[index] = p0.tolerant(sim, text, truth_of(index))
    arm_rows["MEMENTO-style"] = memento

    for family in families:
        indices = [i for i, f in family_of.items() if f == family]
        fold = library(MEMORIES / ("memory_heldout_%s.json" % family))
        grp = library(MEMORIES / ("memory_heldoutgrp_%s.json" % family))
        entry = {
            "family": family,
            "n": len(indices),
            "ours_id": sum(int(ours_id[i].get("reason") == "SOLVED") for i in indices),
            "ours_heldout": sum(int(ours_ho[i].get("reason") == "SOLVED")
                                for i in indices if i in ours_ho),
            "ours_heldout_reasons": dict(Counter(ours_ho[i].get("reason")
                                                 for i in indices if i in ours_ho).most_common()),
            "ours_id_reasons": dict(Counter(ours_id[i].get("reason")
                                            for i in indices).most_common()),
            "fold_library_size": len(fold) if fold is not None else None,
            "fold_library_lost": sorted(set(whole) - set(fold)) if fold is not None else None,
            "grp_library_size": len(grp) if grp is not None else None,
            "grp_library_lost": sorted(set(whole) - set(grp)) if grp is not None else None,
            "baselines": {arm: sum(rows.get(i, 0) for i in indices)
                          for arm, rows in arm_rows.items()},
            "baselines_sibgrp": {arm: sum(rows.get(i, 0) for i in indices)
                                 for arm, rows in grp_rows.items()},
        }
        if ours_grp:
            entry["ours_heldout_sibgrp"] = sum(int(ours_grp[i].get("reason") == "SOLVED")
                                               for i in indices if i in ours_grp)
        report["families"].append(entry)

    report["arm_totals"] = {arm: sum(rows.values()) for arm, rows in arm_rows.items()}
    report["arm_totals_sibgrp"] = {arm: sum(rows.values()) for arm, rows in grp_rows.items()}
    if ours_grp:
        report["arm_totals_sibgrp"]["ours (agent library)"] = sum(
            int(r.get("reason") == "SOLVED") for r in ours_grp.values())
    report["arm_totals"]["ours (agent library)"] = sum(
        int(r.get("reason") == "SOLVED") for r in ours_ho.values())
    report["arm_totals"]["ours (ID, whole library)"] = sum(
        int(r.get("reason") == "SOLVED") for r in ours_id.values())

    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(json.dumps(report, indent=1))

    print("model %s -- held-out column, by family (n=924)\n" % model)
    header = ("family", "n", "ours_ID", "ours_HO", "lost_op", "G-Mem", "MEM", "v1", "trajRAG", "zero")
    print("%-46s %4s %7s %7s %7s %6s %5s %5s %7s %5s" % header)
    for entry in report["families"]:
        base = entry["baselines"]
        print("%-46s %4d %7d %7d %7s %6s %5s %5s %7s %5s" % (
            entry["family"][:46], entry["n"], entry["ours_id"], entry["ours_heldout"],
            len(entry["fold_library_lost"] or []),
            base.get("G-Memory", "-"), base.get("MEMENTO-style", "-"),
            base.get("skill memory v1", "-"), base.get("trajectory RAG", "-"),
            base.get("zero-shot", "-")))
    print()
    for entry in report["families"]:
        if entry["fold_library_lost"]:
            print("%s lost: %s" % (entry["family"], entry["fold_library_lost"]))
    print()
    print("sibling-grouped column, by family (ours / G-Memory):")
    for entry in report["families"]:
        print("%-46s %4d  ours %3s  G-Mem %3s" % (
            entry["family"][:46], entry["n"], entry.get("ours_heldout_sibgrp", "-"),
            entry["baselines_sibgrp"].get("G-Memory", "-")))
    print()
    print(json.dumps(report["arm_totals"], indent=1))
    print(json.dumps(report["arm_totals_sibgrp"], indent=1))
    print("\nwrote %s" % arguments.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
