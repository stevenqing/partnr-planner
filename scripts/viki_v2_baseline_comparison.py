#!/usr/bin/env python3
"""ours (v2 agent library) against every archived baseline. Read-only, zero calls.

Baselines are re-scored from their own archived responses under the JSON-tolerant
convention, the same instrument that produced the numbers in VIKI-L2-PAPER-RESULTS.md, so
what is compared is per-row outcomes rather than two summary rates. Every comparison is an
exact McNemar over the rows both arms attempted.

Written to disk before anything is printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

A11 = ROOT / "results/viki_memory_experiments/amendment11"

OURS = {
    ("72B", "id"): "v2_ours_72B_id", ("72B", "text"): "v2_oursall_72B_text",
    ("72B", "imaged"): "v2_oursall_72B_imaged", ("72B", "heldout"): "v2_ours_72B_heldout",
    ("30B", "id"): "v2_ours_30B_id", ("30B", "text"): "v2_oursall_30B_text",
    ("30B", "imaged"): "v2_oursall_30B_imaged", ("30B", "heldout"): "v2_ours_30B_heldout",
    ("7B", "id"): "v2_ours_7B_id", ("7B", "text"): "v2_oursall_7B_text",
    ("7B", "imaged"): "v2_oursall_7B_imaged", ("7B", "heldout"): "v2_ours_7B_heldout",
}
# The half-memory cells, reported beside the whole-memory ones rather than in place of them.
CURVE = {"comp(cut)": "v2_ourscut_%s_%s", "comp(cut+delivery)": "v2_ours_%s_%s",
         "comp(all)": "v2_oursall_%s_%s"}


def ours_rows(tag: str) -> Optional[Dict[int, int]]:
    path = A11 / ("%s.jsonl" % tag)
    if not path.is_file():
        return None
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            out[int(record["index"])] = int(record.get("reason") == "SOLVED")
    return out or None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / ("results/agent_library_%s" % date.today().isoformat()))
    parser.add_argument("--models", nargs="+", default=["72B", "30B"])
    arguments = parser.parse_args(argv)
    out = arguments.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from habitat_llm.evaluation import viki_bench as bench
    from our_method.skill_memory_v2.simulator import Simulator
    import viki_p0_report as p0

    sim = Simulator(p0.BENCHMARK_ROOT)
    frames = {
        "id": pd.read_parquet(p0.BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet"),
        "imaged": pd.read_parquet(p0.A10 / "recombination.imaged.parquet"),
        "text": pd.read_parquet(p0.A10 / "recombination.text.parquet"),
    }
    cache: Dict[str, Dict[int, Any]] = {}

    def truth_of(split, index):
        table = cache.setdefault(split, {})
        if index not in table:
            table[index] = bench.get_ground_truth(
                bench.to_native(frames[split].iloc[index].to_dict()))
        return table[index]

    # MEMENTO-style is not in p0.METHODS (it is produced by a separate script), so its
    # archives are named here. Omitting the second-strongest baseline from a baseline
    # comparison would be a hole a reader spots immediately.
    MEMENTO = {("72B", "id"): "memento_id.jsonl",
               ("72B", "text"): "memento_recomb_text.jsonl",
               ("72B", "imaged"): "memento_recomb_imaged.jsonl",
               ("30B", "id"): "memento_id_m30.jsonl",
               ("7B", "id"): "memento_id_m7.jsonl"}

    def baseline_rows(model, method, split):
        if method[0] == "MEMENTO-style":
            name = MEMENTO.get((model, split))
            path = (A11 / name) if name else None
        else:
            path = p0.cell_path(model, method, split)
        if path is None or not path.is_file():
            return None, str(path)
        rows = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record["index"])
            text = record.get("response") or record.get("raw") or ""
            rows[index] = p0.tolerant(sim, text, truth_of(split, index))
        return rows, str(path)

    report: Dict[str, Any] = {"generated": date.today().isoformat(), "cells": [],
                              "curve": [], "missing": []}

    for model in arguments.models:
        for split in ("id", "text", "imaged"):
            mine = ours_rows(OURS.get((model, split), ""))
            if mine is None:
                report["missing"].append("ours %s/%s" % (model, split))
                continue
            row = {"model": model, "split": split, "arm": "ours (agent library)",
                   "n": len(mine), "solved": sum(mine.values()),
                   "rate": round(sum(mine.values()) / len(mine), 4)}
            report["cells"].append(row)
            for method in list(p0.METHODS) + [("MEMENTO-style", None, None)]:
                base, where = baseline_rows(model, method, split)
                if base is None:
                    report["missing"].append("%s %s/%s (%s)" % (method[0], model, split, where))
                    continue
                shared = sorted(set(mine) & set(base))
                fix, regress, p = p0.mcnemar({i: mine[i] for i in shared},
                                             {i: base[i] for i in shared})
                report["cells"].append({
                    "model": model, "split": split, "arm": method[0],
                    "n": len(base), "solved": sum(base.values()),
                    "rate": round(sum(base.values()) / len(base), 4),
                    "vs_ours": {"n_paired": len(shared), "ours_wins": fix,
                                "baseline_wins": regress, "p": p,
                                "direction": "ours" if fix > regress else "baseline"}})

    # the half-to-whole curve, ours only
    for model in ("72B", "30B", "7B"):
        for split in ("text", "imaged"):
            entry = {"model": model, "split": split}
            for label, pattern in CURVE.items():
                rows = ours_rows(pattern % (model, split))
                entry[label] = (round(sum(rows.values()) / len(rows), 4) if rows else None)
            report["curve"].append(entry)

    (out / "baseline_comparison.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False))
    write_md(out, report)

    for row in report["cells"]:
        v = row.get("vs_ours")
        print("%-4s %-7s %-22s %3d/%-4d %.4f%s"
              % (row["model"], row["split"], row["arm"], row["solved"], row["n"], row["rate"],
                 ("   ours %d / base %d  p=%.2e" % (v["ours_wins"], v["baseline_wins"], v["p"]))
                 if v else ""))
    print()
    for entry in report["curve"]:
        print("curve %-4s %-7s cut=%s  cut+del=%s  all=%s"
              % (entry["model"], entry["split"], entry.get("comp(cut)"),
                 entry.get("comp(cut+delivery)"), entry.get("comp(all)")))
    if report["missing"]:
        print("\n未执行，缺:")
        for item in report["missing"]:
            print("   ", item)
    print("\nwrote %s" % out)
    return 0


def write_md(out: Path, report) -> None:
    lines = ["# ours vs baselines (v2 agent library)", "",
             "口径 JSON-tolerant，基线由其归档回复逐行重打分；配对为 McNemar exact。", ""]
    for model in ("72B", "30B", "7B"):
        rows = [r for r in report["cells"] if r["model"] == model]
        if not rows:
            continue
        lines += ["## %s" % model, "",
                  "| split | arm | n | solved | rate | ours wins / baseline wins | p |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows:
            v = r.get("vs_ours")
            lines.append("| %s | %s | %d | %d | %.4f | %s | %s |" % (
                r["split"], r["arm"], r["n"], r["solved"], r["rate"],
                ("%d / %d" % (v["ours_wins"], v["baseline_wins"])) if v else "—",
                ("%.2e" % v["p"]) if v else "—"))
        lines.append("")
    lines += ["## comp 半份→全份曲线（ours）", "",
              "| model | split | comp(cut) | comp(cut+delivery) | comp(all) |", "|---|---|---|---|---|"]
    for e in report["curve"]:
        lines.append("| %s | %s | %s | %s | %s |" % (
            e["model"], e["split"], e.get("comp(cut)"), e.get("comp(cut+delivery)"),
            e.get("comp(all)")))
    if report["missing"]:
        lines += ["", "## 未执行", ""] + ["- 缺 %s" % m for m in report["missing"]]
    (out / "BASELINE_COMPARISON.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
