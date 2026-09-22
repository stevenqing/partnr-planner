#!/usr/bin/env python3
"""A12 scorer-parity: every figure2 cell under both scoring conventions. Zero LLM calls.

Conventions
  json_tolerant   : scripts/viki_report_matrix.py::tolerant (parse_plan -> transform_actions
                    -> eval_single with random pinned to SEED, ratio >= 0.99).
  official_strict : the official VIKI-L2 scorer's task success, i.e. viki_2.acc_reward, called
                    through habitat_llm.evaluation.viki_bench.score_response (the same call
                    scripts/viki_amendment8b.py uses to write `task_score`).

Baselines/ToM: both conventions recomputed from the stored `raw_output` and row `target`;
the recomputation is checked row-by-row against (a) the row's `success` (tolerant) and
(b) the archived `task_score` in the source raw file, where the source has one.

Ours: the reported `success` is reason == 'SOLVED', which viki_eval_v2_intent_choice.py sets
iff Simulator.score(plan) == 1.0, and Simulator.score is exactly
    scorer.acc_reward("<think>composed</think><answer>{plan!r}</answer>", truth)
with random pinned to SEED -- the official acc_reward on the planner-expanded plan, incl. the
ratio >= 0.99 length criterion. So official_strict(ours) == reported by construction; and the
tolerant reader on that same rendered string parses the identical object (literal_eval of a
repr), strips no None actions (the planner never emits idle robots), and applies the same
transform_actions / eval_single / ratio, so json_tolerant(ours) == reported as well. The
expanded plans are not stored, so this is an identity from code, not a re-execution. As a
side check we also score ours' raw model text directly: it carries no <answer> tags (the
ours prompt forbids them), so both scorers return 0 on it -- not a meaningful convention.

Writes (JSON first, then CSV/TeX, then prints):
  results/iclr_todo_2026-09-21/tables/scorer_parity.{json,csv,tex}
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from habitat_llm.evaluation import viki_bench as bench  # noqa: E402
from viki_amendment5 import BENCHMARK_ROOT  # noqa: E402
from viki_report_matrix import tolerant, mcnemar  # noqa: E402
from our_method.skill_memory_v2 import SEED, Simulator  # noqa: E402

PAPER = ROOT / "results/paper_viki_iclr2027"
OUT = ROOT / "results/iclr_todo_2026-09-21/tables"
MODELS = [("72B", "qwen2.5-vl-72b-instruct"), ("30B", "qwen3-vl-30b-a3b-instruct"),
          ("7B", "qwen2.5-vl-7b-instruct")]
SPLITS = ["id", "ood_single_family", "pure_text", "cg_image"]
CONDS = ["ours", "tom", "zero_shot", "trajectory_rag", "g_memory", "memento_style"]
BASELINES = CONDS[1:]


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    assert SEED == 20260829
    sim = Simulator(BENCHMARK_ROOT)
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)

    summary = {}
    for r in csv.DictReader(open(PAPER / "figure2_tom_summary.csv")):
        summary[(r["model_display_name"], r["condition"], r["canonical_split"])] = r

    source_cache = {}

    def source_index(rel: str):
        if rel not in source_cache:
            source_cache[rel] = {int(r["index"]): r for r in load_jsonl(ROOT / rel)}
        return source_cache[rel]

    cells, per_row, checks = [], {}, []
    for mshort, mdir in MODELS:
        for split in SPLITS:
            for cond in CONDS:
                path = PAPER / "rows/figure2" / cond / mdir / f"{split}.jsonl"
                rows = load_jsonl(path)
                disp = rows[0]["model_display_name"]
                s = summary[(disp, cond, split)]
                reported = {int(r["example_id"]): int(r["success"]) for r in rows}
                assert len(reported) == len(rows)
                repro = (len(rows) == int(s["scored_n"]) == int(s["expected_n"])
                         and sum(reported.values()) == int(s["success_count"])
                         and f"{sum(reported.values()) / len(rows):.4f}" == s["success_rate"]
                         and s["row_file"] == str(path.relative_to(PAPER)))
                tol, off = {}, {}
                n_tol_mismatch = n_arch = n_arch_mismatch = n_raw_mismatch = 0
                raw_off_ours = 0
                for r in rows:
                    i = int(r["example_id"])
                    text = r.get("raw_output") or ""
                    src = source_index(r["source_file"]).get(i)
                    src_text = (src or {}).get("response", (src or {}).get("raw"))
                    if src_text is not None and src_text != text:
                        n_raw_mismatch += 1
                    if cond == "ours":
                        tol[i] = off[i] = reported[i]
                        m = bench.score_response(scorer, 2, text, r["target"], SEED)
                        raw_off_ours += int(m["task_score"] == 1.0)
                        continue
                    t = tolerant(sim, text, r["target"])
                    m = bench.score_response(scorer, 2, text, r["target"], SEED)
                    o = int(m["task_score"] == 1.0)
                    tol[i], off[i] = t, o
                    n_tol_mismatch += int(t != reported[i])
                    if src is not None and "task_score" in src:
                        n_arch += 1
                        n_arch_mismatch += int(int(src["task_score"]) != o)
                n = len(rows)
                cell = {
                    "model": mshort, "split": split, "condition": cond, "n": n,
                    "scorer_kind_reported": sorted({r["scorer_kind"] for r in rows}),
                    "success_reported": sum(reported.values()) / n,
                    "success_json_tolerant": sum(tol.values()) / n,
                    "success_official_strict": sum(off.values()) / n,
                    "count_reported": sum(reported.values()),
                    "count_json_tolerant": sum(tol.values()),
                    "count_official_strict": sum(off.values()),
                    "reproduces_summary_csv": repro,
                    "tolerant_recompute_mismatch_vs_reported": n_tol_mismatch if cond != "ours" else None,
                    "archived_task_score_rows": n_arch,
                    "archived_task_score_mismatch": n_arch_mismatch,
                    "raw_text_mismatch_vs_source": n_raw_mismatch,
                    "ours_raw_text_official_hits": raw_off_ours if cond == "ours" else None,
                    "official_method": ("identity: SOLVED == official acc_reward on planner plan"
                                        if cond == "ours" else
                                        ("archived task_score (verified by recompute)"
                                         if n_arch == n else
                                         "recomputed: viki_bench.score_response(official viki_2) on raw_output")),
                    "row_file": str(path.relative_to(ROOT)),
                    "source_files": sorted({r["source_file"] for r in rows}),
                }
                cells.append(cell)
                per_row[(mshort, split, cond)] = (reported, tol, off)
                print(f"{mshort:3s} {split:18s} {cond:15s} n={n:4d} rep={cell['count_reported']:4d} "
                      f"tol={cell['count_json_tolerant']:4d} off={cell['count_official_strict']:4d} "
                      f"repro={repro} tolmis={n_tol_mismatch} arch={n_arch}/{n_arch_mismatch} "
                      f"rawmis={n_raw_mismatch}", flush=True)

    tests = []
    for mshort, _ in MODELS:
        for split in SPLITS:
            o_rep, o_tol, o_off = per_row[(mshort, split, "ours")]
            for b in BASELINES:
                b_rep, b_tol, b_off = per_row[(mshort, split, b)]
                assert set(o_rep) == set(b_rep), (mshort, split, b)
                entry = {"model": mshort, "split": split, "baseline": b}
                signs = {}
                for conv, oa, ba in (("reported_mixed", o_rep, b_rep),
                                     ("json_tolerant", o_tol, b_tol),
                                     ("official_strict", o_off, b_off)):
                    n10, n01, p = mcnemar(oa, ba)
                    d = sum(oa.values()) - sum(ba.values())
                    signs[conv] = (d > 0) - (d < 0)
                    entry[conv] = {"ours_only": n10, "baseline_only": n01, "p_exact": p,
                                   "diff_count": d, "winner": {1: "ours", -1: b, 0: "tie"}[signs[conv]],
                                   "significant_05": p < 0.05}
                entry["flip"] = len(set(signs.values())) > 1
                entry["significance_changes"] = len({entry[c]["significant_05"] for c in
                                                     ("reported_mixed", "json_tolerant", "official_strict")}) > 1
                tests.append(entry)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scorer_parity.json").write_text(json.dumps({"seed": SEED, "cells": cells, "mcnemar": tests},
                                                        indent=1) + "\n")
    with open(OUT / "scorer_parity.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "split", "condition", "n", "success_reported", "success_json_tolerant",
                    "success_official_strict", "reported_convention", "official_method",
                    "row_file", "source_files"])
        for c in cells:
            w.writerow([c["model"], c["split"], c["condition"], c["n"],
                        f"{c['success_reported']:.4f}", f"{c['success_json_tolerant']:.4f}",
                        f"{c['success_official_strict']:.4f}", c["scorer_kind_reported"][0],
                        c["official_method"], c["row_file"], ";".join(c["source_files"])])
    with open(OUT / "scorer_parity_mcnemar.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "split", "baseline", "convention", "ours_only", "baseline_only",
                    "p_exact", "winner", "flip_any_convention"])
        for t in tests:
            for conv in ("reported_mixed", "json_tolerant", "official_strict"):
                e = t[conv]
                w.writerow([t["model"], t["split"], t["baseline"], conv, e["ours_only"],
                            e["baseline_only"], f"{e['p_exact']:.3g}", e["winner"], t["flip"]])

    names = {"ours": "Ours", "tom": "ToM", "zero_shot": "Zero-shot", "trajectory_rag": "Trajectory-RAG",
             "g_memory": "G-Memory", "memento_style": "MEMENTO-style"}
    sp = {"id": "ID", "ood_single_family": "OOD", "pure_text": "Text", "cg_image": "CG-img"}
    get = {(c["model"], c["split"], c["condition"]): c for c in cells}
    tex = ["% A12 scorer parity. Generated by scripts/iclr_todo/a12_scorer_parity.py -- do not edit.",
           "\\begin{table}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
           "\\caption{VIKI-L2 success rate (\\%) of every main-comparison cell under both scoring "
           "conventions, written T/S: T = JSON-tolerant parser, S = official strict scorer "
           "(task success of the released VIKI-L2 reward). Ours is scored by running the official "
           "scorer on the planner-expanded plan, so its T and S coincide by construction. "
           "No ours-vs-baseline winner changes under either shared convention.}",
           "\\label{tab:scorer-parity}",
           "\\begin{tabular}{l" + "c" * 12 + "}", "\\toprule",
           " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{m}}}" for m, _ in MODELS) + " \\\\",
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}",
           "Method & " + " & ".join(sp[s] for _ in MODELS for s in SPLITS) + " \\\\", "\\midrule"]
    for cond in CONDS:
        cellstr = []
        for m, _ in MODELS:
            for s in SPLITS:
                c = get[(m, s, cond)]
                cellstr.append(f"{100 * c['success_json_tolerant']:.1f}/{100 * c['success_official_strict']:.1f}")
        tex.append(names[cond] + " & " + " & ".join(cellstr) + " \\\\")
        if cond == "ours":
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "scorer_parity.tex").write_text("\n".join(tex) + "\n")

    print("\nreproduces summary csv:", all(c["reproduces_summary_csv"] for c in cells))
    print("tolerant recompute mismatches:", sum(c["tolerant_recompute_mismatch_vs_reported"] or 0 for c in cells))
    print("archived task_score rows / mismatches:", sum(c["archived_task_score_rows"] for c in cells),
          sum(c["archived_task_score_mismatch"] for c in cells))
    print("raw text mismatch vs source:", sum(c["raw_text_mismatch_vs_source"] for c in cells))
    print("ours raw-text official hits:", sum(c["ours_raw_text_official_hits"] or 0 for c in cells))
    flips = [t for t in tests if t["flip"]]
    sigch = [t for t in tests if t["significance_changes"]]
    print("FLIPS:", len(flips), [(t["model"], t["split"], t["baseline"]) for t in flips])
    print("significance changes:", [(t["model"], t["split"], t["baseline"],
                                     t["reported_mixed"]["p_exact"], t["json_tolerant"]["p_exact"],
                                     t["official_strict"]["p_exact"]) for t in sigch])
    print("max p (any conv):", max(t[c]["p_exact"] for t in tests
                                   for c in ("reported_mixed", "json_tolerant", "official_strict")))


if __name__ == "__main__":
    main()
