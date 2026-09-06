#!/usr/bin/env python3
"""§6/§7/§9 of the agent-library results spec. Read-only; scores what is on disk.

Every cell is (model, split, arm, build seed). Statistics are reported per §6: n, the
per-seed values, mean, SD, SE = SD/sqrt(3) and a 95% t interval on 2 degrees of freedom.
A cell whose SD is the same order as its mean is marked unsuitable for ranking and is not
used to order anything. The only judgement is §7, against G-Memory, on the recombination
columns and the held-out-family column; ID never participates in it.

Anything missing is recorded 未执行 with the path. No substitute data is used.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

A11 = ROOT / "results/viki_memory_experiments/amendment11"
T_CRIT_DF2 = 4.302653                     # two-sided 95%, 3 seeds
SPLIT_LABEL = {"id": "ID", "text": "comp text", "imaged": "comp imaged",
               "heldout_family": "held-out family"}
BASELINE_ARMS = ["zero-shot", "trajectory RAG", "MEMENTO-style", "G-Memory"]


def solved_of(record: Dict[str, Any]) -> int:
    if record.get("reason") == "SOLVED":
        return 1
    try:
        return int(float(record.get("accuracy", 0)) >= 1.0)
    except (TypeError, ValueError):
        return 0


def read_cell(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    per_index = {int(r["index"]): solved_of(r) for r in rows}
    reasons = Counter(r.get("reason") or "UNSET" for r in rows)
    unusable = sum(reasons[k] for k in ("UNPARSEABLE", "NO_USABLE_WORK",
                                        "IMAGE_MARKER_IN_TEXT_SPLIT"))
    reask = sum(1 for r in rows if r.get("reask"))
    return {"path": str(path.relative_to(ROOT)), "n": len(rows),
            "solved": sum(per_index.values()),
            "rate": sum(per_index.values()) / len(rows),
            "per_index": per_index, "reasons": dict(reasons),
            "format_ok": 1 - unusable / len(rows),
            "reask_rows": reask,
            "infeasible_assignment": reasons.get("infeasible_assignment", 0),
            "reask_prompt_tokens": [r["reask_prompt_tokens"] for r in rows
                                    if r.get("reask_prompt_tokens")],
            "reask_completion_tokens": [r["reask_completion_tokens"] for r in rows
                                        if r.get("reask_completion_tokens")]}


def stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n_seeds": 0}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    se = sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    half = T_CRIT_DF2 * se if len(values) == 3 else 0.0
    return {"n_seeds": len(values), "per_seed": [round(v, 4) for v in values],
            "mean": round(mean, 4), "sd": round(sd, 4), "se": round(se, 4),
            "ci95": [round(mean - half, 4), round(mean + half, 4)] if half else None,
            "unsuitable_for_ranking": bool(sd and mean and sd >= 0.5 * abs(mean))}


def mcnemar_pair(a: Dict[int, int], b: Dict[int, int]) -> Dict[str, Any]:
    from viki_p0_report import mcnemar
    shared = sorted(set(a) & set(b))
    fix, regress, p = mcnemar({i: a[i] for i in shared}, {i: b[i] for i in shared})
    return {"n": len(shared), "fix": fix, "regress": regress, "p": p}


def load_ours(seeds: List[int], prefix: str) -> Dict[Tuple[str, str], Dict[int, Any]]:
    out: Dict[Tuple[str, str], Dict[int, Any]] = defaultdict(dict)
    for model in ("72B", "30B", "7B"):
        for split in ("id", "text", "imaged"):
            for seed in seeds:
                cell = read_cell(A11 / f"{prefix}_{model}_{split}_{seed}.jsonl")
                if cell:
                    out[(model, split)][seed] = cell
    return out


def load_baselines(models: List[str]) -> Dict[Tuple[str, str, str], Dict[int, int]]:
    """Baselines scored under the JSON-tolerant convention, per row, from their archives."""
    try:
        import pandas as pd
        from habitat_llm.evaluation import viki_bench as bench
        from our_method.skill_memory_v2.simulator import Simulator
        import viki_p0_report as p0
    except Exception as error:                                       # noqa: BLE001
        return {"__error__": f"{type(error).__name__}: {error}"}     # type: ignore[return-value]

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

    out: Dict[Tuple[str, str, str], Dict[int, int]] = {}
    for model in models:
        for method in p0.METHODS:
            for split in ("id", "imaged", "text"):
                path = p0.cell_path(model, method, split)
                if path is None or not path.is_file():
                    continue
                per_index = {}
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    index = int(record["index"])
                    text = record.get("response") or record.get("raw") or ""
                    per_index[index] = p0.tolerant(sim, text, truth_of(split, index))
                out[(model, split, method[0])] = per_index
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[20260901, 20260902, 20260903])
    parser.add_argument("--out", type=Path,
                        default=ROOT / ("results/agent_library_%s" % date.today().isoformat()))
    parser.add_argument("--no-baselines", action="store_true")
    arguments = parser.parse_args(argv)
    out = arguments.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    arms = {"ours": load_ours(arguments.seeds, "ours"),
            "no-trace control": load_ours(arguments.seeds, "notrace"),
            "w/o Ordering": load_ours(arguments.seeds, "abl_noorder"),
            "w/o Grounding": load_ours(arguments.seeds, "abl_noground")}
    reference = {}
    for split in ("id", "text", "imaged"):
        cell = read_cell(A11 / f"appendix_reference_72B_{split}.jsonl")
        if cell:
            reference[split] = cell

    baselines = {} if arguments.no_baselines else load_baselines(["72B", "30B"])
    baseline_error = baselines.pop("__error__", None) if isinstance(baselines, dict) else None

    # ---- §6 tables
    tables: Dict[str, Any] = {}
    for arm, cells in arms.items():
        rows = []
        for (model, split), by_seed in sorted(cells.items()):
            values = [by_seed[s]["rate"] for s in arguments.seeds if s in by_seed]
            rows.append({"model": model, "split": SPLIT_LABEL[split], "arm": arm,
                         "n": by_seed[arguments.seeds[0]]["n"] if by_seed else None,
                         "seeds_present": sorted(by_seed),
                         **stats(values),
                         "format_ok": round(statistics.mean(
                             [by_seed[s]["format_ok"] for s in by_seed]), 4) if by_seed else None,
                         "format_gate_failed": bool(by_seed) and statistics.mean(
                             [by_seed[s]["format_ok"] for s in by_seed]) < 0.90,
                         "reask_rows": sum(by_seed[s]["reask_rows"] for s in by_seed),
                         "infeasible_assignment": sum(
                             by_seed[s]["infeasible_assignment"] for s in by_seed),
                         "failure_modes": {s: by_seed[s]["reasons"] for s in sorted(by_seed)}})
        tables[arm] = rows

    # ---- §7 judgement, G-Memory only, comp columns; ID never participates
    judgement = {"columns": {}, "note": "held-out family is P2 and was not approved; "
                                        "ID does not participate in the judgement"}
    ours = arms["ours"]
    for split in ("text", "imaged"):
        column = {"per_seed": [], "status": "未执行"}
        gm = baselines.get(("72B", split, "G-Memory")) if baselines else None
        by_seed = ours.get(("72B", split), {})
        if gm and by_seed:
            for seed in arguments.seeds:
                if seed in by_seed:
                    column["per_seed"].append(
                        {"seed": seed, **mcnemar_pair(by_seed[seed]["per_index"], gm)})
            values = [by_seed[s]["rate"] for s in arguments.seeds if s in by_seed]
            gm_rate = sum(gm.values()) / len(gm)
            summary = stats([v - gm_rate for v in values])
            column.update({"status": "judged", "gmemory_rate": round(gm_rate, 4),
                           "ours": stats(values), "difference": summary,
                           "all_seeds_p_lt_05_and_same_direction": bool(
                               column["per_seed"]
                               and all(r["p"] < 0.05 for r in column["per_seed"])
                               and (all(r["fix"] > r["regress"] for r in column["per_seed"])
                                    or all(r["fix"] < r["regress"] for r in column["per_seed"]))),
                           "ci_excludes_zero": bool(summary.get("ci95")
                                                    and summary["ci95"][0] > 0)})
        elif not gm:
            column["missing"] = "G-Memory 72B %s per-row archive" % split
        else:
            column["missing"] = "ours 72B %s cells" % split
        judgement["columns"][split] = column
        column["passed"] = bool(column.get("all_seeds_p_lt_05_and_same_direction")
                                and column.get("ci_excludes_zero"))

    # ---- outcome
    judged = [c for c in judgement["columns"].values() if c["status"] == "judged"]
    if not judged:
        outcome = "未执行：P0 的 comp 两列尚未齐备或缺 G-Memory 逐行归档"
    elif any(c["ours"]["mean"] <= c["gmemory_rate"] for c in judged):
        outcome = "C"
    elif all(c["passed"] and not c["ours"]["unsuitable_for_ranking"] for c in judged):
        outcome = "A"
    else:
        outcome = "B"

    payload = {"outcome": outcome, "seeds": arguments.seeds, "tables": tables,
               "reference_appendix": {k: {"n": v["n"], "rate": round(v["rate"], 4)}
                                      for k, v in reference.items()},
               "judgement": judgement, "baseline_error": baseline_error}
    (out / "results.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    write_report(out, payload)
    print("outcome %s" % outcome)
    print("wrote %s" % out)
    return 0


def write_report(out: Path, payload) -> None:
    import hashlib
    lines = ["# RESULTS_REPORT", "", "结局 %s" % payload["outcome"], ""]
    for arm, rows in payload["tables"].items():
        lines += ["## %s" % arm, ""]
        if not rows:
            lines += ["未执行，缺 该臂的所有格", ""]
            continue
        lines += ["| model | split | n | mean ± SE | 逐 seed | SD | 95% CI | format | "
                  "re-ask | infeasible |", "|---|---|---|---|---|---|---|---|---|---|"]
        for row in rows:
            lines.append("| %s | %s | %s | %s ± %s | %s | %s | %s | %s%s | %s | %s |" % (
                row["model"], row["split"], row["n"],
                row.get("mean"), row.get("se"), row.get("per_seed"), row.get("sd"),
                row.get("ci95"), row.get("format_ok"),
                " **format gate failed**" if row.get("format_gate_failed") else "",
                row.get("reask_rows"), row.get("infeasible_assignment")))
            if row.get("unsuitable_for_ranking"):
                lines.append("| | | | **unsuitable for ranking (SD ~ mean)** | | | | | | |")
        lines.append("")
    lines += ["## §7 判定（只对 G-Memory）", "", "```json",
              json.dumps(payload["judgement"], indent=1, ensure_ascii=False)[:4000], "```", ""]
    lines += ["## 附录：reference 库", "", "```json",
              json.dumps(payload["reference_appendix"], indent=1, ensure_ascii=False), "```", ""]
    if payload.get("baseline_error"):
        lines += ["## 基线", "", "未执行，缺 %s" % payload["baseline_error"], ""]
    lines += ["## 附录：failure_mode 逐格逐 seed", "", "见 `results.json` 的 "
              "`tables[*][*].failure_modes`。", ""]
    path = out / "RESULTS_REPORT.md"
    path.write_text("\n".join(lines))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "RESULTS_REPORT.sha256").write_text(digest + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
