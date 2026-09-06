#!/usr/bin/env python3
"""§3 of the agent-library results spec: find before running. Read-only, no LLM call.

Enumerates every agent-built library, every end-to-end result produced from one, the
baseline archives, and checks each against the four constraints this spec imposes --
build seed, induction pool, dispatch arm, and fallback rule -- then emits INVENTORY.md
and coverage_matrix.csv. A cell that fails any constraint is reported as missing, not
reused. Artefacts land before anything is printed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

A11 = ROOT / "results/viki_memory_experiments/amendment11"
AUDIT = ROOT / "audit/comp_leakage_2026-09-05"

REQUIRED_BUILD_SEEDS = [20260901, 20260902, 20260903]
OBSERVED_SWEEP_SEEDS = "20260829 + sample  (scripts/viki_agentic_rung_sweep.py:49)"
SIM_SEED = 20260829

MODELS = ["72B", "30B", "7B"]
SPLITS = ["id", "heldout_family", "comp_imaged", "comp_text"]
ARMS = ["ours", "zero-shot", "trajectory RAG", "MEMENTO-style", "G-Memory",
        "w/o Ordering", "w/o Grounding", "w/o executable skills", "no-trace control",
        "reference"]

# §5. The rule this spec fixes, against the rule the shipped evaluator implements.
SPEC_FALLBACK = ("re-ask the model once with the infeasibility reason; if still "
                 "infeasible the row is unsolved and recorded infeasible_assignment")
SHIPPED_FALLBACK = ("viki_eval_v2_intent_choice.py:222-232 -- drop temporal constraints "
                    "and retry, then fall back to the memory's own free search (recast), "
                    "scoring the row under memory dispatch")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:                                    # noqa: BLE001
                    return []
        # Only per-row RESULT files carry a score; response-only archives do not.
        return rows if rows and ("accuracy" in rows[0] or "reason" in rows[0]) else []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def rate_of_csv(path: Path) -> Dict[str, Any]:
    try:
        rows = load_rows(path)
    except Exception as error:                                       # noqa: BLE001
        return {"rows": None, "error": str(error)}
    if not rows:
        return {"rows": 0}
    solved = sum(1 for r in rows
                 if (r.get("reason") == "SOLVED"
                     or _as_float(r.get("accuracy")) is not None and _as_float(r.get("accuracy")) >= 1.0))
    return {"rows": len(rows), "solved": solved,
            "rate": round(solved / len(rows), 4),
            "columns": list(rows[0].keys()),
            "reasons": dict(Counter(r.get("reason") for r in rows)),
            "has_cast_by_model": any("cast_by_model" in r for r in rows),
            "cast_by_model_true": sum(1 for r in rows
                                      if str(r.get("cast_by_model", "")).lower() in ("true", "1")),
            "has_recast": any("recast" in r for r in rows),
            "recast_rows": sum(1 for r in rows
                               if str(r.get("recast", "")).lower() in ("true", "1"))}


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ 3.1 libraries

def find_libraries() -> List[Dict[str, Any]]:
    seen, rows = set(), []
    patterns = ["outputs/agentic_library_*.json", "outputs/agentic_memory_*.json",
                "outputs/mem_*.json", "outputs/*agentic*fold*.json",
                "results/**/agentic*.json", "audit/**/agentic*.json",
                "outputs/*no_trace*.json", "outputs/*notrace*.json",
                "outputs/*menu_only*.json", "outputs/*predicate_menu*.json"]
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            if path in seen or path.name.endswith("_assembly.json"):
                continue
            seen.add(path)
            try:
                record = json.loads(path.read_text())
            except Exception:                                        # noqa: BLE001
                continue
            operators = (record.get("operators") if isinstance(record, dict)
                         and "operators" in record else
                         record.get("layer1", {}).get("operators")
                         if isinstance(record, dict) else record)
            if operators is None:
                continue
            supports = [o.get("support", 0) for o in operators]
            assembly = path.with_name(path.stem + "_assembly.json")
            provenance = derive_provenance(assembly)
            rows.append({
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_of(path),
                "kind": ("memory artefact (3 layers)" if isinstance(record, dict)
                         and "layer1" in record else "operator library"),
                "operators": len(operators),
                "effect_keys": dict(Counter((o.get("effect") or {}).get("key") for o in operators)),
                "kinds": dict(Counter(o.get("kind") for o in operators)),
                "support_min": min(supports) if supports else None,
                "support_max": max(supports) if supports else None,
                "support_mean": round(sum(supports) / len(supports), 2) if supports else None,
                "built_from": record.get("built_from") if isinstance(record, dict) else None,
                "layers_2_3_borrowed_from": (record.get("layers_2_3_borrowed_from")
                                             if isinstance(record, dict) else None),
                "recorded_build_seed": (record.get("seed") if isinstance(record, dict) else None),
                "recorded_pool_hash": (record.get("pool_sha256") if isinstance(record, dict) else None),
                "recorded_framework_version": (record.get("framework_sha256")
                                               if isinstance(record, dict) else None),
                **provenance,
            })
    return rows


def derive_provenance(assembly: Path) -> Dict[str, Any]:
    if not assembly.is_file():
        return {"assembly": None, "sampling_seeds": "未记录",
                "inducer_models": None, "rung_tags": None}
    try:
        record = json.loads(assembly.read_text())
    except Exception:                                                # noqa: BLE001
        return {"assembly": str(assembly.relative_to(ROOT)), "sampling_seeds": "未记录",
                "inducer_models": None, "rung_tags": None}
    tags, samples = set(), set()
    for row in record.get("rows", []):
        for source in row.get("sources", []):
            match = re.search(r"agentic_rung/([^/]+)/e(\d+)_s(\d+)/", source)
            if match:
                tags.add(match.group(1))
                samples.add(int(match.group(3)))
    models = sorted({t.split("_")[0] for t in tags})
    return {"assembly": str(assembly.relative_to(ROOT)),
            "assembly_sha256": sha256_of(assembly),
            "rung_tags": sorted(tags),
            "inducer_models": models,
            "sampling_seed_indices": sorted(samples),
            "sampling_seeds": [20260829 + s for s in sorted(samples)],
            "sampling_seed_source": OBSERVED_SWEEP_SEEDS}


# ------------------------------------------------------------------ 3.2 the 74.24% cell

def analyse_headline(libraries) -> Dict[str, Any]:
    path = A11 / "e2e_agentic_runner.csv"
    if not path.is_file():
        return {"status": "未执行，缺 %s" % path}
    stats = rate_of_csv(path)
    memory = ROOT / "outputs/agentic_memory_runner.json"
    checks = {
        "build_seed_in_required_set": {
            "required": REQUIRED_BUILD_SEEDS,
            "observed": OBSERVED_SWEEP_SEEDS,
            "pass": False,
            "why": "the library was assembled from rung runs sampled at 20260829+k; no "
                   "library on disk was built at any of the three required seeds"},
        "dispatch_arm_is_LLM_RA": {
            "observed": "viki_eval_skill_memory_v2.py, memory dispatch (no cast_by_model "
                        "column in the result; that script's crew is the memory's search)",
            "required": "LLM role assignment",
            "pass": bool(stats.get("has_cast_by_model"))},
        "fallback_rule_matches_spec": {
            "required": SPEC_FALLBACK, "observed": SHIPPED_FALLBACK, "pass": False},
        "scoring_and_split": {
            "observed": "VIKI-L2 test split, JSON-tolerant, replayed from "
                        "probe2_zeroshot_v2.jsonl", "pass": True},
    }
    return {"status": "found", "path": str(path.relative_to(ROOT)),
            "sha256": sha256_of(path), **stats,
            "memory_artefact": str(memory.relative_to(ROOT)) if memory.is_file() else "缺",
            "memory_sha256": sha256_of(memory) if memory.is_file() else None,
            "checks": checks,
            "usable_directly": all(c["pass"] for c in checks.values())}


# ------------------------------------------------------------------ 3.3/3.6 result cells

MODEL_FROM = [(r"(^|[_/])m?72b?([_/.]|$)", "72B"), (r"(^|[_/])m?30b?([_/.]|$)", "30B"),
              (r"(^|[_/])m?7b?([_/.]|$)", "7B")]
SPLIT_FROM = [(r"recomb.*imaged|imaged.*recomb", "comp_imaged"),
              (r"recomb.*text|text.*recomb", "comp_text"),
              (r"fold", "heldout_family"), (r"(^|_)id(_|$|\.)", "id"),
              # e2e_* and matrix_* are the ID split by construction of those drivers.
              (r"^e2e_|^matrix_|^oracle_", "id")]
ARM_FROM = [(r"agentic|agent_lib", "ours"), (r"reference|ref_", "reference"),
            (r"gmem|g_memory", "G-Memory"), (r"memento", "MEMENTO-style"),
            (r"zeroshot|zero_shot", "zero-shot"), (r"ragR|traj|rag", "trajectory RAG")]


def classify(name: str) -> Dict[str, Optional[str]]:
    lowered = name.lower()
    def first(table, default=None):
        for pattern, value in table:
            if re.search(pattern, lowered):
                return value
        return default
    return {"model": first(MODEL_FROM, "72B" if "m30" not in lowered and "m7" not in lowered else None),
            "split": first(SPLIT_FROM), "arm": first(ARM_FROM)}


def find_result_cells() -> List[Dict[str, Any]]:
    rows = []
    candidates = (list(A11.glob("*.csv")) + list(ROOT.glob("outputs/recomb/*.csv"))
                  + list(A11.glob("*.jsonl")))
    for path in sorted(candidates):
        tag = path.stem
        info = classify(tag)
        if info["arm"] is None and info["split"] is None:
            continue
        stats = rate_of_csv(path)
        if not stats.get("rows"):
            continue
        run = re.search(r"_r(\d)$", tag)
        rows.append({"tag": tag, "path": str(path.relative_to(ROOT)),
                     "sha256": sha256_of(path), **info,
                     "run_index": int(run.group(1)) if run else None,
                     "rows": stats.get("rows"), "rate": stats.get("rate"),
                     "dispatch": ("LLM RA" if stats.get("has_cast_by_model")
                                  else ("memory RA" if stats.get("rows") else "unknown")),
                     "cast_by_model_true": stats.get("cast_by_model_true"),
                     "recast_rows": stats.get("recast_rows"),
                     "reasons": stats.get("reasons")})
    return rows


def find_responses() -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(list(A11.glob("*.jsonl"))):
        info = classify(path.stem)
        if info["arm"] is None and info["split"] is None:
            continue
        rows.append({"tag": path.stem, "path": str(path.relative_to(ROOT)),
                     "sha256": sha256_of(path), "bytes": path.stat().st_size, **info})
    return rows


# ------------------------------------------------------------------ 3.4/3.5

def find_per_fold_agent() -> Dict[str, Any]:
    searched = ["outputs/agentic_*fold*.json", "outputs/*fold*agentic*.json",
                "results/**/agentic*fold*.json", "outputs/agentic_rung/*fold*"]
    hits = sorted({str(p.relative_to(ROOT)) for pattern in searched for p in ROOT.glob(pattern)})
    return {"searched": searched, "hits": hits,
            "status": "found" if hits else "未执行，缺 per-fold agent Layer 1（盘上不存在）",
            "agrees_with_audit_L6": not hits}


def find_no_trace() -> Dict[str, Any]:
    searched = ["outputs/*no_trace*", "outputs/*notrace*", "outputs/*menu_only*",
                "outputs/*predicate_menu*", "outputs/agentic_rung/*no_trace*",
                "outputs/agentic_rung/*menu*"]
    hits = sorted({str(p.relative_to(ROOT)) for pattern in searched for p in ROOT.glob(pattern)})
    return {"searched": searched, "hits": hits,
            "status": "found" if hits else "未执行，缺 no-trace control（doc §5.3 arm (d) 从未建过）"}


# ------------------------------------------------------------------ 3.3 pool constraint

def check_pools() -> Dict[str, Any]:
    support = AUDIT / "L0_support_pool.json"
    induction = AUDIT / "L0_induction_half.json"
    if not (support.is_file() and induction.is_file()):
        return {"status": "未执行，缺 %s" % AUDIT}
    return {"status": "found",
            "audit_dir": str(AUDIT.relative_to(ROOT)),
            "L0_induction_half_sha256": sha256_of(induction),
            "L0_support_pool_sha256": sha256_of(support),
            "note": "any new build must reproduce these two hashes; a mismatch stops the run"}


# ------------------------------------------------------------------ matrix

def coverage_matrix(cells, headline, no_trace, per_fold) -> List[Dict[str, Any]]:
    have = defaultdict(list)
    for cell in cells:
        if cell["model"] and cell["split"] and cell["arm"]:
            have[(cell["model"], cell["split"], cell["arm"])].append(cell)
    rows = []
    for model in MODELS:
        for split in SPLITS:
            for arm in ARMS:
                found = have.get((model, split, arm), [])
                for seed in REQUIRED_BUILD_SEEDS:
                    # No artefact on disk was built at a required seed, so every "ours"
                    # and every ablation derived from it is missing by seed, whatever
                    # else exists.
                    reusable = False
                    reason = ""
                    if arm in ("ours", "w/o Ordering", "w/o Grounding", "no-trace control"):
                        reason = "build seed not in {20260901,02,03}; dispatch/fallback also differ"
                    elif arm == "reference":
                        reason = "appendix only; rule-based library, no build seed required"
                        reusable = bool(found)
                    else:
                        reason = ("baseline archive; reusable if scoring convention matches"
                                  if found else "no archive located")
                        reusable = bool(found)
                    rows.append({
                        "model": model, "split": split, "arm": arm, "build_seed": seed,
                        "status": "found" if (found and reusable) else "missing",
                        "artefacts_on_disk": len(found),
                        "example_path": found[0]["path"] if found else "",
                        "example_sha256": found[0]["sha256"] if found else "",
                        "observed_dispatch": found[0]["dispatch"] if found else "",
                        "convention_matches_spec": "yes" if reusable else "no",
                        "reason": reason,
                    })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / ("results/agent_library_%s" % date.today().isoformat()))
    arguments = parser.parse_args(argv)
    out = arguments.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    libraries = find_libraries()
    headline = analyse_headline(libraries)
    cells = find_result_cells()
    responses = find_responses()
    per_fold = find_per_fold_agent()
    no_trace = find_no_trace()
    pools = check_pools()
    matrix = coverage_matrix(cells, headline, no_trace, per_fold)

    payload = {"libraries": libraries, "headline_74_24": headline,
               "result_cells": cells, "responses": responses,
               "per_fold_agent_libraries": per_fold, "no_trace_control": no_trace,
               "pool_constraint": pools,
               "required_build_seeds": REQUIRED_BUILD_SEEDS,
               "observed_sweep_seeds": OBSERVED_SWEEP_SEEDS,
               "spec_fallback": SPEC_FALLBACK, "shipped_fallback": SHIPPED_FALLBACK}
    (out / "inventory.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))

    with (out / "coverage_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix[0].keys()))
        writer.writeheader()
        writer.writerows(matrix)

    write_inventory_md(out, payload, matrix)
    print("libraries %d | result cells %d | responses %d" % (len(libraries), len(cells), len(responses)))
    print("coverage rows %d, found %d, missing %d"
          % (len(matrix), sum(1 for r in matrix if r["status"] == "found"),
             sum(1 for r in matrix if r["status"] == "missing")))
    print("wrote %s" % out)
    return 0


def write_inventory_md(out: Path, payload, matrix) -> None:
    lines = ["# INVENTORY", "", "## 3.1 agent 建的库", ""]
    lines.append("| path | sha256 | ops | inducer | sampling seeds | build seed recorded |")
    lines.append("|---|---|---|---|---|---|")
    for row in payload["libraries"]:
        lines.append("| `%s` | `%s` | %s | %s | %s | %s |" % (
            row["path"], row["sha256"][:16], row["operators"],
            ",".join(row.get("inducer_models") or []) or "—",
            row.get("sampling_seeds") or "未记录",
            row.get("recorded_build_seed") if row.get("recorded_build_seed") is not None else "未记录"))
    lines += ["", "要求的建库 seed：%s" % payload["required_build_seeds"],
              "", "盘上观测到的：%s" % payload["observed_sweep_seeds"], ""]

    lines += ["## 3.2 74.24% 的来源", "", "```json",
              json.dumps(payload["headline_74_24"], indent=1, ensure_ascii=False)[:3000],
              "```", ""]
    lines += ["## 3.3 池子约束", "", "```json",
              json.dumps(payload["pool_constraint"], indent=1, ensure_ascii=False), "```", ""]
    lines += ["## 3.4 per-fold agent 库", "", "```json",
              json.dumps(payload["per_fold_agent_libraries"], indent=1, ensure_ascii=False),
              "```", ""]
    lines += ["## 3.5 no-trace control", "", "```json",
              json.dumps(payload["no_trace_control"], indent=1, ensure_ascii=False), "```", ""]
    lines += ["## 3.6 回退规则", "",
              "- spec 要求：%s" % payload["spec_fallback"],
              "- 盘上实现：%s" % payload["shipped_fallback"], ""]
    lines += ["## 3.3/3.6 已有结果格", "",
              "| tag | model | split | arm | run | rows | rate | dispatch | recast |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in payload["result_cells"]:
        lines.append("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["tag"], row["model"] or "—", row["split"] or "—", row["arm"] or "—",
            row["run_index"] if row["run_index"] is not None else "—",
            row["rows"], row["rate"], row["dispatch"],
            row["recast_rows"] if row["recast_rows"] is not None else "—"))
    lines += ["", "## 3.7 覆盖矩阵", "",
              "见 `coverage_matrix.csv`（%d 行，found %d，missing %d）。" % (
                  len(matrix), sum(1 for r in matrix if r["status"] == "found"),
                  sum(1 for r in matrix if r["status"] == "missing")), ""]
    lines += ["## 附录：产物", ""]
    for name in ("inventory.json", "coverage_matrix.csv"):
        path = out / name
        lines.append("- `%s`  sha256 `%s`  %d bytes" % (name, sha256_of(path), path.stat().st_size))
    (out / "INVENTORY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
