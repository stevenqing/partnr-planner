#!/usr/bin/env python3
"""Export the VIKI-L2 ICLR 2027 supplementary results from row-level artifacts.

Everything this script writes is computed from row files named in a cell registry
(`results/paper_viki_iclr2027/cells.json`). There are no result numbers in this file:
expected sample counts come from the split manifests, success comes from the scorer, and
every aggregate, contingency table and p-value is derived from the normalised rows.

Two modes:

  --init-registry   build the first registry from the archive layout (refuses to overwrite
                    an existing registry unless --force-registry). Cells that do not exist
                    yet are registered with status "missing".
  (default)         read the registry, normalise every available cell to the paper row
                    schema, check it (expected_n, duplicate example_id, manifest equality,
                    replay equality, run.json model identity, baseline_comparison.json
                    equality), and write summaries, paired tests, run_manifest.json and
                    paper_viki_results.md. Any failed check makes the cell complete=false,
                    and an incomplete cell publishes no headline aggregate. Missing rows are
                    never filled.

Optional --check-anchors FILE compares recomputed counts against an external anchor list
and only PRINTS the comparison (exit status 3 on mismatch); anchors are never written into
any output.

Must run on the remote box (row artifacts live there):
  /root/venvs/partnr/bin/python scripts/export_paper_viki_results.py --init-registry
  /root/venvs/partnr/bin/python scripts/export_paper_viki_results.py
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DEFAULT = ROOT / "results/paper_viki_iclr2027"
A8B_REL = "results/viki_memory_experiments/amendment8b"
A10_REL = "results/viki_memory_experiments/amendment10"
A11_REL = "results/viki_memory_experiments/amendment11"
V3_MEMORIES_REL = "outputs/v3_memories"
FROZEN_SWEEP_REL = "results/frozen_sweep_v3.json"
BASELINE_COMPARISON_REL = "results/agent_library_v3/baseline_comparison.json"
ID_PARQUET_REL = "../VIKI-R/data/VIKI-R/viki/VIKI-L2/test.parquet"  # resolved via BENCHMARK_ROOT

CANONICAL_SPLITS = ["id", "ood_single_family", "cg_image", "pure_text"]
SPLIT_LABEL = {"id": "ID", "ood_single_family": "OOD", "cg_image": "CG w/ Image",
               "pure_text": "Pure Text"}
EXPERIMENT_CONDITIONS = {
    "figure2": ["ours", "tom", "g_memory", "memento_style", "trajectory_rag", "zero_shot"],
    "rq2": ["full", "no_trace", "no_execution_admission"],
    "rq3": ["full", "no_grounding", "no_order"],
}
PAIRS = {
    "figure2": [("ours", "tom")],
    "rq2": [("full", "no_trace"), ("full", "no_execution_admission")],
    "rq3": [("full", "no_grounding"), ("full", "no_order")],
}
SUMMARY_FILE = {"figure2": "figure2_tom_summary.csv", "rq2": "rq2_summary.csv",
                "rq3": "rq3_summary.csv"}
SUMMARY_COLUMNS = [
    "experiment", "model_display_name", "model_id", "condition", "canonical_split",
    "source_split_name", "repeat", "seed", "expected_n", "scored_n", "success_count",
    "success_rate", "parse_success_count", "parse_failure_count", "complete", "row_file",
    "row_file_sha256", "prompt_sha256", "scorer_sha256", "manifest_sha256",
]
# Appended after the required columns so the required ones keep their order.
SUMMARY_EXTRA_COLUMNS = ["cell_id", "task_display_name", "status", "reuse",
                         "provenance_status", "incomplete_reasons"]
PAIRED_COLUMNS = [
    "experiment", "model_display_name", "canonical_split", "arm_a", "arm_b", "n_paired",
    "both_success", "arm_a_only_success", "arm_b_only_success", "both_failure", "arm_a_rate",
    "arm_b_rate", "absolute_delta", "mcnemar_exact_p", "complete",
]

# Baseline arm name as used by viki_p0_report.METHODS and baseline_comparison.json.
BASELINE_ARMS = {
    "zero_shot": "zero-shot",
    "trajectory_rag": "trajectory RAG",
    "g_memory": "G-Memory",
    "memento_style": "MEMENTO-style",
}
AUDIT_ONLY_ARMS = {"skill_memory_v1": "skill memory v1"}  # never exported to Figure 2
OURS_BC_ARM = "ours (agent library)"
# No-think archives: `*_nt`, `*.m30nt`, `*.m7ntr2`, ... never enter a think cell.
NO_THINK_NAME = re.compile(r"(?:^|[._])(?:m30|m7)?nt(?:r\d+)?(?=$|[._])")
METHOD_DISPLAY = {
    "ours": "Ours", "tom": "ToM", "g_memory": "G-Memory", "memento_style": "MEMENTO-style",
    "trajectory_rag": "Trajectory RAG", "zero_shot": "Zero-shot",
    "skill_memory_v1": "Skill Memory v1", "full": "full", "no_trace": "no_trace",
    "no_execution_admission": "no_execution_admission", "no_grounding": "no_grounding",
    "no_order": "no_order",
}


# ----------------------------------------------------------------------------- utilities
def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def mtime_iso(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows, errors = [], []
    with open(path) as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append("line %d: %s" % (number, error))
                continue
            if not isinstance(record, dict) or "index" not in record:
                errors.append("line %d: not a row with an index" % number)
                continue
            rows.append(record)
    return rows, errors


def mcnemar_exact(arm_a_only: int, arm_b_only: int) -> float:
    """Two-sided exact McNemar: binomial(n=discordant, p=1/2) on the smaller count."""
    n = arm_a_only + arm_b_only
    if n == 0:
        return 1.0
    k = min(arm_a_only, arm_b_only)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / (2 ** n))


def git_info() -> Dict[str, Any]:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                                  text=True, timeout=120).stdout
        except Exception as error:  # noqa: BLE001
            return "ERR %r" % error
    porcelain = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD").strip(),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "dirty": bool(porcelain.strip()),
            "dirty_entries": [l for l in porcelain.splitlines() if l.strip()]}


def env_info() -> Dict[str, Any]:
    info = {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "hostname": platform.node(), "packages": {}}
    for name in ("pandas", "numpy", "scipy", "pyarrow"):
        try:
            info["packages"][name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            info["packages"][name] = None
    return info


def python_constant_sha(path: Path, name: str) -> Optional[str]:
    """SHA-256 of a module-level string constant, read without importing the module."""
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                return sha256_text(ast.literal_eval(node.value))
            except Exception:  # noqa: BLE001
                return None
    return None


# ----------------------------------------------------------------------------- registry
def build_registry() -> Dict[str, Any]:
    import pandas as pd
    import viki_p0_report as p0
    from our_method.skill_memory_v2.simulator import SEED as SIM_SEED

    a8b, a10, a11 = ROOT / A8B_REL, ROOT / A10_REL, ROOT / A11_REL
    frozen = json.loads((ROOT / FROZEN_SWEEP_REL).read_text())
    fold_families = list(frozen["fold_families"])
    fold_dirs = sorted(d.name for d in (a8b / "folds").iterdir() if d.is_dir())
    if sorted(fold_families) != fold_dirs:
        raise SystemExit("fold families in %s != %s/folds dirs" % (FROZEN_SWEEP_REL, A8B_REL))

    id_parquet = p0.BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet"
    manifest_path = a8b / "interactive_manifest.jsonl"
    manifest_n = sum(1 for l in manifest_path.read_text().splitlines() if l.strip())
    imaged_parquet = a10 / "recombination.imaged.parquet"
    text_parquet = a10 / "recombination.text.parquet"
    splits = {
        "id": {"source_split_name": "id", "truth_frame": "id", "expected_n": manifest_n,
               "manifest_path": rel(manifest_path), "truth_path": str(id_parquet),
               "row_key": "index (row of test.parquet, restricted to the manifest)"},
        "ood_single_family": {
            "source_split_name": "heldout", "truth_frame": "id", "expected_n": manifest_n,
            "manifest_path": rel(manifest_path), "truth_path": str(id_parquet),
            "row_key": "index; family X's rows come from the fold whose memory excluded X",
            "fold_families": fold_families},
        "cg_image": {"source_split_name": "imaged", "truth_frame": "imaged",
                     "expected_n": len(pd.read_parquet(imaged_parquet)),
                     "manifest_path": rel(imaged_parquet), "truth_path": rel(imaged_parquet),
                     "row_key": "row number of recombination.imaged.parquet"},
        "pure_text": {"source_split_name": "text", "truth_frame": "text",
                      "expected_n": len(pd.read_parquet(text_parquet)),
                      "manifest_path": rel(text_parquet), "truth_path": rel(text_parquet),
                      "row_key": "row number of recombination.text.parquet"},
    }

    def zs_run_json(tag):
        path = p0.cell_path(tag, p0.METHODS[0], "id")
        rj = Path(str(path) + ".run.json")
        return (json.loads(rj.read_text()), rel(rj)) if rj.is_file() else ({}, None)

    models = {}
    spec = [
        ("72B", "Qwen2.5-VL-72B-Instruct", "Qwen2.5-72B", "qwen2.5-vl-72b-instruct",
         "qwen2.5-vl-72b-amendment3-f2", "http://192.168.32.40:8050/v1", "",
         {"id": "intent_crew_clean.jsonl", "cg_image": "recomb_imaged_agentic.jsonl",
          "pure_text": "recomb_text_agentic.jsonl"}),
        ("30B", "Qwen3-VL-30B-A3B-Instruct", "Qwen3-8B", "qwen3-vl-30b-a3b-instruct",
         "qwen3-vl-30b", "http://127.0.0.1:8062/v1", "m30",
         {"id": "m30_id.jsonl", "cg_image": "m30_recomb_imaged.jsonl",
          "pure_text": "m30_recomb_text.jsonl"}),
        ("7B", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-7B-Instruct", "qwen2.5-vl-7b-instruct",
         "qwen2.5-vl-7b", "http://127.0.0.1:8061/v1", "m7",
         {"id": "m7_id.jsonl", "cg_image": "m7_recomb_imaged.jsonl",
          "pure_text": "m7_recomb_text.jsonl"}),
    ]
    for tag, display, task_display, slug, served, endpoint, archive_tag, replay in spec:
        rj, rj_path = zs_run_json(tag)
        models[tag] = {
            "model_display_name": display, "task_display_name": task_display,
            "model_slug": slug, "model_id": served, "endpoint": endpoint,
            "archive_tag": archive_tag,
            "model_revision": rj.get("model_revision"),
            "model_revision_source": rj_path,
            "ours_replay_sources": {k: "%s/%s" % (A11_REL, v) for k, v in replay.items()},
        }

    prompts = {
        "ours": "scripts/viki_eval_v2_intent_choice.py",
        "baseline_id": "scripts/viki_amendment8b.py",
        "baseline_recomb": "scripts/viki_amendment10_run.py",
        "memento": "scripts/viki_eval_memento.py",
    }

    def run_json_of(path: Path) -> Optional[str]:
        rj = Path(str(path) + ".run.json")
        return rel(rj) if rj.is_file() else None

    def base_entry(experiment, condition, tag, split):
        return {
            "cell_id": "%s/%s/%s/%s" % (experiment, condition, tag, split),
            "experiment": experiment, "condition": condition,
            "method_display": METHOD_DISPLAY[condition], "model_tag": tag,
            "canonical_split": split,
            "source_split_name": splits[split]["source_split_name"],
            "expected_n": splits[split]["expected_n"], "repeat": 1, "seed": None,
            "status": "missing", "reuse": "missing", "provenance_status": "not_applicable",
            "scorer_kind": None, "source": None, "cross_check_file": None,
            "replay_source": None, "prompt_path": None, "library_path": None,
            "run_metadata_path": None, "log_path": None, "command": None,
            "config_path": None, "notes": "",
        }

    def ours_entry(experiment, condition, tag, split):
        e = base_entry(experiment, condition, tag, split)
        e.update({"status": "available", "reuse": "reused", "provenance_status": "verified",
                  "scorer_kind": "simulator_solved", "prompt_path": prompts["ours"],
                  "seed": SIM_SEED, "seed_source": "our_method/skill_memory_v2/simulator.py SEED",
                  "config_path": "scripts/drivers/viki_v2_evaluate.py (--tag-prefix v3, "
                                 "--out-root %s)" % V3_MEMORIES_REL})
        if split == "ood_single_family":
            e["source"] = {"kind": "folds", "fold_rule": "keep record.task_name == fold family",
                           "files": {f: "%s/v3_fold_%s_%s.jsonl" % (A11_REL, tag, f)
                                     for f in fold_families}}
            e["library_path"] = {f: "%s/memory_heldout_%s.json" % (V3_MEMORIES_REL, f)
                                 for f in fold_families}
            e["cross_check_file"] = "%s/v3_ours_%s_heldout.jsonl" % (A11_REL, tag)
            e["replay_source"] = models[tag]["ours_replay_sources"]["id"]
        else:
            stem = {"id": "v3_ours_%s_id", "cg_image": "v3_oursall_%s_imaged",
                    "pure_text": "v3_oursall_%s_text"}[split] % tag
            e["source"] = {"kind": "file", "path": "%s/%s.jsonl" % (A11_REL, stem)}
            e["library_path"] = "%s/memory_all.json" % V3_MEMORIES_REL
            e["replay_source"] = models[tag]["ours_replay_sources"][split]
        return e

    def baseline_entry(experiment, condition, arm, tag, split):
        e = base_entry(experiment, condition, tag, split)
        e.update({"status": "available", "reuse": "reused", "provenance_status": "verified",
                  "scorer_kind": "json_tolerant", "baseline_comparison_arm": arm,
                  "notes": "library/memory is built inside the baseline harness; see prompt_path"})
        method = next((m for m in p0.METHODS if m[0] == arm), None)
        archive_tag = models[tag]["archive_tag"]
        if split == "ood_single_family":
            files = {}
            for fam in fold_families:
                if arm == "MEMENTO-style":
                    name = ("memento_fold_%s.jsonl" % fam if tag == "72B"
                            else "memento_fold_%s_%s.jsonl" % (archive_tag, fam))
                    files[fam] = "%s/%s" % (A11_REL, name)
                else:
                    fold_stem = {"zero-shot": "zero_shot", "trajectory RAG": "trajectory_rag",
                                 "skill memory v1": "skill_memory.fullactions_k8",
                                 "G-Memory": "gmemory"}[arm]
                    arm_stem = {"zero-shot": "zero_shot", "trajectory RAG": "trajectory_rag",
                                "skill memory v1": "skill_memory", "G-Memory": "gmemory"}[arm]
                    name = ("%s.jsonl" % fold_stem if tag == "72B"
                            else "%s.%s.jsonl" % (arm_stem, archive_tag))
                    files[fam] = "%s/folds/%s/%s" % (A8B_REL, fam, name)
            e["source"] = {"kind": "folds", "fold_rule": "all rows of each fold file; every row "
                           "must belong to the fold family", "files": files}
            e["run_metadata_path"] = {f: run_json_of(ROOT / p) for f, p in files.items()}
            e["prompt_path"] = prompts["memento" if arm == "MEMENTO-style" else "baseline_id"]
        else:
            source_split = splits[split]["source_split_name"]
            if arm == "MEMENTO-style":
                name = {("72B", "id"): "memento_id.jsonl",
                        ("72B", "text"): "memento_recomb_text.jsonl",
                        ("72B", "imaged"): "memento_recomb_imaged.jsonl"}.get((tag, source_split))
                if name is None:
                    name = {"id": "memento_id_%s.jsonl", "text": "memento_recomb_text_%s.jsonl",
                            "imaged": "memento_recomb_imaged_%s.jsonl"}[source_split] % archive_tag
                path = ROOT / A11_REL / name
                e["prompt_path"] = prompts["memento"]
            else:
                path = p0.cell_path(tag, method, source_split)
                e["prompt_path"] = prompts["baseline_id" if split == "id" else "baseline_recomb"]
            e["source"] = {"kind": "file", "path": rel(path)}
            e["run_metadata_path"] = run_json_of(path)
        return e

    cells, audit_cells = [], []
    for tag in models:
        for split in CANONICAL_SPLITS:
            # Figure 2
            cells.append(ours_entry("figure2", "ours", tag, split))
            e = base_entry("figure2", "tom", tag, split)
            e.update({"scorer_kind": "json_tolerant", "prompt_path": "prompts/viki_tom_prompt.txt",
                      "notes": "ToM arm not run yet; to be newly_executed"})
            cells.append(e)
            for condition, arm in BASELINE_ARMS.items():
                cells.append(baseline_entry("figure2", condition, arm, tag, split))
            for condition, arm in AUDIT_ONLY_ARMS.items():
                audit_cells.append(baseline_entry("audit", condition, arm, tag, split))
            # RQ2
            cells.append(ours_entry("rq2", "full", tag, split))
            for condition in ("no_trace", "no_execution_admission"):
                e = base_entry("rq2", condition, tag, split)
                e.update({"scorer_kind": "simulator_solved", "prompt_path": prompts["ours"],
                          "notes": "condition not run yet; to be newly_executed"})
                cells.append(e)
            # RQ3
            cells.append(ours_entry("rq3", "full", tag, split))
            for condition, flag, file_tag in (("no_grounding", "--no-grounding", "noground"),
                                              ("no_order", "--no-order", "noorder")):
                e = base_entry("rq3", condition, tag, split)
                e.update({"scorer_kind": "simulator_solved", "prompt_path": prompts["ours"]})
                src_split = {"id": "id", "cg_image": "imaged", "pure_text": "text"}.get(split)
                path = (ROOT / A11_REL / ("v3_ablfull_%s_%s_%s.jsonl" % (file_tag, tag, src_split))
                        if src_split else None)
                if path is not None and path.is_file():
                    e.update({
                        "status": "available", "reuse": "reused",
                        "provenance_status": "verified" if tag == "72B" else "provenance_unverified",
                        "source": {"kind": "file", "path": rel(path)},
                        "library_path": "%s/memory_all.json" % V3_MEMORIES_REL,
                        "library_path_source": "inferred from driver convention "
                                               "`TAG=v3 M=%s`; rows carry no library hash"
                                               % V3_MEMORIES_REL,
                        "replay_source": models[tag]["ours_replay_sources"][split],
                        "seed": SIM_SEED,
                        "seed_source": "our_method/skill_memory_v2/simulator.py SEED",
                        "log_path": "outputs/abl_%s.detail.log" % tag,
                        "command": "TAG=v3 M=%s MODEL=%s bash scripts/drivers/viki_ablations_full.sh"
                                   % (V3_MEMORIES_REL, tag),
                        "config_path": "scripts/drivers/viki_ablations_full.sh (%s)" % flag,
                        "notes": ("" if tag == "72B" else
                                  "rows carry no library hash; to be re-run on the v3 library "
                                  "before use"),
                    })
                else:
                    e["notes"] = "condition not run yet on the current v3 library; to be newly_executed"
                cells.append(e)

    return {
        "format": "paper_viki_iclr2027.cells/1",
        "generated_by": "scripts/export_paper_viki_results.py --init-registry",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canonical_splits": CANONICAL_SPLITS,
        "splits": splits,
        "models": models,
        "fold_families": fold_families,
        "prompts": prompts,
        "baseline_comparison_path": BASELINE_COMPARISON_REL,
        "cells": cells,
        "audit_cells": audit_cells,
        "audit_cells_note": "scored only to check equality with baseline_comparison.json; "
                            "never written to rows/ or any summary (Skill Memory v1 is excluded "
                            "from Figure 2)",
    }


# ----------------------------------------------------------------------------- scoring
class Scorers:
    def __init__(self, registry):
        import pandas as pd
        import viki_p0_report as p0
        from habitat_llm.evaluation import viki_bench as bench
        from our_method.skill_memory_v2 import Simulator
        from viki_amendment9_diag102 import parse_plan
        from viki_amendment11_goalparse import extract_json
        from viki_report_matrix import tolerant

        self.bench, self.parse_plan, self.extract_json, self.tolerant = (
            bench, parse_plan, extract_json, tolerant)
        self.sim = Simulator(p0.BENCHMARK_ROOT)
        self.frames = {
            "id": pd.read_parquet(p0.BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet"),
            "imaged": pd.read_parquet(ROOT / A10_REL / "recombination.imaged.parquet"),
            "text": pd.read_parquet(ROOT / A10_REL / "recombination.text.parquet"),
        }
        self.truths: Dict[str, Dict[int, Any]] = {}
        manifest_path = ROOT / registry["splits"]["id"]["manifest_path"]
        self.manifest = sorted(int(json.loads(l)["index"]) for l in
                               manifest_path.read_text().splitlines() if l.strip())
        self.expected_ids = {
            "id": set(self.manifest), "ood_single_family": set(self.manifest),
            "cg_image": set(range(len(self.frames["imaged"]))),
            "pure_text": set(range(len(self.frames["text"]))),
        }
        # Scorer provenance: the files whose code decides success, per scorer kind.
        official = inspect.getfile(self.sim.scorer.eval_single)
        planner_mod = __import__("our_method.skill_memory_v2.planner", fromlist=["x"])
        simulator_mod = __import__("our_method.skill_memory_v2.simulator", fromlist=["x"])
        self.scorer_files = {
            "json_tolerant": {
                "scorer (tolerant)": ROOT / "scripts/viki_report_matrix.py",
                "parser (parse_plan)": ROOT / "scripts/viki_amendment9_diag102.py",
                "habitat_llm viki_bench": Path(bench.__file__),
                "official scorer eval_single module": Path(official),
            },
            "simulator_solved": {
                "runner (reason == SOLVED)": ROOT / "scripts/viki_eval_v2_intent_choice.py",
                "parser (extract_json)": ROOT / "scripts/viki_amendment11_goalparse.py",
                "simulator": Path(simulator_mod.__file__),
                "planner": Path(planner_mod.__file__),
                "habitat_llm viki_bench": Path(bench.__file__),
            },
        }
        self.scorer_sha = {}
        self.scorer_components = {}
        for kind, files in self.scorer_files.items():
            parts = {label: {"path": str(p), "sha256": sha256_file(p)} for label, p in files.items()}
            self.scorer_components[kind] = parts
            digest = "\n".join("%s %s" % (v["path"], v["sha256"]) for _, v in sorted(parts.items()))
            self.scorer_sha[kind] = sha256_text(digest)

    def truth(self, frame: str, index: int):
        table = self.truths.setdefault(frame, {})
        if index not in table:
            table[index] = self.bench.get_ground_truth(
                self.bench.to_native(self.frames[frame].iloc[index].to_dict()))
        return table[index]


def load_source(entry, root: Path) -> Tuple[List[Tuple[Dict[str, Any], str]], Dict[str, Any]]:
    """Rows as (record, source_path) plus per-file facts. No row is invented."""
    facts: Dict[str, Any] = {"files": [], "errors": [], "missing_files": []}
    source = entry["source"]
    pairs: List[Tuple[Dict[str, Any], str]] = []
    if source["kind"] == "file":
        items = [(None, source["path"])]
    else:
        items = sorted(source["files"].items())
    for family, path_rel in items:
        path = root / path_rel
        if not path.is_file():
            facts["missing_files"].append(path_rel)
            continue
        rows, errors = read_jsonl(path)
        rj_path = Path(str(path) + ".run.json")
        run_json = json.loads(rj_path.read_text()) if rj_path.is_file() else None
        facts["files"].append({"path": path_rel, "sha256": sha256_file(path), "lines": len(rows),
                               "fold_family": family, "mtime": mtime_iso(path),
                               "run_json_path": rel(rj_path) if run_json is not None else None,
                               "run_json": run_json})
        facts["errors"] += ["%s: %s" % (path_rel, e) for e in errors]
        for record in rows:
            record = dict(record)
            record["__fold_family"] = family
            pairs.append((record, path_rel))
    return pairs, facts


def evaluate_cell(entry, registry, scorers: Scorers) -> Dict[str, Any]:
    split = entry["canonical_split"]
    model = registry["models"][entry["model_tag"]]
    frame = registry["splits"][split]["truth_frame"]
    reasons: List[str] = []
    result: Dict[str, Any] = {"entry": entry, "rows": [], "incomplete_reasons": reasons,
                              "checks": {}, "facts": {}}
    if entry["status"] != "available" or not entry.get("source"):
        reasons.append("missing: cell not run")
        return result

    pairs, facts = load_source(entry, ROOT)
    result["facts"] = facts
    result["produced_n"] = sum(f["lines"] for f in facts["files"])
    if facts["missing_files"]:
        reasons.append("missing source files: %d" % len(facts["missing_files"]))
    if facts["errors"]:
        reasons.append("corrupt row lines: %d" % len(facts["errors"]))
    for f in facts["files"]:
        if NO_THINK_NAME.search(Path(f["path"]).name[: -len(".jsonl")]):
            reasons.append("no-think file in a think cell: %s" % f["path"])
        rj = f["run_json"] or {}
        if rj.get("served_model") not in (None, model["model_id"]):
            reasons.append("run.json served_model %r != %r in %s"
                           % (rj.get("served_model"), model["model_id"], f["path"]))
        if rj.get("no_think") or str(rj.get("think", "")).lower() in ("false", "0"):
            reasons.append("run.json says no-think: %s" % f["path"])

    kind = entry["scorer_kind"]
    fold_rule_ours = entry["source"]["kind"] == "folds" and kind == "simulator_solved"
    dropped_other_family = rows_outside_family = family_mismatch = 0
    request_failed = no_archived = reask_error = think_tag_rows = 0
    normalised = []
    for record, path_rel in pairs:
        index = int(record["index"])
        fold_family = record.pop("__fold_family")
        if index not in scorers.expected_ids[split]:
            truth = None
            family = record.get("task_name")
        else:
            truth = scorers.truth(frame, index)
            family = truth.get("task_name")
        if fold_family is not None:
            if fold_rule_ours:
                if record.get("task_name") != fold_family:
                    dropped_other_family += 1
                    continue
            elif family != fold_family:
                rows_outside_family += 1
        if kind == "simulator_solved" and record.get("task_name") not in (None, family):
            family_mismatch += 1
        if kind == "simulator_solved":
            raw = record.get("raw")
            parsed = scorers.extract_json(raw) if isinstance(raw, str) else None
            parse_success = int(isinstance(parsed, dict) and isinstance(parsed.get("work"), list))
            success = int(record.get("reason") == "SOLVED")
            reason = str(record.get("reason", ""))
            request_failed += int(reason.startswith("REQUEST_FAILED"))
            no_archived += int(reason == "NO_ARCHIVED_ANSWER")
            reask_error += int(bool(record.get("reask_error")))
            extra = {"reason": record.get("reason"), "raw_reask": record.get("raw_reask"),
                     "replayed": record.get("replayed"), "plan_len": record.get("plan_len"),
                     "budget": record.get("budget"), "stated": record.get("stated"),
                     "cast_by_model": record.get("cast_by_model")}
        else:
            raw = record.get("response") or record.get("raw") or ""
            parsed = scorers.parse_plan(raw)
            parse_success = int(parsed is not None)
            success = (scorers.tolerant(scorers.sim, raw, truth) if truth is not None else 0)
            think_tag_rows += int("<think>" in raw)
            extra = {"row_prompt_sha256": record.get("prompt_sha256"),
                     "official_format_score": record.get("format_score")}
        rj = next((f["run_json"] for f in facts["files"] if f["path"] == path_rel), None) or {}
        normalised.append({
            "example_id": index,
            "family": family,
            "split": split,
            "canonical_split": split,
            "source_split_name": entry["source_split_name"],
            "fold_family": fold_family,
            "raw_output": raw,
            "parsed_output": parsed,
            "target": truth,
            "success": int(success),
            "parse_success": parse_success,
            "run_id": None,  # filled below
            "model_display_name": model["model_display_name"],
            "task_display_name": model["task_display_name"],
            "model_id": model["model_id"],
            "endpoint": model["endpoint"],
            "model_revision": rj.get("model_revision", model.get("model_revision")),
            "condition": entry["condition"],
            "experiment": entry["experiment"],
            "reask": bool(record.get("reask")),
            "scorer_kind": kind,
            "source_file": path_rel,
            **extra,
        })

    ids = [r["example_id"] for r in normalised]
    counts = Counter(ids)
    duplicates = sorted(i for i, c in counts.items() if c > 1)
    expected = scorers.expected_ids[split]
    id_set = set(ids)
    checks = {
        "expected_n": entry["expected_n"],
        "expected_n_matches_manifest": entry["expected_n"] == len(expected),
        "scored_n": len(normalised),
        "duplicate_example_ids": len(duplicates),
        "missing_example_ids": len(expected - id_set),
        "unexpected_example_ids": len(id_set - expected),
        "manifest_equal": id_set == expected,
        "fold_rows_dropped_other_family": dropped_other_family,
        "fold_rows_outside_family": rows_outside_family,
        "task_name_mismatch_vs_truth": family_mismatch,
        "request_failed_rows": request_failed,
        "no_archived_answer_rows": no_archived,
        "reask_error_rows_warning": reask_error,
        "rows_with_think_tag": think_tag_rows,
        "reask_rows": sum(1 for r in normalised if r["reask"]),
    }
    if not checks["expected_n_matches_manifest"]:
        reasons.append("registry expected_n != manifest size")
    if len(normalised) != entry["expected_n"]:
        reasons.append("scored_n %d != expected_n %d" % (len(normalised), entry["expected_n"]))
    if duplicates:
        reasons.append("duplicate example_id: %d" % len(duplicates))
    if not checks["manifest_equal"]:
        reasons.append("example_id set != manifest (missing %d, unexpected %d)"
                       % (checks["missing_example_ids"], checks["unexpected_example_ids"]))
    if rows_outside_family:
        reasons.append("fold rows outside their fold family: %d" % rows_outside_family)
    if family_mismatch:
        reasons.append("row task_name != ground truth: %d" % family_mismatch)
    if request_failed or no_archived:
        reasons.append("rows without a model answer (REQUEST_FAILED %d, NO_ARCHIVED_ANSWER %d)"
                       % (request_failed, no_archived))

    # Replay equality: an ours/ablation cell must score exactly the archived answers.
    if entry.get("replay_source"):
        src = ROOT / entry["replay_source"]
        if src.is_file():
            archived = {int(r["index"]): (r.get("raw")[-3000:] if isinstance(r.get("raw"), str)
                                          else None) for r in read_jsonl(src)[0]}
            unequal = sum(1 for r in normalised
                          if archived.get(r["example_id"]) != (r["raw_output"] if r["raw_output"]
                                                               is not None else None))
            checks["replay_source_sha256"] = sha256_file(src)
            checks["replay_raw_unequal_rows"] = unequal
            if unequal:
                reasons.append("raw != replay source on %d rows" % unequal)
        else:
            reasons.append("replay source missing: %s" % entry["replay_source"])

    # The fold-assembled column must agree with the archived assembled file, if one exists.
    if entry.get("cross_check_file"):
        cc = ROOT / entry["cross_check_file"]
        if cc.is_file():
            other = {int(r["index"]): int(r.get("reason") == "SOLVED") for r in read_jsonl(cc)[0]}
            mine = {r["example_id"]: r["success"] for r in normalised}
            differing = sum(1 for i in set(mine) | set(other) if mine.get(i) != other.get(i))
            checks["cross_check_file_sha256"] = sha256_file(cc)
            checks["cross_check_rows_differing"] = differing
            if differing:
                reasons.append("fold assembly differs from %s on %d rows"
                               % (entry["cross_check_file"], differing))

    # Library files must exist for simulator-scored cells.
    libs = entry.get("library_path")
    lib_paths = list(libs.values()) if isinstance(libs, dict) else ([libs] if libs else [])
    checks["library_sha256"] = {p: sha256_file(ROOT / p) for p in lib_paths}
    if kind == "simulator_solved" and any(v is None for v in checks["library_sha256"].values()):
        reasons.append("library file missing")

    if entry.get("provenance_status") == "provenance_unverified":
        reasons.append("provenance_unverified: rows carry no library hash")

    run_basis = "|".join("%s:%s" % (f["path"], f["sha256"]) for f in facts["files"])
    run_id = "%s@%s" % (entry["cell_id"], sha256_text(run_basis)[:12])
    for r in normalised:
        r["run_id"] = run_id
    result.update({
        "rows": normalised, "checks": checks, "run_id": run_id,
        "success_count": sum(r["success"] for r in normalised),
        "parse_success_count": sum(r["parse_success"] for r in normalised),
    })
    return result


# ----------------------------------------------------------------------------- export
def seed_of(entry, result) -> Any:
    seeds = {(f.get("run_json") or {}).get("seed") for f in result.get("facts", {}).get("files", [])}
    seeds.discard(None)
    if len(seeds) == 1:
        return seeds.pop()
    if len(seeds) > 1:
        return ";".join(str(s) for s in sorted(seeds))
    return entry.get("seed")


def decoding_of(result) -> Dict[str, Any]:
    out = {}
    for f in result.get("facts", {}).get("files", []):
        rj = f.get("run_json") or {}
        for key in ("temperature", "max_tokens", "seed"):
            if key in rj:
                out.setdefault(key, set()).add(json.dumps(rj[key]))
    return {k: sorted(v) for k, v in out.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--init-registry", action="store_true")
    parser.add_argument("--force-registry", action="store_true")
    parser.add_argument("--check-anchors", type=Path, default=None,
                        help="JSON list of {experiment, condition, model_tag, canonical_split, "
                             "success_count, expected_n}; compared and PRINTED only")
    args = parser.parse_args(argv)
    os.chdir(str(ROOT))
    out = args.out.resolve()
    registry_path = (args.registry or out / "cells.json").resolve()

    if args.init_registry:
        if registry_path.exists() and not args.force_registry:
            print("registry exists, not overwriting: %s (use --force-registry)" % registry_path)
            return 1
        registry = build_registry()
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(registry, indent=1) + "\n")
        print("wrote %s (%d cells, %d audit cells)"
              % (registry_path, len(registry["cells"]), len(registry["audit_cells"])))
        return 0

    registry = json.loads(registry_path.read_text())
    scorers = Scorers(registry)
    started = datetime.now(timezone.utc).isoformat()

    results: Dict[str, Dict[str, Any]] = {}
    for entry in registry["cells"] + registry.get("audit_cells", []):
        if entry["experiment"] == "figure2" and entry["condition"] in AUDIT_ONLY_ARMS:
            raise SystemExit("Skill Memory v1 must not be a Figure 2 cell: %s" % entry["cell_id"])
        results[entry["cell_id"]] = evaluate_cell(entry, registry, scorers)
        print("scored %-48s %s" % (entry["cell_id"], "ok" if not results[entry["cell_id"]]
                                   ["incomplete_reasons"] else "incomplete"), flush=True)

    # ---- equality with baseline_comparison.json (baselines incl. v1, and ours)
    bc_path = ROOT / registry["baseline_comparison_path"]
    bc = json.loads(bc_path.read_text())
    bc_cells = {(c["model"], c["split"], c["arm"]): c for c in bc["cells"]}
    bc_check = {"path": registry["baseline_comparison_path"], "sha256": sha256_file(bc_path),
                "baseline_cells_checked": 0, "baseline_cells_equal": 0,
                "ours_cells_checked": 0, "ours_cells_equal": 0, "mismatches": []}
    seen_ours = set()
    for cell_id, res in results.items():
        e = res["entry"]
        if e["status"] != "available" or "success_count" not in res:
            continue
        key_split = e["source_split_name"]
        if e["scorer_kind"] == "json_tolerant" and e.get("baseline_comparison_arm"):
            key, bucket = (e["model_tag"], key_split, e["baseline_comparison_arm"]), "baseline"
        elif e["condition"] in ("ours", "full"):
            key, bucket = (e["model_tag"], key_split, OURS_BC_ARM), "ours"
        else:
            continue
        ref = bc_cells.get(key)
        mine = (res["success_count"], len(res["rows"]))
        equal = ref is not None and (ref["solved"], ref["n"]) == mine
        if bucket == "ours":
            if key in seen_ours:
                if not equal:
                    res["incomplete_reasons"].append("!= baseline_comparison.json %r" % (key,))
                continue
            seen_ours.add(key)
        bc_check["%s_cells_checked" % bucket] += 1
        bc_check["%s_cells_equal" % bucket] += int(equal)
        res["checks"]["baseline_comparison_equal"] = equal
        if not equal:
            bc_check["mismatches"].append({"key": list(key), "recomputed": list(mine),
                                           "reference": [ref["solved"], ref["n"]] if ref else None})
            res["incomplete_reasons"].append("!= baseline_comparison.json %r" % (key,))

    # ---- write rows
    out.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, List[Dict[str, Any]]] = {k: [] for k in EXPERIMENT_CONDITIONS}
    runs = []
    for entry in registry["cells"]:
        res = results[entry["cell_id"]]
        model = registry["models"][entry["model_tag"]]
        split_def = registry["splits"][entry["canonical_split"]]
        complete = not res["incomplete_reasons"]
        res["complete"] = complete
        row_file = row_sha = None
        if res["rows"]:
            path = (out / "rows" / entry["experiment"] / entry["condition"] / model["model_slug"]
                    / ("%s.jsonl" % entry["canonical_split"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as handle:
                for r in res["rows"]:
                    handle.write(json.dumps(r, default=repr, ensure_ascii=False) + "\n")
            row_file, row_sha = str(path.relative_to(out)), sha256_file(path)
        res["row_file"], res["row_file_sha256"] = row_file, row_sha
        scored_n = len(res["rows"])
        prompt_path = ROOT / entry["prompt_path"] if entry.get("prompt_path") else None
        summary = {
            "experiment": entry["experiment"],
            "model_display_name": model["model_display_name"],
            "model_id": model["model_id"],
            "condition": entry["condition"],
            "canonical_split": entry["canonical_split"],
            "source_split_name": entry["source_split_name"],
            "repeat": entry["repeat"],
            "seed": seed_of(entry, res),
            "expected_n": entry["expected_n"],
            "scored_n": scored_n if res["rows"] else "",
            "success_count": res["success_count"] if complete else "",
            "success_rate": ("%.4f" % (res["success_count"] / scored_n)) if complete else "",
            "parse_success_count": res["parse_success_count"] if complete else "",
            "parse_failure_count": (scored_n - res["parse_success_count"]) if complete else "",
            "complete": complete,
            "row_file": row_file or "",
            "row_file_sha256": row_sha or "",
            "prompt_sha256": (sha256_file(prompt_path) or "") if prompt_path else "",
            "scorer_sha256": scorers.scorer_sha.get(entry["scorer_kind"], ""),
            "manifest_sha256": sha256_file(ROOT / split_def["manifest_path"]) or "",
            "cell_id": entry["cell_id"],
            "task_display_name": model["task_display_name"],
            "status": entry["status"],
            "reuse": entry["reuse"],
            "provenance_status": entry["provenance_status"],
            "incomplete_reasons": "; ".join(res["incomplete_reasons"]),
        }
        summaries[entry["experiment"]].append(summary)
        runs.append({
            "run_id": res.get("run_id"), "cell_id": entry["cell_id"],
            "experiment": entry["experiment"], "condition": entry["condition"],
            "model_tag": entry["model_tag"], "model_display_name": model["model_display_name"],
            "canonical_split": entry["canonical_split"], "status": entry["status"],
            "reuse": entry["reuse"], "provenance_status": entry["provenance_status"],
            "started_at": None,
            "finished_at": max((f["mtime"] for f in res["facts"].get("files", [])), default=None),
            "finished_at_source": "latest source row-file mtime (reused archive)",
            "expected_n": entry["expected_n"], "produced_n": res.get("produced_n"),
            "scored_n": scored_n, "complete": complete,
            "incomplete_reasons": res["incomplete_reasons"],
            "exit_status": None, "endpoint_status": "not recorded in reused archive",
            "endpoint": model["endpoint"], "command": entry.get("command"),
            "config_path": entry.get("config_path"), "log_path": entry.get("log_path"),
            "scorer_kind": entry["scorer_kind"], "prompt_path": entry.get("prompt_path"),
            "library_path": entry.get("library_path"),
            "library_path_source": entry.get("library_path_source"),
            "replay_source": entry.get("replay_source"),
            "decoding_from_run_json": decoding_of(res),
            "source_files": [{k: f[k] for k in ("path", "sha256", "lines", "fold_family",
                                                "mtime", "run_json_path", "run_json")}
                             for f in res["facts"].get("files", [])],
            "checks": res["checks"], "row_file": row_file, "row_file_sha256": row_sha,
            "diagnostic_counts_not_headline": ({"success_count": res["success_count"],
                                                "parse_success_count": res["parse_success_count"],
                                                "scored_n": scored_n}
                                               if "success_count" in res else None),
        })

    # ---- paired tests
    paired = []
    by_key = {(e["experiment"], e["condition"], e["model_tag"], e["canonical_split"]): e
              for e in registry["cells"]}
    for experiment, pairs in PAIRS.items():
        for tag, model in registry["models"].items():
            for split in CANONICAL_SPLITS:
                for arm_a, arm_b in pairs:
                    ea, eb = by_key.get((experiment, arm_a, tag, split)), by_key.get(
                        (experiment, arm_b, tag, split))
                    row = {"experiment": experiment,
                           "model_display_name": model["model_display_name"],
                           "canonical_split": split, "arm_a": arm_a, "arm_b": arm_b}
                    ra = results.get(ea["cell_id"]) if ea else None
                    rb = results.get(eb["cell_id"]) if eb else None
                    ok = bool(ra and rb and ra.get("complete") and rb.get("complete"))
                    if ok:
                        a = {r["example_id"]: r["success"] for r in ra["rows"]}
                        b = {r["example_id"]: r["success"] for r in rb["rows"]}
                        shared = sorted(set(a) & set(b))
                        both = sum(1 for i in shared if a[i] and b[i])
                        a_only = sum(1 for i in shared if a[i] and not b[i])
                        b_only = sum(1 for i in shared if b[i] and not a[i])
                        neither = sum(1 for i in shared if not a[i] and not b[i])
                        n = len(shared)
                        expected_n = ea["expected_n"]
                        ok = (n == expected_n and both + a_only + b_only + neither == n
                              and set(a) == set(b) == scorers.expected_ids[split])
                    if ok:
                        rate_a, rate_b = sum(a[i] for i in shared) / n, sum(b[i] for i in shared) / n
                        row.update({"n_paired": n, "both_success": both,
                                    "arm_a_only_success": a_only, "arm_b_only_success": b_only,
                                    "both_failure": neither, "arm_a_rate": "%.4f" % rate_a,
                                    "arm_b_rate": "%.4f" % rate_b,
                                    "absolute_delta": "%.4f" % (rate_a - rate_b),
                                    "mcnemar_exact_p": "%.6g" % mcnemar_exact(a_only, b_only),
                                    "complete": True})
                    else:
                        row.update({k: "" for k in PAIRED_COLUMNS if k not in row})
                        row["complete"] = False
                    paired.append(row)
    scipy_check = None
    try:
        from scipy.stats import binomtest
        worst = 0.0
        for row in paired:
            if row["complete"]:
                k = min(row["arm_a_only_success"], row["arm_b_only_success"])
                n = row["arm_a_only_success"] + row["arm_b_only_success"]
                ref = binomtest(k, n, 0.5).pvalue if n else 1.0
                worst = max(worst, abs(ref - mcnemar_exact(row["arm_a_only_success"],
                                                           row["arm_b_only_success"])))
        scipy_check = {"max_abs_diff_vs_scipy_binomtest": worst}
    except Exception as error:  # noqa: BLE001
        scipy_check = {"error": repr(error)}

    # ---- write CSVs
    for experiment, rows in summaries.items():
        with open(out / SUMMARY_FILE[experiment], "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS + SUMMARY_EXTRA_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    with open(out / "paired_tests.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(paired)

    # ---- run manifest
    audit = [{"cell_id": e["cell_id"], "source": e["source"],
              "incomplete_reasons": results[e["cell_id"]]["incomplete_reasons"],
              "baseline_comparison_equal": results[e["cell_id"]]["checks"].get(
                  "baseline_comparison_equal")}
             for e in registry.get("audit_cells", [])]
    prompt_paths = sorted({e["prompt_path"] for e in registry["cells"] if e.get("prompt_path")})
    library_paths = sorted({p for e in registry["cells"] for p in (
        list(e["library_path"].values()) if isinstance(e.get("library_path"), dict)
        else [e["library_path"]] if e.get("library_path") else [])})
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "started_at": started,
        "exporter": {"path": rel(Path(__file__)), "sha256": sha256_file(Path(__file__))},
        "registry": {"path": rel(registry_path), "sha256": sha256_file(registry_path)},
        "git": git_info(), "environment": env_info(),
        "scorers": {kind: {"composite_sha256": scorers.scorer_sha[kind],
                           "components": scorers.scorer_components[kind]}
                    for kind in scorers.scorer_sha},
        "scorer_definitions": {
            "json_tolerant": "success = viki_report_matrix.tolerant(sim, response or raw, truth); "
                             "parse_success = viki_amendment9_diag102.parse_plan(text) is not None",
            "simulator_solved": "success = (row.reason == 'SOLVED') as written by "
                                "viki_eval_v2_intent_choice.py; parse_success = "
                                "viki_amendment11_goalparse.extract_json(raw) is a dict whose "
                                "'work' is a list",
        },
        "parser_paths": {
            "json_tolerant": {"path": "scripts/viki_amendment9_diag102.py",
                              "sha256": sha256_file(ROOT / "scripts/viki_amendment9_diag102.py")},
            "simulator_solved": {"path": "scripts/viki_amendment11_goalparse.py",
                                 "sha256": sha256_file(ROOT / "scripts/viki_amendment11_goalparse.py")},
        },
        "prompts": {p: {"sha256": sha256_file(ROOT / p), "exists": (ROOT / p).is_file()}
                    for p in prompt_paths},
        "ours_instruction_constant_sha256": python_constant_sha(
            ROOT / "scripts/viki_eval_v2_intent_choice.py", "INSTRUCTION"),
        "ours_reask_constant_sha256": python_constant_sha(
            ROOT / "scripts/viki_eval_v2_intent_choice.py", "REASK"),
        "model_configs": registry["models"],
        "libraries": {p: sha256_file(ROOT / p) for p in library_paths},
        "split_manifests": {s: {"manifest_path": d["manifest_path"],
                                "manifest_sha256": sha256_file(ROOT / d["manifest_path"]),
                                "truth_path": d["truth_path"],
                                "truth_sha256": sha256_file(Path(d["truth_path"]) if
                                                            Path(d["truth_path"]).is_absolute()
                                                            else ROOT / d["truth_path"]),
                                "expected_n": d["expected_n"]}
                            for s, d in registry["splits"].items()},
        "frozen_sweep": {"path": FROZEN_SWEEP_REL, "sha256": sha256_file(ROOT / FROZEN_SWEEP_REL)},
        "baseline_comparison_check": bc_check,
        "audit_only_cells": audit,
        "mcnemar_scipy_check": scipy_check,
        "runs": runs,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=repr) + "\n")

    # ---- markdown (computed values only)
    write_markdown(out, registry, summaries, paired, manifest)

    # ---- printing comes after everything is on disk
    total = len(registry["cells"])
    n_complete = sum(1 for e in registry["cells"] if results[e["cell_id"]].get("complete"))
    print("\ncells complete %d / %d; baseline_comparison equal: baseline %d/%d, ours %d/%d"
          % (n_complete, total, bc_check["baseline_cells_equal"], bc_check["baseline_cells_checked"],
             bc_check["ours_cells_equal"], bc_check["ours_cells_checked"]))
    print("wrote %s" % out)

    status = 0
    if args.check_anchors:
        anchors = json.loads(args.check_anchors.read_text())
        print("\nanchor check (printed only; not written to any output):")
        for a in anchors:
            key = "%s/%s/%s/%s" % (a["experiment"], a["condition"], a["model_tag"], a["canonical_split"])
            res = results.get(key)
            got = (res.get("success_count"), len(res["rows"])) if res and "success_count" in res else None
            want = (a["success_count"], a["expected_n"])
            ok = got == want
            status = status if ok else 3
            print("  %-4s %-44s recomputed %-12s anchor %-12s complete=%s"
                  % ("OK" if ok else "DIFF", key, "%s/%s" % got if got else "absent",
                     "%s/%s" % want, res.get("complete") if res else None))
    return status


def write_markdown(out: Path, registry, summaries, paired, manifest) -> None:
    models = registry["models"]
    lines = ["# VIKI-L2 supplementary results (ICLR 2027) -- generated", "",
             "Generated by `%s` from `%s`. Every number below is computed from row-level files; "
             "nothing is hand-entered." % (manifest["exporter"]["path"], manifest["registry"]["path"]),
             "",
             "- generated_at: %s" % manifest["generated_at"],
             "- git commit: `%s` (branch %s, dirty=%s)" % (manifest["git"]["commit"],
                                                          manifest["git"]["branch"],
                                                          manifest["git"]["dirty"]),
             "- scorer: VIKI-L2 JSON-tolerant for baselines (composite sha `%s`); simulator "
             "`reason == SOLVED` for Ours / ablations (composite sha `%s`)"
             % (manifest["scorers"]["json_tolerant"]["composite_sha256"][:16],
                manifest["scorers"]["simulator_solved"]["composite_sha256"][:16]),
             "- baseline_comparison.json equality: baseline %d/%d cells, ours %d/%d cells"
             % (manifest["baseline_comparison_check"]["baseline_cells_equal"],
                manifest["baseline_comparison_check"]["baseline_cells_checked"],
                manifest["baseline_comparison_check"]["ours_cells_equal"],
                manifest["baseline_comparison_check"]["ours_cells_checked"]),
             "", "## Model names", "",
             "| tag | model_display_name (actual) | task_display_name | served model_id | endpoint |",
             "|---|---|---|---|---|"]
    for tag, m in models.items():
        lines.append("| %s | %s | %s | %s | %s |" % (tag, m["model_display_name"],
                                                     m["task_display_name"], m["model_id"],
                                                     m["endpoint"]))
    header = "| model | condition | " + " | ".join(SPLIT_LABEL[s] for s in CANONICAL_SPLITS) + " |"
    rule = "|---|---|" + "---|" * len(CANONICAL_SPLITS)
    for experiment, rows in summaries.items():
        index = {(r["model_display_name"], r["condition"], r["canonical_split"]): r for r in rows}
        lines += ["", "## %s -- status and success rate" % experiment, "",
                  "Cell text: `rate (k/n)` when complete; otherwise the status. Incomplete cells "
                  "publish no aggregate.", "", header, rule]
        for tag, m in models.items():
            for condition in EXPERIMENT_CONDITIONS[experiment]:
                cells = []
                for split in CANONICAL_SPLITS:
                    r = index.get((m["model_display_name"], condition, split))
                    if r is None:
                        cells.append("not registered")
                    elif r["complete"]:
                        cells.append("%s (%s/%s)" % (r["success_rate"], r["success_count"],
                                                     r["scored_n"]))
                    elif r["status"] == "missing":
                        cells.append("MISSING")
                    elif r["provenance_status"] == "provenance_unverified":
                        cells.append("UNVERIFIED")
                    else:
                        cells.append("INCOMPLETE")
                lines.append("| %s | %s | %s |" % (m["model_display_name"],
                                                   METHOD_DISPLAY[condition], " | ".join(cells)))
        n_done = sum(1 for r in rows if r["complete"])
        lines += ["", "complete cells: %d / %d" % (n_done, len(rows))]
    lines += ["", "## Paired tests (McNemar exact)", "",
              "| experiment | model | split | A | B | n | both | A only | B only | neither | "
              "rate A | rate B | delta | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in paired:
        if r["complete"]:
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["experiment"], r["model_display_name"], SPLIT_LABEL[r["canonical_split"]],
                r["arm_a"], r["arm_b"], r["n_paired"], r["both_success"], r["arm_a_only_success"],
                r["arm_b_only_success"], r["both_failure"], r["arm_a_rate"], r["arm_b_rate"],
                r["absolute_delta"], r["mcnemar_exact_p"]))
    n_pairs_done = sum(1 for r in paired if r["complete"])
    lines += ["", "complete paired tests: %d / %d (incomplete pairs are omitted from the table "
              "and listed in paired_tests.csv with complete=false)" % (n_pairs_done, len(paired))]
    lines += ["", "## Incomplete and missing cells", ""]
    for run in manifest["runs"]:
        if not run["complete"]:
            lines.append("- `%s` [%s, %s]: %s" % (run["cell_id"], run["status"],
                                                   run["provenance_status"],
                                                   "; ".join(run["incomplete_reasons"])))
    if manifest["baseline_comparison_check"]["mismatches"]:
        lines += ["", "## baseline_comparison.json mismatches", ""]
        for m in manifest["baseline_comparison_check"]["mismatches"]:
            lines.append("- %s recomputed %s reference %s" % (m["key"], m["recomputed"], m["reference"]))
    (out / "paper_viki_results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
