#!/usr/bin/env python3
"""RQ2 induction ledger: proposal -> admission -> dedup -> library, per condition.

Writes results/paper_viki_iclr2027/induction_ledger.json. Every number is computed here
from artefacts on disk -- the sweep definition, the round-two targets and driver defaults,
the rung source, the candidates files written by scripts/viki_rq2_no_exec_admission.py,
the per-family libraries and the unions. Nothing is typed in. The `consistency_check`
block compares against the task's recorded expectations and only reports; it never
replaces a computed value.

    /root/venvs/partnr/bin/python scripts/viki_rq2_induction_ledger.py

no_trace is written as a pending placeholder until its candidates file exists
(scripts/drivers/viki_rq2_no_trace.sh, then the extract command it prints); once it does,
the same function that fills `full` fills it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import viki_fork_guard  # noqa: E402

viki_fork_guard.install()

TRAIN = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet"
PAPER = ROOT / "results/paper_viki_iclr2027"
RQ2 = PAPER / "libraries/rq2"
DEFINITION = ROOT / "results/frozen_sweep_v2.json"
TARGETS = ROOT / "outputs/v3/targets.json"
ROUND2_DRIVER = ROOT / "scripts/drivers/viki_v3_round2.sh"
RUNG_SOURCE = ROOT / "scripts/viki_agentic_rung_abstraction.py"
NO_TRACE_DRIVER = ROOT / "scripts/drivers/viki_rq2_no_trace.sh"

CONDITIONS = {
    "full": {"candidates": RQ2 / "candidates_full_preadmission.jsonl",
             "family_libs": ROOT / "outputs/v3_libraries",
             "memory_dir": ROOT / "outputs/v3_memories", "no_trace": False},
    "no_execution_admission": {"candidates": RQ2 / "candidates_full_preadmission.jsonl",
                               "family_libs": RQ2 / "no_execution_admission/per_family",
                               "memory_dir": RQ2 / "no_execution_admission", "no_trace": False},
    "no_trace": {"candidates": RQ2 / "candidates_no_trace.jsonl",
                 "family_libs": ROOT / "outputs/rq2_notrace/libraries",
                 "memory_dir": RQ2 / "no_trace", "no_trace": True},
}


def sha256(path: Path) -> Optional[str]:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def shell_default(source: str, name: str) -> str:
    match = re.search(r'^%s=\$\{%s:-"?([^"}]*)"?\}' % (name, name), source, re.M)
    return match.group(1)


def proposal_budget() -> Dict[str, Any]:
    definition = json.loads(DEFINITION.read_text())
    targets = json.loads(TARGETS.read_text())
    driver = ROUND2_DRIVER.read_text()
    rung = RUNG_SOURCE.read_text()
    r1_cells = sum(len(entry["rungs"]) for entry in definition["libraries"] if entry["status"] == "ok")
    samples, moves2 = int(shell_default(driver, "SAMPLES")), int(shell_default(driver, "MOVES"))
    keys2 = shell_default(driver, "KEYS").split()
    r2_cells = sum(len(entry["seeds"]) for entry in targets["families"]) * len(keys2) * samples
    max_tokens = int(re.search(r'"--max-tokens", type=int, default=(\d+)', rung).group(1))
    return {
        "unit": "proposal cell = one (family, seed episode, sampling seed, effect key) rung run",
        "round1_cells": r1_cells, "round1_moves_per_cell": definition["moves_per_run"],
        "round1_temperature": definition["temperature"], "round1_source": rel(DEFINITION),
        "round2_cells": r2_cells, "round2_moves_per_cell": moves2,
        "round2_temperature": float(shell_default(driver, "TEMP")),
        "round2_source": "%s defaults x %s" % (rel(ROUND2_DRIVER), rel(TARGETS)),
        "cells_total": r1_cells + r2_cells,
        "max_model_calls_total": r1_cells * definition["moves_per_run"] + r2_cells * moves2,
        "max_completion_tokens_per_call": max_tokens,
    }


def cell_label(source: str) -> Dict[str, Any]:
    match = re.search(r"(outputs/agentic_rung/.+?/e(\d+)_[^/#]+)", source)
    return {"cell": match.group(1), "seed_episode": int(match.group(2))} if match else {"cell": source}


def load_ops(path: Path, layer1: bool) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    record = json.loads(path.read_text())
    return record["layer1"]["operators"] if layer1 else record["operators"]


def skill_records(condition: str, memory: Path, family_libs: Path, candidates: List[Dict[str, Any]],
                  status: str, support_kind: str) -> List[Dict[str, Any]]:
    import viki_union_library as union

    per_family = {p.stem[len("library_"):]: load_ops(p, False) for p in sorted(family_libs.glob("library_*.json"))}
    outcomes: Dict[str, Counter] = {}
    for c in candidates:
        if c.get("type_valid"):
            outcomes.setdefault(c["skill_id"], Counter())[c["full_rejection_reason"] or "admitted"] += 1
    records = []
    for op in load_ops(memory, True):
        signature = union.body_signature(op)
        sid = "sk_" + hashlib.sha256(signature.encode()).hexdigest()[:12]
        donors = [name[len("library_"):-len(".json")] for name in op.get("from_libraries") or []]
        sources, verified = [], {}
        for donor in donors:
            for family_op in per_family.get(donor, []):
                if union.body_signature(family_op) == signature:
                    provenance = family_op.get("provenance") or {}
                    sources += provenance.get("sources") or []
                    verified[donor] = provenance.get("verified_on") or []
        coordinated = bool(op.get("coordinated"))
        records.append({
            "skill_id": sid, "effect": op.get("effect"), "kind": op.get("kind"),
            "coordinated": coordinated,
            "body_verbs": ([a[0] for a in op.get("body", [])] if not coordinated else
                           [[i["action"][0] for i in r["actions"]] for r in op.get("roles", [])]),
            "body_length": (len(op.get("body", [])) if not coordinated
                            else sum(len(r["actions"]) for r in op.get("roles", []))),
            "role_count": len(op.get("roles", [])) if coordinated else 1,
            "donor_families": donors, "donor_family_count": len(donors),
            "model_written_families": op.get("families"),
            "episode_support": op.get("support"), "episode_support_kind": support_kind,
            "seed_episodes": sorted({json.dumps(cell_label(s), sort_keys=True) for s in sources}),
            "verification_examples": verified,
            "admission_status": status, "rejection_reason": None,
            "candidate_outcomes_in_full": dict(outcomes.get(sid, {})),
        })
    for record in records:
        record["seed_episodes"] = [json.loads(s) for s in record["seed_episodes"]]
    return records


def rejected_distinct(candidates: List[Dict[str, Any]], library_ids: set) -> List[Dict[str, Any]]:
    """Every distinct type-valid operator the condition's library does not contain."""
    groups: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        if not c.get("type_valid") or c["skill_id"] in library_ids:
            continue
        op = c["parsed_operator"]
        slot = groups.setdefault(c["skill_id"], {
            "skill_id": c["skill_id"], "effect": op.get("effect"),
            "coordinated": bool(op.get("coordinated")),
            "body_length": (len(op.get("body", [])) if not op.get("coordinated")
                            else sum(len(r["actions"]) for r in op.get("roles", []))),
            "donor_families": set(), "seed_episodes": set(), "candidates": 0,
            "admission_status": "rejected", "rejection_reason": Counter()})
        slot["donor_families"].add(c["family"])
        slot["seed_episodes"].add(c["seed_episode"])
        slot["candidates"] += 1
        slot["rejection_reason"][c["full_rejection_reason"]] += 1
    out = []
    for slot in groups.values():
        slot["donor_families"] = sorted(slot["donor_families"])
        slot["donor_family_count"] = len(slot["donor_families"])
        slot["seed_episodes"] = sorted(slot["seed_episodes"])
        slot["rejection_reason"] = dict(slot["rejection_reason"])
        out.append(slot)
    return sorted(out, key=lambda s: -s["candidates"])


def proposal_side(name: str, spec: Dict[str, Any], exposure: Dict[str, int]) -> Optional[Dict[str, Any]]:
    path = spec["candidates"]
    meta_path = path.with_suffix(".meta.json")
    if not path.is_file() or not meta_path.is_file():
        return None
    candidates = read_jsonl(path)
    meta = json.loads(meta_path.read_text())
    counts = meta["counts"]
    return {"candidates": candidates, "meta": meta, "fields": {
        "exposed_training_episode_count": 0 if spec["no_trace"] else exposure["episodes"],
        "exposed_trace_count": 0 if spec["no_trace"] else exposure["traces"],
        "family_count": len({c["family"] for c in meta["cells"]}),
        "proposal_attempt_count": counts["cells"],
        "proposal_attempt_count_by_round": counts["cells_by_round"],
        "model_call_count": counts["model_calls"], "tool_call_count": counts["tool_calls"],
        "returned_candidate_count": counts["returned_non_tool_replies"],
        "submission_count": counts["submissions"],
        "format_valid_count": counts["format_valid"], "type_valid_count": counts["type_valid"],
        "truncated_submission_count": counts["truncated_submissions"],
        "by_round": counts["by_round"],
        "candidates_file": rel(path), "candidates_sha256": sha256(path),
    }}


def library_side(spec: Dict[str, Any], definition: Dict[str, Any]) -> Dict[str, Any]:
    memory = spec["memory_dir"] / "memory_all.json"
    family_sizes = {p.stem[len("library_"):]: len(load_ops(p, False))
                    for p in sorted(spec["family_libs"].glob("library_*.json"))}
    return {
        "per_family_library_sizes": family_sizes,
        "per_family_distinct_count": sum(family_sizes.values()),
        "deduplicated_skill_count": len(load_ops(memory, True)) if memory.is_file() else None,
        "library_path": rel(memory), "library_sha256": sha256(memory),
        "fold_libraries": {held: {"path": rel(spec["memory_dir"] / ("memory_heldout_%s.json" % held)),
                                  "sha256": sha256(spec["memory_dir"] / ("memory_heldout_%s.json" % held)),
                                  "size": len(load_ops(spec["memory_dir"] / ("memory_heldout_%s.json" % held), True))
                                  if (spec["memory_dir"] / ("memory_heldout_%s.json" % held)).is_file() else None}
                           for held in definition["fold_families"]},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=PAPER / "induction_ledger.json")
    args = parser.parse_args(argv)

    from our_method.skill_memory_v2.build import load_episodes
    import viki_agentic_rung_abstraction as rung

    induction = load_episodes(TRAIN)[::2]
    exposure = {"episodes": len(induction),
                "traces": sum(1 for e in induction if isinstance(e, dict) and e.get("time_steps"))}
    definition = json.loads(DEFINITION.read_text())
    budget = proposal_budget()
    notrace_tools = re.findall(r'\{"tool": "(\w+)"', rung.TOOLS_NO_TRACE)
    leak_note = ("list_episodes(family=null) lists the whole induction half, and show_trace / "
                 "contrast_actors / check_actor accept any index, so a family cell can read other "
                 "families' induction episodes (observed in transcripts, e.g. "
                 "v2_set_plate_and_fork_on_table/e112_pos_name reads cut_fruit_on_board index 6). "
                 "Evaluation episodes are never reachable: the workbench holds episodes[::2] of "
                 "train.parquet only.")
    ledger: Dict[str, Any] = {"generated_at": datetime.now().isoformat(), "git_commit": None,
                              "inputs": {rel(p): sha256(p) for p in (
                                  DEFINITION, TARGETS, ROUND2_DRIVER, RUNG_SOURCE,
                                  ROOT / "scripts/viki_assemble_agentic_library.py",
                                  ROOT / "scripts/viki_union_library.py",
                                  ROOT / "scripts/viki_rq2_no_exec_admission.py", NO_TRACE_DRIVER)},
                              "conditions": {}}
    try:
        ledger["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                              capture_output=True, text=True).stdout.strip() or None
    except Exception:                                                # noqa: BLE001
        pass

    full_prop = proposal_side("full", CONDITIONS["full"], exposure)
    if full_prop is None:
        print("未执行，缺 %s (run scripts/viki_rq2_no_exec_admission.py extract)" % rel(CONDITIONS["full"]["candidates"]))
        return 1
    candidates = full_prop["candidates"]
    counts = full_prop["meta"]["counts"]

    # ---------------------------------------------------------------- full
    full_lib = library_side(CONDITIONS["full"], definition)
    full_skills = skill_records("full", CONDITIONS["full"]["memory_dir"] / "memory_all.json",
                                CONDITIONS["full"]["family_libs"], candidates, "admitted",
                                "measured: induction-half episodes (of the first 60 in the union "
                                "pool) where the operator binds and its effect holds")
    ledger["conditions"]["full"] = dict(
        condition="full", status="complete", no_trace=False,
        exposure_notes=[leak_note], proposal_budget=budget, **full_prop["fields"],
        execution_tested_count=counts["execution_tested"],
        execution_passed_count=counts["rung_passed"],
        execution_rejection_reasons=counts["full_rejection_reason"],
        **full_lib, skills=full_skills,
        rejected_distinct_operators=rejected_distinct(candidates, {s["skill_id"] for s in full_skills}))

    # ---------------------------------------------------------------- no_execution_admission
    arm_spec = CONDITIONS["no_execution_admission"]
    arm_lib = library_side(arm_spec, definition)
    manifest_path = arm_spec["memory_dir"] / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    arm_skills = skill_records("no_execution_admission", arm_spec["memory_dir"] / "memory_all.json",
                               arm_spec["family_libs"], candidates, "admitted_without_execution",
                               "proposal_cells: distinct full-condition proposal cells that submitted "
                               "this operator (no execution)")
    full_ids = {s["skill_id"] for s in full_skills}
    for skill in arm_skills:
        skill["in_full_library"] = skill["skill_id"] in full_ids
        skill["verification_examples"] = {}
    ledger["conditions"]["no_execution_admission"] = dict(
        condition="no_execution_admission",
        status="complete" if arm_lib["library_sha256"] else "pending -- run the build stage",
        no_trace=False, exposure_notes=[leak_note, "proposal records are the full condition's own"],
        proposal_budget=budget, **full_prop["fields"],
        execution_tested_count=0, execution_passed_count=None,
        execution_admission="skipped", support_reprobe="skipped",
        preadmission_candidate_count=counts["type_valid"],
        rejected_by_full_entered_candidate_count=manifest.get("rejected_by_full_entered_candidate_count"),
        operators_not_in_full_count=manifest.get("operators_not_in_full_count"),
        identical_to_full=manifest.get("identical_to_full"),
        **arm_lib, skills=arm_skills, rejected_distinct_operators=[])

    # ---------------------------------------------------------------- no_trace
    nt_spec = CONDITIONS["no_trace"]
    nt_prop = proposal_side("no_trace", nt_spec, exposure)
    stale = sorted((ROOT / "outputs/agentic_rung/v2_notrace").glob("*/verdict.json"))
    stale_ref = {"path": "outputs/agentic_rung/v2_notrace", "cells": len(stale),
                 "passed": sum(str(json.loads(p.read_text()).get("passed")) == "True" for p in stale),
                 "definition": "results/frozen_sweep.json",
                 "why_stale": "56 cells of a different flat sweep, not the 896-cell budget of full"}
    planned_feedback_episodes = sorted(
        {i for entry in definition["libraries"] for r in entry["rungs"] for i in r["holdout"]}
        | {i for entry in json.loads(TARGETS.read_text())["families"] for i in entry["coverage_pool"]})
    no_trace = {"condition": "no_trace", "no_trace": True, "proposal_budget": budget,
                "no_trace_tools_exposed": notrace_tools,
                "exposure_notes": [
                    "--no-traces removes every tool (TOOLS_NO_TRACE lists %d), so no trace or "
                    "episode listing is reachable" % len(notrace_tools),
                    "a SUBMISSION is still executed on the cell's holdout and the result "
                    "(episodes_it_works_on, bound/effect_holds/failure/bindings) is returned to "
                    "the model; bindings name objects of those holdout episodes. Planned "
                    "holdout+coverage episodes across the budget: %d" % len(planned_feedback_episodes)],
                "stale_reference": stale_ref, "driver": rel(NO_TRACE_DRIVER)}
    if nt_prop is None:
        no_trace.update({"status": "pending -- scripts/drivers/viki_rq2_no_trace.sh not run",
                         **{k: None for k in (
                             "exposed_training_episode_count", "exposed_trace_count", "family_count",
                             "proposal_attempt_count", "returned_candidate_count", "format_valid_count",
                             "type_valid_count", "execution_tested_count", "execution_passed_count",
                             "deduplicated_skill_count", "library_sha256")},
                         "skills": []})
    else:
        nt_candidates, nt_counts = nt_prop["candidates"], nt_prop["meta"]["counts"]
        nt_lib = library_side(nt_spec, definition)
        nt_skills = skill_records("no_trace", nt_spec["memory_dir"] / "memory_all.json",
                                  nt_spec["family_libs"], nt_candidates, "admitted",
                                  "measured (same assembler/union re-probe as full)")
        complete = nt_counts["cells"] == budget["cells_total"]
        no_trace.update(dict(
            status="complete" if complete and nt_lib["library_sha256"] else
                   "incomplete: %d of %d cells" % (nt_counts["cells"], budget["cells_total"]),
            **nt_prop["fields"], execution_tested_count=nt_counts["execution_tested"],
            execution_passed_count=nt_counts["rung_passed"],
            execution_rejection_reasons=nt_counts["full_rejection_reason"], **nt_lib, skills=nt_skills,
            rejected_distinct_operators=rejected_distinct(nt_candidates, {s["skill_id"] for s in nt_skills})))
    ledger["conditions"]["no_trace"] = no_trace

    # ---------------------------------------------------------------- expectations (report only)
    expected = {"exposed_trace_count": 3598, "proposal_opportunities": 896,
                "execution_passed_count": 23, "deduplicated_skill_count": 8,
                "stale_no_trace_admitted": 0}
    observed = {"exposed_trace_count": ledger["conditions"]["full"]["exposed_trace_count"],
                "proposal_opportunities": ledger["conditions"]["full"]["proposal_attempt_count"],
                "execution_passed_count": ledger["conditions"]["full"]["execution_passed_count"],
                "deduplicated_skill_count": ledger["conditions"]["full"]["deduplicated_skill_count"],
                "stale_no_trace_admitted": stale_ref["passed"]}
    ledger["consistency_check"] = {
        "source": "TASK-viki-iclr2027-2026-09-14.md section 4 (expectations, not results)",
        "expected": expected, "observed": observed,
        "matches": {k: expected[k] == observed[k] for k in expected},
        "full_budget_matches_attempts": budget["cells_total"] == observed["proposal_opportunities"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ledger, indent=1, ensure_ascii=False, default=str))
    print("wrote %s  sha256 %s" % (rel(args.out), sha256(args.out)))
    for name, c in ledger["conditions"].items():
        print("%-24s %s" % (name, {k: c.get(k) for k in (
            "status", "proposal_attempt_count", "returned_candidate_count", "format_valid_count",
            "type_valid_count", "execution_tested_count", "execution_passed_count",
            "deduplicated_skill_count")}))
    print("consistency", ledger["consistency_check"]["matches"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
