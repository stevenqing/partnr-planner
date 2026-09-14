#!/usr/bin/env python3
"""RQ2 condition C, `no_execution_admission`: the full condition's own proposals, unvalidated.

Zero model calls, no GPU, and nothing existing is modified. Three stages:

  extract  Every reply the FULL condition's proposer returned that was not a tool call --
           round one (`outputs/agentic_rung/v2_<family>/*`, 784 cells) and round two
           (`outputs/agentic_rung/v3/<family>/*`, 112 cells) -- read out of each cell's
           transcript into one structured candidates file. No resampling.

           Two transcript facts the extraction has to handle, both from the rung itself
           (scripts/viki_agentic_rung_abstraction.py):
             * a refused submission is appended to the transcript TWICE (once after the
               execution check, once at the end of the move), so records are keyed by move;
               counting records double-counts submissions;
             * `answer` is stored truncated (`answer[:1500]`). A submission the live run
               parsed but whose stored text no longer parses is kept, marked
               `raw_answer_truncated_in_transcript`, and is format-invalid here, because the
               operator it carried is not recoverable from any artefact. A PASSED submission
               is taken from `verdict.json`, which stores the operator whole.

           Checks kept (static, no episode is touched):
             format_valid  the reply parses to a submission object with the rung's own
                           `extract_request` + `normalise_request`;
             type_valid    the predicate/verb menu the proposer states (`NO_TRACE_NOTE`),
                           the effect key the cell asked for (the rung refuses any other key
                           WITHOUT running it), the operator shape the planner indexes
                           (body / roles with offset and after, dict preconditions/types,
                           numeric cost), and `SkillMemoryV2._usable` -- the loader's own
                           test that every action argument is a variable token.
           Skipped: the rung's execution test (binds and achieves on >= 2 holdout
           episodes), its marginal-coverage/ordering gain, and the assembler's support
           re-probe with its minimum of 2.

  build    Per-family assembly and union through the SAME scripts full used, with the
           opt-in flags `--candidates/--no-support-probe` (assembler) and
           `--no-support-probe/--excluded-family` (union). ID/CG use the 14-family union; each
           of the 8 single-family folds is a union of the OTHER 13 families' per-family
           libraries, whose candidates come only from those families' own proposal cells, with
           Layers 2/3 re-mined on the fold's pool -- exactly the main OOD construction. The
           14-family union is never filtered afterwards.

  check    Loads every built memory the way the Ours evaluator does
           (`SkillMemoryV2.load` in scripts/viki_eval_v2_intent_choice.py), reports fields
           the evaluator/planner index that an operator lacks, renders the menu, and runs a
           crash-only planner smoke on training induction-half episodes (outcome never used
           to select anything).

    /root/venvs/partnr/bin/python scripts/viki_rq2_no_exec_admission.py all
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import viki_fork_guard  # noqa: E402

viki_fork_guard.install()

PY = "/root/venvs/partnr/bin/python" if Path("/root/venvs/partnr/bin/python").is_file() else sys.executable
TRAIN = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet"
BENCHMARK = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R"
RUNG_ROOT = ROOT / "outputs/agentic_rung"
DEFINITION = ROOT / "results/frozen_sweep_v2.json"
FULL_LIBS = ROOT / "outputs/v3_libraries"
FULL_MEMS = ROOT / "outputs/v3_memories"
RQ2 = ROOT / "results/paper_viki_iclr2027/libraries/rq2"
CANDIDATES = RQ2 / "candidates_full_preadmission.jsonl"
ARM = RQ2 / "no_execution_admission"
FULL_COPY = RQ2 / "full"
ASSEMBLE = ROOT / "scripts/viki_assemble_agentic_library.py"
UNION = ROOT / "scripts/viki_union_library.py"
EVALUATOR = ROOT / "scripts/viki_eval_v2_intent_choice.py"
PLANNER_VARIABLE = re.compile(r"^\?(x|y|z\d+|r\d+)$")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def skill_id(operator: Dict[str, Any]) -> str:
    import viki_union_library as union
    return "sk_" + hashlib.sha256(union.body_signature(operator).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- the menu
def proposer_menu(rung) -> Dict[str, Any]:
    """The predicate keys and primitive verbs, read from the proposer's own prompt text."""
    note = rung.NO_TRACE_NOTE
    predicates, verbs = note.split("Predicate vocabulary of the training episodes:")[1].split(
        "Primitive verbs available to a body:")
    keys = [line.split()[0] for line in predicates.strip().splitlines() if line.strip()]
    verbs = [v.strip() for v in verbs.strip().split(",") if v.strip()]
    source = inspect.getsource(rung)
    truncation = int(re.search(r'"answer": answer\[:(\d+)\]', source).group(1))
    tools = re.findall(r'\{"tool": "(\w+)"', rung.TOOLS)
    return {"effect_keys": keys, "verbs": verbs, "answer_truncation": truncation,
            "tool_names": tools, "source": "scripts/viki_agentic_rung_abstraction.py:NO_TRACE_NOTE"}


def type_problems(operator: Any, target_key: Optional[str], menu: Dict[str, Any]) -> List[str]:
    from our_method.skill_memory_v2.memory import SkillMemoryV2

    problems: List[str] = []
    if not isinstance(operator, dict):
        return ["operator is not a JSON object"]
    effect = operator.get("effect")
    if not isinstance(effect, dict):
        problems.append("effect missing or not an object")
    else:
        key = effect.get("key")
        if key not in menu["effect_keys"]:
            problems.append("effect key %r not in the predicate menu" % (key,))
        if target_key is not None and key != target_key:
            problems.append("effect key %r is not the cell's target key %r" % (key, target_key))

    def action(item: Any, where: str) -> None:
        if not isinstance(item, list) or len(item) < 2:
            problems.append("%s: action is not [Verb, arg, ...]" % where)
            return
        if item[0] not in menu["verbs"]:
            problems.append("%s: verb %r not in the primitive verb menu" % (where, item[0]))
        if not all(isinstance(t, str) and t.startswith("?") and not t.startswith("?agent")
                   for t in item[1:]):
            problems.append("%s: an argument is not a variable token" % where)

    if operator.get("coordinated"):
        roles = operator.get("roles")
        if not isinstance(roles, list) or not roles:
            problems.append("coordinated operator without roles")
        else:
            for r, role in enumerate(roles):
                items = role.get("actions") if isinstance(role, dict) else None
                if not isinstance(items, list) or not items:
                    problems.append("role %d has no actions" % r)
                    continue
                for a, item in enumerate(items):
                    if not isinstance(item, dict):
                        problems.append("role %d action %d is not an object" % (r, a))
                        continue
                    action(item.get("action"), "role %d action %d" % (r, a))
                    if not isinstance(item.get("offset"), int) or isinstance(item.get("offset"), bool):
                        problems.append("role %d action %d: offset is not an integer" % (r, a))
                    after = item.get("after")
                    if not isinstance(after, list) or not all(
                            isinstance(p, list) and len(p) == 2 and all(isinstance(v, int) for v in p)
                            for p in after):
                        problems.append("role %d action %d: after is not [[role, count], ...]" % (r, a))
    else:
        body = operator.get("body")
        if not isinstance(body, list) or not body:
            problems.append("body missing or empty")
        else:
            for a, item in enumerate(body):
                action(item, "body action %d" % a)
    if "preconditions" in operator and not isinstance(operator["preconditions"], dict):
        problems.append("preconditions is not an object")
    if "types" in operator and not (isinstance(operator["types"], dict) and
                                    all(isinstance(v, dict) for v in operator["types"].values())):
        problems.append("types is not an object of objects")
    if "cost" in operator and (not isinstance(operator["cost"], (int, float)) or isinstance(operator["cost"], bool)):
        problems.append("cost is not a number")
    if not problems:
        try:
            if not SkillMemoryV2._usable(operator):
                problems.append("SkillMemoryV2._usable refuses it")
        except Exception as error:                                   # noqa: BLE001
            problems.append("SkillMemoryV2._usable raised %s" % type(error).__name__)
    return problems


def tokens_of(operator: Dict[str, Any]) -> List[str]:
    if operator.get("coordinated"):
        return [t for role in operator.get("roles", []) for item in role.get("actions", [])
                for t in (item.get("action") or [])[1:]]
    return [t for item in operator.get("body", []) for t in item[1:]]


# --------------------------------------------------------------------------- extract
def enumerate_cells(prefix: str, keys: List[str]) -> List[Dict[str, Any]]:
    slug_to_key = {k.replace(".", "_"): k for k in keys}
    cells = []
    for verdict in sorted(RUNG_ROOT.glob(prefix + "v2_*/*/verdict.json")):
        label = verdict.parent.parent.name
        if label == "v2_notrace":
            continue
        match = re.match(r"^e(\d+)_(.+)$", verdict.parent.name)
        cells.append({"round": 1, "family": label[len("v2_"):], "cell_id": rel(verdict.parent),
                      "seed_episode": int(match.group(1)), "sample": None,
                      "target_key": slug_to_key[match.group(2)]})
    for verdict in sorted(RUNG_ROOT.glob(prefix + "v3/*/*/verdict.json")):
        match = re.match(r"^e(\d+)_s(\d+)_(.+)$", verdict.parent.name)
        cells.append({"round": 2, "family": verdict.parent.parent.name, "cell_id": rel(verdict.parent),
                      "seed_episode": int(match.group(1)), "sample": int(match.group(2)),
                      "target_key": slug_to_key[match.group(3)]})
    return cells


def extract(args) -> int:
    import viki_agentic_rung_abstraction as rung
    import viki_assemble_agentic_library as assemble
    import viki_union_library as union

    menu = proposer_menu(rung)
    definition = json.loads(DEFINITION.read_text())
    families = definition["build_families"]
    family_signatures = {}
    for family in families:
        path = args.family_libs / ("library_%s.json" % family)
        ops = json.loads(path.read_text())["operators"] if path.is_file() else []
        family_signatures[family] = {repr(assemble.signature(op)) for op in ops}
    union_ops = (json.loads(args.union_memory.read_text())["layer1"]["operators"]
                 if args.union_memory.is_file() else [])
    union_signatures = {union.body_signature(op) for op in union_ops}

    rows, cell_rows = [], []
    for cell in enumerate_cells(args.rung_prefix, menu["effect_keys"]):
        directory = ROOT / cell["cell_id"]
        verdict = json.loads((directory / "verdict.json").read_text())
        transcript_path = directory / "transcript.json"
        transcript = json.loads(transcript_path.read_text()) if transcript_path.is_file() else None
        seen, duplicates, tool_moves = set(), 0, 0
        passed = str(verdict.get("passed")) == "True"
        for record in transcript or []:
            move = record.get("move")
            if move in seen:
                duplicates += 1
                continue
            seen.add(move)
            if record.get("tool") in menu["tool_names"]:
                tool_moves += 1
                continue
            result = record.get("result") if isinstance(record.get("result"), dict) else {}
            submitted = bool(result.get("submitted"))
            answer = record.get("answer") or ""
            request, _ = rung.extract_request(answer)
            request = rung.normalise_request(request)
            parsed = (request["submit"] if isinstance(request, dict)
                      and isinstance(request.get("submit"), dict) else None)
            passed_here = passed and submitted and move == verdict.get("moves_used")
            operator, operator_source, agrees = parsed, "transcript_answer", None
            if passed_here and isinstance(verdict.get("operator"), dict):
                agrees = parsed == verdict["operator"]
                operator, operator_source = verdict["operator"], "verdict_operator"
            truncated = submitted and parsed is None and len(answer) >= menu["answer_truncation"]
            format_valid = submitted and isinstance(operator, dict)
            problems = (type_problems(operator, cell["target_key"], menu) if format_valid
                        else ["answer text in transcript is truncated; operator unrecoverable"
                              if truncated else "not a parseable submission"])
            type_valid = format_valid and not problems

            if not submitted:
                kind = "tool_call_refused" if "tool" in record else "unparsed_reply"
                execution = None
            elif result.get("refused"):
                kind = "submission"
                execution = {"tested": False, "refused_before_execution": result.get("why")}
            else:
                kind = "submission"
                execution = {"tested": True,
                             "episodes_it_works_on": result.get("episodes_it_works_on"),
                             "detail": result.get("detail"), "note": result.get("note"),
                             "interface": result.get("interface"),
                             "episodes_newly_solved": result.get("episodes_newly_solved"),
                             "passed_rung": passed_here}

            if not submitted:
                status, reason = "not_a_submission", (result.get("error") or "not a submission")
            elif result.get("refused"):
                status, reason = "rejected", "rung_refused_target_key"
            elif passed_here:
                try:
                    in_family = repr(assemble.signature(operator)) in family_signatures.get(cell["family"], set())
                except Exception:                                    # noqa: BLE001
                    in_family = False
                if not in_family:
                    status, reason = "rejected", "family_assembly_below_min_support"
                elif union.body_signature(operator) not in union_signatures:
                    status, reason = "rejected", "union_below_min_support"
                else:
                    status, reason = "admitted", None
            elif "episodes_newly_solved" in result:
                status, reason = "rejected", "rung_no_marginal_coverage_or_ordering_gain"
            else:
                status, reason = "rejected", "rung_execution_works_on_fewer_than_2_holdout"

            in_library = None
            if type_valid:
                in_library = union.body_signature(operator) in union_signatures
            rows.append({
                "candidate_id": "%s#m%d" % (cell["cell_id"], move),
                "cell_id": cell["cell_id"], "family": cell["family"], "round": cell["round"],
                "turn": move, "seed_episode": cell["seed_episode"], "sample": cell["sample"],
                "target_key": cell["target_key"], "kind": kind,
                "raw_answer": answer, "raw_answer_truncated_in_transcript": truncated,
                "live_parse_ok": submitted, "parsed_operator": operator,
                "operator_source": operator_source, "transcript_matches_verdict": agrees,
                "format_valid": format_valid, "type_valid": type_valid,
                "type_problems": problems,
                "uses_planner_variables_only": (all(PLANNER_VARIABLE.match(t) for t in tokens_of(operator))
                                                if type_valid else None),
                "skill_id": skill_id(operator) if type_valid else None,
                "execution_outcome": execution,
                "full_status": status, "full_admitted": status == "admitted",
                "full_rejection_reason": reason,
                "signature_in_full_library": in_library,
            })
        cell_rows.append(dict(cell, passed=passed, moves_used=verdict.get("moves_used"),
                              transcript_present=transcript is not None,
                              transcript_records=len(transcript or []),
                              duplicate_records=duplicates, tool_moves=tool_moves))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    counts = {
        "cells": len(cell_rows),
        "cells_by_round": dict(Counter(c["round"] for c in cell_rows)),
        "model_calls": sum(c["moves_used"] or 0 for c in cell_rows),
        "tool_calls": sum(c["tool_moves"] for c in cell_rows),
        "duplicate_transcript_records_skipped": sum(c["duplicate_records"] for c in cell_rows),
        "returned_non_tool_replies": len(rows),
        "submissions": sum(r["kind"] == "submission" for r in rows),
        "format_valid": sum(r["format_valid"] for r in rows),
        "type_valid": sum(r["type_valid"] for r in rows),
        "truncated_submissions": sum(r["raw_answer_truncated_in_transcript"] for r in rows),
        "execution_tested": sum(bool(r["execution_outcome"] and r["execution_outcome"].get("tested")) for r in rows),
        "rung_passed": sum(bool(r["execution_outcome"] and r["execution_outcome"].get("passed_rung")) for r in rows),
        "full_admitted_candidates": sum(r["full_admitted"] for r in rows),
        "full_status": dict(Counter(r["full_status"] for r in rows)),
        "full_rejection_reason": dict(Counter(r["full_rejection_reason"] for r in rows)),
        "type_problem_first": dict(Counter(r["type_problems"][0] for r in rows if r["type_problems"])),
        "by_round": {str(k): {"submissions": sum(r["kind"] == "submission" and r["round"] == k for r in rows),
                              "format_valid": sum(r["format_valid"] and r["round"] == k for r in rows),
                              "type_valid": sum(r["type_valid"] and r["round"] == k for r in rows)}
                     for k in sorted({c["round"] for c in cell_rows})},
    }
    meta = {"candidates_file": rel(args.out), "candidates_sha256": sha256(args.out),
            "rung_prefix": args.rung_prefix, "menu": menu,
            "family_libraries": rel(args.family_libs), "union_memory": rel(args.union_memory),
            "union_memory_sha256": sha256(args.union_memory) if args.union_memory.is_file() else None,
            "definition": rel(DEFINITION), "definition_sha256": sha256(DEFINITION),
            "counts": counts, "cells": cell_rows}
    write_json(args.out.with_suffix(".meta.json"), meta)
    print(json.dumps(counts, indent=1))
    print("wrote %s  sha256 %s" % (rel(args.out), meta["candidates_sha256"]))
    return 0


# --------------------------------------------------------------------------- build
def build(args) -> int:
    import viki_union_library as union

    if not CANDIDATES.is_file():
        print("未执行，缺 %s (run the extract stage)" % rel(CANDIDATES))
        return 1
    definition = json.loads(DEFINITION.read_text())
    families, folds = definition["build_families"], definition["fold_families"]
    per_family = ARM / "per_family"
    per_family.mkdir(parents=True, exist_ok=True)
    log: List[Dict[str, Any]] = []

    for family in families:
        out = per_family / ("library_%s.json" % family)
        if out.is_file():
            log.append({"family": family, "skipped": "exists"})
            continue
        command = [PY, str(ASSEMBLE), "--candidates", str(CANDIDATES), "--family", family,
                   "--no-support-probe", "--out", str(out),
                   "--report", str(per_family / ("assembly_%s.json" % family))]
        proc = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, timeout=3600)
        (per_family / ("assemble_%s.log" % family)).write_text(proc.stdout + proc.stderr)
        log.append({"family": family, "command": command[1:], "returncode": proc.returncode,
                    "built": out.is_file()})

    def union_to(out: Path, members: List[str], excluded: Optional[str]) -> Dict[str, Any]:
        present = [per_family / ("library_%s.json" % f) for f in members
                   if (per_family / ("library_%s.json" % f)).is_file()]
        entry = {"out": rel(out), "families": members, "excluded_family": excluded,
                 "libraries": [p.name for p in present]}
        if out.is_file():
            entry["skipped"] = "exists"
            return entry
        command = [PY, str(UNION), "--libraries", *[str(p) for p in present],
                   "--families", *members, "--out", str(out), "--no-support-probe"]
        if excluded:
            command += ["--excluded-family", excluded]
        proc = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, timeout=10800)
        out.with_suffix(".log").write_text(proc.stdout + proc.stderr)
        entry.update({"command": command[1:], "returncode": proc.returncode})
        return entry

    log.append(union_to(ARM / "memory_all.json", families, None))
    for held in folds:
        log.append(union_to(ARM / ("memory_heldout_%s.json" % held),
                            [f for f in families if f != held], held))

    # The full condition's libraries, copied byte for byte beside the new arm.
    FULL_COPY.mkdir(parents=True, exist_ok=True)
    full_sources = [FULL_MEMS / "memory_all.json"] + [
        FULL_MEMS / ("memory_heldout_%s.json" % held) for held in folds]
    full_sources += sorted(FULL_LIBS.glob("library_*.json"))
    copies = {}
    for source in full_sources:
        target = FULL_COPY / (("per_family/" if source.parent == FULL_LIBS else "") + source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            shutil.copyfile(source, target)
        copies[rel(target)] = {"source": rel(source), "sha256": sha256(target),
                               "matches_source": sha256(target) == sha256(source)}

    def ops(path: Path) -> List[Dict[str, Any]]:
        return json.loads(path.read_text())["layer1"]["operators"] if path.is_file() else []

    def describe(op: Dict[str, Any]) -> Dict[str, Any]:
        return {"skill_id": skill_id(op), "effect": op.get("effect"),
                "coordinated": bool(op.get("coordinated")),
                "verbs": ([a[0] for a in op.get("body", [])] if not op.get("coordinated")
                          else [[i["action"][0] for i in r["actions"]] for r in op.get("roles", [])]),
                "from_libraries": op.get("from_libraries"), "support": op.get("support")}

    def diff(full_path: Path, arm_path: Path) -> Dict[str, Any]:
        full_ops, arm_ops = ops(full_path), ops(arm_path)
        full_sig = {union.body_signature(o): o for o in full_ops}
        arm_sig = {union.body_signature(o): o for o in arm_ops}
        return {"full_path": rel(full_path), "full_sha256": sha256(full_path) if full_path.is_file() else None,
                "arm_path": rel(arm_path), "arm_sha256": sha256(arm_path) if arm_path.is_file() else None,
                "full_size": len(full_ops), "arm_size": len(arm_ops),
                "in_both": len(set(full_sig) & set(arm_sig)),
                "only_full": [describe(full_sig[s]) for s in full_sig if s not in arm_sig],
                "only_arm_count": len([s for s in arm_sig if s not in full_sig]),
                "identical_operator_sets": set(full_sig) == set(arm_sig)}

    union_diff = diff(FULL_MEMS / "memory_all.json", ARM / "memory_all.json")
    fold_diffs = {held: diff(FULL_MEMS / ("memory_heldout_%s.json" % held),
                             ARM / ("memory_heldout_%s.json" % held)) for held in folds}

    candidates = [json.loads(line) for line in CANDIDATES.read_text().splitlines() if line.strip()]
    entered = [c for c in candidates if c["type_valid"] and not c["full_admitted"]]
    with (ARM / "rejected_by_full_entered_candidates.jsonl").open("w") as handle:
        for c in entered:
            handle.write(json.dumps({k: c[k] for k in (
                "candidate_id", "family", "round", "turn", "target_key", "skill_id",
                "full_status", "full_rejection_reason", "signature_in_full_library",
                "uses_planner_variables_only", "parsed_operator")}, ensure_ascii=False) + "\n")
    arm_union = ops(ARM / "memory_all.json")
    full_signatures = {union.body_signature(o) for o in ops(FULL_MEMS / "memory_all.json")}
    reasons_by_skill: Dict[str, Counter] = {}
    for c in candidates:
        if c["type_valid"]:
            reasons_by_skill.setdefault(c["skill_id"], Counter())[c["full_rejection_reason"] or "admitted"] += 1
    not_in_full = [dict(describe(o), full_candidate_outcomes=dict(reasons_by_skill.get(skill_id(o), {})),
                        proposal_cells=len((o.get("provenance") or {}).get("proposal_cells") or []),
                        operator=o)
                   for o in arm_union if union.body_signature(o) not in full_signatures]
    write_json(ARM / "rejected_by_full_entered_operators.json",
               {"description": "operators in the no_execution_admission 14-family union whose "
                               "signature is not in the full v3 library",
                "count": len(not_in_full), "operators": not_in_full})

    outputs = {}
    for path in sorted(ARM.rglob("*.json")) + sorted(ARM.rglob("*.jsonl")):
        if path.name == "build_manifest.json":
            continue
        outputs[rel(path)] = sha256(path)
    per_family_sizes = {}
    for family in families:
        path = per_family / ("library_%s.json" % family)
        per_family_sizes[family] = len(json.loads(path.read_text())["operators"]) if path.is_file() else 0
    manifest = {
        "condition": "no_execution_admission",
        "config": {
            "proposal_records": "full condition, rounds 1+2, no resampling",
            "checks_kept": ["format: rung extract_request + normalise_request",
                            "type: NO_TRACE_NOTE predicate/verb menu, cell target key, operator "
                            "shape, SkillMemoryV2._usable"],
            "skipped": ["rung execution test (>=2 holdout episodes)",
                        "rung marginal coverage/ordering gain",
                        "assembler support re-probe and min_support",
                        "union support re-probe and min_support"],
            "dedup": "viki_assemble_agentic_library.signature per family, then "
                     "viki_union_library.body_signature across families (unchanged code)",
            "support_field": "distinct proposal cells that submitted the operator (no execution)",
            "folds": "union of the other 13 families' per-family libraries, Layers 2/3 re-mined "
                     "on the 13-family pool (same as viki_v2_evaluate.py)",
        },
        "inputs": {"candidates": rel(CANDIDATES), "candidates_sha256": sha256(CANDIDATES),
                   "definition": rel(DEFINITION), "definition_sha256": sha256(DEFINITION)},
        "per_family_sizes": per_family_sizes,
        "union": {"path": rel(ARM / "memory_all.json"),
                  "sha256": sha256(ARM / "memory_all.json") if (ARM / "memory_all.json").is_file() else None,
                  "size": len(arm_union)},
        "folds": {held: {"path": d["arm_path"], "sha256": d["arm_sha256"], "size": d["arm_size"]}
                  for held, d in fold_diffs.items()},
        "diff_vs_full": {"union": union_diff, "folds": fold_diffs},
        "identical_to_full": union_diff["identical_operator_sets"],
        "rejected_by_full_entered_candidate_count": len(entered),
        "operators_not_in_full_count": len(not_in_full),
        "full_copies": copies,
        "outputs_sha256": outputs,
        "build_log": log,
    }
    write_json(ARM / "build_manifest.json", manifest)
    print("per-family sizes  %s" % per_family_sizes)
    print("union             %d operators (full %d), sha256 %s"
          % (len(arm_union), union_diff["full_size"], manifest["union"]["sha256"]))
    for held, d in fold_diffs.items():
        print("fold %-48s %d (full %d)" % (held, d["arm_size"], d["full_size"]))
    print("full-rejected candidates entering the arm: %d; operators not in full: %d"
          % (len(entered), len(not_in_full)))
    if union_diff["identical_operator_sets"]:
        print("WARNING: the unvalidated union has the same operator set as full -- the bypass "
              "must be investigated before this arm is used")
        return 2
    return 0


# --------------------------------------------------------------------------- check
def check(args) -> int:
    from our_method.skill_memory_v2 import SEED, SkillMemoryV2, Simulator, planner
    from our_method.skill_memory_v2.build import load_episodes

    report: Dict[str, Any] = {"evaluator": rel(EVALUATOR), "evaluator_sha256": sha256(EVALUATOR),
                              "model_calls": 0, "memories": {}}
    try:
        import viki_eval_v2_intent_choice as evaluator  # noqa: F401
        report["evaluator_import"] = "ok"
        report["evaluator_loader"] = "SkillMemoryV2.load (viki_eval_v2_intent_choice.py main)"
    except Exception as error:                                       # noqa: BLE001
        report["evaluator_import"] = "%s: %s" % (type(error).__name__, error)

    definition = json.loads(DEFINITION.read_text())
    targets = [("no_execution_admission/memory_all", ARM / "memory_all.json")]
    targets += [("no_execution_admission/memory_heldout_%s" % h, ARM / ("memory_heldout_%s.json" % h))
                for h in definition["fold_families"]]
    targets += [("full/memory_all", FULL_MEMS / "memory_all.json")]
    loaded = {}
    for name, path in targets:
        if not path.is_file():
            report["memories"][name] = {"missing": rel(path)}
            continue
        record = json.loads(path.read_text())
        entry: Dict[str, Any] = {"path": rel(path), "sha256": sha256(path)}
        try:
            memory = SkillMemoryV2.load(path)
        except Exception as error:                                   # noqa: BLE001
            entry["load_error"] = "%s: %s" % (type(error).__name__, error)
            report["memories"][name] = entry
            continue
        operators = record["layer1"]["operators"]
        gaps = []
        for i, op in enumerate(operators):
            for field in ("effect", "preconditions", "cost", "support"):
                if field not in op:
                    gaps.append({"operator": i, "missing": field})
            if op.get("coordinated"):
                for r, role in enumerate(op.get("roles", [])):
                    for a, item in enumerate(role.get("actions", [])):
                        for field in ("action", "offset", "after"):
                            if field not in item:
                                gaps.append({"operator": i, "role": r, "action": a, "missing": field})
        menu_text, catalogue = memory.menu()
        rendered = memory.render()
        entry.update({"format": record.get("format"), "excluded_family": record.get("excluded_family"),
                      "layer1_operators": len(operators), "usable_after_load": len(memory.operators),
                      "dropped_by_loader": len(operators) - len(memory.operators),
                      "field_gaps": gaps, "menu_entries": len(catalogue), "render_chars": len(rendered),
                      "layer2_kept_patterns": len(memory.rules),
                      "layer3_places": len(memory.vocab.get("places", [])),
                      "operators_without_verified_on": sum(
                          1 for op in operators if not (op.get("provenance") or {}).get("verified_on"))})
        report["memories"][name] = entry
        loaded[name] = memory

    if args.smoke:
        from viki_eval_skill_memory_v2 import visits_of
        sim = Simulator(Path(BENCHMARK))
        episodes = [e for e in load_episodes(TRAIN)[::2] if isinstance(e, dict) and e.get("time_steps")]
        sample = episodes[:args.smoke]
        for name in ("no_execution_admission/memory_all", "full/memory_all"):
            memory = loaded.get(name)
            if memory is None:
                continue
            errors: Counter = Counter()
            planned = 0
            for truth in sample:
                blind = {k: v for k, v in truth.items() if k != "time_steps"}
                try:
                    metadata = sim.metadata(blind, SEED)
                    env = sim.world(metadata)
                    requirements = [r["predicate"] for r in planner.collect_requirements(metadata)]
                    memory.order_for(requirements, visits_of(env, requirements, memory))
                    steps, _ = planner.plan(blind, memory, sim, SEED)
                    planned += int(bool(steps))
                except Exception as error:                           # noqa: BLE001
                    errors["%s: %s" % (type(error).__name__, str(error)[:80])] += 1
            report["memories"][name]["planner_smoke"] = {
                "episodes": len(sample), "raised": sum(errors.values()), "exceptions": dict(errors),
                "returned_a_plan": planned,
                "note": "crash check only, on the first training induction-half episodes; "
                        "nothing is selected on it"}

    write_json(ARM / "evaluator_dry_load.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "memories"}, indent=1))
    for name, entry in report["memories"].items():
        print("%-72s ops=%s usable=%s gaps=%d smoke=%s" % (
            name, entry.get("layer1_operators"), entry.get("usable_after_load"),
            len(entry.get("field_gaps", [])), (entry.get("planner_smoke") or {}).get("raised")))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["extract", "build", "check", "all"])
    parser.add_argument("--rung-prefix", default="",
                        help="rung subdirectory of the condition, e.g. 'rq2_notrace/'")
    parser.add_argument("--family-libs", type=Path, default=FULL_LIBS,
                        help="the condition's own per-family libraries (for full_status)")
    parser.add_argument("--union-memory", type=Path, default=FULL_MEMS / "memory_all.json")
    parser.add_argument("--out", type=Path, default=CANDIDATES)
    parser.add_argument("--smoke", type=int, default=40,
                        help="training episodes for the crash-only planner smoke (0 = off)")
    args = parser.parse_args(argv)
    if args.stage in ("extract", "all"):
        status = extract(args)
        if status:
            return status
    if args.stage in ("build", "all"):
        status = build(args)
        if status == 1:
            return status
        if args.stage == "all" and status == 2:
            check(args)
            return 2
    if args.stage in ("check", "all"):
        return check(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
