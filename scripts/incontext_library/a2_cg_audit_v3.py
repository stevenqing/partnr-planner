#!/usr/bin/env python3
"""A2 cg-audit: re-run the 2026-09-05 CG leakage audit against the v3 library.

Zero LLM calls, read-only on every existing artefact. Writes only under
results/incontext_library_2026-09-21/work/A2/.

The Figure-2 CG cells (cg_image / pure_text, condition ours) score with
outputs/v3_memories/memory_all.json. Its inputs, traced from
scripts/drivers/viki_v3_build_and_score.sh -> viki_assemble_agentic_library.py ->
drivers/viki_v2_evaluate.py -> viki_union_library.py:

  R1  round one  outputs/agentic_rung/v2_<family>/e<seed>_<key>/           (frozen_sweep_v2.json)
  R2  round two  outputs/agentic_rung/v3/<family>/e<seed>_s<k>_<key>/      (outputs/v3/targets.json)
      both via scripts/viki_agentic_rung_abstraction.py: seed episode, holdout, coverage pool
      (the "CANNOT solve episodes [...]" list is a subset of holdout|coverage), tool calls.
      Its Workbench borrows layer2/3 from amendment11/skill_memory_v2.json.
  assembly     support probe = workbench indices range(60)                  (per-family libs)
  union        support probe = pool_indices[:60]; Layer 2/3 re-mined on the union pool
               (induction half restricted to the 14 union families)

Every workbench index is a position in train[::2] (Workbench.episodes), global = 2*i.
Logic (L0 comp loading, L1 detectors, transcript index extraction, plan trigrams) is imported
from scripts/viki_comp_leakage_audit.py unchanged.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import viki_comp_leakage_audit as audit  # noqa: E402

OUT = ROOT / "results/incontext_library_2026-09-21/work/A2"
MEMORY = ROOT / "outputs/v3_memories/memory_all.json"
LIBS = ROOT / "outputs/v3_libraries"
RUNG = ROOT / "outputs/agentic_rung"
SWEEP_V2 = ROOT / "results/frozen_sweep_v2.json"
TARGETS_V3 = ROOT / "outputs/v3/targets.json"
REFERENCE = ROOT / "results/viki_memory_experiments/amendment11/skill_memory_v2.json"
CELLS = ROOT / "results/paper_viki_iclr2027/cells.json"
OLD = ROOT / "audit/comp_leakage_2026-09-05"
CELL_RE = re.compile(r"^e(\d+)_(?:s(\d+)_)?(pos_name|is_activated)$")
REQUEST_TOOLS = audit.WORKBENCH_TOOLS_WITH_EPISODE


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_indices(raw) -> set:
    """Indices that appear inside a tool RESULT (repr of a dict; e.g. list_episodes pages).

    The 09-05 scan only caught these when the result was JSON; the transcripts store
    Python reprs, so its count was request-side only. Reported separately, as a superset.
    """
    if raw is None:
        return set()
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except Exception:                                            # noqa: BLE001
            return {int(m.group(1)) for m in re.finditer(
                r"""['"](?:%s)['"]\s*:\s*(\d+)""" % "|".join(audit.INDEX_KEYS), raw)}
    return audit._indices_in(raw)


def scan(families):
    runs = []
    dirs = []
    for family in families:
        for base, rnd in ((RUNG / ("v2_%s" % family), "R1"), (RUNG / "v3" / family, "R2")):
            if base.is_dir():
                dirs += [(d, rnd, family) for d in sorted(base.iterdir()) if d.is_dir()]
    for directory, rnd, family in dirs:
        m = CELL_RE.match(directory.name)
        entry = {"round": rnd, "family": family, "run": directory.name,
                 "seed_episode": int(m.group(1)) if m else None,
                 "verdict": (directory / "verdict.json").is_file(),
                 "transcript": (directory / "transcript.json").is_file(),
                 "request_indices": [], "result_indices": [], "tools": []}
        text = ""
        if entry["transcript"]:
            text = (directory / "transcript.json").read_text()
            try:
                moves = json.loads(text)
            except Exception:                                        # noqa: BLE001
                moves = []
            req, res = set(), set()
            for move in moves if isinstance(moves, list) else []:
                if not isinstance(move, dict):
                    continue
                if move.get("tool"):
                    entry["tools"].append(move["tool"])
                req |= audit._indices_in(move.get("answer"))   # identical to 09-05 rule
                res |= result_indices(move.get("result"))
            entry["request_indices"] = sorted(req)
            entry["result_indices"] = sorted(res)
        entry["_text"] = text
        runs.append(entry)
    return runs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from our_method.skill_memory_v2.simulator import SEED, Simulator
    benchmark = ROOT.parent / "VIKI-R"
    train = benchmark / "data/VIKI-R/viki/VIKI-L2/train.parquet"
    sim = Simulator(benchmark)

    # ---- L0 / L1 recomputed with the audit's own functions (train-only + CG parquets)
    l0l1 = audit.Out(OUT / "l0_l1")
    args = Namespace(test=None, probe=60,
                     manifest=ROOT / "results/viki_memory_experiments/amendment8b/interactive_manifest.jsonl")
    l0 = audit.build_l0(l0l1, sim, SEED, train, args)
    l1 = audit.build_l1(l0l1, sim, SEED, l0["episodes"], l0["comp"])
    episodes = l0["episodes"]
    half = len(range(0, len(episodes), 2))
    comp_ids = {r["task_id"] for r in l0["comp"]}
    flagged = l1["flagged_indices"]
    old_l1 = json.loads((OLD / "L1_train_with_heldout_sig.json").read_text())
    blind_now = json.loads((OUT / "l0_l1/L1_train_with_heldout_sig.json").read_text())
    train_ids_all = {str(t.get("task_id")) for t in episodes if isinstance(t, dict)}

    def assess(name, wb):
        wb = sorted(set(int(i) for i in wb))
        inside = [i for i in wb if 0 <= i < half]
        outside = [i for i in wb if not (0 <= i < half)]
        hits = [{"workbench_index": i, "train_index": 2 * i,
                 "task_id": str(episodes[2 * i].get("task_id"))}
                for i in inside if str((episodes[2 * i] or {}).get("task_id")) in comp_ids]
        sig = [i for i in inside if 2 * i in flagged]
        return {"pool": name, "n_indices": len(wb),
                "intersect_cg_task_ids": len(hits), "cg_hits": hits,
                "intersect_train_with_heldout_sig_decisive": len(sig),
                "indices_outside_induction_half": len(outside), "outside": outside[:50],
                "families": dict(Counter(episodes[2 * i].get("task_name") for i in inside))}

    memory = json.loads(MEMORY.read_text())
    families = memory["union_families"]
    sweep = json.loads(SWEEP_V2.read_text())
    libs_v2 = sweep["libraries"] if isinstance(sweep["libraries"], list) else json.loads(sweep["libraries"])
    targets = json.loads(TARGETS_V3.read_text())["families"]

    r1_seed, r1_hold, r1_cov = [], [], []
    for entry in libs_v2:
        if entry["family"] in families:
            for rung in entry["rungs"]:
                r1_seed.append(rung["seed_episode"])
                r1_hold += rung.get("holdout") or []
                r1_cov += rung.get("coverage_pool") or []
    r2_seed, r2_hold, r2_cov = [], [], []
    for entry in targets:
        if entry["family"] in families:
            r2_seed += entry["seeds"]
            r2_hold += entry["holdout"]
            r2_cov += entry["coverage_pool"]

    runs = scan(families)
    req_all = sorted({i for r in runs for i in r["request_indices"]})
    res_all = sorted({i for r in runs for i in r["result_indices"]})
    per_tool = defaultdict(set)
    for r in runs:
        for i in r["request_indices"]:
            for tool in r["tools"]:
                if tool in REQUEST_TOOLS:
                    per_tool[tool].add(i)
    dir_seeds = sorted({r["seed_episode"] for r in runs if r["seed_episode"] is not None})

    # only the runs whose family library actually exists (dog_push_box admitted nothing)
    contributing = {p.stem[len("library_"):] for p in LIBS.glob("library_*.json")}
    req_contrib = sorted({i for r in runs if r["family"] in contributing
                          for i in r["request_indices"]})

    # accepted-operator provenance, per-family libraries and union
    prov = set()
    for path in LIBS.glob("library_*.json"):
        for op in json.loads(path.read_text())["operators"]:
            prov |= set((op.get("provenance") or {}).get("verified_on") or [])
    for op in memory["layer1"]["operators"]:
        prov |= set((op.get("provenance") or {}).get("verified_on") or [])
    # assembly support probe and union support probe
    induction = episodes[::2]
    pool_idx = [i for i, e in enumerate(induction)
                if isinstance(e, dict) and e.get("task_name") in set(families)]

    reference = json.loads(REFERENCE.read_text())
    table = [
        assess("R1 rung seed episodes (frozen_sweep_v2)", r1_seed),
        assess("R1 holdout", r1_hold),
        assess("R1 coverage_pool", r1_cov),
        assess("R2 rung seed episodes (outputs/v3/targets.json)", r2_seed),
        assess("R2 holdout", r2_hold),
        assess("R2 coverage_pool (superset of the unsolved list shown)", r2_cov),
        assess("seed episodes from run-dir names", dir_seeds),
        assess("all tool-addressed episodes (request side, 09-05 rule)", req_all),
        assess("tool-addressed, contributing families only", req_contrib),
        assess("indices shown inside tool results (e.g. list_episodes pages)", res_all),
    ]
    for tool in ("run_operator", "try_bind", "check_actor", "contrast_actors", "show_trace"):
        table.append(assess("tool:%s" % tool, per_tool.get(tool, set())))
    table += [
        assess("assembly support probe range(60)", range(60)),
        assess("union support probe pool_indices[:60]", pool_idx[:60]),
        assess("accepted-operator provenance verified_on", prov),
        assess("Layer 2/3 mining pool (induction half & 14 union families)", pool_idx),
    ]

    # ---- transcript text: exact CG task_id and plan trigram containment (09-05 L3 rules)
    exact, tri = [], []
    comp_plans = {t: audit._trigrams(audit._plan_tokens(r["truth"]))
                  for t, r in l1["comp_by_task"].items()}
    for r in runs:
        text = r.pop("_text")
        if not text:
            continue
        for t in comp_ids:
            if t and t in text:
                exact.append({"run": "%s/%s" % (r["family"], r["run"]), "task_id": t})
        text_tri = audit._trigrams(re.findall(r"[A-Za-z_][A-Za-z0-9_ ]*", text))
        for t, plan in comp_plans.items():
            if plan and len(plan & text_tri) / len(plan) >= 0.9:
                tri.append({"run": "%s/%s" % (r["family"], r["run"]), "task_id": t})

    total_hits = sum(row["intersect_cg_task_ids"] for row in table)
    total_sig = sum(row["intersect_train_with_heldout_sig_decisive"] for row in table)
    total_out = sum(row["indices_outside_induction_half"] for row in table)
    cg_in_train = sorted(comp_ids & train_ids_all)
    verdict = ("PASS" if total_hits == 0 and not exact and total_out == 0 else "HALTED")

    cells = json.loads(CELLS.read_text())["cells"]
    cg_libs = sorted({c["library_path"] for c in cells if c.get("experiment") == "figure2"
                      and c.get("condition") == "ours"
                      and c.get("canonical_split") in ("cg_image", "pure_text")})
    summary = {
        "task": "A2 cg-audit (v3 library)", "llm_calls": 0, "verdict": verdict,
        "figure2_cg_library_paths": cg_libs,
        "library": {"path": str(MEMORY.relative_to(ROOT)), "sha256": sha(MEMORY),
                    "built_from": memory["built_from"], "union_families": families,
                    "families_with_library": sorted(contributing),
                    "families_without_library": sorted(set(families) - contributing),
                    "layer2_3": "re-mined in place by scripts/viki_union_library.py on the union pool "
                                "(%d episodes); NOT borrowed" % memory["union_report"]["pool_episodes"],
                    "workbench_reference_layers": {"path": str(REFERENCE.relative_to(ROOT)),
                                                   "built_from": reference.get("built_from"),
                                                   "used_by": "viki_agentic_rung_abstraction.py and "
                                                              "viki_assemble_agentic_library.py Workbench"}},
        "appendix_numbers": {
            "tool_addressed_train_indices": {"old": 180, "new": len(req_all),
                                             "contributing_families_only": len(req_contrib),
                                             "including_result_side": len(set(req_all) | set(res_all))},
            "transcripts_run_dirs": {"old": 884, "new_run_dirs": len(runs),
                                     "new_transcripts": sum(r["transcript"] for r in runs),
                                     "new_verdicts": sum(r["verdict"] for r in runs),
                                     "by_round": dict(Counter(r["round"] for r in runs))},
            "actor_blind_sensitivity_train_rows": {
                "old": 1033, "new": l1["report"]["train_with_heldout_sig_actor_blind"],
                "families": l1["report"]["selfcheck"]["sensitivity_families"],
                "indices_identical_to_0905": sorted(old_l1["sensitivity_indices"])
                == sorted(blind_now["sensitivity_indices"])},
            "decisive_detector_train_rows": {"old": 0, "new": len(flagged)},
            "cg_distinct_task_ids": {"old": 295,
                                     "new": l0["report"]["comp_pairing"]["distinct_task_ids_overall"],
                                     "rows_per_form": l0["report"]["comp_pairing"]["rows_per_form"],
                                     "repeated": l0["report"]["comp_pairing"]["text_ids_repeated"]},
        },
        "l1_selfcheck": l1["report"]["selfcheck"],
        "train_rows": len(episodes), "induction_half": half,
        "cg_task_ids_present_anywhere_in_train_7196": len(cg_in_train),
        "pools": table,
        "totals": {"cg_task_id_hits": total_hits, "decisive_sig_hits": total_sig,
                   "indices_outside_induction_half": total_out,
                   "exact_task_id_text_matches": len(exact), "trigram_ge_0.9": len(tri)},
        "text_examples": {"exact": exact[:20], "trigram": tri[:20]},
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p)
                          for p in (MEMORY, SWEEP_V2, TARGETS_V3, REFERENCE)},
    }
    summary["inputs_sha256"]["train.parquet"] = sha(train)
    (OUT / "a2_runs.json").write_text(json.dumps(runs, indent=1))
    (OUT / "a2_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: summary[k] for k in ("verdict", "appendix_numbers", "totals")},
                     indent=1, default=str))
    for row in table:
        print("%-66s n=%-5d cg=%d sig=%d out=%d" % (row["pool"], row["n_indices"],
              row["intersect_cg_task_ids"], row["intersect_train_with_heldout_sig_decisive"],
              row["indices_outside_induction_half"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
