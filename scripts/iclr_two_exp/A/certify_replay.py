#!/usr/bin/env python3
"""Certification of the replay-mined operators (SPEC A1), with the admitted library's own gate code.

Nothing here is new judging logic. Each step calls the function the admitted library went
through, on validation episodes from the induction half:

  execution check   `Workbench.run_operator` (scripts/viki_induction_tools.py): bind on an
                    episode the operator was not taken from, execute, does the effect hold.
                    Pass = holds on >= 2 of the family's 4 holdout episodes, the rung's rule
                    (scripts/viki_agentic_rung_abstraction.py, `len(works) >= 2`).
  (i) solves a validation episode / marginal
                    the rung's `episode_solved` with the ordering gate (goal reached with the
                    official scorer AND the ordering Layer 2 calls for is emitted), over the
                    family's coverage pool: adding the operator to the family's library must
                    turn at least one pool episode from unsolved to solved. As in the admitted
                    pipeline's first round (empty family library), the first operator of a
                    (family, effect key) that passes the execution check is admitted without it.
  (ii) necessary actions
                    no separate code path exists in the admitted pipeline; the ordering gate
                    inside `episode_solved` is what refuses a body that visits too little.
  (iii) recurs across traces
                    the assembler's rule (scripts/viki_assemble_agentic_library.py): support
                    re-measured by `run_operator` on induction-half episodes 0..59, >= 2.

Holdout and coverage pool per family are the ones the admitted library's second round used
(outputs/v3/targets.json), all induction-half indices. Layers 2/3 given to the Workbench are
the ones the rung and the assembler used (amendment11/skill_memory_v2.json).
Per-family admitted libraries are written in the assembler's output format so the column
memories can be built by the unchanged scripts/viki_union_library.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
TRAIN = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet"
BENCH = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R"
REFERENCE = ROOT / "results/viki_memory_experiments/amendment11/skill_memory_v2.json"
TARGETS = ROOT / "outputs/v3/targets.json"
PROBE, MIN_SUPPORT = 60, 2

_BENCH = None


def bench():
    global _BENCH
    if _BENCH is None:
        import viki_fork_guard
        viki_fork_guard.install()
        from viki_induction_tools import Workbench
        from our_method.skill_memory_v2.simulator import SEED
        ref = json.loads(REFERENCE.read_text())
        _BENCH = Workbench(TRAIN, BENCH, SEED, {"layer2": ref["layer2"], "layer3": ref["layer3"]})
    return _BENCH


def strip(op):
    return {k: v for k, v in op.items() if k not in ("episodes", "occurrences", "id")}


def run(op, j):
    try:
        return bench().run_operator(strip(op), j)
    except Exception as error:                                           # noqa: BLE001
        return {"bound": False, "effect_holds": False, "failure": "%s: %s" % (type(error).__name__, error)}


def exec_check(args):
    op, family, holdout = args
    detail = []
    for j in holdout:
        c = run(op, j)
        detail.append({"episode": j, "bound": bool(c.get("bound")), "effect_holds": bool(c.get("effect_holds")),
                       "failure": c.get("failure")})
    works = [d["episode"] for d in detail if d["bound"] and d["effect_holds"]]
    return op["id"], family, works, detail


def episode_solved(operators, j):
    b = bench()
    try:
        goal_ok = b.plan_with([strip(o) for o in operators], j)["official_score"] >= 1.0
    except Exception:                                                    # noqa: BLE001
        return False
    requires, emitted = b.ordering_ok([strip(o) for o in operators], j)
    return bool(goal_ok and (not requires or emitted))


def greedy_family(args):
    family, candidates, passed, pool = args
    library, log = [], []
    for op in candidates:
        if op["id"] not in passed:
            continue
        key = op["effect"]["key"]
        if not any(o["effect"]["key"] == key for o in library):
            library.append(op)
            log.append({"id": op["id"], "family": family, "marginal": "exempt (first executable operator of "
                        "this effect key in the family, as the empty-library first round)", "admitted": True})
            continue
        before = {j: episode_solved(library, j) for j in pool}
        gained = [j for j in pool if not before[j] and episode_solved(library + [op], j)]
        log.append({"id": op["id"], "family": family, "unsolved_before": [j for j in pool if not before[j]],
                    "newly_solved": gained, "admitted": bool(gained),
                    "marginal": "gain" if gained else "no_marginal_gain"})
        if gained:
            library.append(op)
    return family, [o["id"] for o in library], log


def support_probe(op):
    works = [j for j in range(PROBE) if (lambda c: c.get("bound") and c.get("effect_holds"))(run(op, j))]
    return op["id"], works


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mined = json.loads(args.mined.read_text())
    ops = {o["id"]: o for o in mined["operators"]}
    targets = {e["family"]: e for e in json.loads(TARGETS.read_text())["families"]}
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    jobs = [(op, f, targets[f]["holdout"]) for op in ops.values() for f in op["families"] if f in targets]
    no_target = sorted({f for op in ops.values() for f in op["families"] if f not in targets})
    with ProcessPoolExecutor(args.workers) as pool:
        execs = list(pool.map(exec_check, jobs))
    passed = {}
    for oid, fam, works, detail in execs:
        passed.setdefault(fam, {})[oid] = {"works_on": works, "detail": detail, "pass": len(works) >= 2}

    fam_jobs = []
    for fam, entry in targets.items():
        cands = sorted([o for o in ops.values() if fam in o["families"]], key=lambda o: (-o["support"], o["cost"]))
        ok = {oid for oid, r in passed.get(fam, {}).items() if r["pass"]}
        fam_jobs.append((fam, cands, ok, entry["coverage_pool"]))
    with ProcessPoolExecutor(min(args.workers, len(fam_jobs))) as pool:
        greedy = list(pool.map(greedy_family, fam_jobs))
    family_admitted = {fam: ids for fam, ids, _ in greedy}
    marginal_log = [row for _, _, log in greedy for row in log]

    candidates = sorted({oid for ids in family_admitted.values() for oid in ids})
    with ProcessPoolExecutor(args.workers) as pool:
        supports = dict(pool.map(support_probe, [ops[i] for i in candidates]))

    certified = {i for i in candidates if len(supports[i]) >= MIN_SUPPORT}
    per_op = []
    for oid, op in ops.items():
        fams = [f for f in op["families"] if f in targets]
        ex = {f: passed.get(f, {}).get(oid, {}) for f in fams}
        row = {"id": oid, "kind": op["kind"], "effect_key": op["effect"]["key"], "mined_support": op["support"],
               "families": op["families"],
               "exec_check": {f: {"works_on": r.get("works_on"), "pass": r.get("pass")} for f, r in ex.items()},
               "marginal": [m for m in marginal_log if m["id"] == oid],
               "admitted_in_families": [f for f, ids in family_admitted.items() if oid in ids],
               "probe_support": len(supports.get(oid, [])) if oid in supports else None,
               "certified": oid in certified}
        if oid in certified:
            row["reason"] = "pass"
        elif not any(r.get("pass") for r in ex.values()):
            failures = sorted({d.get("failure") or ("not bound" if not d["bound"] else "effect does not hold")
                               for r in ex.values() for d in r.get("detail", [])}, key=str)
            row["reason"] = "execution check: works on <2 of 4 holdout episodes in every source family (%s)" % \
                "; ".join(str(x)[:120] for x in failures[:3])
        elif not row["admitted_in_families"]:
            row["reason"] = "criterion (i): executable but adds no validation episode over the family library"
        else:
            row["reason"] = "criterion (iii): measured support %d < %d on induction-half probe episodes 0..%d" % (
                len(supports.get(oid, [])), MIN_SUPPORT, PROBE - 1)
        per_op.append(row)

    fam_dir = args.out_dir / "family_libraries"
    fam_dir.mkdir(exist_ok=True)
    for fam, ids in family_admitted.items():
        keep = [i for i in ids if i in certified]
        if not keep:
            continue
        lib_ops = []
        for i in keep:
            o = strip(ops[i])
            o["support"] = len(supports[i])
            o["provenance"] = dict(ops[i]["provenance"], mined_id=i, verified_on=supports[i][:20],
                                   certified_by="scripts/iclr_two_exp/A/certify_replay.py",
                                   certify_sha256=script_sha)
            lib_ops.append(o)
        (fam_dir / ("library_%s.json" % fam)).write_text(json.dumps(
            {"operators": lib_ops, "built_by": "certify_replay", "min_support": MIN_SUPPORT,
             "probe_episodes": PROBE}, indent=1))
    cert_ops = []
    for i in sorted(certified, key=lambda i: -len(supports[i])):
        o = strip(ops[i])
        o["support"] = len(supports[i])
        o["provenance"] = dict(ops[i]["provenance"], mined_id=i, certify_sha256=script_sha)
        cert_ops.append(o)
    summary = {"mined": len(ops), "certified": len(certified), "certified_ids": sorted(certified),
               "family_admitted_before_support": family_admitted,
               "families_without_targets": no_target, "certify_script_sha256": script_sha,
               "mined_library_sha256": hashlib.sha256(args.mined.read_bytes()).hexdigest(), "llm_calls": 0}
    (args.out_dir / "replay_library_certified.json").write_text(json.dumps(
        {"format": "replay_library/1", "summary": summary, "operators": cert_ops}, indent=1))
    (args.out_dir / "certification_per_operator.json").write_text(json.dumps(per_op, indent=1))
    (args.out_dir / "certification_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    for row in per_op:
        print(row["id"], row["kind"][:5], row["effect_key"], row["mined_support"], row["certified"], row["reason"][:150])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
