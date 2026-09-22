#!/usr/bin/env python3
"""A1 / A2 of the 2026-09-22 master spec: first-turn-only and reference-library replays, zero LLM calls.

Every row of a published VIKI-L2 cell of ours carries the model's first answer (`raw`) and, when
the fallback fired, the answer to the one re-ask (`raw_reask`). The memory is consulted only after
an answer arrives, so both can be re-planned against any library with no generation.

Modes, per (model, split):
  ours_full   our library, first answer then the archived re-ask, exactly as the live runner
              (`viki_eval_v2_intent_choice.py`) does it. Must reproduce the published cell.
  ours_first  our library, first answer only.
  ref_first   the 19-operator reference library (per-fold variant on the single-family split),
              first answer only. The archived re-ask was written in response to OUR library's
              infeasibility message, so it is not a same-protocol input for another library.

Cells are the ones the main table reads: v3_ours_<M>_{id,heldout,heldout_sibgrp},
v3_oursall_<M>_{text,imaged}.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

A11 = ROOT / "results/viki_memory_experiments/amendment11"
A10 = ROOT / "results/viki_memory_experiments/amendment10"
MEM = ROOT / "outputs/v3_memories"
PRE = ROOT / "results/sibling_folds_preregistration.json"
REF = A11 / "skill_memory_v2.json"

CELL = {"id": "v3_ours_%s_id", "heldout": "v3_ours_%s_heldout",
        "heldout_sibgrp": "v3_ours_%s_heldout_sibgrp",
        "text": "v3_oursall_%s_text", "imaged": "v3_oursall_%s_imaged"}
PARQUET = {"text": A10 / "recombination.text.parquet", "imaged": A10 / "recombination.imaged.parquet"}


def library_path(kind: str, split: str, family: str, affected) -> Path | None:
    if kind == "ours":
        if split in ("id", "text", "imaged"):
            return MEM / "memory_all.json"
        if split == "heldout" or family not in affected:
            return MEM / ("memory_heldout_%s.json" % family)
        return MEM / ("memory_heldoutgrp_%s.json" % family)
    if split in ("id", "text", "imaged"):
        return REF
    if split == "heldout" or family not in affected:
        return A11 / ("skill_memory_v2.fold_%s.json" % family)
    # sibling group: built by scripts/iclr_master/A/a2_build_ref_sibgrp.py (same builder, group exclusion)
    grp = REF_GRP.get(family)
    return grp if grp is not None and grp.is_file() else None


REF_GRP = {f: ROOT / "results/iclr_master_2026-09-22/A/ref_sibgrp" / name for f, name in (
    ("cut_fruit_on_board", "skill_memory_v2.foldgrp_cut.json"),
    ("cut_two_fruits_on_board", "skill_memory_v2.foldgrp_cut.json"),
    ("parallel_human_dual_asset_to_plate_or_bowl", "skill_memory_v2.foldgrp_parallel.json"))}


def run_cell(model: str, split: str, mode: str) -> dict:
    import pandas as pd
    from habitat_llm.evaluation import viki_bench as bench
    from viki_amendment5 import BENCHMARK_ROOT
    from viki_amendment11_goalparse import extract_json
    from viki_eval_v2_intent_choice import to_requirement
    from viki_eval_skill_memory_v2 import visits_of
    from our_method.skill_memory_v2 import SEED, SkillMemoryV2, Simulator, planner

    affected = set(json.loads(PRE.read_text())["affected_eval_folds"])
    tag = CELL[split] % model
    rows = [json.loads(l) for l in (A11 / (tag + ".jsonl")).read_text().splitlines() if l.strip()]
    frame = pd.read_parquet(PARQUET.get(split, BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet"))
    sim = Simulator(BENCHMARK_ROOT)
    kind = mode.split("_")[0]
    cache = {}

    def memory_for(family):
        path = library_path(kind, split, family, affected)
        if path is None:
            return None, None
        if path not in cache:
            cache[path] = json.loads(path.read_text())
        return SkillMemoryV2(cache[path]), path

    out = []
    for record in rows:
        index = int(record["index"])
        memory, path = memory_for(record.get("task_name"))
        if memory is None:
            return {"model": model, "split": split, "mode": mode, "tag": tag, "status": "NOT_FOUND",
                    "why": "no sibling-group variant of the reference library on disk"}
        sample = bench.to_native(frame.iloc[index].to_dict())
        truth = bench.get_ground_truth(sample)
        blind = {k: v for k, v in truth.items() if k != "time_steps"}
        metadata = sim.metadata(blind, SEED)
        scene = sorted(metadata["assets"])

        def attempt(text):
            parsed = extract_json(text or "")
            work = (parsed or {}).get("work") if isinstance(parsed, dict) else None
            if not isinstance(work, list):
                return None, None
            requirements, crew = [], []
            for item in work:
                requirement = to_requirement(memory, item, scene) if isinstance(item, dict) else None
                if requirement is None:
                    continue
                requirements.append(requirement)
                crew.append([n for n in (item.get("robots") or []) if n in metadata["agents"]])
            if not requirements:
                return None, None
            env = sim.world(metadata)
            blind["goal_constraints"] = [[r] for r in requirements]
            blind["temporal_constraints"] = memory.order_for(requirements, visits_of(env, requirements, memory))
            casting = {}
            for requirement, names in zip(requirements, crew):
                if names:
                    casting[planner.predicate_key(requirement)] = names[0]
            return planner.plan(blind, memory, sim, SEED, crew=casting), casting

        try:
            result, casting = attempt(record.get("raw"))
            plan, reason = result if result is not None else (None, "NO_USABLE_WORK")
            reasked = False
            # the live runner re-asks only when the first answer parsed, had a casting and no plan
            if plan is None and result is not None and casting and mode.endswith("_full"):
                reasked = True
                second, _ = attempt(record.get("raw_reask"))
                if second is not None:
                    plan, reason = second
                if plan is None:
                    reason = "infeasible_assignment"
            elif plan is None and result is not None and casting:
                reason = "infeasible_assignment"
            if plan:
                solved = int(sim.score(plan, truth, SEED) > 0)
                reason = "SOLVED" if solved else ("OVER_BUDGET" if len(plan) > len(truth["time_steps"]) else "GOAL_UNMET")
            else:
                solved = 0
        except Exception as error:                                  # noqa: BLE001
            solved, reason, reasked = 0, "EXC_%s" % type(error).__name__, False
        out.append({"index": index, "task_name": record.get("task_name"), "solved": solved,
                    "reason": reason, "reask_would_fire": reasked,
                    "archived_reason": record.get("reason"), "archived_reask": bool(record.get("reask")),
                    "library": str(path.relative_to(ROOT))})
    return {"model": model, "split": split, "mode": mode, "tag": tag, "status": "ok",
            "n": len(out), "solved": sum(r["solved"] for r in out), "rows": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["72B", "30B", "7B"])
    ap.add_argument("--splits", nargs="+", default=list(CELL))
    ap.add_argument("--modes", nargs="+", default=["ours_full", "ours_first", "ref_first"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(m, s, k) for m in args.models for s in args.splits for k in args.modes]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, *job): job for job in jobs}
        for future in as_completed(futures):
            m, s, k = futures[future]
            result = future.result()
            path = args.out / ("replay_%s_%s_%s.json" % (m, s, k))
            path.write_text(json.dumps(result))                     # disk before print
            print(m, s, k, result["status"], result.get("solved"), "/", result.get("n"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
