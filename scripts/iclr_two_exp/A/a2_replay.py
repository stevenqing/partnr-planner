#!/usr/bin/env python3
"""A2 zero-call part: the frozen RQ2 first-turn answers through the planner with the certified
replay-mined memory in place of the admitted one. No model call.

Per row it logs solved (strict SOLVED from `simulator.score`), reason, whether the live runner's
re-ask would fire (first answer parsed, the model cast robots, no plan -- the rule at
scripts/viki_eval_v2_intent_choice.py `if plan is None and casting`), and the operators the
planner's chosen schedule used. Those rows are then sent, unchanged, through the live runner
by a2_reask.sh; everything else here is final.

The row logic is scripts/iclr_master/A/a_replay.py's `attempt`, which reproduced all 15 published
cells row for row; only the memory path and the operator logging are new. Operator logging
wraps `planner.schedule` to keep the chains of the schedule `compose` returns; it does not
change what is searched or returned.

Modes: replay_first (replay-mined memory, first answer) and ours_first (admitted memory, first
answer; used only for the CG same-operator-sequence count).
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
OURS = ROOT / "outputs/v3_memories"
RMEM = ROOT / "results/iclr_two_exp_2026-09-22/A/memories"
CELL = {"id": "v3_ours_%s_id", "ood": "v3_ours_%s_heldout", "cg_image": "v3_oursall_%s_imaged",
        "cg_text": "v3_oursall_%s_text"}
PARQUET = {"cg_text": A10 / "recombination.text.parquet", "cg_image": A10 / "recombination.imaged.parquet"}

_CACHE = {}


def signature(op):
    e = op.get("effect") or {}
    if op.get("coordinated"):
        body = [[i["action"] for i in r["actions"]] for r in op["roles"]]
    else:
        body = op.get("body")
    return json.dumps([e.get("key"), bool(op.get("coordinated")), body])


def _consistent(pattern, actual, binding):
    if len(pattern) != len(actual):
        return False
    for p, a in zip(pattern, actual):
        if p[0] != a[0] or len(p) != len(a):
            return False
        for t, v in zip(p[1:], a[1:]):
            if t.startswith("?r"):
                if v != t:
                    return False
                continue
            if t.startswith("?"):
                if binding.setdefault(t, v) != v:
                    return False
            elif t != v:
                return False
    return len(set(binding.values())) == len(binding)


def match(chain, operators):
    out = []
    for op in operators:
        if op.get("coordinated"):
            if chain.get("group") is None:
                continue
            role = op["roles"][chain["role"]] if chain["role"] < len(op["roles"]) else None
            if role and _consistent([i["action"] for i in role["actions"]], chain["actions"], {}):
                out.append(signature(op))
        elif chain.get("group") is None and _consistent(op["body"], chain["actions"], {}):
            out.append(signature(op))
    return out


def setup():
    if "sim" in _CACHE:
        return _CACHE
    import pandas as pd
    from viki_amendment5 import BENCHMARK_ROOT
    from our_method.skill_memory_v2 import Simulator, planner
    _CACHE["sim"] = Simulator(BENCHMARK_ROOT)
    _CACHE["frames"] = {k: pd.read_parquet(v) for k, v in PARQUET.items()}
    _CACHE["frames"]["id"] = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    original = planner.schedule
    last = {}

    def schedule(metadata, plans, sim, cap=planner.STEP_CAP):
        steps = original(metadata, plans, sim, cap)
        if steps is not None:
            last[id(steps)] = plans
        return steps
    planner.schedule = schedule
    _CACHE["last"] = last
    return _CACHE


def memory_path(mode, split, family):
    base = RMEM if mode.startswith("replay") else OURS
    if split == "ood":
        return base / ("memory_heldout_%s.json" % family)
    return base / "memory_all.json"


def run_chunk(model, split, mode, rows):
    from habitat_llm.evaluation import viki_bench as bench
    from viki_amendment11_goalparse import extract_json
    from viki_eval_v2_intent_choice import to_requirement
    from viki_eval_skill_memory_v2 import visits_of
    from our_method.skill_memory_v2 import SEED, SkillMemoryV2, planner
    c = setup()
    sim, last = c["sim"], c["last"]
    frame = c["frames"]["id" if split in ("id", "ood") else split]
    mems = c.setdefault("mems", {})
    out = []
    for record in rows:
        index = int(record["index"])
        path = memory_path(mode, split, record.get("task_name"))
        if path not in mems:
            mems[path] = SkillMemoryV2(json.loads(path.read_text()))
        memory = mems[path]
        sample = bench.to_native(frame.iloc[index].to_dict())
        truth = bench.get_ground_truth(sample)
        blind = {k: v for k, v in truth.items() if k != "time_steps"}
        metadata = sim.metadata(blind, SEED)
        scene = sorted(metadata["assets"])
        row = {"index": index, "task_name": record.get("task_name"), "memory": str(path.relative_to(ROOT))}
        try:
            parsed = extract_json(record.get("raw") or "")
            work = (parsed or {}).get("work") if isinstance(parsed, dict) else None
            if not isinstance(work, list):
                row.update(solved=0, reason="UNPARSEABLE", reask_would_fire=False)
                out.append(row)
                continue
            requirements, crew = [], []
            for item in work:
                requirement = to_requirement(memory, item, scene) if isinstance(item, dict) else None
                if requirement is None:
                    continue
                requirements.append(requirement)
                crew.append([n for n in (item.get("robots") or []) if n in metadata["agents"]])
            if not requirements:
                row.update(solved=0, reason="NO_USABLE_WORK", reask_would_fire=False)
                out.append(row)
                continue
            env = sim.world(metadata)
            blind["goal_constraints"] = [[r] for r in requirements]
            blind["temporal_constraints"] = memory.order_for(requirements, visits_of(env, requirements, memory))
            casting = {}
            for requirement, names in zip(requirements, crew):
                if names:
                    casting[planner.predicate_key(requirement)] = names[0]
            last.clear()
            plan, reason = planner.plan(blind, memory, sim, SEED, crew=casting)
            plans = last.get(id(plan)) if plan else None
            if plans:
                row["selected"] = [match(ch, memory.operators) for robot in sorted(plans) for ch in plans[robot]]
            if plan is None and casting:
                row.update(solved=0, reason="infeasible_assignment", reask_would_fire=True,
                           planner_reason=reason, casting=casting)
            elif plan:
                solved = int(sim.score(plan, truth, SEED) > 0)
                row.update(solved=solved, reask_would_fire=False, plan_len=len(plan),
                           reason="SOLVED" if solved else ("OVER_BUDGET" if len(plan) > len(truth["time_steps"])
                                                           else "GOAL_UNMET"))
            else:
                row.update(solved=0, reason=reason or "NO_PLAN", reask_would_fire=False)
        except Exception as error:                                   # noqa: BLE001
            row.update(solved=0, reason="EXC_%s" % type(error).__name__, reask_would_fire=False)
        out.append(row)
    return model, split, mode, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["72B", "30B", "7B"])
    ap.add_argument("--splits", nargs="+", default=list(CELL))
    ap.add_argument("--modes", nargs="+", default=["replay_first"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--chunk", type=int, default=24)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs, cells = [], {}
    for m in args.models:
        for s in args.splits:
            rows = [json.loads(l) for l in (A11 / ((CELL[s] % m) + ".jsonl")).read_text().splitlines() if l.strip()]
            for mode in args.modes:
                cells[(m, s, mode)] = []
                for i in range(0, len(rows), args.chunk):
                    jobs.append((m, s, mode, rows[i:i + args.chunk]))
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_chunk, *job) for job in jobs]
        for f in as_completed(futures):
            m, s, mode, out = f.result()
            cells[(m, s, mode)].extend(out)
    for (m, s, mode), rows in cells.items():
        rows.sort(key=lambda r: r["index"])
        result = {"model": m, "split": s, "mode": mode, "n": len(rows), "solved": sum(r["solved"] for r in rows),
                  "reask_would_fire": sum(1 for r in rows if r.get("reask_would_fire")), "rows": rows}
        (args.out / ("%s_%s_%s.json" % (m, s, mode))).write_text(json.dumps(result))
    for (m, s, mode), rows in sorted(cells.items()):
        print(m, s, mode, len(rows), sum(r["solved"] for r in rows),
              "reask", sum(1 for r in rows if r.get("reask_would_fire")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
