#!/usr/bin/env python3
"""Ablate the modules that sit AFTER the model's answer, by replaying archived answers.

Why these four. The failure analysis of 2026-09-17 (`viki_tom_failure_analysis.py`) put the
bottleneck past the surface checks: of the 72B/id plans that already used a legal vocabulary,
named only real objects and were no longer than the reference, 94 % still died in the planner.
And on `ours`, the entire ID->OOD drop is one reason code -- 72B success 721 -> 501 (-220) while
`infeasible_assignment` goes 46 -> 266 (+220). So the modules worth ablating are the ones between
the answer and the plan, not the prompt.

None of them needs a GPU. The memory is consulted only after the answer arrives, so an archived
answer can be re-planned against any library and any of these switches -- which is what makes a
four-arm ablation affordable on a box whose cards are all taken.

  full         everything on, and it must reproduce the published cell or nothing below is usable
  no_casting   the planner searches for the assignment instead of taking the model's `robots`
  no_reask     only the first answer is replayed; the archived re-ask is ignored
  no_ordering  the memory's temporal constraints are dropped
  lib_*        the same answers re-planned against a different library

One arm per memory layer, each keeping the model's own casting so it differs from `full` in that layer only
(added 2026-09-17; `no_casting` is kept runnable but is not a reported arm):

  no_grounding      layer 3: names are taken as the model wrote them, no canonicalisation
  no_preconditions  layer 1: operators are ranked by cost/support only, not by how well their
                    recorded situation matches this one
  no_coordinated    layer 1: relay (multi-role) operators are withheld

`--check` is not decoration: it re-runs `full` and refuses to report if it misses the published
number, because a replay that cannot reproduce the cell it came from is measuring something else.

  python scripts/viki_ablation_replay.py --model 72B --split id \
      --json outputs/viki_ablation/id_72B.json --md outputs/viki_ablation/ID_72B.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from habitat_llm.evaluation import viki_bench as bench  # noqa: E402
from viki_amendment5 import BENCHMARK_ROOT  # noqa: E402
from viki_amendment11_goalparse import extract_json  # noqa: E402
from viki_eval_v2_intent_choice import to_requirement  # noqa: E402
from viki_eval_skill_memory_v2 import visits_of  # noqa: E402
from our_method.skill_memory_v2 import SEED, SkillMemoryV2, Simulator, planner  # noqa: E402

A11 = ROOT / "results/viki_memory_experiments/amendment11"
# v2 is the 4-operator library; the published cells (72B id 0.7803, held-out 0.5422) are v3, the
# 8-operator one. Replaying v2 answers against v2 memories reproduces neither -- 0.6126 and 0.1818 --
# which is what the `full` check is for. The version picks BOTH the archive and the memories, because
# they only make sense as a pair.
VERSION = "v3"
MEMORIES = ROOT / "outputs/v3_memories"
REFERENCE = A11 / "skill_memory_v2.json"

# The published cells this replay has to land on before any ablation is believed.
PUBLISHED = {("72B", "id"): 0.7803, ("72B", "fold"): 0.5422,
             ("30B", "id"): 0.5736, ("30B", "fold"): 0.3539,
             ("7B", "id"): 0.1385, ("7B", "fold"): 0.1169}

ARMS = ["full", "no_grounding", "no_preconditions", "no_coordinated", "no_ordering", "no_reask"]


def load_rows(model: str, split: str) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    if split == "id":
        for line in (A11 / f"{VERSION}_ours_{model}_id.jsonl").read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                rows[int(record["index"])] = record
        return rows
    # One fold file per family, each holding every row; only the family that fold held out is
    # taken from it, which is how the published held-out column is assembled.
    for path in sorted(A11.glob(f"{VERSION}_fold_{model}_*.jsonl")):
        family = path.stem[len(f"{VERSION}_fold_{model}_"):]
        if family.endswith(("_r2", "_r3")):        # repeat rounds, not part of the column
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("task_name") == family:
                record["_fold_family"] = family
                rows[int(record["index"])] = record
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="72B")
    ap.add_argument("--version", default="v3", choices=["v2", "v3"],
                    help="which library generation: archive and memories move together")
    ap.add_argument("--split", default="id", choices=["id", "fold"])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--library", default=None, help="extra arm: re-plan against this library json")
    ap.add_argument("--limit", type=int, default=None, help="first N rows, for a smoke run")
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--check", action="store_true", default=True)
    ap.add_argument("--no-check", dest="check", action="store_false")
    args = ap.parse_args(argv)

    global VERSION, MEMORIES
    VERSION = args.version
    MEMORIES = ROOT / f"outputs/{args.version}_memories"
    rows = load_rows(args.model, args.split)
    if args.limit:
        rows = {k: rows[k] for k in sorted(rows)[:args.limit]}
    ours = json.loads((MEMORIES / "memory_all.json").read_text())
    fold_base = {}
    if args.split == "fold":
        for family in {r["_fold_family"] for r in rows.values()}:
            path = MEMORIES / f"memory_heldout_{family}.json"
            if path.exists():
                fold_base[family] = json.loads(path.read_text())

    sim = Simulator(BENCHMARK_ROOT)
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")

    def memory_for(record: dict, override: Optional[dict]) -> SkillMemoryV2:
        if override is not None:
            return SkillMemoryV2(override)
        base = fold_base.get(record.get("_fold_family"), ours)
        return SkillMemoryV2(base)

    def switch_off(memory: SkillMemoryV2, arm: str) -> SkillMemoryV2:
        if arm == "no_coordinated":
            memory.operators = [o for o in memory.operators if not o.get("coordinated")]
        elif arm == "no_preconditions":
            def unranked(effect_key, facts, coordinated=False, _ops=memory.operators):
                ranked = [(o["cost"], -o["support"], i, o) for i, o in enumerate(_ops)
                          if o["effect"]["key"] == effect_key and bool(o.get("coordinated")) == coordinated]
                return [o for *_, o in sorted(ranked, key=lambda item: item[:3])]
            memory.operators_for = unranked
        return memory

    def solve(index: int, record: dict, arm: str, override: Optional[dict]) -> Dict[str, Any]:
        memory = switch_off(memory_for(record, override), arm)
        sample = bench.to_native(frame.iloc[index].to_dict())
        truth = bench.get_ground_truth(sample)
        blind = {k: v for k, v in truth.items() if k != "time_steps"}
        metadata = sim.metadata(blind, SEED)
        scene = sorted(metadata["assets"])

        def attempt(text: Optional[str]):
            parsed = extract_json(text or "")
            work = (parsed or {}).get("work") if isinstance(parsed, dict) else None
            if not isinstance(work, list):
                return None
            requirements, crew = [], []
            for item in work:
                requirement = (to_requirement(memory, item, scene, ground=arm != "no_grounding")
                               if isinstance(item, dict) else None)
                if requirement is None:
                    continue
                requirements.append(requirement)
                crew.append([n for n in (item.get("robots") or []) if n in metadata["agents"]])
            if not requirements:
                return None
            env = sim.world(metadata)
            blind["goal_constraints"] = [[r] for r in requirements]
            blind["temporal_constraints"] = ([] if arm == "no_ordering" else
                                             memory.order_for(requirements,
                                                              visits_of(env, requirements, memory)))
            casting = {}
            if arm != "no_casting":
                for requirement, names in zip(requirements, crew):
                    if names:
                        casting[planner.predicate_key(requirement)] = names[0]
            return planner.plan(blind, memory, sim, SEED, crew=casting or None)

        result = attempt(record.get("raw"))
        if result is not None and result[0]:
            plan, reason = result
        elif arm == "no_reask":
            plan, reason = None, "NO_PLAN_NO_REASK"
        else:
            second = attempt(record.get("raw_reask"))
            plan, reason = second if second is not None else (None, "NO_PLAN")
        if not plan:
            return {"solved": 0, "reason": reason or "NO_PLAN"}
        accuracy = sim.score(plan, truth, SEED)
        return {"solved": int(accuracy > 0),
                "reason": "SOLVED" if accuracy > 0 else
                          ("OVER_BUDGET" if len(plan) > len(truth["time_steps"]) else "GOAL_UNMET")}

    report: Dict[str, Any] = {"model": args.model, "split": args.split, "version": args.version,
                              "n": len(rows), "arms": {}}
    per_row: Dict[str, Dict[int, int]] = {}
    arms = list(args.arms)
    override_for = {a: None for a in arms}
    if args.library:
        arms.append("lib_" + Path(args.library).stem)
        override_for[arms[-1]] = json.loads(Path(args.library).read_text())

    for arm in arms:
        outcomes, solved = Counter(), {}
        for index in sorted(rows):
            try:
                out = solve(index, rows[index], arm, override_for.get(arm))
            except Exception as error:                              # noqa: BLE001
                out = {"solved": 0, "reason": f"EXC_{type(error).__name__}"}
            outcomes[out["reason"]] += 1
            solved[index] = out["solved"]
        per_row[arm] = solved
        rate = sum(solved.values()) / len(solved) if solved else 0.0
        report["arms"][arm] = {"solved": sum(solved.values()), "n": len(solved),
                               "rate": round(rate, 4), "reasons": dict(outcomes.most_common()),
                               "solved_rows": sorted(i for i, v in solved.items() if v)}
        print(f"[{arm}] {sum(solved.values())}/{len(solved)} = {rate:.4f}  {dict(outcomes.most_common(4))}",
              flush=True)

    # Paired against full: an ablation is read episode by episode, never as two means.
    if "full" in per_row:
        for arm in arms:
            if arm == "full":
                continue
            shared = sorted(set(per_row["full"]) & set(per_row[arm]))
            lost = [i for i in shared if per_row["full"][i] and not per_row[arm][i]]
            gained = [i for i in shared if not per_row["full"][i] and per_row[arm][i]]
            report["arms"][arm]["vs_full"] = {
                "delta": round(report["arms"][arm]["rate"] - report["arms"]["full"]["rate"], 4),
                "lost": len(lost), "gained": len(gained), "n_shared": len(shared)}

    published = PUBLISHED.get((args.model, args.split))
    if published is not None and "full" in report["arms"]:
        got = report["arms"]["full"]["rate"]
        report["published_full"] = published
        report["reproduces_published"] = abs(got - published) <= 0.02
        if args.check and not report["reproduces_published"] and not args.limit:
            report["ALARM"] = (f"full replay {got:.4f} != published {published:.4f}; "
                               "the ablation numbers are not trustworthy")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1))              # disk before print

    lines = [f"# Post-answer ablation, {args.model} / {args.split} (n={report['n']}, zero GPU)", ""]
    if "ALARM" in report:
        lines += [f"> **ALARM** {report['ALARM']}", ""]
    elif published is not None:
        lines += [f"full replay {report['arms']['full']['rate']:.4f} vs published {published:.4f} "
                  f"-- {'reproduces' if report.get('reproduces_published') else 'DOES NOT reproduce'}", ""]
    lines += ["| arm | solved | rate | vs full | lost | gained | top reasons |", "|---|---|---|---|---|---|---|"]
    for arm in arms:
        a = report["arms"][arm]
        v = a.get("vs_full", {})
        top = ", ".join(f"{k} {n}" for k, n in list(a["reasons"].items())[:3])
        lines.append(f"| {arm} | {a['solved']}/{a['n']} | {a['rate']:.4f} | "
                     f"{v.get('delta', ''):} | {v.get('lost', '')} | {v.get('gained', '')} | {top} |")
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.json}" + (f" and {args.md}" if args.md else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
