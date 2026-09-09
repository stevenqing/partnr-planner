#!/usr/bin/env python3
"""Which operator would close the `infeasible_assignment` gap, and is it coordination?

The three families ours scores zero on in the ID column -- `set_plate_and_fork_on_table`,
`ensure_all_fruits_on_table`, `dog_push_box_for_two_panda_transport` -- all fail the same
way: the planner is handed the model's work and its casting and returns no plan at all. The
note in memory says that gap "points at coordination", but the reference library's
coordination operators are credited to only two families (`clear_table`, `dog_push`), so at
most one of the three can be a coordination gap. This settles it by measurement rather than
by reading the library.

Zero model calls. The model's answer does not depend on which library is loaded -- the
memory is consulted only after the answer arrives -- so an archived answer set can be
re-planned against any library. Both the first answer and the archived re-ask are replayed,
which is exactly what the original run planned with.

For each library in {ours, ours + one reference operator, reference}, this reports how many
rows of each family become solvable. The operator that recovers a family names what the gap
actually is.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
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
MEMORIES = ROOT / "outputs/v2_memories"
MANIFEST = ROOT / "results/viki_memory_experiments/amendment8b/interactive_manifest.jsonl"
REFERENCE = A11 / "skill_memory_v2.json"


def describe(operator: Dict[str, Any]) -> str:
    if operator.get("coordinated"):
        return "COORD[%d] %s" % (
            len(operator["roles"]),
            " | ".join(" ".join(item["action"][0] for item in role["actions"])
                       for role in operator["roles"]))
    return "%s %s" % (operator.get("kind", "?"),
                      " ".join(action[0] for action in (operator.get("body") or [])))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="72B")
    parser.add_argument("--families", nargs="+",
                        default=["set_plate_and_fork_on_table", "ensure_all_fruits_on_table",
                                 "dog_push_box_for_two_panda_transport",
                                 "toast_bread_and_set_plate"])
    parser.add_argument("--per-family", type=int, default=20,
                        help="rows per family; the sweep over 19 single additions is a "
                             "screen, so it runs on a sample and the winner is confirmed "
                             "on the whole family with --per-family 0")
    parser.add_argument("--operators", type=int, nargs="*", default=None,
                        help="reference operator indices to try; default all")
    parser.add_argument("--fold", action="store_true",
                        help="score the held-out column instead of ID: each family's rows "
                             "come from the run whose memory excluded it, and the base "
                             "library is that fold's own, so what is measured is what the "
                             "OOD cell would become -- not what ID would.")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    out = arguments.json or ROOT / ("results/viki_coord_gap_%s_%s.json"
                                    % (arguments.model, date.today().isoformat()))

    rows = {}
    if arguments.fold:
        # One fold file per family, each holding all 924 rows; only the family the fold
        # excluded is taken from it, which is how the published column is assembled.
        for family in arguments.families:
            path = A11 / ("v2_fold_%s_%s.jsonl" % (arguments.model, family))
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("task_name") == family:
                    rows[int(record["index"])] = record
    else:
        archive = A11 / ("v2_ours_%s_id.jsonl" % arguments.model)
        for line in archive.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                rows[int(record["index"])] = record

    ours = json.loads((MEMORIES / "memory_all.json").read_text())
    reference = json.loads(REFERENCE.read_text())
    ref_ops = reference["layer1"]["operators"]

    sim = Simulator(BENCHMARK_ROOT)
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")

    # In fold mode the base library is the fold's own, so a family is judged with the
    # library that fold actually had -- adding a reference operator on top of the whole
    # library would answer a question no held-out run ever asks.
    fold_base = {}
    if arguments.fold:
        for family in arguments.families:
            fold_base[family] = json.loads(
                (MEMORIES / ("memory_heldout_%s.json" % family)).read_text())

    def memory_with(operators: List[Dict[str, Any]], family: Optional[str] = None) -> SkillMemoryV2:
        base = fold_base.get(family, ours) if arguments.fold else ours
        record = dict(base)
        record["layer1"] = dict(base["layer1"], operators=operators)
        return SkillMemoryV2(record)

    def solve(index: int, memory: SkillMemoryV2) -> Dict[str, Any]:
        """Replay one archived answer against `memory`, first answer then archived re-ask."""
        record = rows[index]
        sample = bench.to_native(frame.iloc[index].to_dict())
        truth = bench.get_ground_truth(sample)
        blind = {k: v for k, v in truth.items() if k != "time_steps"}
        metadata = sim.metadata(blind, SEED)
        scene = sorted(metadata["assets"])

        def attempt(text: Optional[str]) -> Optional[Any]:
            parsed = extract_json(text or "")
            work = (parsed or {}).get("work") if isinstance(parsed, dict) else None
            if isinstance(parsed, dict) and not isinstance(work, list):
                # The re-ask answers with a bare {predicate_key: robot} object, which is a
                # casting for the work already read, not a new work list.
                return None
            if not isinstance(work, list):
                return None
            requirements, crew = [], []
            for item in work:
                requirement = (to_requirement(memory, item, scene) if isinstance(item, dict)
                               else None)
                if requirement is None:
                    continue
                requirements.append(requirement)
                crew.append([n for n in (item.get("robots") or []) if n in metadata["agents"]])
            if not requirements:
                return None
            env = sim.world(metadata)
            blind["goal_constraints"] = [[r] for r in requirements]
            blind["temporal_constraints"] = memory.order_for(
                requirements, visits_of(env, requirements, memory))
            casting = {}
            for requirement, names in zip(requirements, crew):
                if names:
                    casting[planner.predicate_key(requirement)] = names[0]
            plan, reason = planner.plan(blind, memory, sim, SEED, crew=casting or None)
            return (plan, reason)

        result = attempt(record.get("raw"))
        if result is not None and result[0]:
            plan, reason = result
        else:
            second = attempt(record.get("raw_reask"))
            plan, reason = second if second is not None else (None, "NO_PLAN")
        if not plan:
            return {"solved": 0, "reason": reason or "NO_PLAN"}
        accuracy = sim.score(plan, truth, SEED)
        return {"solved": int(accuracy > 0), "accuracy": accuracy,
                "reason": "SOLVED" if accuracy > 0 else
                          ("OVER_BUDGET" if len(plan) > len(truth["time_steps"]) else "GOAL_UNMET")}

    selected: Dict[str, List[int]] = {}
    for family in arguments.families:
        indices = sorted(i for i, r in rows.items() if r.get("task_name") == family)
        selected[family] = indices[:arguments.per_family] if arguments.per_family else indices

    base_ops = [] if arguments.fold else ours["layer1"]["operators"]
    libraries = [("ours", base_ops)]
    wanted = (range(len(ref_ops)) if arguments.operators is None else arguments.operators)
    for i in wanted:
        libraries.append(("ours+ref%d" % i, base_ops + [ref_ops[i]]))
    if len(wanted) > 1:
        libraries.append(("ours+ref%s" % "+".join(str(i) for i in wanted),
                          base_ops + [ref_ops[i] for i in wanted]))
    libraries.append(("reference19", ref_ops))

    report = {"generated": date.today().isoformat(), "model": arguments.model,
              "per_family": arguments.per_family,
              "reference_operators": {str(i): describe(op) for i, op in enumerate(ref_ops)},
              "selected": {f: len(v) for f, v in selected.items()}, "libraries": []}

    for name, added in libraries:
        entry: Dict[str, Any] = {"library": name, "by_family": {}}
        if name.startswith("ours+ref"):
            entry["added"] = [describe(ref_ops[int(i)])
                              for i in name[len("ours+ref"):].split("+")]
        sizes = set()
        for family, indices in selected.items():
            if arguments.fold:
                base = fold_base[family]["layer1"]["operators"]
                operators = base + [op for op in added if op not in base]
            else:
                operators = added
            sizes.add(len(operators))
            memory = memory_with(operators, family)
            outcomes = [solve(i, memory) for i in indices]
            entry["by_family"][family] = {
                "n": len(indices),
                "solved": sum(o["solved"] for o in outcomes),
                "reasons": dict(Counter(o["reason"] for o in outcomes).most_common(3)),
            }
        entry["solved_total"] = sum(v["solved"] for v in entry["by_family"].values())
        entry["n_operators"] = sorted(sizes)
        report["libraries"].append(entry)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
        print("%-14s ops=%-8s total=%3d   %s" % (
            name, ",".join(str(x) for x in sorted(sizes)), entry["solved_total"],
            "  ".join("%s %d/%d" % (f.split("_")[0], v["solved"], v["n"])
                      for f, v in entry["by_family"].items())), flush=True)

    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
