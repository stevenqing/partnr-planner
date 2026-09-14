"""Turn the ladder's accepted operators into a library the gate can score.

The rung accepts one operator at a time; the gate scores a library. Nothing has ever
connected them, so "what does an agent-built library actually score" has no answer on
disk -- only the blind-write loop's 0/200, which is a different method. This closes that.

What it does, and why each step is mechanical:

  dedup      by effect key and abstracted body. Twelve passes across two models turned out
             to be one operator submitted twelve times; counting that as twelve would be a
             coverage claim the artefact does not support.
  support    recomputed by *running* each operator on episodes from the induction half and
             counting where it binds and its effect then holds. The `support: 1` the model
             wrote is its own say-so and is discarded. Support is an observation here.
  minimum    an operator must work on at least two distinct episodes to enter the library.
             That is the anti-gaming rule in the protocol, and it is enforced here rather
             than trusted, because `works_on` in a verdict was measured against a holdout
             pool chosen by whoever launched that run.

The agent proposes; this file only counts. No operator is admitted, rewritten or repaired
on the strength of anything a model said about it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import viki_fork_guard  # noqa: E402

viki_fork_guard.install()

from viki_induction_tools import Workbench  # noqa: E402
from our_method.skill_memory_v2.simulator import SEED  # noqa: E402

TRAIN = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet"
BENCHMARK = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R"
REFERENCE = Path("results/viki_memory_experiments/amendment11/skill_memory_v2.json")


def signature(operator: Dict[str, Any]):
    """What makes two submissions the same operator: the effect and the abstracted body.

    Deliberately blind to `families`, `runner_types`, `support` and `types` -- those are
    provenance and bookkeeping, and two submissions that differ only there are the same
    operator arrived at from two episodes, which is exactly what support is for.
    """
    effect = operator.get("effect") or {}
    if operator.get("coordinated"):
        body = tuple(
            (item.get("action") or [None])[0]
            for role in operator.get("roles", [])
            for item in role.get("actions", [])
        )
    else:
        body = tuple(tuple(action) for action in operator.get("body", []))
    return (effect.get("key"), effect.get("subject"), str(effect.get("value")),
            bool(operator.get("coordinated")), body)


def collect(roots: List[Path]) -> List[Dict[str, Any]]:
    found = []
    for root in roots:
        for path in sorted(root.rglob("verdict.json")):
            try:
                verdict = json.loads(path.read_text())
            except Exception:
                continue
            if not verdict.get("passed") or "operator" not in verdict:
                continue
            found.append({"operator": verdict["operator"], "source": str(path),
                          "works_on_at_accept": verdict.get("works_on") or []})
    return found


def collect_candidates(path: Path, family) -> List[Dict[str, Any]]:
    """RQ2 no_execution_admission: every format- and type-valid submission of one family.

    Reads the pre-admission candidates file; nothing in it was chosen by execution. The
    family is the proposal cell's own family, not the model-written `families` list.
    """
    found = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if family is not None and record.get("family") != family:
            continue
        if not (record.get("format_valid") and record.get("type_valid")):
            continue
        found.append({"operator": record["parsed_operator"],
                      "source": record["candidate_id"], "works_on_at_accept": []})
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung-root", type=Path, nargs="+",
                        default=[Path("outputs/agentic_rung")])
    parser.add_argument("--probe", type=int, default=60,
                        help="how many induction-half episodes to measure support over")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("outputs/agentic_library.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/agentic_library_assembly.json"))
    # RQ2 condition `no_execution_admission` (2026-09-14), OFF by default: with neither flag
    # this file behaves exactly as before. `--candidates` reads every format- and type-valid
    # submission of one family from the pre-admission candidates file written by
    # scripts/viki_rq2_no_exec_admission.py instead of the verdicts that passed the rung;
    # `--no-support-probe` skips the support re-probe and the minimum, so dedup is the only
    # step left. Support is then the number of distinct proposal cells that submitted the
    # operator -- a count over proposal records, never an execution result.
    parser.add_argument("--candidates", type=Path, default=None,
                        help="pre-admission candidates JSONL (RQ2 no_execution_admission)")
    parser.add_argument("--family", default=None,
                        help="with --candidates: the proposal cells' family to assemble")
    parser.add_argument("--no-support-probe", action="store_true",
                        help="skip the support re-probe and --min-support (no execution)")
    arguments = parser.parse_args()

    if arguments.candidates is not None:
        submissions = collect_candidates(arguments.candidates, arguments.family)
        if not submissions:
            print("no format/type-valid candidates for %s in %s"
                  % (arguments.family, arguments.candidates))
            return 1
    else:
        submissions = collect(arguments.rung_root)
    if not submissions:
        print("no accepted operators found under %s" % arguments.rung_root)
        return 1

    bench = None
    if not arguments.no_support_probe:
        reference = json.loads(REFERENCE.read_text())
        bench = Workbench(TRAIN, BENCHMARK, SEED,
                          {"layer2": reference["layer2"], "layer3": reference["layer3"]})

    grouped: Dict[Any, Dict[str, Any]] = {}
    for entry in submissions:
        key = signature(entry["operator"])
        slot = grouped.setdefault(key, {"operator": entry["operator"], "sources": [],
                                        "families": [], "runner_types": []})
        slot["sources"].append(entry["source"])
        for name in entry["operator"].get("families") or []:
            if name not in slot["families"]:
                slot["families"].append(name)
        for name in entry["operator"].get("runner_types") or []:
            if name not in slot["runner_types"]:
                slot["runner_types"].append(name)

    operators, rows = [], []
    for key, slot in grouped.items():
        operator = dict(slot["operator"])
        works = []
        if bench is not None:
            for index in range(arguments.probe):
                try:
                    outcome = bench.run_operator(operator, index)
                except Exception:
                    continue
                if outcome.get("bound") and outcome.get("effect_holds"):
                    works.append(index)
        row = {"effect_key": key[0], "coordinated": key[3],
               "submissions": len(slot["sources"]), "sources": slot["sources"],
               "measured_support": len(works), "works_on": works[:20],
               "probe_episodes": arguments.probe}
        # Provenance and the contract fields the bench reads. `support` is the measured
        # count, never the model's claim.
        operator["support"] = len(works)
        operator["families"] = slot["families"]
        operator["runner_types"] = slot["runner_types"]
        operator.setdefault("preconditions", {})
        operator.setdefault("cost", len(operator.get("body") or []) or 1)
        operator["provenance"] = {"proposed_by": "agentic_rung", "sources": slot["sources"],
                                 "verified_on": works[:20]}
        row["admitted"] = len(works) >= arguments.min_support
        if arguments.no_support_probe:
            cells = sorted({source.split("#")[0] for source in slot["sources"]})
            operator["support"] = len(cells)
            operator["provenance"]["proposal_cells"] = cells
            operator["provenance"]["admission"] = "no_execution_admission"
            row["proposal_cells"] = len(cells)
            row["probe_episodes"] = 0
            row["admitted"] = True
        rows.append(row)
        if row["admitted"]:
            operators.append(operator)

    operators.sort(key=lambda item: -item.get("support", 0))
    library = {"operators": operators,
               "built_by": "viki_assemble_agentic_library",
               "min_support": arguments.min_support,
               "probe_episodes": arguments.probe}
    if arguments.no_support_probe:
        library.update({"admission": "no_execution_admission", "min_support": None,
                        "probe_episodes": 0,
                        "support_meaning": "distinct proposal cells that submitted this operator"})

    report = {
        "submissions_found": len(submissions),
        "distinct_operators": len(grouped),
        "admitted": len(operators),
        "rejected_below_min_support": sum(1 for r in rows if not r["admitted"]),
        "effect_keys": dict(Counter(o["effect"]["key"] for o in operators)),
        "kinds": dict(Counter(o.get("kind", "achievement") for o in operators)),
        "rows": rows,
    }

    # Disk before stdout. A report that dies after the table and before the dump leaves
    # nothing, and that has already cost this project one full run.
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(library, indent=1))
    arguments.report.write_text(json.dumps(report, indent=1))

    print("accepted submissions   %d" % report["submissions_found"])
    print("distinct operators     %d   (dedup by effect + abstracted body)" % report["distinct_operators"])
    print("admitted to library    %d   (measured support >= %d over %d episodes)"
          % (report["admitted"], arguments.min_support, arguments.probe))
    print("rejected               %d" % report["rejected_below_min_support"])
    print("effect keys            %s" % report["effect_keys"])
    print("kinds                  %s" % report["kinds"])
    print()
    print("%-14s %6s %6s %9s  %s" % ("effect", "subs", "supp", "admitted", "sources"))
    for row in sorted(rows, key=lambda r: -r["measured_support"]):
        print("%-14s %6d %6d %9s  %s"
              % (row["effect_key"], row["submissions"], row["measured_support"],
                 row["admitted"], row["sources"][0].split("/")[-2]))
    print("\n-> %s\n-> %s" % (arguments.out, arguments.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
