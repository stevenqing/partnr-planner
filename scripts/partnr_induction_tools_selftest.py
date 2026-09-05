"""Calibrate the PARTNR workbench in both directions before anything is built on it.

An acceptance oracle that says yes to everything is not an oracle. So it is pointed at
operators whose verdict is known in advance:

  known right   the shipped inducer's own operators, read from `results/partnr_operators.json`.
                These were induced from these very traces by a deterministic attributor, so a
                verifier that cannot confirm them is broken.
  known wrong   the same operators, deliberately spoiled one way at a time -- effect key
                swapped, body reversed, a verb replaced. A verifier that confirms these is
                measuring nothing.

The interesting column is `precision`: of the traces where the body matched an actor's
recorded actions, how often did the recording actually satisfy the operator's effect at the
end of that window. Both `matched` and `predicted` are printed because an operator that
never matches is untested, not refuted, and reporting those as the same thing would hide a
verifier that has simply gone silent.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from partnr_induction_tools import Workbench  # noqa: E402

LIBRARY = Path("results/partnr_operators.json")


def spoil_effect(operator):
    bad = copy.deepcopy(operator)
    bad["effect"] = dict(bad["effect"])
    bad["effect"]["key"] = "is_in_room" if bad["effect"]["key"] != "is_in_room" else "is_on_top"
    return bad, "effect key swapped"


def spoil_order(operator):
    bad = copy.deepcopy(operator)
    bad["body"] = list(reversed(bad["body"]))
    return bad, "body reversed"


def spoil_verb(operator):
    bad = copy.deepcopy(operator)
    if bad["body"]:
        bad["body"] = [list(a) for a in bad["body"]]
        bad["body"][-1][0] = "Navigate"
    return bad, "closing verb replaced by Navigate"


def main() -> int:
    bench = Workbench()
    print("traces on disk: %d\n" % len(bench.paths))
    operators = json.loads(LIBRARY.read_text())["operators"]
    # The three highest-support operators: the ones the shipped library actually leans on.
    chosen = sorted(operators, key=lambda o: -o.get("support", 0))[:3]

    print("%-46s %8s %10s %11s" % ("operator", "matched", "predicted", "precision"))
    rows = []
    for operator in chosen:
        verbs = "/".join(a[0] for a in operator["body"])
        for label, candidate in [("as induced", operator)] + [
            (why, bad) for bad, why in
            (spoil_effect(operator), spoil_order(operator), spoil_verb(operator))
        ]:
            result = bench.score_operator(candidate, exclude=[])
            name = "%s %s" % (operator["effect"]["key"], verbs)
            print("%-46s %8d %10d %11s   %s"
                  % ((name if label == "as induced" else "   ^ " + label)[:46],
                     result["matched"], result["predicted"],
                     result["precision"], ""))
            rows.append({"operator": name, "variant": label, **{
                k: result[k] for k in ("matched", "predicted", "precision")}})
        print()

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/partnr_induction_tools_selftest.json").write_text(json.dumps(rows, indent=1))

    induced = [r for r in rows if r["variant"] == "as induced"]
    spoiled = [r for r in rows if r["variant"] != "as induced"]
    good = [r["precision"] for r in induced if r["precision"] is not None]
    bad = [r["precision"] for r in spoiled if r["precision"] is not None]
    print("as-induced precision: %s" % good)
    print("spoiled precision:    %s" % bad)
    if good and bad and min(good) > max(bad):
        print("\nSEPARATED: every operator as induced scores above every spoiled one.")
    else:
        print("\nNOT SEPARATED -- this verifier cannot yet be used as an acceptance test.")
    print("\n-> outputs/partnr_induction_tools_selftest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
