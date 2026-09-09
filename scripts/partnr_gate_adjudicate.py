#!/usr/bin/env python3
"""Decide which proposed operators the library keeps, from cells already run.

The first version of this rule accepted all six candidates, and it was wrong for the
reason this project has already written down once: a pass test that asks only "did the
score go up" admits duplicates. Four of the six raise a handful of episodes and every one
of those episodes is already raised by the best candidate -- new coverage beyond it, zero.
A fifth raises seventeen and lowers fourteen, which is churn, not capability.

So acceptance asks four things, and a candidate must satisfy all of them:

  complete   its cell covered every episode in the pool. An incomplete cell is not a
             weaker result, it is a different comparison: 25 workers died of CUDA OOM in
             one cell here and the intersection it left had a different baseline mean.
  real       the paired bootstrap interval excludes zero. The band between two identical
             cells is exactly 0.0, but that is reproducibility, not evidence about
             episodes this pool does not contain.
  marginal   it raises at least one episode that no already-accepted operator raises,
             taking candidates best-first. This is the condition the old rule lacked.
  harmless   it does not lower more than it raises.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRIC = "task_percent_complete"


def scores(cell: Path, pool: str) -> dict:
    out = {}
    for path in glob.glob(str(cell / "results" / f"{pool}.json.gz" / "stats" / "*.json")):
        blob = json.load(open(path))
        episode = os.path.basename(path)[:-5]
        out[episode] = json.loads(blob["stats"])[METRIC] if "stats" in blob else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pool", default="gate_iir")
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--library", type=Path, default=ROOT / "results/partnr_operators.json")
    args = ap.parse_args()

    outbase = ROOT / "outputs/gate" / args.tag
    payload = json.loads(args.candidates.read_text())
    base = scores(outbase / "base", args.pool)
    expected = {k for k, v in base.items() if v is not None}

    rows = []
    for index, entry in enumerate(payload["candidates"]):
        cell = outbase / f"cand{index}"
        comparison = json.loads((cell / "compare.json").read_text()) if (cell / "compare.json").is_file() else {}
        cand = scores(cell, args.pool)
        shared = [k for k in expected if cand.get(k) is not None]
        up = {k for k in shared if cand[k] - base[k] > 1e-9}
        down = {k for k in shared if cand[k] - base[k] < -1e-9}
        delta = sum(cand[k] - base[k] for k in shared) / len(shared) if shared else 0.0
        rows.append({"candidate": index, "operator": entry["operator"],
                     "advisory": entry.get("advisory"), "delta": delta,
                     "raises": sorted(up), "lowers": sorted(down),
                     "n_covered": len(shared), "n_expected": len(expected),
                     "complete": len(shared) == len(expected),
                     "ci": comparison.get("paired_bootstrap_95")})

    accepted, covered = [], set()
    for row in sorted(rows, key=lambda r: -r["delta"]):
        reasons = []
        if not row["complete"]:
            reasons.append(f"incomplete cell: {row['n_covered']} of {row['n_expected']} episodes")
        ci = row["ci"]
        if not ci or ci[0] <= 0.0:
            reasons.append(f"the paired interval {ci} does not exclude zero")
        new = set(row["raises"]) - covered
        if not new:
            reasons.append("raises no episode an accepted operator does not already raise")
        if len(row["lowers"]) >= len(row["raises"]):
            reasons.append(f"lowers {len(row['lowers'])} and raises {len(row['raises'])}")
        row["accepted"] = not reasons
        row["why"] = reasons or [f"raises {len(new)} episodes nothing accepted raises, "
                                 f"paired gain {row['delta']:+.4f}"]
        if row["accepted"]:
            accepted.append(row)
            covered |= set(row["raises"])

    operators = json.loads(args.library.read_text())["operators"]
    library = ROOT / f"results/partnr_operators_{args.tag}.json"
    library.write_text(json.dumps({
        "operators": operators + [r["operator"] for r in accepted],
        "accepted": [r["candidate"] for r in accepted],
        "adjudicated_on": args.pool}, indent=1))

    report = {"tag": args.tag, "pool": args.pool, "library": str(library.relative_to(ROOT)),
              "n_accepted": len(accepted),
              "rows": [{k: v for k, v in r.items() if k not in ("raises", "lowers", "operator")}
                       | {"n_raises": len(r["raises"]), "n_lowers": len(r["lowers"])} for r in rows]}
    (outbase / "ADJUDICATION.json").write_text(json.dumps(
        {**report, "detail": rows}, indent=1))          # disk before print
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
