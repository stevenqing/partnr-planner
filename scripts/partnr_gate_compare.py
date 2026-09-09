#!/usr/bin/env python3
"""Compare two PARTNR cells run on the same pool, episode by episode.

The acceptance question is not "is condition A's mean higher than B's". It is "on the
same episodes, did adding this operator move anything, and did it move the episodes it
was supposed to move". So everything here is paired: an episode that neither condition
completes contributes an exact zero, and the noise band is estimated from the cells that
are supposed to be identical rather than assumed from a different split.

Also reported, because on this benchmark it is the sharper signal: how many episodes end
at planner step 0. A memory with no entry for a required predicate does not score badly,
it declines to act -- 61 of the 68 `val_mini` episodes carrying `is_in_room` die at step
0 -- so a recovered entry shows up first as a death that stopped happening.
"""
from __future__ import annotations

import argparse, glob, gzip, json, os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from partnr_task_types import classify

METRIC = "task_percent_complete"


def cell_scores(cell: Path, pool: str) -> dict:
    out = {}
    for path in glob.glob(str(cell / "results" / f"{pool}.json.gz" / "stats" / "*.json")):
        episode = os.path.basename(path)[:-5]
        blob = json.load(open(path))
        if "stats" not in blob:
            out[episode] = None  # crashed, not zero: kept distinct on purpose
            continue
        out[episode] = json.loads(blob["stats"])[METRIC]
    return out


def cell_steps(cell: Path, pool: str) -> dict:
    out = {}
    for path in glob.glob(str(cell / "results" / f"{pool}.json.gz" / "planner-log" / "planner-log-episode_*_0.json")):
        episode = os.path.basename(path).split("episode_")[1].rsplit("_", 1)[0]
        try:
            out[episode] = len(json.load(open(path))["steps"])
        except Exception:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()

    data = json.load(gzip.open(ROOT / f"data/datasets/partnr_episodes/v0_0/{args.pool}.json.gz", "rt"))
    keys = {str(e["episode_id"]): {p.get("function_name") for p in (e.get("evaluation_propositions") or [])}
            for e in data["episodes"]}
    types = {str(e["episode_id"]): classify(e) for e in data["episodes"]}

    expected = {str(e["episode_id"]) for e in data["episodes"]}
    a, b = cell_scores(args.a, args.pool), cell_scores(args.b, args.pool)
    sa, sb = cell_steps(args.a, args.pool), cell_steps(args.b, args.pool)
    shared = sorted(set(a) & set(b))
    both_ok = [k for k in shared if a[k] is not None and b[k] is not None]

    deltas = {k: b[k] - a[k] for k in both_ok}
    moved = sorted(k for k, d in deltas.items() if abs(d) > 1e-9)
    mean_delta = sum(deltas.values()) / len(deltas) if deltas else float("nan")

    rng = random.Random(args.seed)
    values = list(deltas.values())
    boot = []
    for _ in range(args.boot):
        sample = [values[rng.randrange(len(values))] for _ in values] if values else [0.0]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    lo, hi = (boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]) if boot else (0.0, 0.0)

    def deaths(steps, subset):
        return sum(1 for k in subset if steps.get(k, 99) <= 1)

    # A cell that lost workers -- CUDA OOM at startup has done it -- still writes stats for
    # the episodes it did reach, and comparing on the intersection silently changes which
    # episodes the baseline is averaged over. That produced a plausible +0.2177 from a cell
    # holding 32 of 60 episodes. An incomplete cell is reported as incomplete and no verdict
    # is taken from it.
    complete = len(expected - set(a)) == 0 and len(expected - set(b)) == 0
    report = {
        "pool": args.pool, "a": str(args.a), "b": str(args.b),
        "complete": complete, "expected_episodes": len(expected),
        "missing_from_a": sorted(expected - set(a))[:20],
        "missing_from_b": sorted(expected - set(b))[:20],
        "n_missing_a": len(expected - set(a)), "n_missing_b": len(expected - set(b)),
        "n_shared": len(shared), "n_both_scored": len(both_ok),
        "crashed_a": sorted(k for k in shared if a[k] is None),
        "crashed_b": sorted(k for k in shared if b[k] is None),
        "mean_a": sum(a[k] for k in both_ok) / len(both_ok) if both_ok else None,
        "mean_b": sum(b[k] for k in both_ok) / len(both_ok) if both_ok else None,
        "mean_delta": mean_delta,
        "paired_bootstrap_95": [lo, hi],
        "n_episodes_moved": len(moved),
        "episodes_moved": {k: round(deltas[k], 4) for k in moved[:40]},
        "max_abs_delta": max((abs(d) for d in values), default=0.0),
        "step0_deaths_a": deaths(sa, both_ok), "step0_deaths_b": deaths(sb, both_ok),
    }
    for label, subset in (("carrying_is_in_room", [k for k in both_ok if "is_in_room" in keys.get(k, set())]),
                          ("not_carrying", [k for k in both_ok if "is_in_room" not in keys.get(k, set())])):
        if not subset:
            continue
        report[label] = {
            "n": len(subset),
            "mean_a": sum(a[k] for k in subset) / len(subset),
            "mean_b": sum(b[k] for k in subset) / len(subset),
            "mean_delta": sum(deltas[k] for k in subset) / len(subset),
            "moved": sum(1 for k in subset if abs(deltas[k]) > 1e-9),
            "step0_deaths_a": deaths(sa, subset), "step0_deaths_b": deaths(sb, subset),
        }
    report["types"] = sorted({types.get(k, "?") for k in both_ok})

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1))   # on disk BEFORE anything is printed
    print(json.dumps({k: v for k, v in report.items() if k != "episodes_moved"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
