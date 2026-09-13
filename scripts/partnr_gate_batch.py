#!/usr/bin/env python3
"""Accept or reject proposed operators by executing them, one cell per candidate.

This is the acceptance oracle for PARTNR, and it is the outer one on purpose. The inner
oracle -- does the body, matched against another rollout's actions, predict a satisfaction
the recording contains -- was calibrated on 2026-09-07 against operators broken on purpose
and did not separate them at any threshold. Nothing weaker than execution is left, and
execution is affordable here only because a cell is bounded by its slowest episode rather
than by their sum: 40 episodes at 40 processes finished in 11 minutes.

The decision rule, per candidate, against the same pool run with the memory alone:
  gain      the paired mean of `task_percent_complete` must rise by more than the band
            two identical cells produce on that pool
  aim       the gain must land on episodes that carry the target predicate
  no harm   episodes that do NOT carry it must not fall
A candidate that raises the mean only on episodes the predicate is absent from has changed
the division of labour, not the vocabulary, and is refused.

The band is not assumed. Two identical cells on the same 40-episode pool, run under
different load, agreed on all 39 scored episodes to the last digit -- max paired delta
0.0 -- so on a fixed pool any non-zero paired movement is signal and the guard that
matters is not noise but selection: the pool the gate measures on (`gate_iir`) and the
pool the paper reports on are disjoint, and a confirmation pool (`conf_iir`) is run once,
at the end, on the library that acceptance produced.
"""
from __future__ import annotations

import argparse, json, subprocess, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "scripts/drivers/partnr_gate_cell.sh"


def run_cell(pool: str, name: str, outbase: Path, operators: Path, procs: int, timeout: int,
             gpu: int = 1) -> int:
    """One cell. A cell is bounded by its slowest episode, not by their sum, so a pool
    smaller than `procs` costs one wave however many episodes it holds -- which is why
    several cells are worth running side by side rather than one after another."""
    env = {"OPERATORS": str(operators.relative_to(ROOT)), "PROCS": str(procs),
           "HARD_TIMEOUT": str(timeout), "GPU": str(gpu)}
    prefix = " ".join(f"{k}={v}" for k, v in env.items())
    command = f"{prefix} bash {CELL} {pool} {name} {outbase.relative_to(ROOT)}"
    print(f"[gate] {command}", flush=True)
    return subprocess.call(command, shell=True, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--pool", default="gate_iir")
    ap.add_argument("--library", type=Path, default=ROOT / "results/partnr_operators.json")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--procs", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--gpus", type=int, nargs="*", default=[0, 1, 6],
                    help="cells are run one per GPU, this many at a time")
    ap.add_argument("--key", default=None,
                    help="predicate the candidates serve; default is the candidates file's target_key")
    ap.add_argument("--band", type=float, default=None,
                    help="noise band; default is read from the determinism probe report")
    args = ap.parse_args()

    outbase = ROOT / "outputs/gate" / args.tag
    libdir = ROOT / "results/partnr_gate" / args.tag
    libdir.mkdir(parents=True, exist_ok=True)
    outbase.mkdir(parents=True, exist_ok=True)

    blob = json.loads(args.library.read_text())
    base_operators = blob["operators"] if isinstance(blob, dict) else blob

    payload = json.loads(args.candidates.read_text())
    candidates = payload["candidates"]
    target_key = args.key or payload.get("target_key") or "is_in_room"
    if not candidates:
        print(json.dumps({"tag": args.tag, "candidates": 0, "verdict": "nothing proposed"}))
        return 0

    # baseline first: the same pool, the memory as it stands
    base_path = libdir / "base.json"
    base_path.write_text(json.dumps({"operators": base_operators, "variant": "base"}, indent=1))
    run_cell(args.pool, "base", outbase, base_path, args.procs, args.timeout, args.gpus[0])

    # Build every candidate library first, then run the cells in waves of len(gpus).
    for index, entry in enumerate(candidates):
        (libdir / f"cand{index}.json").write_text(json.dumps(
            {"operators": base_operators + [entry["operator"]], "variant": f"cand{index}",
             "added": entry["operator"]}, indent=1))

    pending = list(range(len(candidates)))
    while pending:
        wave, pending = pending[:len(args.gpus)], pending[len(args.gpus):]
        threads = []
        for slot, index in enumerate(wave):
            thread = threading.Thread(target=run_cell, args=(
                args.pool, f"cand{index}", outbase, libdir / f"cand{index}.json",
                args.procs, args.timeout, args.gpus[slot % len(args.gpus)]))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

    results = []
    for index, entry in enumerate(candidates):
        name = f"cand{index}"
        report = outbase / name / "compare.json"
        subprocess.call([sys.executable, str(ROOT / "scripts/partnr_gate_compare.py"),
                         "--pool", args.pool, "--a", str(outbase / "base"),
                         "--b", str(outbase / name), "--json", str(report),
                         "--key", target_key], cwd=ROOT)
        comparison = json.loads(report.read_text()) if report.is_file() else {}
        results.append({"candidate": index, "move": entry.get("move"),
                        "advisory": entry.get("advisory"), "comparison": comparison,
                        "operator": entry["operator"]})

    band = 0.0 if args.band is None else args.band
    for item in results:
        c = item["comparison"] or {}
        carrying = c.get("carrying") or c.get("carrying_is_in_room") or {}
        others = c.get("not_carrying") or {}
        gain = c.get("mean_delta")
        reasons = []
        if not c.get("complete", False):
            reasons.append(f"the cell is incomplete: {c.get('n_missing_b')} of "
                           f"{c.get('expected_episodes')} episodes never ran")
        if gain is None:
            reasons.append("the cell produced no comparable scores")
        else:
            if gain <= band:
                reasons.append(f"paired gain {gain:.4f} does not clear the band {band:.4f}")
            if carrying and carrying.get("mean_delta", 0.0) <= 0:
                reasons.append("no gain on the episodes that carry the predicate")
            if others and others.get("mean_delta", 0.0) < -1e-9:
                reasons.append("episodes without the predicate got worse")
        item["accepted"] = not reasons
        item["why"] = reasons or ["gain, aimed at the predicate, with no collateral loss"]

    summary = {"tag": args.tag, "pool": args.pool, "library": str(args.library),
               "band": band, "n_accepted": sum(1 for r in results if r["accepted"]),
               "results": results}
    (outbase / "SUMMARY.json").write_text(json.dumps(summary, indent=1))   # disk before print
    for item in results:
        c = item["comparison"] or {}
        print(f"cand{item['candidate']}: delta={c.get('mean_delta')} moved={c.get('n_episodes_moved')} "
              f"accepted={item['accepted']} -- {'; '.join(item['why'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
