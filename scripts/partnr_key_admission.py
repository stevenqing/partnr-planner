#!/usr/bin/env python3
"""Per-predicate admission: was THIS key's proposition satisfied, not did the score rise.

The execution gate scores a candidate on `task_percent_complete` over a pool, and for
`is_next_to` that could not tell a correct operator from one broken on purpose: both gained
the same, because the gain came from the menu naming the predicate at all
(HANDOVER-2026-09-13). An operator for a new family therefore has to be admitted on the
predicate it claims -- the fraction of that key's propositions the run actually satisfied.

The evaluator already says which propositions were missed. Its `task_explanation` (last
planner-log step) lists "Missing steps" per proposition, and, when everything held at some
point, the constraints that broke instead. So per episode and per key:

    total     propositions of that key in the episode
    missing   lines of that key under "Missing steps"
    satisfied total - missing                       (an episode with no explanation is all-satisfied)

`--compare A:B` pairs two cells episode by episode on each key's satisfied fraction and
bootstraps the difference, which is the admission reading for a candidate against its base.

    python scripts/partnr_key_admission.py --pool gate_nxt \\
        --cell base=<dir> --cell cand0=<dir> --compare cand0:base --json <out>
"""
from __future__ import annotations

import argparse, glob, gzip, json, os, random, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
DATA = ROOT / "data/datasets/partnr_episodes/v0_0"

# How the evaluator words a proposition that was never satisfied, per key.
MISSING = {
    "is_on_top": re.compile(r"should have been placed on top of"),
    "is_inside": re.compile(r"should have been placed inside"),
    "is_in_room": re.compile(r"should have been moved to the"),
    "is_on_floor": re.compile(r"should have been (?:placed )?on the floor"),
    "is_next_to": re.compile(r"should have been placed next to"),
    "is_clean": re.compile(r"should have been clean\b"),
    "is_dirty": re.compile(r"should have been dirty\b"),
    "is_filled": re.compile(r"should have been filled\b"),
    "is_empty": re.compile(r"should have been empty\b"),
    "is_powered_on": re.compile(r"should have been powered on\b"),
    "is_powered_off": re.compile(r"should have been powered off\b"),
}
BROKEN = re.compile(r"constraints were broken", re.I)


def episode_keys(pool: str):
    with gzip.open(DATA / f"{pool}.json.gz", "rt") as handle:
        episodes = json.load(handle)["episodes"]
    out = {}
    for episode in episodes:
        counts = defaultdict(int)
        for proposition in episode.get("evaluation_propositions") or []:
            key = proposition.get("function_name")
            if key in MISSING:
                counts[key] += 1
        out[str(episode["episode_id"])] = dict(counts)
    return out


def explanation(cell: str, pool: str, episode: str):
    """The evaluator's last explanation for this episode, '' when it scored full marks."""
    path = f"{cell}/results/{pool}.json.gz/planner-log/planner-log-episode_{episode}_0.json"
    if not os.path.exists(path):
        return None
    try:
        steps = json.load(open(path)).get("steps") or []
    except Exception:
        return None
    return str(((steps[-1] if steps else {}).get("stats") or {}).get("task_explanation") or "")


def read_cell(cell: str, pool: str, wanted):
    """{episode: {key: (total, satisfied)}} plus how many episodes had no log at all."""
    per = {}
    absent = 0
    for episode, counts in wanted.items():
        text = explanation(cell, pool, episode)
        if text is None:
            absent += 1
            continue
        broken = bool(BROKEN.search(text))
        row = {}
        for key, total in counts.items():
            missing = len(MISSING[key].findall(text))
            row[key] = (total, max(0, total - min(missing, total)), broken)
        per[episode] = row
    return per, absent


def boot(diffs, n=10000, seed=20260916):
    rng = random.Random(seed)
    m = len(diffs)
    if not m:
        return 0.0, 0.0
    means = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="pool dataset name, e.g. gate_nxt")
    ap.add_argument("--cell", action="append", required=True, help="name=path")
    ap.add_argument("--compare", nargs="*", default=[], help="a:b, delta is a - b")
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()

    wanted = episode_keys(args.pool)
    cells = dict(item.split("=", 1) for item in args.cell)
    report = {"pool": args.pool, "episodes": len(wanted), "cells": {}, "comparisons": {}}
    data = {}
    for name, path in cells.items():
        per, absent = read_cell(path, args.pool, wanted)
        data[name] = per
        keys = defaultdict(lambda: [0, 0, 0, 0])  # props, satisfied, episodes, broken episodes
        for row in per.values():
            for key, (total, satisfied, broken) in row.items():
                acc = keys[key]
                acc[0] += total
                acc[1] += satisfied
                acc[2] += 1
                acc[3] += int(broken)
        report["cells"][name] = {
            "path": path, "episodes_read": len(per), "episodes_without_log": absent,
            "keys": {k: {"props": v[0], "satisfied": v[1],
                         "rate": round(v[1] / v[0], 4) if v[0] else None,
                         "episodes": v[2], "constraint_broken_episodes": v[3]}
                     for k, v in sorted(keys.items())}}
    for spec in args.compare:
        a, b = spec.split(":")
        comp = {}
        for key in MISSING:
            shared = [e for e in data[a] if e in data[b] and key in data[a][e] and key in data[b][e]]
            if not shared:
                continue
            fa = [data[a][e][key][1] / data[a][e][key][0] for e in shared]
            fb = [data[b][e][key][1] / data[b][e][key][0] for e in shared]
            diffs = [x - y for x, y in zip(fa, fb)]
            lo, hi = boot(diffs)
            comp[key] = {"episodes": len(shared), "a": sum(fa) / len(fa), "b": sum(fb) / len(fb),
                         "delta": sum(diffs) / len(diffs), "ci95": [lo, hi],
                         "up": sum(d > 1e-9 for d in diffs), "down": sum(d < -1e-9 for d in diffs)}
        report["comparisons"][spec] = comp
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1))  # disk before print

    for name, cell in report["cells"].items():
        print(f"== {name}  read {cell['episodes_read']}/{report['episodes']}"
              f"  no log {cell['episodes_without_log']}")
        for key, v in cell["keys"].items():
            print(f"   {key:16s} props {v['props']:5d}  satisfied {v['satisfied']:5d}"
                  f"  rate {v['rate']}  episodes {v['episodes']:4d}  constraint-broken {v['constraint_broken_episodes']}")
    for spec, comp in report["comparisons"].items():
        print(f"== {spec}")
        for key, v in comp.items():
            print(f"   {key:16s} n={v['episodes']:4d}  {v['b']:.3f} -> {v['a']:.3f}"
                  f"  d={v['delta']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]  +{v['up']} -{v['down']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
