#!/usr/bin/env python3
"""Paired comparisons of PARTNR cells on one episode pool, split the way the stage work reads.

Every cell is checked first (all pool episodes present); an incomplete comparison is written
with `complete: false` and its numbers are not a result. Groups: all, temporal / control (from
the pool file when it has them), each composite type, and spatial (R_S + R_S_T). Metrics are
percent_complete and state_success, a crashed episode counted as 0, paired bootstrap 10000.

    python scripts/partnr_pair_cells.py --pool <episodes.json> --split train_mini \\
        --cell a=<dir> --cell b=<dir> --compare a:b --json <out>
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT / "scripts"))

from partnr_task_types import classify  # noqa: E402


def load(cell: str, split: str):
    out, crashed = {}, []
    for f in glob.glob(f"{cell}/results/{split}.json.gz/stats/*.json"):
        episode = os.path.basename(f)[:-5]
        stats = json.load(open(f)).get("stats")
        if stats is None:
            crashed.append(episode)
            out[episode] = (0.0, 0.0)
            continue
        stats = json.loads(stats) if isinstance(stats, str) else stats
        out[episode] = (float(stats.get("task_percent_complete", 0)), float(stats.get("task_state_success", 0)))
    return out, crashed


def boot(diffs, n=10000, seed=20260916):
    rng = random.Random(seed)
    m = len(diffs)
    means = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=None, help="episodes.json with ids (and temporal/control)")
    ap.add_argument("--split", default="train_mini")
    ap.add_argument("--cell", action="append", required=True, help="name=path")
    ap.add_argument("--compare", nargs="+", required=True, help="a:b, delta is a - b")
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()

    with gzip.open(ROOT / "data/datasets/partnr_episodes/v0_0" / f"{args.split}.json.gz") as handle:
        episodes = json.load(handle)["episodes"]
    types = {str(e["episode_id"]): classify(e) for e in episodes}
    pool = json.load(open(args.pool)) if args.pool else {"ids": sorted(types, key=int)}
    ids = [str(i) for i in pool["ids"]]
    cells = dict(item.split("=", 1) for item in args.cell)
    data, meta = {}, {}
    for name, path in cells.items():
        d, crashed = load(path, args.split)
        cell_json = {}
        if os.path.exists(f"{path}/CELL.json"):
            cell_json = json.load(open(f"{path}/CELL.json"))
        meta[name] = {"path": path, "present": len(set(ids) & set(d)), "wanted": len(ids),
                      "crashed": sorted(crashed), "done_file": os.path.exists(f"{path}/DONE"), "cell_json": cell_json}
        data[name] = d

    groups = {"all": ids}
    if "temporal" in pool:
        groups["temporal"] = [str(i) for i in pool["temporal"]]
        groups["control"] = [str(i) for i in pool["control"]]
    for t in ("R", "R_S", "R_T", "R_S_T", "H_R"):
        groups[t] = [i for i in ids if types[i] == t]
    groups["spatial"] = [i for i in ids if "S" in types[i].split("_") and "H" not in types[i].split("_")]

    report = {"pool": str(args.pool), "split": args.split, "cells": meta, "comparisons": {}}
    for spec in args.compare:
        a, b = spec.split(":")
        complete = meta[a]["present"] == len(ids) and meta[b]["present"] == len(ids)
        comp = {"complete": complete}
        for group, members in groups.items():
            members = [i for i in members if i in data[a] and i in data[b]]
            if not members:
                continue
            for k, metric in ((0, "pc"), (1, "ss")):
                diffs = [data[a][i][k] - data[b][i][k] for i in members]
                lo, hi = boot(diffs)
                comp[f"{group}/{metric}"] = {
                    "n": len(members),
                    "a": sum(data[a][i][k] for i in members) / len(members),
                    "b": sum(data[b][i][k] for i in members) / len(members),
                    "delta": sum(diffs) / len(diffs), "ci95": [lo, hi],
                    "up": sum(d > 1e-9 for d in diffs), "down": sum(d < -1e-9 for d in diffs)}
        report["comparisons"][spec] = comp
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1))  # disk before print

    for name, m in meta.items():
        print(f"cell {name}: {m['present']}/{m['wanted']} crashed={len(m['crashed'])} DONE={m['done_file']}")
    for spec, comp in report["comparisons"].items():
        print(f"== {spec}  complete={comp['complete']}")
        for key, x in comp.items():
            if key == "complete":
                continue
            print("  %-16s n=%3d %.3f -> %.3f  d=%+.3f [%+.3f,%+.3f]  +%d -%d" % (
                key, x["n"], x["b"], x["a"], x["delta"], x["ci95"][0], x["ci95"][1], x["up"], x["down"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
