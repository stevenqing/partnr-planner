#!/usr/bin/env python3
"""Paired report for the train_mini sim pairing of the requirement interfaces.

Every comparison is over the pool's own episode list (`episodes.json`), episode by episode,
for both metrics -- percent_complete is continuous, so the interval is a paired bootstrap, not
McNemar. Two readings are given because they disagree when crash rates differ: a crashed
episode (stats without a score) counted as zero, and only the episodes both arms finished.
A comparison whose cells are not both complete says so first; its numbers are not a result.

  python scripts/partnr_typed_simpair_report.py \\
      --pair outputs/cand_iface_0914/simpair_train_mini \\
      --compare typed_v7b_7b:intent_7b typed_7b:intent_7b typed_v7b_7b:typed_7b \\
      --json outputs/cand_iface_0914/simpair_train_mini/report.json
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT / "scripts"))
METRICS = ("task_percent_complete", "task_state_success")


def load_cell(cell: Path, pool: str):
    scores, crashed, zero_step = {}, set(), set()
    for path in glob.glob(str(cell / "results" / f"{pool}.json.gz" / "stats" / "*.json")):
        episode = os.path.basename(path)[:-5]
        blob = json.load(open(path))
        stats = blob.get("stats")
        if stats is None:
            crashed.add(episode)
            continue
        stats = json.loads(stats) if isinstance(stats, str) else stats
        scores[episode] = {m: float(stats.get(m, 0.0)) for m in METRICS}
        if stats.get("sim_step_count", 1) == 0:
            zero_step.add(episode)
    return scores, crashed, zero_step


def paired(a, b, ids, boot, seed):
    diffs = [a[i] - b[i] for i in ids]
    n = len(diffs)
    if not n:
        return {"n": 0}
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(boot))
    return {
        "n": n,
        "a": sum(a[i] for i in ids) / n,
        "b": sum(b[i] for i in ids) / n,
        "delta": sum(diffs) / n,
        "ci95": [means[int(0.025 * boot)], means[int(0.975 * boot) - 1]],
        "better": sum(d > 1e-9 for d in diffs),
        "worse": sum(d < -1e-9 for d in diffs),
        "same": sum(abs(d) <= 1e-9 for d in diffs),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", type=Path, required=True)
    ap.add_argument("--pool", default="train_mini")
    ap.add_argument("--compare", nargs="+", required=True, help="armA:armB, delta is A - B")
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    from partnr_task_types import classify

    ids = [str(i) for i in json.load(open(args.pair / "episodes.json"))["ids"]]
    with gzip.open(ROOT / "data/datasets/partnr_episodes/v0_0" / f"{args.pool}.json.gz") as handle:
        kind = {str(e["episode_id"]): classify(e) for e in json.load(handle)["episodes"]}

    cells = {}
    for spec in args.compare:
        for arm in spec.split(":"):
            if arm not in cells:
                scores, crashed, zero = load_cell(args.pair / arm, args.pool)
                cell_json = args.pair / arm / "CELL.json"
                cells[arm] = {
                    "scores": scores, "crashed": crashed, "zero_step": zero,
                    "missing": [i for i in ids if i not in scores and i not in crashed],
                    "cell": json.load(open(cell_json)) if cell_json.exists() else None,
                }

    report = {"pool": args.pool, "episodes": len(ids), "cells": {}, "comparisons": []}
    for arm, c in cells.items():
        report["cells"][arm] = {
            "finished": sum(1 for i in ids if i in c["scores"]), "crashed": sorted(c["crashed"] & set(ids)),
            "missing": len(c["missing"]), "zero_step": len(c["zero_step"] & set(ids)),
            "cell_json": c["cell"],
        }

    for spec in args.compare:
        name_a, name_b = spec.split(":")
        A, B = cells[name_a], cells[name_b]
        complete = not A["missing"] and not B["missing"]
        entry = {"a": name_a, "b": name_b, "complete": complete, "readings": {}}
        both = [i for i in ids if i in A["scores"] and i in B["scores"]]
        for reading, pool_ids in (("crash_as_zero", ids), ("both_finished", both)):
            entry["readings"][reading] = {}
            for metric in METRICS:
                a = {i: A["scores"].get(i, {}).get(metric, 0.0) for i in pool_ids}
                b = {i: B["scores"].get(i, {}).get(metric, 0.0) for i in pool_ids}
                entry["readings"][reading][metric] = paired(a, b, pool_ids, args.boot, args.seed)
        by_type = defaultdict(list)
        for i in ids:
            by_type[kind.get(i, "?")].append(i)
        entry["by_type"] = {}
        for k, members in sorted(by_type.items()):
            a = {i: A["scores"].get(i, {}).get("task_percent_complete", 0.0) for i in members}
            b = {i: B["scores"].get(i, {}).get("task_percent_complete", 0.0) for i in members}
            entry["by_type"][k] = paired(a, b, members, 2000, args.seed)
        report["comparisons"].append(entry)

    args.json.write_text(json.dumps(report, indent=1))  # disk before print

    print(f"pool {args.pool}: {len(ids)} episodes")
    for arm, c in report["cells"].items():
        print(f"  {arm:14s} finished {c['finished']:3d}  crashed {len(c['crashed'])}  missing {c['missing']}"
              f"  zero-step {c['zero_step']}")
    for entry in report["comparisons"]:
        flag = "" if entry["complete"] else "   ** INCOMPLETE: not a result **"
        print(f"\n{entry['a']} - {entry['b']}{flag}")
        for reading, metrics in entry["readings"].items():
            for metric, r in metrics.items():
                if not r.get("n"):
                    continue
                print(f"  {reading:13s} {metric:22s} n={r['n']:3d}  {r['a']:.3f} vs {r['b']:.3f}"
                      f"  delta {r['delta']:+.3f} CI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]"
                      f"  better {r['better']} worse {r['worse']} same {r['same']}")
        print("  by type (percent_complete, crash as zero):")
        for k, r in entry["by_type"].items():
            print(f"    {k:8s} n={r['n']:3d}  {r['a']:.3f} vs {r['b']:.3f}  delta {r['delta']:+.3f}"
                  f" CI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
