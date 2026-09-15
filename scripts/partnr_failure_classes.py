#!/usr/bin/env python3
"""Why each episode failed, from the benchmark's own explanation and the planner's notes.

The evaluator already writes a `task_explanation` into the last planner-log step: which
propositions were never satisfied ("should have been placed on top of", "placed next to",
"moved to", "should have been clean"), or, when every step held at some point, which
constraints broke ("Steps were completed out of order", "Completed steps were later undone",
"... should have been completed after:"). The planner's trace adds what it gave up on
(`abandoned=N`, "never saw X", "no operator for ...").

Each failed episode (state_success 0) gets one primary class, first match wins:

  crashed        no score
  zero_step      simulator never stepped
  order          every step held once, but the temporal constraint was violated
  undone         every step held once, but a completed step was later undone
  state_missing  only heterogeneous (clean / filled / powered) steps missing -- no operator exists
  place_missing  a placement / room move never held
  beside_missing only "next to" steps missing
  other

plus flags that are not exclusive (abandoned, never_saw, no_operator, order_flag, undone_flag).

    python scripts/partnr_failure_classes.py --pool <episodes.json> \\
        --cell stage=<cell dir> --cell priv_partial=<cell dir> ... --json <out>
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT / "scripts"))

from partnr_task_types import classify  # noqa: E402

PLACE = re.compile(r'should have been (?:placed on top of|placed inside|moved to)')
BESIDE = re.compile(r'should have been placed next to')
STATE = re.compile(r'should have been (?:clean|dirty|filled|empty|powered on|powered off)')
ORDER = re.compile(r'out of order|should have been completed after')
UNDONE = re.compile(r'later undone')


def cell_episode(cell: Path, split: str, episode: str):
    base = cell / "results" / f"{split}.json.gz"
    stats_path = base / "stats" / f"{episode}.json"
    if not stats_path.exists():
        return None
    blob = json.load(open(stats_path))
    stats = blob.get("stats")
    stats = (json.loads(stats) if isinstance(stats, str) else stats) if stats is not None else None
    explanation = ""
    log_path = base / "planner-log" / f"planner-log-episode_{episode}_0.json"
    if log_path.exists():
        try:
            steps = json.load(open(log_path)).get("steps") or []
            explanation = str(((steps[-1] if steps else {}).get("stats") or {}).get("task_explanation") or "")
        except Exception:
            explanation = ""
    notes = ""
    for agent in (0, 1):
        trace = base / "traces" / str(agent) / f"trace-episode_{episode}_0-{agent}.txt"
        if trace.exists():
            notes += open(trace, errors="replace").read() + "\n"
    return stats, explanation, notes


def classify_failure(stats, explanation: str, notes: str):
    flags = {
        "abandoned": sum(int(x) for x in re.findall(r"abandoned=(\d+)", notes)),
        "never_saw": len(re.findall(r"never saw", notes)),
        "no_operator": len(re.findall(r"no operator for", notes)),
        "order_flag": bool(ORDER.search(explanation)),
        "undone_flag": bool(UNDONE.search(explanation)),
        "place_lines": len(PLACE.findall(explanation)),
        "beside_lines": len(BESIDE.findall(explanation)),
        "state_lines": len(STATE.findall(explanation)),
    }
    if stats is None:
        return "crashed", flags
    if stats.get("task_state_success", 0) >= 1:
        return "success", flags
    if stats.get("sim_step_count", 1) == 0:
        return "zero_step", flags
    missing = flags["place_lines"] + flags["beside_lines"] + flags["state_lines"]
    if missing == 0 and flags["order_flag"]:
        return "order", flags
    if missing == 0 and flags["undone_flag"]:
        return "undone", flags
    if flags["state_lines"] and not flags["place_lines"] and not flags["beside_lines"]:
        return "state_missing", flags
    if flags["place_lines"]:
        return "place_missing", flags
    if flags["beside_lines"]:
        return "beside_missing", flags
    return "other", flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--split", default="train_mini")
    ap.add_argument("--cell", action="append", required=True, help="name=path")
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()

    pool = json.load(open(args.pool))
    ids = [str(i) for i in pool["ids"]]
    with gzip.open(ROOT / "data/datasets/partnr_episodes/v0_0" / f"{args.split}.json.gz") as handle:
        types = {str(e["episode_id"]): classify(e) for e in json.load(handle)["episodes"]}
    cells = dict(item.split("=", 1) for item in args.cell)

    report = {"pool": str(args.pool), "cells": {}, "episodes": defaultdict(dict)}
    for name, path in cells.items():
        by_type = defaultdict(Counter)
        flags_by_type = defaultdict(Counter)
        missing = 0
        for episode in ids:
            got = cell_episode(Path(path), args.split, episode)
            if got is None:
                missing += 1
                continue
            stats, explanation, notes = got
            label, flags = classify_failure(stats, explanation, notes)
            t = types[episode]
            for group in (t, "all", "temporal" if "T" in t.split("_") else "non_temporal"):
                by_type[group][label] += 1
                by_type[group]["episodes"] += 1
                if label not in ("success",):
                    for k, v in flags.items():
                        flags_by_type[group][k] += int(bool(v))
            report["episodes"][episode][name] = {"type": t, "label": label, "flags": flags,
                                                 "pc": (stats or {}).get("task_percent_complete"),
                                                 "explanation": explanation[:600]}
        report["cells"][name] = {"path": path, "missing": missing,
                                 "classes": {g: dict(c) for g, c in sorted(by_type.items())},
                                 "failure_flags": {g: dict(c) for g, c in sorted(flags_by_type.items())}}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1))  # disk before print

    order = ["episodes", "success", "order", "undone", "place_missing", "beside_missing", "state_missing",
             "zero_step", "crashed", "other"]
    for name, cell in report["cells"].items():
        print(f"== {name}  (missing {cell['missing']})")
        for g in ("all", "temporal", "non_temporal", "R", "R_S", "R_T", "R_S_T"):
            c = cell["classes"].get(g)
            if not c:
                continue
            print("  %-13s " % g + "  ".join(f"{k}={c.get(k, 0)}" for k in order))
            f = cell["failure_flags"].get(g, {})
            print("  %-13s   failed-episode flags: " % "" + "  ".join(f"{k}={v}" for k, v in sorted(f.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
