#!/usr/bin/env python3
"""Did a family donate an operator it derived from another family's episodes?

The held-out column only means anything if the library a fold hands the planner was built
without that family. The build guarantees that by family: an operator enters fold F's
library only when some family other than F submitted it. But the workbench does not fence
the agent in -- `list_episodes` with no family returns the whole induction half, and cells
have been observed reading episodes of families they were not seeded on. If family A's cell
derived its operator from family B's traces, then B's fold is holding an operator that was
read off B's own data, and the column is leaking.

This reads every accepted operator's donor cells, recovers the episode indices those
transcripts actually opened, and reports the families behind them. Only what a cell LOOKED
AT is counted -- the acceptance pool (holdout, coverage) is family-fenced by construction.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import viki_fork_guard  # noqa: E402

viki_fork_guard.install()

from our_method.skill_memory_v2.build import load_episodes  # noqa: E402

TRAIN = "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet"
INDEX = re.compile(r'"index"\s*:\s*(\d+)')


def signature(operator):
    """The same identity the assembler dedups on: effect plus abstracted body."""
    effect = operator.get("effect") or {}
    if operator.get("coordinated"):
        body = tuple((item.get("action") or [None])[0]
                     for role in operator.get("roles", [])
                     for item in role.get("actions", []))
    else:
        body = tuple(tuple(action) for action in operator.get("body", []))
    return (effect.get("key"), effect.get("subject"), str(effect.get("value")),
            bool(operator.get("coordinated")), body)


def donor_of(source: Path) -> str:
    """`outputs/agentic_rung/v2_<family>/<cell>/verdict.json` or `.../v3/<family>/<cell>/`."""
    parts = source.parts
    return parts[2][3:] if parts[2].startswith("v2_") else parts[3]


def main() -> int:
    episodes = load_episodes(TRAIN)[::2]
    family_of = [e.get("task_name") if isinstance(e, dict) else None for e in episodes]

    # Donors are counted from the per-family library files, not from the union: the union
    # keeps one copy of a deduplicated operator and with it only that copy's provenance, so
    # reading `memory_all` under-reports who donated what. Fold survival is decided by which
    # FAMILY LIBRARIES hold the operator, which is exactly what is counted here.
    libraries = {}
    for path in sorted((ROOT / "outputs/v3_libraries").glob("library_*.json")):
        family = path.stem[len("library_"):]
        for operator in json.loads(path.read_text())["operators"]:
            libraries.setdefault(signature(operator), {})[family] = operator

    memory = json.loads((ROOT / "outputs/v3_memories/memory_all.json").read_text())
    report = []
    for operator in memory["layer1"]["operators"]:
        shape = ("COORD[%d] " % len(operator.get("roles", [])) +
                 " | ".join(" ".join(i["action"][0] for i in r["actions"])
                            for r in operator.get("roles", []))
                 if operator.get("coordinated")
                 else " ".join(a[0] for a in (operator.get("body") or [])))
        donated = libraries.get(signature(operator), {})
        sources = []
        for family, copy in sorted(donated.items()):
            sources.extend((copy.get("provenance") or {}).get("sources") or [])
        if not sources:
            sources = (operator.get("provenance") or {}).get("sources") or []
        entry = {"shape": shape, "support": operator.get("support"), "cells": []}
        for source in sources:
            path = Path(source)
            donor = donor_of(path)
            looked = Counter()
            transcript = path.parent / "transcript.json"
            if transcript.is_file():
                for record in json.loads(transcript.read_text()):
                    for match in INDEX.finditer(record.get("answer", "") or ""):
                        index = int(match.group(1))
                        if 0 <= index < len(family_of) and family_of[index]:
                            looked[family_of[index]] += 1
            entry["cells"].append({"donor": donor, "cell": path.parent.name,
                                   "looked_at": dict(looked),
                                   "outside": sorted(set(looked) - {donor})})
        entry["donors"] = sorted({c["donor"] for c in entry["cells"]})
        report.append(entry)

    out = ROOT / "results/viki_donor_leak_audit.json"
    out.write_text(json.dumps(report, indent=1))

    print("%-52s %5s %s" % ("operator", "sup", "donors"))
    for entry in report:
        print("%-52s %5s %s" % (entry["shape"][:52], entry["support"], entry["donors"]))
        for cell in entry["cells"]:
            mark = "LOOKED OUTSIDE" if cell["outside"] else "own family only"
            print("    %-40s %-16s %s" % (cell["donor"][:40], cell["cell"], mark))
            if cell["outside"]:
                print("        %s" % ", ".join(cell["outside"]))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
