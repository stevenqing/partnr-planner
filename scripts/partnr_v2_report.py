#!/usr/bin/env python3
"""Read a sweep's per-episode stats and say what they support, task type by task type.

PARTNR scores an episode on two things: `task_percent_complete`, which is continuous, and
`task_state_success`, which is the all-or-nothing version of it. The continuous one is the
headline here, and it is why the significance test is a paired Wilcoxon over episodes
rather than McNemar -- McNemar is for the binary outcomes the VIKI-L2 work reported, and
using it on a continuous measure would be wrong in the direction that flatters us.

Every comparison is paired on episode id against the ceiling cell, and episodes missing
from either side are dropped from that comparison and counted, so a cell that crashed on
some episodes cannot quietly improve its own average.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

METRICS = ("task_percent_complete", "task_state_success")


def read_cell(root: Path, dataset: str) -> Dict[str, Dict[str, float]]:
    """One cell's per-episode numbers, keyed by episode id."""
    out: Dict[str, Dict[str, float]] = {}
    folder = root / "results" / dataset / "stats"
    for path in sorted(folder.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except Exception:
            continue
        if not record.get("success"):
            continue  # an episode that raised is absent, not zero: it is reported apart
        try:
            stats = json.loads(record["stats"])
        except Exception:
            continue
        out[path.stem] = {key: float(stats.get(key, 0.0)) for key in METRICS}
    return out


def crashed(root: Path, dataset: str) -> List[str]:
    folder = root / "results" / dataset / "stats"
    failed = []
    for path in sorted(folder.glob("*.json")):
        try:
            if not json.loads(path.read_text()).get("success"):
                failed.append(path.stem)
        except Exception:
            failed.append(path.stem)
    return failed


def wilcoxon(pairs: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Paired signed-rank statistic and a normal-approximation two-sided p.

    Written out rather than imported because the evaluation environment has no scipy, and
    because the tie handling is worth being able to read: zero differences are dropped,
    equal magnitudes share an average rank, and the variance correction for ties is
    applied. With fewer than about ten non-zero differences the normal approximation is
    not trustworthy and None is returned instead of a number that looks like evidence.
    """
    differences = [b - a for a, b in pairs if b != a]
    n = len(differences)
    if n < 10:
        return None
    order = sorted(range(n), key=lambda i: abs(differences[i]))
    ranks = [0.0] * n
    tie_correction = 0.0
    position = 0
    while position < n:
        end = position
        while end + 1 < n and abs(differences[order[end + 1]]) == abs(differences[order[position]]):
            end += 1
        average = (position + end) / 2.0 + 1.0
        size = end - position + 1
        for index in range(position, end + 1):
            ranks[order[index]] = average
        tie_correction += size ** 3 - size
        position = end + 1
    positive = sum(rank for rank, difference in zip(ranks, differences) if difference > 0)
    negative = sum(rank for rank, difference in zip(ranks, differences) if difference < 0)
    statistic = min(positive, negative)
    mean = n * (n + 1) / 4.0
    variance = (n * (n + 1) * (2 * n + 1) - tie_correction / 2.0) / 24.0
    if variance <= 0:
        return None
    z = (statistic - mean) / math.sqrt(variance)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return statistic, p


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="outputs/sweep")
    parser.add_argument("--dataset", default="val_mini.json.gz")
    parser.add_argument("--split", default="val_mini")
    parser.add_argument("--types", default="results/partnr_task_types.json")
    parser.add_argument("--baseline", default="ceiling", help="cell every other cell is paired against")
    parser.add_argument("--cells", nargs="*", default=None)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    sweep = Path(arguments.sweep)
    names = arguments.cells or sorted(
        path.name for path in sweep.iterdir() if (path / "results").is_dir()
    )
    cells = {name: read_cell(sweep / name, arguments.dataset) for name in names}

    table = json.loads(Path(arguments.types).read_text()).get(arguments.split, {})
    kind_of = {str(episode): kind for kind, episodes in table.items() for episode in episodes}
    kinds = sorted({kind_of.get(episode, "?") for cell in cells.values() for episode in cell})

    report: Dict[str, Any] = {"cells": {}, "by_type": {}}
    print(f"{'cell':<16}{'n':>5}{'crashed':>9}{'percent_complete':>19}{'state_success':>16}")
    for name in names:
        cell = cells[name]
        failures = crashed(sweep / name, arguments.dataset)
        report["cells"][name] = {
            "episodes": len(cell),
            "crashed": failures,
            **{key: mean([row[key] for row in cell.values()]) for key in METRICS},
        }
        print(
            f"{name:<16}{len(cell):>5}{len(failures):>9}"
            f"{report['cells'][name]['task_percent_complete']:>19.4f}"
            f"{report['cells'][name]['task_state_success']:>16.4f}"
        )

    reference = cells.get(arguments.baseline, {})
    for kind in kinds:
        episodes = [episode for episode in kind_of if kind_of[episode] == kind]
        print(f"\n{kind}  ({len(episodes)} episodes in {arguments.split})")
        row: Dict[str, Any] = {}
        for name in names:
            cell = cells[name]
            shared = [e for e in episodes if e in cell]
            if not shared:
                continue
            entry = {
                "n": len(shared),
                **{key: mean([cell[e][key] for e in shared]) for key in METRICS},
            }
            if name != arguments.baseline and reference:
                paired = [e for e in shared if e in reference]
                entry["paired_n"] = len(paired)
                entry["unpaired"] = len(shared) - len(paired)
                if paired:
                    entry["delta_percent_complete"] = mean(
                        [cell[e]["task_percent_complete"] for e in paired]
                    ) - mean([reference[e]["task_percent_complete"] for e in paired])
                    test = wilcoxon(
                        [
                            (reference[e]["task_percent_complete"], cell[e]["task_percent_complete"])
                            for e in paired
                        ]
                    )
                    if test is not None:
                        entry["wilcoxon_W"], entry["wilcoxon_p"] = test
            row[name] = entry
            delta = entry.get("delta_percent_complete")
            p = entry.get("wilcoxon_p")
            suffix = ""
            if delta is not None:
                suffix = f"   vs {arguments.baseline} {delta:+.4f}"
                suffix += f"  p={p:.4g}" if p is not None else "  p=n/a (too few non-ties)"
            print(
                f"  {name:<16}{entry['n']:>5}"
                f"{entry['task_percent_complete']:>19.4f}"
                f"{entry['task_state_success']:>16.4f}{suffix}"
            )
        report["by_type"][kind] = row

    if arguments.out:
        Path(arguments.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {arguments.out}")


if __name__ == "__main__":
    main()
