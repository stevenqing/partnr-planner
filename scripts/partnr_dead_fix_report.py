#!/usr/bin/env python3
"""Did the explore fallback revive the 70 episodes that died at step 0?

Paired, on exactly those 70 episodes: the archived `priv:v2_memory_R` cell is the `before`
(every one of them scored 0.0 there and ran a single step), the `dead_fix` cell is the
`after`. Two things are reported and they answer different questions:

  dead-at-step-0   whether the mechanism is gone. This is the one the fix is aimed at, and
                   it is a count, not a mean -- a revived episode that still scores badly
                   is a different problem than one that never started.
  percent_complete what that is worth. `task_percent_complete` is continuous, so the
                   paired test is Wilcoxon, never McNemar.

The JSON is written before anything is printed. A report that crashes after the tables and
before the dump leaves nothing behind, which has happened here once already.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Dict

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT / "scripts"))

from partnr_compositional_report import METRIC, episode_types  # noqa: E402
from partnr_v2_report import read_cell  # noqa: E402

SPLIT = "val_mini"
LIBRARY = ROOT / "results/partnr_operators.json"
BEFORE = ROOT / "outputs/sweep_remeasured" / SPLIT / "v2_memory_R"
AFTER = ROOT / "outputs/dead_fix" / SPLIT / "v2_memory_R_deadonly"
DEAD = ROOT / "outputs/partnr_dead_sets.json"
OUT = ROOT / "outputs/partnr_dead_fix.json"


def coverable(split: str) -> Dict[str, bool]:
    """Whether the operator library can ground *anything* this episode asks for.

    The rearrange-only library holds `is_on_top` and `is_inside` and nothing else, so an
    episode made entirely of `is_in_room` has no groundable requirement in it and scores
    zero however healthy the planner is. Reviving such an episode buys nothing, and
    counting it as recoverable would overstate what the step-0 bug cost. `is_next_to` is
    not coverage on its own: it has no operator either, and reaches the world only by
    folding onto a placement that some other requirement already carries.
    """
    keys = {o["effect"]["key"] for o in json.loads(LIBRARY.read_text())["operators"]}
    with gzip.open(ROOT / "data/datasets/partnr_episodes/v0_0" / f"{split}.json.gz") as handle:
        episodes = json.load(handle)["episodes"]
    return {
        str(e["episode_id"]): any(
            p["function_name"] in keys for p in (e.get("evaluation_propositions") or [])
        )
        for e in episodes
    }


def step0_deaths(cell: Path) -> set:
    """Episodes whose entire log is one step in which every agent said `Done`."""
    logs = cell / "results" / f"{SPLIT}.json.gz" / "planner-log"
    dead = set()
    for path in logs.glob("planner-log-episode_*_0.json"):
        steps = json.loads(path.read_text()).get("steps", [])
        if len(steps) == 1 and all(
            action[0] == "Done"
            for action in (steps[0].get("high_level_actions") or {}).values()
        ):
            dead.add(path.name.split("_")[1])
    return dead


def wilcoxon(pairs):
    """Two-sided signed-rank on the non-zero differences, normal approximation.

    Written out rather than imported because scipy is not in this venv, and the sample is
    large enough (n > 20 after dropping ties) for the approximation to be the right call.
    """
    diffs = [after - before for before, after in pairs if after != before]
    n = len(diffs)
    if n == 0:
        return {"n_nonzero": 0, "statistic": None, "z": None, "p": None}
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    w_plus = sum(r for d, r in zip(diffs, ranks) if d > 0)
    w_minus = sum(r for d, r in zip(diffs, ranks) if d < 0)
    statistic = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    sd = (n * (n + 1) * (2 * n + 1) / 24.0) ** 0.5
    z = (statistic - mean) / sd if sd else 0.0
    # Two-sided normal tail without scipy.
    import math

    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"n_nonzero": n, "w_plus": w_plus, "w_minus": w_minus,
            "statistic": statistic, "z": z, "p": p}


def main() -> int:
    targets = json.load(open(DEAD))["v2_memory_R"]
    before_all = {k: v[METRIC] for k, v in read_cell(BEFORE, f"{SPLIT}.json.gz").items()}
    after_all = {k: v[METRIC] for k, v in read_cell(AFTER, f"{SPLIT}.json.gz").items()}

    scored = [e for e in targets if e in before_all and e in after_all]
    missing = [e for e in targets if e not in after_all]
    pairs = [(before_all[e], after_all[e]) for e in scored]

    still_dead = step0_deaths(AFTER) & set(targets)
    types = episode_types(SPLIT)
    can_ground = coverable(SPLIT)
    covered = [e for e in scored if can_ground.get(e)]
    uncovered = [e for e in scored if not can_ground.get(e)]
    by_type = {}
    for episode in scored:
        bucket = by_type.setdefault(types.get(episode, "?"), {"n": 0, "before": 0.0, "after": 0.0, "revived": 0})
        bucket["n"] += 1
        bucket["before"] += before_all[episode]
        bucket["after"] += after_all[episode]
        bucket["revived"] += int(episode not in still_dead)

    report = {
        "targets": len(targets),
        "scored": len(scored),
        "missing_from_after": missing,
        "still_dead_at_step_0": sorted(still_dead, key=int),
        "revived": len(scored) - len(still_dead & set(scored)),
        "before_mean": sum(b for b, _ in pairs) / len(pairs) if pairs else None,
        "after_mean": sum(a for _, a in pairs) / len(pairs) if pairs else None,
        "improved": sum(1 for b, a in pairs if a > b),
        "worsened": sum(1 for b, a in pairs if a < b),
        "unchanged": sum(1 for b, a in pairs if a == b),
        "wilcoxon": wilcoxon(pairs),
        # The split that decides what the bug actually cost.
        "coverable": {
            "n": len(covered),
            "episodes": sorted(covered, key=int),
            "before_mean": (sum(before_all[e] for e in covered) / len(covered)) if covered else None,
            "after_mean": (sum(after_all[e] for e in covered) / len(covered)) if covered else None,
            "wilcoxon": wilcoxon([(before_all[e], after_all[e]) for e in covered]),
        },
        "uncoverable": {
            "n": len(uncovered),
            "episodes": sorted(uncovered, key=int),
            "after_mean": (sum(after_all[e] for e in uncovered) / len(uncovered)) if uncovered else None,
            "note": "no requirement in these episodes has an operator in the library; zero is the correct score",
        },
        "by_type": {
            k: {"n": v["n"], "revived": v["revived"],
                "before": v["before"] / v["n"], "after": v["after"] / v["n"]}
            for k, v in sorted(by_type.items())
        },
        "per_episode": {e: {"before": before_all[e], "after": after_all[e]} for e in scored},
    }

    OUT.write_text(json.dumps(report, indent=1))

    print(f"target episodes      {len(targets)}   scored in both cells {len(scored)}")
    if missing:
        print(f"missing from after   {missing}")
    print(f"still dead at step 0 {len(still_dead)}   revived {report['revived']}")
    print(f"{METRIC}   before {report['before_mean']:.4f}  ->  after {report['after_mean']:.4f}")
    print(f"improved {report['improved']}  worsened {report['worsened']}  unchanged {report['unchanged']}")
    w = report["wilcoxon"]
    if w["p"] is not None:
        print(f"Wilcoxon signed-rank  n={w['n_nonzero']}  W={w['statistic']:.1f}  z={w['z']:.3f}  p={w['p']:.4g}")
    cov, unc = report["coverable"], report["uncoverable"]
    print()
    print(f"library can ground something in {cov['n']} of these episodes; "
          f"the other {unc['n']} ask only for effects it does not have")
    if cov["n"]:
        print(f"  coverable    before {cov['before_mean']:.4f}  ->  after {cov['after_mean']:.4f}"
              f"   (p={cov['wilcoxon']['p']:.4g})" if cov["wilcoxon"]["p"] is not None
              else f"  coverable    before {cov['before_mean']:.4f}  ->  after {cov['after_mean']:.4f}")
    if unc["n"]:
        print(f"  uncoverable  after {unc['after_mean']:.4f}  (zero is the right answer here)")
    print()
    print("%-8s %4s %8s %9s %8s" % ("type", "n", "revived", "before", "after"))
    for name, row in report["by_type"].items():
        print("%-8s %4d %8d %9.4f %8.4f" % (name, row["n"], row["revived"], row["before"], row["after"]))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
