#!/usr/bin/env python3
"""Read the arms on the compositional axis instead of on the leaderboard.

Every memory here is built from the rearrange-only half of `train_mini`. `R` episodes are
therefore the distribution the memory was induced from, and `R_S`, `R_T`, `R_S_T` each add
a way of composing rearrangement that no memory in this comparison has seen. The question
the paper asks is not who scores highest -- that is decided as much by whether an arm
replans every step as by what it remembers -- but whose score falls off least as those
dimensions are added.

So the table that decides it is the normalized one: each arm divided by its own score on
`R`. An arm that is worse everywhere but flat across the axis generalizes compositionally;
an arm that is better everywhere but collapses does not. Absolute scores are printed too,
because the normalized view hides a floor -- an arm at 0.1 everywhere is flat and useless.

`H_R` is kept out of the compositional summary. It does not add a way of composing
rearrangement; it requires a primitive -- cleaning, switching on -- that no rearrange-only
memory contains at all. That is a vocabulary boundary, and averaging it in would blur two
different findings into one number.

## Two error bars, and they measure different things

`react` and `react_rag` turned out to be the same configuration run twice: the few-shot
instruct carries no `{rag_examples}` slot, so the retrieval arm never showed the model
anything it retrieved. That accident is the most useful calibration in the sweep, because
whatever separates those two arms is noise by construction.

  bootstrap CI   resampling episodes, paired across arms. Captures which episodes you
                 happened to evaluate on. Narrow.
  null band      the react/react_rag gap. Captures that too, plus the simulator's own
                 run-to-run stochasticity -- different crashes, different navigation.
                 Wider, and it is the one a difference has to clear.

A gap inside the null band is not a finding no matter what its p-value says.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT / "scripts"))

from partnr_task_types import classify  # noqa: E402
from partnr_v2_report import crashed, read_cell  # noqa: E402

COMPOSITIONAL = ["R_S", "R_T", "R_S_T"]
AXIS = ["R"] + COMPOSITIONAL + ["H_R"]
METRIC = "task_percent_complete"
SEED = 20260903


def episode_types(split: str) -> Dict[str, str]:
    with gzip.open(ROOT / "data/datasets/partnr_episodes/v0_0" / f"{split}.json.gz") as handle:
        return {str(e["episode_id"]): classify(e) for e in json.load(handle)["episodes"]}


def collect(sweep: Path, split: str, prefix: str = ""):
    """{arm: {episode_id: metric}} and {arm: (attempted, crashed)} for cells under `sweep`.

    Attempted and crashed are tracked apart because an arm that crashes often is scored
    on the episodes it survived. A high crash count is both a caveat on that arm's mean
    and, when the crashes are its own prompt breaking the agent, a result in itself.
    """
    arms: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Tuple[int, int]] = {}
    for cell in sorted(sweep.iterdir()):
        if not (cell / "results" / f"{split}.json.gz" / "stats").is_dir():
            continue
        values = read_cell(cell, f"{split}.json.gz")
        if not values:
            continue
        failed = crashed(cell, f"{split}.json.gz")
        arms[prefix + cell.name] = {k: v[METRIC] for k, v in values.items()}
        counts[prefix + cell.name] = (len(values) + len(failed), len(failed))
    return arms, counts


def slope(scores: Dict[str, float], buckets: Dict[str, List[str]]) -> Optional[float]:
    """Mean over the compositional types of (that type's mean / this arm's R mean)."""
    def mean(kind: str) -> Optional[float]:
        values = [scores[e] for e in buckets.get(kind, []) if e in scores]
        return sum(values) / len(values) if values else None

    base = mean("R")
    if not base:
        return None
    ratios = [value / base for value in (mean(k) for k in COMPOSITIONAL) if value is not None]
    return sum(ratios) / len(ratios) if ratios else None


def bootstrap(arms: Dict[str, Dict[str, float]], buckets: Dict[str, List[str]],
              draws: int) -> Dict[str, Tuple[float, float]]:
    """Percentile CI on each arm's slope, resampling episodes paired across arms.

    One resample is drawn per type and applied to every arm, so the interval reflects
    which episodes were evaluated rather than each arm meeting a different draw. An arm
    averages over whichever of the drawn ids it completed: intersecting eleven arms that
    each crash on a different twenty episodes would throw away most of the split to buy a
    pairing that the crashes have already broken.
    """
    rng = random.Random(SEED)
    pools = {kind: list(buckets.get(kind, [])) for kind in AXIS}
    samples: Dict[str, List[float]] = {arm: [] for arm in arms}
    for _ in range(draws):
        drawn = {kind: [rng.choice(pool) for _ in pool] if pool else []
                 for kind, pool in pools.items()}
        for arm, scores in arms.items():
            def mean(kind: str) -> Optional[float]:
                values = [scores[e] for e in drawn[kind] if e in scores]
                return sum(values) / len(values) if values else None

            base = mean("R")
            if not base:
                continue
            ratios = [v / base for v in (mean(k) for k in COMPOSITIONAL) if v is not None]
            if ratios:
                samples[arm].append(sum(ratios) / len(ratios))
    out = {}
    for arm, values in samples.items():
        if len(values) < draws // 2:
            continue
        values.sort()
        out[arm] = (values[int(0.025 * len(values))], values[int(0.975 * len(values))])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path,
                        default=ROOT / "outputs/headtohead/val_mini")
    parser.add_argument("--extra-sweep", type=Path,
                        default=ROOT / "outputs/sweep/val_mini",
                        help="the privileged-goal cells, merged in with a 'priv:' prefix")
    parser.add_argument("--split", default="val_mini")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--null", nargs=2, default=["react", "react_rag"],
                        help="the accidental same-config pair that calibrates noise")
    parser.add_argument("--order", nargs="*", default=None)
    arguments = parser.parse_args()

    types = episode_types(arguments.split)
    buckets: Dict[str, List[str]] = {}
    for episode, kind in types.items():
        buckets.setdefault(kind, []).append(episode)

    arms, counts = collect(arguments.sweep, arguments.split)
    if arguments.extra_sweep and arguments.extra_sweep.is_dir():
        extra, extra_counts = collect(arguments.extra_sweep, arguments.split, prefix="priv:")
        arms.update(extra)
        counts.update(extra_counts)
    if not arms:
        raise SystemExit(f"no cells with results under {arguments.sweep}")

    order = arguments.order or [
        "react", "react_rag", "react_rag_R", "gmemory", "memento", "v2_intent",
        "priv:ceiling", "priv:v2_memory_R", "priv:v2_memory_all",
    ]
    order += [arm for arm in sorted(arms) if arm not in order]
    order = [arm for arm in order if arm in arms]

    shared = set.intersection(*(set(scores) for scores in arms.values()))
    print(f"{len(arms)} arms, {len(types)} episodes in {arguments.split}, "
          f"{len(shared)} completed by every arm")
    print("episodes per type: " + ", ".join(f"{k}={len(buckets.get(k, []))}" for k in AXIS))
    print(f"metric: {METRIC}; every memory built from rearrange-only train_mini\n")

    def mean_for(arm: str, kind: str, pool: Optional[set] = None) -> Optional[float]:
        scores = arms[arm]
        values = [scores[e] for e in buckets.get(kind, [])
                  if e in scores and (pool is None or e in pool)]
        return sum(values) / len(values) if values else None

    print("== absolute (each arm on the episodes it completed) ==")
    print(f"  {'arm':18s}" + "".join(f"{k:>9s}" for k in AXIS)
          + f"{'scored':>8s}{'crashed':>9s}")
    for arm in order:
        cells = "".join(f"{v:9.4f}" if (v := mean_for(arm, k)) is not None else f"{'-':>9s}"
                        for k in AXIS)
        attempted, failed = counts.get(arm, (len(arms[arm]), 0))
        flag = "  <-- scored on what survived" if failed > 2 * min(
            f for _, f in counts.values()) + 20 else ""
        print(f"  {arm:18s}{cells}{len(arms[arm]):8d}{failed:9d}{flag}")

    intervals = bootstrap({a: arms[a] for a in order}, buckets, arguments.draws)
    print(f"\n== relative to each arm's own R -- the compositional axis "
          f"(paired bootstrap, {arguments.draws} draws, n={len(shared)}) ==")
    print(f"  {'arm':18s}" + "".join(f"{k:>9s}" for k in AXIS)
          + f"{'mean comp':>11s}{'95% CI':>18s}")
    slopes: Dict[str, float] = {}
    for arm in order:
        base = mean_for(arm, "R")
        if not base:
            continue
        cells = "".join(f"{v / base:9.3f}" if (v := mean_for(arm, k)) is not None
                        else f"{'-':>9s}" for k in AXIS)
        value = slope(arms[arm], buckets)
        slopes[arm] = value if value is not None else float("nan")
        low, high = intervals.get(arm, (float("nan"), float("nan")))
        print(f"  {arm:18s}{cells}{slopes[arm]:11.3f}   [{low:5.3f}, {high:5.3f}]")

    # Verdicts are withheld until every arm has actually finished. A slope read off the
    # episodes that happened to complete first is not the arm's slope: the workers walk
    # the split in order and the quick episodes land first, so a partial table is biased
    # in a direction nobody can sign.
    # Attempted, not scored: an arm that ran the whole split and crashed on a fifth of it
    # has finished, and its crash rate is reported above rather than mistaken for progress.
    total = len(types)
    incomplete = {arm: counts.get(arm, (0, 0))[0] for arm in order
                  if counts.get(arm, (0, 0))[0] < 0.9 * total}
    if incomplete:
        print(f"\n!! INCOMPLETE -- no verdicts. Arms still running (of {total}): "
              + ", ".join(f"{a}={n}" for a, n in incomplete.items()))
        print("!! The table above is whatever finished first and is not the arms' slopes.")
        return

    left, right = arguments.null
    if left in slopes and right in slopes:
        band = abs(slopes[left] - slopes[right])
        print(f"\n== the null band ==")
        print(f"  {left} and {right} are the same configuration run twice (no "
              f"`{{rag_examples}}` slot,")
        print(f"  so the retrieval arm never used its memory). They differ in mean comp by "
              f"{band:.3f}.")
        print(f"  That is noise by construction, and it is wider than the bootstrap CI "
              f"because it also")
        print(f"  carries the simulator's run-to-run stochasticity. **A gap under "
              f"{band:.3f} is not a finding.**")
        print(f"\n  gaps against ours (priv:v2_memory_R, v2_intent):")
        for ours in ("priv:v2_memory_R", "v2_intent"):
            if ours not in slopes:
                continue
            for arm in order:
                if arm in (ours, left, right) or arm.startswith("priv:ceiling"):
                    continue
                gap = slopes[ours] - slopes.get(arm, float("nan"))
                verdict = "REAL" if abs(gap) > band else "within noise"
                print(f"    {ours:18s} - {arm:18s} = {gap:+.3f}   {verdict}")

    print("\nR_S/R_T/R_S_T add a way of composing rearrangement; H_R needs a primitive no")
    print("rearrange-only memory holds, so it is a vocabulary boundary and is excluded from")
    print("'mean comp'. An arm can be flat here and flat at a useless level -- read both tables.")


if __name__ == "__main__":
    main()
