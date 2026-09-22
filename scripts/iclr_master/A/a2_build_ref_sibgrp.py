#!/usr/bin/env python3
"""A2: the 19-operator reference library with a whole sibling group withheld. Zero LLM calls.

`our_method/skill_memory_v2/build.py` withholds one family by skipping its episodes inside
the even-indexed induction half (induction, dependencies and vocabulary each skip
`task_name == exclude_family`; validation skips it in the odd half). This wrapper does exactly
that with a set of families instead of one, calling the same three layer functions with the
same seed and per-family cap. The frozen builder is not edited.

`--check <family>` rebuilds a single-family fold this way and requires it to equal the
archived `amendment11/skill_memory_v2.fold_<family>.json` layer for layer, which is what makes
the grouped build the same process with a different exclusion unit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from viki_amendment5 import BENCHMARK_ROOT  # noqa: E402
from our_method.skill_memory_v2 import dependencies, induction, vocabulary  # noqa: E402
from our_method.skill_memory_v2.build import load_episodes  # noqa: E402
from our_method.skill_memory_v2.memory import FORMAT, SkillMemoryV2  # noqa: E402
from our_method.skill_memory_v2.simulator import SEED, Simulator  # noqa: E402

A11 = ROOT / "results/viki_memory_experiments/amendment11"
TRAIN = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/train.parquet"


def build(group, per_family=250, seed=SEED, validate=200):
    sim = Simulator(BENCHMARK_ROOT)
    episodes = load_episodes(TRAIN)
    keep = lambda t: isinstance(t, dict) and t.get("task_name") not in group  # noqa: E731
    # the split into halves happens first, then the exclusion, as in build.build
    induction_set = [t for t in episodes[::2] if not isinstance(t, dict) or t.get("task_name") not in group]
    record = {
        "format": FORMAT,
        "built_from": "VIKI-L2 train.parquet, even-indexed episodes",
        "excluded_family": sorted(group)[0] if len(group) == 1 else sorted(group),
        "seed": seed,
        "per_family": per_family,
        "layer1": induction.induce(induction_set, sim, seed, per_family, None),
        "layer2": dependencies.mine(induction_set, sim, seed, per_family, None),
        "layer3": vocabulary.harvest(induction_set, None),
    }
    memory = SkillMemoryV2(record)
    if validate:
        holdout = [t for t in episodes[1::2] if keep(t)][:validate]
        memory.record["self_check"] = memory.validate(holdout, sim, seed)
    return memory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true",
                    help="single family: compare with the archived fold library")
    args = ap.parse_args()
    memory = build(set(args.group))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    memory.save(args.out)
    built = json.loads(args.out.read_text())
    print("wrote", args.out, "operators", len(built["layer1"]["operators"]), "self_check", built.get("self_check", {}).get("rate"))
    if args.check:
        ref = json.loads((A11 / ("skill_memory_v2.fold_%s.json" % args.group[0])).read_text())
        same = {k: built.get(k) == ref.get(k) for k in ("layer1", "layer2", "layer3", "self_check", "excluded_family")}
        print("CHECK vs archived fold:", same)
        return 0 if all(same.values()) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
