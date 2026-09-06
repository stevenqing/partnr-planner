#!/usr/bin/env python3
"""The frozen sweep definition: target effect schema x seed episode. Deterministic, read-only.

Replaces the thirteen ad-hoc rung tags that the agent library was assembled from. Those tags
(`72b_iface_*`, `72b_runner_*`, family-targeted cells) were created while repairing the
harness; they are debugging history and are not inherited.

The definition here is fixed by rule, not by choice:

  rung set          = TARGET_KEYS x seed episodes
  seed episodes     = drawn from the audit-confirmed elementary pool by a deterministic
                      rule -- round-robin across families sorted by name, within a family
                      by ascending workbench index, taking episodes that demonstrate the
                      target effect and replay cleanly
  per effect schema = PER_KEY episodes, so the three libraries have the same run budget as
                      the sweep they replace (170 runs / 3 libraries ~ 57 runs each)
  the three libraries share this seed-episode set exactly and differ ONLY in sample seed,
  so the between-library variance is LLM sampling and nothing else

Elementary is the audit's decisive criterion: a row carrying neither the recombination
split's held-out combination (a two-robot cutting unit plus an independent delivery) nor a
comp task_id. See audit/comp_leakage_2026-09-05/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

AUDIT = ROOT / "audit/comp_leakage_2026-09-05"

TARGET_KEYS = ("pos.name", "is_activated")
PER_KEY = 28                      # 2 x 28 = 56 runs per library
HOLDOUT = 4                       # unseen episodes the acceptance test uses
TEMPERATURE = 0.7                 # >0, or three sample seeds give one library
MOVES = 18
BUILD_SEEDS = (20260901, 20260902, 20260903)
INDUCER = "Qwen2.5-VL-72B"        # induction is 72B only; a new configuration, not the old one


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT.parent / "VIKI-R")
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results/frozen_sweep.json")
    arguments = parser.parse_args(argv)

    from our_method.skill_memory_v2.build import load_episodes
    from our_method.skill_memory_v2.simulator import SEED, Simulator
    from viki_comp_leakage_audit import (carries_heldout_combination, effect_key_of,
                                         goal_predicates, row_signature)

    train = arguments.train or (arguments.benchmark_root / "data/VIKI-R/viki/VIKI-L2/train.parquet")
    sim = Simulator(arguments.benchmark_root)
    episodes = load_episodes(train)

    # §1.3: the pools must be the ones the audit certified.
    pools = {}
    for name in ("L0_induction_half.json", "L0_support_pool.json"):
        path = AUDIT / name
        if not path.is_file():
            print("未执行，缺 %s" % path)
            return 1
        pools[name] = sha256_file(path)

    comp_ids = set()
    comp_path = AUDIT / "L0_comp_297.json"
    if comp_path.is_file():
        comp_ids = {row["task_id"] for row in json.loads(comp_path.read_text())["rows"]}

    # Elementary induction-half episodes, by workbench index.
    elementary: List[Dict[str, Any]] = []
    for workbench_index in range(0, len(episodes) // 2 + 1):
        global_index = 2 * workbench_index
        if global_index >= len(episodes):
            break
        truth = episodes[global_index]
        if not isinstance(truth, dict) or not truth.get("time_steps"):
            continue
        if str(truth.get("task_id")) in comp_ids:
            continue
        signature = row_signature(truth, sim, SEED)
        if signature["replay"] != "OK" or carries_heldout_combination(signature):
            continue
        keys = {effect_key_of(p) for p in goal_predicates(truth)} - {None}
        elementary.append({"workbench_index": workbench_index, "global_index": global_index,
                           "task_id": str(truth.get("task_id")),
                           "task_name": truth.get("task_name"),
                           "effect_keys": sorted(keys)})

    rungs, selection = [], {}
    for key in TARGET_KEYS:
        by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in elementary:
            if key in row["effect_keys"]:
                by_family[row["task_name"]].append(row)
        for family in by_family:
            by_family[family].sort(key=lambda r: r["workbench_index"])
        families = sorted(by_family)
        picked, position = [], 0
        while len(picked) < PER_KEY and families:
            progressed = False
            for family in families:
                if position < len(by_family[family]) and len(picked) < PER_KEY:
                    picked.append(by_family[family][position])
                    progressed = True
            if not progressed:
                break
            position += 1
        # Holdout: the acceptance test needs episodes the run was not shown, demonstrating
        # the SAME effect. The shipped default [4, 6, 8, 10] is fixed regardless of target
        # key, so for `is_activated` it can hold episodes that never demonstrate it. Rule:
        # per rung, the four lowest-indexed elementary episodes of the same family that
        # demonstrate this key and are not in the seed set for this key.
        chosen = {row["workbench_index"] for row in picked}
        selection[key] = {"families_available": families,
                          "episodes_available": sum(len(v) for v in by_family.values()),
                          "picked": len(picked)}
        for row in picked:
            pool = [other["workbench_index"] for other in by_family[row["task_name"]]
                    if other["workbench_index"] not in chosen]
            holdout = sorted(pool)[:HOLDOUT]
            rungs.append({"target_key": key, "seed_episode": row["workbench_index"],
                          "task_name": row["task_name"], "task_id": row["task_id"],
                          "holdout": holdout,
                          "holdout_short": len(holdout) < HOLDOUT})

    definition = {
        "frozen_on": date.today().isoformat(),
        "inducer_model": INDUCER,
        "inducer_note": "72B only. The superseded library mixed 72B and 30B inducers, which "
                        "was debugging history; this is a new configuration and does not "
                        "inherit from it. The evaluation still runs three models.",
        "target_keys": list(TARGET_KEYS),
        "per_key": PER_KEY,
        "build_seeds": list(BUILD_SEEDS),
        "runs_per_library": len(rungs),
        "runs_total": len(rungs) * len(BUILD_SEEDS),
        "moves_per_run": MOVES,
        "temperature": TEMPERATURE,
        "max_calls_total": len(rungs) * len(BUILD_SEEDS) * MOVES,
        "seed_episode_rule": "round-robin across families sorted by name, within family by "
                             "ascending workbench index, over elementary induction-half "
                             "episodes demonstrating the target effect and replaying cleanly",
        "elementary_rule": "induction half, replay OK, task_id not in comp_297, and not "
                           "carrying the held-out combination (audit decisive detector)",
        "elementary_episodes": len(elementary),
        "selection": selection,
        "acceptance": "marginal contribution (--library), rungs processed in the fixed order "
                      "below, each seed's library grown incrementally",
        "holdout_rule": "per rung, the four lowest-indexed elementary episodes of the same "
                        "family demonstrating the same target key that are not seed episodes "
                        "for that key; the shipped default [4,6,8,10] is not used",
        "holdout_per_rung": HOLDOUT,
        "pool_hashes": pools,
        "rungs": rungs,
    }
    definition["definition_sha256"] = sha256_text(json.dumps(
        {k: v for k, v in definition.items() if k != "definition_sha256"}, sort_keys=True))

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(definition, indent=1, ensure_ascii=False))

    print("elementary induction-half episodes %d" % len(elementary))
    for key in TARGET_KEYS:
        info = selection[key]
        print("  %-14s families %2d  available %4d  picked %d"
              % (key, len(info["families_available"]), info["episodes_available"], info["picked"]))
    short = [r for r in rungs if r.get("holdout_short")]
    print("rungs with a short holdout %d" % len(short))
    print("runs per library %d | libraries %d | max calls %d"
          % (len(rungs), len(BUILD_SEEDS), definition["max_calls_total"]))
    print("definition sha256 %s" % definition["definition_sha256"][:16])
    print("wrote %s" % arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
