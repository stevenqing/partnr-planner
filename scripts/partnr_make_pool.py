#!/usr/bin/env python3
"""Carve a small, named, reproducible episode pool out of a PARTNR split.

Every pool this writes is a real dataset file, so a cell that runs one needs no episode
filter and pays nothing to load. The manifest is the point: pools are allocated against a
ledger, and a pool may exclude every episode any earlier pool took, which is how the
induction pool, the acceptance gate and the confirmation set are kept disjoint by
construction rather than by memory.

`val_mini` is a subset of `val` -- 365 of its 365 episodes appear there, checked by
instruction and scene -- so nothing selected on `val_mini` may be reported on `val`.
Pools for anything that selects are therefore carved from `train`, which shares no
episode with either.
"""
from __future__ import annotations

import argparse, gzip, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from partnr_task_types import classify

DATA = ROOT / "data/datasets/partnr_episodes/v0_0"
LEDGER = ROOT / "results/partnr_pools/ledger.json"


def load_ledger() -> dict:
    if LEDGER.is_file():
        return json.loads(LEDGER.read_text())
    return {"pools": {}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--source", default="train")
    ap.add_argument("--types", nargs="*", default=None, help="episode types to keep, e.g. R R_T")
    ap.add_argument("--require-key", nargs="*", default=None, help="keep only episodes carrying all of these proposition keys")
    ap.add_argument("--forbid-key", nargs="*", default=None)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--exclude-pools", nargs="*", default=None, help="pool names whose episodes must not be reused")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ledger = load_ledger()
    if args.name in ledger["pools"] and not args.force:
        print(json.dumps({"skipped": args.name, "already": ledger["pools"][args.name]["n"]}))
        return 0

    src = DATA / f"{args.source}.json.gz"
    blob = json.load(gzip.open(src, "rt"))
    episodes = blob["episodes"]

    taken = set()
    for name in args.exclude_pools or []:
        entry = ledger["pools"].get(name)
        if entry is None:
            raise SystemExit(f"--exclude-pools names an unknown pool {name!r}")
        if entry["source"] != args.source:
            raise SystemExit(f"pool {name!r} was carved from {entry['source']}, not {args.source}")
        taken.update(entry["episode_ids"])

    keep = []
    for episode in episodes:
        if str(episode["episode_id"]) in taken:
            continue
        if args.types and classify(episode) not in args.types:
            continue
        keys = {p.get("function_name") for p in (episode.get("evaluation_propositions") or [])}
        if args.require_key and not set(args.require_key) <= keys:
            continue
        if args.forbid_key and (set(args.forbid_key) & keys):
            continue
        keep.append(episode)

    if len(keep) < args.n:
        raise SystemExit(f"only {len(keep)} episodes match; asked for {args.n}")
    random.Random(args.seed).shuffle(keep)
    chosen = keep[: args.n]

    out = DATA / f"{args.name}.json.gz"
    with gzip.open(out, "wt") as handle:
        json.dump({**{k: v for k, v in blob.items() if k != "episodes"}, "episodes": chosen}, handle)

    entry = {"source": args.source, "types": args.types, "require_key": args.require_key,
             "forbid_key": args.forbid_key, "n": len(chosen), "seed": args.seed,
             "excluded_pools": args.exclude_pools or [], "path": str(out.relative_to(ROOT)),
             "episode_ids": [str(e["episode_id"]) for e in chosen],
             "candidates": len(keep)}
    ledger["pools"][args.name] = entry
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=1))
    print(json.dumps({k: v for k, v in entry.items() if k != "episode_ids"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
