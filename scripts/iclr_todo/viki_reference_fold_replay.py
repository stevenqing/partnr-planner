#!/usr/bin/env python3
"""A10 (ICLR TODO spec 2026-09-21): the 19-operator reference library on the single-family OOD split,
with the held-out family removed from it the same way it is removed from ours.

`skill_memory_v2.json` is not hand-written: it is built from VIKI-L2 train (even-indexed episodes) and
ships eight per-fold variants `skill_memory_v2.fold_<family>.json` with that family excluded. The replay
arm of 2026-09-17 scored the OOD split against the FULL reference library, i.e. a library that saw the
held-out family. This re-plans the same archived v3 answers against the per-fold reference variants.

Zero LLM calls: the memory is consulted only after the archived answer, exactly as in
`scripts/viki_ablation_replay.py`, whose code is reused unchanged. It is pointed at a directory whose
`memory_all.json` / `memory_heldout_<family>.json` are the reference library and its folds, so its
`full` arm IS the reference-fold arm here (and so its published-number check is switched off).

  python scripts/iclr_todo/viki_reference_fold_replay.py --model 72B --json <out.json>
"""
import argparse
import sys
from pathlib import Path

REPO = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(REPO / "scripts"))
import viki_ablation_replay as replay  # noqa: E402

A11 = REPO / "results/viki_memory_experiments/amendment11"
FAMILIES = ["clear_table_with_two_robots_and_put_in_cabinet", "cut_fruit_on_board",
            "cut_two_fruits_on_board", "dog_push_box_for_two_panda_transport",
            "ensure_all_fruits_on_table", "parallel_human_dual_asset_to_plate_or_bowl",
            "set_plate_and_fork_on_table", "toast_bread_and_set_plate"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True,
                    help="scratch root; <root>/outputs/v3_memories gets symlinks to the reference files")
    args = ap.parse_args()
    mem = args.root / "outputs/v3_memories"
    mem.mkdir(parents=True, exist_ok=True)
    links = {"memory_all.json": A11 / "skill_memory_v2.json"}
    links.update({f"memory_heldout_{f}.json": A11 / f"skill_memory_v2.fold_{f}.json" for f in FAMILIES})
    for name, target in links.items():
        assert target.is_file(), target
        path = mem / name
        if path.is_symlink() or path.exists():
            path.unlink()
        path.symlink_to(target)
    replay.ROOT = args.root          # MEMORIES is derived from ROOT inside main(); archives are not
    return replay.main(["--model", args.model, "--split", "fold", "--arms", "full", "--no-check",
                        "--json", str(args.json)])


if __name__ == "__main__":
    raise SystemExit(main())
