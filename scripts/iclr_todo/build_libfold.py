#!/usr/bin/env python3
"""Pair our v3 library with the PER-FOLD 19-operator reference library on single-family OOD.

`outputs/viki_ablation/v3_lib_fold_<M>.json` (2026-09-17) scored the OOD split against the full
reference library, which saw the held-out family. `viki_reference_fold_replay.py` re-planned it against
the per-fold variants but kept only that one arm. This replays our `full` arm once more (zero LLM calls,
published-number check ON, so it must reproduce the main table) to get its solved rows, and writes
`outputs/viki_ablation/v3_libfold_fold_<M>.json` in the same schema as `v3_lib_fold_<M>.json`, with the
reference arm named `lib_skill_memory_v2_perfold`.

  python scripts/iclr_todo/build_libfold.py --model 72B
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(REPO / "scripts"))
import viki_ablation_replay as replay  # noqa: E402

WORK = REPO / "results/iclr_todo_2026-09-21/work"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    ref = json.loads((WORK / f"ref_fold_{args.model}.json").read_text())
    ref_arm = ref["arms"]["full"]
    assert ref["split"] == "fold" and ref["n"] == ref_arm["n"] and "solved_rows" in ref_arm

    with tempfile.TemporaryDirectory() as tmp:
        ours_path = Path(tmp) / "ours.json"
        rc = replay.main(["--model", args.model, "--split", "fold", "--arms", "full",
                          "--json", str(ours_path)])
        assert rc == 0, rc
        ours = json.loads(ours_path.read_text())
    assert ours["reproduces_published"] and "ALARM" not in ours, ours.get("ALARM")
    full = ours["arms"]["full"]
    assert full["n"] == ref_arm["n"]

    a, b = set(full["solved_rows"]), set(ref_arm["solved_rows"])
    ref_arm = dict(ref_arm)
    ref_arm["vs_full"] = {"delta": round(ref_arm["rate"] - full["rate"], 4),
                          "lost": len(a - b), "gained": len(b - a), "n_shared": full["n"]}
    ours["arms"]["lib_skill_memory_v2_perfold"] = ref_arm
    ours["reference_source"] = str((WORK / f"ref_fold_{args.model}.json").relative_to(REPO))
    ours["note"] = ("reference arm = results/viki_memory_experiments/amendment11/skill_memory_v2.fold_<family>.json "
                    "per held-out family; our arm replayed with the published-number check on")
    out = REPO / f"outputs/viki_ablation/v3_libfold_fold_{args.model}.json"
    out.write_text(json.dumps(ours, indent=1))
    print(out, "ours", full["rate"], "ref", ref_arm["rate"], ref_arm["vs_full"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
