#!/usr/bin/env python3
"""In-context-library arm (P) on the two VIKI-L2 CG splits, spec 2026-09-21 §4.

The 72B baselines on CG were run by scripts/viki_amendment10_run.py. This reuses that
runner unchanged -- message assembly, partner prefix, memory insertion
(viki_memory_skill.add_memory_to_messages), model, temperature, max_tokens, seed,
official scorer -- and swaps only the memory provider: every row gets the same frozen
rendering of the admitted library (scripts/incontext_library/render.py). Outputs go to
results/incontext_library_2026-09-21/runs/<split>/ through a directory of symlinks to the
same split parquets, so nothing under amendment10/ is written.

  # zero calls: rebuild the zero-shot prompts and compare them with the archived run
  python scripts/incontext_library/run_p_arm.py --split text --check-template
  # smoke, then the full split
  python scripts/incontext_library/run_p_arm.py --split text --limit 20 --tag smoke20
  python scripts/incontext_library/run_p_arm.py --split text
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import viki_amendment10_run as run10  # noqa: E402

OUT = REPO / "results/incontext_library_2026-09-21"
BLOCK = OUT / "prompts/library_block.txt"
BLOCK_SHA256 = "13e33587b2a7123ba6ff8e36e290abb86153d76fbe13129a09a5e23b11d712a1"
ARCHIVE = REPO / "results/viki_memory_experiments/amendment10"


class LibraryBlock:
    """Memory provider: the whole admitted library, identical for every row."""

    def __init__(self):
        text = BLOCK.read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == BLOCK_SHA256, "library block changed"
        self.text = text.rstrip("\n")

    def prompt(self, index, sample):
        return self.text


def point_runner_at_outputs():
    root = OUT / "runs"
    root.mkdir(parents=True, exist_ok=True)
    for variant in ("text", "imaged"):
        link = root / f"recombination.{variant}.parquet"
        target = ARCHIVE / f"recombination.{variant}.parquet"
        if not link.exists():
            link.symlink_to(target)
    run10.SPLIT_DIR = root


def check_template(variant):
    """Zero calls. Rebuild each row's zero-shot messages with today's code and compare
    prompt_sha256 with the archived zero_shot run on the same split."""
    import pandas as pd
    frame = pd.read_parquet(ARCHIVE / f"recombination.{variant}.parquet")
    manifest = run10.manifest_for(variant)
    archived = {}
    with (ARCHIVE / variant / "zero_shot.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                archived[int(r["index"])] = r["prompt_sha256"]
    same = diff = 0
    for index in sorted(manifest):
        sample = run10.native(frame.iloc[index].to_dict())
        messages = run10.build_messages(sample, manifest[index]["partner_prefix"], "",
                                        variant == "imaged")
        if run10.messages_sha256(messages) == archived.get(index):
            same += 1
        else:
            diff += 1
    result = {"split": variant, "rows": len(manifest), "archived_rows": len(archived),
              "zero_shot_prompt_sha_identical": same, "different": diff}
    print(json.dumps(result))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["text", "imaged"], required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=None, help="first N manifest rows (smoke)")
    ap.add_argument("--check-template", action="store_true")
    args = ap.parse_args()

    point_runner_at_outputs()
    if args.check_template:
        check_template(args.split)
        return

    original_manifest = run10.manifest_for
    if args.limit:
        run10.manifest_for = lambda v: dict(sorted(original_manifest(v).items())[: args.limit])
    run10.make_memory = lambda arm, client: LibraryBlock()
    run10.run(args.split, "incontext_library", args.base_url, args.workers, args.tag)


if __name__ == "__main__":
    main()
