#!/usr/bin/env python3
"""Export a few cutting scenes' images so the last open question can be looked at.

Metadata says the narrow transplant route covers every cutting scene: each one has
four or more assets its layout always places in the same spot and that the cutting
task does not touch. What metadata cannot say is whether such an asset is actually
rendered in a row that leaves its init_pos null. If it is, null means only "not
needed by this task" and the transplant keeps the scene and its picture consistent.
If it is not, the object is absent and asking a model to fetch it would make the task
unsolvable by perception.

The rows carry their images, so this writes them out alongside the asset lists.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import MEMORY_PARQUET, SOURCE_PARQUET, load_manifest, native
from viki_amendment9_folds import train_family_by_index
from viki_amendment10_pairs import live
from viki_amendment10_recombine import CUTTING, delivery_goals

OUT = Path("/tmp/viki_images")


def save(images, stem: str) -> list:
    written = []
    entries = images if isinstance(images, (list, tuple)) else [images]
    for number, entry in enumerate(entries):
        payload = None
        if isinstance(entry, dict):
            payload = entry.get("bytes") or entry.get("path")
        elif isinstance(entry, (bytes, bytearray)):
            payload = bytes(entry)
        if isinstance(payload, (bytes, bytearray)):
            path = OUT / f"{stem}_{number}.png"
            path.write_bytes(bytes(payload))
            written.append(str(path))
        elif isinstance(payload, str):
            written.append(f"(path reference) {payload}")
        else:
            written.append(f"(unhandled type {type(entry).__name__})")
    return written


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(MEMORY_PARQUET)
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()
    catalogue = delivery_goals(train, train_family_by_index())

    shown = 0
    for index in sorted(manifest):
        row = native(test.iloc[index].to_dict())
        truth = row["reward_model"]["ground_truth"]
        if truth.get("task_name") not in CUTTING:
            continue
        assets = live(truth)
        used = {
            str(a[1])
            for step in truth["time_steps"]
            for a in step["actions"].values()
            if a is not None and len(a) > 1
        }
        nulls = sorted(
            {
                name.rsplit("_", 1)[0]
                for name, positions in (truth.get("init_pos") or {}).items()
                if positions is None
                and not (name.startswith("R") and name[1:].isdigit())
            }
            & set(catalogue)
        )
        files = save(row.get("images"), f"row{index}")
        print(f"row {index}  layout {truth.get('layout_id')}  {truth['task_name']}")
        print(f"  description: {truth['description']}")
        print(f"  live assets: {sorted(assets)}")
        print(f"  used by the plan: {sorted(used)}")
        print(f"  deliverable but null in this row: {nulls}")
        print(f"  images: {files}")
        print()
        shown += 1
        if shown >= 3:
            break


if __name__ == "__main__":
    main()
