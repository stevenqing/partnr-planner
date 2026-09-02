#!/usr/bin/env python3
"""Does a row's image depend on which assets that row makes live?

The transplant route hangs on one question: is an asset left null in a row rendered
in that row's picture? Looking at the pictures did not settle it -- they are 600x337
and the objects are small and generic, so naming one is guesswork.

There is an objective form of the same question. Rows share layouts, and rows of one
layout differ in which assets they place. If the image is rendered from the placement,
then rows of a layout with different live sets must have different images. If instead
a layout has only a handful of distinct images while its rows carry many different
live sets, the picture does not track the objects, and null means "not used by this
task" rather than "absent from the scene".

This compares image bytes. It needs no object recognition and the answer is binary.
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import SOURCE_PARQUET, load_manifest, native
from viki_amendment10_pairs import live


def image_digest(images: Any) -> str:
    entries = images if isinstance(images, (list, tuple)) else [images]
    hasher = hashlib.sha256()
    for entry in entries:
        payload = None
        if isinstance(entry, dict):
            payload = entry.get("bytes") or entry.get("path")
        elif isinstance(entry, (bytes, bytearray)):
            payload = bytes(entry)
        if isinstance(payload, (bytes, bytearray)):
            hasher.update(bytes(payload))
        elif isinstance(payload, str):
            hasher.update(payload.encode())
    return hasher.hexdigest()


def main() -> None:
    test = pd.read_parquet(SOURCE_PARQUET)
    manifest = load_manifest()

    by_layout: Dict[Any, Dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "images": set(), "live_sets": set(), "pairs": set()}
    )
    for index in sorted(manifest):
        row = native(test.iloc[index].to_dict())
        truth = row["reward_model"]["ground_truth"]
        layout = truth.get("layout_id")
        digest = image_digest(row.get("images"))
        assets = frozenset(live(truth))
        entry = by_layout[layout]
        entry["rows"] += 1
        entry["images"].add(digest)
        entry["live_sets"].add(assets)
        entry["pairs"].add((digest, assets))

    print(f"{'layout':>8s} {'rows':>6s} {'images':>8s} {'live sets':>10s} "
          f"{'(image,set) pairs':>18s}")
    for layout, entry in sorted(by_layout.items(), key=lambda kv: str(kv[0])):
        print(
            f"{str(layout):>8s} {entry['rows']:6d} {len(entry['images']):8d} "
            f"{len(entry['live_sets']):10d} {len(entry['pairs']):18d}"
        )

    print()
    by_image: Dict[str, Set[frozenset]] = defaultdict(set)
    by_set: Dict[frozenset, Set[str]] = defaultdict(set)
    for entry in by_layout.values():
        for digest, assets in entry["pairs"]:
            by_image[digest].add(assets)
            by_set[assets].add(digest)
    same_image_many_sets = sum(1 for sets in by_image.values() if len(sets) > 1)
    same_set_many_images = sum(1 for digests in by_set.values() if len(digests) > 1)
    print(f"distinct images across the manifest: {len(by_image)}")
    print(f"distinct live-asset sets:            {len(by_set)}")
    print(
        f"images used by more than one live-asset set: {same_image_many_sets}"
        f"  ({100*same_image_many_sets/max(1,len(by_image)):.1f}%)"
    )
    print(
        f"live-asset sets appearing under more than one image: {same_set_many_images}"
    )
    spread = Counter(len(sets) for sets in by_image.values())
    print(f"live-asset sets per image: {sorted(spread.items())}")


if __name__ == "__main__":
    main()
