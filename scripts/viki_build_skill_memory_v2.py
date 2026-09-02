#!/usr/bin/env python3
"""Driver: build skill memory v2 against this checkout's VIKI-R.

The package takes paths rather than finding them, so that it can be built against any
checkout; this fills them in for ours.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from viki_amendment5 import BENCHMARK_ROOT  # noqa: E402
from our_method.skill_memory_v2 import build as build_module  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--train" not in argv:
        argv += ["--train", str(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/train.parquet")]
    if "--benchmark-root" not in argv:
        argv += ["--benchmark-root", str(BENCHMARK_ROOT)]
    if "--out" not in argv:
        argv += ["--out", str(ROOT / "results/viki_memory_experiments/amendment11/skill_memory_v2.json")]
    build_module.main(argv)
