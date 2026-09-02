#!/usr/bin/env python3
"""Print the two arms' memory text for one row, so the content metrics can be checked.

The shows_a_plan metric keys on an explicit `step: N` structure. Our renderer could
be showing an ordered plan in some other notation, which would make that metric a
statement about formatting rather than content, so the rendered text is read here
before anything is concluded from it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from viki_amendment8b import EXPOSED_STEPS, OUTPUT_DIR, SELF_ROBOT, SOURCE_PARQUET, native
from viki_amendment9_diag_content import OfflineGMemory
from viki_amendment9_fold_memory import fold_bank
from viki_amendment9_folds import folds, rows_of

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "queryfix_k8"
WANT_FAMILY = sys.argv[2] if len(sys.argv) > 2 else "cut_fruit_on_board"

diag = json.loads((OUTPUT_DIR / "folds" / f"diag102.{VARIANT}.json").read_text())
focus = set(diag["gmemory_only"])
rows = [i for i in rows_of(WANT_FAMILY) if i in focus][:1]
if not rows:
    raise SystemExit(f"no disagreement rows in {WANT_FAMILY}")
index = rows[0]

test = pd.read_parquet(SOURCE_PARQUET)
sample = native(test.iloc[index].to_dict())
truth = sample["reward_model"]["ground_truth"]

import viki_amendment8_memory as memories

ours = memories.SkillMemory(fold_bank(WANT_FAMILY), EXPOSED_STEPS, SELF_ROBOT)
theirs = OfflineGMemory(WANT_FAMILY)

print("=" * 78)
print(f"row {index}  fold {WANT_FAMILY}")
print("=" * 78)
print("GROUND TRUTH PLAN")
print(json.dumps(truth["time_steps"], indent=1)[:1500])
print()
print("=" * 78)
print("OUR MEMORY")
print("=" * 78)
print(ours.prompt(index, sample))
print()
print("=" * 78)
print("GMEMORY (first permitted candidate)")
print("=" * 78)
print(theirs.candidates(sample)[0])
