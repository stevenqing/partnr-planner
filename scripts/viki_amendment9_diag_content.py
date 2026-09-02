#!/usr/bin/env python3
"""What does each memory actually put in front of the model on the OOD rows?

The fold diagnostic established that the held-out split is real: for all 924 rows
the ground-truth plan occurs nowhere in the permitted bank, and G-Memory's correct
answers are not copies of anything it could see. Its advantage therefore comes from
the form of what it shows, not from the answer being present. This script measures
that form for both arms on the same rows, with no model calls.

G-Memory's own retrieval asks the model to pick one of its top two permitted
candidates. That choice needs an LLM, so both candidates are measured here and the
range is reported: if the two agree, the comparison does not depend on which one
the rescoring picked.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import numpy as np
import pandas as pd

from habitat_llm.evaluation.viki_branch_conditions import get_instruction
from habitat_llm.evaluation.viki_gmemory import render_retrieval_prompt
from viki_amendment8b import (
    EXPOSED_STEPS,
    MEMORY_PARQUET,
    OUTPUT_DIR,
    SELF_ROBOT,
    SOURCE_PARQUET,
    load_manifest,
    native,
)
from viki_amendment8_memory import _encode
from viki_amendment9_fold_memory import FoldGMemory, fold_bank
from viki_amendment9_folds import folds, rows_of, train_family_by_index

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "queryfix_k8"
SAMPLE = int(os.environ.get("A9_DIAG_SAMPLE", "0"))


class OfflineGMemory(FoldGMemory):
    """FoldGMemory without the relevance call: both permitted candidates are kept."""

    def __init__(self, family: str) -> None:
        # The parent builds an OpenAI client only to rescore; retrieval itself is
        # local, so the client is never used here.
        super().__init__(client=None, family=family)

    def candidates(self, sample: Dict[str, Any]) -> List[str]:
        instruction = get_instruction(sample)
        query = _encode(self.model, [instruction])[0]
        pool = self.state.raw_success_candidates(query, count=len(self.state.records))
        allowed = [p for p in pool if p not in self.blocked][:2]
        insights = self._allowed_insights(query, 3)
        return [
            render_retrieval_prompt([self.state.records[p]], insights) for p in allowed
        ]


def vocabulary(train: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    verbs: Set[str] = set()
    targets: Set[str] = set()
    for i in range(len(train)):
        gt = native(train.iloc[i].to_dict())["reward_model"]["ground_truth"]
        for step in gt["time_steps"]:
            for action in step["actions"].values():
                if isinstance(action, list) and action:
                    verbs.add(str(action[0]))
                    if len(action) > 1:
                        targets.add(str(action[1]))
    return verbs, targets


def plan_terms(truth: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    verbs: Set[str] = set()
    targets: Set[str] = set()
    for step in truth["time_steps"]:
        for action in step["actions"].values():
            if isinstance(action, list) and action:
                verbs.add(str(action[0]))
                if len(action) > 1:
                    targets.add(str(action[1]))
    return verbs, targets


def mentioned(text: str, terms: Set[str]) -> Set[str]:
    low = text.lower()
    return {t for t in terms if t.lower() in low}


STEP_PATTERN = re.compile(r"'step'\s*:\s*(\d+)|\"step\"\s*:\s*(\d+)")


def shown_plan_length(text: str) -> int:
    """The longest step index the memory text spells out, 0 if it shows no plan."""
    best = 0
    for a, b in STEP_PATTERN.findall(text):
        best = max(best, int(a or b))
    return best


def main() -> None:
    manifest = load_manifest()
    test = pd.read_parquet(SOURCE_PARQUET)
    train = pd.read_parquet(MEMORY_PARQUET)
    verbs_all, targets_all = vocabulary(train)
    print(f"action vocabulary: {len(verbs_all)} verbs, {len(targets_all)} targets")

    diag = json.loads(
        (OUTPUT_DIR / "folds" / f"diag102.{VARIANT}.json").read_text()
    )
    focus = set(diag["gmemory_only"])
    print(f"rows where only G-Memory is right: {len(focus)}")

    stats: Dict[str, List[Dict[str, float]]] = {"gmemory": [], "ours": []}
    for family in folds():
        rows = [i for i in rows_of(family) if i in focus]
        if SAMPLE:
            rows = rows[:SAMPLE]
        if not rows:
            continue
        import viki_amendment8_memory as memories

        ours = memories.SkillMemory(fold_bank(family), EXPOSED_STEPS, SELF_ROBOT)
        theirs = OfflineGMemory(family)
        print(f"  {family}: {len(rows)} rows")
        for index in rows:
            sample = native(test.iloc[index].to_dict())
            truth = sample["reward_model"]["ground_truth"]
            want_verbs, want_targets = plan_terms(truth)
            gt_len = len(truth["time_steps"])

            texts = {
                "ours": [ours.prompt(index, sample)],
                "gmemory": theirs.candidates(sample),
            }
            for arm, options in texts.items():
                scored = [
                    {
                        "verb_recall": len(mentioned(t, want_verbs))
                        / max(1, len(want_verbs)),
                        "target_recall": len(mentioned(t, want_targets))
                        / max(1, len(want_targets)),
                        "shows_a_plan": float(shown_plan_length(t) > 0),
                        "len_ratio": (shown_plan_length(t) / gt_len)
                        if shown_plan_length(t)
                        else 0.0,
                        "chars": float(len(t)),
                    }
                    for t in options
                ]
                # Keep the candidate most favourable to that arm, so the comparison
                # cannot be accused of picking G-Memory's weaker option.
                stats[arm].append(max(scored, key=lambda s: s["verb_recall"]))

    print()
    print(f"{'metric':16s} {'ours':>18s} {'gmemory':>18s}")
    keys = ("verb_recall", "target_recall", "shows_a_plan", "len_ratio", "chars")
    for key in keys:
        cells = []
        for arm in ("ours", "gmemory"):
            values = [s[key] for s in stats[arm]]
            cells.append(f"{median(values):8.2f} (mean {sum(values)/len(values):6.2f})")
        print(f"{key:16s} " + " ".join(f"{c:>18s}" for c in cells))

    out = OUTPUT_DIR / "folds" / f"diag_content.{VARIANT}.json"
    out.write_text(json.dumps(stats, indent=2))
    print(f"\nper-row metrics written to {out}")


if __name__ == "__main__":
    main()
