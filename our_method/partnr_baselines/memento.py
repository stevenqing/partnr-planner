#!/usr/bin/env python3
"""MEMENTO's user-profile memory, ported from VIKI-L2 to PARTNR.

The retrieval and rendering are the ones already written for VIKI-L2 in
`scripts/viki_memento_rag.py` -- `MementoRAG` only needs a list of typed nodes, so it is
imported and reused verbatim rather than reimplemented. What is written here is the one
part that cannot be shared: turning PARTNR rollouts into MEMENTO's two node types.

  object_semantics   what is known about a kind of thing: the furniture it is usually
                     found on, whether it is a container, whether it has to be opened.
                     Counted over the same rollouts skill memory v2 induces from, so the
                     two arms differ in representation, not in what they were shown.
  user_pattern       what is known about a kind of task: the instruction, what had to end
                     up true, and the plan that was carried out. Written per episode, not
                     per family, because the authors are explicit that episodic memory
                     carries an in-context learning benefit as well as the knowledge, and
                     averaging it away would understate them.

The same disclaimer the VIKI-L2 port carries applies here and should be repeated in the
paper: MEMENTO's memory is about a *person* -- a grandmother's vase, a bedtime routine --
and PARTNR's val split has no persistent user. This is MEMENTO's architecture carrying
PARTNR's content. It is a port, not MEMENTO evaluated on its own benchmark, and no number
from it may be reported as one. What it does support is the comparison that matters here:
type-separated knowledge-graph retrieval rendered into a prompt, against the same
knowledge held as operators and executed.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from viki_memento_rag import EXTRACTION_PROMPT, TOP_K, MementoRAG  # noqa: E402

from . import substrate  # noqa: E402


def build(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The two node types, over the shared training pile."""
    where = substrate.where_found(records)
    containers = substrate.container_kinds(records)
    opened = substrate.opened_kinds(records)

    nodes: List[Dict[str, Any]] = []
    for kind, counter in sorted(where.items()):
        spots = ", ".join(name for name, _ in counter.most_common(3))
        text = f"The {kind} is usually found on: {spots}."
        if kind in containers:
            text += f" The {kind} is a container that other things can be put into."
        nodes.append({"knowledge": f"{kind}: {spots}", "text": text,
                      "knowledge_type": "object_semantics", "entity": kind,
                      "support": int(sum(counter.values()))})
    for kind in sorted(opened):
        # Kept as its own node rather than folded into the placement node above, because
        # the furniture that has to be opened is rarely a kind anything starts on, so
        # folding would drop it entirely.
        nodes.append({
            "knowledge": f"{kind}: opening",
            "text": f"The {kind} has to be opened before anything can cross it.",
            "knowledge_type": "object_semantics", "entity": kind, "support": 1})

    for record in records:
        if not record["instruction"]:
            continue
        goals = "; ".join(record["goals"])
        text = (f"Asked: \"{record['instruction']}\"\n"
                f"  what had to end up true: {goals}\n"
                f"  the plan that was carried out ({len(record['sketch'])} steps): "
                f"{json.dumps(record['sketch'])}")
        nodes.append({"knowledge": record["instruction"], "text": text,
                      "knowledge_type": "user_pattern", "entity": record["episode_id"],
                      "support": 1})

    return {"nodes": nodes,
            "counts": {t: sum(1 for n in nodes if n["knowledge_type"] == t)
                       for t in ("object_semantics", "user_pattern")},
            "built_from": {"episodes": len(records),
                           "ids": [r["episode_id"] for r in records]}}


class PartnrMemento:
    """Build once, then answer an instruction with the retrieved subgraph as prose."""

    def __init__(self, record: Dict[str, Any], extract, top_k: int = TOP_K,
                 device: str = "cpu"):
        self.rag = MementoRAG(record, device=device)
        self.extract = extract
        self.top_k = top_k
        self._cache: Dict[str, str] = {}

    def render(self, instruction: str) -> str:
        if instruction in self._cache:
            return self._cache[instruction]
        try:
            extracted = self.extract(EXTRACTION_PROMPT.format(instruction=instruction))
        except Exception:
            # The authors' pipeline names the personalized entities with a model. If that
            # call fails the honest fallback is the instruction itself as a single query
            # of both types, not an empty memory: an empty one would silently turn this
            # arm into the no-memory baseline.
            extracted = {"object_semantics": [instruction], "user_pattern": [instruction]}
        text = self.rag.to_natural_language(self.rag.retrieve(extracted, top_k=self.top_k))
        self._cache[instruction] = text
        return text


def parse_extraction(text: str) -> Dict[str, List[str]]:
    """The authors' output format is a JSON object; models wrap it in prose."""
    start, end = str(text).find("{"), str(text).rfind("}")
    if start < 0 or end <= start:
        return {"object_semantics": [], "user_pattern": []}
    try:
        value = json.loads(text[start:end + 1])
    except Exception:
        return {"object_semantics": [], "user_pattern": []}
    return {key: [str(item) for item in (value.get(key) or []) if item]
            for key in ("object_semantics", "user_pattern")}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train_mini")
    parser.add_argument("--types", nargs="*", default=["R"])
    parser.add_argument("--out", type=Path,
                        default=Path("results/partnr_memento_train_mini_R.json"))
    arguments = parser.parse_args()

    records = substrate.load(arguments.split, arguments.types or None)
    built = build(records)
    arguments.out.write_text(json.dumps(built, indent=1) + "\n")
    print(f"{built['counts']} nodes from {len(records)} episodes -> {arguments.out}")
    for kind in ("object_semantics", "user_pattern"):
        example = next(n for n in built["nodes"] if n["knowledge_type"] == kind)
        print(f"\n[{kind}] {example['text'][:400]}")
