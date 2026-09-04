#!/usr/bin/env python3
"""One socket, three memories.

PARTNR's ReAct prompt has a `{rag_examples}` slot, and `LLMPlanner` fills it by asking
`self.rag` for the nearest example. Nothing about that code is specific to trajectory
retrieval: it needs an object with `retrieve_top_k_given_query` and a `data_dict` whose
entry carries a `trace` string. So G-Memory and the MEMENTO-style port are plugged into
the same socket, behind the same wrapper sentence, feeding the same planner.

That is the point of doing it this way. On VIKI-L2 every baseline and our method wrote a
plan through one interface and differed only in what the memory held. Reproducing that on
PARTNR means holding the consumption fixed -- same planner, same loop, same prompt slot,
same rendering position -- so that the arms differ in the memory and nothing else. The
one comparison this cannot make fair is against skill memory v2 itself, which does not
consume a prompt at all; that difference is the method, and it is why the compositional
axis is read as a degradation slope per arm rather than as absolute scores.

Retrieval is cached per instruction. PARTNR replans up to `replanning_threshold` times
per episode and the instruction never changes within one, so retrieving once per episode
rather than once per replan is the same memory, cheaper -- and it keeps a query-time model
call from being made fifty times for one answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class PromptMemory:
    """The duck type `LLMPlanner` already knows how to talk to.

    `retrieve_top_k_given_query` returns `(text, [index])` and the planner then reads
    `data_dict[index]["trace"]`, so a memory that renders one block of text per query is
    served by handing back a fresh index each time.
    """

    def __init__(self, render, empty: str = "") -> None:
        self._render = render
        self._empty = empty
        self.data_dict: Dict[int, Dict[str, Any]] = {}
        self._by_query: Dict[str, int] = {}

    def retrieve_top_k_given_query(self, query: str, top_k: int = 1, agent_id: int = 0
                                   ) -> Tuple[str, List[int]]:
        assert query != "", "query text is an empty string"
        if query not in self._by_query:
            try:
                text = self._render(query) or self._empty
            except Exception as error:
                # A memory that silently renders nothing is the no-memory baseline wearing
                # this arm's name, which would be reported as a memory result. Say so.
                text = f"[memory unavailable: {type(error).__name__}]"
            index = len(self.data_dict)
            self.data_dict[index] = {"trace": text, "instruction": query}
            self._by_query[query] = index
        index = self._by_query[query]
        return self.data_dict[index]["trace"], [index]


def _sentence_embedder(device: str = "cpu"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-mpnet-base-v2", device=device)

    def embed(texts: List[str]):
        return model.encode(list(texts), convert_to_numpy=True,
                            show_progress_bar=False, normalize_embeddings=True)

    return embed


def build_prompt_memory(plan_config: Any) -> Optional[PromptMemory]:
    """Whichever memory `example_type` names, ready to answer instructions.

    Returns None for the example types PARTNR's own RAG handles, so the caller falls
    through to it unchanged.
    """
    kind = str(getattr(plan_config, "example_type", "") or "")
    if kind not in ("gmemory", "memento", "v2_prompt"):
        return None
    store = Path(str(plan_config.memory_store))
    device = str(getattr(plan_config, "memory_device", "cpu") or "cpu")

    if kind == "v2_prompt":
        from .operator_prompt import PartnrOperatorPrompt

        return PromptMemory(PartnrOperatorPrompt(str(store)))

    if kind == "memento":
        from .memento import PartnrMemento

        built = json.loads(store.read_text())
        extractions = {}
        cache = getattr(plan_config, "memory_extractions", None)
        if cache:
            extractions = json.loads(Path(str(cache)).read_text())

        def extract(prompt: str) -> Dict[str, List[str]]:
            # The prompt carries the instruction at its tail; the precomputed table is
            # keyed on the instruction itself. Missing means the extraction step was not
            # run for this split, and the caller's fallback is the honest one.
            instruction = prompt.rsplit("Instruction: ", 1)[-1].rsplit("\nOutput:", 1)[0]
            if instruction not in extractions:
                raise KeyError(instruction)
            return extractions[instruction]

        memory = PartnrMemento(built, extract, device=device)
        return PromptMemory(memory.render)

    from .gmemory import PartnrGMemory, restore

    memory = PartnrGMemory(restore(store), _sentence_embedder(device))
    return PromptMemory(memory.render)
