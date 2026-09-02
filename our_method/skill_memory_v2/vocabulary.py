"""Layer 3: the names this world uses.

The judge compares the name a plan gives a place against the name the goal asks for, so
`kitchen island` and `kitchen island area` are two different places to it however
plainly they are the same counter to a reader. Nothing in an instruction says which
spelling is official. It is a fact about the domain, stated over and over in the
training episodes, and it is the cheapest thing a memory can be asked to hold.

Assets and places are kept apart because a prediction snaps onto them differently: an
object matching nothing is a wrong object and is dropped, while a place matching nothing
is still a place and is kept, in its cleaned spelling.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional


def harvest(episodes: Iterable[Dict[str, Any]], exclude_family: Optional[str] = None) -> Dict[str, Any]:
    assets, positions, goal_targets = Counter(), Counter(), Counter()
    episodes_used = 0
    for truth in episodes:
        if not isinstance(truth, dict):
            continue
        if exclude_family and truth.get("task_name") == exclude_family:
            continue
        episodes_used += 1
        for name, where in (truth.get("init_pos") or {}).items():
            if where is None or (name.startswith("R") and name[1:].isdigit()):
                continue
            assets[name.rsplit("_", 1)[0]] += 1
            for item in where:
                if isinstance(item, str):
                    positions[item] += 1
        stack = list(truth.get("goal_constraints") or [])
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                target = (node.get("status") or {}).get("pos.name")
                if isinstance(target, str):
                    goal_targets[target] += 1
    places = Counter(positions)
    places.update(goal_targets)
    for asset, count in assets.items():
        places[asset] += count
    return {
        "episodes": episodes_used,
        "assets": [name for name, _ in assets.most_common()],
        "places": [name for name, _ in places.most_common()],
        "goal_targets": dict(goal_targets.most_common()),
        "self_check_every_goal_target_known": all(name in places for name in goal_targets),
    }


def clean(name: str) -> str:
    """The identifier spelling a model reaches for, written the way the world spells it."""
    return " ".join(name.replace("_", " ").replace("-", " ").split())


def canonical(name: Any, known: List[str]) -> Optional[str]:
    """Snap a predicted name onto a known one, but only when it plainly means it.

    Exact, case, and whole-word containment are accepted and nothing else, so a wrong
    object stays wrong: this is grounding, not a second guess at what was meant.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    raw = name.strip()
    for candidate in (raw, clean(raw)):
        if candidate in known:
            return candidate
    lowered = {item.lower(): item for item in known}
    for candidate in (raw, clean(raw)):
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    words = set(re.findall(r"[a-z0-9]+", clean(raw).lower()))
    best, best_score = None, 0
    for item in known:
        item_words = set(re.findall(r"[a-z0-9]+", item.lower()))
        if not item_words:
            continue
        overlap = len(words & item_words)
        if overlap > best_score and (words <= item_words or item_words <= words):
            best, best_score = item, overlap
    return best
