"""Layer 1 for PARTNR: operators induced from recorded privileged rollouts.

The recorder writes, per episode, the propositions the episode is judged on, the
simulation step at which each became satisfied, the high-level action each agent held at
each step, and the map from simulator handles to world-graph names. That is enough to
attribute exactly: the proposition says what became true and when, the action log says
who was doing what at that moment, and the name map lets the two be compared.

An operator is the shortest run of one agent's actions ending in the action that made a
proposition true, written over variables. `?x` is the proposition's subject, `?y` its
destination, and any other entity the body names becomes `?z1`, `?z2`. Preconditions are
read off the body rather than assumed: a body that opens the destination before placing
into it was solving a shut container, and that is recorded as an observation about when
the body applies, not as a rule about the world.

Nothing here is specific to the Rearrange cell. Every proposition type PARTNR evaluates
yields its own operators, so the same traces serve the heterogeneous and spatial cells
without being collected again.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Verbs that bring a proposition about rather than prepare for one. A segment ends at one.
COMPLETING = {"Place", "PowerOn", "PowerOff", "Clean", "Fill", "Pour", "Open", "Close"}
ENTITY_ARGS = ("object_handles", "receptacle_handles", "entity_handles_a",
               "entity_handles_b", "room_ids")


def resolve(handles: Any, names: Dict[str, str]) -> List[str]:
    """Handles as the world graph names them, dropping any the graph never saw."""
    if isinstance(handles, str):
        handles = [handles]
    out = []
    for handle in handles or []:
        name = names.get(str(handle))
        if name:
            out.append(name)
    return out


def action_entities(action: List[str]) -> List[str]:
    """The entities an action names, however the skill happens to spell its arguments."""
    out: List[str] = []
    for part in action[1:]:
        for piece in str(part).split(","):
            piece = piece.strip()
            if piece and piece not in ("none", "on", "within", "next_to", ""):
                out.append(piece)
    return out


def induce_from_trace(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every operator this episode demonstrates, one per proposition it satisfied."""
    names = trace.get("handle_to_name") or {}
    steps = trace.get("steps") or []
    satisfied = trace.get("proposition_satisfied_at")
    propositions = trace.get("propositions") or []
    if not steps or not satisfied or len(satisfied) != len(propositions):
        return []

    # When each agent last finished something, so a segment starts after it.
    finished_at: Dict[str, int] = defaultdict(lambda: -1)
    order = sorted(
        (int(when), index) for index, when in enumerate(satisfied) if int(when) >= 0
    )
    induced: List[Dict[str, Any]] = []

    for when, index in order:
        proposition = propositions[index]
        wanted: Dict[str, List[str]] = {
            key: resolve(proposition.get("args", {}).get(key), names)
            for key in ENTITY_ARGS
        }
        subjects = wanted["object_handles"] or wanted["entity_handles_a"]
        targets = wanted["receptacle_handles"] or wanted["entity_handles_b"] or wanted["room_ids"]
        if not subjects:
            continue

        # The last recorded action change at or before the satisfying step, per agent.
        position = max((i for i, s in enumerate(steps) if s.get("sim_step", 0) <= when),
                       default=None)
        if position is None:
            continue
        actor, actor_action = None, None
        for agent, action in (steps[position].get("actions") or {}).items():
            entities = action_entities(action)
            touches = any(name in entities for name in subjects + targets)
            if action and action[0] in COMPLETING and touches:
                actor, actor_action = agent, action
                break
        if actor is None:  # a proposition nobody's action at that step explains
            continue

        start = finished_at[actor] + 1
        body: List[List[str]] = []
        for step in steps[start : position + 1]:
            action = (step.get("actions") or {}).get(actor)
            if action and (not body or body[-1] != action):
                body.append(list(action))
        finished_at[actor] = position
        if not body:
            continue

        mapping: Dict[str, str] = {}
        for name in subjects:
            mapping[name] = "?x"
        for name in targets:
            mapping.setdefault(name, "?y")
        spare = 0
        abstracted: List[List[str]] = []
        for action in body:
            rewritten = [action[0]]
            for part in action[1:]:
                pieces = []
                for piece in str(part).split(","):
                    piece = piece.strip()
                    if not piece or piece in ("none", "on", "within", "next_to"):
                        pieces.append(piece)
                        continue
                    if piece not in mapping:
                        spare += 1
                        mapping[piece] = f"?z{spare}"
                    pieces.append(mapping[piece])
                rewritten.append(", ".join(p for p in pieces if p != ""))
            abstracted.append(rewritten)
        if not any("?x" in part for action in abstracted for part in action[1:]):
            continue  # a body that never names the subject is not this proposition's

        induced.append({
            "effect": {"key": proposition.get("function_name"),
                       "subject": "?x",
                       "value": "?y" if targets else True},
            "body": abstracted,
            "preconditions": {
                # Read off the body: opening before placing means the destination was shut.
                "target_starts_shut": any(a[0] == "Open" for a in abstracted),
                "needs_exploration": any(a[0] == "Explore" for a in abstracted),
            },
            "cost": len(abstracted),
            "episode": trace.get("episode_id"),
            "agent": actor,
        })
    return induced


def build_library(traces: Iterable[Dict[str, Any]], task_type_of=None) -> Dict[str, Any]:
    library: Dict[str, Dict[str, Any]] = {}
    seen, kept, skipped = 0, 0, Counter()
    for trace in traces:
        seen += 1
        satisfied = trace.get("proposition_satisfied_at")
        propositions = trace.get("propositions") or []
        if not satisfied or len(satisfied) != len(propositions):
            skipped["misaligned_trace"] += 1
            continue
        found = induce_from_trace(trace)
        if not found:
            skipped["nothing_attributable"] += 1
        for operator in found:
            kept += 1
            key = json.dumps([operator["effect"], operator["body"],
                              operator["preconditions"]], sort_keys=True)
            entry = library.setdefault(key, {
                "effect": operator["effect"], "body": operator["body"],
                "preconditions": operator["preconditions"], "cost": operator["cost"],
                "support": 0, "episodes": [], "task_types": [],
            })
            entry["support"] += 1
            if len(entry["episodes"]) < 8:
                entry["episodes"].append(operator["episode"])
            if task_type_of:
                kind = task_type_of(str(operator["episode"]))
                if kind and kind not in entry["task_types"]:
                    entry["task_types"].append(kind)
    operators = sorted(library.values(), key=lambda item: -item["support"])
    return {"operators": operators, "traces_read": seen, "operators_attributed": kept,
            "skipped": dict(skipped)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True, help="directory of recorder output")
    parser.add_argument("--types", default="results/partnr_task_types.json")
    parser.add_argument("--split", default="train_2k")
    parser.add_argument("--only-type", default=None, help="induce from one task type only")
    parser.add_argument("--out", default="results/partnr_operators.json")
    arguments = parser.parse_args()

    index: Dict[str, str] = {}
    types_path = Path(arguments.types)
    if types_path.is_file():
        table = json.loads(types_path.read_text()).get(arguments.split, {})
        for kind, episodes in table.items():
            for episode in episodes:
                index[str(episode)] = kind

    traces = []
    for path in sorted(Path(arguments.traces).glob("*.json")):
        try:
            trace = json.loads(path.read_text())
        except Exception:
            continue
        if arguments.only_type and index.get(str(trace.get("episode_id"))) != arguments.only_type:
            continue
        traces.append(trace)

    record = build_library(traces, lambda e: index.get(e))
    record["induced_from"] = {"traces": arguments.traces, "split": arguments.split,
                              "only_type": arguments.only_type}
    Path(arguments.out).write_text(json.dumps(record, indent=2) + "\n")

    print(f"traces read           {record['traces_read']}")
    print(f"operators attributed  {record['operators_attributed']}")
    print(f"skipped               {record['skipped']}")
    print(f"distinct operators    {len(record['operators'])}\n")
    for operator in record["operators"][:14]:
        effect = operator["effect"]
        when = ", ".join(k for k, v in operator["preconditions"].items() if v) or "-"
        body = "  ".join("[" + " ".join(a) + "]" for a in operator["body"])
        print(f"  {effect['key']}({effect['subject']}, {effect['value']})   "
              f"cost {operator['cost']}   support {operator['support']}   when {when}")
        print(f"      {body[:150]}")
    print(f"\nwrote {arguments.out}")


if __name__ == "__main__":
    main()
