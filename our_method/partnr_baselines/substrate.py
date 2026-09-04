#!/usr/bin/env python3
"""The one pile of training material every memory on PARTNR is built from.

Skill memory v2 induces its operators from the rearrange-only half of `train_mini`,
recorded by `our_method/skill_memory_v2/partnr_recorder.py`. A baseline built from a
different pile is not a comparison of memories, so this module exists to make the pile
literally shared: G-Memory, the MEMENTO-style port and skill memory v2 all read the same
episodes through the same loader, and each one only decides what to keep from them.

Two things are repaired on the way out, both of them cheap and both of them wrong if
skipped:

  * `constraints` and `dependencies` in a recorded trace belong to whichever episode the
    environment had already advanced to when the recorder flushed -- 343 of `train_mini`'s
    399 traces carry another episode's. They are re-read from the dataset by episode id.
    `propositions` are snapshotted at record time and are not affected.
  * the starting placement of every object, which no trace holds at all, is read from the
    dataset's `name_to_receptacle` and mapped through the trace's own `handle_to_name`.
    That is what lets an object-semantics memory say where a kind of thing is usually
    found -- the one piece of knowledge MEMENTO's first node type is made of.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
DATASET = ROOT / "data/datasets/partnr_episodes/v0_0"
ROLLOUTS = ROOT / "results/partnr_rollouts"
TASK_TYPES = ROOT / "results/partnr_task_types.json"

INDEXED = re.compile(r"^(.*)_\d+$")


def category(name: Optional[str]) -> str:
    if name is None:
        return ""
    match = INDEXED.match(str(name))
    return match.group(1) if match else str(name)


def load_episodes(split: str) -> Dict[str, Dict[str, Any]]:
    with gzip.open(DATASET / f"{split}.json.gz") as handle:
        return {str(e["episode_id"]): e for e in json.load(handle)["episodes"]}


def allowed_ids(split: str, types: Optional[Iterable[str]]) -> Optional[set]:
    if not types:
        return None
    classification = json.loads(TASK_TYPES.read_text())[split]
    keep = {str(e) for kind in types for e in classification.get(kind, [])}
    if not keep:
        raise SystemExit(f"no {split} episodes of type {list(types)}")
    return keep


def temporal_edges(episode: Dict[str, Any]) -> List[List[int]]:
    """The proposition DAG, off the dataset rather than the trace."""
    for constraint in episode.get("evaluation_constraints") or []:
        if constraint.get("type") != "TemporalConstraint":
            continue
        arguments = constraint.get("args") or {}
        inner = arguments.get("args") or arguments
        edges = inner.get("dag_edges") or []
        return [list(edge) for edge in edges]
    return []


def proposition_text(proposition: Dict[str, Any], handle_to_name: Dict[str, str]) -> str:
    """One proposition as a sentence, in the world's own names."""

    def names(handles) -> List[str]:
        if isinstance(handles, str):
            handles = [handles]
        return [handle_to_name.get(str(h), str(h)) for h in handles or []]

    arguments = proposition.get("args") or {}
    key = proposition.get("function_name", "?")
    subjects = names(arguments.get("object_handles")) or names(arguments.get("entity_handles_a"))
    targets = (names(arguments.get("receptacle_handles"))
               or names(arguments.get("entity_handles_b"))
               or [str(r) for r in arguments.get("room_ids") or []])
    subject = " or ".join(subjects) if subjects else "?"
    if not targets:
        return f"{subject}: {key}"
    verb = {"is_on_top": "on top of", "is_inside": "inside", "is_next_to": "next to",
            "is_in_room": "in", "is_on_floor": "on the floor of"}.get(key, key)
    return f"{subject} {verb} {' or '.join(targets)}"


def plan_sketch(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The high-level actions, with the repeats the simulator loop produces collapsed.

    `get_next_action` is called once per simulation step and a single high-level action
    spans hundreds of them, so a raw trace is the same line several hundred times over.
    """
    out: List[Dict[str, Any]] = []
    for step in steps:
        actions = {agent: list(action) for agent, action in (step.get("actions") or {}).items()
                   if action}
        if not actions:
            continue
        if out and out[-1]["actions"] == actions:
            continue
        out.append({"step": len(out), "actions": actions})
    return out


def load(split: str = "train_mini", types: Optional[Iterable[str]] = ("R",),
         rollouts: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every recorded rollout of the requested types, repaired and flattened.

    The returned record is deliberately memory-agnostic: it says what was asked, what had
    to become true, what was actually done and where everything started. Which of those a
    memory keeps is the thing under test.
    """
    source = Path(rollouts or (ROLLOUTS / split))
    episodes = load_episodes(split)
    keep = allowed_ids(split, types)
    records: List[Dict[str, Any]] = []
    skipped: Counter = Counter()

    for path in sorted(source.glob("episode_*.json")):
        episode_id = path.stem[len("episode_"):]
        if keep is not None and episode_id not in keep:
            continue
        episode = episodes.get(episode_id)
        if episode is None:
            skipped["not in the dataset"] += 1
            continue
        try:
            trace = json.loads(path.read_text())
        except Exception:
            skipped["unreadable trace"] += 1
            continue
        handle_to_name = dict(trace.get("handle_to_name") or {})
        sketch = plan_sketch(trace.get("steps") or [])
        if not sketch:
            skipped["no actions recorded"] += 1
            continue

        # Where each object started, in the world's names. `name_to_receptacle` maps an
        # object handle to `<furniture handle>|receptacle_mesh_...`, so the furniture
        # handle is the part before the pipe.
        started_at: Dict[str, str] = {}
        for object_handle, receptacle in (episode.get("name_to_receptacle") or {}).items():
            object_name = handle_to_name.get(str(object_handle))
            furniture = handle_to_name.get(str(receptacle).split("|")[0])
            if object_name and furniture:
                started_at[object_name] = furniture

        propositions = trace.get("propositions") or episode.get("evaluation_propositions") or []
        records.append({
            "episode_id": episode_id,
            "instruction": (trace.get("instruction") or episode.get("instruction") or "").strip(),
            "handle_to_name": handle_to_name,
            "propositions": propositions,
            "goals": [proposition_text(p, handle_to_name) for p in propositions],
            # Off the dataset, never off the trace -- see the module docstring.
            "temporal_edges": temporal_edges(episode),
            "sketch": sketch,
            "started_at": started_at,
            "sim_steps": trace.get("sim_steps"),
        })

    records.sort(key=lambda record: int(record["episode_id"]))
    if skipped:
        print(f"substrate: skipped {dict(skipped)}")
    return records


def opened_kinds(records: Iterable[Dict[str, Any]]) -> set:
    """Furniture kinds some rollout had to open. Read off the traces, not asserted."""
    kinds = set()
    for record in records:
        for step in record["sketch"]:
            for action in step["actions"].values():
                if action and action[0] == "Open" and len(action) > 1:
                    kinds.add(category(action[1]))
    return kinds


def container_kinds(records: Iterable[Dict[str, Any]]) -> set:
    """Furniture kinds something was required to end up inside. Also read off the data."""
    kinds = set()
    for record in records:
        for proposition in record["propositions"]:
            if proposition.get("function_name") != "is_inside":
                continue
            arguments = proposition.get("args") or {}
            for handle in arguments.get("receptacle_handles") or []:
                name = record["handle_to_name"].get(str(handle))
                if name:
                    kinds.add(category(name))
    return kinds


def where_found(records: Iterable[Dict[str, Any]]) -> Dict[str, Counter]:
    """For each kind of object, the furniture kinds it started on, counted."""
    where: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        for object_name, furniture in record["started_at"].items():
            where[category(object_name)][category(furniture)] += 1
    return where


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train_mini")
    parser.add_argument("--types", nargs="*", default=["R"])
    arguments = parser.parse_args()

    records = load(arguments.split, arguments.types or None)
    print(f"{len(records)} records from {arguments.split} types={arguments.types or 'all'}")
    print(f"  goals per episode      {sum(len(r['goals']) for r in records) / max(len(records),1):.2f}")
    print(f"  plan steps per episode {sum(len(r['sketch']) for r in records) / max(len(records),1):.2f}")
    print(f"  with a temporal DAG    {sum(1 for r in records if r['temporal_edges'])}")
    print(f"  object kinds placed    {len(where_found(records))}")
    print(f"  container kinds        {sorted(container_kinds(records))}")
    print(f"  kinds that were opened {sorted(opened_kinds(records))}")
    example = records[0]
    print(f"\n  example {example['episode_id']}: {example['instruction']}")
    print(f"    goals:  {example['goals']}")
    print(f"    sketch: {json.dumps(example['sketch'])[:300]}")
