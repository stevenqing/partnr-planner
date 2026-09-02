"""Record what the privileged planner did, and which proposition each action satisfied.

Operator induction needs to know not just what a solver did but which requirement each
piece of the work brought about. On VIKI-L2 that came from replaying a reference plan and
watching predicates turn true. PARTNR ships no reference plans, but it ships something
better: a planner that reads the ground-truth propositions and solves the episode, and a
measure that records the step at which each proposition became satisfied. Running the two
together gives the attribution exactly rather than by inference.

It has to be done online. The episode file states its propositions over simulator handles
(`Squirt_Strain_Fruit_Basket_:0000`) while a plan names world-graph entities
(`squeezer_1`), and nothing written to disk after a run carries the correspondence. Inside
a run both are present: the world graph's nodes hold `name` and `sim_handle` together, so
the mapping is read off the graph and written out with the trace.

What is recorded per episode: the instruction, the propositions and constraints, the
handle-to-name map, the high-level action each agent issued at each step, and the step at
which each proposition became true. That is everything induction needs and nothing about
how it will be used, so the same traces serve any later change to the operator schema.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from habitat_llm.planner.scripted_centralized_planner import ScriptedCentralizedPlanner


def _dig(root: Any, *names: str) -> Any:
    """Follow an attribute path, giving up quietly rather than raising mid-episode."""
    node = root
    for name in names:
        node = getattr(node, name, None)
        if node is None:
            return None
    return node


class RecordingScriptedPlanner(ScriptedCentralizedPlanner):
    """The privileged planner, plus a trace written for every episode it solves.

    Nothing about its planning changes -- it is the same solver producing the same
    actions -- so the traces describe the behaviour the baseline itself exhibits, and a
    memory induced from them inherits no choices of ours beyond what to write down.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        # The base planner's constructor calls reset(), and this subclass's reset flushes
        # a trace, so the trace's own state has to exist before the base runs.
        self._record_dir = Path(os.environ.get("PARTNR_RECORD_DIR", "results/partnr_rollouts"))
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._steps: List[Dict[str, Any]] = []
        self._instruction: Optional[str] = None
        self._names: Dict[str, str] = {}
        self._episode_id: Optional[str] = None
        self._satisfied: Optional[List[int]] = None
        self._sim_steps: int = 0
        self._propositions: List[Any] = []
        self._constraints: List[Any] = []
        self._dependencies: List[Any] = []
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ plumbing

    def _episode(self):
        return _dig(self, "env_interface", "env", "env", "env", "_env", "current_episode")

    def _measures(self):
        task = _dig(self, "env_interface", "env", "env", "env", "_env", "task")
        return getattr(getattr(task, "measurements", None), "measures", {}) or {}

    def _harvest_names(self, world_graph) -> None:
        """The handle-to-name correspondence, which exists only while the run is live."""
        graphs = world_graph.values() if isinstance(world_graph, dict) else [world_graph]
        for graph in graphs:
            for getter in ("get_all_objects", "get_all_furnitures", "get_all_receptacles",
                           "get_all_rooms"):
                try:
                    nodes = getattr(graph, getter)()
                except Exception:
                    continue
                for node in nodes or []:
                    handle = getattr(node, "sim_handle", None)
                    name = getattr(node, "name", None)
                    if handle and name:
                        self._names[str(handle)] = str(name)

    # ------------------------------------------------------------------ recording

    def reset(self) -> None:
        self._flush()
        super().reset()
        # The base reset may run before this subclass's state exists; guard the rebuild.
        self._steps = []
        self._instruction = None
        self._names = {}
        self._episode_id = None
        self._satisfied = None
        self._sim_steps = 0

    def get_next_action(self, instruction, observations, world_graph, **kwargs):
        result = super().get_next_action(instruction, observations, world_graph, **kwargs)
        try:
            episode = self._episode()
            identifier = str(getattr(episode, "episode_id", "?")) if episode is not None else None
            # Episode boundaries are detected here rather than in reset(), which this
            # planner is not guaranteed to receive between episodes -- relying on it let a
            # satisfaction vector from one episode be written into the next one's trace.
            if identifier is not None and identifier != self._episode_id:
                self._flush()
                self._steps, self._names = [], {}
                self._satisfied, self._sim_steps = None, 0
                self._episode_id = identifier
                # Everything about the episode is snapshotted here, at its first step.
                # Reading any of it at flush time reads the *next* episode: the
                # environment advances before this planner is told, which is the same
                # trap the satisfaction vector fell into below. Traces written before
                # this fix carry the wrong episode's constraints and dependencies; the
                # propositions in them are correct, and the constraints are recoverable
                # offline from the dataset by episode id.
                self._propositions = list(
                    getattr(episode, "evaluation_propositions", None) or []
                )
                self._constraints = list(
                    getattr(episode, "evaluation_constraints", None) or []
                )
                self._dependencies = list(
                    getattr(episode, "evaluation_proposition_dependencies", None) or []
                )
            self._instruction = instruction
            self._harvest_names(world_graph)
            actions = {
                str(agent): list(action) if isinstance(action, (list, tuple)) else [str(action)]
                for agent, action in (getattr(self, "last_high_level_actions", {}) or {}).items()
            }
            # get_next_action is called once per simulation step, and a high-level action
            # spans hundreds of them. Only a change is a new entry; `sim_step` keeps the
            # timing so the satisfaction step still lines up with the action that caused it.
            if not self._steps or self._steps[-1].get("actions") != actions:
                self._steps.append({"step": len(self._steps), "sim_step": self._sim_steps,
                                    "actions": actions})
            self._sim_steps += 1
            # The measure is reset when the next episode begins, which happens before this
            # planner's reset, so reading satisfaction at flush time reads a cleared
            # tracker. Snapshot it every step and keep the last one that says anything.
            tracker = self._measures().get("auto_eval_proposition_tracker")
            current = getattr(tracker, "_proposition_satisfied_at", None) if tracker else None
            # A snapshot whose length disagrees with this episode's proposition count
            # belongs to a different episode and is refused rather than written down.
            if (current is not None and len(current) == len(self._propositions)
                    and any(int(v) >= 0 for v in current)):
                self._satisfied = _plain(current)
        except Exception as error:  # a trace is never worth losing an episode over
            self._steps.append({"step": len(self._steps), "error": type(error).__name__})
        return result

    def _flush(self) -> None:
        if not getattr(self, "_steps", None) or getattr(self, "_episode_id", None) is None:
            return
        record: Dict[str, Any] = {
            "episode_id": self._episode_id,
            "instruction": self._instruction,
            "handle_to_name": self._names,
            "steps": self._steps,
        }
        # All three come from the snapshot taken at the episode's first step, never from
        # the live episode: by flush time the environment may already hold the next one.
        record["propositions"] = [
            {"function_name": p.function_name, "args": _plain(p.args)}
            for p in self._propositions
        ]
        record["constraints"] = [
            {"type": type(c).__name__, "args": _plain(getattr(c, "__dict__", {}))}
            for c in self._constraints
        ]
        record["dependencies"] = [
            _plain(getattr(d, "__dict__", {})) for d in self._dependencies
        ]
        record["proposition_satisfied_at"] = self._satisfied
        record["sim_steps"] = self._sim_steps
        path = self._record_dir / f"episode_{self._episode_id}.json"
        path.write_text(json.dumps(record) + "\n")
        self._steps = []

    def __del__(self):  # the last episode has no reset after it
        try:
            self._flush()
        except Exception:
            pass


def _plain(value: Any) -> Any:
    """JSON-safe, without pulling numpy or attrs into the trace."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
