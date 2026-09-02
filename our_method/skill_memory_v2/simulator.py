"""The benchmark's own simulator, loaded once and shared by every layer.

The memory is induced by replaying plans and validated by replaying operators, so it
needs the same world the judge uses. Loading it here rather than re-implementing it is
deliberate: a memory whose model of the world disagrees with the evaluator's would be
wrong in a way no amount of testing on its own terms would reveal.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import random
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

SEED = 20260829


class Simulator:
    """Handles onto the judge's scorer, world model, feasibility checker and entities."""

    def __init__(self, benchmark_root: Path, level: int = 2):
        self.benchmark_root = Path(benchmark_root)
        reward_root = self.benchmark_root / "verl/verl/utils/reward_score"
        if not reward_root.is_dir():
            raise FileNotFoundError(f"VIKI reward directory not found: {reward_root}")
        for name, path in (
            ("verl", reward_root.parents[1]),
            ("verl.utils", reward_root.parent),
            ("verl.utils.reward_score", reward_root),
            ("verl.utils.reward_score.utils", reward_root / "utils"),
            ("verl.utils.reward_score.utils.eval", reward_root / "utils/eval"),
        ):
            if name not in sys.modules:
                module = types.ModuleType(name)
                module.__path__ = [str(path)]
                sys.modules[name] = module
        importlib.import_module("verl.utils.reward_score.utils.eval.eval_viki_2")
        module_name = f"verl.utils.reward_score.viki_{level}"
        if module_name in sys.modules:
            self.scorer = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(
                module_name, reward_root / f"viki_{level}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self.scorer = module
        self.SimEnv = importlib.import_module("verl.utils.reward_score.utils.eval.env").SimEnv
        self.Checker = importlib.import_module("verl.utils.reward_score.utils.eval.checker").Checker
        self.entities = importlib.import_module("verl.utils.reward_score.utils.eval.entities")
        self.Eval = importlib.import_module("verl.utils.reward_score.utils.eval.eval").Eval
        self.viki2 = importlib.import_module("verl.utils.reward_score.utils.eval.eval_viki_2")

    # ------------------------------------------------------------------ world building

    def metadata(self, truth: Dict[str, Any], seed: int = SEED) -> Dict[str, Any]:
        """The environment eval_single builds, reproduced under a fixed seed."""
        data = self.viki2.filter_none_values(copy.deepcopy(truth))
        generator = random.Random(seed)
        record: Dict[str, Any] = {"agents": {}, "assets": {}}
        for robot_id, robot_type in data["robots"].items():
            record["agents"][robot_id] = {"type": robot_type, "pos": {"name": robot_id}}
        for name, positions in data["init_pos"].items():
            if name.startswith("R") and name[1:].isdigit():
                continue
            kind = name.rsplit("_", 1)[0]
            record["assets"][kind] = {"pos": {"name": generator.choice(positions)}}
            if kind in self.viki2.CONTAINER_ASSETS:
                record["assets"][kind]["params"] = {
                    "is_container": True,
                    "position_kwargs": {"name": kind, "isolated": kind == "cabinet"},
                }
        record["goal_constraints"] = data["goal_constraints"]
        record["temporal_constraints"] = data["temporal_constraints"]
        return record

    def world(self, metadata: Dict[str, Any]):
        return self.SimEnv(metadata=copy.deepcopy(metadata))

    def score(self, plan: List[Dict[str, Any]], truth: Dict[str, Any], seed: int = SEED) -> float:
        """The official accuracy, with the scorer's own randomness pinned to `seed`."""
        response = f"<think>composed</think><answer>{plan!r}</answer>"
        globals_ = self.scorer.eval_single.__globals__
        original = globals_["random"]
        try:
            globals_["random"] = random.Random(seed)
            return float(self.scorer.acc_reward(response, truth))
        finally:
            globals_["random"] = original


# ------------------------------------------------------------------------- predicates

def flatten_predicates(node: Any) -> List[Dict[str, Any]]:
    if isinstance(node, dict):
        return [node]
    if isinstance(node, list):
        out: List[Dict[str, Any]] = []
        for item in node:
            out.extend(flatten_predicates(item))
        return out
    return []


def predicate_status(predicate: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (predicate.get("status") or {}).items() if v is not None}


def holds(env, predicate: Dict[str, Any]) -> bool:
    """The judge's own check_constraint, for one predicate, on a live world."""
    table = getattr(env, f"{predicate['type']}s", None)
    if table is None or predicate["name"] not in table:
        return False
    entity = table[predicate["name"]]
    positive = bool(predicate.get("is_satisfied", True))
    for attribute, value in predicate_status(predicate).items():
        current = entity
        for part in attribute.split("."):
            current = getattr(current, part)
        if (current == value) != positive:
            return False
    return True


def state_facts(env, predicate: Dict[str, Any]) -> Dict[str, bool]:
    """The few facts about a requirement's objects that tell operator variants apart."""
    name = predicate["name"]
    target = predicate_status(predicate).get("pos.name")
    subject = env.assets.get(name)
    holder = env.assets.get(subject.pos.name) if subject is not None and subject.pos.name in env.assets else None
    destination = env.assets.get(target) if isinstance(target, str) else None
    return {
        "subject_sealed": bool(subject is not None and subject.pos.isolated),
        "subject_in_container": bool(holder is not None and getattr(holder, "is_container", False)),
        "subject_on_agent": bool(subject is not None and subject.pos.name in env.agents),
        "target_sealed": bool(
            destination is not None
            and getattr(destination, "container_position", None) is not None
            and destination.container_position.isolated
        ),
        "target_on_agent": bool(destination is not None and destination.pos.name in env.agents),
    }


def object_properties(entity) -> Dict[str, bool]:
    container = getattr(entity, "container_position", None)
    return {
        "is_container": bool(getattr(entity, "is_container", False)),
        "isolated": bool(container is not None and container.isolated),
        "pushable": entity.name in {"box", "cardboardbox"},
    }
