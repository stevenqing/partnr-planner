#!/usr/bin/env python3
"""Probe 1: can a symbolic composer, handed the goal predicates, out-plan the LLM?

The official judge does not compare a prediction to the reference plan. It runs the
prediction in a simulator and asks whether the goal predicates hold at the end
(eval.py:Eval.eval), subject to one outer rule -- the prediction may not be longer
than the reference (viki_2.py:acc_reward). So the task is reachability in a small
deterministic world, and every arm so far has been solving it by asking a language
model to write the answer out of retrieved text.

This composes the answer instead. Requirements become macro chains, chains are handed
to robots, and a scheduler advances each robot by one action per step. Feasibility and
compatibility are decided by the benchmark's own Checker on its own SimEnv rather than
by a re-implementation, so the planner cannot disagree with the judge about what is
legal -- the failure mode that would otherwise make this number meaningless.

Three things the first draft got wrong, each found by reading the judge rather than
the results:

  Temporal constraints are requirements, not just an order. `eval` fails the moment a
  later stage holds while an earlier one does not, and demands every stage hold
  together at some step. A stage can name a predicate that appears in no goal --
  bread in the toaster is only ever stated as stage one -- so composing from the goals
  alone omits work the judge requires. Stages are collected as requirements here, and
  the action that completes a stage carries the earlier stages as a guard.

  Chains have to interleave. Opening the cabinet is not a separate errand: a two-armed
  robot carrying the pumpkin reaches and opens with its free hand, in the same trip.
  Each robot therefore holds several chains and advances whichever one it can, which
  turns the open into a splice rather than a detour and saves the step the reference
  plan also saves.

  A move to where the robot already stands is dropped rather than executed, since the
  budget counts steps and that step buys nothing.

The composer never sees `time_steps`; it is handed a truth dict with that key removed.
The reference length re-enters only in the judge, as the budget.

Probe 1a (this file) writes the macro library by hand. That measures the ceiling of
symbolic composition itself, independent of memory quality: if it is low, the
direction is dead and no memory-derived operator library can rescue it. Probe 1b swaps
the hand-written macros for ones compiled out of the skill memory, and the gap between
them is what the memory contributes.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import permutations, product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT

SEED = 20260829  # the seed the archived ID arms were scored under
MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "results/viki_memory_experiments/amendment8b/interactive_manifest.jsonl"
)
OUT = Path(__file__).resolve().parents[1] / "results/viki_memory_experiments/amendment11"
STEP_CAP = 16  # no reference plan exceeds 9; this only bounds the search
MAX_CANDIDATES = 4000


# --------------------------------------------------------------------------- setup

def load_sim():
    """The judge's own simulator classes, loaded the way viki_bench loads the scorer."""
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    env_mod = importlib.import_module("verl.utils.reward_score.utils.eval.env")
    checker_mod = importlib.import_module("verl.utils.reward_score.utils.eval.checker")
    entities = importlib.import_module("verl.utils.reward_score.utils.eval.entities")
    eval_mod = importlib.import_module("verl.utils.reward_score.utils.eval.eval")
    viki2 = importlib.import_module("verl.utils.reward_score.utils.eval.eval_viki_2")
    return scorer, env_mod.SimEnv, checker_mod.Checker, entities, eval_mod.Eval, viki2


def build_metadata(truth: Dict[str, Any], viki2, seed: int) -> Dict[str, Any]:
    """The environment eval_single builds, reproduced under a fixed seed.

    eval_single draws each asset's position with random.choice, so a planner that wants
    to reason about the same world has to draw the same numbers in the same order. The
    scorer is called with random re-seeded to `seed` (viki_bench:score_response), so
    seeding identically here puts planner and judge on one realization.
    """
    data = viki2.filter_none_values(copy.deepcopy(truth))
    generator = random.Random(seed)
    metadata: Dict[str, Any] = {"agents": {}, "assets": {}}
    for robot_id, robot_type in data["robots"].items():
        metadata["agents"][robot_id] = {"type": robot_type, "pos": {"name": robot_id}}
    for name, positions in data["init_pos"].items():
        if name.startswith("R") and name[1:].isdigit():
            continue
        kind = name.rsplit("_", 1)[0]
        metadata["assets"][kind] = {"pos": {"name": generator.choice(positions)}}
        if kind in viki2.CONTAINER_ASSETS:
            metadata["assets"][kind]["params"] = {
                "is_container": True,
                "position_kwargs": {"name": kind, "isolated": kind == "cabinet"},
            }
    metadata["goal_constraints"] = data["goal_constraints"]
    metadata["temporal_constraints"] = data["temporal_constraints"]
    return metadata


# ----------------------------------------------------------------------- goal model

def flatten_predicates(node: Any) -> List[Dict[str, Any]]:
    if isinstance(node, dict):
        return [node]
    if isinstance(node, list):
        out: List[Dict[str, Any]] = []
        for item in node:
            out.extend(flatten_predicates(item))
        return out
    return []


def flatten_goals(goal_constraints: List[Any]) -> List[Dict[str, Any]]:
    return flatten_predicates(goal_constraints)


def predicate_key(predicate: Dict[str, Any]) -> str:
    status = {k: v for k, v in predicate.get("status", {}).items() if v is not None}
    return json.dumps(
        [predicate.get("type"), predicate.get("name"), predicate.get("is_satisfied", True), status],
        sort_keys=True,
    )


def requirement_holds(env, predicate: Dict[str, Any]) -> bool:
    """The judge's own check_constraint, for one predicate, on a live env."""
    table = getattr(env, f"{predicate['type']}s", None)
    if table is None or predicate["name"] not in table:
        return False
    entity = table[predicate["name"]]
    positive = bool(predicate.get("is_satisfied", True))
    for attribute, value in predicate.get("status", {}).items():
        if value is None:
            continue
        current = entity
        for part in attribute.split("."):
            current = getattr(current, part)
        if (current == value) != positive:
            return False
    return True


def collect_requirements(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every predicate the judge will check, with the stages that must precede it.

    Temporal stages are gathered as requirements in their own right because `eval`
    demands they all hold, and because a stage can name work that no goal mentions.
    """
    requirements: Dict[str, Dict[str, Any]] = {}

    def register(predicate: Dict[str, Any], guard: List[Dict[str, Any]]) -> None:
        key = predicate_key(predicate)
        entry = requirements.setdefault(key, {"predicate": predicate, "guard": []})
        for item in guard:
            if predicate_key(item) != key and all(
                predicate_key(item) != predicate_key(existing) for existing in entry["guard"]
            ):
                entry["guard"].append(item)

    for constraint in metadata.get("temporal_constraints") or []:
        earlier: List[Dict[str, Any]] = []
        for stage in constraint:
            stage_predicates = flatten_predicates(stage)
            for predicate in stage_predicates:
                register(predicate, list(earlier))
            earlier.extend(stage_predicates)
    for predicate in flatten_goals(metadata["goal_constraints"]):
        register(predicate, [])
    return list(requirements.values())


# --------------------------------------------------------------------- macro library

OPERATORS_PATH = OUT / "operators.json"
PUSHABLE = {"box", "cardboardbox"}


def load_operators(path: Path = OPERATORS_PATH) -> Optional[List[Dict[str, Any]]]:
    """Layer 1, if it has been induced: bodies observed in training, over variables.

    An induced body whose targets are not all variables is refused. Those are the
    segments where the runner walked to a place named in that episode and nowhere else;
    keeping them would carry a training scene's furniture into every later plan, which
    is the overfitting the abstraction step exists to prevent.
    """
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    usable = []
    for operator in record.get("operators", []):
        if operator.get("coordinated"):
            targets = [
                item
                for role in operator["roles"]
                for entry in role["actions"]
                for item in entry["action"][1:]
            ]
        else:
            targets = [item for action in operator["body"] for item in action[1:]]
        if targets and all(item.startswith("?") and not item.startswith("?agent") for item in targets):
            usable.append(operator)
    return usable


def state_preconditions(env, predicate: Dict[str, Any]) -> Dict[str, bool]:
    """The same few facts the induction recorded, read off the current world."""
    name = predicate["name"]
    target = (predicate.get("status") or {}).get("pos.name")
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


def induced_chains_for(
    env,
    requirement: Dict[str, Any],
    operators: List[Dict[str, Any]],
    limit: int = 4,
    coordinated: bool = True,
) -> Optional[List[List[Dict[str, Any]]]]:
    """Every way the induced library offers to make this requirement hold.

    Alternatives are returned rather than one answer, because the library holds several
    bodies for the same effect -- a delivery into an open cupboard, and the longer one
    that opens the cupboard on the way -- and which of them fits depends on the scene
    and on what the other robots are doing. Ranking puts the bodies whose recorded
    situation matches this one first, then the cheap ones, then the well-supported ones,
    and the composer's own search settles the rest.
    """
    predicate = requirement["predicate"]
    if predicate.get("type") != "asset" or not predicate.get("is_satisfied", True):
        return None
    status = {k: v for k, v in (predicate.get("status") or {}).items() if v is not None}
    if predicate["name"] not in env.assets:
        return None
    if "pos.name" in status:
        wanted, value = "pos.name", status["pos.name"]
    elif status.get("is_activated") is True:
        wanted, value = "is_activated", True
    else:
        return None

    here = state_preconditions(env, predicate)

    # A shut container has to be opened before anything crosses its lip, and the body
    # that opens it is an operator of its own. It is offered alongside the delivery
    # rather than spliced into it, so any robot with a free hand can do it -- which is
    # what lets the two pieces of work overlap instead of queueing.
    unseal: List[Dict[str, Any]] = []
    sealed = None
    if here.get("target_sealed") and isinstance(value, str):
        sealed = value
    elif here.get("subject_sealed"):
        holder = env.assets[predicate["name"]].pos.name
        sealed = holder if holder in env.assets else None
    if sealed is not None:
        repairs = sorted(
            (item for item in operators if item["effect"]["key"] == "unsealed"),
            key=lambda item: (item["cost"], -item["support"]),
        )
        if repairs:
            unseal = [
                {
                    "actions": [
                        [action[0]] + [sealed if target == "?x" else target for target in action[1:]]
                        for action in repairs[0]["body"]
                    ],
                    "guard": [],
                }
            ]

    ranked = []
    for operator in operators:
        if operator["effect"]["key"] != wanted:
            continue
        if bool(operator.get("coordinated")) != coordinated:
            continue
        mismatch = sum(1 for key, was in operator["preconditions"].items() if here.get(key) != was)
        ranked.append((mismatch, operator["cost"], -operator["support"], operator))
    ranked.sort(key=lambda item: item[:3])

    alternatives: List[List[Dict[str, Any]]] = []
    for rank, (_, _, _, operator) in enumerate(ranked):
        binding = {"?x": predicate["name"]}
        if wanted == "pos.name":
            binding["?y"] = value
        if operator.get("coordinated"):
            targets = [
                item
                for role in operator["roles"]
                for entry in role["actions"]
                for item in entry["action"][1:]
            ]
        else:
            targets = [item for action in operator["body"] for item in action[1:]]
        spares = sorted({item for item in targets if item.startswith("?z")})
        options: List[Dict[str, str]] = [binding]
        for spare in spares:
            wants = operator["types"].get(spare, {})
            def suits(asset) -> bool:
                for key, observed in (("is_container", wants.get("is_container")),
                                      ("pushable", wants.get("pushable"))):
                    if observed is None:
                        continue
                    actual = (
                        bool(getattr(asset, "is_container", False)) if key == "is_container"
                        else asset.name in PUSHABLE
                    )
                    if actual != observed:
                        return False
                return True

            fits = [
                name for name, asset in env.assets.items()
                if name not in binding.values() and suits(asset)
            ]
            options = [dict(option, **{spare: name}) for option in options for name in fits[:2]]
        for number, option in enumerate(options[:2]):
            if operator.get("coordinated"):
                group = f"g{rank}:{number}"
                last = max(
                    range(len(operator["roles"])),
                    key=lambda position: max(
                        entry["offset"] for entry in operator["roles"][position]["actions"]
                    ),
                )
                chains, broken = [], False
                for position, role in enumerate(operator["roles"]):
                    actions, after = [], []
                    for entry in role["actions"]:
                        verb, targets = entry["action"][0], entry["action"][1:]
                        # A role variable names whichever robot takes that role, and
                        # which robot that is only becomes known when the schedule is
                        # laid out, so it is left standing and resolved there.
                        bound = [
                            item if item.startswith("?r") else option.get(item, item)
                            for item in targets
                        ]
                        if any(item.startswith("?") and not item.startswith("?r") for item in bound):
                            broken = True
                        actions.append([verb] + bound)
                        after.append(entry["after"])
                    chains.append(
                        {
                            "actions": actions,
                            "after": after,
                            "guard": requirement["guard"] if position == last else [],
                            "group": group,
                            "role": position,
                        }
                    )
                if not broken:
                    alternatives.append(chains)
                continue
            body = [[action[0]] + [option.get(item, item) for item in action[1:]] for action in operator["body"]]
            if any(item.startswith("?") for action in body for item in action[1:]):
                continue
            alternatives.append(unseal + [{"actions": body, "guard": requirement["guard"]}])
        if len(alternatives) >= limit:
            break
    return alternatives or None


def relay_chains(
    env, name: str, target: str, guard: List[Dict[str, Any]]
) -> Optional[List[Dict[str, Any]]]:
    """Hand an asset between two robots that cannot walk, using a pushed container.

    A panda is an arm bolted in place: it has no `move`, so it can only touch what is
    already beside it. When the asset starts at one such arm and the destination sits
    beside another, no carry exists -- the reach check will refuse every step of it.
    What does exist is a legged robot with `push`, which relocates a container and
    itself in one action, and a container is exactly the vehicle the two arms can both
    load and unload. The second push is guarded on the asset actually being inside,
    since nothing in the feasibility rules stops a dog from shoving an empty box.
    """
    asset = env.assets[name]
    source = asset.pos.name if asset.pos.name in env.agents else None
    destination = None
    if target in env.assets:
        holder = env.assets[target].pos.name
        if holder in env.agents:
            destination = holder
    if source is None or destination is None or source == destination:
        return None
    pusher = next(
        (robot for robot, agent in env.agents.items() if "push" in agent.avail_actions), None
    )
    carrier = next(
        (
            item
            for item, entity in env.assets.items()
            if item in PUSHABLE and getattr(entity, "is_container", False)
        ),
        None,
    )
    if pusher is None or carrier is None:
        return None
    loaded = {
        "type": "asset",
        "name": name,
        "is_satisfied": True,
        "status": {"pos.name": carrier},
    }
    return [
        {
            "actions": [["Move", carrier], ["Push", carrier, source], ["Push", carrier, destination]],
            "guard": [loaded],
        },
        {"actions": [["Reach", name], ["Grasp", name], ["Place", carrier]], "guard": []},
        {"actions": [["Reach", name], ["Grasp", name], ["Place", target]], "guard": guard},
    ]


def chains_for(env, requirement: Dict[str, Any], relay: bool = True) -> Optional[List[Dict[str, Any]]]:
    """Chains that would make one requirement hold.

    A chain is a sequence one robot runs; a requirement may need more than one, since
    an asset bound for a closed cabinet needs the cabinet opened and that can be
    another robot's work. The guard rides on the chain that completes the requirement,
    and is checked before its last action -- the one that makes the predicate true --
    so the approach to a target can still happen early and cost no extra step.
    """
    predicate = requirement["predicate"]
    guard = requirement["guard"]
    if predicate.get("type") != "asset" or not predicate.get("is_satisfied", True):
        return None
    name = predicate["name"]
    if name not in env.assets:
        return None
    asset = env.assets[name]
    status = {k: v for k, v in predicate.get("status", {}).items() if v is not None}
    unknown = set(status) - {"pos.name", "is_activated"}
    if unknown:
        return None
    chains: List[Dict[str, Any]] = []

    def unseal(position_name: str) -> None:
        """A sealed container has to be opened before anything crosses its lip."""
        if position_name in env.assets:
            holder = env.assets[position_name]
            if getattr(holder, "is_container", False) and holder.container_position.isolated:
                chains.append(
                    {
                        "actions": [["Move", position_name], ["Reach", position_name], ["Open", position_name]],
                        "guard": [],
                    }
                )

    target = status.get("pos.name")
    if target is not None and relay:
        relayed = relay_chains(env, name, target, guard)
        if relayed is not None:
            return relayed
    if target is not None:
        unseal(asset.pos.name)
        unseal(target)
        chains.append(
            {
                "actions": [
                    ["Move", name],
                    ["Reach", name],
                    ["Grasp", name],
                    ["Move", target],
                    ["Place", target],
                ],
                "guard": guard,
            }
        )
    if status.get("is_activated") is True:
        chains.append({"actions": [["Move", name], ["Interact", name]], "guard": guard})
    return chains or None


# ------------------------------------------------------------------------ scheduler

def schedule(
    metadata: Dict[str, Any],
    plans: Dict[str, List[Dict[str, Any]]],
    SimEnv,
    Checker,
    entities,
    cap: int = STEP_CAP,
) -> Optional[List[Dict[str, Any]]]:
    """Advance every robot one action per step, letting the judge's checker decide.

    A robot holding several chains tries them in order and runs the first whose next
    action is legal now, which is what splices the cabinet open into a carrying trip.
    A robot with nothing legal to do stands still and tries again, so the waiting a
    task needs appears on its own: a robot bound for a closed cabinet cannot place
    until someone opens it, and idles exactly that long with no rule about waiting
    written anywhere here.
    """
    env = SimEnv(metadata=copy.deepcopy(metadata))
    checker = Checker()
    Position = entities.Position
    queues = {robot: [copy.deepcopy(chain) for chain in chains] for robot, chains in plans.items()}
    # `place` releases everything the robot is holding at once (env.py:sim_step), so a
    # robot part-way through one carry cannot start another: the first place would put
    # both objects down together and the second chain would arrive empty-handed. Its
    # hands therefore belong to one carrying chain from that chain's grasp until its
    # place. Reaching and opening stay free, which is what lets a two-armed robot open
    # the cabinet mid-trip while still holding the pumpkin.
    holding: Dict[str, Optional[int]] = {robot: None for robot in metadata["agents"]}
    # A coordinated operator names its runners by role. The schedule decides which robot
    # takes which role, so the mapping is read off the assignment and the precedence
    # between roles is enforced by counting how much of each role has actually run.
    role_robot = {
        (chain["group"], chain["role"]): robot
        for robot, group in plans.items()
        for chain in group
        if chain.get("group") is not None
    }
    progress: Dict[Tuple[str, int], int] = defaultdict(int)

    def resolve(target: str, operation: str):
        if target in env.agents:
            return env.agents[target]
        if target in env.assets:
            return env.assets[target]
        if operation in ("move", "place"):
            return Position(name=target)
        return None

    steps: List[Dict[str, Any]] = []
    while any(chain["actions"] for chains in queues.values() for chain in chains):
        if len(steps) >= cap:
            return None
        step_actions: Dict[str, List[str]] = {}
        step_commands: List[List[Any]] = []
        for robot in metadata["agents"]:
            agent = env.agents[robot]
            for position, chain in enumerate(queues.get(robot, [])):
                # A move to where the robot already stands costs a step and buys
                # nothing, so it is dropped rather than scheduled.
                while (
                    chain["actions"]
                    and chain["actions"][0][0] == "Move"
                    and len(chain["actions"][0]) == 2
                    and agent.pos.name == chain["actions"][0][1]
                ):
                    chain["actions"].pop(0)
                if not chain["actions"]:
                    continue
                verb, *targets = chain["actions"][0]
                if chain.get("after"):
                    if any(
                        progress[(chain["group"], role)] < count
                        for role, count in chain["after"][0]
                    ):
                        continue
                    targets = [
                        role_robot.get((chain["group"], int(item[2:])), item)
                        if item.startswith("?r") else item
                        for item in targets
                    ]
                    if any(item.startswith("?") for item in targets):
                        continue
                if (
                    verb in ("Grasp", "Place")
                    and holding[robot] is not None
                    and holding[robot] != position
                ):
                    continue
                if len(chain["actions"]) == 1 and any(
                    not requirement_holds(env, predicate) for predicate in chain["guard"]
                ):
                    continue  # the stage this completes is not allowed to land yet
                operation = verb.lower()
                resolved = [resolve(item, operation) for item in targets]
                if any(item is None for item in resolved):
                    continue
                params = [agent] + resolved
                # Placing onto a plain asset sets the carried object's position to that
                # asset rather than to a Position (env.py:sim_step), so a later reach
                # for it walks into an attribute that is not there and the checker
                # raises. The judge wraps the same call and scores such a plan zero, so
                # a raised exception here means exactly what a refusal means: this
                # action cannot be part of a plan that scores.
                try:
                    if not checker.check_operation(operation, params, env.assets, env.agents):
                        continue
                    candidate = step_commands + [[operation] + params]
                    if not checker.check_compatible_constraints(candidate, env.assets, env.agents):
                        continue
                except Exception:
                    continue
                step_commands = candidate
                step_actions[robot] = [verb, *targets]
                chain["_taken"] = True
                break
        if not step_commands:
            return None
        for robot in step_actions:
            for position, chain in enumerate(queues[robot]):
                if chain.pop("_taken", False):
                    verb = chain["actions"].pop(0)[0]
                    if chain.get("after"):
                        chain["after"].pop(0)
                        progress[(chain["group"], chain["role"])] += 1
                    if verb == "Grasp":
                        holding[robot] = position
                    elif verb == "Place":
                        holding[robot] = None
                    break
        try:
            env.sim_step(step_commands)
        except Exception:
            return None
        steps.append({"step": len(steps) + 1, "actions": step_actions})
    return steps


def compose(
    truth_without_plan: Dict[str, Any],
    viki2,
    SimEnv,
    Checker,
    entities,
    seed: int = SEED,
    relay: bool = True,
    blind_state: bool = False,
    operators: Optional[List[Dict[str, Any]]] = None,
    coordinated: bool = False,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Requirements -> chains -> the shortest schedule any assignment produces.

    With an induced library this runs twice. The first pass offers only the bodies a
    single robot ran end to end, which is what nearly every requirement needs and is
    cheap to search. Only when that yields nothing are the coordinated operators
    brought in, and they are expensive precisely because they are several chains that
    have to be laid out against each other. Simplest sufficient operator first, and the
    cost of the relay is paid only by the tasks that are actually relays.
    """
    metadata = build_metadata(truth_without_plan, viki2, seed)
    env = SimEnv(metadata=copy.deepcopy(metadata))

    choices: List[List[List[Dict[str, Any]]]] = []
    for requirement in collect_requirements(metadata):
        # Skipping a requirement that already holds is the one decision here that
        # needs to know where things are, and the language arms are not told that --
        # they have to read it off the image. `blind_state` drops the skip and plans
        # the work anyway, which costs steps but owes perception nothing, and prices
        # how much of this result rests on being handed the initial positions.
        if not blind_state and requirement_holds(env, requirement["predicate"]):
            continue
        if operators is not None:
            produced = induced_chains_for(env, requirement, operators, coordinated=coordinated)
        else:
            written = chains_for(env, requirement, relay)
            produced = [written] if written else None
        if not produced:
            return None, "UNSUPPORTED_PREDICATE"
        choices.append(produced)
    if not choices:
        return [], "ALREADY_SATISFIED"

    robots = list(metadata["agents"])
    best: Optional[List[Dict[str, Any]]] = None
    examined = 0
    for selection in product(*choices):
        chains, seen = [], set()
        for group in selection:
            for chain in group:
                key = json.dumps(chain, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    chains.append(chain)
        if not chains:
            continue
        floor = max(len(chain["actions"]) for chain in chains)
        for order in permutations(range(len(chains))):
            ordered = [chains[i] for i in order]
            for assignment in product(robots, repeat=len(chains)):
                examined += 1
                if examined > MAX_CANDIDATES:
                    break
                taken: Dict[Tuple[str, int], str] = {}
                clash = False
                for chain, robot in zip(ordered, assignment):
                    if chain.get("group") is None:
                        continue
                    if taken.get((chain["group"], robot)) is not None:
                        clash = True
                        break
                    taken[(chain["group"], robot)] = chain["role"]
                if clash:
                    continue
                plans: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for chain, robot in zip(ordered, assignment):
                    plans[robot].append(chain)
                steps = schedule(metadata, dict(plans), SimEnv, Checker, entities)
                if steps is not None and (best is None or len(steps) < len(best)):
                    best = steps
                    if len(best) <= floor:
                        break
            if (best is not None and len(best) <= floor) or examined > MAX_CANDIDATES:
                break
        if examined > MAX_CANDIDATES:
            break
    if best is None:
        return None, "NO_SCHEDULE"
    return best, "OK"


def plan_for(
    truth_without_plan: Dict[str, Any],
    viki2,
    SimEnv,
    Checker,
    entities,
    seed: int = SEED,
    relay: bool = True,
    blind_state: bool = False,
    operators: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Single-runner bodies first; coordinated ones only if those cannot do it."""
    plan, reason = compose(
        truth_without_plan, viki2, SimEnv, Checker, entities, seed, relay, blind_state,
        operators, coordinated=False,
    )
    if plan or operators is None:
        return plan, reason
    return compose(
        truth_without_plan, viki2, SimEnv, Checker, entities, seed, relay, blind_state,
        operators, coordinated=True,
    )


# -------------------------------------------------------------------------- scoring

def score(scorer, plan: List[Dict[str, Any]], truth: Dict[str, Any], seed: int) -> Tuple[float, float]:
    response = f"<think>composed</think><answer>{plan!r}</answer>"
    evaluator_globals = scorer.eval_single.__globals__
    original = evaluator_globals["random"]
    try:
        evaluator_globals["random"] = random.Random(seed)
        accuracy = float(scorer.acc_reward(response, truth))
    finally:
        evaluator_globals["random"] = original
    return accuracy, float(scorer.format_reward(response))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="probe1a")
    parser.add_argument("--induced", action="store_true",
                        help="plan with the induced operator library instead of the written macros")
    parser.add_argument("--shuffle-goals", action="store_true",
                        help="control: plan for another row's goals in this row's scene, "
                             "scored against this row's truth; a composer that is really "
                             "driven by the goal predicates has to collapse here")
    parser.add_argument("--blind-state", action="store_true",
                        help="plan every requirement, without checking what already holds")
    parser.add_argument("--no-relay", action="store_true",
                        help="drop the push-relay macro, to price the general library alone")
    arguments = parser.parse_args()

    scorer, SimEnv, Checker, entities, Eval, viki2 = load_sim()
    operators = load_operators() if arguments.induced else None
    print(f"operators: {'induced, ' + str(len(operators)) + ' usable' if operators else 'hand-written'}")
    indices = [json.loads(line)["index"] for line in MANIFEST.read_text().splitlines() if line.strip()]
    if arguments.limit:
        indices = indices[: arguments.limit]
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")

    # Self-check: the reference plan must score under this harness. If it does not,
    # the harness is wrong and every number below is noise.
    reference_failures = 0
    for index in indices[:40]:
        truth = bench.get_ground_truth(bench.to_native(frame.iloc[index].to_dict()))
        oracle = [
            {"step": step["step"], "actions": {r: a for r, a in step["actions"].items() if a is not None}}
            for step in truth["time_steps"]
        ]
        if score(scorer, oracle, truth, arguments.seed)[0] != 1.0:
            reference_failures += 1
    print(f"self-check: reference plan rejected on {reference_failures}/40 rows")

    records = []
    reasons = Counter()
    by_family = defaultdict(lambda: [0, 0])
    for position, index in enumerate(indices):
        truth = bench.get_ground_truth(bench.to_native(frame.iloc[index].to_dict()))
        blind = {k: v for k, v in truth.items() if k != "time_steps"}
        if arguments.shuffle_goals:
            donor = bench.get_ground_truth(
                bench.to_native(frame.iloc[indices[(position + 457) % len(indices)]].to_dict())
            )
            blind = dict(blind)
            blind["goal_constraints"] = donor["goal_constraints"]
            blind["temporal_constraints"] = donor["temporal_constraints"]
        plan, reason = plan_for(
            blind,
            viki2,
            SimEnv,
            Checker,
            entities,
            arguments.seed,
            not arguments.no_relay,
            arguments.blind_state,
            operators,
        )
        accuracy = 0.0
        formatted = 0.0
        if plan:
            accuracy, formatted = score(scorer, plan, truth, arguments.seed)
        budget = len(truth["time_steps"])
        length = len(plan) if plan else 0
        if plan and accuracy == 0.0:
            reason = "OVER_BUDGET" if length > budget else "GOAL_UNMET"
        outcome = "SOLVED" if accuracy == 1.0 else reason
        reasons[outcome] += 1
        family = truth["task_name"]
        by_family[family][1] += 1
        by_family[family][0] += int(accuracy == 1.0)
        records.append(
            {
                "index": index,
                "task_name": family,
                "accuracy": accuracy,
                "format": formatted,
                "plan_len": length,
                "budget": budget,
                "reason": outcome,
            }
        )
        if (position + 1) % 200 == 0:
            solved = sum(r["accuracy"] for r in records)
            print(f"  {position + 1}/{len(indices)}  solved={solved:.0f} ({solved / len(records) * 100:.1f}%)")

    frame_out = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    frame_out.to_csv(OUT / f"{arguments.tag}.csv", index=False)

    solved = int(frame_out["accuracy"].sum())
    print(f"\n=== {arguments.tag}: hand-written macros + oracle goals, {len(indices)} rows ===")
    print(f"accuracy       {solved}/{len(indices)} = {solved / len(indices) * 100:.2f}%")
    print(f"format         {frame_out['format'].mean() * 100:.2f}%")
    print("\noutcome:")
    for reason, count in reasons.most_common():
        print(f"  {reason:<22} {count:>5}  {count / len(indices) * 100:5.1f}%")
    print("\nby family:")
    for family, (hit, total) in sorted(by_family.items(), key=lambda kv: -kv[1][1]):
        print(f"  {family:<48} {hit:>4}/{total:<4} {hit / total * 100:5.1f}%")
    fits = frame_out[frame_out["plan_len"] > 0]
    if len(fits):
        print(
            f"\nplan length vs budget: composed mean {fits['plan_len'].mean():.2f}, "
            f"reference mean {fits['budget'].mean():.2f}, "
            f"within budget {(fits['plan_len'] <= fits['budget']).mean() * 100:.1f}%"
        )
    print(f"\nwrote {OUT / (arguments.tag + '.csv')}")


if __name__ == "__main__":
    main()
