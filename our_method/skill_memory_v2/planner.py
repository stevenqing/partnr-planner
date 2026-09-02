"""The consumer: turn requirements into a plan using nothing but the memory.

The planner is shipped with the memory rather than beside it, because an operator
library is only meaningful together with something that can execute it, and because the
thing being claimed is that this memory is executable at all. Its own contribution is
deliberately small: it picks bodies out of the memory, hands them to robots, and steps
everybody forward one action at a time. Feasibility is decided by the benchmark's own
checker on its own world model, so the planner cannot disagree with the judge about what
is legal.

Two behaviours are worth naming because they are not written anywhere as rules. A robot
whose next action is not yet legal stands still and tries again, so waiting appears by
itself: a robot bound for a shut cupboard idles exactly until somebody opens it. And a
robot holds several chains at once and advances whichever it can, so opening the cupboard
becomes a splice into a carrying trip rather than a separate errand -- which is where the
step that a step-budget scores on is saved.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from itertools import permutations, product
from typing import Any, Dict, List, Optional, Tuple

from .simulator import Simulator, flatten_predicates, holds, predicate_status, state_facts

STEP_CAP = 16
MAX_CANDIDATES = 4000


def predicate_key(predicate: Dict[str, Any]) -> str:
    return json.dumps(
        [predicate.get("type"), predicate.get("name"), predicate.get("is_satisfied", True),
         predicate_status(predicate)],
        sort_keys=True,
    )


def collect_requirements(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every predicate the judge will check, with the stages that must precede it.

    Temporal stages are requirements in their own right, not merely an order: `eval`
    demands they all hold, and a stage can name work no goal mentions -- bread in the
    toaster is only ever stated as stage one.
    """
    requirements: Dict[str, Dict[str, Any]] = {}

    def register(predicate, guard):
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
    for predicate in flatten_predicates(metadata["goal_constraints"]):
        register(predicate, [])
    return list(requirements.values())


def chains_for(env, requirement, memory, coordinated: bool, limit: int = 4):
    """Every way the memory offers to make one requirement hold, best first."""
    predicate = requirement["predicate"]
    if predicate.get("type") != "asset" or not predicate.get("is_satisfied", True):
        return None
    status = predicate_status(predicate)
    if predicate["name"] not in env.assets:
        return None
    if "pos.name" in status:
        wanted, value = "pos.name", status["pos.name"]
    elif status.get("is_activated") is True:
        wanted, value = "is_activated", True
    else:
        return None

    facts = state_facts(env, predicate)

    # A shut container is opened by an operator of its own, offered alongside the
    # delivery rather than spliced into it, so any robot with a free hand can do it.
    unseal: List[Dict[str, Any]] = []
    sealed = None
    if facts.get("target_sealed") and isinstance(value, str):
        sealed = value
    elif facts.get("subject_sealed"):
        holder = env.assets[predicate["name"]].pos.name
        sealed = holder if holder in env.assets else None
    if sealed is not None:
        repair = memory.repair_operator()
        if repair is not None:
            unseal = [{
                "actions": [[action[0]] + [sealed if t == "?x" else t for t in action[1:]]
                            for action in repair["body"]],
                "guard": [],
            }]

    alternatives: List[List[Dict[str, Any]]] = []
    for rank, operator in enumerate(memory.operators_for(wanted, facts, coordinated)):
        binding = {"?x": predicate["name"]}
        if wanted == "pos.name":
            binding["?y"] = value
        options = [binding]
        for spare in memory.spare_variables(operator):
            wants = operator.get("types", {}).get(spare, {})
            fits = [name for name, asset in env.assets.items()
                    if name not in binding.values() and memory.suits(asset, wants)]
            options = [dict(option, **{spare: name}) for option in options for name in fits[:2]]
        for number, option in enumerate(options[:2]):
            if operator.get("coordinated"):
                group = f"g{rank}:{number}"
                last = max(range(len(operator["roles"])),
                           key=lambda slot: max(item["offset"] for item in operator["roles"][slot]["actions"]))
                chains, broken = [], False
                for slot, role in enumerate(operator["roles"]):
                    actions, after = [], []
                    for item in role["actions"]:
                        verb, targets = item["action"][0], item["action"][1:]
                        # A role variable names whichever robot takes that role, and which
                        # robot that is only becomes known when the schedule is laid out.
                        bound = [t if t.startswith("?r") else option.get(t, t) for t in targets]
                        if any(t.startswith("?") and not t.startswith("?r") for t in bound):
                            broken = True
                        actions.append([verb] + bound)
                        after.append(item["after"])
                    chains.append({"actions": actions, "after": after, "group": group, "role": slot,
                                   "guard": requirement["guard"] if slot == last else []})
                if not broken:
                    alternatives.append(chains)
                continue
            body = [[action[0]] + [option.get(t, t) for t in action[1:]] for action in operator["body"]]
            if any(t.startswith("?") for action in body for t in action[1:]):
                continue
            alternatives.append(unseal + [{"actions": body, "guard": requirement["guard"]}])
        if len(alternatives) >= limit:
            break
    return alternatives or None


def schedule(metadata, plans, sim: Simulator, cap: int = STEP_CAP):
    """Advance every robot one action per step, letting the judge's checker decide."""
    env = sim.world(metadata)
    checker = sim.Checker()
    Position = sim.entities.Position
    queues = {robot: [copy.deepcopy(chain) for chain in group] for robot, group in plans.items()}
    # `place` releases everything the robot holds at once, so its hands belong to one
    # carrying chain from that chain's grasp until its place; reaching and opening stay
    # free, which is what allows the cupboard to be opened mid-trip.
    holding: Dict[str, Optional[int]] = {robot: None for robot in metadata["agents"]}
    role_robot = {(chain["group"], chain["role"]): robot
                  for robot, group in plans.items() for chain in group if chain.get("group") is not None}
    progress: Dict[Tuple[str, int], int] = defaultdict(int)

    def resolve(target, operation):
        if target in env.agents:
            return env.agents[target]
        if target in env.assets:
            return env.assets[target]
        if operation in ("move", "place"):
            return Position(name=target)
        return None

    steps: List[Dict[str, Any]] = []
    while any(chain["actions"] for group in queues.values() for chain in group):
        if len(steps) >= cap:
            return None
        step_actions, step_commands = {}, []
        for robot in metadata["agents"]:
            agent = env.agents[robot]
            for slot, chain in enumerate(queues.get(robot, [])):
                # A move to where the robot already stands costs a step and buys nothing.
                while (chain["actions"] and chain["actions"][0][0] == "Move"
                       and len(chain["actions"][0]) == 2 and agent.pos.name == chain["actions"][0][1]):
                    chain["actions"].pop(0)
                    if chain.get("after"):
                        chain["after"].pop(0)
                if not chain["actions"]:
                    continue
                verb, *targets = chain["actions"][0]
                if chain.get("after"):
                    if any(progress[(chain["group"], role)] < count for role, count in chain["after"][0]):
                        continue
                    targets = [role_robot.get((chain["group"], int(t[2:])), t) if t.startswith("?r") else t
                               for t in targets]
                    if any(t.startswith("?") for t in targets):
                        continue
                if verb in ("Grasp", "Place") and holding[robot] is not None and holding[robot] != slot:
                    continue
                if len(chain["actions"]) == 1 and any(not holds(env, p) for p in chain["guard"]):
                    continue
                operation = verb.lower()
                resolved = [resolve(t, operation) for t in targets]
                if any(item is None for item in resolved):
                    continue
                params = [agent] + resolved
                # Placing onto a plain asset sets the carried object's position to that
                # asset rather than to a Position, so a later reach for it raises inside
                # the checker. The judge wraps the same call and scores such a plan zero,
                # so a raised exception means what a refusal means.
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
            for slot, chain in enumerate(queues[robot]):
                if chain.pop("_taken", False):
                    verb = chain["actions"].pop(0)[0]
                    if chain.get("after"):
                        chain["after"].pop(0)
                        progress[(chain["group"], chain["role"])] += 1
                    if verb == "Grasp":
                        holding[robot] = slot
                    elif verb == "Place":
                        holding[robot] = None
                    break
        try:
            env.sim_step(step_commands)
        except Exception:
            return None
        steps.append({"step": len(steps) + 1, "actions": step_actions})
    return steps


def compose(truth, memory, sim: Simulator, seed: int, coordinated: bool = False,
            crew: Optional[Dict[str, str]] = None):
    """Requirements to a plan. `crew` fixes who runs what, when somebody else decided.

    Whoever names the work may also name the robot for it, and when they do the search
    over assignments collapses to their answer. That is not a convenience: it is what
    makes the division of labour measurable, since the same requirements planned with
    and without their casting say exactly what their casting was worth.
    """
    metadata = sim.metadata(truth, seed)
    env = sim.world(metadata)
    choices, wanted = [], []
    for requirement in collect_requirements(metadata):
        if holds(env, requirement["predicate"]):
            continue
        produced = chains_for(env, requirement, memory, coordinated)
        if not produced:
            return None, "UNSUPPORTED_PREDICATE"
        choices.append(produced)
        wanted.append((crew or {}).get(predicate_key(requirement["predicate"])))
    if not choices:
        return [], "ALREADY_SATISFIED"

    robots = list(metadata["agents"])
    best, examined = None, 0
    for selection in product(*choices):
        chains, seen, forced = [], set(), []
        for group, robot in zip(selection, wanted):
            for chain in group:
                key = json.dumps(chain, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    chains.append(chain)
                    # A coordinated body needs one robot per role, so a single name
                    # cannot cast it; those keep the search.
                    forced.append(robot if chain.get("group") is None else None)
        if not chains:
            continue
        floor = max(len(chain["actions"]) for chain in chains)
        for order in permutations(range(len(chains))):
            ordered = [chains[i] for i in order]
            candidates = [[forced[i]] if forced[i] else robots for i in order]
            for assignment in product(*candidates):
                examined += 1
                if examined > MAX_CANDIDATES:
                    break
                taken, clash = {}, False
                for chain, robot in zip(ordered, assignment):
                    if chain.get("group") is None:
                        continue
                    if (chain["group"], robot) in taken:
                        clash = True
                        break
                    taken[(chain["group"], robot)] = chain["role"]
                if clash:
                    continue
                plans = defaultdict(list)
                for chain, robot in zip(ordered, assignment):
                    plans[robot].append(chain)
                steps = schedule(metadata, dict(plans), sim)
                if steps is not None and (best is None or len(steps) < len(best)):
                    best = steps
                    if len(best) <= floor:
                        break
            if (best is not None and len(best) <= floor) or examined > MAX_CANDIDATES:
                break
        if examined > MAX_CANDIDATES:
            break
    return (best, "OK") if best is not None else (None, "NO_SCHEDULE")


def plan(truth, memory, sim: Simulator, seed: int, crew: Optional[Dict[str, str]] = None):
    """Single-runner bodies first; coordinated ones only if those cannot do it.

    Coordinated operators are several chains that must be laid out against each other,
    and searching them is expensive. Nearly every requirement never needs one, so the
    cost of the relay is paid only by the tasks that are actually relays.
    """
    result, reason = compose(truth, memory, sim, seed, coordinated=False, crew=crew)
    if result:
        return result, reason
    return compose(truth, memory, sim, seed, coordinated=True, crew=crew)
