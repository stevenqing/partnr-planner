"""Layer 1: operators, induced by replaying training plans in the simulator.

An operator is a body of primitive actions over variables, an effect it brings about,
the world facts that held when it was seen to work, the capability its runner needs and
what it costs in steps. Three kinds are mined, and the three exist because three
different things went missing when only the first was:

  achievement   the shortest run of one robot's actions that made a required predicate
                true. Nearly everything is this.
  repair        work that completes no requirement of its own but unblocks somebody
                else's -- opening a container being the whole of it here. Goal
                attribution never reaches this, because there is no goal to attribute
                it to, and a memory without it cannot plan any task whose destination
                happens to be shut.
  coordination  a requirement no single robot achieved alone. The test is literal:
                replay that robot's actions by themselves and see whether the predicate
                still turns true. When it does not, the whole window is kept as one
                operator with roles, and each action remembers how much of every other
                role had already run -- which is the part a trajectory written out as
                text cannot be executed from.

Every operator carries its support and the families it was seen in, because the claim
this memory makes is about transfer, and transfer is only checkable if provenance
survives induction.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .simulator import (
    Simulator,
    flatten_predicates,
    holds,
    object_properties,
    predicate_status,
    state_facts,
)


def requirements_of(truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every predicate the episode is judged on, goals and temporal stages alike."""
    seen, out = set(), []
    for source in (truth.get("goal_constraints") or [], truth.get("temporal_constraints") or []):
        for predicate in flatten_predicates(source):
            if predicate.get("type") != "asset" or not predicate.get("is_satisfied", True):
                continue
            status = predicate_status(predicate)
            if not status:
                continue
            key = json.dumps([predicate["name"], status], sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append({"type": "asset", "name": predicate["name"], "is_satisfied": True,
                            "status": status})
    return out


def _resolve(env, verb: str, targets: List[str], entities):
    out = []
    for target in targets:
        if target in env.agents:
            out.append(env.agents[target])
        elif target in env.assets:
            out.append(env.assets[target])
        elif verb.lower() in ("move", "place"):
            out.append(entities.Position(name=target))
        else:
            return None
    return out


def replay(truth: Dict[str, Any], sim: Simulator, seed: int):
    """Run the reference plan, recording per step what turned true and who turned it."""
    metadata = sim.metadata({k: v for k, v in truth.items() if k != "time_steps"}, seed)
    env = sim.world(metadata)
    checker = sim.Checker()
    outstanding = [p for p in requirements_of(truth) if not holds(env, p)]
    history: List[Dict[str, Any]] = []
    states: List[Any] = []
    completions: List[Tuple[int, Optional[str], Dict[str, Any]]] = []

    for step in truth.get("time_steps") or []:
        actions = {r: a for r, a in (step.get("actions") or {}).items() if a is not None}
        if not actions:
            continue
        states.append(copy.deepcopy(env))
        commands, carried = [], {}
        for robot, action in actions.items():
            agent = env.agents.get(robot)
            if agent is None:
                return None, "UNKNOWN_ROBOT"
            carried[robot] = [item.name for item in agent.get_carried_objects()]
            resolved = _resolve(env, action[0], list(action[1:]), sim.entities)
            if resolved is None:
                return None, "UNKNOWN_TARGET"
            if not checker.check_operation(action[0].lower(), [agent] + resolved, env.assets, env.agents):
                return None, "REFERENCE_INFEASIBLE"
            commands.append([action[0].lower(), agent] + resolved)
        env.sim_step(commands)
        history.append({"actions": actions, "carried_before": carried})
        index = len(history) - 1
        for predicate in list(outstanding):
            if not holds(env, predicate):
                continue
            outstanding.remove(predicate)
            actor = None
            for robot, action in actions.items():
                if predicate["name"] in list(action[1:]) or predicate["name"] in carried[robot]:
                    actor = robot
                    break
            completions.append((index, actor, predicate))
    return {"metadata": metadata, "history": history, "states": states,
            "completions": completions, "unmet": outstanding}, "OK"


def _segment_start(completions, index: int, actor: str) -> int:
    start = 0
    for earlier_index, earlier_actor, _ in completions:
        if earlier_actor == actor and earlier_index < index:
            start = max(start, earlier_index + 1)
    return start


def _runs_alone(state, history, start, index, actor, predicate, sim) -> bool:
    """Would the runner's own actions, by themselves, have made this true?"""
    env = copy.deepcopy(state)
    checker = sim.Checker()
    for step in range(start, index + 1):
        action = history[step]["actions"].get(actor)
        if action is None:
            continue
        resolved = _resolve(env, action[0], list(action[1:]), sim.entities)
        if resolved is None:
            return False
        try:
            if not checker.check_operation(action[0].lower(), [env.agents[actor]] + resolved,
                                           env.assets, env.agents):
                return False
            env.sim_step([[action[0].lower(), env.agents[actor]] + resolved])
        except Exception:
            return False
    return holds(env, predicate)


def _bind(predicate) -> Tuple[Dict[str, str], Dict[str, Any]]:
    status = predicate_status(predicate)
    mapping = {predicate["name"]: "?x"}
    target = status.get("pos.name")
    if isinstance(target, str) and target not in mapping:
        mapping[target] = "?y"
    effect = ({"key": "pos.name", "subject": "?x", "value": "?y"} if "pos.name" in status
              else {"key": "is_activated", "subject": "?x", "value": True}
              if status.get("is_activated") is True else None)
    return mapping, effect


def _abstract(actions, mapping, types, env, counter: List[int]) -> List[List[str]]:
    out = []
    for action in actions:
        rewritten = [action[0]]
        for item in action[1:]:
            if item not in mapping:
                if item in env.assets:
                    counter[0] += 1
                    mapping[item] = f"?z{counter[0]}"
                    types[mapping[item]] = object_properties(env.assets[item])
                elif item in env.agents:
                    mapping[item] = f"?agent:{item}"
                else:
                    mapping[item] = item
            rewritten.append(mapping[item])
        out.append(rewritten)
    return out


def induce(
    episodes,
    sim: Simulator,
    seed: int,
    per_family: int = 250,
    exclude_family: Optional[str] = None,
    progress=None,
) -> Dict[str, Any]:
    """Replay episodes and collect the operator library, with provenance."""
    library: Dict[str, Dict[str, Any]] = {}
    seen_family, outcomes = Counter(), Counter()

    def entry(key, template):
        return library.setdefault(key, dict(template, support=0, families=[], runner_types=[]))

    def credit(record, family, runner_type):
        record["support"] += 1
        if family not in record["families"]:
            record["families"].append(family)
        if runner_type and runner_type not in record["runner_types"]:
            record["runner_types"].append(runner_type)

    for position, truth in enumerate(episodes):
        if not isinstance(truth, dict) or not truth.get("time_steps"):
            outcomes["NO_PLAN"] += 1
            continue
        family = truth.get("task_name", "?")
        if exclude_family and family == exclude_family:
            continue
        if seen_family[family] >= per_family:
            continue
        seen_family[family] += 1
        trace, status = replay(truth, sim, seed)
        outcomes[status] += 1
        if trace is None:
            continue
        if progress:
            progress(position, family)

        # --- repair: the work that unblocks, which no goal will ever point at --------
        for index, step in enumerate(trace["history"]):
            for robot, action in step["actions"].items():
                if action[0] != "Open" or len(action) < 2:
                    continue
                container = action[1]
                body = []
                for earlier in range(index, -1, -1):
                    previous = trace["history"][earlier]["actions"].get(robot)
                    if previous is None or container not in list(previous[1:]):
                        break
                    body.insert(0, list(previous))
                if not body:
                    continue
                state = trace["states"][index - len(body) + 1]
                if container not in state.assets:
                    continue
                abstracted = [[item[0]] + ["?x" if t == container else t for t in item[1:]]
                              for item in body]
                if any(not t.startswith("?") for item in abstracted for t in item[1:]):
                    continue
                record = entry(
                    json.dumps(["unsealed", abstracted], sort_keys=True),
                    {
                        "kind": "repair",
                        "effect": {"key": "unsealed", "subject": "?x", "value": True},
                        "body": abstracted,
                        "preconditions": {"target_sealed": True},
                        "types": {"?x": object_properties(state.assets[container])},
                        "requires": sorted({item[0].lower() for item in abstracted}),
                        "carries": False,
                        "cost": len(abstracted),
                    },
                )
                credit(record, family, trace["metadata"]["agents"][robot]["type"])

        # --- achievement and coordination -------------------------------------------
        for index, actor, predicate in trace["completions"]:
            if actor is None:
                continue
            start = _segment_start(trace["completions"], index, actor)
            state = trace["states"][start]
            facts = state_facts(state, predicate)
            mapping, effect = _bind(predicate)
            if effect is None:
                continue
            types: Dict[str, Any] = {}
            if predicate["name"] in state.assets:
                types["?x"] = object_properties(state.assets[predicate["name"]])
            target = predicate_status(predicate).get("pos.name")
            if isinstance(target, str):
                types["?y"] = (object_properties(state.assets[target]) if target in state.assets
                               else {"is_container": False, "isolated": False, "pushable": False})
            counter = [0]

            if _runs_alone(state, trace["history"], start, index, actor, predicate, sim):
                body = [list(trace["history"][s]["actions"][actor])
                        for s in range(start, index + 1)
                        if actor in trace["history"][s]["actions"]]
                if not body:
                    continue
                abstracted = _abstract(body, mapping, types, state, counter)
                record = entry(
                    json.dumps(["achieve", effect, abstracted, facts], sort_keys=True),
                    {
                        "kind": "achievement",
                        "effect": effect,
                        "body": abstracted,
                        "preconditions": facts,
                        "types": dict(types),
                        "requires": sorted({item[0].lower() for item in abstracted}),
                        "carries": any(item[0] == "Grasp" for item in abstracted),
                        "cost": len(abstracted),
                    },
                )
                credit(record, family, trace["metadata"]["agents"][actor]["type"])
            else:
                window = []
                for robot in sorted({name for s in range(start, index + 1)
                                     for name in trace["history"][s]["actions"]}):
                    acts = [(s, list(trace["history"][s]["actions"][robot]))
                            for s in range(start, index + 1)
                            if robot in trace["history"][s]["actions"]]
                    if acts:
                        window.append({"robot": robot, "acts": acts})
                for slot, role in enumerate(window):
                    mapping[role["robot"]] = f"?r{slot}"
                roles = []
                for slot, role in enumerate(window):
                    rewritten = _abstract([act for _, act in role["acts"]], mapping, types, state, counter)
                    roles.append({
                        "variable": f"?r{slot}",
                        "actions": [{"action": action, "offset": offset}
                                    for action, (offset, _) in zip(rewritten, role["acts"])],
                    })
                for slot, role in enumerate(roles):
                    for item in role["actions"]:
                        item["after"] = [
                            [other, sum(1 for peer in roles[other]["actions"]
                                        if peer["offset"] < item["offset"])]
                            for other in range(len(roles)) if other != slot
                        ]
                        item["after"] = [pair for pair in item["after"] if pair[1]]
                record = entry(
                    json.dumps(["coordinate", effect, roles, facts], sort_keys=True),
                    {
                        "kind": "coordination",
                        "coordinated": True,
                        "effect": effect,
                        "roles": roles,
                        "preconditions": facts,
                        "types": dict(types),
                        "cost": max(item["offset"] for role in roles for item in role["actions"]) - start + 1,
                    },
                )
                credit(record, family, None)
                record.setdefault("role_types", {})
                for role, source in zip(roles, window):
                    kind = trace["metadata"]["agents"][source["robot"]]["type"]
                    record["role_types"].setdefault(role["variable"], [])
                    if kind not in record["role_types"][role["variable"]]:
                        record["role_types"][role["variable"]].append(kind)

    # A property only generalises if it held every time it was seen.
    operators = sorted(library.values(), key=lambda item: -item["support"])
    return {
        "operators": operators,
        "episodes_replayed": sum(seen_family.values()),
        "replay_outcomes": dict(outcomes),
        "families_seen": dict(seen_family),
    }
