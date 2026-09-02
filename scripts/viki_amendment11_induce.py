#!/usr/bin/env python3
"""Layer 1: induce the operator library by replaying training plans in the simulator.

The hand-written macros of probe 1 are what a reviewer will point at, and rightly: they
cover exactly the eight families of this split, so a held-out family proves nothing about
them. This derives the same kind of object from data instead.

Each training episode carries a reference plan and the predicates that plan is meant to
make true. Replaying it against the benchmark's own SimEnv says exactly when a predicate
turns true and which robot's action turned it; walking back from that action to the
runner's previous finished piece of work gives the shortest run of actions that produced
the effect -- an operator whose body is observed rather than written.

Arguments are then abstracted. The effect's subject and target become variables, and any
other object the body names becomes a further variable carrying the properties it had
when it was bound, so a body that reached into a container is not later handed a pear.
The state before the run becomes preconditions, and the primitives used become the
capability its runner needs.

What this buys over writing the macros out: an operator induced from `cut_fruit` and one
induced from `clear_table` collapse into the same entry, so the library a held-out family
is planned with can be built without ever seeing that family. That is the claim this line
rests on, and it is only worth making if the library is built this way.

Nothing here reads the test split.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "our_method"))

import pandas as pd

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment11_composer import (
    OUT,
    SEED,
    build_metadata,
    flatten_predicates,
    load_sim,
    requirement_holds,
)

DATA = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2"


# ------------------------------------------------------------------ what to attribute

def tracked_predicates(truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every predicate the episode is judged on, goals and temporal stages alike."""
    seen, out = set(), []
    for source in (truth.get("goal_constraints") or [], truth.get("temporal_constraints") or []):
        for predicate in flatten_predicates(source):
            if predicate.get("type") != "asset" or not predicate.get("is_satisfied", True):
                continue
            status = {k: v for k, v in (predicate.get("status") or {}).items() if v is not None}
            if not status:
                continue
            key = json.dumps([predicate["name"], status], sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append({"type": "asset", "name": predicate["name"], "is_satisfied": True, "status": status})
    return out


def properties(entity) -> Dict[str, bool]:
    """The properties a variable's binding relied on, and must preserve when rebound."""
    container = getattr(entity, "container_position", None)
    return {
        "is_container": bool(getattr(entity, "is_container", False)),
        "isolated": bool(container is not None and container.isolated),
        "pushable": entity.name in {"box", "cardboardbox"},
    }


# ------------------------------------------------------------------------- the replay

def replay(truth: Dict[str, Any], viki2, SimEnv, Checker, entities, seed: int):
    """Run the reference plan, recording per step what turned true and who turned it."""
    metadata = build_metadata({k: v for k, v in truth.items() if k != "time_steps"}, viki2, seed)
    env = SimEnv(metadata=copy.deepcopy(metadata))
    start = SimEnv(metadata=copy.deepcopy(metadata))
    checker = Checker()
    Position = entities.Position

    outstanding = [p for p in tracked_predicates(truth) if not requirement_holds(env, p)]
    history: List[Dict[str, Any]] = []
    completions: List[Tuple[int, str, Dict[str, Any]]] = []
    states: List[Any] = []

    for step in truth.get("time_steps") or []:
        actions = {r: a for r, a in (step.get("actions") or {}).items() if a is not None}
        if not actions:
            continue
        states.append(copy.deepcopy(env))
        commands, carried_before = [], {}
        for robot, action in actions.items():
            verb, targets = action[0], list(action[1:])
            agent = env.agents.get(robot)
            if agent is None:
                return None, "UNKNOWN_ROBOT"
            carried_before[robot] = [item.name for item in agent.get_carried_objects()]
            resolved = []
            for target in targets:
                if target in env.agents:
                    resolved.append(env.agents[target])
                elif target in env.assets:
                    resolved.append(env.assets[target])
                elif verb.lower() in ("move", "place"):
                    resolved.append(Position(name=target))
                else:
                    return None, "UNKNOWN_TARGET"
            if not checker.check_operation(verb.lower(), [agent] + resolved, env.assets, env.agents):
                return None, "REFERENCE_INFEASIBLE"
            commands.append([verb.lower(), agent] + resolved)
        env.sim_step(commands)
        history.append({"actions": actions, "carried_before": carried_before})
        index = len(history) - 1
        for predicate in list(outstanding):
            if not requirement_holds(env, predicate):
                continue
            outstanding.remove(predicate)
            actor = None
            for robot, action in actions.items():
                if predicate["name"] in list(action[1:]) or predicate["name"] in carried_before[robot]:
                    actor = robot
                    break
            completions.append((index, actor, predicate))
    return (
        {"metadata": metadata, "start": start, "history": history, "states": states,
         "completions": completions, "unmet": outstanding},
        "OK",
    )


def segment_bounds(completions, index: int, actor: str) -> int:
    """Where the runner's current piece of work began: just after its previous one."""
    start = 0
    for earlier_index, earlier_actor, _ in completions:
        if earlier_actor == actor and earlier_index < index:
            start = max(start, earlier_index + 1)
    return start


def run_alone(state, history, start: int, index: int, actor: str, predicate, SimEnv, Checker, entities) -> bool:
    """Would the runner's own actions, by themselves, have made the predicate true?

    This is the test that separates a piece of work one robot did from a piece of work
    that only looked like one robot's. In the relay the arm that finally places the pear
    does three plain actions, and replaying just those three achieves nothing, because a
    legged robot had to push the box within its reach first. A false here is the signal
    to induce the whole window as one coordinated operator instead.
    """
    env = copy.deepcopy(state)
    checker = Checker()
    Position = entities.Position
    for step in range(start, index + 1):
        action = history[step]["actions"].get(actor)
        if action is None:
            continue
        verb, targets = action[0], list(action[1:])
        agent = env.agents[actor]
        resolved = []
        for target in targets:
            if target in env.agents:
                resolved.append(env.agents[target])
            elif target in env.assets:
                resolved.append(env.assets[target])
            elif verb.lower() in ("move", "place"):
                resolved.append(Position(name=target))
            else:
                return False
        try:
            if not checker.check_operation(verb.lower(), [agent] + resolved, env.assets, env.agents):
                return False
            env.sim_step([[verb.lower(), agent] + resolved])
        except Exception:
            return False
    return requirement_holds(env, predicate)


def abstract_roles(window, predicate, env) -> Optional[Dict[str, Any]]:
    """One operator over several runners, with the order they were seen in kept.

    Roles are variables like everything else: the body says a robot that can push moves
    the container to whichever role holds the object, not that R2 does. What has to be
    preserved is the interleaving, so each action records how many of every other role's
    actions had already happened when it ran -- which is the relay's real content, and
    the part a trajectory written out as text cannot be executed from.
    """
    status = predicate["status"]
    mapping: Dict[str, str] = {predicate["name"]: "?x"}
    types: Dict[str, Dict[str, bool]] = {}
    if predicate["name"] in env.assets:
        types["?x"] = properties(env.assets[predicate["name"]])
    target = status.get("pos.name")
    if isinstance(target, str) and target not in mapping:
        mapping[target] = "?y"
        types["?y"] = (
            properties(env.assets[target]) if target in env.assets
            else {"is_container": False, "isolated": False, "pushable": False}
        )
    for position, role in enumerate(window):
        mapping[role["robot"]] = f"?r{position}"
    spare = 0
    roles = []
    for position, role in enumerate(window):
        actions = []
        for offset, action in role["acts"]:
            verb, targets = action[0], list(action[1:])
            rewritten = [verb]
            for item in targets:
                if item not in mapping:
                    if item in env.assets:
                        spare += 1
                        mapping[item] = f"?z{spare}"
                        types[mapping[item]] = properties(env.assets[item])
                    else:
                        mapping[item] = item
                rewritten.append(mapping[item])
            actions.append({"action": rewritten, "offset": offset})
        roles.append({"variable": f"?r{position}", "actions": actions})

    # Each action remembers how much of every other role had already run.
    for position, role in enumerate(roles):
        for slot, entry in enumerate(role["actions"]):
            after = []
            for other, other_role in enumerate(roles):
                if other == position:
                    continue
                count = sum(1 for item in other_role["actions"] if item["offset"] < entry["offset"])
                if count:
                    after.append([other, count])
            entry["after"] = after
    if "pos.name" in status:
        effect = {"key": "pos.name", "subject": "?x", "value": "?y"}
    elif status.get("is_activated") is True:
        effect = {"key": "is_activated", "subject": "?x", "value": True}
    else:
        return None
    return {"effect": effect, "roles": roles, "types": types}


def abstract(body, predicate, env) -> Optional[Dict[str, Any]]:
    """Rewrite a body over variables, keeping what each binding's properties were."""
    status = predicate["status"]
    mapping: Dict[str, str] = {predicate["name"]: "?x"}
    types: Dict[str, Dict[str, bool]] = {}
    if predicate["name"] in env.assets:
        types["?x"] = properties(env.assets[predicate["name"]])
    target = status.get("pos.name")
    if isinstance(target, str):
        if target not in mapping:
            mapping[target] = "?y"
        types["?y"] = (
            properties(env.assets[target]) if target in env.assets
            else {"is_container": False, "isolated": False, "pushable": False}
        )
    spare = 0
    abstracted: List[List[str]] = []
    for action in body:
        verb, targets = action[0], list(action[1:])
        rewritten = [verb]
        for item in targets:
            if item not in mapping:
                if item in env.assets:
                    spare += 1
                    mapping[item] = f"?z{spare}"
                    types[mapping[item]] = properties(env.assets[item])
                elif item in env.agents:
                    mapping[item] = f"?agent:{item}"
                else:
                    mapping[item] = item  # a free-form place keeps its own name
            rewritten.append(mapping[item])
        abstracted.append(rewritten)
    if "pos.name" in status:
        effect = {"key": "pos.name", "subject": "?x", "value": "?y"}
    elif status.get("is_activated") is True:
        effect = {"key": "is_activated", "subject": "?x", "value": True}
    else:
        return None
    return {"effect": effect, "body": abstracted, "types": types}


def preconditions_at(state, predicate, env_at_start) -> Dict[str, bool]:
    """The few facts about the bound objects that tell the body variants apart."""
    assets = state.assets
    agents = state.agents
    name = predicate["name"]
    target = predicate["status"].get("pos.name")
    subject = assets.get(name)
    holder = assets.get(subject.pos.name) if subject is not None and subject.pos.name in assets else None
    destination = assets.get(target) if isinstance(target, str) else None
    return {
        "subject_sealed": bool(subject is not None and subject.pos.isolated),
        "subject_in_container": bool(holder is not None and getattr(holder, "is_container", False)),
        "subject_on_agent": bool(subject is not None and subject.pos.name in agents),
        "target_sealed": bool(
            destination is not None
            and getattr(destination, "container_position", None) is not None
            and destination.container_position.isolated
        ),
        "target_on_agent": bool(destination is not None and destination.pos.name in agents),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-family", type=int, default=250)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default="operators.json")
    parser.add_argument("--exclude-family", default=None,
                        help="induce as if this family had never been seen (held-out fold)")
    arguments = parser.parse_args()

    scorer, SimEnv, Checker, entities, Eval, viki2 = load_sim()
    train = pd.read_parquet(DATA / "train.parquet")
    total = len(train) if arguments.limit is None else min(len(train), arguments.limit)

    library: Dict[str, Dict[str, Any]] = {}
    seen_family = Counter()
    outcomes = Counter()
    unattributed = Counter()
    unmet_families = Counter()

    for position in range(total):
        truth = bench.get_ground_truth(bench.to_native(train.iloc[position].to_dict()))
        if not isinstance(truth, dict) or not truth.get("time_steps"):
            outcomes["NO_PLAN"] += 1
            continue
        family = truth.get("task_name", "?")
        if arguments.exclude_family and family == arguments.exclude_family:
            continue
        if seen_family[family] >= arguments.per_family:
            continue
        seen_family[family] += 1
        trace, status = replay(truth, viki2, SimEnv, Checker, entities, arguments.seed)
        outcomes[status] += 1
        if trace is None:
            continue
        if trace["unmet"]:
            unmet_families[family] += 1
        # Opening a container completes no goal of its own, so goal attribution never
        # reaches it, and the only bodies that carry an Open are the ones where the same
        # robot went on to finish something. That is why holding out the family whose
        # robots open the destination loses the ability entirely, even though three other
        # families open containers all the time -- there, the open sits inside a fetch.
        # Mined on its own, it is the same short piece of work in all four: address the
        # container, reach it, open it. It is stored as an operator whose effect is that
        # the container is no longer shut.
        for index, entry in enumerate(trace["history"]):
            for robot, action in entry["actions"].items():
                if action[0] != "Open" or len(action) < 2:
                    continue
                container = action[1]
                body = []
                for step in range(index, -1, -1):
                    earlier = trace["history"][step]["actions"].get(robot)
                    if earlier is None or container not in list(earlier[1:]):
                        break
                    body.insert(0, list(earlier))
                if not body or container not in trace["states"][index - len(body) + 1].assets:
                    continue
                state = trace["states"][index - len(body) + 1]
                abstracted = [
                    [item[0]] + ["?x" if target == container else target for target in item[1:]]
                    for item in body
                ]
                if any(not target.startswith("?") for item in abstracted for target in item[1:]):
                    continue
                key = json.dumps(["unsealed", abstracted], sort_keys=True)
                repair = library.setdefault(
                    key,
                    {
                        "effect": {"key": "unsealed", "subject": "?x", "value": True},
                        "body": abstracted,
                        "preconditions": {"target_sealed": True},
                        "types": {"?x": properties(state.assets[container])},
                        "requires": sorted({item[0].lower() for item in abstracted}),
                        "carries": False,
                        "cost": len(abstracted),
                        "support": 0,
                        "families": [],
                        "runner_types": [],
                    },
                )
                repair["support"] += 1
                if family not in repair["families"]:
                    repair["families"].append(family)
                kind = trace["metadata"]["agents"][robot]["type"]
                if kind not in repair["runner_types"]:
                    repair["runner_types"].append(kind)

        for index, actor, predicate in trace["completions"]:
            if actor is None:
                unattributed[family] += 1
                continue
            start = segment_bounds(trace["completions"], index, actor)
            body = [
                list(trace["history"][step]["actions"][actor])
                for step in range(start, index + 1)
                if actor in trace["history"][step]["actions"]
            ]
            if not body:
                continue
            state = trace["states"][start]
            pre = preconditions_at(state, predicate, trace["start"])
            if not run_alone(state, trace["history"], start, index, actor, predicate,
                             SimEnv, Checker, entities):
                window = []
                for robot in sorted({
                    name for step in range(start, index + 1) for name in trace["history"][step]["actions"]
                }):
                    acts = [
                        (step, list(trace["history"][step]["actions"][robot]))
                        for step in range(start, index + 1)
                        if robot in trace["history"][step]["actions"]
                    ]
                    if acts:
                        window.append({"robot": robot, "acts": acts})
                shape = abstract_roles(window, predicate, state)
                if shape is None:
                    continue
                key = json.dumps([shape["effect"], shape["roles"], pre], sort_keys=True)
                entry = library.setdefault(
                    key,
                    {
                        "effect": shape["effect"],
                        "roles": shape["roles"],
                        "preconditions": pre,
                        "types": shape["types"],
                        "cost": max(
                            item["offset"] for role in shape["roles"] for item in role["actions"]
                        ) - start + 1,
                        "support": 0,
                        "families": [],
                        "runner_types": [],
                        "coordinated": True,
                    },
                )
                entry["support"] += 1
                if family not in entry["families"]:
                    entry["families"].append(family)
                for role, source in zip(shape["roles"], window):
                    kind = trace["metadata"]["agents"][source["robot"]]["type"]
                    entry.setdefault("role_types", {}).setdefault(role["variable"], [])
                    if kind not in entry["role_types"][role["variable"]]:
                        entry["role_types"][role["variable"]].append(kind)
                continue
            shape = abstract(body, predicate, state)
            if shape is None:
                continue
            key = json.dumps(
                [shape["effect"], shape["body"], pre], sort_keys=True
            )
            entry = library.setdefault(
                key,
                {
                    "effect": shape["effect"],
                    "body": shape["body"],
                    "preconditions": pre,
                    "types": shape["types"],
                    "requires": sorted({action[0].lower() for action in shape["body"]}),
                    "carries": any(action[0] == "Grasp" for action in shape["body"]),
                    "cost": len(shape["body"]),
                    "support": 0,
                    "families": [],
                    "runner_types": [],
                },
            )
            entry["support"] += 1
            if family not in entry["families"]:
                entry["families"].append(family)
            runner = trace["metadata"]["agents"][actor]["type"]
            if runner not in entry["runner_types"]:
                entry["runner_types"].append(runner)
            # A property only generalises if it held every time; keep the intersection.
            for variable, observed in shape["types"].items():
                kept = entry["types"].setdefault(variable, dict(observed))
                for prop, value in list(kept.items()):
                    if observed.get(prop) != value:
                        kept[prop] = None

    operators = sorted(library.values(), key=lambda item: -item["support"])
    record = {
        "built_from": f"VIKI-L2 train.parquet, up to {arguments.per_family} episodes per family",
        "excluded_family": arguments.exclude_family,
        "episodes_replayed": sum(seen_family.values()),
        "replay_outcomes": dict(outcomes),
        "operators": operators,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / arguments.out).write_text(json.dumps(record, indent=2) + "\n")

    print(f"episodes replayed   {sum(seen_family.values())}")
    print(f"replay outcomes     {dict(outcomes)}")
    print(f"episodes whose reference plan left a tracked predicate unmet: {sum(unmet_families.values())}")
    if unmet_families:
        print(f"  {dict(unmet_families)}")
    if unattributed:
        print(f"completions with no attributable runner: {dict(unattributed)}")
    single = [item for item in operators if not item.get("coordinated")]
    coordinated = [item for item in operators if item.get("coordinated")]
    print(f"\ndistinct operators  {len(operators)}  "
          f"({len(single)} single-runner, {len(coordinated)} coordinated)")
    for operator in coordinated:
        effect = operator["effect"]
        written = (f"{effect['subject']} @ {effect['value']}" if effect["key"] == "pos.name"
                   else f"{effect['subject']} activated")
        pre = ", ".join(k for k, v in operator["preconditions"].items() if v) or "-"
        print(f"\n  COORDINATED  effect {written}   makespan {operator['cost']}   "
              f"support {operator['support']}")
        print(f"  when     {pre}")
        for role in operator["roles"]:
            line = "  ".join(
                "[" + " ".join(item["action"]) + "]"
                + ("" if not item["after"] else "^" + str(item["after"]))
                for item in role["actions"]
            )
            print(f"  {role['variable']:<5} {operator.get('role_types', {}).get(role['variable'], [])}  {line}")
        print(f"  families {operator['families']}")
    for operator in single:
        body = "  ".join("[" + " ".join(action) + "]" for action in operator["body"])
        effect = operator["effect"]
        written = (
            f"{effect['subject']} @ {effect['value']}" if effect["key"] == "pos.name"
            else f"{effect['subject']} activated"
        )
        pre = ", ".join(k for k, v in operator["preconditions"].items() if v) or "-"
        print(f"\n  effect   {written}   cost {operator['cost']}   support {operator['support']}")
        print(f"  when     {pre}")
        print(f"  body     {body}")
        print(f"  runners  {operator['runner_types']}   uses {operator['requires']}")
        print(f"  families {len(operator['families'])}: {operator['families']}")
    print(f"\nwrote {OUT / arguments.out}")


if __name__ == "__main__":
    main()
