#!/usr/bin/env python3
"""Run the decentralized v2 planner against a stand-in world, with no simulator.

Habitat is not importable on the machine this is usually edited on, and a planner whose
first execution is inside a two-hour rollout is a planner whose bugs are found expensively.
So the two things the planner needs from `habitat_llm` -- a base class that turns high
level actions into responses, and the world-model types it checks with `isinstance` -- are
supplied here as stubs, and the real planner module is imported on top of them unchanged.

What this exercises is exactly the part that is ours: which requirements each agent claims
without talking to the other, whether the graph query drops work the partner has already
done, whether a refusal at a shut cupboard is answered from the memory, and whether a
spatial requirement folds into the placement that carries it. What it cannot exercise is
Habitat's own skills, which is why it is a self-test and not an evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SUCCESS = "Successful execution!"


# --------------------------------------------------------------------- the stand-in world


class Node:
    def __init__(self, name, kind, room=None, states=None, components=None):
        self.name, self.kind = name, kind
        self.properties = {"type": kind, "states": dict(states or {}),
                           "components": list(components or [])}
        self.room = room

    def __repr__(self):
        return f"<{self.kind} {self.name}>"


class FakeWorld:
    """A house, plus the parts of `WorldGraph` the planner actually calls."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.container: Dict[str, str] = {}  # object -> furniture
        self.held: Dict[str, Optional[int]] = {}  # object -> agent uid holding it
        self.shut: Dict[str, bool] = {}
        self.beside: Dict[str, str] = {}
        self.at: Dict[int, Optional[str]] = {}
        # What a pick out of a shut container says. The real skills phrase this in a way
        # `_repair` recognises; a scenario can replace it with something the ladder has
        # never seen, which is the case the closed loop exists for.
        self.shut_pick_message = (
            "Failed to pick! Object is in a closed furniture, "
            "you need to open the furniture first."
        )

    # ---- construction
    def room(self, name):
        self.nodes[name] = Node(name, "room")
        return self

    def furniture(self, name, room, shut=False, components=()):
        self.nodes[name] = Node(name, "furniture", room=room, components=components)
        self.shut[name] = shut
        return self

    def floor(self, name, room):
        self.nodes[name] = Node(name, "floor", room=room)
        self.shut[name] = False
        return self

    def obj(self, name, container, states=None):
        self.nodes[name] = Node(name, "object", states=states)
        self.container[name] = container
        self.held[name] = None
        return self

    # ---- the WorldGraph surface the planner uses
    @property
    def graph(self):
        table = {}
        for name, node in self.nodes.items():
            neighbours = {}
            if node.kind == "object" and self.beside.get(name):
                neighbours[self.nodes[self.beside[name]]] = "next to"
            table[node] = neighbours
        return table

    def get_node_from_name(self, name):
        if name not in self.nodes:
            raise ValueError(name)
        return self.nodes[name]

    def find_furniture_for_object(self, node):
        if self.held.get(node.name) is not None:
            return None
        holder = self.container.get(node.name)
        return self.nodes[holder] if holder else None

    def get_room_for_entity(self, node):
        if node.kind == "object":
            holder = self.container.get(node.name)
            node = self.nodes[holder] if holder else node
        if node.kind == "room":
            return node
        if node.room is None:
            raise ValueError(node.name)
        return self.nodes[node.room]

    def get_furniture_in_room(self, node):
        return [item for item in self.nodes.values()
                if item.kind in ("furniture", "floor") and item.room == node.name]

    def get_all_furnitures(self):
        return [item for item in self.nodes.values() if item.kind in ("furniture", "floor")]

    def get_all_rooms(self):
        return [item for item in self.nodes.values() if item.kind == "room"]

    def get_all_objects(self):
        return [item for item in self.nodes.values() if item.kind == "object"]

    def is_object_with_agent(self, node, agent_type="any"):
        return self.held.get(node.name) is not None

    # ---- execution: what Habitat's skills would have decided
    def execute(self, uid: int, verb: str, argument: str) -> str:
        pieces = [piece.strip() for piece in str(argument or "").split(",")]
        target = pieces[0] if pieces else ""
        if verb in ("Wait", "Explore"):
            return SUCCESS
        if verb == "Navigate":
            self.at[uid] = target
            return SUCCESS
        if verb == "Open":
            self.shut[target] = False
            return SUCCESS
        if verb == "Pick":
            holder = self.container.get(target)
            if holder and self.shut.get(holder):
                return self.shut_pick_message
            if any(held == uid for held in self.held.values()):
                return "Failed to pick! Your hands are full."
            self.held[target] = uid
            self.container.pop(target, None)
            return SUCCESS
        if verb == "Place":
            subject = pieces[0]
            destination = pieces[2] if len(pieces) > 2 else None
            if self.held.get(subject) != uid:
                return "Failed to place! The agent is not holding any object."
            if self.shut.get(destination):
                return "Failed to place! Furniture is closed, you need to open it first."
            self.held[subject] = None
            self.container[subject] = destination
            if len(pieces) > 4 and pieces[3] == "next_to" and pieces[4] not in ("none", ""):
                self.beside[subject] = pieces[4]
                self.beside[pieces[4]] = subject
            return SUCCESS
        if verb in ("Clean", "Fill", "PowerOn", "PowerOff"):
            key = {"Clean": "is_clean", "Fill": "is_filled",
                   "PowerOn": "is_powered_on", "PowerOff": "is_powered_on"}[verb]
            self.nodes[target].properties["states"][key] = verb != "PowerOff"
            return SUCCESS
        return f"Failed! Unknown skill {verb}."


# --------------------------------------------------------------------- the stubs


# The stub base class is installed into `sys.modules` once and the planner module is
# imported against it once, so the world it acts on has to be looked up rather than
# closed over -- otherwise every scenario after the first would drive the first one's house.
CURRENT: Dict[str, FakeWorld] = {}


def install_stubs(world: FakeWorld) -> None:
    """Put just enough of `habitat_llm` on the import path to load the planner."""
    CURRENT["world"] = world

    kinds = {"room": "Room", "furniture": "Furniture", "floor": "Floor", "object": "Object"}

    class Meta(type):
        def __instancecheck__(cls, instance):
            wanted = cls._kind
            got = getattr(instance, "kind", None)
            if wanted == "furniture":  # a floor is a furniture in the real model too
                return got in ("furniture", "floor")
            return got == wanted

    world_model = types.ModuleType("habitat_llm.world_model")
    for kind, name in kinds.items():
        world_model.__dict__[name] = Meta(name, (), {"_kind": kind})

    class Planner:
        def __init__(self, plan_config, env_interface):
            self.planner_config = plan_config
            self.env_interface = env_interface
            self._agents = []
            self.is_done = False
            self.last_high_level_actions = {}

        @property
        def agent_indices(self):
            return [agent.uid for agent in self._agents]

        @property
        def agents(self):
            return self._agents

        @agents.setter
        def agents(self, value):
            self._agents = value

        def process_high_level_actions(self, hl_actions, observations):
            responses = {}
            for uid, (verb, argument, _) in (hl_actions or {}).items():
                responses[uid] = CURRENT["world"].execute(uid, verb, argument)
            return ({uid: 1 for uid in responses}, responses)

        def update_world(self, responses):
            return None

    planner_module = types.ModuleType("habitat_llm.planner.planner")
    planner_module.Planner = Planner

    package = types.ModuleType("habitat_llm")
    package.__path__ = []
    planner_package = types.ModuleType("habitat_llm.planner")
    planner_package.__path__ = []
    sys.modules.setdefault("habitat_llm", package)
    sys.modules.setdefault("habitat_llm.planner", planner_package)
    sys.modules.setdefault("habitat_llm.planner.planner", planner_module)
    sys.modules.setdefault("habitat_llm.world_model", world_model)


class FakeAgent:
    def __init__(self, uid):
        self.uid = uid

    def reset(self):
        return None

    def get_last_state_description(self):
        return "idle"


class FakeAgentsManager(list):
    pass


class FakeEnvInterface:
    def __init__(self, uids, episode_id="selftest"):
        self.partial_obs = False
        self.sim = types.SimpleNamespace(agents_mgr=FakeAgentsManager(uids))
        self.perception = types.SimpleNamespace(sim_handle_to_name={}, region_id_to_name={})
        # The planner detects episode boundaries by id, so the stub has to have one.
        episode = types.SimpleNamespace(episode_id=episode_id, evaluation_propositions=[])
        inner = types.SimpleNamespace(_env=types.SimpleNamespace(current_episode=episode))
        self.env = types.SimpleNamespace(env=types.SimpleNamespace(env=inner))


# --------------------------------------------------------------------- the scenarios


def scenario_shut_cupboard(world: FakeWorld):
    """Two deliveries, one of them out of a cupboard nobody said was shut."""
    (world.room("kitchen_1").room("living_room_1")
        .furniture("counter_1", "kitchen_1").furniture("cabinet_1", "kitchen_1", shut=True)
        .furniture("table_1", "living_room_1").floor("floor_kitchen_1", "kitchen_1")
        .obj("cup_1", "cabinet_1").obj("book_1", "counter_1"))
    return [
        {"key": "is_on_top", "subject": "cup_1", "target": "table_1",
         "alternatives": ["table_1"], "next_to": None, "proposition": 0},
        {"key": "is_on_top", "subject": "book_1", "target": "table_1",
         "alternatives": ["table_1"], "next_to": None, "proposition": 1},
    ]


def scenario_spatial(world: FakeWorld):
    """A rearrangement plus a `next to`, which an R-only memory must fold in."""
    (world.room("kitchen_1").room("living_room_1")
        .furniture("counter_1", "kitchen_1").furniture("table_1", "living_room_1")
        .floor("floor_kitchen_1", "kitchen_1")
        .obj("cup_1", "counter_1").obj("plate_1", "counter_1"))
    return [
        {"key": "is_on_top", "subject": "cup_1", "target": "table_1",
         "alternatives": ["table_1"], "next_to": None, "proposition": 0},
        {"key": "is_on_top", "subject": "plate_1", "target": "table_1",
         "alternatives": ["table_1"], "next_to": None, "proposition": 1},
        {"key": "is_next_to", "subject": "cup_1", "target": "plate_1",
         "alternatives": ["plate_1"], "next_to": None, "proposition": 2},
    ]


def scenario_partner_beat_me(world: FakeWorld):
    """Two errands; something outside the planners finishes one of them mid-walk.

    This is the coordination query on its own. The agent that had claimed the cup is two
    actions into fetching it when the cup appears on the table, and the only thing that
    can tell it to stop is the graph.
    """
    (world.room("kitchen_1").furniture("counter_1", "kitchen_1")
        .furniture("table_1", "kitchen_1").furniture("shelf_1", "kitchen_1")
        .floor("floor_kitchen_1", "kitchen_1")
        .obj("cup_1", "counter_1").obj("book_1", "shelf_1"))
    return [
        {"key": "is_on_top", "subject": "cup_1", "target": "table_1",
         "alternatives": ["table_1"], "next_to": None, "proposition": 0},
        {"key": "is_on_top", "subject": "book_1", "target": "table_1",
         "alternatives": ["table_1"], "next_to": None, "proposition": 1},
    ]


def interrupt_partner_beat_me(step: int, world: FakeWorld) -> None:
    if step == 2:
        world.held["cup_1"] = None
        world.container["cup_1"] = "table_1"


def scenario_capability(world: FakeWorld):
    """A wash and a move. Only the human can wash, and nobody told the planner so.

    The split has to come out of the memory: the cheapest body for `is_clean` uses the
    `Clean` skill, which the benchmark gives to the human alone, so the requirement can
    only go to agent 1 -- and it has to go there without either agent being told which
    requirements the other took. Needs a memory that has seen a state change, so run this
    one against the full library rather than the rearrange-only one.
    """
    (world.room("kitchen_1").furniture("counter_1", "kitchen_1")
        .furniture("sink_1", "kitchen_1", components=["faucet"])
        .furniture("table_1", "kitchen_1").floor("floor_kitchen_1", "kitchen_1")
        .obj("plate_1", "counter_1", states={"is_clean": False})
        .obj("book_1", "counter_1"))
    return [
        {"key": "is_clean", "subject": "plate_1", "target": None,
         "alternatives": [], "next_to": None, "proposition": 0},
        {"key": "is_on_top", "subject": "book_1", "target": "table_1",
         "alternatives": ["table_1"], "next_to": None, "proposition": 1},
    ]


def scenario_unrecognised_refusal(world: FakeWorld):
    """The cupboard case again, but the refusal is phrased in words the ladder cannot read.

    `_repair` is keyed on the phrases the shipped skills use. A refusal outside that
    vocabulary is not repairable, and the open-loop composer answers it by counting the
    requirement down to `retry_limit` and abandoning it -- with the body that would have
    worked sitting unused in the memory the whole time. Closed-loop, the refusal costs the
    *body* rather than the requirement: it is burned, the next one opens the cupboard, and
    the work gets done. Both arms are handed the same memory and the same world, so the
    difference between them is only what a refusal is taken to mean.
    """
    (world.room("kitchen_1").room("living_room_1")
        .furniture("counter_1", "kitchen_1").furniture("cabinet_1", "kitchen_1", shut=True)
        .furniture("table_1", "living_room_1").floor("floor_kitchen_1", "kitchen_1")
        .obj("cup_1", "cabinet_1"))
    world.shut_pick_message = "Failed to pick! Something is in the way."
    return [
        {"key": "is_on_top", "subject": "cup_1", "target": "table_1",
         "alternatives": ["table_1"], "next_to": None, "proposition": 0},
    ]


SCENARIOS = {
    "unrecognised_refusal": scenario_unrecognised_refusal,
    "shut_cupboard": scenario_shut_cupboard,
    "spatial": scenario_spatial,
    "partner_beat_me": scenario_partner_beat_me,
    "capability": scenario_capability,
}

# Scenarios that need an operator library richer than the rearrange-only one.
NEEDS_FULL_MEMORY = {"capability"}

INTERRUPTS = {"partner_beat_me": interrupt_partner_beat_me}


def run(name: str, operators: str, steps: int, verbose: bool,
        closed_loop: bool = False) -> Dict[str, Any]:
    world = FakeWorld()
    requirements = SCENARIOS[name](world)
    install_stubs(world)

    from our_method.skill_memory_v2.partnr_planner import GraphView, SkillMemoryV2Planner

    uids = [0, 1]
    environment = FakeEnvInterface(uids)
    planners = []
    for uid in uids:
        planner = SkillMemoryV2Planner(
            {"operators": operators, "goal_source": "given", "closed_loop": closed_loop},
            environment)
        planner.agents = [FakeAgent(uid)]
        planner.episode_id = "selftest"
        planner._build_from(requirements, GraphView(world, uid))
        planners.append(planner)

    interrupt = INTERRUPTS.get(name)
    log: List[str] = []
    for step in range(steps):
        if interrupt:
            interrupt(step, world)
        finished = True
        for planner in planners:
            _, info, done = planner.get_next_action("selftest", {}, {uid: world for uid in uids})
            finished = finished and done
            if verbose and info.get("print"):
                log.append(info["print"].rstrip())
        if finished:
            break

    view = GraphView(world, 0)
    outcome = {
        "scenario": name,
        "satisfied": sum(1 for item in requirements if view.holds(item)),
        "requirements": len(requirements),
        "split": {planner.uid: sorted(planner.mine) for planner in planners},
        "folded_spatial": planners[0].folded_spatial,
        "abandoned": {planner.uid: sorted(planner.abandoned) for planner in planners},
        "closed_loop": closed_loop,
        "replans": {planner.uid: sum(planner.replans.values()) for planner in planners},
        "notes": {planner.uid: planner.notes for planner in planners},
    }
    if verbose:
        outcome["log"] = log
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", default="results/partnr_operators_train_mini_R.json")
    parser.add_argument("--full-operators", default="results/partnr_operators_train_mini_all.json",
                        help="library used for scenarios that need a state-change operator")
    parser.add_argument("--scenario", default=None, choices=sorted(SCENARIOS))
    parser.add_argument("--closed-loop", action="store_true",
                        help="run with the composer replanning instead of abandoning")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()

    wanted = [arguments.scenario] if arguments.scenario else sorted(SCENARIOS)
    failures = 0
    for name in wanted:
        library = (
            arguments.full_operators if name in NEEDS_FULL_MEMORY else arguments.operators
        )
        outcome = run(name, library, arguments.steps, arguments.verbose,
                      closed_loop=arguments.closed_loop)
        ok = outcome["satisfied"] == outcome["requirements"]
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        print(json.dumps(outcome, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
