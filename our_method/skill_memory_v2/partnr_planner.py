"""A decentralized planner that gets its competence from the operator memory.

PARTNR already ships a symbolic composer: `ScriptedCentralizedPlanner` turns the
episode's evaluation propositions into Navigate/Pick/Navigate/Place and splits the result
across the agents. What makes it the ceiling rather than a method is that it reads the
propositions out of the episode, and it plans for everybody at once. This planner is the
same shape with both privileges removed. One instance drives one agent; it sees only that
agent's world graph; and where the scripted planner consults ground truth for the body of
a rearrangement, this one consults the memory induced from rollouts.

**Coordination is a graph query, not an inference.** Each agent holds the whole
requirement list and, before it starts anything, asks its own world graph whether that
requirement already holds. If the partner has just put the cup on the table, the cup is
on the table in the graph, and the requirement is dropped without a word being exchanged.
The earlier prompt-based memory had to infer the partner's progress from a natural
language description of what changed; here the operators carry typed effects, so the same
question is a lookup. This is what makes the decentralized setting cheaper for v2 than
for v1 rather than more expensive.

**Division of labour is a function, not a negotiation.** Both agents run the same
assignment over the same requirement list, so they reach the same split without
communicating -- the memory supplies each requirement's cost and the benchmark supplies
each agent's capabilities, and greedy balancing on those is deterministic. Where the two
agents' graphs disagree the split can differ, and then the satisfaction check above
absorbs it: the loser of a duplicated errand discovers the work already done.

**Failure is answered from the memory.** A cupboard's openness is not in the world graph,
so the planner does not pretend to know it. It offers the cheap body first, and when the
world refuses -- "Object is in a closed furniture" -- it asks the memory for the variant
that opens something, which is the operator induction learned as `target_starts_shut`.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from habitat_llm.planner.planner import Planner
from habitat_llm.world_model import Floor, Furniture, Object, Room

from .partnr_memory import PartnrSkillMemory, join_argument, split_argument

# Which agent may run which verb. This is not our rule: the benchmark's own action parser
# (`habitat_llm/llm/instruct/utils.py`) refuses state changes for agent 0 and tells it to
# hand them to its partner, and the agent configs give the state-change tools to the human
# only. Encoding it here lets the assignment respect it instead of discovering it as a
# string of failures.
HUMAN_ONLY = {"Clean", "Fill", "Pour", "PowerOn", "PowerOff"}

SUCCESS = "Successful execution!"
# How the world says no, and what each refusal means the plan should do about it. These
# are the benchmark's own termination messages, quoted from the skills that raise them.
SHUT_AT_SOURCE = ("in a closed furniture",)
SHUT_AT_TARGET = ("furniture is closed",)
# The destination is fine but the reference object is not on it yet. Nothing to repair:
# the thing to do is stand still and place again once the partner has landed theirs.
NOT_YET_BESIDE = ("spatial constraint", "has not yet been placed")
# Arrived, but not close enough to act. Walking there again is the whole of the fix.
TOO_FAR = ("not close enough", "occluded")
# The skill does not recognise the name. Under partial observation that usually means the
# thing has not been seen yet rather than that the plan is wrong, so the answer is to go
# and look, not to give up on the requirement.
UNSEEN = ("may not exist", "explore the house", "not present in the graph", "no node with name")

PLACEMENT = {"is_on_top", "is_inside", "is_in_room", "is_on_floor", "is_next_to"}
STATE_OF = {
    "is_clean": ("is_clean", True),
    "is_dirty": ("is_clean", False),
    "is_filled": ("is_filled", True),
    "is_empty": ("is_filled", False),
    "is_powered_on": ("is_powered_on", True),
    "is_powered_off": ("is_powered_on", False),
}


class GraphView:
    """One agent's world graph, answering the questions the memory asks of a scene.

    Every method is a read of this agent's own graph. Under partial observation the
    honest answer is often "I don't know", and it is returned as None rather than
    guessed: an operator that cannot be made concrete is simply not offered yet.
    """

    def __init__(self, graph, uid: int):
        self.graph = graph
        self.uid = uid

    def _node(self, name: Optional[str]):
        if not name:
            return None
        try:
            return self.graph.get_node_from_name(name)
        except Exception:
            return None

    def knows(self, name: str) -> bool:
        return self._node(name) is not None

    def is_furniture(self, name: str) -> bool:
        return isinstance(self._node(name), Furniture)

    def is_room(self, name: str) -> bool:
        return isinstance(self._node(name), Room)

    def container_of(self, name: str) -> Optional[str]:
        node = self._node(name)
        if node is None or not isinstance(node, Object):
            return None
        try:
            furniture = self.graph.find_furniture_for_object(node)
        except Exception:
            return None
        return furniture.name if furniture is not None else None

    def room_of(self, name: str) -> Optional[str]:
        node = self._node(name)
        if node is None:
            return None
        try:
            return self.graph.get_room_for_entity(node).name
        except Exception:
            return None

    def furniture_in_room(self, room: str) -> List[str]:
        node = self._node(room)
        if node is None or not isinstance(node, Room):
            return []
        try:
            return [
                furniture.name
                for furniture in self.graph.get_furniture_in_room(node)
                if not isinstance(furniture, Floor)
            ]
        except Exception:
            return []

    def floor_of(self, room: str) -> Optional[str]:
        node = self._node(room)
        if node is None:
            return None
        try:
            floors = [
                furniture.name
                for furniture in self.graph.get_furniture_in_room(node)
                if isinstance(furniture, Floor)
            ]
        except Exception:
            return None
        return sorted(floors)[0] if floors else None

    def faucet_furniture(self) -> Optional[str]:
        try:
            with_faucet = [
                furniture.name
                for furniture in self.graph.get_all_furnitures()
                if "faucet" in (furniture.properties.get("components") or [])
            ]
        except Exception:
            return None
        return sorted(with_faucet)[0] if with_faucet else None

    def all_names(self) -> List[str]:
        names: List[str] = []
        for getter in ("get_all_objects", "get_all_furnitures", "get_all_rooms"):
            try:
                names.extend(node.name for node in getattr(self.graph, getter)())
            except Exception:
                continue
        return names

    def resolve(self, asked: Optional[str], exclude=()) -> Optional[str]:
        """The entity in this graph that `asked` refers to, if it can be found yet.

        The model is asked what must become true before anything has been seen, so it
        answers in the words of the instruction -- `candle`, `plant` -- while the world
        will come to call them `candle_0` and `plant_container_2`. Refusing those names
        was what made this arm score zero: the requirements were correct and none of them
        survived. Resolution is by instance-of-category, tried in order of how much it
        assumes, and it returns None rather than a guess when nothing matches, which
        leaves the requirement waiting for the room that holds it to be explored.

        `exclude` carries the instances other requirements have already taken, so "the two
        toy animals" becomes two requirements over two different animals rather than one
        animal twice.
        """
        if asked is None:
            return None
        if self.knows(asked) and asked not in exclude:
            return asked
        want = str(asked).strip().lower().replace(" ", "_")
        names = [name for name in self.all_names() if name not in exclude]
        for candidates in (
            [n for n in names if n.lower() == want],
            # `candle_0` is an instance of `candle`; the trailing index is the instance.
            [n for n in names if n.lower().rsplit("_", 1)[0] == want],
            # `plant` for `plant_container_2`: the asked name begins the instance's name.
            [n for n in names if n.lower().startswith(want + "_")],
        ):
            if candidates:
                return sorted(candidates)[0]
        return None

    def rooms(self) -> List[str]:
        try:
            return sorted(room.name for room in self.graph.get_all_rooms())
        except Exception:
            return []

    def held_by_agent(self, name: str) -> bool:
        node = self._node(name)
        if node is None:
            return False
        try:
            return self.graph.is_object_with_agent(node, "any")
        except Exception:
            return False

    # ------------------------------------------------------- the coordination query

    def holds(self, requirement: Dict[str, Any]) -> bool:
        """Whether this requirement is already true as far as this agent can see.

        Answering False when the graph does not know is deliberate: it costs a wasted
        trip at worst, whereas answering True on ignorance silently abandons work.
        """
        effect, subject, target = (
            requirement["key"],
            requirement["subject"],
            requirement.get("target"),
        )
        node = self._node(subject)
        if node is None:
            return False
        if effect in STATE_OF:
            key, wanted = STATE_OF[effect]
            states = node.properties.get("states") or {}
            return key in states and bool(states[key]) is wanted
        # An object in somebody's hand is not resting anywhere, whatever the graph
        # remembers about where it used to be.
        if self.held_by_agent(subject):
            return False
        if effect in ("is_on_top", "is_inside"):
            return self.container_of(subject) == target
        if effect == "is_in_room":
            return self.room_of(subject) == target
        if effect == "is_on_floor":
            container = self.container_of(subject)
            return container is not None and isinstance(self._node(container), Floor)
        if effect == "is_next_to":
            # Only the concept-graph world model records this relation; under the
            # ground-truth graph the answer is always "not that I can see", which costs a
            # missed hand-off at worst and never a false claim of completion.
            neighbour = self._node(target)
            if neighbour is None:
                return False
            try:
                return any(
                    other.name == target and label == "next to"
                    for other, label in self.graph.graph[node].items()
                )
            except Exception:
                return False
        return False


def requirement_key(requirement: Dict[str, Any]) -> str:
    return json.dumps(
        [requirement["key"], requirement["subject"], requirement.get("target")],
        sort_keys=True,
    )


def temporal_order(constraints: List[Any]) -> Dict[int, List[int]]:
    """Which propositions have to be true before which, as the episode states it.

    A `TemporalConstraint` holds a DAG whose edge (i, j) means j must become true after i,
    and it is not decoration: an episode that says "move them to the bedroom table, then
    to the closet" scores both stages zero if the second is reached without the first ever
    having held. Nothing in the world state at the end distinguishes the two, so an agent
    that ignores this cannot recover by being careful -- it has to be told, or it has to
    have learned the pattern, which is what Layer 2 is for in the arm that has no episode
    to read.
    """
    order: Dict[int, List[int]] = {}
    for constraint in constraints or []:
        edges = getattr(constraint, "args", {}) or {}
        edges = edges.get("dag_edges") if isinstance(edges, dict) else None
        dependencies = getattr(constraint, "dependencies", None)
        if isinstance(dependencies, dict) and dependencies:
            # `dependencies` is already the transitive closure the judge applies.
            for node, ancestors in dependencies.items():
                order.setdefault(int(node), []).extend(int(a) for a in ancestors)
        elif edges:
            for earlier, later in edges:
                order.setdefault(int(later), []).append(int(earlier))
    return {node: sorted(set(ancestors)) for node, ancestors in order.items() if ancestors}


def requirements_from_propositions(
    propositions: List[Any],
    handle_to_name: Dict[str, str],
    room_id_to_name: Optional[Dict[str, str]] = None,
    order: Optional[Dict[int, List[int]]] = None,
) -> List[Dict[str, Any]]:
    """The episode's own propositions as requirements. Privileged: an upper bound only.

    This is the arm that answers "how much of the gap is the memory and how much is
    reading the instruction", by handing the planner exactly what the scripted ceiling
    reads. The `number` field is respected -- a proposition over three candidate objects
    asking for one is one requirement, not three -- and the candidates are ordered by
    name so both agents choose the same one.
    """

    def resolve(handles) -> List[str]:
        if isinstance(handles, str):
            handles = [handles]
        return sorted(
            name for name in (handle_to_name.get(str(h)) for h in handles or []) if name
        )

    requirements = []
    for index, proposition in enumerate(propositions):
        arguments = getattr(proposition, "args", None) or {}
        name = getattr(proposition, "function_name", None)
        subjects = resolve(arguments.get("object_handles")) or resolve(
            arguments.get("entity_handles_a")
        )
        targets = (
            resolve(arguments.get("receptacle_handles"))
            or resolve(arguments.get("entity_handles_b"))
            or sorted(
                (room_id_to_name or {}).get(str(room), str(room))
                for room in (arguments.get("room_ids") or [])
            )
        )
        if not subjects:
            continue
        number = int(arguments.get("number", 1) or 1)
        for subject in subjects[:number]:
            requirements.append(
                {
                    "key": name,
                    "subject": subject,
                    "target": targets[0] if targets else None,
                    "alternatives": targets,
                    "next_to": None,
                    "bound": True,
                    "proposition": index,
                    # Recorded over proposition indices rather than positions in this
                    # list, because folding will renumber the list and these must survive.
                    "after_propositions": list((order or {}).get(index, [])),
                }
            )
    return requirements


def fold_spatial(requirements: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Attach `is_next_to` to the placement that will carry it, where one exists.

    A spatial requirement over two objects that are both being moved is not separate
    work: it is a constraint on where one of them lands, and the induced `Place` already
    carries the slot for it. Folding it in is what lets a memory built from rearrangement
    alone satisfy a rearrange-and-place-beside episode. Requirements that cannot be
    folded -- neither entity is being placed -- are left standing to be planned on their
    own, and the count of folds is reported so the effect is measurable.
    """
    placements = {
        requirement["subject"]: requirement
        for requirement in requirements
        if requirement["key"] in ("is_on_top", "is_inside")
    }
    kept = [r for r in requirements if r["key"] != "is_next_to"]
    folded = 0
    for requirement in requirements:
        if requirement["key"] != "is_next_to":
            continue
        subject, target = requirement["subject"], requirement.get("target")
        holder = placements.get(subject) or placements.get(target)
        if holder is None:
            kept.append(requirement)
            continue
        beside = target if placements.get(subject) is holder else subject
        holder["next_to"] = beside
        # The folded requirement's own ordering does not vanish with it: whatever had to
        # happen before the `next to` still has to happen before the placement that now
        # carries it.
        for ancestor in requirement.get("after_propositions") or []:
            holder["after_propositions"].append(ancestor)
        # Placing beside something only works once that something has landed -- the place
        # skill refuses a spatial constraint whose reference is not on the receptacle yet.
        # When the reference is itself being placed, that is an ordering the two
        # requirements imply between them, and recording it is what makes one agent wait
        # for the other instead of failing at the table.
        anchor = placements.get(beside)
        if anchor is not None and anchor is not holder:
            holder["after_propositions"].append(anchor["proposition"])
        folded += 1
    return kept, folded


def link_order(requirements: List[Dict[str, Any]]) -> int:
    """Turn proposition-level orderings into indices into this list. Returns the count."""
    where: Dict[int, List[int]] = defaultdict(list)
    for index, requirement in enumerate(requirements):
        where[requirement["proposition"]].append(index)
    links = 0
    for index, requirement in enumerate(requirements):
        guards = []
        for ancestor in requirement.get("after_propositions") or []:
            guards.extend(other for other in where.get(ancestor, []) if other != index)
        requirement["after"] = sorted(set(guards))
        links += len(requirement["after"])
    return links


def assign(
    requirements: List[Dict[str, Any]],
    uids: List[int],
    cost_of,
    capable_of,
) -> Dict[int, List[int]]:
    """Who does what, computed the same way by everybody, without anybody being told.

    Greedy balancing on the memory's own costs: the most expensive requirement goes to
    the least-loaded agent that can run it. Both agents evaluate this function over the
    same list and reach the same answer, so the split needs no message; and because the
    input is the memory's cost and the benchmark's capabilities, changing the memory
    changes the division of labour, which is what "the memory dispatches" means.
    """
    load = {uid: 0 for uid in uids}
    plan: Dict[int, List[int]] = {uid: [] for uid in uids}
    order = sorted(
        range(len(requirements)),
        key=lambda index: (-cost_of(requirements[index]), requirement_key(requirements[index])),
    )
    for index in order:
        allowed = [uid for uid in uids if capable_of(requirements[index], uid)] or uids
        chosen = min(allowed, key=lambda uid: (load[uid], uid))
        plan[chosen].append(index)
        load[chosen] += cost_of(requirements[index])
    for uid in uids:
        plan[uid].sort(key=lambda index: requirement_key(requirements[index]))
    return plan


class SkillMemoryV2Planner(Planner):
    """One agent, one graph, one memory. No central schedule and no ground truth."""

    def __init__(self, plan_config, env_interface):
        super().__init__(plan_config, env_interface)
        self.config = plan_config
        path = self._setting("operators", os.environ.get("PARTNR_OPERATORS", "results/partnr_operators.json"))
        self.memory = PartnrSkillMemory.load(path)
        self.goal_source = self._setting("goal_source", "oracle")
        self.retry_limit = int(self._setting("retry_limit", 2))
        self.patience = int(self._setting("patience", 3))
        self.repair_limit = int(self._setting("repair_limit", 3))
        self.allow_spatial_composition = bool(self._setting("allow_spatial_composition", True))
        # Reading the episode's temporal DAG is privileged, like reading its propositions.
        # Switching it off is the ablation that says what an ordering is worth, and so
        # what the mined ordering rules will have to recover for the arm that has none.
        self.use_episode_order = bool(self._setting("use_episode_order", True))
        self.llm = None
        if self.goal_source == "llm":
            from hydra.utils import instantiate

            llm_conf = self.config.llm
            self.llm = instantiate(llm_conf.llm)(llm_conf)
        self._reset_state()

    def _setting(self, name: str, default: Any) -> Any:
        try:
            value = self.config[name] if name in self.config else default
        except Exception:
            value = default
        return default if value is None else value

    # ------------------------------------------------------------------ lifecycle

    def _reset_state(self) -> None:
        self.requirements: List[Dict[str, Any]] = []
        self.mine: List[int] = []
        self.chain: List[List[str]] = []
        self.current: Optional[int] = None
        self.attempts: Dict[int, int] = defaultdict(int)
        self.repairs: Dict[int, int] = defaultdict(int)
        self.abandoned: set = set()
        self.done_indices: set = set()
        self.unexplored: List[str] = []
        self.episode_id: Optional[str] = None
        self.replan_required = True
        self.is_done = False
        self.last_high_level_actions = {}
        self.latest_responses: Dict[int, str] = {}
        self.folded_spatial = 0
        self.ordered = 0
        self.idle_rounds = 0
        self.built = False
        self.trace: List[str] = []
        self.notes: List[str] = []

    def reset(self) -> None:
        self._reset_state()
        for agent in self._agents:
            agent.reset()

    @property
    def uid(self) -> int:
        return self._agents[0].uid

    def _episode(self):
        try:
            return self.env_interface.env.env.env._env.current_episode
        except Exception:
            return None

    def _all_uids(self) -> List[int]:
        try:
            return sorted(range(len(self.env_interface.sim.agents_mgr)))
        except Exception:
            return [self.uid]

    # ------------------------------------------------------------------ requirements

    def _cost_of(self, requirement: Dict[str, Any]) -> int:
        options = self.memory.operators_for(requirement["key"])
        return options[0]["cost"] if options else 8

    def _capable_of(self, requirement: Dict[str, Any], uid: int) -> bool:
        options = self.memory.operators_for(requirement["key"])
        if not options:
            return True
        verbs = {action[0] for action in options[0]["body"]}
        return uid != 0 if verbs & HUMAN_ONLY else True

    def _build(self, instruction: str, view: GraphView) -> None:
        if self.goal_source == "llm":
            requirements = self._requirements_from_llm(instruction, view)
        else:
            episode = self._episode()
            perception = self.env_interface.perception
            requirements = requirements_from_propositions(
                list(getattr(episode, "evaluation_propositions", None) or []),
                dict(getattr(perception, "sim_handle_to_name", {}) or {}),
                dict(getattr(perception, "region_id_to_name", {}) or {}),
                temporal_order(list(getattr(episode, "evaluation_constraints", None) or []))
                if self.use_episode_order
                else {},
            )
        self._build_from(requirements, view)

    def _build_from(self, requirements: List[Dict[str, Any]], view: GraphView) -> None:
        """Fold, split, and note the result. Separated so it can be driven a requirement
        list at a time, which is what the offline self-test does."""
        for requirement in requirements:
            requirement.setdefault("after_propositions", [])
        if self.allow_spatial_composition:
            requirements, self.folded_spatial = fold_spatial(requirements)
        self.ordered = link_order(requirements)
        self.requirements = requirements
        self.built = True
        uids = self._all_uids()
        plan = assign(requirements, uids, self._cost_of, self._capable_of)
        self.mine = list(plan.get(self.uid, []))
        self.unexplored = view.rooms() if getattr(self.env_interface, "partial_obs", False) else []
        self.notes.append(
            f"{len(requirements)} requirements, {len(self.mine)} mine, "
            f"{self.folded_spatial} spatial folded, {self.ordered} ordering links"
        )

    def _requirements_from_llm(self, instruction: str, view: GraphView) -> List[Dict[str, Any]]:
        """Ask the model what must become true, and nothing else.

        The model is given the world and the predicates the memory can bring about, and
        answers in those predicates. It never writes an action, never chooses a variant
        and never names an agent: which body, which cupboard and who goes are the
        memory's to decide. That division is the method -- see the delegation curve --
        and it is why the prompt is this short.
        """
        from habitat_llm.llm.instruct.utils import get_world_descr

        menu = "\n".join(
            f"  {item['effect']}({'object, target' if item['arity'] == 2 else 'object'})"
            for item in self.memory.intents()
        )
        world = get_world_descr(view.graph, agent_uid=self.uid, include_room_name=True,
                                add_state_info=True)
        prompt = (
            f"{world}\n\n"
            f"Task: {instruction}\n\n"
            "State what must be true when the task is done, one predicate per line, using "
            "only these predicates and only names from the description above:\n"
            f"{menu}\n\n"
            "Write nothing else: no actions, no order, no agent assignment.\n"
            "Requirements:\n"
        )
        text = ""
        try:
            text = self.llm.generate(prompt, stop="\n\n", max_length=256) or ""
        except Exception as error:
            self.notes.append(f"llm failed: {type(error).__name__}")
        self.trace.append(prompt + text)
        return self._parse_requirements(text, view)

    def _parse_requirements(self, text: str, view: GraphView) -> List[Dict[str, Any]]:
        known = set(self.memory.effects())
        out: List[Dict[str, Any]] = []
        for line in str(text).splitlines():
            line = line.strip().strip("-*. ")
            if "(" not in line or not line.endswith(")"):
                continue
            head, _, rest = line.partition("(")
            head = head.strip()
            if head not in known:
                continue
            arguments = [piece.strip() for piece in rest[:-1].split(",") if piece.strip()]
            if not arguments:
                continue
            # Names are kept as the model wrote them and resolved later, when the room
            # holding the thing has been explored. Dropping them here is what made this
            # arm answer correctly and score nothing.
            target = arguments[1] if len(arguments) > 1 else None
            out.append(
                {
                    "key": head,
                    "subject": arguments[0],
                    "target": target,
                    "alternatives": [target] if target else [],
                    "next_to": None,
                    "proposition": len(out),
                    # No ordering: the model is asked what must be true, not in what
                    # sequence, and the memory has no mined ordering rules for PARTNR yet.
                    # Temporal episodes are therefore expected to lose their first stage
                    # in this arm, and that gap is what Layer 2 is for.
                    "after_propositions": [],
                }
            )
        return out

    # ------------------------------------------------------------------ execution

    def _bind(self, index: int, view: GraphView) -> bool:
        """Fix this requirement's names to entities in this agent's graph, if it can be.

        Done once and remembered, so that indices, guards and the split stay stable while
        the world reveals itself: only the names inside a requirement change, never the
        list of requirements.
        """
        requirement = self.requirements[index]
        if requirement.get("bound"):
            return True
        taken = {
            other["subject"]
            for position, other in enumerate(self.requirements)
            if position != index and other.get("bound")
        }
        subject = view.resolve(requirement["subject"], taken)
        if subject is None:
            return False
        target = requirement.get("target")
        if target is not None:
            target = view.resolve(target)
            if target is None:
                return False
        beside = requirement.get("next_to")
        if beside is not None:
            beside = view.resolve(beside, {subject})
        requirement["asked"] = requirement.get("asked", requirement["subject"])
        requirement["subject"], requirement["target"] = subject, target
        requirement["next_to"] = beside
        requirement["bound"] = True
        return True

    def _ground(self, index: int, view: GraphView) -> Optional[List[List[str]]]:
        requirement = self.requirements[index]
        for operator in self.memory.operators_for(requirement["key"]):
            actions = self.memory.ground(operator, requirement, view)
            if actions is None:
                continue
            if requirement.get("next_to"):
                actions = self.memory.with_beside(actions, requirement["next_to"])
            return actions
        return None

    def _claim(self, view: GraphView) -> None:
        """Take the next piece of work that is still worth doing.

        A requirement guarded by one the partner holds is skipped, not failed: the agent
        goes and does something else, or goes idle at no cost, and picks this up when the
        guard clears. That is the whole of the temporal handling on this side.
        """
        for index in self._queue(view):
            if index in self.abandoned or index in self.done_indices:
                continue
            requirement = self.requirements[index]
            if requirement.get("bound") and view.holds(requirement):
                self.done_indices.add(index)
                continue
            # An ordering this requirement carries is a reason to leave it alone, not a
            # reason to give up: the agent falls through to other work, or waits. A guard
            # counts as met once it has *ever* held, which is what the judge tests -- the
            # first stage of "move it here, then there" is false again the moment the
            # second stage succeeds, and re-blocking on it would deadlock a retry.
            if any(
                guard not in self.abandoned
                and guard not in self.done_indices
                and not (
                    self.requirements[guard].get("bound")
                    and view.holds(self.requirements[guard])
                )
                for guard in requirement.get("after") or []
            ):
                continue
            if not self._bind(index, view):
                if self.unexplored:
                    self.chain = [["Explore", self.unexplored.pop(0)]]
                    self.current = None
                    self.idle_rounds = 0
                    return
                # Every room has been searched and the name still resolves to nothing.
                # Letting the requirement go is what hands it to the partner, whose graph
                # may well hold it: an agent that keeps a requirement it cannot see keeps
                # it away from the only agent that can act on it.
                self.attempts[index] += 1
                if self.attempts[index] >= self.retry_limit:
                    self.abandoned.add(index)
                    self.notes.append(f"never saw {requirement['subject']}")
                continue
            actions = self._ground(index, view)
            if actions is None:
                self.attempts[index] += 1
                if self.attempts[index] >= self.retry_limit:
                    self.abandoned.add(index)
                    self.notes.append(f"no operator for {requirement_key(requirement)}")
                continue
            self.current, self.chain = index, actions
            self.idle_rounds = 0
            return
        # Nothing claimable from what this agent can currently see. Under partial
        # observation that is ignorance far more often than idleness, and the explore
        # branch above only answers it for requirements this planner had to bind itself:
        # privileged requirements arrive already bound, so `_bind` returns early and that
        # branch never fires for them. An agent whose objects are all still unseen then
        # finds nothing to do on its very first call and reports done -- and when the
        # partner does the same, the runner ends the episode at step 0 with nothing
        # attempted. 70 of 366 `val_mini` episodes died exactly that way, every one of
        # them scoring zero. So while work remains, go and look. `idle_rounds` keeps
        # rising rather than resetting, because exploring is not claiming: the agent
        # should still become willing to take over the partner's share on schedule while
        # it searches.
        if self.unexplored and any(
            index not in self.done_indices and index not in self.abandoned
            for index in range(len(self.requirements))
        ):
            self.current, self.chain = None, [["Explore", self.unexplored.pop(0)]]
            self.idle_rounds += 1
            return
        self.current, self.chain = None, []
        self.idle_rounds += 1

    def _queue(self, view: GraphView) -> List[int]:
        """My own share, and -- once I have been idle a while -- what nobody is on.

        Taking over the partner's work is how the pair recovers when one of them fails,
        and it is also the obvious way to make two agents collide on one object. Two
        decentralized conditions keep it from being the second thing. An agent only looks
        at somebody else's share after it has found nothing of its own for `patience`
        rounds, so a partner mid-errand is normally finished or visibly holding the
        object by then; and a requirement whose subject is in somebody's hands is not
        offered at all, which is the same graph query the coordination rests on. Neither
        condition needs a message, and both are wrong in the safe direction: the cost of
        being too shy is a late start, the cost of being too eager is two robots reaching
        for the same cup.
        """
        if self.idle_rounds < self.patience:
            return list(self.mine)
        leftovers = [
            index
            for index in range(len(self.requirements))
            if index not in self.mine
            and index not in self.done_indices
            and index not in self.abandoned
            and not (
                self.requirements[index].get("bound")
                and view.held_by_agent(self.requirements[index]["subject"])
            )
        ]
        if self.uid != 0:
            leftovers.reverse()
        return list(self.mine) + leftovers

    def _repair(self, response: str, view: GraphView) -> bool:
        """Answer a refusal with whatever the memory or the message says to do instead.

        Every answer is counted, and after `repair_limit` of them the refusal is treated
        as unanswerable. Without that bound the planner can be perfectly reasonable and
        still never stop: a `Pick` that reports "not close enough" is repaired by walking
        there again, the walk succeeds, the pick reports it again, and the pair spends the
        whole episode budget being sensible at each other. Measured on one pilot episode:
        498 rounds, eighteen thousand simulation steps, while the partner stood finished.
        """
        if self.current is None or not self.chain:
            return False
        if self.repairs[self.current] >= self.repair_limit:
            return self._place_anyway(response)
        lowered = response.lower()
        if any(hint in lowered for hint in NOT_YET_BESIDE):
            # The plan is right and the world is early. Keep the action and come back.
            self.notes.append("waiting for the reference object to land")
            self.repairs[self.current] += 1
            return True
        if any(hint in lowered for hint in UNSEEN) and self.unexplored:
            room = self.unexplored.pop(0)
            self.chain.insert(0, ["Explore", room])
            self.notes.append(f"looking in {room} for it")
            self.repairs[self.current] += 1
            return True
        if any(hint in lowered for hint in TOO_FAR):
            verb, argument = self.chain[0]
            pieces = split_argument(argument)
            destination = pieces[2] if verb == "Place" and len(pieces) > 2 else pieces[0]
            if destination:
                self.chain.insert(0, ["Navigate", destination])
                self.notes.append(f"walking to {destination} again")
                self.repairs[self.current] += 1
                return True
            return False
        opened = None
        if any(hint in lowered for hint in SHUT_AT_SOURCE):
            opened = "subject"
        elif any(hint in lowered for hint in SHUT_AT_TARGET):
            opened = "target"
        if opened is None:
            return False
        requirement = self.requirements[self.current]
        for operator in self.memory.shut_variant(requirement["key"], opened):
            actions = self.memory.ground(operator, requirement, view)
            if actions is None:
                continue
            if requirement.get("next_to"):
                actions = self.memory.with_beside(actions, requirement["next_to"])
            self.chain = actions
            self.notes.append(f"repaired {requirement['key']} at {opened}")
            self.repairs[self.current] += 1
            return True
        return False

    def _place_anyway(self, response: str) -> bool:
        """Out of repairs on a spatial placement: land the object without the constraint.

        Half a requirement is worth more than none of it under a continuous measure, and
        it is the honest half -- the object really is on the right receptacle, it is just
        not beside the thing that never arrived. Offered once, and only for the refusal
        that says the reference is missing, so it cannot quietly become the usual path.
        """
        if not any(hint in response.lower() for hint in NOT_YET_BESIDE):
            return False
        verb, argument = self.chain[0]
        pieces = split_argument(argument)
        if verb != "Place" or len(pieces) < 5 or pieces[4] in ("none", ""):
            return False
        self.chain[0] = [verb, join_argument(pieces[:3] + ["none", "none"])]
        self.notes.append("placing without the spatial constraint")
        self.repairs[self.current] += 1
        return True

    def _advance(self, view: GraphView) -> None:
        """Consume the last response, then decide what this agent does next."""
        self.is_done = False
        response = str(self.latest_responses.get(self.uid, "") or "")
        if response and self.chain:
            if response.startswith(SUCCESS):
                self.chain.pop(0)
                if not self.chain and self.current is not None:
                    # A body that ran to the end has done its requirement, and that is
                    # recorded here rather than re-derived from the graph. The graph
                    # check exists to notice the *partner's* work; asking it to confirm
                    # our own would strand any requirement the graph cannot represent --
                    # the ground-truth world model carries no `next to` edges at all, so
                    # a spatial requirement would be replanned forever.
                    self.done_indices.add(self.current)
                    self.current = None
            elif not self._repair(response, view):
                # Only a refusal the memory has no answer for counts against the
                # requirement. A repair is the plan working, not the plan failing.
                self.chain = []
                if self.current is not None:
                    self.attempts[self.current] += 1
                    if self.attempts[self.current] >= self.retry_limit:
                        self.abandoned.add(self.current)
                        self.notes.append(f"gave up on {self.current}: {response[:60]}")
                    self.current = None

        # Whatever else has changed, the partner may have finished this while we walked.
        if (
            self.current is not None
            and self.requirements[self.current].get("bound")
            and view.holds(self.requirements[self.current])
        ):
            self.done_indices.add(self.current)
            self.current, self.chain = None, []
            self.notes.append("partner satisfied it first")

        if not self.chain:
            self._claim(view)

        if self.chain:
            verb, argument = self.chain[0]
            self.last_high_level_actions = {self.uid: (verb, argument, "")}
            return
        # Nothing claimable and nowhere left to look. Report done and issue no action at
        # all, rather than idling in the simulator: `Wait` is a skill that costs 600
        # simulation steps every time it succeeds, and an agent whose partner is doing the
        # remaining work would spend the episode's whole budget standing still.
        #
        # Being done is re-decided from the world on every call -- `_advance` clears it at
        # the top -- so this agent picks work straight back up if the partner fails or an
        # ordering guard clears. That re-decision is *not* a safety net for reporting done
        # too eagerly, though, and reading it as one is what let the step-0 death above go
        # unnoticed: the runner ends the episode the moment every planner says done at
        # once, so a premature done that both agents reach together is never revisited.
        # By the time control arrives here the agent has searched every room it knows of.
        self.is_done = True
        self.last_high_level_actions = {self.uid: ("Done", None, None)}

    # ------------------------------------------------------------------ the interface

    def get_next_action(self, instruction, observations, world_graph, verbose: bool = False):
        graph = world_graph[self.uid] if isinstance(world_graph, dict) else world_graph
        view = GraphView(graph, self.uid)

        episode = self._episode()
        identifier = str(getattr(episode, "episode_id", "?")) if episode is not None else None
        # The runner does reset planners between episodes, but the recorder that produced
        # this memory was bitten by assuming so; the check costs nothing and the failure
        # it prevents is silent.
        if identifier != self.episode_id:
            self._reset_state()
            self.episode_id = identifier
        if not self.built:
            self._build(instruction, view)

        replanned = bool(self.replan_required)
        if self.replan_required:
            self._advance(view)

        if self.is_done:
            # Reported as *not* having replanned even though this call re-decided things.
            # The runner logs an action for every planner that says it replanned and then
            # insists each logged action eventually receives a response; an idle agent
            # issues no action, so logging "Done" for it would leave an entry that can
            # never be answered, and the run raises the moment this agent picks work back
            # up. Nothing was planned, so nothing is claimed to have been.
            return {}, self._info(False, {}, ""), True

        actions = {
            uid: action
            for uid, action in self.last_high_level_actions.items()
            if action[0] != "Done"
        }
        low_level_actions, responses = self.process_high_level_actions(actions, observations)
        self.latest_responses = responses
        self.replan_required = any(responses.values())

        printed = ""
        if replanned:
            verb, argument, _ = self.last_high_level_actions[self.uid]
            printed = f"Agent_{self.uid}_Action: {verb}[{argument or ''}]\n"
        for uid, text in responses.items():
            if text:
                printed += f"Agent_{uid}_Observation: {text}\n"
        return low_level_actions, self._info(replanned, responses, printed), False

    def _info(self, replanned: bool, responses: Dict[int, str], printed: str) -> Dict[str, Any]:
        for uid in self.agent_indices:
            responses.setdefault(uid, "")
        summary = (
            f"requirements={len(self.requirements)} mine={len(self.mine)} "
            f"done={len(self.done_indices)} abandoned={len(self.abandoned)} "
            f"folded_spatial={self.folded_spatial} ordering_links={self.ordered}"
        )
        info: Dict[str, Any] = {
            "replanned": {self.uid: replanned},
            "replan_required": {self.uid: self.replan_required},
            "responses": responses,
            "is_done": {self.uid: self.is_done},
            "high_level_actions": dict(self.last_high_level_actions),
            "prompts": {self.uid: "\n".join(self.trace)},
            "traces": {self.uid: summary + "\n" + "\n".join(self.notes)},
            "agent_states": {
                agent.uid: agent.get_last_state_description() for agent in self._agents
            },
        }
        if printed:
            info["print"] = printed
        return info
