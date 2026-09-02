"""What the induced PARTNR operators can be asked, and how one is made concrete.

`partnr_induction.py` writes down what the privileged planner did; this reads that back
as something a planner can query. Three things happen here that the induction file
deliberately does not do, because they are about use rather than about evidence.

**Waiting is dropped.** A recorded body sometimes opens with `Wait`, which is the
centralized planner holding one agent still until the other catches up. That is a fact
about the solver that produced the trace, not about the skill, and a decentralized agent
that inherited it would stand around for no reason.

**Bodies that handle something other than their own subject are refused.** The privileged
planner interleaves two errands, so a segment attributed to one proposition sometimes
carries a `Pick` or a `Place` belonging to the other. Such a body is two jobs written as
one and does not transfer. `Navigate` and `Open` on a spare entity are kept: those are
work done *for* the subject -- opening the cupboard it sits in, walking to the sink it
must be filled at -- which is exactly the kind of enabling step the memory should own.

**Spare variables are given roles.** A body says `[Navigate ?z1][Open ?z1][Navigate ?x]
[Pick ?x]...`, and `?z1` is not free: the body itself says what it is, because a thing
you open before picking `?x` up is the container `?x` is in. Reading the role off the
body is what lets an operator be instantiated in a scene it was never recorded in, and it
is why the memory has to be consulted with a world graph rather than with a string.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Verbs that change what is true of their subject. A body may only apply these to `?x`.
MANIPULATING = {"Pick", "Place", "Clean", "Fill", "Pour", "PowerOn", "PowerOff"}
# Verbs that prepare the world for somebody else's subject, and so may name a spare.
ENABLING = {"Navigate", "Open", "Close", "Explore"}
# Verbs whose subject must be reached before acting, which is what marks the `station`.
AT_STATION = {"Fill", "Clean", "Pour"}

CONTAINER, BESIDE, RECEPTACLE, STATION = "container", "beside", "receptacle", "station"
# Which role wins when a body gives one variable more than one job. A variable that is
# opened is a container whatever else it does, because nothing else can be opened.
ROLE_ORDER = [CONTAINER, BESIDE, RECEPTACLE, STATION]


class WorldView:
    """What the memory needs to know about a scene to make an operator concrete.

    Everything here is answerable from one agent's own world graph, which is what keeps
    the memory usable in the decentralized, partially observed setting: no query asks
    about the other agent, the simulator, or the episode's ground truth.
    """

    def knows(self, name: str) -> bool:
        raise NotImplementedError

    def is_furniture(self, name: str) -> bool:
        raise NotImplementedError

    def container_of(self, name: str) -> Optional[str]:
        raise NotImplementedError

    def room_of(self, name: str) -> Optional[str]:
        raise NotImplementedError

    def furniture_in_room(self, room: str) -> List[str]:
        raise NotImplementedError

    def floor_of(self, room: str) -> Optional[str]:
        raise NotImplementedError

    def faucet_furniture(self) -> Optional[str]:
        raise NotImplementedError


def split_argument(argument: str) -> List[str]:
    return [piece.strip() for piece in str(argument).split(",")]


def join_argument(pieces: Sequence[str]) -> str:
    return ", ".join(pieces)


def _subject_of(verb: str, pieces: List[str]) -> Optional[str]:
    """The entity a verb acts on, which for every skill here is its first argument."""
    return pieces[0] if pieces and pieces[0] else None


def _roles_in(body: List[List[str]]) -> Optional[Dict[str, str]]:
    """What each spare variable is for, or None if the body does more than one job."""
    proposed: Dict[str, List[str]] = defaultdict(list)
    completing = body[-1][0] if body else ""
    for index, action in enumerate(body):
        verb, pieces = action[0], split_argument(action[1])
        subject = _subject_of(verb, pieces)
        if verb in MANIPULATING and subject not in ("?x", None):
            return None  # this body is carrying somebody else's errand
        if verb in ("Open", "Close") and subject and subject.startswith("?z"):
            proposed[subject].append(CONTAINER)
        if verb == "Place" and len(pieces) >= 5:
            if pieces[2].startswith("?z"):
                proposed[pieces[2]].append(RECEPTACLE)
            if pieces[4].startswith("?z"):
                proposed[pieces[4]].append(BESIDE)
        if (
            verb == "Navigate"
            and subject
            and subject.startswith("?z")
            and completing in AT_STATION
            and index + 1 < len(body)
            and body[index + 1][0] == completing
        ):
            proposed[subject].append(STATION)
    spares = {
        piece
        for action in body
        for piece in split_argument(action[1])
        if piece.startswith("?z")
    }
    roles: Dict[str, str] = {}
    for spare in sorted(spares):
        jobs = set(proposed.get(spare, []))
        # A variable that is both opened and placed into is the destination, which
        # happened to be shut -- not the cupboard the subject started in. Without this
        # the body would be instantiated with the source container as its target.
        if CONTAINER in jobs and RECEPTACLE in jobs:
            jobs.discard(CONTAINER)
        found = [role for role in ROLE_ORDER if role in jobs]
        if not found:
            return None  # a spare the body never explains is a spare we cannot bind
        roles[spare] = found[0]
    return roles


def normalize(operator: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One recorded operator as something instantiable, or None if it is not."""
    body = [
        [action[0], action[1] if len(action) > 1 else ""]
        for action in operator["body"]
        if action[0] != "Wait"
    ]
    if not body:
        return None
    # A body has to end by acting on its own subject. One recorded segment ends at
    # `[Open ?y]` -- the attribution caught the moment the destination was opened and the
    # placement landed outside the window -- and a body that never places cannot deliver
    # a placement, however plausible its first three steps look.
    last_verb, last_pieces = body[-1][0], split_argument(body[-1][1])
    if last_verb not in MANIPULATING or _subject_of(last_verb, last_pieces) != "?x":
        return None
    roles = _roles_in(body)
    if roles is None:
        return None
    return {
        "effect": operator["effect"],
        "body": body,
        "roles": roles,
        "preconditions": dict(operator.get("preconditions") or {}),
        "cost": len(body),
        "support": int(operator.get("support", 0)),
        "task_types": list(operator.get("task_types") or []),
    }


class PartnrSkillMemory:
    """The induced operators, indexed by the effect they bring about.

    Nothing is indexed by instruction text. An operator is found by naming the predicate
    that has to become true, which is why holding out a whole family of tasks costs
    almost nothing: the predicates are shared even when the tasks are not.
    """

    def __init__(self, operators: List[Dict[str, Any]], provenance: Any = None):
        merged: Dict[str, Dict[str, Any]] = {}
        self.refused = 0
        for operator in operators:
            clean = normalize(operator)
            if clean is None:
                self.refused += 1
                continue
            # Dropping `Wait` makes bodies collide that the recorder kept apart; their
            # support belongs together, since they demonstrate the same skill.
            key = json.dumps([clean["effect"], clean["body"]], sort_keys=True)
            entry = merged.setdefault(key, clean)
            if entry is not clean:
                entry["support"] += clean["support"]
                for kind in clean["task_types"]:
                    if kind not in entry["task_types"]:
                        entry["task_types"].append(kind)
        self.operators = sorted(
            merged.values(), key=lambda item: (item["cost"], -item["support"])
        )
        self.by_effect: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for operator in self.operators:
            self.by_effect[operator["effect"]["key"]].append(operator)
        self.provenance = provenance

    @classmethod
    def load(cls, path: str) -> "PartnrSkillMemory":
        record = json.loads(Path(path).read_text())
        return cls(record.get("operators") or [], record.get("induced_from"))

    # ------------------------------------------------------------------ queries

    def effects(self) -> List[str]:
        """The predicates this memory knows how to bring about."""
        return sorted(self.by_effect)

    def operators_for(
        self, effect: str, facts: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Every way to make `effect` true, cheapest first, filtered by what is known.

        `facts` carries only what the agent's own graph can answer. When it says nothing
        about a container being shut -- the usual case, since openness is not in the
        graph -- the cheap body is offered first and the shut variant is kept in reserve
        for `shut_variant` to produce if the cheap one is refused by the world.
        """
        facts = facts or {}
        candidates = list(self.by_effect.get(effect, []))
        shut = facts.get("target_starts_shut")
        if shut is not None:
            candidates = [
                operator
                for operator in candidates
                if bool(operator["preconditions"].get("target_starts_shut")) == bool(shut)
            ] or candidates
        if not facts.get("needs_exploration"):
            candidates = [
                operator
                for operator in candidates
                if not operator["preconditions"].get("needs_exploration")
            ] or candidates
        return candidates

    def shut_variant(self, effect: str, opened: str) -> List[Dict[str, Any]]:
        """The bodies that open something before acting, for when the world refused.

        `opened` says which entity the failure was about -- the object's container, or
        the destination itself -- and only the bodies that open that one are offered, so
        a refusal at the destination is not answered by opening the source cupboard.
        """
        wanted = {
            "subject": CONTAINER,  # the body opens a spare: the container `?x` is in
            "target": "?y",  # the body opens the destination by name
        }[opened]
        out = []
        for operator in self.by_effect.get(effect, []):
            if not operator["preconditions"].get("target_starts_shut"):
                continue
            opens = {
                split_argument(action[1])[0]
                for action in operator["body"]
                if action[0] == "Open"
            }
            if wanted == "?y" and "?y" in opens:
                out.append(operator)
            elif wanted == CONTAINER and any(
                operator["roles"].get(name) == CONTAINER for name in opens
            ):
                out.append(operator)
        return out

    def ground(
        self,
        operator: Dict[str, Any],
        requirement: Dict[str, Any],
        view: WorldView,
    ) -> Optional[List[List[str]]]:
        """This operator as actions over real entities, or None if the scene cannot.

        A body is refused rather than guessed at. If the graph cannot say what container
        the object is in, the operator that opens one is simply not available yet, and
        the planner falls through to a cheaper body -- which is the right behaviour under
        partial observation, where the answer often arrives a few steps later.
        """
        subject, target = requirement.get("subject"), requirement.get("target")
        binding: Dict[str, str] = {"?x": subject}
        if target:
            binding["?y"] = target
        for spare, role in operator["roles"].items():
            value = self._resolve(role, requirement, view)
            if not value:
                return None
            binding[spare] = value
        grounded: List[List[str]] = []
        for action in operator["body"]:
            pieces = [binding.get(piece, piece) for piece in split_argument(action[1])]
            if any(piece.startswith("?") for piece in pieces):
                return None
            grounded.append([action[0], join_argument(pieces)])
        return grounded

    def _resolve(
        self, role: str, requirement: Dict[str, Any], view: WorldView
    ) -> Optional[str]:
        subject, target = requirement.get("subject"), requirement.get("target")
        effect = requirement.get("key")
        if role == CONTAINER:
            return view.container_of(subject)
        if role == BESIDE:
            return requirement.get("next_to")
        if role == STATION:
            return view.faucet_furniture()
        if role == RECEPTACLE:
            if effect == "is_in_room" and target:
                furniture = [name for name in view.furniture_in_room(target)]
                return sorted(furniture)[0] if furniture else None
            if effect == "is_on_floor":
                room = target if target and view.furniture_in_room(target) else view.room_of(subject)
                return view.floor_of(room) if room else None
            if effect == "is_next_to" and target:
                # Beside something means on whatever that something is on.
                return view.container_of(target)
            return target if target and view.is_furniture(target) else None
        return None

    def with_beside(self, actions: List[List[str]], reference: str) -> List[List[str]]:
        """The same placement, made to land next to `reference`.

        The induced `Place` already carries the slot -- the recorded bodies write it as
        `next_to, none` when nothing was asked for -- so filling it composes a spatial
        requirement onto a rearrangement the memory already knows, rather than needing a
        spatial operator of its own. This is the one piece of schema knowledge the memory
        asserts rather than induces, and the planner counts every time it is used.
        """
        out = []
        for verb, argument in actions:
            pieces = split_argument(argument)
            if verb == "Place" and len(pieces) >= 5 and pieces[4] in ("none", ""):
                pieces = pieces[:3] + ["next_to", reference]
                argument = join_argument(pieces)
            out.append([verb, argument])
        return out

    # ------------------------------------------------------------------ reporting

    def intents(self) -> List[Dict[str, Any]]:
        """The menu an LLM is shown: what can be made true, and what it takes."""
        menu = []
        for effect in self.effects():
            best = self.by_effect[effect][0]
            menu.append(
                {
                    "effect": effect,
                    "arity": 2 if best["effect"]["value"] != True else 1,  # noqa: E712
                    "cheapest": best["cost"],
                    "support": sum(op["support"] for op in self.by_effect[effect]),
                    "variants": len(self.by_effect[effect]),
                }
            )
        return menu

    def summary(self) -> Dict[str, Any]:
        return {
            "operators": len(self.operators),
            "effects": {
                effect: len(operators) for effect, operators in sorted(self.by_effect.items())
            },
            "support": sum(operator["support"] for operator in self.operators),
            "refused_as_multi_job_or_unbindable": self.refused,
            "induced_from": self.provenance,
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", default="results/partnr_operators.json")
    arguments = parser.parse_args()
    memory = PartnrSkillMemory.load(arguments.operators)
    print(json.dumps(memory.summary(), indent=2))
    for effect in memory.effects():
        print(f"\n{effect}")
        for operator in memory.by_effect[effect][:4]:
            body = "  ".join(f"[{verb} {argument}]" for verb, argument in operator["body"])
            roles = ", ".join(f"{k}={v}" for k, v in sorted(operator["roles"].items())) or "-"
            print(f"  cost {operator['cost']:>2}  support {operator['support']:>4}  {roles}")
            print(f"      {body}")


if __name__ == "__main__":
    main()
