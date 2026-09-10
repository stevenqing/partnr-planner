"""A workbench of mechanical oracles over the induction half of VIKI-L2.

The point of these is credit assignment. A six-stage inducer scored only by the memory's
self-check gets one bit of signal for six decisions, which is why a candidate that writes
`?apple_0` for a variable learns nothing from its 0.0: the binding failure never reaches
it. Each oracle here answers one decision, cheaply -- a full replay of the induction half
is 3.5 seconds, so a single episode is milliseconds and these can be called freely.

Everything here reads the *induction* half only (`episodes[::2]`). The half the self-check
plans is never exposed, so no amount of tool use can fit to the gate.

No oracle is a judgement. `check_actor` is a counterfactual replay, `try_bind` is the
planner's own `chains_for`, and `run_operator` executes the bound body in the simulator and
asks whether the effect then holds.
"""
from __future__ import annotations

import copy
from itertools import permutations
from typing import Any, Dict, List, Optional

from our_method.skill_memory_v2 import induction, planner as planner_module
from our_method.skill_memory_v2.build import load_episodes
from our_method.skill_memory_v2.memory import FORMAT, SkillMemoryV2
from our_method.skill_memory_v2.simulator import (
    Simulator, holds, object_properties, predicate_status, state_facts,
)


def metadata_agents(sim, metadata) -> List[str]:
    """The robots this episode has, without building a world just to ask."""
    try:
        return list(sim.world(metadata).agents)
    except Exception:
        return []


def _bound_tokens(operator: Dict[str, Any], option: Dict[str, str]) -> List[str]:
    """Exactly the substitution `chains_for` performs, so leftovers explain its silence."""
    if operator.get("coordinated"):
        actions = [item["action"] for role in operator.get("roles", []) for item in role["actions"]]
    else:
        actions = operator.get("body", [])
    left = []
    for action in actions:
        for token in action[1:]:
            if not isinstance(token, str):
                continue
            value = option.get(token, token)
            if value.startswith("?") and not value.startswith("?r"):
                left.append(value)
    return sorted(set(left))


def _roles_attempted(chains: Any) -> List[List[Any]]:
    """The bound actions of each role, so a relay with a role missing is visible as one."""
    return [list(one.get("actions") or []) for one in
            (chains if isinstance(chains, list) else [chains])]


def _bindings(operator: Dict[str, Any], chain: Any) -> Dict[str, Any]:
    """What each variable was actually bound to, read off the chain the planner built.

    `?x` and `?y` are not the operator's to choose: the planner binds `?x` to the subject
    the effect is about and `?y` to its target, and everything else is a spare it fills by
    what the body does with it. An operator that uses `?x` for the container it opens is
    therefore opening the object it was supposed to fetch -- which is exactly what a run of
    `Open plate` means, and what the refusal used to leave the reader to work out. Saying
    what the variables became is an observation about the binding, not a hint about the
    body.
    """
    chains = chain if isinstance(chain, list) else [chain]
    if operator.get("coordinated"):
        abstract = [item["action"] for role in operator.get("roles", [])
                    for item in role.get("actions", [])]
    else:
        abstract = list(operator.get("body") or [])
    concrete = [action for one in chains for action in (one.get("actions") or [])]
    out: Dict[str, Any] = {}
    for written, done in zip(abstract, concrete):
        for token, value in zip(written[1:], list(done)[1:]):
            if isinstance(token, str) and token.startswith("?") and token not in out:
                out[token] = value
    return out


class Workbench:
    def __init__(self, train, benchmark_root, seed, reference_layers=None):
        self.sim = Simulator(benchmark_root)
        self.seed = seed
        self.episodes = load_episodes(train)[::2]   # the induction half, and only that
        self.reference_layers = reference_layers or {"layer2": {"rules": []}, "layer3": {}}
        self._traces: Dict[int, Any] = {}

    # ------------------------------------------------------------------ reading
    def trace(self, index: int):
        if index not in self._traces:
            truth = self.episodes[index]
            if not isinstance(truth, dict) or not truth.get("time_steps"):
                self._traces[index] = (None, "NO_PLAN")
            else:
                self._traces[index] = induction.replay(truth, self.sim, self.seed)
        return self._traces[index]

    def list_episodes(self, family: Optional[str] = None, limit: int = 20, start: int = 0):
        """The induction half, listed. `start` is an offset into the listing.

        It exists because a run seeded at an episode outside the first page could not
        reach it: models ask for `{"limit": 10, "start": 10}`, the argument was silently
        dropped, and the same first page came back until the move budget ran out. Silently
        discarding part of a well-formed request is the same defect as discarding a
        submission on its key -- the tool has to either honour the argument or say it does
        not exist. `total` is returned so the range is knowable without probing for it.
        """
        matches = []
        for index, truth in enumerate(self.episodes):
            if not isinstance(truth, dict) or not truth.get("time_steps"):
                continue
            if family and truth.get("task_name") != family:
                continue
            matches.append({"index": index, "task_name": truth.get("task_name"),
                            "steps": len(truth["time_steps"])})
        start = max(0, int(start))
        return {"total": len(matches), "start": start,
                "episodes": matches[start:start + max(1, int(limit))]}

    def show_trace(self, index: int, max_steps: int = 12):
        trace, status = self.trace(index)
        if trace is None:
            return {"status": status}
        return {
            "status": status,
            "task_name": self.episodes[index].get("task_name"),
            "history": [{"step": s, "actions": trace["history"][s]["actions"],
                         "carried_before": trace["history"][s]["carried_before"]}
                        for s in range(min(max_steps, len(trace["history"])))],
            "history_len": len(trace["history"]),
            "completions": [
                {"completion": k, "history_index": i, "actor_guess": a,
                 "predicate_name": p.get("name"), "status": predicate_status(p),
                 "facts_at_step_0": state_facts(trace["states"][0], p)}
                for k, (i, a, p) in enumerate(trace["completions"])],
            "assets": sorted(trace["states"][0].assets)[:40],
            "agents": {r: v.get("type") for r, v in trace["metadata"]["agents"].items()},
        }

    # ------------------------------------------------------------------ oracles
    def check_actor(self, index: int, completion: int, actor: str, start: Optional[int] = None):
        """Would `actor`'s own actions over [start..completion step] have made it true?"""
        trace, status = self.trace(index)
        if trace is None:
            return {"error": status}
        step, _, predicate = trace["completions"][completion]
        if start is None:
            start = induction._segment_start(trace["completions"], step, actor)
        verdict = induction._runs_alone(trace["states"][start], trace["history"],
                                        start, step, actor, predicate, self.sim)
        body = [list(trace["history"][s]["actions"][actor])
                for s in range(start, step + 1) if actor in trace["history"][s]["actions"]]
        return {"runs_alone": bool(verdict), "start": start, "step": step,
                "actor_actions": body}

    def contrast_actors(self, index: int) -> Dict[str, Any]:
        """Every actor's own decision sequence, side by side, repeats collapsed.

        `show_trace` returns the episode step by step, which is faithful and, it turns out,
        unreadable: a model given it called `check_actor` on the first robot, submitted that
        robot's body, was refused, then re-read the same trace four times and submitted the
        identical body again -- never once looking at the second robot, whose actions in that
        very episode were the variant the memory was missing. The information was always
        there; it was not legible. This lays the actors next to each other so a difference
        between them is one call rather than a diff the model has to do in its head.
        """
        trace, status = self.trace(index)
        if trace is None:
            return {"status": status}
        actors: Dict[str, List[List[str]]] = {}
        for step in trace["history"]:
            for robot, action in (step["actions"] or {}).items():
                bucket = actors.setdefault(robot, [])
                if not bucket or bucket[-1] != list(action):
                    bucket.append(list(action))
        return {
            "index": index,
            "task_name": self.episodes[index].get("task_name"),
            "actors": {robot: {"verbs": [a[0] for a in actions], "actions": actions}
                       for robot, actions in actors.items()},
            "note": "different actors in one episode often do different things; a body only "
                    "one of them performed is still an operator",
        }

    @staticmethod
    def _normalised(operator: Dict[str, Any]) -> Dict[str, Any]:
        """Fill in the bookkeeping fields a submission may leave out.

        `support` and `cost` only order the candidates a memory offers, and a memory built
        for one operator has nothing to order; `preconditions` absent means the operator
        claims to apply anywhere, which is what an empty mismatch count already says. None
        of these can make a wrong body look right -- but a missing one used to raise
        KeyError inside `operators_for` and kill the whole cell, which loses an induction
        run to a typo. A malformed submission has to be refused, not fatal.
        """
        filled = dict(operator)
        filled.setdefault("support", 1)
        filled.setdefault("preconditions", {})
        filled.setdefault("types", {})
        if "cost" not in filled:
            filled["cost"] = len(filled.get("body") or []) or max(
                (len(role.get("actions") or []) for role in filled.get("roles") or []),
                default=1)
        return filled

    def _memory_of(self, operator: Dict[str, Any]) -> SkillMemoryV2:
        operator = self._normalised(operator)
        record = {"format": FORMAT, "built_from": "workbench", "excluded_family": None,
                  "seed": self.seed, "per_family": 0,
                  "layer1": {"operators": [operator]},
                  "layer2": self.reference_layers.get("layer2", {"rules": []}),
                  "layer3": self.reference_layers.get("layer3", {})}
        return SkillMemoryV2(record)

    def try_bind(self, operator: Dict[str, Any], index: int):
        """Run the planner's own `chains_for` and, when it is silent, say what was left over."""
        truth = self.episodes[index]
        metadata = self.sim.metadata({k: v for k, v in truth.items() if k != "time_steps"}, self.seed)
        env = self.sim.world(metadata)
        memory = self._memory_of(operator)
        key = (operator.get("effect") or {}).get("key")
        report = {"episode": index, "task_name": truth.get("task_name"),
                  "effect_key": key, "requirements": [], "bound": False}
        for requirement in planner_module.collect_requirements(metadata):
            predicate = requirement["predicate"]
            if holds(env, predicate):
                continue
            status = predicate_status(predicate)
            wanted = ("pos.name" if "pos.name" in status
                      else "is_activated" if status.get("is_activated") is True else None)
            if wanted != key:
                continue
            chains = planner_module.chains_for(env, requirement, memory,
                                               bool(operator.get("coordinated")))
            entry = {"predicate_name": predicate.get("name"), "wanted": wanted,
                     "facts": state_facts(env, predicate), "chains": len(chains or [])}
            if not chains:
                binding = {"?x": predicate["name"]}
                if wanted == "pos.name":
                    binding["?y"] = status["pos.name"]
                leftover = _bound_tokens(operator, binding)
                entry["unbound_after_substitution"] = leftover
                entry["why"] = (
                    "tokens left unbound; the planner only binds ?x (the subject), ?y (the "
                    "target for pos.name) and spare variables named ?z1, ?z2, ... -- every "
                    "other ? token makes the operator unusable"
                    if leftover else
                    "operator offered no chain (preconditions or types may not fit)")
            else:
                report["bound"] = True
                entry["first_chain"] = chains[0]
            report["requirements"].append(entry)
            if len(report["requirements"]) >= 4:
                break
        if not report["requirements"]:
            report["why"] = "this episode has no outstanding requirement with that effect key"
        return report

    def run_operator(self, operator: Dict[str, Any], index: int):
        """Bind on episode `index`, execute the bound body, and ask whether the effect holds.

        This is the abstraction test: an operator induced from one episode is only an
        operator if it still works on another.
        """
        binding = self.try_bind(operator, index)
        if not binding.get("bound"):
            return {"bound": False, "binding": binding}
        truth = self.episodes[index]
        metadata = self.sim.metadata({k: v for k, v in truth.items() if k != "time_steps"}, self.seed)
        target = next(r for r in binding["requirements"] if r.get("first_chain"))
        chain = target["first_chain"]
        predicate = next(p["predicate"] for p in planner_module.collect_requirements(metadata)
                         if p["predicate"].get("name") == target["predicate_name"])

        # A coordinated body is several role chains that only mean anything laid out
        # against each other, one robot per role. Running the first role's actions with a
        # single robot -- which is what this did until 2026-09-09 -- asks a question the
        # operator does not answer, and the reference library's own coordination operators
        # fail it 0/6 while achieving their effect perfectly well through the planner. So
        # a coordinated candidate is executed the way the consumer executes it, through
        # `planner.schedule`, and the resulting steps are replayed to observe the effect.
        if operator.get("coordinated"):
            return self._run_coordinated(operator, chain, metadata, predicate)

        actions = chain[0]["actions"] if isinstance(chain, list) else chain["actions"]

        # Try every robot, not just the first. This used to run `sorted(env.agents)[0]` and
        # nothing else, which quietly failed a whole class of correct operators: the
        # reference library's own sealed-target variant -- the one that takes all four
        # `clear_table` holdout episodes from 0.00 to 1.00 under `plan_with` -- was refused
        # here on every one of them with "checker refused ['Open', 'cabinet']", because the
        # first robot is not one of the two types that can open it. A model submitted a
        # byte-identical copy of that operator and was told it did not work. The planner
        # assigns an agent; a verifier that fixes one is asking a different question than
        # the library will.
        attempts, executed, failure, env, runner = [], [], None, None, None
        for candidate in sorted(metadata_agents(self.sim, metadata)):
            env = self.sim.world(metadata)
            checker = self.sim.Checker()
            executed, failure = [], None
            for action in actions:
                resolved = induction._resolve(env, action[0], list(action[1:]), self.sim.entities)
                if resolved is None:
                    failure = "cannot resolve %s" % (action,)
                    break
                agent = env.agents[candidate]
                if not checker.check_operation(action[0].lower(), [agent] + resolved,
                                               env.assets, env.agents):
                    failure = "checker refused %s" % (action,)
                    break
                env.sim_step([[action[0].lower(), agent] + resolved])
                executed.append(action)
            runner = candidate
            attempts.append({"runner": candidate, "failure": failure,
                             "effect_holds": bool(holds(env, predicate))})
            if failure is None and holds(env, predicate):
                break
        # What the effect predicate actually became. This is an observation about the
        # world after the body ran, the same attribute walk the judge's `holds` performs.
        # It is not a hint: it says where the object ended up, never what the body should
        # have been.
        observed = {}
        for attribute, wanted_value in predicate_status(predicate).items():
            current = env.assets.get(predicate["name"])
            try:
                for part in attribute.split("."):
                    current = getattr(current, part)
            except Exception:
                current = None
            observed[attribute] = {"wanted": wanted_value, "observed": current}
        return {"bound": True, "executed": executed, "failure": failure,
                "effect_holds": bool(holds(env, predicate)),
                "effect_state": observed,
                "bindings": _bindings(operator, chain),
                "predicate_name": target["predicate_name"], "runner": runner,
                "runners_tried": attempts}

    # ------------------------------------------------- coordinated bodies
    def _replay(self, metadata, steps):
        """Re-execute a schedule and hand back the world it produced.

        `schedule` decides feasibility on a world of its own and returns only the steps, so
        the effect is observed here on a fresh world -- the same list of steps the judge
        would be handed, run again.
        """
        env = self.sim.world(metadata)
        for step in steps:
            commands = []
            for robot, action in (step.get("actions") or {}).items():
                resolved = induction._resolve(env, action[0], list(action[1:]), self.sim.entities)
                if resolved is None:
                    return env, "cannot resolve %s for %s" % (action, robot)
                commands.append([action[0].lower(), env.agents[robot]] + resolved)
            if not commands:
                continue
            try:
                env.sim_step(commands)
            except Exception as error:                                    # noqa: BLE001
                return env, "%s: %s" % (type(error).__name__, error)
        return env, None

    def _run_coordinated(self, operator, chain, metadata, predicate):
        """Every injective casting of robots to roles, until one makes the effect hold."""
        chains = chain if isinstance(chain, list) else [chain]
        robots = sorted(metadata_agents(self.sim, metadata))
        attempts: List[Dict[str, Any]] = []
        if len(robots) < len(chains):
            return {"bound": True, "effect_holds": False, "executed": [],
                    "failure": "this episode has %d robots and the operator needs %d roles"
                               % (len(robots), len(chains)),
                    "bindings": _bindings(operator, chains),
                    "roles_attempted": _roles_attempted(chains),
                    "predicate_name": predicate.get("name"), "runner": None,
                    "runners_tried": attempts}
        best = None
        for assignment in permutations(robots, len(chains)):
            plans = {robot: [copy.deepcopy(one)] for robot, one in zip(assignment, chains)}
            try:
                steps = planner_module.schedule(metadata, plans, self.sim)
            except Exception as error:                                    # noqa: BLE001
                steps, note = None, "%s: %s" % (type(error).__name__, error)
            else:
                note = None if steps is not None else "no schedule lays these roles out"
            if steps is None:
                attempts.append({"runner": list(assignment), "failure": note,
                                 "effect_holds": False})
                continue
            env, failure = self._replay(metadata, steps)
            achieved = bool(holds(env, predicate))
            attempts.append({"runner": list(assignment), "failure": failure,
                             "effect_holds": achieved, "steps": len(steps)})
            best = {"bound": True,
                    "executed": [a for step in steps for a in step.get("actions", {}).values()],
                    "failure": failure, "effect_holds": achieved,
                    "bindings": _bindings(operator, chains),
                    "roles_attempted": _roles_attempted(chains),
                    "predicate_name": predicate.get("name"),
                    "runner": list(assignment), "runners_tried": attempts}
            if failure is None and achieved:
                return best
        return best or {"bound": True, "effect_holds": False, "executed": [],
                        "failure": "no casting of robots to roles could be scheduled. The "
                                   "roles as they were bound are below: read them against "
                                   "what `contrast_actors` shows each robot doing, since a "
                                   "relay written with a role missing can never be laid out.",
                        "bindings": _bindings(operator, chains),
                        "roles_attempted": _roles_attempted(chains),
                        "predicate_name": predicate.get("name"), "runner": None,
                        "runners_tried": attempts}

    # ------------------------------------------------- the ordering oracle
    def _ordering_probe(self, probe: int):
        """Episodes that are *known* to have an ordering, with their world built once.

        The episode's own `temporal_constraints` are training-set ground truth, so asking
        whether a memory recovers them needs no model and no reference library. Cached
        because the acceptance test calls this on every submission.
        """
        key = ("ordering_probe", probe)
        if key in self._traces:
            return self._traces[key]
        from our_method.skill_memory_v2 import planner as planner_module

        rows = []
        for index in range(min(probe, len(self.episodes))):
            episode = self.episodes[index]
            if not isinstance(episode, dict) or not episode.get("time_steps"):
                continue
            if not (episode.get("temporal_constraints") or []):
                continue
            blind = {k: v for k, v in episode.items() if k != "time_steps"}
            metadata = self.sim.metadata(blind, self.seed)
            env = self.sim.world(metadata)
            requirements = [r["predicate"] for r in planner_module.collect_requirements(metadata)]
            rows.append((index, env, requirements))
        self._traces[key] = rows
        return rows

    def ordering_ok(self, operators: List[Dict[str, Any]], index: int):
        """Does this episode require an ordering, and does this library emit one?

        The per-episode form of `ordering_score`, so the acceptance gate can count an
        episode unsolved when the ordering Layer 2's rules call for is never emitted --
        which is invisible to every other test here. `run_operator` cannot see it (a short
        body and a carrying body both achieve the effect) and the coverage test cannot
        either (on the induction half the episode's own goals carry their constraints, so
        `order_for` is never consulted).

        Reads the episode's own `temporal_constraints`, which are training-set ground
        truth, plus the library. No model, no reference library, no test split.
        """
        from viki_eval_skill_memory_v2 import visits_of
        from our_method.skill_memory_v2 import planner as planner_module

        truth = self.episodes[index]
        if not isinstance(truth, dict) or not (truth.get("temporal_constraints") or []):
            return False, False
        key = ("ordering_row", index)
        if key not in self._traces:
            blind = {k: v for k, v in truth.items() if k != "time_steps"}
            metadata = self.sim.metadata(blind, self.seed)
            env = self.sim.world(metadata)
            requirements = [r["predicate"]
                            for r in planner_module.collect_requirements(metadata)]
            self._traces[key] = (env, requirements)
        env, requirements = self._traces[key]
        memory = SkillMemoryV2({
            "format": FORMAT, "built_from": "ordering_probe", "excluded_family": None,
            "seed": self.seed, "per_family": 0, "layer1": {"operators": operators},
            "layer2": self.reference_layers.get("layer2", {"rules": []}),
            "layer3": self.reference_layers.get("layer3", {})})
        try:
            order = memory.order_for(requirements, visits_of(env, requirements, memory))
        except Exception:                                            # noqa: BLE001
            return True, False
        return True, bool(order and any(len(group) > 1 for group in order))

    def ordering_score(self, operators: List[Dict[str, Any]], probe: int = 200):
        """How often this library emits an ordering on episodes known to have one.

        This is the one property that separates a carrying operator from a short one, and
        no other test in this workbench can see it. `run_operator` cannot: both achieve the
        effect. The marginal-coverage test cannot either -- on the induction half the
        episode's own goals carry their own temporal constraints, so `order_for` is never
        consulted and the deficiency never shows. Measured: the 7-operator agent library
        scores 40/71, the same library plus one carrying `is_activated` operator scores
        71/71, and so does the reference.

        What it does NOT check is whether the emitted ordering is *correct*; it checks that
        one is emitted at all. That is the difference the visit sets create, and sharpening
        it further would need the episode's constraint graph compared edge by edge.
        """
        from viki_eval_skill_memory_v2 import visits_of

        memory = SkillMemoryV2({
            "format": FORMAT, "built_from": "ordering_probe", "excluded_family": None,
            "seed": self.seed, "per_family": 0, "layer1": {"operators": operators},
            "layer2": self.reference_layers.get("layer2", {"rules": []}),
            "layer3": self.reference_layers.get("layer3", {})})
        rows = self._ordering_probe(probe)
        emitted, seen = 0, 0
        for index, env, requirements in rows:
            try:
                order = memory.order_for(requirements, visits_of(env, requirements, memory))
            except Exception:
                continue
            seen += 1
            if order and any(len(group) > 1 for group in order):
                emitted += 1
        return {"episodes_with_a_known_ordering": seen, "memory_emits_one": emitted,
                "rate": round(emitted / seen, 4) if seen else None}

    def plan_with(self, operators: List[Dict[str, Any]], index: int):
        record = {"format": FORMAT, "built_from": "workbench", "excluded_family": None,
                  "seed": self.seed, "per_family": 0, "layer1": {"operators": operators},
                  "layer2": self.reference_layers.get("layer2", {"rules": []}),
                  "layer3": self.reference_layers.get("layer3", {})}
        memory = SkillMemoryV2(record)
        truth = self.episodes[index]
        blind = {k: v for k, v in truth.items() if k != "time_steps"}
        steps, reason = planner_module.plan(blind, memory, self.sim, self.seed)
        accuracy = self.sim.score(steps, truth, self.seed) if steps else 0.0
        return {"episode": index, "reason": reason, "steps": len(steps) if steps else 0,
                "official_score": accuracy}
