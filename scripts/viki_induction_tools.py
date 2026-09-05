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

    def list_episodes(self, family: Optional[str] = None, limit: int = 20):
        out = []
        for index, truth in enumerate(self.episodes):
            if not isinstance(truth, dict) or not truth.get("time_steps"):
                continue
            if family and truth.get("task_name") != family:
                continue
            out.append({"index": index, "task_name": truth.get("task_name"),
                        "steps": len(truth["time_steps"])})
            if len(out) >= limit:
                break
        return out

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

    def _memory_of(self, operator: Dict[str, Any]) -> SkillMemoryV2:
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
        actions = chain[0]["actions"] if isinstance(chain, list) else chain["actions"]
        predicate = next(p["predicate"] for p in planner_module.collect_requirements(metadata)
                         if p["predicate"].get("name") == target["predicate_name"])

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
                "predicate_name": target["predicate_name"], "runner": runner,
                "runners_tried": attempts}

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
