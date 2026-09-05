"""Mechanical oracles over PARTNR's recorded rollouts, for an agent to propose against.

VIKI's workbench could afford counterfactual replay: the whole induction half re-executes in
3.5 seconds. PARTNR cannot -- its simulator is habitat, and one episode is a minute or two,
so an oracle that replays is not an oracle, it is an experiment.

The recordings make replay unnecessary for the question that matters. Each trace carries
`proposition_satisfied_at`: the step at which every judged proposition actually became true.
So an operator can be tested against recorded ground truth instead of against a simulator --
find, in a trace it was NOT derived from, an actor whose consecutive actions match the
operator's body under some binding, and ask whether the matching proposition was in fact
recorded as satisfied at the end of that window. That is the same question `run_operator`
asks on VIKI ("does it still work on another episode"), answered from the recording, in
microseconds.

What this is NOT: a causal test. It cannot say the actor's actions *alone* would have
sufficed, because nothing here re-executes anything. It says the operator predicts, on
episodes it has not seen, a satisfaction the world actually produced. Where a claim needs
the causal version, the outer gate is the real one -- rebuild the library and run the
privileged sweep against the post-fix baseline.

Everything reads recorded traces from `results/partnr_rollouts/train_mini`. The evaluation
split is never opened here.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from our_method.skill_memory_v2.partnr_induction import (
    COMPLETING, action_entities, resolve,
)

ROOT = Path("results/partnr_rollouts/train_mini")


class Workbench:
    """Recorded PARTNR rollouts, with the questions an inducer needs to ask of them."""

    def __init__(self, root: Path = ROOT, limit: Optional[int] = None):
        self.paths = sorted(root.glob("*.json"))
        if limit:
            self.paths = self.paths[:limit]
        self._cache: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ reading
    def trace(self, index: int) -> Dict[str, Any]:
        if index not in self._cache:
            self._cache[index] = json.loads(self.paths[index].read_text())
        return self._cache[index]

    def usable(self, index: int) -> bool:
        trace = self.trace(index)
        steps, satisfied = trace.get("steps"), trace.get("proposition_satisfied_at")
        propositions = trace.get("propositions") or []
        return bool(steps and satisfied and len(satisfied) == len(propositions))

    def list_traces(self, effect: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        out = []
        for index in range(len(self.paths)):
            if not self.usable(index):
                continue
            trace = self.trace(index)
            keys = [p.get("function_name") for p in trace["propositions"]]
            if effect and effect not in keys:
                continue
            out.append({"index": index, "episode_id": trace.get("episode_id"),
                        "steps": len(trace["steps"]), "effects": sorted(set(keys)),
                        "instruction": (trace.get("instruction") or "")[:110]})
            if len(out) >= limit:
                break
        return out

    def show_trace(self, index: int, max_steps: int = 20) -> Dict[str, Any]:
        trace = self.trace(index)
        names = trace.get("handle_to_name") or {}
        satisfied = trace.get("proposition_satisfied_at") or []
        return {
            "index": index,
            "instruction": trace.get("instruction"),
            "agents": sorted({uid for s in trace["steps"] for uid in (s.get("actions") or {})}),
            "steps": [{"step": s["step"], "actions": s.get("actions")}
                      for s in trace["steps"][:max_steps]],
            "n_steps": len(trace["steps"]),
            "propositions": [
                {"index": i, "function_name": p.get("function_name"),
                 "entities": resolve(p.get("args", {}).get("object_handles"), names),
                 "targets": resolve(p.get("args", {}).get("receptacle_handles"), names)
                            or (p.get("args", {}).get("room_ids") or []),
                 "satisfied_at_step": int(satisfied[i]) if i < len(satisfied) else -1}
                for i, p in enumerate(trace["propositions"])],
            # The whitelist the shipped inducer attributes with, stated so a proposal can
            # be about *why* it fails rather than a guess at what it is.
            "completing_verbs": sorted(COMPLETING),
        }

    # ------------------------------------------------------------------ oracles
    def held_by(self, index: int, entity: str, step: int) -> Dict[str, Any]:
        """Who was holding `entity` at `step`, from the recorded Pick/Place history.

        `is_in_room` is satisfied by carrying an object while navigating, so the actor is
        never named by the action at the satisfying step. It is named by the `Pick` that
        put the object in that agent's hand, and that is recoverable exactly.
        """
        trace = self.trace(index)
        holder, since = None, None
        for position in range(min(step + 1, len(trace["steps"]))):
            for uid, action in (trace["steps"][position].get("actions") or {}).items():
                if not action:
                    continue
                if action[0] == "Pick" and entity in action_entities(action):
                    holder, since = uid, position
                elif action[0] == "Place" and entity in action_entities(action) and holder == uid:
                    holder, since = None, None
        return {"held_by": holder, "since_step": since, "entity": entity, "at_step": step}

    def actor_window(self, index: int, actor: str, end: int, back: int = 12) -> Dict[str, Any]:
        """That actor's own consecutive actions ending at `end`, skipping its idle steps."""
        trace = self.trace(index)
        window = []
        for position in range(max(0, end - back), min(end + 1, len(trace["steps"]))):
            action = (trace["steps"][position].get("actions") or {}).get(actor)
            if action and action[0] not in ("Wait", "Done"):
                window.append({"step": position, "action": list(action)})
        return {"index": index, "actor": actor, "end": end, "actions": window}

    def step_of_sim(self, index: int, when: int) -> Optional[int]:
        """`proposition_satisfied_at` is in SIM steps, not planner steps.

        A trace with 49 planner steps carries satisfactions at 591, 563, 906 against
        `sim_steps: 1407`. Comparing the two directly is what made the first version of this
        verifier score 0.0 on the shipped inducer's own operators. The conversion is taken
        verbatim from `induce_from_trace` -- the last planner step that had begun by then --
        so this agrees with the attributor it is meant to be measured against.
        """
        steps = self.trace(index)["steps"]
        return max((i for i, step in enumerate(steps) if step.get("sim_step", 0) <= when),
                   default=None)

    def _runs(self, index: int) -> Dict[str, List[Dict[str, Any]]]:
        """Each actor's actions with consecutive repeats collapsed into one run.

        A PARTNR skill occupies many planner steps with the same action while it executes --
        `Navigate chest_of_drawers_44` appears at step 0 and again at step 1 -- so the raw
        per-step list is not a sequence of decisions and matching a body against it is
        meaningless. Collapsing gives the decision sequence, and keeps `last_step` because
        that is when the effect lands.
        """
        trace = self.trace(index)
        runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for step in trace["steps"]:
            for uid, action in (step.get("actions") or {}).items():
                if not action or action[0] in ("Wait", "Done"):
                    continue
                action = list(action)
                bucket = runs[uid]
                if bucket and bucket[-1]["action"] == action:
                    bucket[-1]["last_step"] = step["step"]
                else:
                    bucket.append({"action": action, "first_step": step["step"],
                                   "last_step": step["step"]})
        return runs

    # --------------------------------------------------------- the acceptance oracle
    def predicts(self, operator: Dict[str, Any], index: int,
                 tolerance: int = 2) -> Dict[str, Any]:
        """Does this operator predict, on a trace it was not derived from, a satisfaction
        the recording actually contains?

        Match the body's verb sequence against one actor's recorded actions, bind the
        variables to the entities those actions name, and then ask the recording whether the
        proposition with this operator's effect over those same entities became true at the
        end of that window (within `tolerance` steps, because a proposition is checked after
        the step that causes it).

        Returns `matched` and `predicted` separately on purpose: an operator that never
        matches is untested, which is a different thing from one that matches and is wrong.
        """
        if not self.usable(index):
            return {"index": index, "matched": False, "predicted": False, "why": "unusable trace"}
        trace = self.trace(index)
        names = trace.get("handle_to_name") or {}
        satisfied = trace["proposition_satisfied_at"]
        verbs = [action[0] for action in operator.get("body") or []]
        if not verbs:
            return {"index": index, "matched": False, "predicted": False, "why": "empty body"}

        by_actor = self._runs(index)

        effect = (operator.get("effect") or {}).get("key") or operator.get("key")
        for actor, actions in by_actor.items():
            for start in range(0, max(0, len(actions) - len(verbs) + 1)):
                window = actions[start:start + len(verbs)]
                if [run["action"][0] for run in window] != verbs:
                    continue
                touched = {name for run in window for name in action_entities(run["action"])}
                # The LAST step of the closing run: a skill occupies many planner steps with
                # the same action while it executes, and the proposition is recorded true
                # when it finishes, not when it was first issued. Matching on the first step
                # is what made every operator -- including the shipped ones -- score 0.0.
                end_step = window[-1]["last_step"]
                for position, proposition in enumerate(trace["propositions"]):
                    if proposition.get("function_name") != effect:
                        continue
                    when = int(satisfied[position])
                    arguments = proposition.get("args", {}) or {}
                    entities = set(resolve(arguments.get("object_handles"), names)) or set(
                        resolve(arguments.get("entity_handles_a"), names))
                    if not entities & touched:
                        continue
                    # The target has to be named by the body too. Without this the verifier
                    # confirms any *co-satisfied* proposition: placing `toy_food_0` on
                    # `table_52` does make `is_in_room(toy_food_0, bedroom)` true at that
                    # very step, so an operator that claims `is_in_room` scored 1.0 on a
                    # pick-and-place body. That is a real fact about the world and exactly
                    # the inference the agentic attribution has to make explicit -- but a
                    # body naming `table_52` does not, on its own, explain a proposition
                    # about a room, and an acceptance test that cannot tell those apart
                    # accepts everything.
                    targets = (set(resolve(arguments.get("receptacle_handles"), names))
                               or set(resolve(arguments.get("entity_handles_b"), names))
                               or set(arguments.get("room_ids") or []))
                    if targets and not (targets & touched):
                        continue
                    if when < 0:
                        continue
                    when_step = self.step_of_sim(index, when)
                    if when_step is None:
                        continue
                    if abs(when_step - end_step) <= tolerance:
                        return {"index": index, "matched": True, "predicted": True,
                                "actor": actor, "window": [run["action"] for run in window],
                                "end_step": end_step, "proposition": position,
                                "satisfied_at_sim": when, "satisfied_at_step": when_step,
                                "effect": effect}
                    return {"index": index, "matched": True, "predicted": False,
                            "actor": actor, "window": [run["action"] for run in window],
                            "end_step": end_step, "proposition": position,
                            "satisfied_at_sim": when, "satisfied_at_step": when_step,
                            "effect": effect,
                            "why": "body matched but the recording satisfies it elsewhere"}
        return {"index": index, "matched": False, "predicted": False,
                "why": "no actor's recorded actions match this body"}

    def score_operator(self, operator: Dict[str, Any], exclude: List[int],
                       probe: int = 120) -> Dict[str, Any]:
        """`predicts` over many traces at once, which is what an acceptance test needs."""
        matched, correct, rows = 0, 0, []
        for index in range(min(probe, len(self.paths))):
            if index in exclude:
                continue
            outcome = self.predicts(operator, index)
            if outcome["matched"]:
                matched += 1
                correct += int(outcome["predicted"])
                rows.append(outcome)
        return {"traces_probed": min(probe, len(self.paths)) - len(exclude),
                "matched": matched, "predicted": correct,
                "precision": round(correct / matched, 4) if matched else None,
                "examples": rows[:5]}
