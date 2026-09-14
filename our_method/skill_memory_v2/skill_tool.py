"""The skill memory as a tool a ReAct planner calls, instead of a goal compiler it answers once.

The intent arm asks the model once, at step 0, for every predicate the task needs, and then
executes blind. On PARTNR that loses to closed-loop ReAct by half (DIAG-why-v2-loses-2026-09-13):
the model cannot name what it has not seen, cannot order what it writes as end states, and at 7B
cannot pick the right predicate or target even with the whole world in front of it
(memory `partnr-reask-does-not-help-7b`). The memory itself is not the problem -- fed correct
requirements it beats ReAct on the episodes it covers (0.967 vs 0.884).

So this keeps ReAct's loop and hands the memory only the part it is good at. The model calls

    Achieve[<predicate>, <object>, <target>]

whenever it decides something should become true, and the memory does the rest:

  * names are resolved the planner's own way (`GraphView.resolve`), so `cup` binds to the seen
    `cup_1`;
  * a name nothing answers to yet is searched for: the tool explores the house's own rooms, one
    at a time, re-resolving after each. The first 7B smoke run (09-13) looped on
    `Explore[living_room_0]` -- a room name copied from the few-shot examples that the scene
    does not have -- so exploration is the memory's job here, not the model's;
  * the cheapest operator whose spare roles the agent's graph can fill is grounded
    (`PartnrSkillMemory.ground`) -- for `is_in_room` that includes choosing the furniture;
  * the body runs through this tool's own oracle Navigate / Pick / Place / Open / Close / Explore,
    one sub-skill at a time, exactly the way `OracleRearrangeSkill` sequences its four;
  * if the world refuses because something is shut, the memory's `shut_variant` is tried once.

The tool owns its sub-skills rather than borrowing the agent's, so nothing in `habitat_llm`
changes and the baseline arms run on identical code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hydra.utils import instantiate
from omegaconf import OmegaConf

from habitat_llm.tools import Tool
from our_method.skill_memory_v2.partnr_memory import PartnrSkillMemory

ROOT = Path(__file__).resolve().parents[2]
SUB_SKILLS = {
    "Navigate": "oracle_nav",
    "Pick": "oracle_pick",
    "Place": "oracle_place",
    "Open": "oracle_open",
    "Close": "oracle_close",
    "Explore": "oracle_explore",
}
REFUSED_SHUT = ("closed", "open the", "not open", "articulated")


class _EpisodeState:
    """What `Agent.reset` touches: it calls `tool.skill.reset([0])` on every tool that has one.

    That happens once per episode, so it is also where the record of explored rooms is dropped;
    a single call clearing its queue must not forget which rooms this agent already searched.
    """

    def __init__(self, tool: "MemorySkillTool"):
        self.tool = tool

    def reset(self, batch_idxs) -> None:
        self.tool._explored = set()
        self.tool._clear()


class MemorySkillTool(Tool):
    def __init__(self, skill_config):
        super().__init__(skill_config.name)
        self.skill_config = skill_config
        operators = Path(str(skill_config.operators))
        self.memory = PartnrSkillMemory.load(str(operators if operators.is_absolute() else ROOT / operators))
        self.env = None
        self.subs: Dict[str, Any] = {}
        self.skill = _EpisodeState(self)
        self._explored: Set[str] = set()
        self._clear()

    # ------------------------------------------------------------------ Tool interface

    @property
    def description(self) -> str:
        return f"{self.skill_config.description} Predicates: {', '.join(self.memory.effects())}."

    @property
    def argument_types(self) -> List[str]:
        return []

    def set_environment(self, env) -> None:
        self.env = env
        for verb, key in SUB_SKILLS.items():
            config = OmegaConf.load(ROOT / f"habitat_llm/conf/tools/motor_skills/{key}.yaml")[key]
            tool = instantiate(config)
            tool.agent_uid = self.agent_uid
            tool.set_environment(env)
            tool.to(env.device)
            self.subs[verb] = tool

    def to(self, device) -> None:  # sub-skills are moved when they are built
        return

    def get_state_description(self) -> str:
        if not self._queue:
            return "Idle"
        verb = self._queue[0][0]
        try:
            return self.subs[verb].get_state_description()
        except Exception:
            return f"{self.name}: {verb}"

    def process_high_level_action(self, input_query, observations) -> Tuple[Any, str]:
        if self._input != input_query or not self._queue:
            self._clear()
            error = self._plan(input_query)
            if error:
                self._clear()
                return None, error
        return self._step(observations)

    # ------------------------------------------------------------------ internals

    def _clear(self) -> None:
        self._input: Optional[str] = None
        self._queue: List[List[str]] = []
        self._done: List[str] = []
        self._requirement: Optional[Dict[str, Any]] = None
        self._pending: Optional[Tuple[str, str, Optional[str]]] = None
        self._retried_shut = False
        for tool in getattr(self, "subs", {}).values():
            try:
                tool.skill.reset([0])
            except Exception:
                pass

    def _view(self):
        from our_method.skill_memory_v2.partnr_planner import GraphView

        return GraphView(self.env.world_graph[self.agent_uid], self.agent_uid)

    def _plan(self, input_query: str) -> Optional[str]:
        """Fill the queue for this call, or return the observation that explains why not.

        Called again after every room explored on behalf of a name that did not resolve yet.
        """
        pieces = [piece.strip() for piece in str(input_query).split(",") if piece.strip()]
        if len(pieces) not in (2, 3):
            return (f"Wrong use of {self.name}. Use {self.name}[<predicate>, <object>, <target>], "
                    f"predicate one of: {', '.join(self.memory.effects())}.")
        key, asked_subject = pieces[0], pieces[1]
        asked_target = pieces[2] if len(pieces) == 3 else None
        if key not in self.memory.effects():
            return f"No stored skill achieves {key}. Known predicates: {', '.join(self.memory.effects())}."
        view = self._view()
        subject = view.resolve(asked_subject)
        if subject is None:
            rooms = [room for room in view.rooms() if room not in self._explored]
            if not rooms:
                return (f"{asked_subject} was not found in any room. Check the name, "
                        f"or use FindObjectTool to see what is in the house.")
            self._input, self._pending = input_query, (key, asked_subject, asked_target)
            self._queue = [["Explore", room] for room in rooms]
            return None
        target = view.resolve(asked_target, {subject}) if asked_target else None
        if asked_target and target is None:
            return f"{asked_target} is not a known furniture or room. Use a name from the house description."
        requirement = {"key": key, "subject": subject, "target": target,
                       "alternatives": [target] if target else [], "next_to": None}
        for operator in self.memory.operators_for(key):
            actions = self.memory.ground(operator, requirement, view)
            if actions:
                self._input, self._queue, self._requirement = input_query, actions, requirement
                self._pending = None
                return None
        return f"No stored skill can achieve {key}({subject}, {target}) from what has been seen so far."

    def _replan_shut(self, failed_verb: str) -> bool:
        if self._retried_shut or self._requirement is None:
            return False
        self._retried_shut = True
        which = "target" if failed_verb == "Place" else "subject"
        view = self._view()
        for operator in self.memory.shut_variant(self._requirement["key"], which):
            actions = self.memory.ground(operator, self._requirement, view)
            if actions:
                self._queue = actions
                return True
        return False

    def _step(self, observations) -> Tuple[Any, str]:
        last_action = None
        while self._queue:
            verb, argument = self._queue[0]
            tool = self.subs.get(verb)
            if tool is None:
                message = f"{self.name} cannot run {verb}[{argument}]."
                self._clear()
                return None, message
            action, response = tool.process_high_level_action(argument, observations)
            if response == "":
                if action is None:
                    message = f"{self.name}: {verb}[{argument}] produced no action."
                    self._clear()
                    return None, message
                return action, ""
            if response.startswith("Successful"):
                self._done.append(f"{verb}[{argument}]")
                self._queue.pop(0)
                last_action = action
                if verb in ("Pick", "Place"):
                    # The planner updates both agents' graphs from the action name it issued,
                    # and "Achieve" matches none of the names it knows (place / rearrange /
                    # ...), so without this a placed object stays in the agent's hand in its
                    # own graph and never lands in the partner's. OracleRearrangeSkill reports
                    # its sub-skills the same way; the planner reads and clears this per step.
                    self.env._composite_action_response = {
                        self.agent_uid: (verb, argument, response),
                    }
                if verb == "Explore":
                    self._explored.add(argument)
                    if self._pending is not None:
                        # Try the name again now that one more room has been seen; `_plan`
                        # either grounds the body or queues the rooms still unexplored.
                        error = self._plan(self._input)
                        if error:
                            self._clear()
                            return action, error
                continue
            if any(word in response.lower() for word in REFUSED_SHUT) and self._replan_shut(verb):
                self._done.append(f"{verb}[{argument}] refused, opening first")
                continue
            message = f"{self.name} stopped at {verb}[{argument}]: {response}"
            self._clear()
            return action, message
        requirement = self._requirement or {}
        message = (f"Successful execution! {requirement.get('key')}({requirement.get('subject')}, "
                   f"{requirement.get('target')}) via {' -> '.join(self._done)}")
        self._clear()
        return last_action, message
