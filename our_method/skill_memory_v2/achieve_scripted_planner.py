"""The privileged planner with every body run through the Achieve tool: no model anywhere.

Two 7B smoke runs of ReAct + Achieve (09-13) never called the tool, so its execution path --
resolving names, exploring for what is unseen, grounding an operator, sequencing oracle
sub-skills, retrying a shut container, reporting to the world graph -- had never run. This
drives it without a model. The requirements, the split between agents and the ordering are
`SkillMemoryV2Planner`'s with the episode's propositions (goal_source=oracle); the only change
is that a claimed requirement is issued as one `Achieve[key, subject, target]` instead of the
grounded chain, and the planner's own repairs are off because the tool does its own shut retry.

Read against `outputs/report/priv_iir1/accepted` (same requirements, same library, chain
executed by the planner): an episode that scores lower here is a defect of the tool, not of
the memory. Known differences, by construction, that can move a score without being a defect:
the tool carries no `next_to` (folded spatial constraints are dropped), only the first of
several alternative targets, and none of the planner's too-far / not-yet-beside repairs.
"""

from __future__ import annotations

from typing import List, Optional

from .partnr_planner import GraphView, SkillMemoryV2Planner


class AchieveScriptedPlanner(SkillMemoryV2Planner):
    def _ground(self, index: int, view: GraphView) -> Optional[List[List[str]]]:
        requirement = self.requirements[index]
        pieces = [requirement["key"], requirement["subject"]]
        if requirement.get("target"):
            pieces.append(requirement["target"])
        if requirement.get("next_to"):
            self.notes.append(f"next_to {requirement['next_to']} dropped for {requirement['subject']}")
        return [["Achieve", ", ".join(pieces)]]

    def _repair(self, response: str, view: GraphView) -> bool:
        if response:
            self.notes.append(f"achieve refused: {response[:160]}")
        return False
