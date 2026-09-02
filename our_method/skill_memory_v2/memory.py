"""Skill memory v2: three layers behind one query interface.

The interface is the redesign. v1 answered "what did we do somewhere that reads like
this?", and everything it returned was text for a prompt; seven rounds of work went into
rendering that text better and none of it helped, because the index was wrong rather than
the rendering. v2 answers three questions instead, and none of them is about similarity:

    operators_for(effect, facts)   what makes this predicate true here, and what it costs
    repair_operator()              what unblocks work that a shut container is blocking
    canonical_place / _asset       what this world calls that
    order_for(requirements)        which of these must happen before which

Only the last of those is retrieval in the old sense, and even it returns learned
patterns rather than a neighbour. Nothing here is indexed by instruction text, which is
why a family the memory has never seen costs it almost nothing: the body that delivers a
pear to a board is the body that delivers a pumpkin to a cupboard, and it was induced
from both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import dependencies, vocabulary
from .simulator import Simulator

FORMAT = "skill-memory-v2"


class SkillMemoryV2:
    def __init__(self, record: Dict[str, Any]):
        self.record = record
        self.operators: List[Dict[str, Any]] = [
            operator for operator in record.get("layer1", {}).get("operators", [])
            if self._usable(operator)
        ]
        self.vocab: Dict[str, Any] = record.get("layer3", {})
        self.rules: List[str] = record.get("layer2", {}).get("kept_patterns", [])

    # ------------------------------------------------------------------ persistence

    @classmethod
    def load(cls, path: Path) -> "SkillMemoryV2":
        record = json.loads(Path(path).read_text())
        if record.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} artefact: {path}")
        return cls(record)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.record, indent=2) + "\n")

    # ------------------------------------------------------------------ layer 1

    @staticmethod
    def _usable(operator: Dict[str, Any]) -> bool:
        """An operator whose body still names a training scene's furniture is refused.

        Those are the segments where the runner walked somewhere named in that episode
        and nowhere else; keeping them would carry one scene into every later plan.
        """
        if operator.get("coordinated"):
            targets = [t for role in operator["roles"] for item in role["actions"] for t in item["action"][1:]]
        else:
            targets = [t for action in operator["body"] for t in action[1:]]
        return bool(targets) and all(
            t.startswith("?") and not t.startswith("?agent") for t in targets
        )

    def operators_for(self, effect_key: str, facts: Dict[str, bool], coordinated: bool = False):
        """Bodies for this effect, those whose recorded situation matches this one first."""
        ranked = []
        for operator in self.operators:
            if operator["effect"]["key"] != effect_key:
                continue
            if bool(operator.get("coordinated")) != coordinated:
                continue
            mismatch = sum(1 for key, was in operator["preconditions"].items() if facts.get(key) != was)
            ranked.append((mismatch, operator["cost"], -operator["support"], operator))
        ranked.sort(key=lambda item: item[:3])
        return [operator for _, _, _, operator in ranked]

    def repair_operator(self) -> Optional[Dict[str, Any]]:
        repairs = sorted(
            (item for item in self.operators if item["effect"]["key"] == "unsealed"),
            key=lambda item: (item["cost"], -item["support"]),
        )
        return repairs[0] if repairs else None

    @staticmethod
    def spare_variables(operator: Dict[str, Any]) -> List[str]:
        if operator.get("coordinated"):
            targets = [t for role in operator["roles"] for item in role["actions"] for t in item["action"][1:]]
        else:
            targets = [t for action in operator["body"] for t in action[1:]]
        return sorted({t for t in targets if t.startswith("?z")})

    @staticmethod
    def suits(asset, wants: Dict[str, Any]) -> bool:
        """A spare variable may only be rebound to something with the same properties."""
        for key, observed in (("is_container", wants.get("is_container")),
                              ("pushable", wants.get("pushable"))):
            if observed is None:
                continue
            actual = (bool(getattr(asset, "is_container", False)) if key == "is_container"
                      else asset.name in {"box", "cardboardbox"})
            if actual != observed:
                return False
        return True

    # ------------------------------------------------------------------ layer 3

    def canonical_place(self, name: str) -> str:
        known = sorted(set(self.vocab.get("places", [])) | set(self.vocab.get("assets", [])))
        return vocabulary.canonical(name, known) or vocabulary.clean(name)

    def canonical_asset(self, name: str, scene_assets: List[str]) -> Optional[str]:
        return vocabulary.canonical(name, scene_assets)

    # ------------------------------------------------------------------ layer 2

    def order_for(self, requirements, visits) -> List[Any]:
        return dependencies.order_for(requirements, visits, self.rules)

    # ------------------------------------------------------------------ rendering

    def render(self, limit: int = 0) -> str:
        """The memory written out for a language model to plan from.

        The same records the planner executes, in prose. This exists because the claim is
        about the representation, not the executor: if effect-keyed typed operators are
        better memory than trajectory text, they should be better when the reader is a
        model too. The whole library fits in a prompt -- nineteen entries against v1's
        eight hundred -- so nothing is retrieved and nothing is truncated, and what the
        model sees cannot be credited to a lucky nearest neighbour.
        """
        def phrase(effect):
            if effect["key"] == "pos.name":
                return "get object X to place Y"
            if effect["key"] == "unsealed":
                return "make what is inside container X reachable"
            return "get object X used, switched on, or operated"

        words = {
            "target_sealed": "Y is a container that starts shut",
            "subject_sealed": "X starts inside a shut container",
            "subject_in_container": "X starts inside a container",
            "subject_on_agent": "X starts beside a robot that cannot walk",
            "target_on_agent": "Y is beside a robot that cannot walk",
        }

        def when(operator):
            facts = [words.get(k, k) for k, v in operator["preconditions"].items() if v]
            return "; ".join(facts) or "nothing special about the scene"

        def spell(action, roles=None):
            out = []
            for item in action[1:]:
                if roles is not None and item.startswith("?r"):
                    out.append(f"robot {chr(65 + int(item[2:]))}")
                else:
                    out.append(item.replace("?", ""))
            return f"{action[0]}({', '.join(out)})"

        lines = [
            "Ways of making something true, distilled from tasks already carried out.",
            "Each was executed successfully; the step count is what it cost, and steps are",
            "what this task is scored on, so prefer the cheapest way that fits the scene.",
            "",
            "ONE ROBOT",
        ]
        singles = [o for o in self.operators if not o.get("coordinated")]
        for operator in sorted(singles, key=lambda o: (o["effect"]["key"], o["cost"])):
            if limit and operator["support"] < limit:
                continue
            body = ", ".join(spell(action) for action in operator["body"])
            lines.append(f"- To {phrase(operator['effect'])}, when {when(operator)}:")
            lines.append(f"    {body}   [{operator['cost']} steps, used {operator['support']} times]")

        pairs, seen = [], set()
        for operator in sorted((o for o in self.operators if o.get("coordinated")),
                               key=lambda o: (o["effect"]["key"], o["cost"], -o["support"])):
            # The same pattern is induced once per ordering of its roles, and which
            # robot happened to be role 0 is not part of what was learned. Blanking the
            # role indices before comparing collapses those back into one entry.
            signature = json.dumps(sorted(
                json.dumps([[item["action"][0]] +
                            ["?r" if t.startswith("?r") else t for t in item["action"][1:]]
                            for item in role["actions"]])
                for role in operator["roles"]
            ))
            if signature in seen:
                continue
            seen.add(signature)
            pairs.append(operator)
        if pairs:
            lines += ["", "SEVERAL ROBOTS TOGETHER"]
        for operator in pairs:
            lines.append(f"- To {phrase(operator['effect'])}, when {when(operator)}:  "
                         f"[{operator['cost']} steps in total]")
            for slot, role in enumerate(operator["roles"]):
                pieces = []
                for item in role["actions"]:
                    text = spell(item["action"], roles=True)
                    waits = [f"after robot {chr(65 + other)}'s action {count}"
                             for other, count in item["after"]]
                    pieces.append(text + (f" ({waits[0]})" if waits else ""))
                lines.append(f"    robot {chr(65 + slot)}: " + ", ".join(pieces))

        if self.rules:
            lines += ["", "ORDER THAT MUST BE RESPECTED"]
            said = []
            for pattern in self.rules:
                rule = json.loads(pattern)
                if rule["b_key"] == "act" and rule.get("a_target_is_b_subject"):
                    text = "put a thing into a device before switching that device on"
                elif rule["b_key"] == "act" and rule.get("b_visits_a_target"):
                    text = ("put a thing where it belongs before using a tool on it, when "
                            "using that tool means going to where it belongs")
                elif rule.get("a_subject_is_b_target"):
                    text = "put a container in its place before putting anything into it"
                else:
                    continue
                if text not in said:
                    said.append(text)
            lines += [f"- {text}" for text in said]
            lines.append("- Anything else may happen at the same time. Two objects going to "
                         "the same place do not wait for each other.")

        targets = list(self.vocab.get("goal_targets", {}))
        if targets:
            lines += ["", "A thing is asked to end up at one of these, spelled exactly so: "
                      + ", ".join(targets) + "."]
        return "\n".join(lines)

    def menu(self) -> tuple:
        """The operators as a numbered vocabulary a model can plan in.

        Rendering the bodies as prose and asking for actions makes the model worse than
        no memory at all -- it reads an abstract schema where the arms it is compared
        against read a concrete trajectory, and it invents object names to fill the
        variables. So the bodies are not shown at all here. What is offered is a list of
        things the memory knows how to do, by name, with what each costs and when it
        applies; the model chooses among them and says what to bind and who should run
        it, and the memory expands the choice into actions it knows are legal. The model
        keeps the decisions and gives up only the transcription.
        """
        words = {
            "target_sealed": "the destination is a container that starts shut",
            "subject_sealed": "the object starts inside a shut container",
            "subject_in_container": "the object starts inside a container",
            "subject_on_agent": "the object starts beside a robot that cannot walk",
            "target_on_agent": "the destination is beside a robot that cannot walk",
        }
        catalogue: Dict[str, Dict[str, Any]] = {}
        lines: List[str] = []
        singles = sorted((o for o in self.operators if not o.get("coordinated")),
                         key=lambda o: (o["effect"]["key"], o["cost"], -o["support"]))
        pairs, seen = [], set()
        for operator in sorted((o for o in self.operators if o.get("coordinated")),
                               key=lambda o: (o["effect"]["key"], o["cost"], -o["support"])):
            signature = json.dumps(sorted(
                json.dumps([[item["action"][0]] +
                            ["?r" if t.startswith("?r") else t for t in item["action"][1:]]
                            for item in role["actions"]])
                for role in operator["roles"]))
            if signature not in seen:
                seen.add(signature)
                pairs.append(operator)

        def condition(operator):
            facts = [words[k] for k, v in operator["preconditions"].items() if v and k in words]
            return "; ".join(facts) or "the ordinary case"

        for number, operator in enumerate(singles, 1):
            name = f"S{number}"
            catalogue[name] = operator
            spares = self.spare_variables(operator)
            arguments = ["X"] + (["Y"] if operator["effect"]["key"] == "pos.name" else []) + \
                        [f"Z{i + 1}" for i in range(len(spares))]
            what = ("put X at Y" if operator["effect"]["key"] == "pos.name"
                    else "open container X, so what is inside can be reached"
                    if operator["effect"]["key"] == "unsealed"
                    else "use, operate or switch on X")
            extra = ("  (Z1 is where X has to be taken to be used)"
                     if spares and operator["effect"]["key"] == "is_activated"
                     else "  (Z1 is the container X starts inside)" if spares else "")
            lines.append(f"  {name}  one robot, {operator['cost']} steps: {what}{extra}")
            lines.append(f"        use when {condition(operator)};  arguments: {', '.join(arguments)}")
        for number, operator in enumerate(pairs, 1):
            name = f"T{number}"
            catalogue[name] = operator
            spares = self.spare_variables(operator)
            arguments = ["X", "Y"] + [f"Z{i + 1}" for i in range(len(spares))]
            lines.append(f"  {name}  {len(operator['roles'])} robots together, "
                         f"{operator['cost']} steps: put X at Y")
            lines.append(f"        use when {condition(operator)};  arguments: "
                         f"{', '.join(arguments)}, and one robot per role")
        return "\n".join(lines), catalogue

    # ------------------------------------------------------------------ reporting

    def summary(self) -> Dict[str, Any]:
        layer1 = self.record.get("layer1", {})
        kinds: Dict[str, int] = {}
        for operator in self.operators:
            kinds[operator.get("kind", "achievement")] = kinds.get(operator.get("kind", "achievement"), 0) + 1
        families = sorted({family for operator in self.operators for family in operator["families"]})
        return {
            "format": FORMAT,
            "excluded_family": self.record.get("excluded_family"),
            "layer1_operators": len(self.operators),
            "layer1_by_kind": kinds,
            "layer1_episodes_replayed": layer1.get("episodes_replayed"),
            "layer1_replay_outcomes": layer1.get("replay_outcomes"),
            "layer1_families": len(families),
            "layer1_max_support": max((o["support"] for o in self.operators), default=0),
            "layer2_kept_patterns": len(self.rules),
            "layer2_recall": self.record.get("layer2", {}).get("recall"),
            "layer2_false_orderings": self.record.get("layer2", {}).get("false_orderings"),
            "layer3_assets": len(self.vocab.get("assets", [])),
            "layer3_places": len(self.vocab.get("places", [])),
        }

    # ------------------------------------------------------------------ self-check

    def validate(self, episodes: List[Dict[str, Any]], sim: Simulator, seed: int) -> Dict[str, Any]:
        """Plan held-out training episodes from their own goals and score them officially.

        An operator library can look plausible and plan nothing, so the artefact is not
        finished until it has been asked to do the job on episodes it was not induced
        from. This is the memory's own regression test, and it uses the judge, not a
        notion of correctness of its own.
        """
        from . import planner

        solved, attempted, reasons = 0, 0, {}
        for truth in episodes:
            if not isinstance(truth, dict) or not truth.get("time_steps"):
                continue
            attempted += 1
            blind = {k: v for k, v in truth.items() if k != "time_steps"}
            steps, reason = planner.plan(blind, self, sim, seed)
            accuracy = sim.score(steps, truth, seed) if steps else 0.0
            if accuracy != 1.0 and steps:
                reason = "OVER_BUDGET" if len(steps) > len(truth["time_steps"]) else "GOAL_UNMET"
            reasons[reason if accuracy != 1.0 else "SOLVED"] = reasons.get(
                reason if accuracy != 1.0 else "SOLVED", 0
            ) + 1
            solved += int(accuracy == 1.0)
        return {"episodes": attempted, "solved": solved,
                "rate": round(solved / attempted, 4) if attempted else 0.0, "outcomes": reasons}
