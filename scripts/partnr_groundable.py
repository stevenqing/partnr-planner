#!/usr/bin/env python3
"""Can this operator be made concrete at all? The free half of acceptance.

An execution cell costs ten minutes; this costs nothing, and it answers the question that
must be true before execution is worth spending. The memory instantiates an operator by
reading each variable's role off the body -- a thing you `Open` before picking `?x` up is
the container `?x` is in, a thing you `Place` onto is the receptacle -- and then asking the
scene for something to fill it. A body whose variables cannot be filled produces no actions
at all, and running the benchmark to discover that would be running it to watch nothing
happen.

The scene here is a stub, not a simulator: two rooms, furniture in each, one object in a
closed container. That is enough for the only question being asked -- whether grounding
returns actions -- and nothing here says the actions are correct. Correctness is the outer
gate's business.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from our_method.skill_memory_v2.partnr_memory import PartnrSkillMemory, WorldView

ROOMS = {"kitchen_1": ["counter_1", "fridge_2", "sink_3"],
         "living_room_1": ["table_10", "couch_11", "cabinet_12"]}
CONTAINED = {"cup_0": "cabinet_12",
             # `is_next_to` asks for a placement beside another object, so the stub
             # needs one object that sits on furniture rather than inside a container.
             "plate_1": "table_10"}
FLOORS = {"kitchen_1": "floor_kitchen_1", "living_room_1": "floor_living_room_1"}


class StubView(WorldView):
    def knows(self, name: str) -> bool:
        return name in self._all()

    def _all(self) -> List[str]:
        names = list(ROOMS) + list(FLOORS.values()) + list(CONTAINED)
        for furniture in ROOMS.values():
            names += furniture
        return names

    def is_furniture(self, name: str) -> bool:
        return any(name in furniture for furniture in ROOMS.values()) or name in FLOORS.values()

    def container_of(self, name: str) -> Optional[str]:
        return CONTAINED.get(name)

    def room_of(self, name: str) -> Optional[str]:
        for room, furniture in ROOMS.items():
            if name in furniture:
                return room
        return CONTAINED.get(name) and self.room_of(CONTAINED[name])

    def furniture_in_room(self, room: str) -> List[str]:
        return list(ROOMS.get(room, []))

    def floor_of(self, room: str) -> Optional[str]:
        return FLOORS.get(room)

    def faucet_furniture(self) -> Optional[str]:
        return "sink_3"


TARGETS = {"is_in_room": "kitchen_1", "is_on_floor": "kitchen_1", "is_next_to": "plate_1"}


def _variables(operator: Dict[str, Any]) -> List[str]:
    out = []
    for action in operator.get("body") or []:
        for piece in str(action[1]).split(","):
            piece = piece.strip()
            if piece.startswith("?") and piece not in out:
                out.append(piece)
    return out


def check(operator: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Refusals here name the variable, because a refusal that does not is unusable.

    The memory reads a spare variable's job off the body and only recognises spares
    written `?z1`, `?z2`, ...; a variable under any other name is never given a role and
    the body cannot be instantiated. That is a naming convention, so the refusal states
    it -- what the variable should stand for is the proposer's problem, not ours.
    """
    unknown = [v for v in _variables(operator)
               if v not in ("?x", "?y") and not v.startswith("?z")]
    naming = (f"{', '.join(unknown)} is not a variable the memory can bind -- the subject is "
              "`?x`, its target is `?y`, and any other variable must be written `?z1`, `?z2`, "
              "... because that is how the body is read for what the variable is for")
    memory = PartnrSkillMemory([operator])
    loaded = memory.operators_for(key)
    if not loaded:
        why = "the memory refused the operator on load"
        return {"groundable": False, "why": why + ("; " + naming if unknown else ""),
                "unbindable": unknown}
    requirement = {"key": key, "subject": "cup_0", "target": TARGETS.get(key, "table_10"),
                   "alternatives": [TARGETS.get(key, "table_10")], "next_to": None, "bound": True}
    actions = memory.ground(loaded[0], requirement, StubView())
    if not actions:
        why = "the operator loaded but grounding produced no actions"
        if unknown:
            why += "; " + naming
        else:
            why += ": a variable it kept has no role this scene can fill"
        return {"groundable": False, "unbindable": unknown, "why": why}
    return {"groundable": True, "actions": actions}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    payload = json.loads(args.candidates.read_text())
    key = payload.get("target_key", "is_in_room")
    out = []
    for index, entry in enumerate(payload["candidates"]):
        verdict = check(entry["operator"], key)
        out.append({"candidate": index, "move": entry.get("move"),
                    "body": [a[0] for a in entry["operator"].get("body") or []], **verdict})
    report = {"candidates": args.candidates.name, "key": key, "results": out,
              "groundable": sum(1 for o in out if o["groundable"])}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
