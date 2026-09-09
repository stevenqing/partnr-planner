#!/usr/bin/env python3
"""Step 1: is the `is_in_room` material readable through the workbench's own tools?

The question is not whether the recording contains the satisfaction -- it does, 127 times --
but whether an agent restricted to `show_trace` / `held_by` / `actor_window` can SEE the
mechanism: an actor picked an object and then navigated somewhere in the target room.
09-05's lesson: faithful != readable.
"""
import json, sys
from collections import Counter
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
from partnr_induction_tools import Workbench

bench = Workbench()
out = {"traces": len(bench.paths), "cases": [], "summary": {}}
stat = Counter()

for index in range(len(bench.paths)):
    if not bench.usable(index):
        stat["unusable"] += 1
        continue
    trace = bench.trace(index)
    names = trace.get("handle_to_name") or {}
    satisfied = trace["proposition_satisfied_at"]
    for i, prop in enumerate(trace["propositions"]):
        if prop.get("function_name") != "is_in_room":
            continue
        stat["propositions"] += 1
        when_sim = satisfied[i] if i < len(satisfied) else -1
        if when_sim is None or when_sim < 0:
            stat["never_satisfied"] += 1
            continue
        step = bench.step_of_sim(index, when_sim)
        if step is None:
            stat["no_step_for_sim"] += 1
            continue
        stat["converted"] += 1
        handles = (prop.get("args") or {}).get("object_handles") or []
        entities = [names.get(str(h), str(h)) for h in handles]
        rooms = (prop.get("args") or {}).get("room_ids") or []
        # Who was holding it, and can we see the carry?
        case = {"index": index, "episode_id": trace.get("episode_id"),
                "instruction": (trace.get("instruction") or "")[:100],
                "entity": entities[0] if entities else None, "rooms": rooms,
                "sat_sim_step": when_sim, "sat_step": step, "n_steps": len(trace["steps"])}
        holder = bench.held_by(index, case["entity"] or "", step) if entities else {}
        case["held_by"] = holder.get("held_by")
        case["picked_at"] = holder.get("since_step")
        if holder.get("held_by") is None:
            stat["holder_unknown"] += 1
        else:
            stat["holder_known"] += 1
            window = bench.actor_window(index, holder["held_by"], step, back=12)["actions"]
            case["window"] = window
            verbs = [a["action"][0] for a in window]
            case["verbs"] = verbs
            # readable == the window shows a Pick of the entity, then a Navigate after it
            has_pick = any(a["action"][0] == "Pick" and case["entity"] in " ".join(str(x) for x in a["action"]) for a in window)
            nav_after = False
            seen_pick = False
            for a in window:
                if a["action"][0] == "Pick":
                    seen_pick = True
                elif seen_pick and a["action"][0] == "Navigate":
                    nav_after = True
            case["pick_in_window"] = has_pick
            case["navigate_after_pick"] = nav_after
            stat["pick_in_window"] += int(has_pick)
            stat["navigate_after_pick"] += int(nav_after)
            # is the room even nameable from the window's navigate target?
            navs = [a["action"][1] for a in window if a["action"][0] == "Navigate"]
            case["navigate_targets"] = navs[-3:]
        out["cases"].append(case)

out["summary"] = dict(stat)
json.dump(out, open("outputs/partnr_is_in_room_readability.json", "w"), indent=1, default=str)
print(json.dumps(out["summary"], indent=1))
print("---- 6 sample cases ----")
for c in out["cases"][:6]:
    print(json.dumps(c, indent=1, default=str)[:1200])
