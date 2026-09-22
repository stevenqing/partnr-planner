#!/usr/bin/env python3
"""A1 of results/iclr_two_exp_2026-09-22/SPEC.md: the replay-mined operator set. Zero LLM calls.

Why not `our_method/skill_memory_v2/build.py` as is (it is the builder of the 19-operator
reference library): it caps every family at 250 episodes, deduplicates by
(effect, body, preconditions) rather than by (effect, body), leaves non-asset place names
literal (the memory then refuses those operators at load), and mines a `repair` kind that is
not a goal predicate turning true. This script keeps its replay and segmentation primitives
(`induction.replay`, `_segment_start`, `_runs_alone`, `_bind`) untouched and changes only
those four things:

  1. every one of the 3,598 even-indexed training episodes is replayed (no per-family cap);
  2. a segment ends wherever a judged predicate of the episode (goal or temporal stage)
     becomes true; the body is the actor's own actions since its previous completion, or,
     when those actions alone do not reproduce the predicate, the whole window with one role
     per robot (the builder's coordination rule);
  3. every asset AND place name is lifted: the effect's subject is ?x, its target ?y, every
     other asset or place ?z1, ?z2, ... (typed with the object's properties; a place that is
     not an asset gets an empty type); robots in a coordination window become ?r0, ?r1, ...;
     a robot named as an action target in a single-robot body is lifted to ?agent:<name>
     exactly as the builder does (the planner cannot bind it, so certification drops it);
  4. deduplicate by (effect, lifted body); support = number of distinct episodes whose
     replay produced that body. Preconditions / types of the merged operator are the facts
     that held with the same value in every supporting occurrence.

No operator is edited after this step. Output: replay_library.json (+ script sha256).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
TRAIN = Path("/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet")
BENCH = Path("/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R")

_SIM = None
_EPISODES = None


def _init():
    global _SIM, _EPISODES
    from our_method.skill_memory_v2.build import load_episodes
    from our_method.skill_memory_v2.simulator import Simulator
    _SIM = Simulator(BENCH)
    _EPISODES = load_episodes(TRAIN)[::2]


def _lift(actions, mapping, types, state):
    from our_method.skill_memory_v2.simulator import object_properties
    out = []
    for action in actions:
        rewritten = [action[0]]
        for item in action[1:]:
            if item not in mapping:
                if item in state.agents:
                    mapping[item] = "?agent:%s" % item
                else:
                    n = 1 + sum(1 for v in mapping.values() if v.startswith("?z"))
                    mapping[item] = "?z%d" % n
                    types[mapping[item]] = (object_properties(state.assets[item])
                                            if item in state.assets else {})
            rewritten.append(mapping[item])
        out.append(rewritten)
    return out


def mine_episode(position: int):
    from our_method.skill_memory_v2 import induction
    from our_method.skill_memory_v2.simulator import SEED, object_properties, predicate_status, state_facts
    truth = _EPISODES[position]
    if not isinstance(truth, dict) or not truth.get("time_steps"):
        return {"position": position, "status": "NO_PLAN", "candidates": []}
    family = truth.get("task_name", "?")
    try:
        trace, status = induction.replay(truth, _SIM, SEED)
    except Exception as error:                                         # noqa: BLE001
        return {"position": position, "family": family, "status": "EXC_%s" % type(error).__name__,
                "candidates": []}
    if trace is None:
        return {"position": position, "family": family, "status": status, "candidates": []}
    found = []
    for index, actor, predicate in trace["completions"]:
        if actor is None:
            found.append({"skip": "no_actor"})
            continue
        start = induction._segment_start(trace["completions"], index, actor)
        state = trace["states"][start]
        facts = state_facts(state, predicate)
        mapping, effect = induction._bind(predicate)
        if effect is None:
            found.append({"skip": "effect_not_liftable"})
            continue
        types = {}
        if predicate["name"] in state.assets:
            types["?x"] = object_properties(state.assets[predicate["name"]])
        target = predicate_status(predicate).get("pos.name")
        if isinstance(target, str):
            types["?y"] = (object_properties(state.assets[target]) if target in state.assets
                           else {"is_container": False, "isolated": False, "pushable": False})
        if induction._runs_alone(state, trace["history"], start, index, actor, predicate, _SIM):
            body = [list(trace["history"][s]["actions"][actor]) for s in range(start, index + 1)
                    if actor in trace["history"][s]["actions"]]
            if not body:
                found.append({"skip": "empty_body"})
                continue
            lifted = _lift(body, mapping, types, state)
            found.append({"kind": "achievement", "effect": effect, "body": lifted,
                          "preconditions": facts, "types": types,
                          "runner_type": trace["metadata"]["agents"][actor]["type"],
                          "window": [start, index]})
        else:
            window = []
            for robot in sorted({n for s in range(start, index + 1) for n in trace["history"][s]["actions"]}):
                acts = [(s, list(trace["history"][s]["actions"][robot])) for s in range(start, index + 1)
                        if robot in trace["history"][s]["actions"]]
                if acts:
                    window.append({"robot": robot, "acts": acts})
            for slot, role in enumerate(window):
                mapping[role["robot"]] = "?r%d" % slot
            roles = []
            for slot, role in enumerate(window):
                rewritten = _lift([a for _, a in role["acts"]], mapping, types, state)
                roles.append({"variable": "?r%d" % slot,
                              "actions": [{"action": a, "offset": off - start}
                                          for a, (off, _) in zip(rewritten, role["acts"])]})
            for slot, role in enumerate(roles):
                for item in role["actions"]:
                    item["after"] = [[other, sum(1 for peer in roles[other]["actions"]
                                                 if peer["offset"] < item["offset"])]
                                     for other in range(len(roles)) if other != slot]
                    item["after"] = [p for p in item["after"] if p[1]]
            role_types = {r["variable"]: [trace["metadata"]["agents"][w["robot"]]["type"]]
                          for r, w in zip(roles, window)}
            found.append({"kind": "coordination", "coordinated": True, "effect": effect,
                          "roles": roles, "preconditions": facts, "types": types,
                          "role_types": role_types, "window": [start, index]})
    return {"position": position, "family": family, "status": status, "candidates": found}


def signature(op) -> str:
    """(effect, lifted body) -- the dedup key SPEC A1.4 names; same form as viki_union_library."""
    if op.get("coordinated"):
        body = [[item["action"] for item in role["actions"]] for role in op["roles"]]
    else:
        body = op["body"]
    e = op["effect"]
    return json.dumps([e.get("key"), e.get("subject"), e.get("value"), bool(op.get("coordinated")), body],
                      sort_keys=True)


def _intersect(a, b):
    return {k: v for k, v in a.items() if k in b and b[k] == v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    from our_method.skill_memory_v2.build import load_episodes
    n = len(load_episodes(TRAIN)[::2])
    positions = list(range(n if args.limit is None else min(n, args.limit)))
    with Pool(args.workers, initializer=_init) as pool:
        results = pool.map(mine_episode, positions, chunksize=4)
    merged, statuses, skips = {}, Counter(), Counter()
    occurrences = 0
    for res in results:
        statuses[res["status"]] += 1
        seen_here = set()
        for cand in res["candidates"]:
            if "skip" in cand:
                skips[cand["skip"]] += 1
                continue
            occurrences += 1
            sig = signature(cand)
            slot = merged.get(sig)
            if slot is None:
                slot = {k: copy.deepcopy(v) for k, v in cand.items() if k not in ("runner_type", "window")}
                slot.update(support=0, occurrences=0, families=[], runner_types=[], episodes=[])
                if not cand.get("coordinated"):
                    slot["requires"] = sorted({a[0].lower() for a in cand["body"]})
                    slot["carries"] = any(a[0] == "Grasp" for a in cand["body"])
                    slot["cost"] = len(cand["body"])
                else:
                    slot["cost"] = max(i["offset"] for r in cand["roles"] for i in r["actions"]) + 1
                merged[sig] = slot
            else:
                slot["preconditions"] = _intersect(slot["preconditions"], cand["preconditions"])
                slot["types"] = {v: _intersect(t, cand["types"].get(v, {}))
                                 for v, t in slot["types"].items() if v in cand["types"]}
                for var, kinds in (cand.get("role_types") or {}).items():
                    for k in kinds:
                        if k not in slot.setdefault("role_types", {}).setdefault(var, []):
                            slot["role_types"][var].append(k)
            slot["occurrences"] += 1
            if (res["position"], sig) not in seen_here:
                seen_here.add((res["position"], sig))
                slot["support"] += 1
                slot["episodes"].append(res["position"])
            if res["family"] not in slot["families"]:
                slot["families"].append(res["family"])
            rt = cand.get("runner_type")
            if rt and rt not in slot["runner_types"]:
                slot["runner_types"].append(rt)
    operators = sorted(merged.values(), key=lambda o: (-o["support"], o["cost"]))
    for i, op in enumerate(operators):
        op["id"] = "m%04d" % i
        op["provenance"] = {"proposed_by": "replay_mining", "script": "scripts/iclr_two_exp/A/mine_replay.py",
                            "script_sha256": script_sha}
    by_family = defaultdict(int)
    for op in operators:
        for f in op["families"]:
            by_family[f] += 1
    record = {"format": "replay_library/1", "built_from": "VIKI-L2 train.parquet, even-indexed episodes",
              "episodes": len(positions), "script": "scripts/iclr_two_exp/A/mine_replay.py",
              "script_sha256": script_sha, "llm_calls": 0, "manual_edits": 0,
              "replay_outcomes": dict(statuses), "segments": occurrences, "skipped_segments": dict(skips),
              "operators_mined": len(operators),
              "effect_keys": dict(Counter(o["effect"]["key"] for o in operators)),
              "kinds": dict(Counter(o["kind"] for o in operators)),
              "operators_per_family": dict(sorted(by_family.items())),
              "operators": operators}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "operators"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
