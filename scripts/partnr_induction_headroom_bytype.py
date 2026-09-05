"""Same counting, split by the training episode's task type.

The R-only library reads only the rearrange traces, so the question that decides whether
the lost demonstrations matter for the compositional claim is: do the rearrange traces
themselves demonstrate the spatial and room predicates, and does the inducer drop those?
"""
import sys, json
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from pathlib import Path
from collections import Counter, defaultdict
from our_method.skill_memory_v2.partnr_induction import resolve, action_entities, ENTITY_ARGS, COMPLETING
from partnr_compositional_report import episode_types

types = episode_types("train_mini")
ROOT = Path("results/partnr_rollouts/train_mini")

seen = defaultdict(Counter)   # tasktype -> effect -> n
lost = defaultdict(Counter)
traces_by_type = Counter()

for path in sorted(ROOT.glob("*.json")):
    trace = json.loads(path.read_text())
    eid = str(trace.get("episode_id"))
    ttype = types.get(eid, "?")
    traces_by_type[ttype] += 1
    names = trace.get("handle_to_name") or {}
    steps = trace.get("steps") or []
    satisfied = trace.get("proposition_satisfied_at")
    propositions = trace.get("propositions") or []
    if not steps or not satisfied or len(satisfied) != len(propositions):
        continue
    finished_at = defaultdict(lambda: -1)
    for when, index in sorted((int(w), i) for i, w in enumerate(satisfied) if int(w) >= 0):
        p = propositions[index]
        fname = p.get("function_name")
        seen[ttype][fname] += 1
        wanted = {k: resolve(p.get("args", {}).get(k), names) for k in ENTITY_ARGS}
        subjects = wanted["object_handles"] or wanted["entity_handles_a"]
        targets = wanted["receptacle_handles"] or wanted["entity_handles_b"] or wanted["room_ids"]
        drop = False
        if not subjects:
            drop = True
        else:
            position = max((i for i, s in enumerate(steps) if s.get("sim_step", 0) <= when), default=None)
            if position is None:
                drop = True
            else:
                actor = None
                for agent, action in (steps[position].get("actions") or {}).items():
                    ents = action_entities(action)
                    if action and action[0] in COMPLETING and any(n in ents for n in subjects + targets):
                        actor = agent
                        break
                if actor is None:
                    drop = True
                else:
                    start = finished_at[actor] + 1
                    body = []
                    for step in steps[start:position + 1]:
                        a = (step.get("actions") or {}).get(actor)
                        if a and (not body or body[-1] != a):
                            body.append(list(a))
                    finished_at[actor] = position
                    if not body:
                        drop = True
        if drop:
            lost[ttype][fname] += 1

print("traces by training task type:", dict(traces_by_type))
print()
for ttype in ["R", "R_S", "R_T", "R_S_T", "H_R", "?"]:
    if ttype not in seen:
        continue
    tot = sum(seen[ttype].values())
    los = sum(lost[ttype].values())
    print("== %s  (%d traces, %d propositions, %d lost = %.1f%%) ==" %
          (ttype, traces_by_type[ttype], tot, los, 100.0 * los / max(tot, 1)))
    for fname, n in seen[ttype].most_common():
        l = lost[ttype].get(fname, 0)
        print("   %-22s seen %5d   lost %5d  (%5.1f%%)" % (fname, n, l, 100.0 * l / max(n, 1)))
    print()
