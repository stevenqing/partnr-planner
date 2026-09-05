"""Where does PARTNR's replay induction give up, and how often?

Mirrors `induce_from_trace` branch for branch, counting instead of inducing, so the
numbers describe the shipped inducer and not a paraphrase of it. Nothing is modified:
the module's own helpers are imported and the traces are read from disk.

Each count is an upper bound on what an agentic proposal step could recover at that point.
"""
import sys, json
sys.path.insert(0, ".")
from pathlib import Path
from collections import Counter, defaultdict
from our_method.skill_memory_v2.partnr_induction import (
    resolve, action_entities, ENTITY_ARGS, COMPLETING,
)

ROOT = Path("results/partnr_rollouts/train_mini")
files = sorted(ROOT.glob("*.json"))
print("trace files:", len(files))

give_up = Counter()
prop_total = 0
prop_kept = 0
traces_seen = 0
traces_zero = 0
effects_seen = Counter()
effects_lost = Counter()
completing_verbs = Counter()
missed_verbs = Counter()

for path in files:
    trace = json.loads(path.read_text())
    traces_seen += 1
    names = trace.get("handle_to_name") or {}
    steps = trace.get("steps") or []
    satisfied = trace.get("proposition_satisfied_at")
    propositions = trace.get("propositions") or []
    if not steps or not satisfied or len(satisfied) != len(propositions):
        give_up["trace: misaligned"] += 1
        continue

    finished_at = defaultdict(lambda: -1)
    order = sorted((int(w), i) for i, w in enumerate(satisfied) if int(w) >= 0)
    prop_total += len(order)
    kept_here = 0

    for when, index in order:
        proposition = propositions[index]
        fname = proposition.get("function_name")
        effects_seen[fname] += 1
        wanted = {k: resolve(proposition.get("args", {}).get(k), names) for k in ENTITY_ARGS}
        subjects = wanted["object_handles"] or wanted["entity_handles_a"]
        targets = wanted["receptacle_handles"] or wanted["entity_handles_b"] or wanted["room_ids"]
        if not subjects:
            give_up["prop: no resolvable subject"] += 1
            effects_lost[fname] += 1
            continue
        position = max((i for i, s in enumerate(steps) if s.get("sim_step", 0) <= when), default=None)
        if position is None:
            give_up["prop: no step at/before satisfaction"] += 1
            effects_lost[fname] += 1
            continue
        actor = None
        for agent, action in (steps[position].get("actions") or {}).items():
            entities = action_entities(action)
            touches = any(n in entities for n in subjects + targets)
            if action and action[0] in COMPLETING and touches:
                actor = agent
                break
        if actor is None:
            give_up["prop: no action explains it (attribution)"] += 1
            effects_lost[fname] += 1
            for agent, action in (steps[position].get("actions") or {}).items():
                if action:
                    ents = action_entities(action)
                    tag = "verb-not-completing" if action[0] not in COMPLETING else "does-not-touch"
                    missed_verbs["%s/%s" % (action[0], tag)] += 1
            continue
        start = finished_at[actor] + 1
        body = []
        for step in steps[start:position + 1]:
            action = (step.get("actions") or {}).get(actor)
            if action and (not body or body[-1] != action):
                body.append(list(action))
        finished_at[actor] = position
        if not body:
            give_up["prop: empty body"] += 1
            effects_lost[fname] += 1
            continue
        if not any(n in str(body) for n in subjects):
            give_up["prop: body never names subject (approx)"] += 1
            effects_lost[fname] += 1
            continue
        completing_verbs[body[-1][0]] += 1
        prop_kept += 1
        kept_here += 1
    if kept_here == 0:
        traces_zero += 1

print()
print("traces read              %5d" % traces_seen)
print("traces yielding nothing  %5d  (%.1f%%)" % (traces_zero, 100.0 * traces_zero / max(traces_seen, 1)))
print("propositions satisfied   %5d" % prop_total)
print("  induced into operators %5d  (%.1f%%)" % (prop_kept, 100.0 * prop_kept / max(prop_total, 1)))
print("  dropped                %5d  (%.1f%%)" % (prop_total - prop_kept,
      100.0 * (prop_total - prop_kept) / max(prop_total, 1)))
print()
print("== where it gives up ==")
for k, n in give_up.most_common():
    print("  %-42s %5d" % (k, n))
print()
print("== propositions by effect: seen vs lost ==")
print("  %-28s %7s %7s %7s" % ("function_name", "seen", "lost", "lost%"))
for fname, n in effects_seen.most_common():
    lost = effects_lost.get(fname, 0)
    print("  %-28s %7d %7d %6.1f%%" % (fname, n, lost, 100.0 * lost / max(n, 1)))
print()
print("== verbs that closed a kept operator ==")
for v, n in completing_verbs.most_common(8):
    print("  %-20s %5d" % (v, n))
print()
print("== at attribution failures, what the step's actions looked like ==")
for v, n in missed_verbs.most_common(10):
    print("  %-40s %5d" % (v, n))
