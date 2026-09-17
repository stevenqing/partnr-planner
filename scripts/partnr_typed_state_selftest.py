"""Offline self-test for the typed_state interface: prompt -> parse -> project -> requirements -> actions.

No endpoint, no simulator. The model's answer is a literal string here, so a failure is the
interface's, not the model's -- the rule from CLAUDE.md: put a known-correct answer through the
same judge before concluding anything.
"""
import sys
sys.path.insert(0, ".")
from our_method.skill_memory_v2.partnr_typed_goals import (
    STATE_KEYS, parse_typed, project, as_requirements, typed_prompt, StepZero)
from our_method.skill_memory_v2.partnr_memory import PartnrSkillMemory

memory = PartnrSkillMemory.load("results/partnr_operators_llm5.json")
effects = memory.effects()
scene = StepZero(rooms=["kitchen_1", "bedroom_2"], furniture=["table_7", "counter_3", "cabinet_9"],
                 openable=["cabinet_9"],
                 room_of_furniture={"table_7": "kitchen_1", "counter_3": "kitchen_1", "cabinet_9": "bedroom_2"})
kinds = ["mug", "lamp", "book"]
answer = "mug | clean | -\nlamp | powered_on | -\nbook | on | table_7\n"

fails = []
def check(name, got, want):
    ok = got == want
    print(("  ok   " if ok else "  FAIL ") + f"{name}: {got!r}" + ("" if ok else f" != {want!r}"))
    if not ok: fails.append(name)

print("== switch OFF: every step must behave exactly as before ==")
off = parse_typed(answer, effects, state=False)
check("parse drops the two state lines", [r["key"] for r in off], ["is_on_top"])
kept_off, _ = project(off, scene, kinds, effects, None, "", state=False)
check("project keeps only the placement", [r["key"] for r in kept_off], ["is_on_top"])
p_off = typed_prompt("world", "Clean the mug.", kinds, scene, effects, examples="RS", state=False)
check("prompt offers no state relation", ("clean" in p_off), False)

print("== switch ON ==")
on = parse_typed(answer, effects, state=True)
check("parse keeps all three", [r["key"] for r in on], ["is_clean", "is_powered_on", "is_on_top"])
check("state lines carry no target", [r["target"] for r in on[:2]], [None, None])
kept, counts = project(on, scene, kinds, effects, None, "", state=True)
check("project keeps all three", [r["key"] for r in kept], ["is_clean", "is_powered_on", "is_on_top"])
check("state target stays None", kept[0]["target"], None)
p_on = typed_prompt("world", "Clean the mug.", kinds, scene, effects, examples="RS", state=True)
check("prompt offers clean", ("clean    -- no place" in p_on), True)
check("prompt offers powered_on", ("powered_on -- no place" in p_on), True)
check("prompt carries a state example", ("mug | clean | -" in p_on), True)

print("== a state subject may be furniture: every is_clean proposition in gate_H/conf_H names one ==")
for line, want_subject in (("table_7 | clean | -", "table_7"), ("counter_3 | clean | -", "counter_3"),
                           ("mug | clean | -", "mug"), ("lamp | powered_on | -", "lamp")):
    got, _ = project(parse_typed(line, effects, state=True), scene, kinds, effects, None, "", state=True)
    check(f"kept {line!r}", [(r["key"], r["subject"], r["target"]) for r in got],
          [(("is_clean" if "clean" in line else "is_powered_on"), want_subject, None)])
placement, _ = project(parse_typed("table_7 | on | counter_3", effects, state=True), scene, kinds,
                       effects, None, "", state=True)
check("a placement with a furniture subject is still dropped", placement, [])

print("== a library without H operators must not offer them even with the switch on ==")
iir1 = PartnrSkillMemory.load("results/partnr_operators_iir1.json")
p_iir1 = typed_prompt("world", "Clean the mug.", kinds, scene, iir1.effects(), examples="RS", state=True)
check("iir1 prompt offers no clean", ("clean" in p_iir1), False)
kept_iir1, _ = project(parse_typed(answer, iir1.effects(), state=True), scene, kinds,
                       iir1.effects(), None, "", state=True)
check("iir1 keeps only the placement", [r["key"] for r in kept_iir1], ["is_on_top"])

print("== requirements and grounding ==")
for stages in (False, True):
    reqs = as_requirements(kept, "Clean the mug and switch on the lamp, then put the book on the table.",
                           stages=stages, same_object=stages)
    check(f"as_requirements(stages={stages}) keeps three", [r["key"] for r in reqs],
          ["is_clean", "is_powered_on", "is_on_top"])

class FakeView:
    def resolve(self, name, taken=None): return {"mug": "mug_1", "lamp": "lamp_4", "table_7": "table_7"}.get(name)
    def container_of(self, name): return None
    def faucet_furniture(self): return "sink_2"
reqs = as_requirements(kept, "Clean the mug.", stages=False, same_object=False)
bound = [dict(r, subject={"mug": "mug_1", "lamp": "lamp_4", "book": "book_3"}[r["subject"]],
              target=(r["target"] if r["target"] is None else r["target"])) for r in reqs]
for r in bound[:2]:
    ops = memory.operators_for(r["key"])
    actions = memory.ground(ops[0], r, FakeView()) if ops else None
    check(f"ground {r['key']}", actions,
          [["Navigate", r["subject"]], ["Clean" if r["key"] == "is_clean" else "PowerOn", r["subject"]]])

print()
print("FAILED: " + ", ".join(fails) if fails else "ALL CHECKS PASSED")
sys.exit(1 if fails else 0)
