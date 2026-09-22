#!/usr/bin/env python3
"""Render the admitted VIKI-L2 library as an in-context memory block (in-context-library arm).

Spec 2026-09-21 §4.2, frozen once written:
- one paragraph per operator: name, effect, preconditions, ordered demo with variables kept
  as written (?x, ?y, ...), never bound to scene objects;
- the two-role relay operator in the same format, each role listed separately;
- each KEPT ordering rule of layer 2 as one sentence, after the operators;
- no name-canonicalisation vocabulary (layer 3 is a planner step, not library content);
- the block opens with exactly: "These are reusable skills and ordering rules. Use them to write the plan."

Operators carry no name field in the library, so the name is derived deterministically from the
position and the kind (e.g. "Skill 2 (achievement)"); the effect follows on its own line.

  python scripts/incontext_library/render.py --library outputs/v3_memories/memory_all.json --out <block.txt>
"""
import argparse
import hashlib
import json
from pathlib import Path

HEADER = "These are reusable skills and ordering rules. Use them to write the plan."


def effect_text(effect):
    key, subject = effect["key"], effect["subject"]
    if key == "pos.name":
        return f"{subject} is at {effect['value']}"
    if key == "is_activated":
        return f"{subject} is activated"
    return f"{key}({subject}) = {effect.get('value')}"


def action_text(action):
    return f"{action[0]}({', '.join(action[1:])})"


def precondition_text(op):
    parts = []
    for flag, value in sorted((op.get("preconditions") or {}).items()):
        parts.append(f"{flag} = {str(value).lower()}")
    for var, props in sorted((op.get("types") or {}).items()):
        for prop, value in sorted(props.items()):
            parts.append(f"{var}.{prop} = {str(value).lower()}")
    return "; ".join(parts) if parts else "none"


def operator_paragraph(index, op):
    kind = op.get("kind", "achievement")
    lines = [f"Skill {index} ({kind})",
             f"  Effect: {effect_text(op['effect'])}",
             f"  Preconditions: {precondition_text(op)}"]
    if op.get("roles"):
        for role in op["roles"]:
            steps = []
            for step in role["actions"]:
                text = action_text(step["action"])
                if step.get("after"):
                    waits = ", ".join(f"role {r} step {s + 1}" for r, s in step["after"])
                    text += f" [after {waits}]"
                steps.append(text)
            lines.append(f"  Role {role['variable']} demo: " + " -> ".join(steps))
    else:
        lines.append("  Demo: " + " -> ".join(action_text(a) for a in op["body"]))
    return "\n".join(lines)


def rule_sentence(p):
    """One sentence per kept layer-2 pattern: requirement A (a placement or an activation) is
    ordered before requirement B whenever the stated relations hold."""
    a = "placing an object A at a place P" if p["a_key"] == "pos" else "activating an object A"
    b = "placing an object B at a place Q" if p["b_key"] == "pos" else "activating an object B"
    cond = []
    if p.get("a_subject_is_b_target"):
        cond.append("A is the place Q" if p["b_key"] == "pos" else "A is B")
    if p.get("a_target_is_b_subject"):
        cond.append("P is the object B")
    if p.get("a_target_is_b_target"):
        cond.append("P is Q")
    if p.get("b_visits_a_target"):
        cond.append("the skill for B visits P")
    when = " and ".join(cond) if cond else "always"
    return f"Do {a} before {b} when {when}."


def render(library):
    ops = library["layer1"]["operators"]
    paragraphs = [HEADER, ""]
    for i, op in enumerate(ops, 1):
        paragraphs.append(operator_paragraph(i, op))
        paragraphs.append("")
    kept = [r["pattern"] for r in library["layer2"]["rules"] if r.get("kept")]
    assert len(kept) == len(library["layer2"]["kept_patterns"]), "kept rules != kept_patterns"
    paragraphs.append("Ordering rules:")
    for r in kept:
        paragraphs.append("- " + rule_sentence(r))
    return "\n".join(paragraphs).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    block = render(json.loads(args.library.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(block)
    sha = hashlib.sha256(block.encode()).hexdigest()
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(sha + "\n")
    print(sha, len(block), "chars")


if __name__ == "__main__":
    main()
