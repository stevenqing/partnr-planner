"""Sub-libraries of the reference, so "how far can an achievement-only library get" is a
number rather than a guess.

The agentic ladder can currently produce one shape: a single-robot achievement operator.
That is not a choice, it is what the workbench supports -- `run_operator` executes with one
runner (`sorted(env.agents)[0]`), so a coordination operator cannot be verified, and
`try_bind` maps requirements onto `pos.name` and `is_activated` only, so a repair effect
never meets a requirement to bind against.

Before spending a night of sampling on that shape it is worth knowing what it could score
at best. Deleting operators from the *reference* library answers it exactly, costs no model
call, and cannot be confounded by whether a model happened to write a good body: every
operator here is one the shipped inducer produced and the gate already scores at 1.0.

Each sub-library is written out for `viki_library_shim.py` to present to the gate. This
file only slices; the scoring is the gate's, unchanged.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REFERENCE = Path("results/viki_memory_experiments/amendment11/skill_memory_v2.json")

# name -> (predicate over an operator, what the slice is asking)
SLICES = {
    "ref_all":
        (lambda o: True,
         "every operator; the gate's own 1.0, re-scored through the shim as a control"),
    "ref_achievement":
        (lambda o: o.get("kind", "achievement") == "achievement",
         "single-robot achievement only -- the exact shape the ladder can produce today"),
    "ref_achievement_repair":
        (lambda o: o.get("kind", "achievement") in ("achievement", "repair"),
         "everything except coordination; isolates what the two-robot operators are worth"),
    "ref_posname_only":
        (lambda o: (o.get("effect") or {}).get("key") == "pos.name",
         "one effect key, all kinds; isolates what is_activated and unsealed are worth"),
    "ref_achievement_posname":
        (lambda o: o.get("kind", "achievement") == "achievement"
                   and (o.get("effect") or {}).get("key") == "pos.name",
         "where the ladder actually stands today: one shape, one key"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/reference_ablation"))
    arguments = parser.parse_args()

    reference = json.loads(REFERENCE.read_text())
    operators = reference["layer1"]["operators"]
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name, (keep, why) in SLICES.items():
        kept = [o for o in operators if keep(o)]
        path = arguments.out_dir / f"{name}.json"
        path.write_text(json.dumps({"operators": kept, "sliced_from": str(REFERENCE),
                                    "slice": name, "why": why}, indent=1))
        manifest[name] = {
            "operators": len(kept),
            "path": str(path),
            "why": why,
            "kinds": dict(Counter(o.get("kind", "achievement") for o in kept)),
            "effect_keys": dict(Counter((o.get("effect") or {}).get("key") for o in kept)),
        }

    (arguments.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("%-26s %5s  %s" % ("slice", "ops", "coverage"))
    for name, row in manifest.items():
        print("%-26s %5d  %s %s" % (name, row["operators"], row["kinds"], row["effect_keys"]))
    print("\n-> %s" % (arguments.out_dir / "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
