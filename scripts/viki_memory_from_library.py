"""Wrap an operator library as a full skill-memory artefact so the end-to-end eval can load it.

The gate scores Layer 1 alone. The benchmark arm needs all three layers, so Layers 2 and 3
are taken from a reference artefact and only Layer 1 varies -- the same substitution
`viki_inducer_bench.py` makes, and for the same reason: what is being compared is the
operator library, and letting the ordering rules or the name grounding differ too would
make the comparison unreadable.

**Why the substitution is sound, and what it is not.** Neither layer is a function of
Layer 1 at build time: `dependencies.mine(episodes, sim, seed, per_family, exclude_family)`
and `vocabulary.harvest(episodes, exclude_family)` take no library argument, and Layer 2's
mining reads its `visits` off the replayed training trace, not off any operator body. So
rebuilding those layers for this library reproduces the reference's byte for byte, and
borrowing them is a computational shortcut rather than an import of hand-authored content.
`scripts/viki_layer23_equivalence.py` checks that claim mechanically; run it before quoting
a number produced here.

What the substitution IS: Layer 2's hypothesis space (`dependencies._describe`'s six
features) and Layer 3's harvesting rules are authored. They are shared inductive bias, held
fixed across every arm; they are not memory entries. An agent-built Layer 1 scored this way
is an agent-built operator library carried by mined ordering and grounding. Say so wherever
the number is quoted.

**Folds.** The reference default is the FULL memory. A held-out-family cell must pass
`--reference results/.../skill_memory_v2.fold_<family>.json`, because the fold rebuilds all
three layers with that family excluded; grafting a fold Layer 1 onto the full Layer 2/3
leaks the held-out family into the ordering rules and the vocabulary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REFERENCE = Path("results/viki_memory_experiments/amendment11/skill_memory_v2.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE,
                        help="artefact supplying Layers 2 and 3. Default: the full memory. "
                             "For a held-out-family cell pass that family's fold artefact, "
                             "or the held-out family leaks into ordering and vocabulary.")
    arguments = parser.parse_args()

    reference = json.loads(arguments.reference.read_text())
    library = json.loads(arguments.library.read_text())
    operators = library["operators"] if isinstance(library, dict) else library

    record = dict(reference)
    record["layer1"] = {"operators": operators}
    record["built_from"] = str(arguments.library)
    record["layers_2_3_borrowed_from"] = str(arguments.reference)
    record.pop("self_check", None)          # the reference's, not this library's
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(record, indent=1))
    print("%d operators from %s -> %s (layers 2 and 3 from %s)"
          % (len(operators), arguments.library, arguments.out, arguments.reference))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
