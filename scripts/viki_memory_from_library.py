"""Wrap an operator library as a full skill-memory artefact so the end-to-end eval can load it.

The gate scores Layer 1 alone. The benchmark arm needs all three layers, so Layers 2 and 3
are taken from the reference artefact and only Layer 1 varies -- the same substitution
`viki_inducer_bench.py` makes, and for the same reason: what is being compared is the
operator library, and letting the ordering rules or the name grounding differ too would
make the comparison unreadable.

That substitution is a **limit on the claim**, not a detail. An agent-built Layer 1 scored
this way is not an agent-built memory; it is an agent-built operator library carried by the
reference's ordering and grounding. Say so wherever the number is quoted.
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
    arguments = parser.parse_args()

    reference = json.loads(REFERENCE.read_text())
    library = json.loads(arguments.library.read_text())
    operators = library["operators"] if isinstance(library, dict) else library

    record = dict(reference)
    record["layer1"] = {"operators": operators}
    record["built_from"] = str(arguments.library)
    record["layers_2_3_borrowed_from"] = str(REFERENCE)
    record.pop("self_check", None)          # the reference's, not this library's
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(record, indent=1))
    print("%d operators from %s -> %s (layers 2 and 3 from the reference)"
          % (len(operators), arguments.library, arguments.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
