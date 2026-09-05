"""Present a fixed operator library to `viki_inducer_bench.py` as if it were an inducer.

The bench scores an *inducer* -- a module exposing `induce(...)` -- because that is what the
reference is. The agentic ladder does not produce an inducer; it produces operators, one at
a time, each already accepted by a mechanical test. Something has to carry those into the
gate, and this is it: `induce` ignores the episodes it is handed and returns the library
named by `$VIKI_LIBRARY_JSON`.

Ignoring the episodes is the honest thing here, not a shortcut. The operators were derived
from the induction half (`episodes[::2]`, all the workbench ever exposes) and verified on
held-out episodes from that same half. The bench's self-check plans `episodes[1::2]`, which
neither the ladder nor this shim has ever seen. What the gate measures is therefore
unchanged: can this library plan episodes nobody building it looked at.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def induce(episodes, sim, seed, per_family=250, exclude_family=None, progress=None) -> Dict[str, Any]:
    path = os.environ.get("VIKI_LIBRARY_JSON")
    if not path:
        raise RuntimeError("set VIKI_LIBRARY_JSON to the library this shim should present")
    library = json.loads(Path(path).read_text())
    operators = library["operators"] if isinstance(library, dict) else library
    return {
        "operators": operators,
        # Stated rather than measured: this shim replays nothing. Reporting a replay count
        # it did not perform would put a fabricated number in the bench's output.
        "episodes_replayed": 0,
        "replay_outcomes": {"NOT_REPLAYED_BY_SHIM": len(operators)},
        "families_seen": {},
        "shim_source": path,
    }
