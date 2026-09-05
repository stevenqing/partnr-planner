"""The same operator library against every model's own goal parses, on the ID split.

Until now the agent-built library had been scored on exactly one cell: 924 ID rows against
`probe2_zeroshot_v2`. That is one reasoning model, and it says nothing about whether the
library survives a weaker one -- which matters, because Layer 1 is supposed to carry work the
model no longer has to do, so a memory that only helps the strongest model is not the claim.

Two axes are being kept apart and must not be confused when this is quoted:

  the INDUCER model   which model derived the operators (72B here, with a few from 30B)
  the REASONING model whose goal parses the library is then asked to execute -- what varies
                      across the columns below

Responses are replayed from disk, so no endpoint is called and every column sees byte-identical
model output regardless of which library is being scored. Layers 2 and 3 come from the
reference artefact in every row, so only Layer 1 varies.

The recombination split is NOT covered here: it is 297 rows under a different harness
(`viki_amendment10_run.py`) that calls the model rather than replaying it, so pointing a
library at it is a separate piece of work, not a flag.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")

MODELS = [
    ("72B", "probe2_zeroshot_v2"),
    ("30B", "m30_id"),
    ("7B", "m7_id"),
]
LIBRARIES = [
    ("reference", "results/viki_memory_experiments/amendment11/skill_memory_v2.json"),
    ("agentic", "outputs/agentic_memory_runner.json"),
]


def usable(responses: str) -> str:
    """Refuse a response set that is not goal parses, instead of scoring it as zero.

    The first version of this script ran `m30_id` and `m7_id` and reported 0.00% for every
    library including the reference -- which is the tell that the test is broken, not the
    library. Those files come from the intent/delegation arm and carry a different schema
    (`{"work": [{"do": "put", "X": ..., "Y": ..., "robots": [...]}]}`, plus `cast_by_model`),
    so `to_predicates` extracts nothing and every episode scores zero for every memory.

    A reasoning-model column can only be filled by a response set in the goal-parse schema
    this arm reads. Only `probe2_*` on disk is one, and all of those are the same model.
    """
    path = ROOT / "results/viki_memory_experiments/amendment11" / (responses + ".jsonl")
    if not path.is_file():
        return "missing: %s" % path
    with path.open() as handle:
        first = json.loads(handle.readline())
    raw = str(first.get("raw") or "")
    if '"goals"' not in raw:
        return ("wrong schema: `raw` carries %s, not goal parses. This arm reads "
                "{\"goals\": [...]}; scoring it would report zeros that are the "
                "harness, not the memory."
                % ("a `work` plan" if '"work"' in raw else "an unrecognised shape"))
    return ""


def run(memory: str, responses: str, tag: str):
    out = subprocess.run(
        [sys.executable, "scripts/viki_eval_skill_memory_v2.py",
         "--memory", memory, "--responses", responses, "--tag", tag],
        capture_output=True, text=True, cwd=ROOT)
    solved = total = None
    families = {}
    for line in out.stdout.splitlines():
        if line.strip().startswith("accuracy"):
            piece = line.split()[1]
            solved, total = (int(x) for x in piece.split("/"))
        parts = line.split()
        if len(parts) == 3 and "/" in parts[1] and parts[2].endswith("%"):
            got, seen = parts[1].split("/")
            families[parts[0]] = (int(got), int(seen))
    if solved is None:
        return {"error": (out.stdout + out.stderr)[-400:]}
    return {"solved": solved, "total": total,
            "rate": round(solved / total, 4), "by_family": families}


def main() -> int:
    table = {}
    for model, responses in MODELS:
        for library, memory in LIBRARIES:
            tag = "matrix_%s_%s" % (library, model)
            problem = usable(responses)
            if problem:
                print("SKIP    %-10s x %-9s -- %s" % (library, model, problem), flush=True)
                table["%s/%s" % (library, model)] = {
                    "model": model, "library": library, "responses": responses,
                    "memory": memory, "unusable": problem}
                continue
            print("running %-10s x %-9s ..." % (library, model), flush=True)
            table["%s/%s" % (library, model)] = dict(
                run(memory, responses, tag), model=model, library=library,
                responses=responses, memory=memory)

    # Disk before stdout.
    (ROOT / "outputs/viki_library_model_matrix.json").write_text(json.dumps(table, indent=1))

    print()
    print("ID split, 924 rows, responses replayed from disk (no endpoint called)")
    print("Layers 2 and 3 are the reference's in every row; only Layer 1 varies.\n")
    print("%-12s %18s %18s %18s" % ("library", "72B", "30B", "7B"))
    for library, _ in LIBRARIES:
        cells = []
        for model, _ in MODELS:
            row = table["%s/%s" % (library, model)]
            cells.append("%s/%s = %.2f%%" % (row["solved"], row["total"], 100 * row["rate"])
                         if "solved" in row else
                         ("not measurable" if row.get("unusable") else "ERROR"))
        print("%-12s %18s %18s %18s" % (library, *cells))
    blocked = sorted({r["responses"] for r in table.values() if r.get("unusable")})
    if blocked:
        print("\nno goal-parse response set exists for: %s" % ", ".join(blocked))
        print("Those columns need the models run over the 924 ID rows in this arm's own")
        print("prompt; they cannot be filled from what is on disk.")
    print("\n-> outputs/viki_library_model_matrix.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
