#!/usr/bin/env python3
"""v2 evaluation: union the family libraries per column, then score offline.

No library is rebuilt for a split. Each column names the families it may use; Layer 1 is
their union and Layers 2 and 3 are re-mined on the same pool. Scoring replays archived
answers, which is sound because the prompt never contains the library -- so every cell here
is zero generation except the per-row fallback re-ask.

The held-out-family column is assembled the way the fold table is: family X's rows are
scored by the memory built without X, and the eight slices are concatenated into one
924-row cell.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = "/root/venvs/partnr/bin/python"
A11 = ROOT / "results/viki_memory_experiments/amendment11"
UNION = ROOT / "scripts/viki_union_library.py"
EVAL = ROOT / "scripts/viki_eval_v2_intent_choice.py"

REPLAY = {
    ("72B", "id"): "intent_crew_clean.jsonl",
    ("72B", "recombination-text"): "recomb_text_agentic.jsonl",
    ("72B", "recombination-imaged"): "recomb_imaged_agentic.jsonl",
    ("30B", "id"): "m30_id.jsonl",
    ("30B", "recombination-text"): "m30_recomb_text.jsonl",
    ("30B", "recombination-imaged"): "m30_recomb_imaged.jsonl",
    ("7B", "id"): "m7_id.jsonl",
    ("7B", "recombination-text"): "m7_recomb_text.jsonl",
    ("7B", "recombination-imaged"): "m7_recomb_imaged.jsonl",
}
TAG = "v2"

ENDPOINT = {"72B": ("http://192.168.32.40:8050/v1", "qwen2.5-vl-72b-amendment3-f2"),
            "30B": ("http://127.0.0.1:8062/v1", "qwen3-vl-30b"),
            "7B": ("http://127.0.0.1:8061/v1", "qwen2.5-vl-7b")}


def say(message: str) -> None:
    print("[%s] %s" % (datetime.now().strftime("%m-%d %H:%M:%S"), message), flush=True)


def union(libraries, families, out, libs_root):
    """A family that yielded no operator contributes none; it does not abort the union.

    A missing library file means the family's sweep admitted nothing, which is a result and
    not a failure. Treating it as a missing input aborted every column here once. The
    families whose episodes Layers 2 and 3 are mined over are unchanged either way -- the
    pool is the family set, not the set that happened to produce operators.
    """
    if out.is_file():
        say("skip union %s" % out.name)
        return True
    paths = [libs_root / ("library_%s.json" % f) for f in libraries]
    present = [p for p in paths if p.is_file()]
    empty = [p.name for p in paths if not p.is_file()]
    if empty:
        say("families with no admitted operator (contributing nothing): %s" % ", ".join(empty))
    if not present:
        say("未执行，缺 任何一个族库 for %s" % out.name)
        return False
    command = [PY, str(UNION), "--libraries", *[str(p) for p in present],
               "--families", *families, "--out", str(out)]
    return subprocess.run(command, cwd=str(ROOT), timeout=10800).returncode == 0


def cell(tag, memory, model, split, extra=()):
    if (A11 / ("%s.jsonl" % tag)).is_file():
        say("skip   %s" % tag)
        return True
    source = A11 / REPLAY.get((model, split), "")
    if not source.is_file():
        say("未执行，缺 replay source for %s/%s" % (model, split))
        return False
    if not memory.is_file():
        say("未执行，缺 %s" % memory)
        return False
    url, served = ENDPOINT[model]
    say("start  %s" % tag)
    command = [PY, str(EVAL), "--memory", str(memory), "--split", split, "--tag", tag,
               "--replay", str(source), "--model", served, "--base-url", url,
               "--workers", "8", *extra]
    return subprocess.run(command, cwd=str(ROOT), timeout=10800).returncode == 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, default=ROOT / "results/frozen_sweep_v2.json")
    parser.add_argument("--libs-root", type=Path, default=ROOT / "outputs/v2_libraries")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs/v2_memories")
    parser.add_argument("--models", nargs="+", default=["72B", "30B", "7B"])
    # Which build these cells belong to. Default "v2" keeps every path this driver has ever
    # written byte-identical; a different prefix lands a whole build beside it rather than
    # on top of it, which is what a rebuilt library needs -- the v2 cells stay readable and
    # the two can be compared row by row.
    parser.add_argument("--tag-prefix", default="v2")
    args = parser.parse_args(argv)
    global TAG
    TAG = args.tag_prefix

    definition = json.loads(args.definition.read_text())
    build_families = definition["build_families"]
    fold_families = definition["fold_families"]
    cutting = definition["cutting_families"]
    delivery = definition["delivery_family"]
    args.out_root.mkdir(parents=True, exist_ok=True)

    # ---- ID column: all 14, matching the pool the baselines build from
    id_memory = args.out_root / "memory_all.json"
    say("=== union: all %d families (ID and comp(all) share this artefact) ===" % len(build_families))
    union(build_families, build_families, id_memory, args.libs_root)

    # ---- comp columns: half, most, whole
    comp_cd = args.out_root / "memory_comp_cut_delivery.json"
    comp_c = args.out_root / "memory_comp_cut.json"
    say("=== union: comp(cut+delivery) and comp(cut) ===")
    union(list(cutting) + [delivery], list(cutting) + [delivery], comp_cd, args.libs_root)
    union(list(cutting), list(cutting), comp_c, args.libs_root)

    # ---- held-out family: the other 13, so ID and this column differ by one family only
    fold_memories = {}
    for held in fold_families:
        rest = [f for f in build_families if f != held]
        out = args.out_root / ("memory_heldout_%s.json" % held)
        say("=== union: held-out %s (%d families) ===" % (held, len(rest)))
        if union(rest, rest, out, args.libs_root):
            fold_memories[held] = out

    # ---- no-trace control, arm (d), on the ID pool so it is comparable
    notrace = args.libs_root / "library_notrace.json"
    notrace_memory = args.out_root / "memory_notrace.json"
    if notrace.is_file() and not notrace_memory.is_file():
        say("=== union: no-trace control ===")
        subprocess.run([PY, str(UNION), "--libraries", str(notrace),
                        "--families", *build_families, "--out", str(notrace_memory)],
                       cwd=str(ROOT), timeout=10800)

    # ---- cells
    for model in args.models:
        say("=== cells for %s ===" % model)
        cell("%s_ours_%s_id" % (TAG, model), id_memory, model, "id")
        # The three comp cells are the half-to-whole curve: cut only, cut+delivery, all.
        cell("%s_ourscut_%s_text" % (TAG, model), comp_c, model, "recombination-text")
        cell("%s_ourscut_%s_imaged" % (TAG, model), comp_c, model, "recombination-imaged")
        cell("%s_ours_%s_text" % (TAG, model), comp_cd, model, "recombination-text")
        cell("%s_ours_%s_imaged" % (TAG, model), comp_cd, model, "recombination-imaged")
        cell("%s_oursall_%s_text" % (TAG, model), id_memory, model, "recombination-text")
        cell("%s_oursall_%s_imaged" % (TAG, model), id_memory, model, "recombination-imaged")
        if notrace_memory.is_file():
            cell("%s_notrace_%s_id" % (TAG, model), notrace_memory, model, "id")
            cell("%s_notrace_%s_text" % (TAG, model), notrace_memory, model, "recombination-text")
            cell("%s_notrace_%s_imaged" % (TAG, model), notrace_memory, model, "recombination-imaged")
        if model == "72B":
            cell("%s_abl_noorder_%s_id" % (TAG, model), id_memory, model, "id", ("--no-order",))
            cell("%s_abl_noorder_%s_text" % (TAG, model), comp_cd, model, "recombination-text",
                 ("--no-order",))
            cell("%s_abl_noorder_%s_imaged" % (TAG, model), comp_cd, model, "recombination-imaged",
                 ("--no-order",))
            cell("%s_abl_noground_%s_id" % (TAG, model), id_memory, model, "id", ("--no-grounding",))
            cell("%s_abl_noground_%s_text" % (TAG, model), comp_cd, model, "recombination-text",
                 ("--no-grounding",))
            cell("%s_abl_noground_%s_imaged" % (TAG, model), comp_cd, model, "recombination-imaged",
                 ("--no-grounding",))
        # held-out family: score each family's rows with the memory built without it
        for held, memory in fold_memories.items():
            cell("%s_fold_%s_%s" % (TAG, model, held), memory, model, "id")
        assemble_fold(model, fold_memories)
    say("done")
    return 0


def assemble_fold(model, fold_memories) -> None:
    """One 924-row cell: family X's rows taken from the run whose memory excluded X."""
    if not fold_memories:
        return
    out = A11 / ("%s_ours_%s_heldout.jsonl" % (TAG, model))
    if out.is_file():
        say("skip   fold assembly %s" % out.name)
        return
    rows = []
    for held in fold_memories:
        path = A11 / ("%s_fold_%s_%s.jsonl" % (TAG, model, held))
        if not path.is_file():
            say("未执行，缺 %s" % path)
            return
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("task_name") == held:
                    rows.append(record)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    say("held-out column for %s: %d rows" % (model, len(rows)))


if __name__ == "__main__":
    raise SystemExit(main())
