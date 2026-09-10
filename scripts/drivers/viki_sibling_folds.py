#!/usr/bin/env python3
"""The held-out column with the whole sibling group removed, for our arm.

Groups, threshold and predictions are fixed in results/sibling_folds_preregistration.json
and come from instruction embeddings alone. Only three evaluation folds have a sibling, so
only those three memories are rebuilt; the other five folds are unchanged by construction
and their existing cells are reused when the column is assembled.

Our cells are replays of archived responses, so this half costs no model calls.
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
PRE = ROOT / "results/sibling_folds_preregistration.json"

REPLAY = {"72B": "intent_crew_clean.jsonl", "30B": "m30_id.jsonl", "7B": "m7_id.jsonl"}
ENDPOINT = {"72B": ("http://192.168.32.40:8050/v1", "qwen2.5-vl-72b-amendment3-f2"),
            "30B": ("http://127.0.0.1:8062/v1", "qwen3-vl-30b"),
            "7B": ("http://127.0.0.1:8061/v1", "qwen2.5-vl-7b")}


def say(message: str) -> None:
    print("[%s] %s" % (datetime.now().strftime("%m-%d %H:%M:%S"), message), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, default=ROOT / "results/frozen_sweep_v2.json")
    parser.add_argument("--libs-root", type=Path, default=ROOT / "outputs/v2_libraries")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs/v2_memories")
    parser.add_argument("--models", nargs="+", default=["72B", "30B", "7B"])
    # Which build these cells belong to; "v2" keeps every path this driver has written.
    parser.add_argument("--tag-prefix", default="v2")
    args = parser.parse_args()
    tag_prefix = args.tag_prefix

    pre = json.loads(PRE.read_text())
    groups = {f: g for g in pre["groups"] for f in g}
    affected = pre["affected_eval_folds"]
    build = json.loads(args.definition.read_text())["build_families"]
    say("pre-registration: %s" % PRE)
    say("affected folds: %s" % ", ".join(affected))

    memories = {}
    for held in affected:
        group = groups.get(held, [held])
        rest = [f for f in build if f not in group]
        out = args.out_root / ("memory_heldoutgrp_%s.json" % held)
        say("union: hiding %s (%d families left)" % ("+".join(group), len(rest)))
        if out.is_file():
            say("skip union %s" % out.name)
            memories[held] = out
            continue
        paths = [args.libs_root / ("library_%s.json" % f) for f in rest]
        present = [p for p in paths if p.is_file()]
        if not present:
            say("未执行，缺 任何一个族库 for %s" % out.name)
            return 1
        command = [PY, str(UNION), "--libraries", *[str(p) for p in present],
                   "--families", *rest, "--out", str(out)]
        if subprocess.run(command, cwd=str(ROOT), timeout=10800).returncode != 0:
            say("FAILED union %s" % out.name)
            return 1
        memories[held] = out

    for model in args.models:
        source = A11 / REPLAY[model]
        if not source.is_file():
            say("未执行，缺 replay source %s" % source)
            continue
        url, served = ENDPOINT[model]
        for held, memory in memories.items():
            tag = "%s_foldgrp_%s_%s" % (tag_prefix, model, held)
            if (A11 / ("%s.jsonl" % tag)).is_file():
                say("skip   %s" % tag)
                continue
            say("start  %s" % tag)
            command = [PY, str(EVAL), "--memory", str(memory), "--split", "id", "--tag", tag,
                       "--replay", str(source), "--model", served, "--base-url", url,
                       "--workers", "8"]
            if subprocess.run(command, cwd=str(ROOT), timeout=10800).returncode != 0:
                say("FAILED %s" % tag)

    # One 924-row column: an affected family's rows come from its group-held-out run, an
    # unaffected family's from the single-family run it already had -- those two are the
    # same experiment for a family with no sibling.
    for model in args.models:
        out = A11 / ("%s_ours_%s_heldout_sibgrp.jsonl" % (tag_prefix, model))
        rows, missing = [], []
        for family in pre["affected_eval_folds"] + pre["unaffected_eval_folds_reused_as_is"]:
            stem = ("%s_foldgrp" % tag_prefix) if family in affected else ("%s_fold" % tag_prefix)
            path = A11 / ("%s_%s_%s.jsonl" % (stem, model, family))
            if not path.is_file():
                missing.append(path.name)
                continue
            for line in path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record.get("task_name") == family:
                        rows.append(record)
        if missing:
            say("未执行 column %s, 缺 %s" % (out.name, ", ".join(missing)))
            continue
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        solved = sum(1 for r in rows if r.get("reason") == "SOLVED")
        say("column %s: %d rows, %d solved = %.4f" % (out.name, len(rows), solved,
                                                      solved / max(len(rows), 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
