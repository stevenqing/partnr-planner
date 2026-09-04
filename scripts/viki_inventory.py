#!/usr/bin/env python3
"""What has already been generated, before anything else is.

The cross-model main table needs 60 cells -- three models, five methods, four splits --
and the point of this script is to find out how many of them are already sitting on disk
as raw responses. A cell with an archive is free: it is re-scored offline under the
JSON-tolerant convention and never touches a generation budget. Only the genuinely
missing cells cost anything, and this is what says which those are.

Nothing here calls a model, and nothing here re-scores: scoring needs the simulator and
belongs in `viki_report_matrix.py`, which is the one place a number in the paper may come
from. What is reported per cell is only what can be read straight off the archive --
rows, format compliance, token means, and the served model the run recorded -- so this
can never disagree with the report; it only says where to point it.

Layout, read off `viki_report_matrix.py` and the fold driver:

  amendment8b/<arm>.jsonl                    ID, prompt-shaped arms
  amendment8b/folds/<family>/<arm>.jsonl     held-out-family OOD, eight families
  amendment10/{imaged,text}/<arm>.jsonl      recombination, the paired pair
  amendment11/<tag>.jsonl                    skill memory v2 and the MEMENTO-style port

The model is not in the record; it is in the sibling `.run.json` as `served_model`, and
where that file is missing the archive is reported as unlabelled rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
EXP = ROOT / "results/viki_memory_experiments"
A8B, A10, A11 = EXP / "amendment8b", EXP / "amendment10", EXP / "amendment11"

MODELS = ["72B", "30B", "7B", "GPT-4o"]
METHODS = ["zero-shot", "trajectory RAG", "skill memory v1", "G-Memory", "MEMENTO-style"]
SPLITS = ["ID", "OOD folds", "recomb imaged", "recomb text"]

# The file stem each method is archived under, for the prompt-shaped arms.
STEM = {
    "zero-shot": "zero_shot",
    "trajectory RAG": "trajectory_rag",
    "skill memory v1": "skill_memory.fullactions_k8",
    "G-Memory": "gmemory",
}
# 72B is the unsuffixed archive; the other models were only ever run for MEMENTO, and
# under their own tags. A suffix that does not exist is exactly the finding.
SUFFIX = {"72B": "", "30B": "_m30", "7B": "_m7", "GPT-4o": "_gpt4o"}


def fold_names() -> List[str]:
    folder = A8B / "folds"
    return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []


def paths_for(model: str, method: str, split: str) -> List[Path]:
    """Every file this cell would live in. A cell is present when all of them are."""
    suffix = SUFFIX[model]
    if method == "MEMENTO-style":
        if split == "ID":
            return [A11 / f"memento_id{suffix}.jsonl"]
        if split == "recomb imaged":
            return [A11 / f"memento_recomb_imaged{suffix}.jsonl"]
        if split == "recomb text":
            return [A11 / f"memento_recomb_text{suffix}.jsonl"]
        return [A11 / f"memento_fold_{name}{suffix}.jsonl" for name in fold_names()]
    stem = STEM[method]
    # The prompt-shaped arms carry no model in the filename, so a non-72B cell can only
    # be present under a suffixed name -- which is how a missing model shows up as absent
    # rather than silently reading the 72B archive.
    stem = stem if model == "72B" else f"{stem}{suffix}"
    if split == "ID":
        return [A8B / f"{stem}.jsonl"]
    if split == "recomb imaged":
        return [A10 / "imaged" / f"{stem}.jsonl"]
    if split == "recomb text":
        return [A10 / "text" / f"{stem}.jsonl"]
    return [A8B / "folds" / name / f"{stem}.jsonl" for name in fold_names()]


def read(path: Path) -> Optional[Dict[str, Any]]:
    """Rows, format compliance and token means, straight off the archive."""
    if not path.is_file():
        return None
    rows = 0
    format_ok = 0
    scored = 0
    prompt_tokens: List[float] = []
    completion_tokens: List[float] = []
    has_score = False
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        rows += 1
        if "format_score" in record:
            format_ok += int(round(float(record.get("format_score") or 0)))
        for key, sink in (("prompt_tokens", prompt_tokens),
                          ("completion_tokens", completion_tokens)):
            value = record.get(key)
            if isinstance(value, (int, float)):
                sink.append(float(value))
        for key in ("score", "accuracy"):
            if key in record:
                has_score = True
                scored += int(round(float(record.get(key) or 0)))
                break
    run = path.with_suffix(".jsonl.run.json")
    served = None
    if run.is_file():
        try:
            served = json.load(run.open()).get("served_model")
        except Exception:
            served = None
    return {
        "rows": rows,
        "format": format_ok / rows if rows else 0.0,
        "archived_score": scored if has_score else None,
        "prompt_tokens": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None,
        "completion_tokens": (sum(completion_tokens) / len(completion_tokens)
                              if completion_tokens else None),
        "served_model": served,
        "path": str(path.relative_to(ROOT)),
    }


def cell(model: str, method: str, split: str) -> Dict[str, Any]:
    wanted = paths_for(model, method, split)
    found = [read(path) for path in wanted]
    present = [f for f in found if f]
    return {
        "model": model, "method": method, "split": split,
        "wanted": len(wanted), "present": len(present),
        "rows": sum(f["rows"] for f in present),
        "parts": present,
        "missing": [str(p.relative_to(ROOT)) for p, f in zip(wanted, found) if not f],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", action="store_true",
                        help="per-cell metadata for the cells that exist")
    arguments = parser.parse_args()

    folds = fold_names()
    print(f"eight-fold families: {len(folds)} -> {', '.join(folds)}\n")

    grid: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for model in MODELS:
        for method in METHODS:
            for split in SPLITS:
                grid[(model, method, split)] = cell(model, method, split)

    def mark(c: Dict[str, Any]) -> str:
        if c["present"] == 0:
            return "-"
        if c["present"] < c["wanted"]:
            return f"partial {c['present']}/{c['wanted']}"
        return f"yes ({c['rows']})"

    for model in MODELS:
        print(f"== {model} ==")
        print("  " + "method".ljust(18) + "".join(s.rjust(16) for s in SPLITS))
        for method in METHODS:
            cells = [mark(grid[(model, method, split)]) for split in SPLITS]
            print("  " + method.ljust(18) + "".join(c.rjust(16) for c in cells))
        print()

    have = sum(1 for c in grid.values() if c["present"] == c["wanted"])
    part = sum(1 for c in grid.values() if 0 < c["present"] < c["wanted"])
    print(f"{have}/{len(grid)} cells fully archived, {part} partial, "
          f"{len(grid) - have - part} absent")

    if arguments.detail:
        print("\n== per-cell metadata (archived cells only) ==")
        for key in sorted(grid):
            c = grid[key]
            if not c["present"]:
                continue
            first = c["parts"][0]
            served = {p["served_model"] for p in c["parts"] if p["served_model"]}
            score = sum(p["archived_score"] for p in c["parts"]
                        if p["archived_score"] is not None) or None
            print(f"\n  {c['model']} / {c['method']} / {c['split']}")
            print(f"    rows {c['rows']}   files {c['present']}/{c['wanted']}")
            fmt = sum(p['format'] * p['rows'] for p in c['parts']) / max(c['rows'], 1)
            print(f"    format {fmt:.4f}"
                  + (f"   archived score {score}" if score is not None else "   (responses only)"))
            print(f"    served_model {sorted(served) if served else 'unlabelled'}")
            if first["prompt_tokens"]:
                print(f"    prompt_tokens {first['prompt_tokens']:.0f}   "
                      f"completion_tokens {first['completion_tokens']:.0f}")
            print(f"    {first['path']}")
            if c["missing"]:
                print(f"    MISSING {len(c['missing'])}: {c['missing'][0]} ...")


if __name__ == "__main__":
    main()
