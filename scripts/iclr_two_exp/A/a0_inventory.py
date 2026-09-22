#!/usr/bin/env python3
"""A0 of results/iclr_two_exp_2026-09-22/SPEC.md: inventory, zero model calls.

Writes into results/iclr_two_exp_2026-09-22/A/:
  a0_first_turn_cells.csv   the 12 first-turn cells (path, rows, sha256) and the Figure 3 check
  a0_artifacts.csv          planner / re-ask / ordering rules / vocabulary / schema files with sha256
  a0_family_effects.csv     14 training families x effects in the reference plans, induction half only
  a0_inventory.json         everything above plus the gate verdict
Disk before stdout.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/iclr_two_exp_2026-09-22/A"
A11 = ROOT / "results/viki_memory_experiments/amendment11"
REG = ROOT / "results/paper_viki_iclr2027"
TRAIN = Path("/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet")

MODELS = {"72B": "qwen2.5-vl-72b-instruct", "30B": "qwen3-vl-30b-a3b-instruct", "7B": "qwen2.5-vl-7b-instruct"}
SPLITS = {"id": ("v3_ours_%s_id.jsonl", "id", 924),
          "ood_single_family": ("v3_ours_%s_heldout.jsonl", "ood_single_family", 924),
          "cg_image": ("v3_oursall_%s_imaged.jsonl", "cg_image", 297),
          "cg_text": ("v3_oursall_%s_text.jsonl", "pure_text", 297)}
REPLAY = {("72B", "id"): "intent_crew_clean.jsonl", ("72B", "cg_text"): "recomb_text_agentic.jsonl",
          ("72B", "cg_image"): "recomb_imaged_agentic.jsonl", ("30B", "id"): "m30_id.jsonl",
          ("30B", "cg_text"): "m30_recomb_text.jsonl", ("30B", "cg_image"): "m30_recomb_imaged.jsonl",
          ("7B", "id"): "m7_id.jsonl", ("7B", "cg_text"): "m7_recomb_text.jsonl",
          ("7B", "cg_image"): "m7_recomb_imaged.jsonl"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def canon_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sums = {}
    for line in (REG / "SHA256SUMS.txt").read_text().splitlines():
        h, p = line.split(None, 1)
        sums[p.strip().lstrip("./")] = h
    registry = {c["cell_id"]: c for c in json.loads((REG / "cells.json").read_text())["cells"]}

    cells, gate_problems = [], []
    for model, long in MODELS.items():
        for split, (pattern, regsplit, n_expected) in SPLITS.items():
            path = A11 / (pattern % model)
            row = {"model": model, "split": split, "path": str(path.relative_to(ROOT))}
            if not path.is_file():
                row["status"] = "MISSING"
                gate_problems.append("missing %s" % path)
                cells.append(row)
                continue
            rows = jsonl(path)
            row.update(rows=len(rows), expected=n_expected, sha256=sha(path),
                       solved=sum(1 for r in rows if r.get("reason") == "SOLVED"),
                       reask_rows=sum(1 for r in rows if r.get("reask")),
                       rows_with_raw=sum(1 for r in rows if r.get("raw") is not None))
            # Figure 3 is the RQ2 figure (fig:viki_rq2); its full arm is the registered
            # rows/rq2/full/<model>/<split>.jsonl, whose SHA256SUMS entry we also check.
            reg_path = REG / "rows/rq2/full" / long / ("%s.jsonl" % regsplit)
            rel = str(reg_path.relative_to(REG))
            row["fig3_rows"] = str(reg_path.relative_to(ROOT))
            row["fig3_sha256"] = sha(reg_path) if reg_path.is_file() else None
            row["fig3_sha_matches_SHA256SUMS"] = row["fig3_sha256"] == sums.get(rel)
            reg_cell = registry.get("rq2/full/%s/%s" % (model, regsplit)) or {}
            row["registry_source"] = (reg_cell.get("source") or {}).get("path")
            row["registry_replay_source"] = reg_cell.get("replay_source")
            same, differ, missing = 0, 0, 0
            if reg_path.is_file():
                reg = {int(r["example_id"]): r for r in jsonl(reg_path)}
                row["fig3_n"] = len(reg)
                row["fig3_solved"] = sum(1 for r in reg.values()
                                         if r.get("success") in (1, True, 1.0) or r.get("strict_solved") in (1, True))
                for r in rows:
                    other = reg.get(int(r["index"]))
                    if other is None:
                        missing += 1
                    elif other.get("raw_output") == r.get("raw"):
                        same += 1
                    else:
                        differ += 1
            row.update(fig3_first_turn_identical=same, fig3_first_turn_differ=differ, fig3_missing=missing)
            # the archived answer set the cell replays (first turn); OOD rows replay the ID set
            src_name = REPLAY.get((model, "id" if split == "ood_single_family" else split))
            src = A11 / src_name
            row["first_turn_source"] = str(src.relative_to(ROOT))
            row["first_turn_source_sha256"] = sha(src)
            archived = {int(r["index"]): r.get("raw") for r in jsonl(src)}
            row["source_rows"] = len(archived)
            row["raw_equals_source"] = sum(1 for r in rows
                                          if archived.get(int(r["index"])) is not None
                                          and archived[int(r["index"])][-3000:] == r.get("raw"))
            if len(rows) != n_expected or differ or missing or not row["fig3_sha_matches_SHA256SUMS"]:
                gate_problems.append("%s %s: rows=%d differ=%d missing=%d sumsok=%s"
                                     % (model, split, len(rows), differ, missing,
                                        row["fig3_sha_matches_SHA256SUMS"]))
            row["status"] = "ok"
            cells.append(row)

    # ---- artefacts
    mem = json.loads((ROOT / "outputs/v3_memories/memory_all.json").read_text())
    runner = (ROOT / "scripts/viki_eval_v2_intent_choice.py").read_text().splitlines()
    reask_lines = [i + 1 for i, l in enumerate(runner) if "if plan is None and casting" in l
                   or "REASK %" in l or l.startswith("REASK =")]
    artefacts = []

    def add(role, rel, note=""):
        p = ROOT / rel
        artefacts.append({"role": role, "path": rel, "exists": p.exists(),
                          "sha256": sha(p) if p.is_file() else None, "note": note})

    add("planner entry (consumer)", "our_method/skill_memory_v2/planner.py", "plan() -> compose() -> chains_for()/schedule()")
    add("memory class (operator selection, binding of ?z, _usable filter)", "our_method/skill_memory_v2/memory.py")
    add("runner: prompt, answer parsing, re-ask", "scripts/viki_eval_v2_intent_choice.py",
        "re-ask: REASK constant and single second turn at lines %s" % reask_lines)
    add("ordering rules + vocabulary + layer1 (admitted, ID/CG column)", "outputs/v3_memories/memory_all.json",
        "layer2 canonical sha %s (%d kept rules); layer3 canonical sha %s; layer1 %d operators"
        % (canon_sha(mem["layer2"]), sum(1 for r in mem["layer2"]["rules"] if r.get("kept")),
           canon_sha(mem["layer3"]), len(mem["layer1"]["operators"])))
    for fam in sorted((ROOT / "outputs/v3_memories").glob("memory_heldout_*.json")):
        m = json.loads(fam.read_text())
        add("admitted memory, OOD fold", str(fam.relative_to(ROOT)),
            "layer1 %d ops; layer2 kept %d" % (len(m["layer1"]["operators"]),
                                              sum(1 for r in m["layer2"]["rules"] if r.get("kept"))))
    add("ordering rule application (layer 2)", "our_method/skill_memory_v2/dependencies.py")
    add("vocabulary grounding (layer 3)", "our_method/skill_memory_v2/vocabulary.py")
    add("simulator wrapper, state_facts/object_properties", "our_method/skill_memory_v2/simulator.py")
    add("replay-based operator mining (reference-library builder)", "our_method/skill_memory_v2/induction.py")
    add("reference-library build entry", "our_method/skill_memory_v2/build.py")
    add("execution check: Workbench.run_operator / plan_with / ordering_ok", "scripts/viki_induction_tools.py")
    add("admission: works on >=2 holdout + marginal coverage/ordering gain", "scripts/viki_agentic_rung_abstraction.py")
    add("admission: dedup + measured support >=2 over 60 probe episodes", "scripts/viki_assemble_agentic_library.py")
    add("union of family libraries into a column memory", "scripts/viki_union_library.py")
    add("column build driver", "scripts/drivers/viki_v2_evaluate.py")
    add("round-2 validation episodes per family (holdout, coverage pool)", "outputs/v3/targets.json")
    add("19-operator mined reference library (not an admitted library)", "results/viki_memory_experiments/amendment11/skill_memory_v2.json")

    # ---- families x effects, induction half only
    import pandas as pd
    from habitat_llm.evaluation import viki_bench as bench
    from our_method.skill_memory_v2.induction import requirements_of
    frame = pd.read_parquet(TRAIN)
    half = [bench.get_ground_truth(bench.to_native(frame.iloc[i].to_dict())) for i in range(0, len(frame), 2)]
    fam_n, eff = Counter(), defaultdict(Counter)
    strip = lambda s: re.sub(r"_\d+$", "", str(s))
    for truth in half:
        fam = truth.get("task_name", "?")
        fam_n[fam] += 1
        seen = set()
        for p in requirements_of(truth):
            st = p["status"]
            if "pos.name" in st:
                e = "pos.name(%s -> %s)" % (strip(p["name"]), strip(st["pos.name"]))
            elif st.get("is_activated") is True:
                e = "is_activated(%s)" % strip(p["name"])
            else:
                e = "other(%s)" % json.dumps(st, sort_keys=True)
            seen.add(e)
        for e in seen:
            eff[fam][e] += 1
    with (OUT / "a0_family_effects.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["family", "episodes_in_induction_half", "effect_key", "effect_instance", "episodes_with_effect"])
        for fam in sorted(fam_n):
            for e, c in sorted(eff[fam].items(), key=lambda kv: -kv[1]):
                w.writerow([fam, fam_n[fam], e.split("(")[0], e, c])
    effect_keys = {fam: sorted({e.split("(")[0] for e in eff[fam]}) for fam in fam_n}

    with (OUT / "a0_first_turn_cells.csv").open("w", newline="") as h:
        keys = sorted({k for c in cells for k in c})
        first = ["model", "split", "path", "rows", "sha256", "solved", "reask_rows"]
        w = csv.DictWriter(h, fieldnames=first + [k for k in keys if k not in first])
        w.writeheader()
        w.writerows(cells)
    with (OUT / "a0_artifacts.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["role", "path", "exists", "sha256", "note"])
        w.writeheader()
        w.writerows(artefacts)
    gate = {"four_split_files_x3": all(c.get("status") == "ok" for c in cells),
            "planner_exists": (ROOT / "our_method/skill_memory_v2/planner.py").is_file(),
            "schema_exists": bool(mem["layer1"]["operators"]),
            "fig3_first_turn_identical": all(c.get("fig3_first_turn_differ") == 0 and c.get("fig3_missing") == 0
                                             for c in cells),
            "problems": gate_problems}
    gate["PASS"] = all(v for k, v in gate.items() if k != "problems") and not gate_problems
    record = {"cells": cells, "artefacts": artefacts, "train_rows": len(frame),
              "induction_half": len(half), "families": dict(fam_n), "family_effect_keys": effect_keys,
              "gate": gate}
    (OUT / "a0_inventory.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(gate, indent=1))
    print("train rows %d, induction half %d, families %d" % (len(frame), len(half), len(fam_n)))
    for c in cells:
        print(c["model"], c["split"], c.get("rows"), c.get("solved"), c.get("fig3_first_turn_identical"),
              c.get("fig3_first_turn_differ"), c.get("raw_equals_source"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
