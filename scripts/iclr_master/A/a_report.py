#!/usr/bin/env python3
"""A1 / A2 / A4 tables of the 2026-09-22 master spec, from the replay JSONs and the archived cells.

Zero LLM calls. Writes answers.json and the per-row CSVs before printing anything.
"""
from __future__ import annotations

import csv
import json
from math import comb
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
A11 = ROOT / "results/viki_memory_experiments/amendment11"
OUT = ROOT / "results/iclr_master_2026-09-22/A"
REPLAY = OUT / "replay"
MODELS = ["72B", "30B", "7B"]
SPLITS = ["id", "heldout", "heldout_sibgrp", "imaged", "text"]
LABEL = {"id": "ID", "heldout": "OOD (single family)", "heldout_sibgrp": "OOD (sibling group)",
         "imaged": "CG w/ Image", "text": "CG w/o Image"}
CELL = {"id": "v3_ours_%s_id", "heldout": "v3_ours_%s_heldout",
        "heldout_sibgrp": "v3_ours_%s_heldout_sibgrp",
        "text": "v3_oursall_%s_text", "imaged": "v3_oursall_%s_imaged"}
CURVE = [("comp(cut)", "v3_ourscut_%s_%s", "outputs/v3_memories/memory_comp_cut.json"),
         ("comp(cut+delivery)", "v3_ours_%s_%s", "outputs/v3_memories/memory_comp_cut_delivery.json"),
         ("comp(all)", "v3_oursall_%s_%s", "outputs/v3_memories/memory_all.json")]


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def archive(tag):
    path = A11 / (tag + ".jsonl")
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["index"])] = r
    return rows, path


def replay(model, split, mode):
    path = REPLAY / ("replay_%s_%s_%s.json" % (model, split, mode))
    if not path.is_file():
        return None, path
    d = json.loads(path.read_text())
    if d.get("status") != "ok":
        return d, path
    return {int(r["index"]): r for r in d["rows"]}, path


def mcnemar(a, b):
    shared = sorted(set(a) & set(b))
    wins = sum(1 for i in shared if a[i] and not b[i])
    losses = sum(1 for i in shared if b[i] and not a[i])
    n = wins + losses
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n)
    return len(shared), wins, losses, p


def num(value, numerator, denominator, source, how):
    return {"value": value, "numerator": numerator, "denominator": denominator,
            "source_path": source, "computation": how}


def main() -> int:
    answers = {"status": "complete", "ending": None, "numbers": {}, "notes": []}
    N = answers["numbers"]
    a1_rows, a2_rows, a4_rows = [], [], []
    a1_table, a2_table, a4_table = [], [], []
    alarms = []

    # ---------------- A1
    for m in MODELS:
        for s in SPLITS:
            arch, apath = archive(CELL[s] % m)
            full, fpath = replay(m, s, "ours_full")
            first, rpath = replay(m, s, "ours_first")
            pub = sum(r.get("reason") == "SOLVED" for r in arch.values())
            n = len(arch)
            reask = sum(bool(r.get("reask")) for r in arch.values())
            # first-turn-only from the archive alone: the live runner re-asks only when the first
            # answer produced no plan, so a re-asked row is unsolved on its first answer.
            first_arch = {i: int(r.get("reason") == "SOLVED" and not r.get("reask")) for i, r in arch.items()}
            full_arch = {i: int(r.get("reason") == "SOLVED") for i, r in arch.items()}
            rep_full = sum(r["solved"] for r in full.values())
            rep_first = {i: r["solved"] for i, r in first.items()}
            mismatch = sum(1 for i in arch if full[i]["solved"] != full_arch[i])
            if mismatch:
                alarms.append("A1 %s %s: ours_full replay differs from the archive on %d rows" % (m, s, mismatch))
            fmis = sum(1 for i in arch if rep_first[i] != first_arch[i])
            if fmis:
                alarms.append("A1 %s %s: first-turn replay differs from archive-derived first-turn on %d rows" % (m, s, fmis))
            gained = sum(1 for i in arch if full_arch[i] and not rep_first[i])
            lost = sum(1 for i in arch if rep_first[i] and not full_arch[i])
            nf = sum(rep_first.values())
            a1_table.append((m, s, n, reask, pub, rep_full, nf, gained, lost))
            key = "A1.%s.%s" % (m, s)
            N[key + ".reask_rows"] = num(round(reask / n, 4), reask, n, rel(apath), "rows with reask=true")
            N[key + ".full"] = num(round(pub / n, 4), pub, n, rel(apath), "reason == SOLVED")
            N[key + ".first_turn_only"] = num(round(nf / n, 4), nf, n, rel(rpath),
                                              "replay of `raw` only against the same library, SOLVED")
            N[key + ".gained_by_reask"] = num(gained, gained, n, rel(rpath), "full solved and first-turn not")
            N[key + ".lost_by_reask"] = num(lost, lost, n, rel(rpath), "first-turn solved and full not")
            for i in sorted(arch):
                a1_rows.append({"model": m, "split": s, "index": i, "task_name": arch[i].get("task_name"),
                                "reask": int(bool(arch[i].get("reask"))), "full_solved": full_arch[i],
                                "full_replay_solved": full[i]["solved"], "first_turn_solved": rep_first[i],
                                "archived_reason": arch[i].get("reason"), "first_turn_reason": first[i]["reason"]})

    # ---------------- A2
    for m in MODELS:
        for s in SPLITS:
            first, _ = replay(m, s, "ours_first")
            ref, rpath = replay(m, s, "ref_first")
            arch, apath = archive(CELL[s] % m)
            full = {i: int(r.get("reason") == "SOLVED") for i, r in arch.items()}
            if not isinstance(ref, dict) or (ref.get("status") if "status" in ref else "ok") != "ok":
                a2_table.append((m, s, "NOT_FOUND"))
                N["A2.%s.%s.reference_first_turn" % (m, s)] = "NOT_FOUND"
                continue
            refd = {i: r["solved"] for i, r in ref.items()}
            ours = {i: r["solved"] for i, r in first.items()}
            n = len(refd)
            nr, no, nfull = sum(refd.values()), sum(ours.values()), sum(full.values())
            _, w, l, p = mcnemar(ours, refd)
            _, w2, l2, p2 = mcnemar(full, refd)
            a2_table.append((m, s, n, no, nfull, nr, w, l, p, w2, l2, p2))
            key = "A2.%s.%s" % (m, s)
            N[key + ".reference_first_turn"] = num(round(nr / n, 4), nr, n, rel(rpath),
                                                   "19-op reference library, first answer only, SOLVED")
            N[key + ".ours_first_turn"] = num(round(no / n, 4), no, n, rel(rpath).replace("ref_first", "ours_first"),
                                              "v3 library, first answer only, SOLVED")
            N[key + ".mcnemar_ours_first_vs_ref"] = {"ours_only": w, "ref_only": l, "p_exact": p}
            N[key + ".mcnemar_ours_full_vs_ref"] = {"ours_only": w2, "ref_only": l2, "p_exact": p2}
            libs = sorted({r["library"] for r in ref.values()})
            for i in sorted(refd):
                a2_rows.append({"model": m, "split": s, "index": i, "task_name": arch[i].get("task_name"),
                                "ours_first_turn": ours[i], "ours_full": full[i], "reference_first_turn": refd[i],
                                "reference_reason": ref[i]["reason"], "reference_library": ref[i]["library"]})

    # on-disk reference cells written earlier, for cross-checking the replay (not the protocol here)
    disk = {}
    for tag in ("appendix_reference_72B_id", "appendix_reference_72B_imaged", "appendix_reference_72B_text"):
        arch, p = archive(tag)
        solved = sum(r.get("reason") == "SOLVED" for r in arch.values())
        solved_first = sum(r.get("reason") == "SOLVED" and not r.get("reask") for r in arch.values())
        disk[tag] = num(round(solved / len(arch), 4), solved, len(arch), rel(p), "reason == SOLVED (live re-ask included)")
        disk[tag + ".first_turn"] = num(round(solved_first / len(arch), 4), solved_first, len(arch), rel(p),
                                        "SOLVED and reask false")
    for m in MODELS:
        for name in ("v3_lib_id_%s.json", "v3_libfold_fold_%s.json"):
            p = ROOT / "outputs/viki_ablation" / (name % m)
            if p.is_file():
                d = json.loads(p.read_text())
                for arm, a in d["arms"].items():
                    if arm.startswith("lib_"):
                        disk["%s.%s" % (p.stem, arm)] = num(a["rate"], a["solved"], a["n"], rel(p),
                                                            "archived replay (re-ask included)")
    N["A2.disk_reference_cells"] = disk

    # ---------------- A4
    for m in MODELS:
        for s in ("text", "imaged"):
            base = None
            for label, pat, mem in CURVE:
                arch, p = archive(pat % (m, s))
                solved = {i: int(r.get("reason") == "SOLVED") for i, r in arch.items()}
                k = sum(solved.values())
                ops = len(json.loads((ROOT / mem).read_text())["layer1"]["operators"])
                fams = json.loads((ROOT / mem).read_text()).get("union_families")
                row = [m, s, label, len(ops) if isinstance(ops, list) else ops, k, len(arch), rel(p)]
                if base is not None:
                    _, w, l, pv = mcnemar(solved, base)
                    row += [w, l, pv]
                else:
                    row += ["", "", ""]
                base = solved          # each point is paired against the previous one
                a4_table.append(row)
                N["A4.%s.%s.%s" % (m, s, label)] = num(round(k / len(arch), 4), k, len(arch), rel(p), "reason == SOLVED")
                N["A4.%s.%s.%s.library" % (m, s, label)] = {"path": mem, "operators": ops, "families": fams}
                for i in sorted(arch):
                    a4_rows.append({"model": m, "split": s, "curve": label, "index": i,
                                    "task_name": arch[i].get("task_name"), "solved": solved[i]})

    answers["notes"] = alarms
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "answers.json").write_text(json.dumps(answers, indent=1))
    for name, rows in (("a1_per_row.csv", a1_rows), ("a2_per_row.csv", a2_rows), ("a4_per_row.csv", a4_rows)):
        if rows:
            with (OUT / name).open("w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    (OUT / "tables.json").write_text(json.dumps({"a1": a1_table, "a2": a2_table, "a4": a4_table}, indent=0))

    print("ALARMS", alarms)
    print("\nA1  model split n reask published replay_full first_only gained lost")
    for r in a1_table:
        print(*r)
    print("\nA2  model split n ours_first ours_full ref_first ours>ref ref>ours p | full: ours>ref ref>ours p")
    for r in a2_table:
        print(*r)
    print("\nA4")
    for r in a4_table:
        print(*r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
