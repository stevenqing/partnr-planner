#!/usr/bin/env python3
"""A3/A4: merge first-turn replay with the live re-ask rows, score, and write the tables.

Per cell: replay-mined final row = first-turn replay row, except rows whose re-ask fired, which
take the live runner's row (a2_reask/<model>/<tag>.jsonl). Ours = the published cell
(amendment11 v3 file, reason == SOLVED). Strict SOLVED throughout.
Selected operators of re-asked rows come from planning the runner's second answer with the same
memory (a2_replay.run_chunk on raw_reask); its solved flag is checked against the runner's.

Writes tables/replay_vs_ours.csv, tables/replay_gap_by_family.csv, tables/reask_rates.csv,
tables/cg_same_operator_sequence.csv, a3_scores.json. Disk before stdout.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT / "scripts/iclr_two_exp/A"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
BASE = ROOT / "results/iclr_two_exp_2026-09-22/A"
A11 = ROOT / "results/viki_memory_experiments/amendment11"
CELL = {"id": "v3_ours_%s_id", "ood": "v3_ours_%s_heldout", "cg_image": "v3_oursall_%s_imaged",
        "cg_text": "v3_oursall_%s_text"}
SPLIT_NAME = {"id": "ID", "ood": "single-family OOD", "cg_image": "CG w/ Image", "cg_text": "CG w/o Image"}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def short(sig):
    key, coord, body = json.loads(sig)
    if coord:
        return "%s:COORD[%d] %s" % (key, len(body), " | ".join(" ".join(a[0] + ("(" + ",".join(a[1:]) + ")") for a in r) for r in body))
    return "%s:%s" % (key, " ".join(a[0] + "(" + ",".join(a[1:]) + ")" for a in body))


def main() -> int:
    import a2_replay
    (BASE / "tables").mkdir(exist_ok=True)
    rows_out, fam_rows, reask_rows, cg_rows, complete = [], [], [], [], True
    detail = {}
    for model in ("72B", "30B", "7B"):
        runner_rows = {}
        for path in sorted((BASE / "a2_reask" / model).glob("*.jsonl")):
            if path.name.endswith(".partial.jsonl") or path.name.endswith(".replay.jsonl"):
                continue
            parts = path.stem.split("_")
            split = parts[1] if parts[1] != "cg" else "cg_" + parts[2]
            for l in path.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    if r.get("reason") != "NO_ARCHIVED_ANSWER":
                        runner_rows[(split, int(r["index"]))] = r
        for split in CELL:
            ours = {int(json.loads(l)["index"]): json.loads(l)
                    for l in (A11 / ((CELL[split] % model) + ".jsonl")).read_text().splitlines() if l.strip()}
            first = json.loads((BASE / "a2_first" / ("%s_%s_replay_first.json" % (model, split))).read_text())
            final, missing, mismatch = {}, [], 0
            second_in = []
            for r in first["rows"]:
                row = dict(r)
                if r.get("reask_would_fire"):
                    live = runner_rows.get((split, r["index"]))
                    if live is None or not live.get("reask"):
                        missing.append(r["index"])
                        row.update(reask_done=False)
                    else:
                        row.update(reask_done=True, solved=int(live.get("reason") == "SOLVED"),
                                   reason=live.get("reason"), raw_reask=live.get("raw_reask"),
                                   reask_error=live.get("reask_error"))
                        if live.get("raw_reask"):
                            second_in.append({"index": r["index"], "task_name": r["task_name"],
                                              "raw": live["raw_reask"]})
                final[r["index"]] = row
            if second_in:
                _, _, _, second = a2_replay.run_chunk(model, split, "replay_first", second_in)
                for s in second:
                    f = final[s["index"]]
                    f["selected"] = s.get("selected")
                    if s.get("solved") != f["solved"]:
                        mismatch += 1
            if missing:
                complete = False
            n = len(final)
            o = {i: int(ours[i].get("reason") == "SOLVED") for i in ours}
            p = {i: int(final[i]["solved"]) for i in final}
            b = sum(1 for i in final if o[i] and not p[i])
            c = sum(1 for i in final if p[i] and not o[i])
            rows_out.append({"model": model, "split": SPLIT_NAME[split], "n": n, "ours": sum(o.values()),
                             "replay": sum(p.values()), "ours_only": b, "replay_only": c,
                             "p": "%.3g" % mcnemar(b, c)})
            reask_rows.append({"model": model, "split": SPLIT_NAME[split], "n": n,
                               "replay_reask_fired": sum(1 for r in final.values() if r.get("reask_would_fire")),
                               "replay_reask_done": sum(1 for r in final.values() if r.get("reask_done")),
                               "replay_reask_rate": "%.4f" % (sum(1 for r in final.values() if r.get("reask_would_fire")) / n),
                               "ours_reask": sum(1 for r in ours.values() if r.get("reask")),
                               "ours_reask_rate": "%.4f" % (sum(1 for r in ours.values() if r.get("reask")) / n),
                               "reask_rows_missing": len(missing), "second_answer_replan_mismatch": mismatch})
            if split in ("id", "ood"):
                by = defaultdict(lambda: [0, 0, 0, 0, Counter()])
                for i, r in final.items():
                    f = by[r["task_name"]]
                    f[0] += 1
                    f[1] += o[i]
                    f[2] += p[i]
                    if p[i] and not o[i]:
                        f[3] += 1
                        for chain in r.get("selected") or []:
                            f[4][" / ".join(short(s) for s in chain) or "?"] += 1
                for fam, (nn, oo, pp, only, ops) in sorted(by.items()):
                    ours_only = sum(1 for i, r in final.items() if r["task_name"] == fam and o[i] and not p[i])
                    fam_rows.append({"model": model, "split": SPLIT_NAME[split], "family": fam, "n": nn,
                                     "ours": oo, "replay": pp, "replay_only": only, "ours_only": ours_only,
                                     "operators_used_on_replay_only_rows": "; ".join(
                                         "%s x%d" % (k, v) for k, v in ops.most_common())})
            else:
                of = json.loads((BASE / "a2_first" / ("%s_%s_ours_first.json" % (model, split))).read_text())
                osel = {r["index"]: r.get("selected") for r in of["rows"]}
                same = sum(1 for i, r in final.items()
                           if r.get("selected") is not None and osel.get(i) is not None
                           and sorted(map(tuple, map(sorted, r["selected"]))) == sorted(map(tuple, map(sorted, osel[i]))))
                both_planned = sum(1 for i, r in final.items() if r.get("selected") is not None and osel.get(i) is not None)
                ops = Counter()
                for i, r in final.items():
                    if p[i] and not o[i]:
                        for chain in r.get("selected") or []:
                            ops[" / ".join(short(s) for s in chain)] += 1
                cg_rows.append({"model": model, "split": SPLIT_NAME[split], "n": n,
                                "rows_both_arms_planned": both_planned, "same_operator_sequence": same,
                                "replay_only": c, "ours_only": b,
                                "operators_used_on_replay_only_rows": "; ".join("%s x%d" % kv for kv in ops.most_common())})
                fam_rows.append({"model": model, "split": SPLIT_NAME[split], "family": "recombine_cut_and_deliver",
                                 "n": n, "ours": sum(o.values()), "replay": sum(p.values()), "replay_only": c,
                                 "ours_only": b, "operators_used_on_replay_only_rows": cg_rows[-1][
                                     "operators_used_on_replay_only_rows"]})
            detail["%s/%s" % (model, split)] = {"missing_reask_rows": missing, "rows": list(final.values())}

    def write(name, rows):
        with (BASE / "tables" / name).open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    write("replay_vs_ours.csv", rows_out)
    write("replay_gap_by_family.csv", fam_rows)
    write("reask_rates.csv", reask_rows)
    write("cg_same_operator_sequence.csv", cg_rows)
    (BASE / "a3_scores.json").write_text(json.dumps({"complete": complete, "cells": rows_out, "reask": reask_rows,
                                                    "cg": cg_rows, "detail": detail}))
    print("complete", complete)
    for r in rows_out:
        print(r)
    for r in reask_rows:
        print(r)
    for r in cg_rows:
        print({k: v for k, v in r.items() if k != "operators_used_on_replay_only_rows"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
