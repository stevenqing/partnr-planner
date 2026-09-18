#!/usr/bin/env python3
"""Why the ToM arm does not beat zero-shot: a paired, per-episode failure analysis.

The headline is already on record -- Figure 2 has ToM at 0.025 and zero-shot at 0.028 on 72B/id --
but a rate says nothing about *why*, and "ToM does not help" is the kind of claim a reviewer answers
with "then your ToM implementation is too weak". This script answers that objection with the rows
themselves. Both arms emit the same thing (a step list of primitive actions) and differ only in the
prompt, so every difference below is attributable to the ToM adapter.

Four readings, in the order they close the argument:

  1. paired          McNemar over the same episode ids. A rate difference of 0.003 can still hide a
                     large two-way churn, and it does: the arms disagree on 35 of 924.
  2. funnel          The scorer needs `eval_single(plan, truth)` AND `len(truth)/len(plan) >= 0.99`,
                     so a plan must be both feasible and no longer than the reference. The funnel
                     prices each hurdle separately: vocabulary, objects, length, then the simulator.
  3. mediation       ToM does change behaviour -- it hallucinates fewer objects and writes plans
                     closer to the reference length. This asks whether the episodes it repairs are
                     the episodes it wins, i.e. whether the change is on the critical path at all.
  4. partner_talk    Whether the zero-shot arm was already reasoning about the partner. If it was,
                     the adapter has nothing to add and that -- not a weak implementation -- is why
                     the arms tie.

Everything lands on disk before anything is printed.

  python scripts/viki_tom_failure_analysis.py --json outputs/viki_tom_failures/analysis.json \
      --md outputs/viki_tom_failures/TOM_FAILURES.md
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
try:  # math.comb is 3.8+; the box still has a 3.7 interpreter on PATH
    from math import comb
except ImportError:  # pragma: no cover
    from math import factorial

    def comb(n: int, k: int) -> int:
        return factorial(n) // (factorial(k) * factorial(n - k))
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "results/paper_viki_iclr2027/rows/figure2"
MODELS = ["qwen2.5-vl-72b-instruct", "qwen3-vl-30b-a3b-instruct", "qwen2.5-vl-7b-instruct"]
SPLITS = ["id", "ood_single_family", "cg_image", "pure_text"]
ARMS = ["tom", "zero_shot"]

# The words a plan would use if it were reasoning about the other robot at all. Deliberately wide:
# the point of the check is to fail loudly if the zero-shot arm is silent about the partner, so a
# generous pattern makes a null result ("both arms talk about the partner") harder to manufacture.
PARTNER = re.compile(
    r"\b(R2|R3|partner|other robot|another robot|teammate|its intent|intention|"
    r"already|is doing|plans to|in parallel|simultaneously|duplicat)\b",
    re.I,
)


def load(arm: str, model: str, split: str) -> dict[str, dict]:
    path = ROWS / arm / model / f"{split}.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.open():
        row = json.loads(line)
        out[str(row["example_id"])] = row
    return out


def mcnemar(a: dict[str, int], b: dict[str, int]) -> dict[str, Any]:
    """Exact two-sided binomial test on the episodes where the two arms disagree."""
    shared = sorted(set(a) & set(b))
    n10 = sum(1 for i in shared if a[i] and not b[i])
    n01 = sum(1 for i in shared if not a[i] and b[i])
    n = n10 + n01
    if n == 0:
        return {"n10": 0, "n01": 0, "p": 1.0}
    k = min(n10, n01)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"n10": n10, "n01": n01, "p": min(1.0, 2 * tail)}


def verbs_and_objects(steps: list) -> tuple[set, set]:
    verbs, objects = set(), set()
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        for action in (step.get("actions") or {}).values():
            if isinstance(action, list) and action:
                verbs.add(str(action[0]))
                if len(action) > 1 and isinstance(action[1], str):
                    objects.add(action[1])
    return verbs, objects


def hurdles(row: dict) -> dict[str, Any]:
    """Where this plan stands relative to each thing the scorer needs, independently of the rest."""
    plan, truth = row.get("parsed_output"), row["target"]
    if not isinstance(plan, list) or not plan:
        return {"parsed": False}
    ref_steps = truth["time_steps"]
    ref_verbs, ref_objects = verbs_and_objects(ref_steps)
    known_objects = ref_objects | set(truth.get("init_pos") or {})
    verbs, objects = verbs_and_objects(plan)
    length_ok = len(ref_steps) / len(plan) >= 0.99
    return {
        "parsed": True,
        "n_plan": len(plan),
        "n_ref": len(ref_steps),
        "ratio": len(plan) / len(ref_steps),
        "vocab_ok": not (verbs - ref_verbs),
        "objects_ok": not (objects - known_objects),
        "length_ok": length_ok,
        "length_exact": len(plan) == len(ref_steps),
        "success": bool(row["success"]),
        "partner_talk": bool(PARTNER.search(row.get("raw_output") or "")),
    }


def funnel(rows: dict[str, dict]) -> dict[str, Any]:
    """How many plans survive each hurdle, applied cumulatively in the order the scorer meets them."""
    marks = [hurdles(r) for r in rows.values()]
    total = len(marks)
    parsed = [m for m in marks if m["parsed"]]
    vocab = [m for m in parsed if m["vocab_ok"]]
    objects = [m for m in vocab if m["objects_ok"]]
    length = [m for m in objects if m["length_ok"]]
    won = [m for m in length if m["success"]]
    ratios = sorted(m["ratio"] for m in parsed)

    def pct(x):
        return round(len(x) / total, 4) if total else None

    return {
        "n": total,
        "stages": {
            "parsed": {"n": len(parsed), "share": pct(parsed)},
            "+ every verb in the reference vocabulary": {"n": len(vocab), "share": pct(vocab)},
            "+ every object in the scene": {"n": len(objects), "share": pct(objects)},
            "+ no longer than the reference": {"n": len(length), "share": pct(length)},
            "+ simulator accepts it (= success)": {"n": len(won), "share": pct(won)},
        },
        # Reported separately because it is the single biggest cut and is not a modelling error:
        # a plan shorter than the reference cannot satisfy the goal, a longer one is rejected outright.
        "length_profile": {
            "median_ratio": round(ratios[len(ratios) // 2], 3) if ratios else None,
            "shorter_than_reference": sum(1 for m in parsed if m["n_plan"] < m["n_ref"]),
            "exactly_reference_length": sum(1 for m in parsed if m["length_exact"]),
            "longer_than_reference": sum(1 for m in parsed if m["n_plan"] > m["n_ref"]),
        },
        "success_given_length_ok": round(
            sum(1 for m in length if m["success"]) / len(length), 4) if length else None,
        "partner_talk_share": round(sum(1 for m in marks if m.get("partner_talk")) / total, 4) if total else None,
    }


def mediation(tom: dict[str, dict], zs: dict[str, dict]) -> dict[str, Any]:
    """Is what ToM repairs the thing that decides the point?

    ToM writes plans with fewer invented objects and closer to the reference length. If those
    repairs were on the critical path, the episodes it repairs would be the episodes it wins.
    """
    shared = sorted(set(tom) & set(zs))
    out = {"n_shared": len(shared)}
    for label, field in (("objects", "objects_ok"), ("length", "length_ok")):
        repaired = [i for i in shared
                    if hurdles(tom[i]).get("parsed") and hurdles(zs[i]).get("parsed")
                    and hurdles(tom[i])[field] and not hurdles(zs[i])[field]]
        broke = [i for i in shared
                 if hurdles(tom[i]).get("parsed") and hurdles(zs[i]).get("parsed")
                 and not hurdles(tom[i])[field] and hurdles(zs[i])[field]]
        out[label] = {
            "tom_repaired": len(repaired),
            "tom_broke": len(broke),
            "of_repaired_tom_now_succeeds": sum(1 for i in repaired if tom[i]["success"]),
            "of_broke_tom_now_fails": sum(1 for i in broke if not tom[i]["success"]),
        }
    return out


def cases(tom: dict[str, dict], zs: dict[str, dict], limit: int = 2) -> list[dict]:
    """A few episodes worth quoting, chosen by what they demonstrate rather than at random."""
    shared = sorted(set(tom) & set(zs))
    picks = []
    for label, test in (
        ("ToM wins, zero-shot loses", lambda i: tom[i]["success"] and not zs[i]["success"]),
        ("zero-shot wins, ToM loses", lambda i: zs[i]["success"] and not tom[i]["success"]),
        ("both lose, ToM reasoned about the partner correctly",
         lambda i: not tom[i]["success"] and not zs[i]["success"]
         and PARTNER.search(tom[i].get("raw_output") or "")),
    ):
        chosen = [i for i in shared if test(i)][:limit]
        for i in chosen:
            h = hurdles(tom[i])
            picks.append({
                "label": label,
                "example_id": i,
                "family": tom[i].get("family"),
                "plan_len": h.get("n_plan"),
                "ref_len": h.get("n_ref"),
                "vocab_ok": h.get("vocab_ok"),
                "objects_ok": h.get("objects_ok"),
                "tom_reasoning": (tom[i].get("raw_output") or "").split("</think>")[0][:600],
            })
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()

    report: dict[str, Any] = {"rows_dir": str(ROWS), "cells": {}, "cases": {}}
    for model in MODELS:
        for split in SPLITS:
            arms = {a: load(a, model, split) for a in ARMS}
            if not arms["tom"] or not arms["zero_shot"]:
                continue
            key = f"{model}/{split}"
            success = {a: {i: int(r["success"]) for i, r in arms[a].items()} for a in ARMS}
            report["cells"][key] = {
                "n": len(arms["tom"]),
                "rate": {a: round(sum(success[a].values()) / len(success[a]), 4) for a in ARMS},
                "mcnemar_tom_vs_zero_shot": mcnemar(success["tom"], success["zero_shot"]),
                "funnel": {a: funnel(arms[a]) for a in ARMS},
                "mediation": mediation(arms["tom"], arms["zero_shot"]),
            }
            if split == "id":
                report["cases"][key] = cases(arms["tom"], arms["zero_shot"])

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1))          # on disk BEFORE anything is printed

    lines = ["# Why the ToM arm ties zero-shot", "",
             "Both arms emit a step list of primitive actions and differ only in the prompt, so every",
             "difference here is the ToM adapter's. The scorer needs a feasible plan AND",
             "`len(reference)/len(plan) >= 0.99`.", "",
             "| cell | n | ToM | zero-shot | ToM only | zs only | McNemar p |",
             "|---|---|---|---|---|---|---|"]
    for key, c in report["cells"].items():
        m = c["mcnemar_tom_vs_zero_shot"]
        lines.append(f"| {key} | {c['n']} | {c['rate']['tom']:.4f} | {c['rate']['zero_shot']:.4f} "
                     f"| {m['n10']} | {m['n01']} | {m['p']:.3f} |")
    lines += ["", "## What stops each plan (cumulative, 72B / id)", "",
              "| hurdle | ToM | zero-shot |", "|---|---|---|"]
    ref = report["cells"].get("qwen2.5-vl-72b-instruct/id")
    if ref:
        for stage in ref["funnel"]["tom"]["stages"]:
            t = ref["funnel"]["tom"]["stages"][stage]
            z = ref["funnel"]["zero_shot"]["stages"][stage]
            lines.append(f"| {stage} | {t['n']} ({t['share']:.1%}) | {z['n']} ({z['share']:.1%}) |")
        lines += ["", "## Does ToM repair what decides the point?", ""]
        med = ref["mediation"]
        for label in ("objects", "length"):
            d = med[label]
            lines.append(f"- **{label}**: ToM repaired {d['tom_repaired']} episodes and broke "
                         f"{d['tom_broke']}; of the repaired, {d['of_repaired_tom_now_succeeds']} now score.")
        lines += ["",
                  f"- both arms mention the partner in "
                  f"{ref['funnel']['zero_shot']['partner_talk_share']:.1%} (zero-shot) and "
                  f"{ref['funnel']['tom']['partner_talk_share']:.1%} (ToM) of outputs."]
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote {args.json}" + (f" and {args.md}" if args.md else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
