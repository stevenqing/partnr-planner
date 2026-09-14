#!/usr/bin/env python3
"""RQ3 equivalence evidence: row-by-row agreement between ablation runs on the same rows.

Rows are replays, so everything up to the planner is deterministic; the only live step is
the per-row re-ask. Agreement is therefore reported separately for rows re-asked in
neither run (should be exact) and rows re-asked in at least one (where nondeterminism of
the live re-ask can show up, measured as raw_reask text identity).

Writes results/paper_viki_iclr2027/raw/rq3/equivalence/7B/agreement.json, then prints.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A11 = ROOT / "results/viki_memory_experiments/amendment11"
RAW = ROOT / "results/paper_viki_iclr2027/raw/rq3"
EQ = RAW / "equivalence/7B"


def cell(kind_dir: Path, tag: str) -> Path:
    return kind_dir / ("%s.jsonl" % tag)


PAIRS = [
    # (name, expectation, a, b)
    ("flag+full_lib vs archived v3_ablfull (no_grounding, id)", "equivalent up to re-ask",
     cell(EQ / "flag_no_grounding__full_lib", "rq3eq_flag_no_grounding__full_lib_7B_id"),
     A11 / "v3_ablfull_noground_7B_id.jsonl"),
    ("flag+full_lib vs archived v3_ablfull (no_order, id)", "equivalent up to re-ask",
     cell(EQ / "flag_no_order__full_lib", "rq3eq_flag_no_order__full_lib_7B_id"),
     A11 / "v3_ablfull_noorder_7B_id.jsonl"),
    ("transformed no_grounding lib WITHOUT flag vs flag+full_lib (id)", "NOT expected equivalent",
     cell(EQ / "noflag__no_grounding_lib", "rq3eq_noflag__no_grounding_lib_7B_id"),
     cell(EQ / "flag_no_grounding__full_lib", "rq3eq_flag_no_grounding__full_lib_7B_id")),
    ("transformed no_order lib WITHOUT flag vs flag+full_lib (id)", "equivalent up to re-ask",
     cell(EQ / "noflag__no_order_lib", "rq3eq_noflag__no_order_lib_7B_id"),
     cell(EQ / "flag_no_order__full_lib", "rq3eq_flag_no_order__full_lib_7B_id")),
    ("driver cell (transformed lib + flag) vs flag+full_lib (no_grounding, id)", "equivalent up to re-ask",
     cell(RAW / "7B/no_grounding/id", "rq3_no_grounding_7B_id"),
     cell(EQ / "flag_no_grounding__full_lib", "rq3eq_flag_no_grounding__full_lib_7B_id")),
    ("driver cell (transformed lib + flag) vs flag+full_lib (no_order, id)", "equivalent up to re-ask",
     cell(RAW / "7B/no_order/id", "rq3_no_order_7B_id"),
     cell(EQ / "flag_no_order__full_lib", "rq3eq_flag_no_order__full_lib_7B_id")),
    ("driver cell vs archived v3_ablfull (no_grounding, cg_image)", "equivalent up to re-ask",
     cell(RAW / "7B/no_grounding/cg_image", "rq3_no_grounding_7B_cg_image"),
     A11 / "v3_ablfull_noground_7B_imaged.jsonl"),
    ("driver cell vs archived v3_ablfull (no_grounding, pure_text)", "equivalent up to re-ask",
     cell(RAW / "7B/no_grounding/pure_text", "rq3_no_grounding_7B_pure_text"),
     A11 / "v3_ablfull_noground_7B_text.jsonl"),
    ("driver cell vs archived v3_ablfull (no_order, cg_image)", "equivalent up to re-ask",
     cell(RAW / "7B/no_order/cg_image", "rq3_no_order_7B_cg_image"),
     A11 / "v3_ablfull_noorder_7B_imaged.jsonl"),
    ("driver cell vs archived v3_ablfull (no_order, pure_text)", "equivalent up to re-ask",
     cell(RAW / "7B/no_order/pure_text", "rq3_no_order_7B_pure_text"),
     A11 / "v3_ablfull_noorder_7B_text.jsonl"),
]


def load(path: Path):
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["index"])] = r
    return rows


def compare(a, b):
    common = sorted(set(a) & set(b))
    out = {"n_a": len(a), "n_b": len(b), "n_common": len(common),
           "success_a": sum(a[i].get("reason") == "SOLVED" for i in common),
           "success_b": sum(b[i].get("reason") == "SOLVED" for i in common)}
    both = a_only = b_only = neither = 0
    reason_same = 0
    groups = {"reasked_in_neither": [0, 0], "reasked_in_either": [0, 0]}
    reask_text_same = reask_text_total = 0
    disagree = []
    for i in common:
        sa, sb = a[i].get("reason") == "SOLVED", b[i].get("reason") == "SOLVED"
        both += sa and sb
        a_only += sa and not sb
        b_only += sb and not sa
        neither += not sa and not sb
        reason_same += a[i].get("reason") == b[i].get("reason")
        g = "reasked_in_either" if (a[i].get("reask") or b[i].get("reask")) else "reasked_in_neither"
        groups[g][0] += sa == sb
        groups[g][1] += 1
        if a[i].get("reask") and b[i].get("reask"):
            reask_text_total += 1
            reask_text_same += a[i].get("raw_reask") == b[i].get("raw_reask")
        if sa != sb or a[i].get("reason") != b[i].get("reason"):
            disagree.append({"index": i, "task_name": a[i].get("task_name"),
                             "reason_a": a[i].get("reason"), "reason_b": b[i].get("reason"),
                             "reask_a": bool(a[i].get("reask")), "reask_b": bool(b[i].get("reask")),
                             "raw_reask_same": a[i].get("raw_reask") == b[i].get("raw_reask")})
    out.update({
        "success_agree": both + neither, "both_success": both, "a_only_success": a_only,
        "b_only_success": b_only, "both_failure": neither, "reason_agree": reason_same,
        "success_agree_reasked_in_neither": "%d/%d" % tuple(groups["reasked_in_neither"]),
        "success_agree_reasked_in_either": "%d/%d" % tuple(groups["reasked_in_either"]),
        "raw_reask_identical_when_both_reasked": "%d/%d" % (reask_text_same, reask_text_total),
        "disagreements": disagree,
    })
    return out


def main() -> int:
    report = {"pairs": []}
    for name, expectation, pa, pb in PAIRS:
        entry = {"name": name, "expectation": expectation, "a": str(pa), "b": str(pb)}
        if not (pa.is_file() and pb.is_file()):
            entry["missing"] = [str(p) for p in (pa, pb) if not p.is_file()]
        else:
            entry["a_sha256"] = hashlib.sha256(pa.read_bytes()).hexdigest()
            entry["b_sha256"] = hashlib.sha256(pb.read_bytes()).hexdigest()
            entry.update(compare(load(pa), load(pb)))
        report["pairs"].append(entry)
    EQ.mkdir(parents=True, exist_ok=True)
    (EQ / "agreement.json").write_text(json.dumps(report, indent=2) + "\n")
    for e in report["pairs"]:
        if "missing" in e:
            print("%s: MISSING %s" % (e["name"], e["missing"]))
            continue
        print("%s [%s]\n  n=%d/%d common=%d  solved a=%d b=%d  success-agree=%d (both=%d a_only=%d b_only=%d)"
              "  reason-agree=%d  no-reask rows %s  reask rows %s  reask text same %s" % (
                  e["name"], e["expectation"], e["n_a"], e["n_b"], e["n_common"], e["success_a"],
                  e["success_b"], e["success_agree"], e["both_success"], e["a_only_success"],
                  e["b_only_success"], e["reason_agree"], e["success_agree_reasked_in_neither"],
                  e["success_agree_reasked_in_either"], e["raw_reask_identical_when_both_reasked"]))
    print("wrote", EQ / "agreement.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
