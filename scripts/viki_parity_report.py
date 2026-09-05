"""One table: how close is an agent-built library to the rule-based one, and what is missing.

Two things are read side by side and they answer different questions.

  reference ablations   sub-libraries of the SHIPPED library. These bound the problem: if
                        `ref_achievement` -- every achievement operator the reference
                        induced, and nothing else -- cannot reach 1.0, then the shape the
                        ladder can currently produce cannot reach parity no matter how well
                        a model writes it, and the gap is a workbench limitation rather than
                        a model one. That distinction decides what to build next.
  agentic libraries     what the ladder actually assembled, through the same gate.

Parity is `self_check.rate == 1.0`, not operator count. A library of 25 that solves 200/200
is parity; one of exactly 19 that solves 140 is not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")

ORDER = [
    ("reference_control", "reference, all 19, through the shim (control)"),
    ("ref_posname_only", "reference minus is_activated and unsealed"),
    ("ref_achievement_repair", "reference minus coordination"),
    ("ref_achievement", "reference, single-robot achievement only"),
    ("ref_achievement_posname", "reference, the ladder's exact shape today"),
    ("agentic_frozen", "agent-built, the 12 frozen-rung passes"),
    ("agentic_all", "agent-built, after the coverage sweep"),
    ("agentic_div", "agent-built, after residual-seeded diversity sweep"),
    ("agentic_marg", "agent-built, marginal-contribution acceptance"),
    ("agentic_iface", "agent-built, after the interface fix"),
    ("agentic_runner", "agent-built, after run_operator tries every robot"),
    ("agentic_fmt", "agent-built, after the submission-format fix"),
]


def load(name: str):
    path = ROOT / "outputs/parity" / f"bench_{name}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/viki_parity.json")
    arguments = parser.parse_args()

    rows, report = [], {}
    for name, why in ORDER:
        data = load(name)
        if data is None:
            rows.append({"name": name, "why": why, "missing": True})
            continue
        check = data.get("self_check") or {}
        row = {
            "name": name, "why": why, "missing": False,
            "operators": data.get("operators"),
            "by_kind": data.get("by_kind"),
            "effect_keys": data.get("effect_keys_offered"),
            "solved": check.get("solved"), "episodes": check.get("episodes"),
            "rate": check.get("rate"),
            "outcomes": check.get("outcomes"),
            "contract_violations": data.get("contract_violations"),
            "support": data.get("support"),
        }
        rows.append(row)
        report[name] = row

    control = report.get("reference_control", {})
    # The best agent-built library on disk, not a hardcoded preference order -- the first
    # draft named `agentic_all` and kept reporting it after `agentic_div` had overtaken it.
    agentic = max(
        (row for name, row in report.items()
         if name.startswith("agentic") and row.get("rate") is not None),
        key=lambda row: row["rate"], default={})
    ceiling = report.get("ref_achievement", {})
    payload = {
        "bar": "self_check.rate == 1.0 (the reference solves 200/200)",
        "rows": rows,
        "parity_reached": bool(agentic.get("rate") == 1.0),
        "achievement_only_ceiling": ceiling.get("rate"),
        "control_reproduces_reference": bool(control.get("rate") == 1.0),
    }

    # Disk before stdout, always: a report that dies after the table and before the dump
    # leaves nothing behind, which has cost this project a full run once already.
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=1))

    print("bar: self_check rate 1.0 -- the reference solves 200/200 held-out episodes\n")
    print("%-26s %5s %7s %8s  %s" % ("library", "ops", "solved", "rate", "coverage"))
    for row in rows:
        if row.get("missing"):
            print("%-26s %5s %7s %8s  (not scored)" % (row["name"], "-", "-", "-"))
            continue
        print("%-26s %5s %7s %8s  %s"
              % (row["name"], row["operators"],
                 "%s/%s" % (row["solved"], row["episodes"]),
                 row["rate"], row["effect_keys"]))
    print()
    for row in rows:
        if not row.get("missing"):
            print("  %-26s %s" % (row["name"], row["why"]))

    print()
    if control and control.get("rate") is not None and control.get("rate") != 1.0:
        print("!! the control did not reproduce the reference's 1.0 through the shim.")
        print("   Nothing below is interpretable until that is explained -- the shim or the")
        print("   gate changed, not the method.")
    # What each missing capability is worth, read off the ablations rather than asserted.
    # The first draft of this text claimed parity needed "coordination and repair"; the
    # ablation says repair is worth exactly nothing, so the claim is now computed.
    solved = lambda name: (report.get(name) or {}).get("solved")
    full, achievement = solved("reference_control"), solved("ref_achievement")
    with_repair, posname_only = solved("ref_achievement_repair"), solved("ref_achievement_posname")
    worth = {}
    if achievement is not None and posname_only is not None:
        worth["is_activated"] = achievement - posname_only
    if with_repair is not None and achievement is not None:
        worth["repair (unsealed)"] = with_repair - achievement
    if full is not None and with_repair is not None:
        worth["coordination"] = full - with_repair
    if worth:
        print("what each capability is worth, in held-out episodes solved:")
        for name, value in sorted(worth.items(), key=lambda kv: -kv[1]):
            print("    %-20s %+d" % (name, value))
    if ceiling.get("rate") is not None:
        print("\nachievement-only ceiling: %s (%s/%s)."
              % (ceiling["rate"], achievement, ceiling.get("episodes")), end=" ")
        if ceiling["rate"] == 1.0:
            print("Single-robot achievement operators are sufficient\n"
                  "for parity, so the remaining gap is the model's and more sampling is the answer.")
        else:
            print("Single-robot achievement operators -- the only\n"
                  "shape the workbench can verify today -- reach this even when the shipped inducer\n"
                  "writes them. So that is the ceiling for the current rung, and the distance from\n"
                  "it to the agent-built library is a model/sampling gap; the distance from it to\n"
                  "1.0 is a workbench gap, and the table above says exactly which capability that\n"
                  "last stretch is (`run_operator` runs one robot, so coordination cannot be\n"
                  "verified at all). Build the rung that buys the most episodes, not the one with\n"
                  "the most operators.")
    if agentic:
        print("\nagent-built best: %s/%s (rate %s) with %s"
              % (agentic.get("solved"), agentic.get("episodes"), agentic.get("rate"),
                 agentic.get("effect_keys")))
    print("\n-> %s" % arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
