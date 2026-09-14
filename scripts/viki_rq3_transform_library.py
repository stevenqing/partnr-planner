#!/usr/bin/env python3
"""RQ3: write the no_grounding / no_order libraries, with structural checks and SHA-256.

What the evaluator's ablation flags actually bypass (scripts/viki_eval_v2_intent_choice.py):

  --no-grounding   to_requirement() skips SkillMemoryV2.canonical_asset (vocabulary.canonical
                   against the SCENE's asset names -- runtime data, not library data) and
                   SkillMemoryV2.canonical_place (vocabulary.canonical against the library's
                   layer3.places | layer3.assets, falling back to vocabulary.clean). Under the
                   flag X is kept only on an exact scene-name match and Y is used as
                   `where.strip()`.
  --no-order       temporal_constraints = [] instead of SkillMemoryV2.order_for, whose only
                   library input is layer2.kept_patterns.

So the transforms are:

  no_grounding     clear layer3.assets and layer3.places (the only library fields the flag
                   bypasses). Layer 1 and Layer 2 are byte-for-byte the source's.
                   NOT equivalent to the flag as a pure data transform: with an empty
                   vocabulary canonical_place still applies vocabulary.clean (underscores and
                   hyphens -> spaces), and canonical_asset still does case / whole-word
                   matching against the scene. The runtime flag is authoritative; the saved
                   library documents which library fields the flag disables.
  no_order         clear layer2.kept_patterns (and mark those rules kept=false so the file is
                   self-consistent). With kept_patterns == [] order_for returns [] for every
                   requirement set, which is exactly what the flag emits, so the transform is
                   equivalent by construction. Serialization handed to the planner:
                   "temporal_constraints": [] (removed, not permuted; no seed involved).
                   Operator bodies (intra-skill action order) are untouched: the existing
                   repository ablation removes Layer-2 inter-requirement ordering only.

The removed values are kept in the report, not in the transformed library.

Run (remote):  /root/venvs/partnr/bin/python scripts/viki_rq3_transform_library.py
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from our_method.skill_memory_v2.memory import SkillMemoryV2  # noqa: E402

SOURCE = ROOT / "outputs/v3_memories"
OUT = ROOT / "results/paper_viki_iclr2027/libraries/rq3"
FOLD_FAMILIES = [
    "clear_table_with_two_robots_and_put_in_cabinet",
    "cut_fruit_on_board",
    "cut_two_fruits_on_board",
    "dog_push_box_for_two_panda_transport",
    "ensure_all_fruits_on_table",
    "parallel_human_dual_asset_to_plate_or_bowl",
    "set_plate_and_fork_on_table",
    "toast_bread_and_set_plate",
]
EXPECTED_FULL_SHA = "d580b15c90b539aee5cb9f761d460f4b8e57b36b84af6ca4b8379f59d8237c87"

GROUNDING_FIELDS = ["layer3.assets", "layer3.places"]
ORDER_FIELDS = ["layer2.kept_patterns", "layer2.rules[*].kept (true -> false)"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body_sequences(record):
    """Per operator, its action sequence(s) exactly as serialized."""
    out = []
    for operator in record["layer1"]["operators"]:
        if operator.get("coordinated"):
            out.append([[item["action"] for item in role["actions"]] for role in operator["roles"]])
        else:
            out.append(operator["body"])
    return out


def action_multisets(record):
    out = []
    for operator in record["layer1"]["operators"]:
        if operator.get("coordinated"):
            actions = [item["action"] for role in operator["roles"] for item in role["actions"]]
        else:
            actions = operator["body"]
        out.append(Counter(json.dumps(a) for a in actions))
    return out


def transform_no_grounding(record):
    new = copy.deepcopy(record)
    removed = {"layer3.assets": new["layer3"]["assets"], "layer3.places": new["layer3"]["places"]}
    new["layer3"]["assets"] = []
    new["layer3"]["places"] = []
    return new, removed


def transform_no_order(record):
    new = copy.deepcopy(record)
    removed = {"layer2.kept_patterns": new["layer2"]["kept_patterns"],
               "layer2.rules_kept_true_indices": [i for i, r in enumerate(new["layer2"]["rules"])
                                                  if r.get("kept")]}
    new["layer2"]["kept_patterns"] = []
    for rule in new["layer2"]["rules"]:
        rule["kept"] = False
    return new, removed


def order_probe(memory: SkillMemoryV2, source_patterns):
    """For every source kept pattern, a two-requirement set that triggers exactly it."""
    hits = []
    for text in source_patterns:
        p = json.loads(text)
        a_name, b_name = "obj_a", "obj_b"
        a_target = "place_a"
        if p["a_key"] == "pos":
            a = {"type": "asset", "name": a_name, "is_satisfied": True, "status": {"pos.name": a_target}}
        else:
            a = {"type": "asset", "name": a_name, "is_satisfied": True, "status": {"is_activated": True}}
            a_target = None
        if p["a_target_is_b_subject"] and a_target is not None:
            b_name = a_target
        b_target = "place_b"
        if p["a_subject_is_b_target"]:
            b_target = a_name
        if p["a_target_is_b_target"] and a_target is not None:
            b_target = a_target
        if p["b_key"] == "pos":
            b = {"type": "asset", "name": b_name, "is_satisfied": True, "status": {"pos.name": b_target}}
        else:
            b = {"type": "asset", "name": b_name, "is_satisfied": True, "status": {"is_activated": True}}
        visits = {0: set(), 1: ({a_target} if p["b_visits_a_target"] and a_target else set())}
        hits.append(len(memory.order_for([a, b], visits)))
    return hits


def check(condition, src_record, new_record, src_path, new_path):
    src_mem, new_mem = SkillMemoryV2.load(src_path), SkillMemoryV2.load(new_path)
    c = {}
    c["format_same"] = src_record["format"] == new_record["format"]
    c["skill_count_source"] = len(src_record["layer1"]["operators"])
    c["skill_count_transformed"] = len(new_record["layer1"]["operators"])
    c["loaded_skill_count_source"] = len(src_mem.operators)
    c["loaded_skill_count_transformed"] = len(new_mem.operators)
    c["skill_count_equal"] = (c["skill_count_source"] == c["skill_count_transformed"]
                              and c["loaded_skill_count_source"] == c["loaded_skill_count_transformed"])
    c["layer1_identical"] = src_record["layer1"] == new_record["layer1"]
    c["no_empty_action_body"] = all(len(b) > 0 for b in body_sequences(new_record))
    c["file_bytes_differ"] = sha(src_path) != sha(new_path)
    c["other_top_level_identical"] = all(src_record[k] == new_record[k] for k in src_record
                                         if k not in ("layer2", "layer3"))
    if condition == "no_grounding":
        c["action_order_identical_per_skill"] = body_sequences(src_record) == body_sequences(new_record)
        c["source_grounding_nonempty"] = bool(src_record["layer3"]["assets"]) and bool(src_record["layer3"]["places"])
        c["grounding_fields_cleared"] = (new_record["layer3"]["assets"] == []
                                         and new_record["layer3"]["places"] == [])
        c["layer2_identical"] = src_record["layer2"] == new_record["layer2"]
        c["layer3_other_fields_identical"] = all(
            src_record["layer3"][k] == new_record["layer3"][k]
            for k in src_record["layer3"] if k not in ("assets", "places"))
        probe = "kitchen island"
        c["probe_canonical_place"] = {"input": probe, "source": src_mem.canonical_place(probe),
                                      "transformed": new_mem.canonical_place(probe),
                                      "runtime_flag_would_use": probe.strip()}
        c["probe_grounding_disabled"] = (c["probe_canonical_place"]["source"] != probe
                                         and c["probe_canonical_place"]["transformed"] == probe)
        keys = ["format_same", "skill_count_equal", "layer1_identical", "no_empty_action_body",
                "file_bytes_differ", "other_top_level_identical", "action_order_identical_per_skill",
                "source_grounding_nonempty", "grounding_fields_cleared", "layer2_identical",
                "layer3_other_fields_identical", "probe_grounding_disabled"]
    else:
        c["action_multiset_identical_per_skill"] = action_multisets(src_record) == action_multisets(new_record)
        c["grounding_identical"] = src_record["layer3"] == new_record["layer3"]
        c["source_order_signal_nonempty"] = bool(src_record["layer2"]["kept_patterns"])
        c["order_signal_cleared"] = (new_record["layer2"]["kept_patterns"] == []
                                     and new_mem.rules == []
                                     and not any(r.get("kept") for r in new_record["layer2"]["rules"]))
        src_hits = order_probe(src_mem, src_record["layer2"]["kept_patterns"])
        new_hits = order_probe(new_mem, src_record["layer2"]["kept_patterns"])
        c["probe_order_constraints"] = {"source": src_hits, "transformed": new_hits}
        c["probe_order_disabled"] = all(h > 0 for h in src_hits) and all(h == 0 for h in new_hits)
        keys = ["format_same", "skill_count_equal", "layer1_identical", "no_empty_action_body",
                "file_bytes_differ", "other_top_level_identical",
                "action_multiset_identical_per_skill", "grounding_identical",
                "source_order_signal_nonempty", "order_signal_cleared", "probe_order_disabled"]
    c["all_pass"] = all(bool(c[k]) for k in keys)
    c["checked_keys"] = keys
    return c


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    names = ["memory_all.json"] + ["memory_heldout_%s.json" % f for f in FOLD_FAMILIES]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha(Path(__file__)),
        "source_dir": str(args.source),
        "authoritative_mechanism": {
            "no_grounding": {
                "mechanism": "runtime flag --no-grounding (scripts/viki_eval_v2_intent_choice.py)",
                "library_transform_equivalent": False,
                "why": ("the flag changes matching logic, not only data: canonical_asset matches "
                        "against the scene's asset names (not library data) and canonical_place "
                        "falls back to vocabulary.clean() even with an empty vocabulary, whereas "
                        "the flag keeps X only on an exact scene match and uses Y.strip()"),
                "cells_use": "transformed library + --no-grounding (the flag makes layer3 unread, "
                             "so the library bytes do not change the result)",
            },
            "no_order": {
                "mechanism": "runtime flag --no-order (equivalent library transform also saved)",
                "library_transform_equivalent": True,
                "why": ("order_for's only library input is layer2.kept_patterns; with it empty "
                        "order_for returns [] for every input, which is what the flag emits"),
                "cells_use": "transformed library + --no-order",
            },
        },
        "fields_removed": {"no_grounding": GROUNDING_FIELDS, "no_order": ORDER_FIELDS},
        "fields_retained_note": {
            "no_grounding": ("layer3.goal_targets / episodes / self_check are retained: the "
                             "intent-choice evaluator never reads them (goal_targets is used "
                             "only by SkillMemoryV2.render, a prompt path this arm does not use); "
                             "operator `types` (spare-variable property filters) are Layer 1 and "
                             "untouched by the flag"),
            "no_order": ("layer2.rules[*] mining statistics (ordered/seen/precision/families) and "
                         "the summary recall/false_orderings are retained as descriptive "
                         "metadata; nothing reads them at planning time. Operator bodies and "
                         "coordinated `after` links (intra-skill order) are untouched"),
        },
        "no_order_serialization": {"temporal_constraints": [], "permutation": None, "seed": None},
        "files": {},
    }
    ok = True
    for name in names:
        src_path = args.source / name
        src_record = json.loads(src_path.read_text())
        full_dir = args.out / "full"
        full_dir.mkdir(parents=True, exist_ok=True)
        full_copy = full_dir / name
        if not full_copy.is_file():
            shutil.copyfile(src_path, full_copy)
        entry = {"source": str(src_path), "source_sha256": sha(src_path),
                 "full_copy": str(full_copy), "full_copy_sha256": sha(full_copy),
                 "excluded_family": src_record.get("excluded_family")}
        entry["full_copy_identical"] = entry["full_copy_sha256"] == entry["source_sha256"]
        ok &= entry["full_copy_identical"]
        if name == "memory_all.json":
            entry["matches_expected_full_sha"] = entry["source_sha256"] == EXPECTED_FULL_SHA
            ok &= entry["matches_expected_full_sha"]
        else:
            fam = name[len("memory_heldout_"):-len(".json")]
            # Union libraries carry excluded_family = null; the fold is in union_families.
            fams = src_record.get("union_families") or []
            entry["held_out_family"] = fam
            entry["union_family_count"] = len(fams)
            entry["held_out_family_absent_from_union"] = fam not in fams and len(fams) == 13
            ok &= entry["held_out_family_absent_from_union"]
        for condition, fn in (("no_grounding", transform_no_grounding), ("no_order", transform_no_order)):
            new_record, removed = fn(src_record)
            out_dir = args.out / condition
            out_dir.mkdir(parents=True, exist_ok=True)
            new_path = out_dir / name
            new_path.write_text(json.dumps(new_record, indent=2) + "\n")
            checks = check(condition, src_record, new_record, src_path, new_path)
            ok &= checks["all_pass"]
            entry[condition] = {"path": str(new_path), "sha256": sha(new_path),
                                "removed_values": removed, "structural_checks": checks}
        report["files"][name] = entry
    report["all_checks_pass"] = bool(ok)
    report_path = args.out / "transform_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for name, entry in report["files"].items():
        print("%-72s full %s  ng %s %s  no %s %s" % (
            name, entry["source_sha256"][:12],
            entry["no_grounding"]["sha256"][:12], entry["no_grounding"]["structural_checks"]["all_pass"],
            entry["no_order"]["sha256"][:12], entry["no_order"]["structural_checks"]["all_pass"]))
    print("all_checks_pass", ok, "->", report_path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
