#!/usr/bin/env python3
"""Read-only audit: did agentic induction ever see the recombination split's held-out combination?

Executes the spec `VIKI-L2 Agentic Induction 组合泄漏审计` (L0-L6) and writes every artefact,
with sha256, to a single directory. Makes no LLM call, mutates no frozen artefact, changes no
split, seed or threshold. A step whose input is absent is recorded as 未执行 with the missing
path; no substitute data is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

A11 = ROOT / "results/viki_memory_experiments/amendment11"
A10 = ROOT / "results/viki_memory_experiments/amendment10"
RUNG_ROOT = ROOT / "outputs/agentic_rung"

# The held-out rule, taken from the split's own generator rather than inferred.
# scripts/viki_amendment10_recombine.py:40-41 and its module docstring.
CUTTING = ("cut_fruit_on_board", "cut_two_fruits_on_board")
DELIVERY = "single_move_asset_to_target"
RULE_SOURCE = "scripts/viki_amendment10_recombine.py:40-41 (CUTTING x DELIVERY), docstring lines 3-14"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class Out:
    """Every artefact lands here, hashed, before any verdict is computed."""

    def __init__(self, directory: Path):
        self.dir = Path(directory).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index: Dict[str, Dict[str, Any]] = {}

    def write(self, name: str, payload: Any) -> Dict[str, Any]:
        path = self.dir / name
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str))
        try:
            shown = str(path.relative_to(ROOT))
        except ValueError:
            shown = str(path)
        entry = {"path": shown, "sha256": sha256_of(path), "bytes": path.stat().st_size}
        self.index[name] = entry
        return entry


# --------------------------------------------------------------------------- signatures

def goal_predicates(truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    from our_method.skill_memory_v2.simulator import flatten_predicates
    out = []
    for group in truth.get("goal_constraints") or []:
        out.extend(flatten_predicates(group))
    return out


def effect_key_of(predicate: Dict[str, Any]) -> Optional[str]:
    from our_method.skill_memory_v2.simulator import predicate_status
    status = predicate_status(predicate)
    if status.get("pos.name"):
        return "pos.name"
    if status.get("is_activated"):
        return "is_activated"
    return None


def row_signature(truth: Dict[str, Any], sim, seed: int) -> Dict[str, Any]:
    """(effect key, solo|coop) per goal predicate, decided by the project's own criterion.

    The mode comes from replaying the row's own reference plan and applying `_runs_alone`,
    the counterfactual that distinguishes an operator one robot can run from one that needs
    several. It is not a label read off `task_name`.
    """
    from our_method.skill_memory_v2 import induction
    from our_method.skill_memory_v2.simulator import predicate_status

    def key(predicate):
        return canonical([predicate.get("name"), predicate_status(predicate)])

    units: Dict[str, Tuple[str, str]] = {}
    actor_of: Dict[str, str] = {}
    replay_status = "NO_PLAN"
    if isinstance(truth, dict) and truth.get("time_steps"):
        try:
            trace, replay_status = induction.replay(truth, sim, seed)
        except Exception as error:                                   # noqa: BLE001
            trace, replay_status = None, f"REPLAY_ERROR:{type(error).__name__}"
        if trace is not None:
            replay_status = "OK"
            for index, actor, predicate in trace["completions"]:
                effect = effect_key_of(predicate)
                if effect is None or actor is None:
                    continue
                start = induction._segment_start(trace["completions"], index, actor)
                state = trace["states"][start]
                try:
                    alone = induction._runs_alone(state, trace["history"], start, index,
                                                  actor, predicate, sim)
                except Exception:                                    # noqa: BLE001
                    alone = None
                mode = "solo" if alone is True else ("coop" if alone is False else "unknown")
                units[key(predicate)] = (effect, mode)
                actor_of[key(predicate)] = actor

    # Goals never completed in the trace still count as demanded units, mode unknown.
    for predicate in goal_predicates(truth):
        effect = effect_key_of(predicate)
        if effect is not None:
            units.setdefault(key(predicate), (effect, "unknown"))

    multiset = sorted(units.values())
    ordered = _ordered_units(truth, units, key)
    structural = structural_units(truth, actor_of)
    return {"replay": replay_status,
            "multiset": multiset,
            "ordered": ordered,
            "units": structural,
            # Decisive. The (effect, solo/coop) multiset below is a DIAGNOSTIC only: see
            # the L1 self-check, which shows it cannot characterise the comp rows.
            "sig_structural": canonical([structural["has_two_robot_cut_unit"],
                                         structural["has_independent_delivery"]]),
            "sig_multiset": canonical(multiset),
            "sig_ordered": canonical(ordered)}


def _ordered_units(truth, units, key) -> List[Tuple[str, str]]:
    from our_method.skill_memory_v2.simulator import flatten_predicates
    rank: Dict[str, int] = {}
    for constraint in truth.get("temporal_constraints") or []:
        for position, stage in enumerate(constraint):
            for predicate in flatten_predicates(stage):
                identifier = key(predicate)
                rank[identifier] = min(rank.get(identifier, position), position)
    return [units[identifier] for identifier in
            sorted(units, key=lambda i: (rank.get(i, 99), i))]


def temporal_edges(truth: Dict[str, Any], key) -> Set[Tuple[str, str]]:
    """(earlier, later) goal-predicate pairs this row orders, as `dependencies` reads them."""
    from our_method.skill_memory_v2.simulator import flatten_predicates
    edges: Set[Tuple[str, str]] = set()
    for constraint in truth.get("temporal_constraints") or []:
        stages = [[key(p) for p in flatten_predicates(stage)] for stage in constraint]
        for position, earlier in enumerate(stages):
            for later in stages[position + 1:]:
                for one in earlier:
                    for two in later:
                        edges.add((one, two))
    return edges


def structural_units(truth: Dict[str, Any], actor_of: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The two units the split withholds the co-occurrence of, read off the row's own goals.

    Taken from the generator rather than from `task_name`, which cannot express a pair:
    a comp instance is a CUTTING donor (a placement that must precede an activation --
    the fruit reaches the board before the knife is used) extended with a DELIVERY (a
    placement entangled with no activation at all). See RULE_SOURCE.

    Deliberately replay-free: it reads goal_constraints and temporal_constraints only, so
    it cannot be perturbed by simulator behaviour.
    """
    from our_method.skill_memory_v2.simulator import predicate_status

    def key(predicate):
        return canonical([predicate.get("name"), predicate_status(predicate)])

    # The universe is goal_constraints UNION the predicates named in temporal_constraints.
    # Verified necessary: in a comp row the cutting placement (pear -> board) appears ONLY
    # in temporal_constraints, so a goal-only universe cannot see the cut unit at all and
    # scores 0 on every row it is meant to detect.
    from our_method.skill_memory_v2.simulator import flatten_predicates

    kinds: Dict[str, str] = {}
    universe = list(goal_predicates(truth))
    for constraint in truth.get("temporal_constraints") or []:
        for stage in constraint:
            universe.extend(flatten_predicates(stage))
    for predicate in universe:
        effect = effect_key_of(predicate)
        if effect:
            kinds[key(predicate)] = effect
    activations = {k for k, v in kinds.items() if v == "is_activated"}
    placements = {k for k, v in kinds.items() if v == "pos.name"}
    goal_keys = {key(p) for p in goal_predicates(truth)}
    edges = temporal_edges(truth, key)

    cut_pairs = {(a, b) for a, b in edges if a in placements and b in activations}
    # The generator's rule names a TWO-ROBOT cutting pattern: one robot carries the fruit
    # to the board, another works the knife, and that split is what forces a reallocation.
    # An activation gated by a placement the SAME robot performed does not pose it.
    actor_of = actor_of or {}
    two_robot_pairs = {(a, b) for a, b in cut_pairs
                       if actor_of.get(a) and actor_of.get(b) and actor_of[a] != actor_of[b]}
    entangled = {a for a, _ in cut_pairs} | {b for _, b in cut_pairs}
    for a, b in edges:
        if a in activations and b in placements:
            entangled |= {a, b}
    # A delivery must be DEMANDED by the row, so it is taken from the goals, not from a
    # predicate that only appears as somebody's ordering constraint.
    independent = (placements - entangled) & goal_keys
    return {"has_cut_unit": bool(cut_pairs),
            "has_two_robot_cut_unit": bool(two_robot_pairs),
            "has_independent_delivery": bool(independent),
            "n_cut_pairs": len(cut_pairs),
            "n_two_robot_cut_pairs": len(two_robot_pairs),
            "n_activations": len(activations),
            "n_placements": len(placements),
            "n_independent_deliveries": len(independent)}


def carries_heldout_combination(signature: Dict[str, Any]) -> bool:
    """The generator's rule in full: a TWO-ROBOT cutting unit plus an independent delivery."""
    units = signature.get("units") or {}
    return bool(units.get("has_two_robot_cut_unit") and units.get("has_independent_delivery"))


def carries_actor_blind_combination(signature: Dict[str, Any]) -> bool:
    """The same without the two-robot requirement. Reported as a sensitivity bound."""
    units = signature.get("units") or {}
    return bool(units.get("has_cut_unit") and units.get("has_independent_delivery"))


# --------------------------------------------------------------------------- L0

def build_l0(out: Out, sim, seed: int, train_path: Path, args) -> Dict[str, Any]:
    from our_method.skill_memory_v2.build import load_episodes

    report: Dict[str, Any] = {"missing": []}
    episodes = load_episodes(train_path)
    report["train_rows"] = len(episodes)

    def ident(truth, index):
        return {"train_index": index,
                "task_id": (truth.get("task_id") if isinstance(truth, dict) else None),
                "task_name": (truth.get("task_name") if isinstance(truth, dict) else None)}

    induction_half = [ident(episodes[i], i) for i in range(0, len(episodes), 2)]
    selfcheck_half = [ident(episodes[i], i) for i in range(1, len(episodes), 2)]
    out.write("L0_induction_half.json",
              {"n": len(induction_half), "workbench_index_is_position_in_this_list": True,
               "global_index_rule": "global = 2 * workbench_index", "rows": induction_half})
    out.write("L0_selfcheck_half.json", {"n": len(selfcheck_half), "rows": selfcheck_half})
    report["induction_half"] = len(induction_half)
    report["selfcheck_half"] = len(selfcheck_half)
    report["halves_disjoint"] = not (
        {r["train_index"] for r in induction_half} & {r["train_index"] for r in selfcheck_half})
    report["halves_cover_train"] = (
        len(induction_half) + len(selfcheck_half) == len(episodes))

    # 3 test split
    test_path = args.test or (train_path.parent / "test.parquet")
    if Path(test_path).is_file():
        test_episodes = load_episodes(Path(test_path))
        out.write("L0_test_split.json",
                  {"n": len(test_episodes), "source": str(test_path),
                   "rows": [ident(t, i) for i, t in enumerate(test_episodes)]})
        report["test_split"] = len(test_episodes)
    else:
        report["missing"].append(str(test_path))
        report["test_split"] = "未执行，缺 %s" % test_path

    # 4 id manifest 924
    manifest = args.manifest
    have_manifest = bool(manifest and Path(manifest).is_file())
    if have_manifest:
        rows_manifest = []
        with Path(manifest).open() as source:
            for line in source:
                if line.strip():
                    record = json.loads(line)
                    rows_manifest.append({"index": int(record["index"]),
                                          "task_id": record.get("task_id"),
                                          "task_name": record.get("task_name")})
        out.write("L0_id_manifest_924.json",
                  {"n": len(rows_manifest), "source": str(manifest),
                   "source_sha256": sha256_of(Path(manifest)), "rows": rows_manifest})
        report["id_manifest_924"] = len(rows_manifest)
    else:
        report["missing"].append(str(manifest) if manifest else "id_manifest_924 (path not supplied)")
        report["id_manifest_924"] = "未执行，缺 id_manifest_924"

    # 5 held-out family folds
    folds = sorted(A11.glob("skill_memory_v2.fold_*.json"))
    fold_rows = []
    for path in folds:
        record = json.loads(path.read_text())
        fold_rows.append({"artefact": str(path.relative_to(ROOT)),
                          "excluded_family": record.get("excluded_family"),
                          "layer1_operators": len(record.get("layer1", {}).get("operators", [])),
                          "layer1_sha256": hashlib.sha256(
                              canonical(record.get("layer1")).encode()).hexdigest()})
    out.write("L0_heldout_family_folds.json", {"n": len(fold_rows), "rows": fold_rows})
    report["heldout_family_folds"] = len(fold_rows)

    # 6 comp 297
    comp = load_comp(out, report)
    report["comp_297"] = (len(comp) if comp is not None else report.get("comp_297"))

    # 7 support pool
    probe = args.probe
    support_pool = [{"workbench_index": i, "global_index": 2 * i,
                     **ident(episodes[2 * i], 2 * i)} for i in range(probe)
                    if 2 * i < len(episodes)]
    out.write("L0_support_pool.json",
              {"n": len(support_pool), "probe": probe,
               "source": "scripts/viki_assemble_agentic_library.py --probe (default 60), "
                         "range(probe) over Workbench.episodes = episodes[::2]",
               "rows": support_pool})
    report["support_pool"] = len(support_pool)

    # 8 unsolved lists, per rung
    unsolved, unsolved_missing = load_unsolved()
    out.write("L0_unsolved_lists.json", unsolved)
    report["unsolved_lists"] = {"files": len(unsolved["files"]),
                                "rungs": len(unsolved["rungs"])}
    if unsolved_missing:
        report["missing"].extend(unsolved_missing)
    # The spec's L0 condition: all eight pools materialised, halves disjoint, halves cover
    # train. A missing pool fails L0 even if every other pool is clean.
    eight = {"induction_half": bool(induction_half), "selfcheck_half": bool(selfcheck_half),
             "test_split": Path(test_path).is_file(), "id_manifest_924": have_manifest,
             "heldout_family_folds": bool(fold_rows), "comp_297": comp is not None,
             "support_pool": bool(support_pool),
             "unsolved_lists": bool(unsolved["rungs"])}
    report["eight_pools"] = eight
    report["pass"] = bool(all(eight.values()) and report["halves_disjoint"]
                          and report["halves_cover_train"])
    return {"report": report, "episodes": episodes, "comp": comp,
            "induction_half": induction_half, "support_pool": support_pool,
            "unsolved": unsolved}


def load_comp(out: Out, report: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    import pandas as pd
    from viki_amendment8b import native

    rows: List[Dict[str, Any]] = []
    for form in ("text", "imaged"):
        path = A10 / f"recombination.{form}.parquet"
        if not path.is_file():
            report["missing"].append(str(path))
            report["comp_297"] = "未执行，缺 %s" % path
            return None
        frame = pd.read_parquet(path)
        for position in range(len(frame)):
            truth = native(frame.iloc[position].to_dict())["reward_model"]["ground_truth"]
            rows.append({"form": form, "row": position,
                         "task_id": str(truth.get("task_id")),
                         "task_name": str(truth.get("task_name")),
                         "source_row": truth.get("source_row"),
                         "layout_id": truth.get("layout_id"),
                         "added_asset": (truth.get("added_asset") or {}).get("name"),
                         "truth": truth})
    text_ids = [r["task_id"] for r in rows if r["form"] == "text"]
    imaged_ids = [r["task_id"] for r in rows if r["form"] == "imaged"]
    pairing = {"rows_per_form": dict(Counter(r["form"] for r in rows)),
               "distinct_task_ids_text": len(set(text_ids)),
               "distinct_task_ids_imaged": len(set(imaged_ids)),
               "distinct_task_ids_overall": len(set(text_ids) | set(imaged_ids)),
               "text_ids_repeated": {t: c for t, c in Counter(text_ids).items() if c > 1},
               "imaged_ids_repeated": {t: c for t, c in Counter(imaged_ids).items() if c > 1},
               "text_only": sorted(set(text_ids) - set(imaged_ids)),
               "imaged_only": sorted(set(imaged_ids) - set(text_ids))}
    out.write("L0_comp_297.json",
              {"n": len(rows), "paired_by": "task_id", "pairing": pairing,
               "rows": [{k: v for k, v in r.items() if k != "truth"} for r in rows]})
    report["comp_pairing"] = pairing
    return rows


def load_unsolved() -> Tuple[Dict[str, Any], List[str]]:
    """The 'the memory cannot solve these' lists shown to the agent, per rung."""
    files, rungs, missing = [], {}, []
    for path in sorted(ROOT.glob("outputs/rung_*targets*.json")):
        try:
            record = json.loads(path.read_text())
        except Exception:                                            # noqa: BLE001
            continue
        files.append(str(path.relative_to(ROOT)))
        targets = record.get("targets") or {}
        for family, entry in (targets.items() if isinstance(targets, dict) else []):
            if not isinstance(entry, dict):
                continue
            rungs.setdefault(f"{path.name}:{family}", {
                "library": record.get("library"),
                "seeds": entry.get("seeds", []),
                "holdout": entry.get("holdout", []),
                "target_key": entry.get("target_key"),
            })
    if not files:
        missing.append("outputs/rung_*targets*.json")
    return {"files": files, "rungs": rungs}, missing


# --------------------------------------------------------------------------- L1

def build_l1(out: Out, sim, seed: int, episodes, comp) -> Dict[str, Any]:
    report: Dict[str, Any] = {"rule_case": "a",
                              "rule": "held out = the co-occurrence of a CUTTING family "
                                      "(%s) with %s in one instance" % (", ".join(CUTTING), DELIVERY),
                              "rule_source": RULE_SOURCE}
    comp_by_task: Dict[str, Dict[str, Any]] = {}
    for row in comp:
        comp_by_task.setdefault(row["task_id"], row)

    comp_sigs = {}
    for task_id, row in comp_by_task.items():
        comp_sigs[task_id] = row_signature(row["truth"], sim, seed)
    H_multiset = sorted({s["sig_multiset"] for s in comp_sigs.values()})
    H_ordered = sorted({s["sig_ordered"] for s in comp_sigs.values()})
    comp_carrying = {t for t, s in comp_sigs.items() if carries_heldout_combination(s)}

    train_sigs = []
    for index, truth in enumerate(episodes):
        signature = row_signature(truth, sim, seed) if isinstance(truth, dict) else {
            "replay": "NOT_A_DICT", "multiset": [], "ordered": [],
            "sig_multiset": "[]", "sig_ordered": "[]"}
        train_sigs.append({"train_index": index,
                           "task_id": (truth.get("task_id") if isinstance(truth, dict) else None),
                           "task_name": (truth.get("task_name") if isinstance(truth, dict) else None),
                           "half": ("induction" if index % 2 == 0 else "selfcheck"),
                           **signature})

    flagged_multiset = [r for r in train_sigs if r["sig_multiset"] in set(H_multiset)]
    flagged_ordered = [r for r in train_sigs if r["sig_ordered"] in set(H_ordered)]
    flagged_pattern = [r for r in train_sigs if carries_heldout_combination(r)]

    # ---- L1 self-check: the detector must fire on the rows it is meant to characterise.
    # Running the known-positive case through the criterion before trusting any zero or any
    # hit it produces. A detector that misses the comp rows cannot be used to flag train
    # rows, and no verdict is emitted from one that fails here.
    comp_detected = len(comp_carrying)
    comp_rate = comp_detected / max(1, len(comp_by_task))
    flagged_actor_blind = [r for r in train_sigs if carries_actor_blind_combination(r)]
    comp_blind = sum(1 for s in comp_sigs.values() if carries_actor_blind_combination(s))
    diag_comp_detected = sum(
        1 for s in comp_sigs.values()
        if ("is_activated", "coop") in set(map(tuple, s["multiset"]))
        and ("pos.name", "solo") in set(map(tuple, s["multiset"])))
    selfcheck = {
        "decisive_detector": "structural: cut unit (placement ordered before an activation) "
                             "AND an independent delivery (placement entangled with no activation)",
        "comp_rows": len(comp_by_task),
        "comp_rows_detected": comp_detected,
        "comp_detection_rate": round(comp_rate, 4),
        "donor_family_rows_detected": {
            family: sum(1 for r in train_sigs
                        if r["task_name"] == family and carries_heldout_combination(r))
            for family in CUTTING},
        "sensitivity_actor_blind_detector":
            "cut unit WITHOUT the two-robot requirement -- reported as an upper bound",
        "sensitivity_comp_rows_detected": comp_blind,
        "sensitivity_train_rows_flagged": len(flagged_actor_blind),
        "sensitivity_families": dict(Counter(r["task_name"] for r in flagged_actor_blind)),
        "superseded_detector": "(effect, solo|coop) multiset -- unfit, kept as the record",
        "superseded_comp_rows_detected": diag_comp_detected,
        "superseded_is_unfit": diag_comp_detected == 0,
        "pass": comp_rate >= 0.95,
    }

    # Only the decisive detector feeds L2/L4/L5.
    union = {r["train_index"] for r in flagged_pattern}

    H_structural = sorted({s["sig_structural"] for s in comp_sigs.values()})
    out.write("L1_H.json", {"case": "a", "rule_source": RULE_SOURCE,
                            "H_structural": H_structural,
                            "H_multiset_diagnostic": H_multiset,
                            "H_ordered_diagnostic": H_ordered,
                            "comp_rows_carrying_the_heldout_combination": len(comp_carrying),
                            "comp_distinct_task_ids": len(comp_by_task),
                            "selfcheck": selfcheck})
    out.write("L1_comp_signatures.json",
              {t: {k: v for k, v in s.items() if k in ("replay", "multiset", "ordered", "units")}
               for t, s in comp_sigs.items()})
    out.write("L1_train_signatures.json", {"n": len(train_sigs), "rows": train_sigs})
    out.write("L1_train_with_heldout_sig.json",
              {"decisive_two_robot_cut_plus_independent_delivery": len(flagged_pattern),
               "sensitivity_actor_blind": len(flagged_actor_blind),
               "superseded_by_multiset": len(flagged_multiset),
               "superseded_by_ordered": len(flagged_ordered),
               "decisive_indices": sorted(union),
               "decisive_rows": [{k: v for k, v in r.items() if k != "ordered"}
                                 for r in flagged_pattern[:400]],
               "sensitivity_families": dict(Counter(r["task_name"] for r in flagged_actor_blind)),
               "sensitivity_indices": sorted(r["train_index"] for r in flagged_actor_blind)})

    report.update({
        "selfcheck": selfcheck,
        "H_structural": H_structural,
        "H_multiset_size": len(H_multiset),
        "H_ordered_size": len(H_ordered),
        "comp_distinct_task_ids": len(comp_by_task),
        "comp_rows_carrying_the_pattern": len(comp_carrying),
        "train_with_heldout_sig_by_multiset": len(flagged_multiset),
        "train_with_heldout_sig_by_ordered": len(flagged_ordered),
        "train_with_heldout_sig_by_pattern": len(flagged_pattern),
        "train_with_heldout_sig_decisive": len(union),
        "train_with_heldout_sig_actor_blind": len(flagged_actor_blind),
        "families_of_flagged_decisive": Counter(
            r["task_name"] for r in flagged_pattern),
        "families_of_flagged_diagnostic_only": Counter(
            r["task_name"] for r in flagged_multiset),
        "pass": bool(selfcheck["pass"]),
    })
    return {"report": report, "H_multiset": set(H_multiset), "H_ordered": set(H_ordered),
            "flagged_indices": union, "train_sigs": train_sigs,
            "comp_by_task": comp_by_task, "comp_sigs": comp_sigs}


# --------------------------------------------------------------------------- L2/L3 pools

WORKBENCH_TOOLS_WITH_EPISODE = (
    "show_trace", "check_actor", "contrast_actors", "run_operator", "try_bind",
    "episode_state", "list_episodes", "plan_with", "describe_episode",
)
INDEX_KEYS = ("index", "episode", "episode_index", "on", "target", "at")


def scan_transcripts(rung_root: Path) -> Dict[str, Any]:
    """Every workbench episode index the agent addressed, per run, with the raw text kept."""
    runs = []
    for verdict in sorted(rung_root.glob("*/*/verdict.json")):
        directory = verdict.parent
        transcript_path = directory / "transcript.json"
        tag, run = directory.parent.name, directory.name
        seed_match = re.match(r"e(\d+)_s(\d+)$", run)
        entry: Dict[str, Any] = {
            "tag": tag, "run": run,
            "seed_episode": int(seed_match.group(1)) if seed_match else None,
            "sample": int(seed_match.group(2)) if seed_match else None,
            "transcript": transcript_path.is_file(),
            "tool_indices": [], "tools": [], "text_len": 0,
        }
        if transcript_path.is_file():
            raw = transcript_path.read_text()
            entry["text_len"] = len(raw)
            entry["text"] = raw
            try:
                moves = json.loads(raw)
            except Exception:                                        # noqa: BLE001
                moves = []
            indices: Set[int] = set()
            for move in moves if isinstance(moves, list) else []:
                if not isinstance(move, dict):
                    continue
                if move.get("tool"):
                    entry["tools"].append(move["tool"])
                for blob in (move.get("answer"), move.get("result")):
                    indices |= _indices_in(blob)
            entry["tool_indices"] = sorted(indices)
        runs.append(entry)
    return {"runs": runs}


def _indices_in(blob: Any) -> Set[int]:
    """Episode indices named in a tool request or returned by one."""
    found: Set[int] = set()
    if blob is None:
        return found
    if isinstance(blob, str):
        try:
            blob = json.loads(blob)
        except Exception:                                            # noqa: BLE001
            for match in re.finditer(r'"(?:%s)"\s*:\s*(\d+)' % "|".join(INDEX_KEYS), blob):
                found.add(int(match.group(1)))
            return found
    if isinstance(blob, dict):
        for key, value in blob.items():
            if key in INDEX_KEYS and isinstance(value, int):
                found.add(value)
            else:
                found |= _indices_in(value)
    elif isinstance(blob, list):
        for item in blob:
            found |= _indices_in(item)
    return found


def build_l2_l3(out: Out, l0, l1, rung_root: Path) -> Dict[str, Any]:
    episodes = l0["episodes"]
    comp_task_ids = {row["task_id"] for row in l0["comp"]}
    comp_by_task = l1["comp_by_task"]
    flagged = l1["flagged_indices"]
    train_task_ids = {}
    for index, truth in enumerate(episodes):
        if isinstance(truth, dict) and truth.get("task_id") is not None:
            train_task_ids.setdefault(str(truth["task_id"]), index)

    scan = scan_transcripts(rung_root)
    runs = scan["runs"]

    def to_global(workbench_index: int) -> Optional[int]:
        target = 2 * workbench_index
        return target if target < len(episodes) else None

    def assess(name: str, workbench_indices: List[int]) -> Dict[str, Any]:
        globals_ = [g for g in (to_global(i) for i in workbench_indices) if g is not None]
        by_task = sum(1 for g in globals_
                      if str((episodes[g] or {}).get("task_id")) in comp_task_ids)
        by_sig = sum(1 for g in globals_ if g in flagged)
        out_of_half = [i for i in workbench_indices if to_global(i) is None]
        return {"pool": name, "n_indices": len(set(workbench_indices)),
                "intersect_comp_297_by_task_id": by_task,
                "intersect_train_with_heldout_sig": by_sig,
                "indices_outside_induction_half": len(out_of_half),
                "families": dict(Counter((episodes[g] or {}).get("task_name") for g in globals_))}

    seeds = sorted({r["seed_episode"] for r in runs if r["seed_episode"] is not None})
    unsolved_seeds, unsolved_holdout = [], []
    for entry in l0["unsolved"]["rungs"].values():
        unsolved_seeds += list(entry.get("seeds") or [])
        unsolved_holdout += list(entry.get("holdout") or [])
    tool_indices = sorted({i for r in runs for i in r["tool_indices"]})
    per_tool: Dict[str, List[int]] = defaultdict(list)
    for run in runs:
        for index in run["tool_indices"]:
            for tool in run["tools"]:
                if tool in WORKBENCH_TOOLS_WITH_EPISODE:
                    per_tool[tool].append(index)

    table = [
        assess("rung seed episodes", seeds),
        assess("unsolved_list (seeds shown)", sorted(set(unsolved_seeds))),
        assess("unsolved_list (holdout shown)", sorted(set(unsolved_holdout))),
        assess("support_pool", [r["workbench_index"] for r in l0["support_pool"]]),
        assess("minimality / all tool-addressed episodes", tool_indices),
    ]
    for tool in ("run_operator", "try_bind", "check_actor", "contrast_actors"):
        table.append(assess(f"tool:{tool}", sorted(set(per_tool.get(tool, [])))))

    l2_pass = all(row["intersect_comp_297_by_task_id"] == 0
                  and row["intersect_train_with_heldout_sig"] == 0 for row in table)

    # ---- L3
    outside = [i for i in tool_indices if to_global(i) is None]
    comp_hits = [row for row in table if row["intersect_comp_297_by_task_id"]]
    exact_text_hits, trigram_hits = [], []
    comp_plans = {t: _plan_tokens(r["truth"]) for t, r in comp_by_task.items()}
    for run in runs:
        text = run.get("text") or ""
        if not text:
            continue
        for task_id in comp_task_ids:
            if task_id and task_id in text:
                exact_text_hits.append({"run": f'{run["tag"]}/{run["run"]}', "task_id": task_id})
        text_tri = _trigrams(re.findall(r"[A-Za-z_][A-Za-z0-9_ ]*", text))
        for task_id, tokens in comp_plans.items():
            plan_tri = _trigrams(tokens)
            if not plan_tri:
                continue
            containment = len(plan_tri & text_tri) / len(plan_tri)
            if containment >= 0.9:
                trigram_hits.append({"run": f'{run["tag"]}/{run["run"]}',
                                     "task_id": task_id, "containment": round(containment, 4)})

    l3 = {"transcripts_found": sum(1 for r in runs if r["transcript"]),
          "run_dirs": len(runs),
          "tool_addressed_indices": len(tool_indices),
          "indices_outside_induction_half": len(outside),
          "indices_in_comp_297": sum(row["intersect_comp_297_by_task_id"] for row in table),
          "indices_with_heldout_signature": sum(
              row["intersect_train_with_heldout_sig"] for row in table),
          "exact_task_id_text_matches": len(exact_text_hits),
          "trigram_plan_matches_ge_0.9": len(trigram_hits),
          "examples": {"exact": exact_text_hits[:20], "trigram": trigram_hits[:20]}}
    l3["pass"] = (l3["indices_outside_induction_half"] == 0
                  and l3["indices_in_comp_297"] == 0
                  and l3["indices_with_heldout_signature"] == 0
                  and l3["exact_task_id_text_matches"] == 0
                  and l3["trigram_plan_matches_ge_0.9"] == 0)

    out.write("L2_pool_intersections.json", {"rows": table, "pass": l2_pass})
    out.write("L3_transcript_audit.json", l3)
    out.write("L3_runs.json", {"n": len(runs),
                               "rows": [{k: v for k, v in r.items() if k != "text"} for r in runs]})
    return {"l2": {"rows": table, "pass": l2_pass}, "l3": l3}


def _plan_verbs(truth: Dict[str, Any]) -> List[str]:
    verbs = []
    for step in truth.get("time_steps") or []:
        for robot, action in sorted((step.get("actions") or {}).items()):
            if action is not None and len(action):
                verbs.append(str(action[0]))
    return verbs


def _plan_tokens(truth: Dict[str, Any]) -> List[str]:
    tokens = []
    for step in truth.get("time_steps") or []:
        for robot, action in sorted((step.get("actions") or {}).items()):
            if action is None:
                continue
            tokens.extend([str(x) for x in list(action)])
    return tokens


def _trigrams(tokens: List[str]) -> Set[Tuple[str, str, str]]:
    return {tuple(tokens[i:i + 3]) for i in range(max(0, len(tokens) - 2))}


# --------------------------------------------------------------------------- L4

def build_l4(out: Out, l0, l1, sim, seed: int, memory_path: Path) -> Dict[str, Any]:
    from our_method.skill_memory_v2 import dependencies, vocabulary

    if not memory_path.is_file():
        return {"status": "未执行", "missing": str(memory_path)}
    record = json.loads(memory_path.read_text())
    build_args = {"seed": record.get("seed", seed), "per_family": record.get("per_family", 250),
                  "excluded_family": record.get("excluded_family")}
    episodes = l0["episodes"]
    flagged = l1["flagged_indices"]
    induction_set = [episodes[i] for i in range(0, len(episodes), 2)]
    restricted = [episodes[i] for i in range(0, len(episodes), 2) if i not in flagged]

    rebuilt = {
        "layer2": dependencies.mine(restricted, sim, build_args["seed"],
                                    build_args["per_family"], build_args["excluded_family"]),
        "layer3": vocabulary.harvest(restricted, build_args["excluded_family"]),
    }
    result: Dict[str, Any] = {
        "status": "RAN",
        "memory": str(memory_path.relative_to(ROOT)),
        "layers_2_3_borrowed_from": record.get("layers_2_3_borrowed_from"),
        "build_args": build_args,
        "induction_set": len(induction_set),
        "restricted_pool": len(restricted),
        "rows_removed": len(induction_set) - len(restricted),
    }
    for layer in ("layer2", "layer3"):
        stored, made = record.get(layer), rebuilt[layer]
        identical = canonical(stored) == canonical(made)
        entry = {"identical": identical,
                 "sha_stored": hashlib.sha256(canonical(stored).encode()).hexdigest()[:16],
                 "sha_rebuilt": hashlib.sha256(canonical(made).encode()).hexdigest()[:16]}
        if layer == "layer2" and not identical:
            kept_stored = set((stored or {}).get("kept_patterns", []))
            kept_made = set(made.get("kept_patterns", []))
            entry["kept_patterns_only_with_heldout_rows"] = sorted(kept_stored - kept_made)
            entry["kept_patterns_only_without"] = sorted(kept_made - kept_stored)
            entry["rule_count_stored"] = len(kept_stored)
            entry["rule_count_rebuilt"] = len(kept_made)
        if layer == "layer3" and not identical:
            for field in ("assets", "places"):
                entry[f"{field}_only_with_heldout_rows"] = sorted(
                    set((stored or {}).get(field, [])) - set(made.get(field, [])))
        result[layer] = entry
    result["pass"] = bool(result["layer2"]["identical"] and result["layer3"]["identical"])
    out.write("L4_mining_pool.json", result)
    return result


# --------------------------------------------------------------------------- L5

def build_l5(out: Out, l0, l1, library_path: Path, assembly_path: Path) -> Dict[str, Any]:
    if not library_path.is_file():
        return {"status": "未执行", "missing": str(library_path)}
    library = json.loads(library_path.read_text())
    operators = library["operators"] if isinstance(library, dict) else library
    episodes = l0["episodes"]
    comp_task_ids = {row["task_id"] for row in l0["comp"]}
    flagged = l1["flagged_indices"]

    provenance_rows, provenance_hits = [], 0
    if assembly_path.is_file():
        assembly = json.loads(assembly_path.read_text())
        for row in assembly.get("rows", []):
            seeds = []
            for source in row.get("sources", []):
                match = re.search(r"/e(\d+)_s\d+/verdict\.json$", source)
                if match:
                    seeds.append(int(match.group(1)))
            works = row.get("works_on", [])
            addressed = sorted(set(seeds) | set(works))
            globals_ = [2 * i for i in addressed if 2 * i < len(episodes)]
            hits_comp = sum(1 for g in globals_
                            if str((episodes[g] or {}).get("task_id")) in comp_task_ids)
            hits_sig = sum(1 for g in globals_ if g in flagged)
            provenance_hits += hits_comp + hits_sig
            provenance_rows.append({"effect_key": row.get("effect_key"),
                                    "seed_episodes": sorted(set(seeds)),
                                    "verified_on": works,
                                    "intersect_comp_297": hits_comp,
                                    "intersect_heldout_sig": hits_sig})
    else:
        provenance_rows = [{"note": "未执行，缺 %s" % assembly_path}]

    comp_plans = {row["task_id"]: _plan_tokens(row["truth"]) for row in l0["comp"]}
    comp_verbs = {row["task_id"]: _plan_verbs(row["truth"]) for row in l0["comp"]}
    overlaps, exact_pairs, verb_pairs = [], [], []
    for position, operator in enumerate(operators):
        body = operator.get("body") or []
        body_tri = _trigrams([str(x) for action in body for x in action])
        verb_tri = _trigrams([str(action[0]) for action in body if action])
        best_token, best_verb = 0.0, 0.0
        for task_id in comp_plans:
            if body_tri:
                containment = len(body_tri & _trigrams(comp_plans[task_id])) / len(body_tri)
                best_token = max(best_token, containment)
                if containment >= 1.0 and len(body_tri) >= 3:
                    exact_pairs.append({"operator": position, "task_id": task_id})
            if verb_tri:
                containment = len(verb_tri & _trigrams(comp_verbs[task_id])) / len(verb_tri)
                best_verb = max(best_verb, containment)
                if containment >= 1.0 and len(verb_tri) >= 3:
                    verb_pairs.append({"operator": position, "task_id": task_id})
        overlaps.append({"operator": position, "effect": (operator.get("effect") or {}).get("key"),
                         "body_len": len(body),
                         "max_containment_tokens": round(best_token, 4),
                         "max_containment_verbs_only": round(best_verb, 4)})

    schema_rows = []
    for position, operator in enumerate(operators):
        key = (operator.get("effect") or {}).get("key")
        with_flag = sum(1 for i in flagged
                        if any(effect_key_of(p) == key for p in goal_predicates(episodes[i] or {})))
        without = sum(1 for i in range(0, len(episodes), 2) if i not in flagged
                      and any(effect_key_of(p) == key for p in goal_predicates(episodes[i] or {})))
        schema_rows.append({"operator": position, "effect_key": key,
                            "supporting_rows_with_heldout_sig": with_flag,
                            "supporting_rows_without": without,
                            "only_supported_by_heldout_rows": bool(with_flag and not without)})

    result = {"status": "RAN", "library": str(library_path.relative_to(ROOT)),
              "operators": len(operators),
              "provenance_intersections": provenance_hits,
              "provenance_rows": provenance_rows,
              "body_overlap": overlaps,
              "overlap_equal_1.0_pairs": exact_pairs,
              "body_overlap_token_level_is_non_discriminating":
                  "operator bodies are variable-abstracted (?x/?y) and comp reference plans "
                  "are ground, so token trigrams cannot intersect. Max observed: %s. This "
                  "variant establishes nothing either way."
                  % max([r["max_containment_tokens"] for r in overlaps] or [0.0]),
              "overlap_verbs_only_equal_1.0_pairs": len(verb_pairs),
              "overlap_verbs_only_is_expected_not_evidence":
                  "the primitive set is 5-6 verbs shared by every family by construction, so "
                  "a verb-sequence match is not evidence of having seen a comp row",
              "max_containment_verbs_only":
                  max([r["max_containment_verbs_only"] for r in overlaps] or [0.0]),
              "effect_schema": schema_rows,
              "schemas_only_supported_by_heldout_rows":
                  sum(1 for r in schema_rows if r["only_supported_by_heldout_rows"])}
    result["pass"] = (provenance_hits == 0 and not exact_pairs
                      and result["schemas_only_supported_by_heldout_rows"] == 0)
    out.write("L5_library_covering.json", result)
    return result


# --------------------------------------------------------------------------- L6

def build_l6(out: Out, l0, library_path: Path) -> Dict[str, Any]:
    agent_folds = sorted(ROOT.glob("outputs/agentic_*fold*.json"))
    if not agent_folds:
        result = {"status": "未执行",
                  "missing": "outputs/agentic_*fold*.json",
                  "finding": "no per-fold agent-built Layer 1 exists on disk; the agent "
                             "library has no held-out-family column, so there is nothing "
                             "to check for reuse",
                  "affects": "held-out-family column only; does not bear on comp"}
        out.write("L6_fold_reuse.json", result)
        return result
    library = json.loads(library_path.read_text()) if library_path.is_file() else {}
    base = hashlib.sha256(canonical(library.get("operators", library)).encode()).hexdigest()
    rows = []
    for path in agent_folds:
        record = json.loads(path.read_text())
        ops = record.get("layer1", {}).get("operators", record.get("operators", []))
        rows.append({"artefact": str(path.relative_to(ROOT)),
                     "layer1_sha256": hashlib.sha256(canonical(ops).encode()).hexdigest(),
                     "same_as_full_library": hashlib.sha256(
                         canonical(ops).encode()).hexdigest() == base})
    result = {"status": "RAN", "full_library_sha256": base, "rows": rows,
              "pass": all(not r["same_as_full_library"] for r in rows)}
    out.write("L6_fold_reuse.json", result)
    return result


# --------------------------------------------------------------------------- report

def verdict_of(l1, l2, l3, l4, l5) -> str:
    if not (l1.get("selfcheck") or {}).get("pass"):
        return "UNFIT (L1 detector fails its own self-check; no verdict is emitted)"
    hard = [l2.get("pass"), l3.get("pass"), l5.get("pass") if l5.get("status") == "RAN" else None]
    if any(value is False for value in hard):
        return "C"
    if l4.get("status") == "RAN" and l4.get("pass") is False:
        return "B"
    return "A"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT.parent / "VIKI-R")
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "results/viki_memory_experiments/amendment8b/interactive_manifest.jsonl")
    parser.add_argument("--library", type=Path, default=ROOT / "outputs/agentic_library_runner.json")
    parser.add_argument("--assembly", type=Path,
                        default=ROOT / "outputs/agentic_library_runner_assembly.json")
    parser.add_argument("--memory", type=Path, default=ROOT / "outputs/agentic_memory_runner.json")
    parser.add_argument("--rung-root", type=Path, default=RUNG_ROOT)
    parser.add_argument("--probe", type=int, default=60)
    parser.add_argument("--out", type=Path,
                        default=ROOT / ("audit/comp_leakage_%s" % date.today().isoformat()))
    arguments = parser.parse_args(argv)

    from our_method.skill_memory_v2.simulator import SEED, Simulator

    train = arguments.train or (arguments.benchmark_root / "data/VIKI-R/viki/VIKI-L2/train.parquet")
    sim = Simulator(arguments.benchmark_root)
    out = Out(arguments.out)

    steps: Dict[str, Any] = {}
    l0 = build_l0(out, sim, SEED, train, arguments)
    steps["L0"] = l0["report"]
    l1 = build_l1(out, sim, SEED, l0["episodes"], l0["comp"])
    steps["L1"] = l1["report"]
    l23 = build_l2_l3(out, l0, l1, arguments.rung_root)
    steps["L2"], steps["L3"] = l23["l2"], l23["l3"]
    steps["L4"] = build_l4(out, l0, l1, sim, SEED, arguments.memory)
    steps["L5"] = build_l5(out, l0, l1, arguments.library, arguments.assembly)
    steps["L6"] = build_l6(out, l0, arguments.library)

    result = verdict_of(steps["L1"], steps["L2"], steps["L3"], steps["L4"], steps["L5"])
    summary = {"verdict": result, "steps": steps, "artefacts": out.index}
    out.write("AUDIT_SUMMARY.json", summary)
    write_markdown(out, result, steps)
    print("verdict %s" % result)
    print("wrote %s" % out.dir)
    return 0


def write_markdown(out: Out, result: str, steps: Dict[str, Any]) -> None:
    lines = ["# AUDIT_REPORT", "", "结局 %s" % result, ""]
    for name in ("L0", "L1", "L2", "L3", "L4", "L5", "L6"):
        step = steps.get(name, {})
        lines.append("## %s" % name)
        status = step.get("status")
        if status == "未执行":
            lines.append("未执行，缺 %s" % step.get("missing"))
        else:
            lines.append("通过：%s" % step.get("pass"))
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(step, indent=1, ensure_ascii=False, default=str)[:6000])
        lines.append("```")
        lines.append("")
    lines.append("## 附录：产物")
    lines.append("")
    for name, entry in sorted(out.index.items()):
        lines.append("- `%s`  sha256 `%s`  %d bytes" % (entry["path"], entry["sha256"], entry["bytes"]))
    (out.dir / "AUDIT_REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
