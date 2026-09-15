#!/usr/bin/env python3
"""Recover the temporal DAG of a PARTNR episode from the instruction's words alone.

Every temporal episode in train_mini and val_mini has the same shape: the closure of its
`dag_edges` is exactly "stage(u) < stage(v)", the stages are contiguous in proposition order,
and the instruction carries an order word (first / then / finally ...). So the whole DAG is
one stage number per requirement. The typed arm loses it because `as_requirements` only
orders two lines about the same object, while most edges join different objects.

The rule here needs no temporal example and no model call: cut the instruction into clauses
at order words, give each clause a stage, and give each requirement line the stage of the
clause that mentions its kind of object. It is measured two ways on train_mini only:

  oracle   the ground-truth requirements as the lines -- what the rule can reach at best
  typedR / typedRS
           the stored 7B answers, re-projected with the flags of `examples/run.sh`, matched to
           ground truth the way `grade_typed` matches -- what the planner would actually get

against `same_subject`, the ordering the typed arm uses today. Pairs are over lines matched to
distinct ground-truth requirements; a spurious order on a non-temporal episode is counted
separately, because an edge that is not in the task can only block execution.

    /root/venvs/partnr/bin/python scripts/partnr_stage_recovery.py \\
      --out outputs/cand_iface_0914/train_mini/stage_recovery
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "our_method"))

import partnr_typed_intent as ti  # noqa: E402
from partnr_task_types import classify  # noqa: E402

R1 = ROOT / "outputs/cand_iface_0914/train_mini/r1"

# ------------------------------------------------------------------ the rule

# The rule lives in the planner's module so what is measured here is what runs
# (`goal_source: typed`, `typed_stages: True`). This script's own copy was frozen at v2 and
# moved there unchanged; the summary from that copy is outputs/.../stage_recovery_v2.
from skill_memory_v2.partnr_typed_goals import as_requirements, beside_requirements, stage_lines  # noqa: E402

MOVES = ("is_on_top", "is_inside", "is_in_room", "is_on_floor")


def gt_beside(episode: Dict[str, Any], scene) -> List[Tuple[set, set, bool]]:
    """Each `is_next_to` as (kinds a, kinds b, foldable: both entities are moved)."""
    props = episode.get("evaluation_propositions") or []
    moved = set()
    for p in props:
        if p.get("function_name") in MOVES:
            moved |= {str(h) for h in (p.get("args") or {}).get("object_handles") or []}
    out = []
    for p in props:
        if p.get("function_name") != "is_next_to":
            continue
        args = p.get("args") or {}
        a = [str(h) for h in args.get("entity_handles_a") or []]
        b = [str(h) for h in args.get("entity_handles_b") or []]
        kinds = lambda hs: {ti.category(scene.handle_to_name[h]) for h in hs if h in scene.handle_to_name}
        out.append((kinds(a), kinds(b), bool(set(a) & moved and set(b) & moved)))
    return out


def score_beside(gt: List[Tuple[set, set, bool]], pred: List[Tuple[str, str]], spatial: bool) -> Counter:
    same = lambda x, y, ka, kb: (x in ka and y in kb) or (x in kb and y in ka)
    c = Counter(episodes=1)
    for ka, kb, foldable in gt:
        hit = any(same(x, y, ka, kb) for x, y in pred)
        c["gt"] += 1
        c["gt_foldable"] += int(foldable)
        c["hit"] += int(hit)
        c["hit_foldable"] += int(hit and foldable)
    for x, y in pred:
        c["pred"] += 1
        c["pred_correct"] += int(any(same(x, y, ka, kb) for ka, kb, _ in gt))
    if not spatial:
        c["nonS_episodes_with_pred"] += int(bool(pred))
    return c


def assign_stages(instruction, kinds, targets=None):
    return stage_lines(instruction, kinds, targets)


# ------------------------------------------------------------------ ground truth and matching

def dag_levels(episode: Dict[str, Any]) -> Dict[int, int]:
    n = len(episode.get("evaluation_propositions") or [])
    edges = []
    for constraint in episode.get("evaluation_constraints") or []:
        if constraint.get("type") == "TemporalConstraint":
            edges = constraint.get("args", {}).get("dag_edges") or []
    level = {i: 0 for i in range(n)}
    for _ in range(n):
        for u, v in edges:
            level[v] = max(level[v], level[u] + 1)
    return level


def match(gt: List[Dict[str, Any]], pred: List[Dict[str, Any]], scene) -> Dict[int, int]:
    """Line index -> ground-truth index, exactly as `grade_typed` pairs them."""
    taken: set = set()
    open_gt = list(range(len(gt)))
    pairs: Dict[int, int] = {}
    left = []
    for j, item in enumerate(pred):
        subject = scene.resolve(item["subject"], scene.objects, taken)
        kind = ti.category(subject) if subject else ti.category(item["subject"])
        target = scene.resolve(item.get("target"), scene.names) if item.get("target") else None
        hit = next((i for i in open_gt if gt[i]["key"] == item["key"] and kind in gt[i]["subject_kinds"]
                    and (target in gt[i]["target_ok"] or (not gt[i]["target_ok"] and target is None))), None)
        if hit is not None:
            open_gt.remove(hit)
            pairs[j] = hit
            if subject:
                taken.add(subject)
        else:
            left.append((j, item, kind, target))
    for j, item, kind, target in left:
        if item["key"] not in ti.PLACEMENTS:
            continue
        room = scene.room_of_furniture.get(target)
        hit = next((i for i in open_gt if gt[i]["key"] == "is_in_room" and kind in gt[i]["subject_kinds"]
                    and room in gt[i]["target_ok"]), None)
        if hit is not None:
            open_gt.remove(hit)
            pairs[j] = hit
    return pairs


def same_subject_order(pred: List[Dict[str, Any]]) -> List[List[int]]:
    """`as_requirements`: a line is after the previous line on the same subject."""
    last: Dict[str, int] = {}
    after = []
    for j, item in enumerate(pred):
        after.append([last[item["subject"]]] if item["subject"] in last else [])
        last[item["subject"]] = j
    return after


def closure(after: List[List[int]]) -> set:
    edges = {(a, j) for j, parents in enumerate(after) for a in parents}
    changed = True
    while changed:
        changed = False
        for a, b in list(edges):
            for c, d in list(edges):
                if b == c and (a, d) not in edges:
                    edges.add((a, d))
                    changed = True
    return edges


def score_pairs(pairs: Dict[int, int], gt_stage: Dict[int, int], pred_before) -> Counter:
    """Compare ordering over lines matched to distinct ground-truth requirements.

    `pred_before(j1, j2)` is True when the arm orders line j1 before line j2.
    """
    c = Counter()
    lines = sorted(pairs)
    for x, j1 in enumerate(lines):
        for j2 in lines[x + 1:]:
            g1, g2 = gt_stage[pairs[j1]], gt_stage[pairs[j2]]
            p12, p21 = pred_before(j1, j2), pred_before(j2, j1)
            if g1 != g2:
                c["gt_ordered"] += 1
                right = p12 if g1 < g2 else p21
                wrong = p21 if g1 < g2 else p12
                c["recovered"] += int(right)
                c["reversed"] += int(wrong)
                c["missed"] += int(not right and not wrong)
            else:
                c["gt_same_stage"] += 1
                c["spurious"] += int(p12 or p21)
            if p12 or p21:
                c["pred_ordered"] += 1
                c["pred_correct"] += int(g1 != g2 and (p12 if g1 < g2 else p21))
    return c


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=R1 / "examples/merged/rows.jsonl")
    ap.add_argument("--graphs", type=Path, default=R1 / "examples/SNAP_ALL/detailed_traces")
    ap.add_argument("--vocab-graphs", type=Path, nargs="+", default=[
        ROOT / "outputs/record/rec_R_iir/results/rec_R_iir.json.gz/detailed_traces",
        ROOT / "outputs/record/rec_R_plain/results/rec_R_plain.json.gz/detailed_traces"])
    ap.add_argument("--vocab-csv", type=Path, default=ROOT / "data/hssd-hab/metadata/object_categories_filtered.csv")
    ap.add_argument("--inside-prior", type=Path, default=ROOT / "results/partnr_inside_prior_train_R_only.json")
    ap.add_argument("--operators", default="results/partnr_operators_iir1.json")
    ap.add_argument("--arms", nargs="+", default=["typedR", "typedRS", "typedRST"])
    ap.add_argument("--split", default="train_mini")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import csv
    from skill_memory_v2.partnr_memory import PartnrSkillMemory

    # The projection flags of examples/run.sh, the run that produced these answers.
    ti.INSIDE_PRIOR = json.loads(args.inside_prior.read_text())
    ti.KEEP_DUPLICATES = ti.SNAP_FURNITURE = ti.ROOM_SNAP = ti.COLLAPSE_PLACES = ti.SNAP_FIRST = True
    effects = PartnrSkillMemory.load(str(ROOT / args.operators)).effects()
    kinds_cache = args.out / "kinds.json"
    if kinds_cache.exists():
        kinds = json.loads(kinds_cache.read_text())
    else:
        kinds = {k for d in args.vocab_graphs for k in ti.vocabulary(d)}
        with open(args.vocab_csv) as handle:
            kinds |= {ti.category(r["clean_category"]) for r in csv.DictReader(handle)
                      if r.get("clean_category", "").strip()}
        kinds = sorted(kinds)
        kinds_cache.write_text(json.dumps(kinds))

    with gzip.open(ti.DATASET / f"{args.split}.json.gz") as handle:
        episodes = {str(e["episode_id"]): e for e in json.load(handle)["episodes"]}
    rows = [json.loads(line) for line in open(args.rows)]
    rows = [r for r in rows if r["status"] == "ok"]

    totals: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    details = []
    selfcheck = Counter()
    for row in rows:
        key = row["episode"]
        episode = episodes[key.split("_")[1]]
        etype = classify(episode)
        temporal = "T" in etype
        group = "T" if temporal else "nonT"
        graph = ti.first_graph(args.graphs / f"detailed_trace-{key}.pkl")
        scene = ti.Scene(graph)
        gt = ti.ground_truth(episode, scene)
        level = dag_levels(episode)
        gt_stage = {i: level.get(r["proposition"], 0) for i, r in enumerate(gt)}
        instruction = episode["instruction"]
        record = {"episode": key, "type": etype, "instruction": instruction, "gt_stages": list(gt_stage.values()),
                  "gt_lines": [f"{ti.category(r['subject'])} | {r['key']} | {r['target']}" for r in gt], "arms": {}}

        gt_pairs_beside = gt_beside(episode, scene)
        spatial = "S" in etype.split("_")
        # oracle: the ground-truth requirements are the lines, in proposition order
        oracle_kinds = [ti.category(r["subject"]) for r in gt]
        oracle_lines = [{"key": r["key"], "subject": k, "target": r["target"]} for r, k in zip(gt, oracle_kinds)]
        bc = score_beside(gt_pairs_beside, [(r["subject"], r["target"]) for r in beside_requirements(instruction, oracle_lines)], spatial)
        totals["oracle:beside"]["S" if spatial else "nonS"] += bc
        totals["oracle:beside"][etype] += bc
        stages, parts = assign_stages(instruction, oracle_kinds, [r["target"] for r in gt])
        record["clauses"] = [(p["lead"], p.get("stage"), p["text"]) for p in parts]
        record["arms"]["oracle"] = stages
        identity = {i: i for i in range(len(gt))}
        c = score_pairs(identity, gt_stage, lambda a, b, s=stages: s[a] < s[b])
        c["episodes"] += 1
        c["exact"] += int(c["reversed"] == 0 and c["missed"] == 0 and c["spurious"] == 0)
        totals["oracle"][group] += c
        totals["oracle"][etype] += c

        for arm in args.arms:
            stored = row["arms"][arm]
            raw = ti.parse_typed(stored["answer"], effects)
            projected, _ = ti.project(raw, scene, kinds, effects, instruction)
            pairs = match(gt, projected, scene)
            selfcheck[f"{arm} matched"] += len(pairs)
            selfcheck[f"{arm} stored typed_proj.matched"] += stored["typed_proj"]["matched"]
            line_kinds = [ti.category(item["subject"]) for item in projected]
            stages, _ = assign_stages(instruction, line_kinds, [item["target"] for item in projected])
            beside = beside_requirements(instruction, projected)
            bc = score_beside(gt_pairs_beside, [(r["subject"], r["target"]) for r in beside], spatial)
            totals[f"{arm}:beside"]["S" if spatial else "nonS"] += bc
            totals[f"{arm}:beside"][etype] += bc
            reach = closure(same_subject_order(projected))
            # What the planner will actually guard on: the closure of its after-lists must be
            # exactly "stage a < stage b", or the sim cell is not running what was measured.
            planned = as_requirements(projected, instruction, stages=True)
            planner_edges = closure([r["after_propositions"] for r in planned])
            want_edges = {(a, b) for a in range(len(stages)) for b in range(len(stages)) if stages[a] < stages[b]}
            selfcheck[f"{arm} planner_dag_mismatch"] += int(planner_edges != want_edges)
            # An intermediate stop: an earlier place of an object that the task moves again.
            # A missing one cannot be ordered at all, whatever the rule does.
            inter = {i for i, r in enumerate(gt)
                     if any(o["subject"] == r["subject"] and gt_stage[j] > gt_stage[i] for j, o in enumerate(gt))}
            if inter:
                totals[f"{arm}:stops"][group]["intermediate_gt"] += len(inter)
                totals[f"{arm}:stops"][group]["intermediate_matched"] += len(inter & set(pairs.values()))
                totals[f"{arm}:stops"][etype]["intermediate_gt"] += len(inter)
                totals[f"{arm}:stops"][etype]["intermediate_matched"] += len(inter & set(pairs.values()))
            for name, before in (("stage_rule", lambda a, b, s=stages: s[a] < s[b]),
                                 ("same_subject", lambda a, b, e=reach: (a, b) in e)):
                c = score_pairs(pairs, gt_stage, before)
                c["episodes"] += 1
                c["exact"] += int(c["reversed"] == 0 and c["missed"] == 0 and c["spurious"] == 0)
                # Order edges over all lines, matched or not, on episodes that have no order.
                if not temporal:
                    n = len(projected)
                    c["nonT_any_edge_episode"] += int(any(before(a, b) for a in range(n) for b in range(n) if a != b))
                totals[f"{arm}:{name}"][group] += c
                totals[f"{arm}:{name}"][etype] += c
            record["arms"][arm] = {"lines": [f"{i['subject']} | {i['key']} | {i['target']}" for i in projected],
                                   "stages": stages, "matched": {str(j): i for j, i in pairs.items()}}
        details.append(record)

    def rates(c: Counter) -> Dict[str, Any]:
        out = dict(c)
        if c["gt_ordered"]:
            out["pair_recall"] = round(c["recovered"] / c["gt_ordered"], 4)
        if c["pred_ordered"]:
            out["pair_precision"] = round(c["pred_correct"] / c["pred_ordered"], 4)
        if c["episodes"]:
            out["exact_rate"] = round(c["exact"] / c["episodes"], 4)
        return out

    summary = {"split": args.split, "rows": str(args.rows), "episodes": len(rows),
               "selfcheck": dict(selfcheck),
               "results": {arm: {g: rates(c) for g, c in sorted(groups.items())} for arm, groups in totals.items()}}
    # disk before print
    with open(args.out / "episodes.jsonl", "w") as handle:
        for record in details:
            handle.write(json.dumps(record) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"{args.split}: {len(rows)} episodes; selfcheck {dict(selfcheck)}")
    for arm, groups in summary["results"].items():
        if arm.endswith(":beside") or arm.endswith(":stops"):
            for g in ("S", "nonS", "T", "R_S", "R_S_T"):
                if g in groups:
                    print(f"  {arm:24s} {g:6s} {groups[g]}")
    for arm, groups in summary["results"].items():
        for g in ("T", "nonT", "R_T", "R_S_T"):
            if g in groups:
                x = groups[g]
                print(f"  {arm:24s} {g:6s} n={x.get('episodes',0):3d} gt_ordered={x.get('gt_ordered',0):5d} "
                      f"recall={x.get('pair_recall','-')} precision={x.get('pair_precision','-')} "
                      f"reversed={x.get('reversed',0)} spurious={x.get('spurious',0)} exact={x.get('exact_rate','-')} "
                      f"nonT_edge_eps={x.get('nonT_any_edge_episode','-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
