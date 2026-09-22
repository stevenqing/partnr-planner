#!/usr/bin/env python3
"""A2 follow-up: are the CG task_ids found in train the same episodes? Zero LLM calls.

CG rows (amendment10 recombination) inherit task_id from their TEST source row
(`source_row` indexes test.parquet). VIKI-L2 task_ids are not unique across train/test,
so a task_id match needs a content check. Merges the result into a2_summary.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd  # noqa: E402
from viki_amendment8b import native  # noqa: E402
from our_method.skill_memory_v2.build import load_episodes  # noqa: E402

OUT = ROOT / "results/incontext_library_2026-09-21/work/A2"
DATA = ROOT.parent / "VIKI-R/data/VIKI-R/viki/VIKI-L2"
FIELDS = ("description", "init_pos", "goal_constraints", "time_steps", "layout_id", "task_name")


def main():
    tr, te = load_episodes(DATA / "train.parquet"), load_episodes(DATA / "test.parquet")
    frame = pd.read_parquet(ROOT / "results/viki_memory_experiments/amendment10/recombination.text.parquet")
    cg = {}
    for i in range(len(frame)):
        t = native(frame.iloc[i].to_dict())["reward_model"]["ground_truth"]
        cg.setdefault(str(t["task_id"]), t)
    by_id = {}
    for i, t in enumerate(tr):
        if isinstance(t, dict):
            by_id.setdefault(str(t.get("task_id")), []).append(i)
    tr_ids = set(by_id)
    te_ids = {str(t.get("task_id")) for t in te if isinstance(t, dict)}

    def key(x):
        return {f: json.dumps(x.get(f), sort_keys=True) for f in FIELDS}

    rows = []
    for tid in sorted(set(cg) & tr_ids):
        c = cg[tid]; s = te[c["source_row"]]
        for i in by_id[tid]:
            t = tr[i]; kt, ks, kc = key(t), key(s), key(c)
            rows.append({"task_id": tid, "train_index": i,
                         "half": "induction" if i % 2 == 0 else "selfcheck",
                         "workbench_index": i // 2 if i % 2 == 0 else None,
                         "train_task_name": t.get("task_name"),
                         "cg_source_test_row": c["source_row"],
                         "test_source_task_id": str(s.get("task_id")),
                         "train_equals_test_source": {f: kt[f] == ks[f] for f in FIELDS},
                         "train_equals_cg": {f: kt[f] == kc[f] for f in FIELDS},
                         "train_description": t.get("description"),
                         "cg_description": c.get("description")})
    result = {"cg_ids_in_test": len(set(cg) & te_ids), "cg_ids": len(cg),
              "train_distinct_ids": len(tr_ids), "test_distinct_ids": len(te_ids),
              "train_test_shared_ids": len(tr_ids & te_ids),
              "cg_ids_in_train": len(rows), "rows": rows,
              "same_episode_count": sum(all(r["train_equals_cg"].values()) or
                                        all(r["train_equals_test_source"].values()) for r in rows)}
    (OUT / "a2_taskid_collisions.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    summary_path = OUT / "a2_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["taskid_collision_analysis"] = {k: v for k, v in result.items() if k != "rows"}
    summary["taskid_collision_analysis"]["rows_brief"] = [
        {k: r[k] for k in ("task_id", "train_index", "half", "workbench_index", "train_task_name",
                           "cg_source_test_row")} | {"identical_fields_vs_cg": [f for f, v in r["train_equals_cg"].items() if v]}
        for r in rows]
    summary["verdict_rule_applied"] = ("task_id intersection non-zero -> HALTED (literal rule); "
                                       "content check: same_episode_count=%d" % result["same_episode_count"])
    summary["verdict"] = "HALTED" if rows and any(r["half"] == "induction" for r in rows) else summary["verdict"]
    summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(summary["taskid_collision_analysis"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
