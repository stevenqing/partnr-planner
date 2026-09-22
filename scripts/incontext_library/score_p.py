#!/usr/bin/env python3
"""Score the in-context-library arm (P) and compare it row by row with the Figure 2 arms.

Zero LLM calls. For each CG split:
- P scored two ways: official strict (task_score from the runner) and JSON-tolerant
  (viki_report_matrix.tolerant, the parser Figure 2 uses for baselines);
- G-Memory, zero-shot and ours read from results/paper_viki_iclr2027/rows/figure2/*/72B,
  keyed by example_id == parquet row index; G-Memory and zero-shot are re-scored with the
  same tolerant() from their archived amendment10 responses as a check on the pairing;
- exact McNemar for P vs G-Memory, P vs zero-shot, ours vs P, on 297 rows and on 295 rows
  (second occurrence of the two repeated task_ids dropped);
- P-canon / G-canon (imaged split only, descriptive): every action argument snapped to the
  scene assets plus library places with the library's vocabulary.canonical, then tolerant().
Writes per_row/<split>.csv, work/B/score_summary.json, prompts/<split>_first5.txt.
"""
import csv
import hashlib
import json
import sys
from math import comb
from pathlib import Path

REPO = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]
import pandas as pd  # noqa: E402
import viki_amendment10_run as run10  # noqa: E402
from viki_amendment5 import BENCHMARK_ROOT  # noqa: E402
from viki_report_matrix import tolerant  # noqa: E402
from viki_amendment9_diag102 import parse_plan  # noqa: E402
from our_method.skill_memory_v2 import SEED, Simulator, vocabulary  # noqa: E402

OUT = REPO / "results/incontext_library_2026-09-21"
ARCH = REPO / "results/viki_memory_experiments/amendment10"
FIG2 = REPO / "results/paper_viki_iclr2027/rows/figure2"
LIB = REPO / "outputs/v3_memories/memory_all.json"
CANON = {"text": "pure_text", "imaged": "cg_image"}
DUPES = {"4307_8-1", "1426_9-1"}


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def mcnemar(a, b, keys):
    """Exact two-sided; returns (a_only, b_only, p)."""
    a_only = sum(1 for k in keys if a[k] and not b[k])
    b_only = sum(1 for k in keys if b[k] and not a[k])
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    tail = sum(comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2 ** n
    return a_only, b_only, min(1.0, 2 * tail)


def canon_response(response, known):
    parsed = parse_plan(response or "")
    if not isinstance(parsed, list):
        return response
    for step in parsed:
        acts = step.get("actions") if isinstance(step, dict) else None
        if not isinstance(acts, dict):
            continue
        for robot, act in acts.items():
            if isinstance(act, list) and act:
                acts[robot] = [act[0]] + [vocabulary.canonical(x, known) or x if isinstance(x, str) else x
                                          for x in act[1:]]
    return "<answer>\n" + json.dumps(parsed) + "\n</answer>"


def main():
    sim = Simulator(BENCHMARK_ROOT)
    library = json.loads(LIB.read_text())
    lib_places = sorted(set(library["layer3"].get("places", [])))
    summary = {}
    for split in ("text", "imaged"):
        frame = pd.read_parquet(ARCH / f"recombination.{split}.parquet")
        p_path = OUT / "runs" / split / "incontext_library.jsonl"
        p_rows = {r["index"]: r for r in jl(p_path)}
        manifest = run10.manifest_for(split)
        complete = set(p_rows) == set(manifest)
        truth, task_id, known = {}, {}, {}
        for i in manifest:
            gt = run10.native(frame.iloc[i].to_dict())["reward_model"]["ground_truth"]
            truth[i], task_id[i] = gt, gt.get("task_id")
            blind = {k: v for k, v in gt.items() if k != "time_steps"}
            known[i] = sorted(set(sim.metadata(blind, SEED)["assets"]) | set(lib_places))
        keys = sorted(manifest)

        p_off = {i: int(p_rows[i]["task_score"]) for i in keys if i in p_rows}
        p_tol = {i: tolerant(sim, p_rows[i]["response"], truth[i]) for i in keys if i in p_rows}

        fig = {}
        for cond in ("g_memory", "zero_shot", "ours"):
            rows = {int(r["example_id"]): r for r in jl(FIG2 / cond / "qwen2.5-vl-72b-instruct" / f"{CANON[split]}.jsonl")}
            fig[cond] = {i: int(bool(rows[i]["success"])) for i in keys}
        arch = {c: {r["index"]: r for r in jl(ARCH / split / f"{c}.jsonl")} for c in ("gmemory", "zero_shot")}
        g_tol = {i: tolerant(sim, arch["gmemory"][i]["response"], truth[i]) for i in keys}
        z_tol = {i: tolerant(sim, arch["zero_shot"][i]["response"], truth[i]) for i in keys}
        pairing_check = {"g_memory_tolerant_recomputed_equals_fig2": sum(g_tol[i] == fig["g_memory"][i] for i in keys),
                         "zero_shot_tolerant_recomputed_equals_fig2": sum(z_tol[i] == fig["zero_shot"][i] for i in keys),
                         "rows": len(keys)}
        g_off = {i: int(arch["gmemory"][i]["task_score"]) for i in keys}
        z_off = {i: int(arch["zero_shot"][i]["task_score"]) for i in keys}

        seen, k295 = set(), []
        for i in keys:
            if task_id[i] in DUPES and task_id[i] in seen:
                continue
            seen.add(task_id[i])
            k295.append(i)

        def tests(ks):
            out = {}
            for name, a, b in (("P_vs_GMemory", p_tol, fig["g_memory"]), ("P_vs_zero_shot", p_tol, fig["zero_shot"]),
                               ("ours_vs_P", fig["ours"], p_tol)):
                a_only, b_only, p = mcnemar(a, b, ks)
                out[name] = {"first_only": a_only, "second_only": b_only, "p": p}
            return out

        cell = {"complete": complete, "rows": len(p_rows), "manifest": len(manifest),
                "P_json_tolerant": [sum(p_tol.values()), len(keys)], "P_official": [sum(p_off.values()), len(keys)],
                "GMemory_json_tolerant": [sum(fig["g_memory"].values()), len(keys)], "GMemory_official": [sum(g_off.values()), len(keys)],
                "zero_shot_json_tolerant": [sum(fig["zero_shot"].values()), len(keys)], "zero_shot_official": [sum(z_off.values()), len(keys)],
                "ours_SOLVED": [sum(fig["ours"].values()), len(keys)],
                "pairing_check": pairing_check,
                "mcnemar_297": tests(keys), "mcnemar_295": tests(k295), "rows_295": len(k295),
                "format_compliance_official": [sum(int(p_rows[i]["format_score"]) for i in keys), len(keys)],
                "truncated_ge_2000_tokens": sum(p_rows[i]["completion_tokens"] >= 2000 for i in keys),
                "prompt_tokens": {"P_mean": sum(p_rows[i]["prompt_tokens"] for i in keys) / len(keys),
                                  "P_max": max(p_rows[i]["prompt_tokens"] for i in keys),
                                  "GMemory_mean": sum(arch["gmemory"][i]["prompt_tokens"] for i in keys) / len(keys),
                                  "GMemory_max": max(arch["gmemory"][i]["prompt_tokens"] for i in keys)},
                "memory_prompt_sha256_distinct_P": sorted({p_rows[i]["memory_prompt_sha256"] for i in keys})}
        if split == "imaged":
            p_can = {i: tolerant(sim, canon_response(p_rows[i]["response"], known[i]), truth[i]) for i in keys}
            g_can = {i: tolerant(sim, canon_response(arch["gmemory"][i]["response"], known[i]), truth[i]) for i in keys}
            cell["descriptive_canon"] = {"P_canon": [sum(p_can.values()), len(keys)], "G_canon": [sum(g_can.values()), len(keys)]}
        summary[split] = cell

        (OUT / "per_row").mkdir(exist_ok=True)
        with open(OUT / "per_row" / f"{split}.csv", "w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["index", "task_id", "P_response", "P_official", "P_json_tolerant",
                        "GMemory_official", "GMemory_json_tolerant", "zero_shot_official", "zero_shot_json_tolerant", "ours_SOLVED"])
            for i in keys:
                w.writerow([i, task_id[i], f"runs/{split}/incontext_library.jsonl#index={i}", p_off.get(i), p_tol.get(i),
                            g_off[i], fig["g_memory"][i], z_off[i], fig["zero_shot"][i], fig["ours"][i]])

        block = (OUT / "prompts/library_block.txt").read_text().rstrip("\n")
        with open(OUT / "prompts" / f"{split}_first5.txt", "w") as handle:
            for i in keys[:5]:
                sample = run10.native(frame.iloc[i].to_dict())
                msgs = run10.build_messages(sample, manifest[i]["partner_prefix"], block, split == "imaged")
                assert run10.messages_sha256(msgs) == p_rows[i]["prompt_sha256"], i
                handle.write(f"===== row {i} task_id {task_id[i]} prompt_sha256 {p_rows[i]['prompt_sha256']} =====\n")
                for m in msgs:
                    content = m["content"]
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if item.get("type") == "text":
                                parts.append(item["text"])
                            else:
                                url = json.dumps(item)
                                parts.append(f"[image omitted, sha256 {hashlib.sha256(url.encode()).hexdigest()}]")
                        content = "\n".join(parts)
                    handle.write(f"--- {m['role']} ---\n{content}\n")
                handle.write("\n")

    (OUT / "work/B").mkdir(parents=True, exist_ok=True)
    (OUT / "work/B/score_summary.json").write_text(json.dumps(summary, indent=1))   # disk before print
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
