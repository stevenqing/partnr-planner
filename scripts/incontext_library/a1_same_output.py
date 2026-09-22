"""A1: do the three RQ2 arms (full, no_trace, no_execution_admission) see the same model output?

Zero LLM calls. Read-only on existing artifacts; writes only the summary JSON.

For every model x split in results/paper_viki_iclr2027/cells.json (experiment == "rq2"):
  * per row index, sha256 of the first-turn response (`raw`) in each arm, and against the
    archived replay source (`raw[-3000:]`, which is what the evaluator stores);
  * whether the row was re-asked in each arm, and sha256 of `raw_reask`;
  * with --reconstruct (needs the remote venv, the simulator and the libraries), rebuild the
    re-ask prompt of every re-asked row in each arm -- the planner's reason code and the
    casting JSON that REASK embeds -- by re-running grounding + ordering + planner.plan
    with that arm's library (no model call), so a re-ask text difference can be attributed
    to a different prompt or to a same-prompt / different-output serving effect.
It also rebuilds the first N first-turn prompts exactly as work_on() does and scans them
for library content (operator bodies, effect keys, layer-2 pattern keys).

Run on remote:
  /root/venvs/partnr/bin/python scripts/incontext_library/a1_same_output.py --reconstruct
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CELLS = ROOT / "results/paper_viki_iclr2027/cells.json"
OUT = ROOT / "results/incontext_library_2026-09-21/work/A1/a1_summary.json"
ARMS = ("full", "no_trace", "no_execution_admission")
SPLIT_ARG = {"id": "id", "ood_single_family": "id", "cg_image": "recombination-imaged",
             "pure_text": "recombination-text"}


def sha(text):
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def rel(p):
    p = str(p)
    prefix = str(ROOT) + "/"
    return p[len(prefix):] if p.startswith(prefix) else p


def load_arm(cell):
    """rows keyed by index, plus {index: library path} used for that row."""
    src = cell["source"]
    rows, libs = {}, {}
    if src["kind"] == "folds":
        for fam, path in src["files"].items():
            for r in read_jsonl(ROOT / path):
                if r["task_name"] == fam:
                    rows[int(r["index"])] = r
                    libs[int(r["index"])] = cell["library_path"][fam]
        return rows, libs, sorted(src["files"].values())
    path = ROOT / src["path"]
    for r in read_jsonl(path):
        rows[int(r["index"])] = r
    lib = cell.get("library_path")
    if cell["canonical_split"] == "ood_single_family":
        # the assembled file's library is per fold; read it from each fold's run.json
        fold_root = path.parent.parent
        for run in sorted(fold_root.glob("fold_*/run.json")):
            meta = json.loads(run.read_text())
            fam = meta["fold_family"]
            for i, r in rows.items():
                if r["task_name"] == fam:
                    libs[i] = meta["library_path"]
    else:
        libs = {i: lib for i in rows}
    return rows, libs, [src["path"]]


def replay_path(cell):
    cmd = cell.get("command")
    if cmd and "--replay" in cmd:
        return rel(cmd[cmd.index("--replay") + 1])
    if cell.get("replay_source"):
        return cell["replay_source"]
    if cell["canonical_split"] == "ood_single_family":
        fold_root = (ROOT / cell["source"]["path"]).parent.parent
        seen = {rel(json.loads(r.read_text())["replay_source"])
                for r in fold_root.glob("fold_*/run.json")}
        return seen.pop() if len(seen) == 1 else sorted(seen)
    return None


class Reconstructor:
    """Rebuild (reason, casting) that REASK embeds, per arm library. No model calls."""

    def __init__(self):
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "scripts"))
        import pandas as pd
        import viki_eval_v2_intent_choice as V
        from our_method.skill_memory_v2 import SEED, SkillMemoryV2, Simulator, planner
        from viki_eval_skill_memory_v2 import visits_of
        from viki_amendment11_goalparse import extract_json
        from habitat_llm.evaluation import viki_bench as bench
        self.V, self.pd, self.SEED, self.planner = V, pd, SEED, planner
        self.SkillMemoryV2, self.visits_of, self.extract_json = SkillMemoryV2, visits_of, extract_json
        self.bench = bench
        self.sim = Simulator(V.BENCHMARK_ROOT)
        self.frames, self.memories = {}, {}

    def frame(self, split):
        if split not in self.frames:
            V = self.V
            if split == "id":
                self.frames[split] = self.pd.read_parquet(
                    V.BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
            else:
                self.frames[split] = self.pd.read_parquet(
                    V.ROOT / "results/viki_memory_experiments/amendment10" / V.SPLITS[split])
        return self.frames[split]

    def memory(self, path):
        if path not in self.memories:
            self.memories[path] = self.SkillMemoryV2.load(ROOT / path)
        return self.memories[path]

    def messages(self, split, index, text_only_check=True):
        V, bench = self.V, self.bench
        sample = bench.to_native(self.frame(split).iloc[index].to_dict())
        if split != "recombination-text":
            messages = V.drop_think_rule(bench.get_messages(sample))
        else:
            messages = [{"role": m["role"], "content": m["content"]}
                        for m in bench.to_native(sample["prompt"])]
        user = next(m for m in reversed(messages) if m["role"] == "user")
        content = user["content"]
        if isinstance(content, list):
            item = next(i for i in content if i["type"] == "text")
            item["text"] = f"{item['text']}\n\n{V.INSTRUCTION}"
        else:
            user["content"] = f"{content}\n\n{V.INSTRUCTION}"
        return messages

    def reask_inputs(self, split, index, text, lib_path):
        V, bench = self.V, self.bench
        memory = self.memory(lib_path)
        sample = bench.to_native(self.frame(split).iloc[index].to_dict())
        truth = bench.get_ground_truth(sample)
        blind = {k: v for k, v in truth.items() if k != "time_steps"}
        metadata = self.sim.metadata(blind, self.SEED)
        parsed = self.extract_json(text)
        work = (parsed or {}).get("work") if isinstance(parsed, dict) else None
        if not isinstance(work, list):
            return {"status": "unparseable"}
        scene = sorted(metadata["assets"])
        requirements, crew = [], []
        for item in work:
            req = V.to_requirement(memory, item, scene) if isinstance(item, dict) else None
            if req is None:
                continue
            requirements.append(req)
            crew.append([n for n in (item.get("robots") or []) if n in metadata["agents"]])
        if not requirements:
            return {"status": "no_usable_work"}
        env = self.sim.world(metadata)
        blind["goal_constraints"] = [[r] for r in requirements]
        blind["temporal_constraints"] = memory.order_for(
            requirements, self.visits_of(env, requirements, memory))
        casting = {}
        for r, names in zip(requirements, crew):
            if names:
                casting[self.planner.predicate_key(r)] = names[0]
        plan, reason = self.planner.plan(blind, memory, self.sim, self.SEED, crew=casting)
        return {"status": "reask" if (plan is None and casting) else "no_reask",
                "reason": reason, "casting": json.dumps(casting, sort_keys=True),
                "reask_user_turn": V.REASK % (reason or "no feasible plan",
                                              json.dumps(casting, sort_keys=True))}


def library_signatures(path):
    d = json.loads((ROOT / path).read_text())
    sigs = set()
    for op in d["layer1"]["operators"]:
        body = op.get("body") or []
        steps = [" ".join(str(x) for x in s) for s in body]
        if steps:
            sigs.add(" ".join(steps))            # whole demo body
            sigs.add(steps[0] + " " + steps[1] if len(steps) > 1 else steps[0])
        eff = op.get("effect") or {}
        if eff.get("key"):
            sigs.add(str(eff["key"]))
        for s in body:
            if any(str(x).startswith("?") for x in s):
                sigs.add(" ".join(str(x) for x in s))   # e.g. "Grasp ?x"
    for rule in d["layer2"].get("rules", []):
        sigs.update(k for k in (rule.get("pattern") or {}))  # a_key, b_visits_a_target...
    for w in ("subject_on_agent", "target_sealed", "runner_types", "precondition", "layer1",
              "layer2", "operator"):
        sigs.add(w)
    return sorted(sigs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reconstruct", action="store_true")
    ap.add_argument("--n-prompts", type=int, default=5)
    args = ap.parse_args()
    cells = [c for c in json.loads(CELLS.read_text())["cells"] if c.get("experiment") == "rq2"]
    by = {}
    for c in cells:
        by.setdefault((c["model_tag"], c["canonical_split"]), {})[c["condition"]] = c
    recon = Reconstructor() if args.reconstruct else None
    summary = {"cells_json": rel(CELLS), "per_cell": [], "llm_calls": 0}
    lib_paths = set()
    for (model, split), arms in sorted(by.items()):
        loaded = {}
        for a in ARMS:
            rows, libs, files = load_arm(arms[a])
            loaded[a] = (rows, libs, files)
            lib_paths.update(set(libs.values()))
        replays = {a: replay_path(arms[a]) for a in ARMS}
        rp = replays["full"]
        replay_raw = {}
        if isinstance(rp, str):
            for r in read_jsonl(ROOT / rp):
                if r.get("raw") is not None:
                    replay_raw[int(r["index"])] = r["raw"][-3000:]
        keys = set(loaded["full"][0]) & set(loaded["no_trace"][0]) & set(loaded["no_execution_admission"][0])
        union = set().union(*(set(loaded[a][0]) for a in ARMS))
        first_same = first_diff = first_eq_replay = 0
        reask = {a: 0 for a in ARMS}
        replayed_flag = {a: 0 for a in ARMS}
        pair = {}
        for a, b in (("full", "no_trace"), ("full", "no_execution_admission"),
                     ("no_trace", "no_execution_admission")):
            pair[f"{a}~{b}"] = {"both_reasked": 0, "same_reask_text": 0,
                                "same_reask_prompt": 0, "same_prompt_diff_text": 0,
                                "diff_prompt_diff_text": 0, "diff_prompt_same_text": 0,
                                "diff_reason_code": 0, "diff_casting": 0,
                                "reask_only_in_one": 0}
        all3_reask = all3_same_text = 0
        diff_examples = []
        rebuilt = {a: {} for a in ARMS}
        recon_mismatch = Counter()
        for i in sorted(keys):
            recs = {a: loaded[a][0][i] for a in ARMS}
            h = {a: sha(recs[a].get("raw")) for a in ARMS}
            if len(set(h.values())) == 1:
                first_same += 1
            else:
                first_diff += 1
                if len(diff_examples) < 5:
                    diff_examples.append({"index": i, "sha": h})
            if i in replay_raw and all(h[a] == sha(replay_raw[i]) for a in ARMS):
                first_eq_replay += 1
            for a in ARMS:
                if recs[a].get("reask"):
                    reask[a] += 1
                if recs[a].get("replayed"):
                    replayed_flag[a] += 1
                if recon and recs[a].get("reask"):
                    ri = recon.reask_inputs(SPLIT_ARG[split], i, recs[a]["raw"], loaded[a][1][i])
                    rebuilt[a][i] = ri
                    if ri["status"] != "reask":
                        recon_mismatch[f"{a}:{ri['status']}"] += 1
            if all(recs[a].get("reask") for a in ARMS):
                all3_reask += 1
                if len({sha(recs[a].get("raw_reask")) for a in ARMS}) == 1:
                    all3_same_text += 1
            for key, p in pair.items():
                a, b = key.split("~")
                ra, rb = bool(recs[a].get("reask")), bool(recs[b].get("reask"))
                if ra != rb:
                    p["reask_only_in_one"] += 1
                if not (ra and rb):
                    continue
                p["both_reasked"] += 1
                same_text = sha(recs[a].get("raw_reask")) == sha(recs[b].get("raw_reask"))
                p["same_reask_text"] += int(same_text)
                if recon:
                    pa, pb = rebuilt[a][i], rebuilt[b][i]
                    same_prompt = pa.get("reask_user_turn") == pb.get("reask_user_turn")
                    p["same_reask_prompt"] += int(same_prompt)
                    p["diff_reason_code"] += int(pa.get("reason") != pb.get("reason"))
                    p["diff_casting"] += int(pa.get("casting") != pb.get("casting"))
                    if same_prompt and not same_text:
                        p["same_prompt_diff_text"] += 1
                    elif not same_prompt and not same_text:
                        p["diff_prompt_diff_text"] += 1
                    elif not same_prompt and same_text:
                        p["diff_prompt_same_text"] += 1
        reason_codes = {a: dict(Counter(v.get("reason") for v in rebuilt[a].values()))
                        for a in ARMS} if recon else None
        summary["per_cell"].append({
            "model": model, "split": split,
            "files": {a: loaded[a][2] for a in ARMS},
            "replay_source": replays,
            "rows": {a: len(loaded[a][0]) for a in ARMS},
            "rows_in_all_three": len(keys), "rows_in_union": len(union),
            "replayed_flag_true": replayed_flag,
            "first_response_identical_all3": first_same,
            "first_response_differs": first_diff,
            "first_response_equals_replay_source_all3": first_eq_replay,
            "first_diff_examples": diff_examples,
            "reasked": reask,
            "reasked_all3": all3_reask, "reask_text_identical_all3": all3_same_text,
            "pairwise_reask": pair,
            "reconstructed_reask_reason_codes": reason_codes,
            "reconstruction_disagrees_with_record": dict(recon_mismatch) if recon else None,
        })
        c = summary["per_cell"][-1]
        print(f"{model:4} {split:18} rows={c['rows_in_all_three']:4} first_same={first_same:4} "
              f"=replay={first_eq_replay:4} reask={reask} all3_reask={all3_reask} "
              f"same_text={all3_same_text}", flush=True)
        if recon:
            print("      pairs:", {k: v for k, v in pair.items()}, "codes:", reason_codes,
                  "mismatch:", dict(recon_mismatch), flush=True)

    # ---- prompt scan -------------------------------------------------------------------
    src = (ROOT / "scripts/viki_eval_v2_intent_choice.py").read_text()
    lines = src.splitlines()
    scan = {"libraries_scanned": sorted(lib_paths)}
    sigs = set()
    for p in sorted(lib_paths):
        if (ROOT / p).is_file():
            sigs.update(library_signatures(p))
    scan["n_signatures"] = len(sigs)
    V_instruction = src.split('INSTRUCTION = """', 1)[1].split('"""', 1)[0]
    V_reask = src.split('REASK = """', 1)[1].split('"""', 1)[0]
    scan["instruction_hits"] = sorted(s for s in sigs if s in V_instruction)
    scan["reask_template_hits"] = sorted(s for s in sigs if s in V_reask)
    scan["memory_refs_before_first_call"] = [
        f"scripts/viki_eval_v2_intent_choice.py:{n}: {l.strip()}"
        for n, l in enumerate(lines, 1) if 224 <= n <= 270 and "memory" in l]
    if recon:
        prompts = []
        for idx in range(args.n_prompts):
            idx_real = sorted(json.loads(l)["index"] for l in
                              recon.V.MANIFEST.read_text().splitlines() if l.strip())[idx]
            msgs = recon.messages("id", idx_real)
            text = json.dumps([{"role": m["role"],
                                "content": m["content"] if isinstance(m["content"], str)
                                else [x for x in m["content"] if x["type"] == "text"]}
                               for m in msgs])
            base = text.replace(json.dumps(recon.V.INSTRUCTION)[1:-1], "")
            hits = sorted(s for s in sigs if s in text)
            hits_outside_instruction = sorted(s for s in sigs if s in base)
            prompts.append({"index": idx_real, "sha256": sha(text), "chars": len(text),
                            "library_signature_hits": hits,
                            "hits_in_benchmark_text": hits_outside_instruction})
        scan["first_prompts"] = prompts
    summary["prompt_scan"] = scan
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1))
    print(json.dumps(scan, indent=1)[:4000])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
