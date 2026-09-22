"""Part B, B0 + B1 (zero model calls). Run on the remote box.

B0: the 197 H_R episode ids and instructions, the heuristic trajectories the archived H_R library was built from,
    the archived library manifest (file hashes, counts, e_src breakdown).
B1: split by sha256(episode_id) mod 2, where episode_id is the dataset's own string field (e.g. "433"), encoded
    UTF-8, digest read as a big-endian integer. Remainder 0 -> fold A, 1 -> fold B.
    Instruction-text check: if one exact instruction string occurs in both folds, every copy is moved to the fold
    of the copy with the smallest integer id; every move is logged.
Also writes per-fold id files, the per-fold build inputs (heuristic log rows), and the predicted proposal-call count
of the archived builder (--include-failed --patch-failed): 3 calls per successful episode (2 individual + 1
cooperation extraction), 1 per failed episode (patch_failed_episode). This formula reproduces the 214 calls of the
09-22 route-Q build on 98 episodes (58 successful, 40 failed).
Usage: b01_inventory_split.py <out_dir>
"""
import collections, csv, gzip, hashlib, json, os, sys

P = "/mnt/pfs/devs/pn5wp/shishuqing"
REPO = f"{P}/partnr-isambard-C"
DATA = f"{REPO}/task_classification_datasets/heterogeneous+rerange.json.gz"
HEUR = f"{P}/iclr_master_C_build/src/episode_result_log.csv"   # copy of ~/restore_isambard/heuristic_dataset/2025-12-30_21-10-23-heterogeneous+rerange.json/results
TRACES = f"{P}/iclr_master_C_build/src/traces"
LIB = f"{REPO}/data/hierarchical_skill_memory/hierarchical_heterogeneous_rerange"
out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


d = json.load(gzip.open(DATA))
eps = d["episodes"]
assert len(eps) == 197
assert all(isinstance(e["episode_id"], str) for e in eps)
heur = {r["episode_id"]: r for r in csv.DictReader(open(HEUR))}


def has_trace(eid):
    return all(os.path.exists(f"{TRACES}/{a}/trace-episode_{eid}_0-{a}.txt") for a in (0, 1))


fold = {}
rows = []
for e in eps:
    eid = e["episode_id"]
    r = int(hashlib.sha256(eid.encode("utf-8")).hexdigest(), 16) % 2
    fold[eid] = "A" if r == 0 else "B"

# instruction-text check and moves
by_text = collections.defaultdict(list)
for e in eps:
    by_text[e["instruction"]].append(e["episode_id"])
moves = []
dup_groups = []
for text, ids in by_text.items():
    if len(ids) < 2:
        continue
    folds = {fold[i] for i in ids}
    dup_groups.append({"ids": sorted(ids, key=int), "folds_before": sorted(folds)})
    if len(folds) > 1:
        anchor = min(ids, key=int)
        for i in ids:
            if fold[i] != fold[anchor]:
                moves.append({"episode_id": i, "from": fold[i], "to": fold[anchor], "anchor": anchor,
                              "instruction": text})
                fold[i] = fold[anchor]
for text, ids in by_text.items():
    assert len({fold[i] for i in ids}) == 1

with open(f"{out}/hr_episodes.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["episode_id", "fold", "sha256_mod2", "in_heuristic_log", "has_traces", "heuristic_success",
                "heuristic_percent_complete", "instruction"])
    for e in sorted(eps, key=lambda e: int(e["episode_id"])):
        eid = e["episode_id"]
        h = heur.get(eid)
        w.writerow([eid, fold[eid], int(hashlib.sha256(eid.encode()).hexdigest(), 16) % 2, h is not None,
                    has_trace(eid), h["task_state_success"] if h else "", h["task_percent_complete"] if h else "",
                    e["instruction"]])

ids = {k: sorted([i for i in fold if fold[i] == k], key=int) for k in ("A", "B")}
calls = {}
for k in ("A", "B"):
    build = [i for i in ids[k] if i in heur and has_trace(i)]
    succ = sum(float(heur[i]["task_state_success"]) >= 1.0 for i in build)
    calls[k] = {"episodes": len(ids[k]), "build_inputs": len(build), "successful": succ,
                "failed": len(build) - succ, "predicted_calls": 3 * succ + (len(build) - succ),
                "missing_from_heuristic_log": [i for i in ids[k] if i not in heur],
                "missing_traces": [i for i in ids[k] if i in heur and not has_trace(i)]}
    open(f"{out}/fold{k}_ids.txt", "w").write("\n".join(ids[k]) + "\n")
    with open(f"{out}/fold{k}_build_log.csv", "w", newline="") as f:   # builder input, source order kept
        src = list(csv.DictReader(open(HEUR)))
        w = csv.DictWriter(f, fieldnames=list(src[0]))
        w.writeheader()
        w.writerows([r for r in src if r["episode_id"] in set(build)])

split = {
    "rule": "fold = 'A' if int(sha256(episode_id.encode('utf-8')).hexdigest(), 16) % 2 == 0 else 'B'",
    "episode_id_form": "the dataset's episode_id field, a decimal string such as '433' (no prefix, no scene id)",
    "dataset": DATA, "dataset_sha256": sha(DATA),
    "n": len(fold), "fold_sizes": {k: len(v) for k, v in ids.items()},
    "moves": moves, "duplicate_instruction_groups": dup_groups,
    "folds": ids,
    "evaluation": {"A": "evaluated with the library built from fold B", "B": "evaluated with the library built from fold A"},
}
json.dump(split, open(f"{out}/split.json", "w"), indent=1)


# archived library manifest
def lib_manifest(lib):
    m = {"path": lib, "files": {}}
    for fn in sorted(os.listdir(lib)):
        p = f"{lib}/{fn}"
        if os.path.isfile(p):
            m["files"][fn] = sha(p)
    esrc = collections.Counter()
    counts = {}
    for f in ("L_ind_skills", "L_coop_skills"):
        L = json.load(gzip.open(f"{lib}/{f}.json.gz"))
        counts[f] = {"skills": len(L), "instances": sum(len(s["instances"]) for s in L.values())}
        for s in L.values():
            for i in s["instances"]:
                x = i["e_src"]
                esrc["episode_<id>_partial/patched" if x.startswith("episode_") else x] += 1
    em = json.load(gzip.open(f"{lib}/episodic_memory.json.gz"))
    m.update(counts=counts, e_src=dict(esrc), episodic_memory_episodes=len(em),
             episodic_memory_patched=sum(bool(v.get("patched")) for v in em.values()),
             source_episode_ids=sorted(em, key=int),
             summary=json.load(open(f"{lib}/memory_summary.json")) if os.path.exists(f"{lib}/memory_summary.json") else None)
    return m


man = lib_manifest(LIB)
man["summary"] = {k: v for k, v in (man["summary"] or {}).items() if not k.endswith("_names")}
man["heuristic_log"] = {"path": HEUR, "sha256": sha(HEUR), "rows": len(heur)}
man["builder"] = {"path": f"{REPO}/our_method/build_hierarchical_skill_memory.py",
                  "sha256": sha(f"{REPO}/our_method/build_hierarchical_skill_memory.py"),
                  "extractor_sha256": sha(f"{REPO}/our_method/llm_skill_extractor.py"),
                  "archived_command": "rebuild_all_memories.sh H_R block: --include-failed --use-llm --use-api --patch-failed, "
                                      "meta-llama/Llama-3.3-70B-Instruct, vLLM TP 4, bf16, max-model-len 8192",
                  "gate": "none: the builder has no execution check and no admission criteria; every LLM-extracted "
                          "skill is merged by name (skill_key) and saved"}
em_ids = set(man["source_episode_ids"])
man["coverage_of_197"] = {"in_library": len(em_ids & set(fold)), "not_in_library": sorted(set(fold) - em_ids, key=int)}
json.dump(man, open(f"{out}/archived_hr_library_manifest.json", "w"), indent=1)
json.dump(calls, open(f"{out}/b2_predicted_calls.json", "w"), indent=1)
print(json.dumps({"fold_sizes": split["fold_sizes"], "moves": len(moves), "dup_groups": len(dup_groups),
                  "calls": {k: v["predicted_calls"] for k, v in calls.items()},
                  "build_inputs": {k: v["build_inputs"] for k, v in calls.items()},
                  "lib_counts": man["counts"], "coverage": man["coverage_of_197"]["in_library"]}))
