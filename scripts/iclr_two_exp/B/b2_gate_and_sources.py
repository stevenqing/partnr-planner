"""Part B2 checks on the two fold libraries (zero model calls). Run on the remote box after b2_build.sh.

Gate B2 (coverage): each fold library must hold >= 1 individual and >= 1 cooperation skill for Pickup, Place and
every state-changing action the H_R goals require. Required actions come from the H_R dataset's evaluation
propositions: is_clean -> Clean, is_filled -> Fill, is_powered_on -> PowerOn, is_powered_off -> PowerOff
(plus Pick and Place). A skill covers action X when one of its instances has an `action_sequence` entry
`X[...]` (strict). The loose count also accepts X's verb in the skill name/description/demo text; only strict
decides the gate. The archived H_R library is scored the same way as a reference.

Sources: every instance's source episode (provenance.json from build_with_provenance.py), every episodic-memory
key, and every explicit id in e_src / context.original_episode must lie in the library's own fold; the library
used to evaluate fold X is the other fold's. Writes gate_b2.json, sources_check.json and the frozen manifests
hr_foldA_library.json / hr_foldB_library.json, and installs each library at
<repo>/data/hierarchical_skill_memory_iclr2exp/lib_<k>/hierarchical_heterogeneous_rerange/ (refuses to overwrite).
Usage: b2_gate_and_sources.py <build_dir> <results_B_dir> <folds: A|B|AB>
Outputs are per fold (gate_b2_fold<k>.json, sources_check_fold<k>.json); a library is installed only if its sources check passes.
"""
import collections, gzip, hashlib, json, os, re, shutil, sys

W, RES, FOLDS = sys.argv[1], sys.argv[2], sys.argv[3]
P = "/mnt/pfs/devs/pn5wp/shishuqing"
REPO = f"{P}/partnr-isambard-C"
ARCH = f"{REPO}/data/hierarchical_skill_memory/hierarchical_heterogeneous_rerange"
split = json.load(open(f"{RES}/split.json"))
folds = {k: set(v) for k, v in split["folds"].items()}
PRED2ACT = {"is_clean": "Clean", "is_filled": "Fill", "is_powered_on": "PowerOn", "is_powered_off": "PowerOff"}
WORDS = {"Pick": r"\bpick", "Place": r"\bplace|\bput\b", "Clean": r"\bclean|\bwash|\bdust|\bwipe",
         "Fill": r"\bfill", "PowerOn": r"power.?on|turn.?on|switch.?on", "PowerOff": r"power.?off|turn.?off|switch.?off"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def required_actions():
    d = json.load(gzip.open(split["dataset"]))
    fns = collections.Counter(p["function_name"] for e in d["episodes"] for p in e["evaluation_propositions"])
    return ["Pick", "Place"] + [a for f, a in PRED2ACT.items() if fns.get(f)], dict(fns)


def load(lib):
    return {f: json.load(gzip.open(f"{lib}/{f}.json.gz")) for f in ("L_ind_skills", "L_coop_skills")}


def coverage(lib, acts):
    L = load(lib)
    out = {}
    for f, kind in (("L_ind_skills", "individual"), ("L_coop_skills", "cooperation")):
        for a in acts:
            strict = [k for k, s in L[f].items()
                      if any(re.match(rf"^\s*{a}\[", x) for i in s["instances"]
                             for x in (i["context"].get("action_sequence") or []))]
            loose = [k for k, s in L[f].items()
                     if k in strict or re.search(WORDS[a], " ".join([s["name"], s.get("description", "")]
                                                                     + [i["demo"] for i in s["instances"]]), re.I)]
            out[f"{kind}:{a}"] = {"strict": len(strict), "loose": len(loose), "strict_examples": strict[:3]}
    return out, {f: {"skills": len(v), "instances": sum(len(s["instances"]) for s in v.values())} for f, v in L.items()}


acts, fns = required_actions()
gate = {"required_actions": acts, "proposition_counts": fns, "rule": "strict count >= 1 for every kind:action"}
ref_cov, ref_counts = coverage(ARCH, acts)
gate["archived_reference"] = {"counts": ref_counts, "coverage": ref_cov,
                              "passes": all(v["strict"] >= 1 for v in ref_cov.values())}
src_report = {}
for k in FOLDS:
    lib = f"{W}/lib_{k}"
    cov, counts = coverage(lib, acts)
    miss = [key for key, v in cov.items() if v["strict"] < 1]
    gate[f"fold{k}"] = {"counts": counts, "coverage": cov, "missing": miss, "passes": not miss}
    # sources
    L = load(lib)
    prov = json.load(open(f"{lib}/provenance.json"))
    em = json.load(gzip.open(f"{lib}/episodic_memory.json.gz"))
    srcs, explicit, bad = collections.Counter(), collections.Counter(), []
    for f, lb in L.items():
        for sk, s in lb.items():
            pv = prov[f][sk]
            assert len(pv) == len(s["instances"]), (f, sk)
            for j, (inst, ep) in enumerate(zip(s["instances"], pv)):
                srcs[ep] += 1
                m = re.match(r"episode_(\w+?)_(partial|patched)$", inst["e_src"])
                ex = m.group(1) if m else inst["context"].get("original_episode")
                if ex is not None:
                    explicit[str(ex)] += 1
                    if str(ex) != ep:
                        bad.append((f, sk, j, ep, ex))
    ids = set(srcs) | set(em) | set(explicit)
    other = "B" if k == "A" else "A"
    src_report[f"lib_{k}"] = {
        "instances": sum(srcs.values()), "source_episodes": len(set(srcs)), "episodic_memory": len(em),
        "all_in_own_fold": ids <= folds[k], "outside_own_fold": sorted(ids - folds[k], key=int),
        "overlap_with_evaluated_fold": sorted(ids & folds[other], key=int),
        "explicit_id_mismatch": bad[:10], "evaluates_fold": other,
        "own_fold_episodes_without_instances": sorted(folds[k] - set(srcs), key=int)}
    man = {"library_dir": lib, "built_from_fold": k, "evaluates_fold": other,
           "files": {fn: sha(f"{lib}/{fn}") for fn in sorted(os.listdir(lib)) if os.path.isfile(f"{lib}/{fn}")},
           "counts": counts, "source_episode_ids": sorted(set(srcs), key=int),
           "builder": "our_method/build_hierarchical_skill_memory.py via build_with_provenance.py, "
                      "--include-failed --use-llm --use-api --patch-failed, Llama-3.3-70B-Instruct",
           "gate": "none (see deviations B-D2)"}
    ok = src_report[f"lib_{k}"]["all_in_own_fold"] and not src_report[f"lib_{k}"]["overlap_with_evaluated_fold"] \
        and not src_report[f"lib_{k}"]["explicit_id_mismatch"]
    src_report[f"lib_{k}"]["sources_ok"] = ok
    dst = f"{REPO}/data/hierarchical_skill_memory_iclr2exp/lib_{k}/hierarchical_heterogeneous_rerange"
    if ok and not os.path.exists(dst):
        shutil.copytree(lib, dst)
    man["installed_at"] = dst if ok else None
    man["installed_identical"] = ok and all(sha(f"{dst}/{fn}") == h for fn, h in man["files"].items())
    json.dump(man, open(f"{RES}/hr_fold{k}_library.json", "w"), indent=1)
    json.dump({kk: v for kk, v in gate.items() if not kk.startswith("fold") or kk == f"fold{k}"},
              open(f"{RES}/gate_b2_fold{k}.json", "w"), indent=1)
    json.dump(src_report[f"lib_{k}"], open(f"{RES}/sources_check_fold{k}.json", "w"), indent=1)
    print(json.dumps({"fold": k, "gate_passes": gate[f"fold{k}"]["passes"], "missing": gate[f"fold{k}"]["missing"],
                      "counts": gate[f"fold{k}"]["counts"], "sources_ok": ok,
                      "installed_identical": man["installed_identical"]}))
print(json.dumps({"archived_reference_passes": gate["archived_reference"]["passes"]}))
