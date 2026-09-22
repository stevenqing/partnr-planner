"""C.1.1 route P' (reconstructed provenance): keep an instance only when every
candidate source episode is in the build half. synchronization_cooperation has
exactly one instance per standard-path episode, so its j-th instance is the j-th
non-patched episode in processing order (exact). Writes the held-out bank."""
import collections, gzip, json, os, sys
bank, attr, outdir, resdir = sys.argv[1:5]
rows = json.load(open(attr))
ev = set(open(f"{resdir}/hr_eval_ids.txt").read().split())
bd = set(open(f"{resdir}/hr_build_ids.txt").read().split())
L = lambda f: json.load(gzip.open(f"{bank}/{f}.json.gz"))
ind, coop, em = L("L_ind_skills"), L("L_coop_skills"), L("episodic_memory")
std = [k for k, v in em.items() if not v.get("patched")]
sync = coop["synchronization_cooperation"]["instances"]
assert len(sync) == len(std), (len(sync), len(std))
side = {}
for r in rows:
    cs = set(r["cands"])
    if r["kind"] == "cooperation" and r["skill"] == "synchronization_cooperation":
        cs = {std[r["inst"]]}; r["cands"] = sorted(cs); r["how"] = "order_exact"
    s = "undetermined" if not cs else "eval" if cs <= ev else "build" if cs <= bd else "mixed"
    r["side"] = s; side[(r["kind"], r["skill"], r["inst"])] = s
json.dump(rows, open(f"{resdir}/attribution_sided.json", "w"))
stats = collections.Counter(); first_side = collections.Counter()
out = {}
for kind, lib, fname in (("individual", ind, "L_ind_skills"), ("cooperation", coop, "L_coop_skills")):
    new = {}
    for k, s in lib.items():
        keep = [i for j, i in enumerate(s["instances"]) if side[(kind, k, j)] == "build"]
        stats[(kind, "skills_before")] += 1; stats[(kind, "inst_before")] += len(s["instances"])
        if keep:
            s2 = dict(s); s2["instances"] = keep; s2["instance_count"] = len(keep); new[k] = s2
            stats[(kind, "skills_after")] += 1; stats[(kind, "inst_after")] += len(keep)
            # name/description are set by the first episode that produced the skill
            f = side[(kind, k, 0)]
            first_side[(kind, f)] += 1
    out[fname] = new
os.makedirs(outdir, exist_ok=True)
for fname, lib in out.items():
    with gzip.open(f"{outdir}/{fname}.json.gz", "wt") as f: json.dump(lib, f)
em2 = {k: v for k, v in em.items() if k in bd}
with gzip.open(f"{outdir}/episodic_memory.json.gz", "wt") as f: json.dump(em2, f)
summ = dict(total_individual_skills=len(out["L_ind_skills"]), total_cooperation_skills=len(out["L_coop_skills"]),
            total_episodes=len(em2), individual_skill_names=sorted(out["L_ind_skills"]),
            cooperation_skill_names=sorted(out["L_coop_skills"]), built_from="H_R bank minus instances not provably from build half (iclr_master C.1.1 route P')")
json.dump(summ, open(f"{outdir}/memory_summary.json", "w"), indent=1)
res = {f"{a}.{b}": v for (a, b), v in sorted(stats.items())}
res.update({f"retained_first_instance_side.{a}.{b}": v for (a, b), v in sorted(first_side.items())})
res["instance_side"] = {f"{a}.{b}": v for (a, b), v in sorted(collections.Counter((r["kind"], r["side"]) for r in rows).items())}
json.dump(res, open(f"{resdir}/heldout_library_stats.json", "w"), indent=1)
print(json.dumps(res, indent=1))
