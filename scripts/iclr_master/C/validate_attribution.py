"""C.1.1 check: run the content matcher on instances whose source episode is recorded, hiding the record,
and count how often its candidate set contains the true episode and whether it would put it on the wrong side."""
import gzip, json, re, sys
sys.path.insert(0, "scripts/iclr_master/C")
import attribute_instances as A
bank, resdir = sys.argv[1:3]
ind, coop, em = A.load(bank)
order = list(em); idx = A.ep_index(em)
for e in idx.values(): e["patched"] = False  # let the matcher see every episode
ev = set(open(f"{resdir}/hr_eval_ids.txt").read().split()); bd = set(open(f"{resdir}/hr_build_ids.txt").read().split())
n = hit = uniq = uniq_ok = wrong_side_build = 0
for lib in (ind, coop):
    for s in lib.values():
        for inst in s["instances"]:
            m = re.match(r"episode_(\w+?)_(partial|patched)$", inst["e_src"])
            if not m: continue
            true = m.group(1)
            fake = dict(inst); fake["e_src"] = "agent_0_llm" if "partial" in inst["e_src"] else "multi_agent_llm"
            fake["context"] = {k: v for k, v in inst["context"].items() if k != "original_episode"}
            c, how = A.candidates(fake, None, idx, order)
            if not c: continue
            n += 1; hit += true in c
            if set(c) <= bd and true in ev: wrong_side_build += 1
            if len(c) == 1: uniq += 1; uniq_ok += c[0] == true
print(json.dumps(dict(matched_with_candidates=n, true_in_candidates=hit, unique=uniq, unique_correct=uniq_ok,
                      eval_source_labelled_build=wrong_side_build)))
