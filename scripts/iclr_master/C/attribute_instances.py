"""C.1.1: attribute each H_R-bank instance to its source episode, zero LLM calls.

Explicit ids: e_src 'episode_<id>_partial' / context.original_episode.
Otherwise content matching against episodic_memory (entity ids + actions),
then the builder's append order (instances extend in episode processing order).
"""
import argparse, collections, gzip, json, re, sys

ENT = re.compile(r"^[a-z][a-z_]*_\d+$")

def ents_of_action(a):
    m = re.match(r"^\w+\[(.*)\]$", a.strip())
    if not m: return set()
    return {x.strip() for x in m.group(1).split(",") if ENT.match(x.strip())}

def load(bank):
    L = lambda f: json.load(gzip.open(f"{bank}/{f}.json.gz"))
    return L("L_ind_skills"), L("L_coop_skills"), L("episodic_memory")

def ep_index(em):
    out = {}
    for eid, ep in em.items():
        es = set(ep["env_state"].get("furniture", {})) | set(ep["env_state"].get("objects", {}))
        acts = {0: set(), 1: set(), None: set()}
        for a in ep["actions"]:
            ra = (a.get("raw_action") or "").replace(" ", "")
            acts[a["agent_id"]].add(ra); acts[None].add(ra)
            es |= ents_of_action(a.get("raw_action") or "")
        out[eid] = dict(ents=es, acts=acts, patched=ep.get("patched", False), task=ep["task"])
    return out

def candidates(inst, kind, idx, order):
    src = inst["e_src"]
    m = re.match(r"episode_(\w+?)_(partial|patched)$", src)
    if m: return [m.group(1)], "explicit"
    oe = inst["context"].get("original_episode")
    if oe is not None: return [str(oe)], "explicit"
    ctx = inst["context"]
    toks = {o for o in ctx.get("objects", []) if ENT.match(o)}
    seq = [a.replace(" ", "") for a in ctx.get("action_sequence", [])]
    for a in seq: toks |= ents_of_action(a)
    agent = None
    mm = re.match(r"agent_(\d)_llm", src)
    if mm: agent = int(mm.group(1))
    if not toks and not seq:
        return None, "no_content"
    cands = []
    for eid in order:
        e = idx[eid]
        if e["patched"]: continue  # standard path only runs on non-patched episodes
        if not toks <= e["ents"]: continue
        cands.append(eid)
    # tighten with exact action strings when they carry entities
    spec = [a for a in seq if ents_of_action(a)]
    if spec and cands:
        tight = [c for c in cands if all(a in idx[c]["acts"][agent] for a in spec)]
        if tight: cands = tight
    return cands, ("content" if toks else "content_noent")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ind, coop, em = load(a.bank)
    order = list(em.keys())  # insertion order == processing order
    pos = {e: i for i, e in enumerate(order)}
    idx = ep_index(em)
    rows = []
    for kind, lib in (("individual", ind), ("cooperation", coop)):
        for skey, s in lib.items():
            prev_lo = 0
            recs = []
            for j, inst in enumerate(s["instances"]):
                c, how = candidates(inst, kind, idx, order)
                recs.append([j, c, how])
            # monotone pass: instances were appended in episode order within a skill
            lo = 0
            for r in recs:
                if r[1]:
                    r.append([x for x in r[1] if pos[x] >= lo] or r[1])
                    lo = min(pos[x] for x in r[3])
                else:
                    r.append(None)
            hi = len(order)
            for r in reversed(recs):
                if r[3]:
                    r[3] = [x for x in r[3] if pos[x] <= hi] or r[3]
                    hi = max(pos[x] for x in r[3])
            for j, c, how, c2 in recs:
                rows.append(dict(kind=kind, skill=skey, inst=j, how=how,
                                 n_cand=(len(c2) if c2 else 0), cands=c2 or []))
    json.dump(rows, open(a.out, "w"))
    cnt = collections.Counter((r["kind"], r["how"], "unique" if r["n_cand"] == 1 else ("none" if r["n_cand"] == 0 else "multi")) for r in rows)
    for k in sorted(cnt): print(k, cnt[k])

if __name__ == "__main__":
    main()
