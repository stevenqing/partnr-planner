"""D.1.2: can the Isambard text-demonstration banks (R 450, S 20) be executed by the A-repo gate?

Per instance, zero LLM calls:
  demo_parse    the `demo` string itself yields >=1 tool call of the form Verb[args] with a PARTNR tool verb
                (the spec's literal question; demos are written in natural language)
  seq_parse     every entry of context.action_sequence is Verb[args] with a PARTNR tool verb and
                concrete (non-placeholder) arguments -- the structured field closest to a tool-call sequence
  provenance    the instance records its source episode (and a step range)
  transition    a claimed transition in the gate's form (predicate, subject, value) follows from the
                instance's own fields: Place[x, on|within, y] -> is_on_top/is_inside(x, y),
                Clean -> is_clean, Fill -> is_filled, PowerOn -> is_powered_on, PowerOff -> not powered
  single_errand the manipulating verbs of the sequence touch exactly one object, which is what the gate's
                operator form (one subject ?x, spares only for enabling verbs) can hold
A skill passes the feasibility bar when >=1 of its instances has (demo_parse or seq_parse) and transition.
The spec's bar is demo_parse and transition; both are reported.
"""
import collections, csv, gzip, json, re, sys

VERBS = {"Navigate", "Pick", "Place", "Open", "Close", "Clean", "Fill", "Pour", "PowerOn", "PowerOff",
         "Rearrange", "Explore", "Wait", "Done", "FindObjectTool", "FindReceptacleTool", "FindRoomTool"}
MANIP = {"Pick", "Place", "Clean", "Fill", "Pour", "PowerOn", "PowerOff", "Rearrange"}
ACT = re.compile(r"\b([A-Z][A-Za-z]+)\[([^\]]*)\]")
ENT = re.compile(r"^[a-z][a-z_]*_\d+$")
PLACEHOLDER = {"target", "object", "object_0", "furniture_0", "x", "y", "?x", "?y", ""}


def transitions(seq):
    out = []
    for a in seq:
        m = ACT.fullmatch(a.strip())
        if not m: continue
        v, args = m.group(1), [x.strip() for x in m.group(2).split(",")]
        if v in ("Place", "Rearrange") and len(args) >= 3 and ENT.match(args[0]) and ENT.match(args[2]):
            rel = args[1].lower()
            out.append(("is_inside" if rel in ("within", "in", "inside") else "is_on_top", args[0], args[2]))
        elif v in ("Clean", "Fill", "PowerOn", "PowerOff") and args and ENT.match(args[0]):
            out.append(({"Clean": "is_clean", "Fill": "is_filled", "PowerOn": "is_powered_on",
                         "PowerOff": "is_powered_off"}[v], args[0], True))
    return out


def seq_ok(seq):
    if not seq: return False
    for a in seq:
        m = ACT.fullmatch(a.strip())
        if not m or m.group(1) not in VERBS: return False
        args = [x.strip() for x in m.group(2).split(",")]
        if m.group(1) not in ("Wait", "Done") and (not args or args[0] in PLACEHOLDER): return False
    return True


def manip_objects(seq):
    objs = set()
    for a in seq:
        m = ACT.fullmatch(a.strip())
        if m and m.group(1) in MANIP:
            objs.add(m.group(2).split(",")[0].strip())
    return objs


def main():
    banks = dict(a.split("=") for a in sys.argv[1:-1]); out = sys.argv[-1]
    rows, skills = [], collections.defaultdict(list)
    for bname, path in banks.items():
        for f, kind in (("L_ind_skills", "individual"), ("L_coop_skills", "cooperation")):
            lib = json.load(gzip.open(f"{path}/{f}.json.gz"))
            for sk, s in lib.items():
                for j, inst in enumerate(s["instances"]):
                    ctx = inst.get("context", {}); seq = ctx.get("action_sequence") or []
                    demo_calls = [m for m in ACT.findall(inst.get("demo", "")) if m[0] in VERBS]
                    tr = transitions(seq)
                    prov = bool(re.match(r"episode_\w+_(partial|patched)$", inst.get("e_src", ""))) or "original_episode" in ctx
                    r = dict(bank=bname, kind=kind, skill=sk, inst=j, e_src=inst.get("e_src"),
                             demo_parse=int(bool(demo_calls)), seq_parse=int(seq_ok(seq)), provenance_episode=int(prov),
                             provenance_steps=0, transition=int(bool(tr)), n_transitions=len(tr),
                             single_errand=int(len(manip_objects(seq)) == 1))
                    rows.append(r); skills[(bname, kind, sk)].append(r)
    with open(out + "/per_instance.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = collections.OrderedDict()
    for b in banks:
        for k in ("individual", "cooperation", "all"):
            R = [r for r in rows if r["bank"] == b and (k == "all" or r["kind"] == k)]
            S = {key: v for key, v in skills.items() if key[0] == b and (k == "all" or key[1] == k)}
            d = dict(instances=len(R), skills=len(S))
            for fld in ("demo_parse", "seq_parse", "provenance_episode", "provenance_steps", "transition", "single_errand"):
                d[f"inst_{fld}"] = sum(r[fld] for r in R)
            d["skills_bar_spec(demo_parse&transition)"] = sum(any(r["demo_parse"] and r["transition"] for r in v) for v in S.values())
            d["skills_bar_seq(seq_parse&transition)"] = sum(any(r["seq_parse"] and r["transition"] for r in v) for v in S.values())
            d["skills_bar_seq_single(seq_parse&transition&single_errand)"] = sum(any(r["seq_parse"] and r["transition"] and r["single_errand"] for r in v) for v in S.values())
            summ[f"{b}.{k}"] = d
    tot = {x: sum(summ[f"{b}.all"][x] for b in banks) for x in summ[f"{list(banks)[0]}.all"]}
    for x in [k for k in tot if k.startswith("skills_bar")]:
        tot[x.replace("skills_bar", "frac_bar")] = round(tot[x] / tot["skills"], 4)
    summ["R+S.all"] = tot
    json.dump(summ, open(out + "/executability_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
