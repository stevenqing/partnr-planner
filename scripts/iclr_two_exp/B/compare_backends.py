"""Compare the HF (in-process, transformers-CFG) and vLLM (xgrammar) backends on the same episodes and config.
Reads each cell's prompts/<agent>/prompt-episode_<id>_0-<agent>.txt (full conversation) and stats/<id>.json.
Per episode and backend: assistant turns, turns whose next user message reports a parse error ("SyntaxError" or
"not a valid" / "could not parse"), mean response chars, success, percent complete, the parsed action list, and the
first-divergence turn of the action lists (agent 0 and agent 1).
Usage: compare_backends.py <hf_cell_dir> <vllm_cell_dir> <out_json>"""
import glob, json, os, re, sys

hf, vl, out = sys.argv[1:4]
ERR = re.compile(r"SyntaxError|could not parse|not a valid|Invalid action|unable to parse", re.I)
ACT = re.compile(r"\b(Navigate|Explore|Pick|Place|Open|Close|Clean|Fill|PowerOn|PowerOff|Pour|Rearrange|Wait|Done|FindObjectTool|FindReceptacleTool|FindRoomTool)\[[^\]\n]*\]")


def res_dir(cell):
    d = glob.glob(f"{cell}/hydra/results/*.json.gz")
    return d[0] if d else None


def parse_prompt(path):
    s = open(path, errors="ignore").read()
    parts = s.split("<|start_header_id|>assistant<|end_header_id|>")
    turns = []
    for p in parts[1:]:
        resp, _, rest = p.partition("<|eot_id|>")
        nxt = rest.split("<|start_header_id|>assistant<|end_header_id|>")[0]
        turns.append({"chars": len(resp.strip()), "err": bool(ERR.search(nxt)),
                      "action": (ACT.findall(resp) and ACT.search(resp).group(0)) or None})
    return turns


def cell(cdir):
    r = res_dir(cdir)
    eps = {}
    if not r:
        return eps
    for f in glob.glob(f"{r}/stats/*.json"):
        eid = os.path.basename(f)[:-5]
        d = json.load(open(f))
        st = d.get("stats")
        st = json.loads(st) if isinstance(st, str) else st
        e = {"stats_ok": st is not None,
             "success": (st or {}).get("task_state_success"), "pc": (st or {}).get("task_percent_complete"),
             "sim_steps": (st or {}).get("sim_step_count"), "runtime": (st or {}).get("runtime"),
             "info": None if st else d.get("info", "")[-200:]}
        for a in (0, 1):
            p = f"{r}/prompts/{a}/prompt-episode_{eid}_0-{a}.txt"
            t = parse_prompt(p) if os.path.exists(p) else []
            e[f"agent{a}"] = {"turns": len(t), "err_turns": sum(x["err"] for x in t),
                              "mean_chars": round(sum(x["chars"] for x in t) / len(t), 1) if t else None,
                              "actions": [x["action"] for x in t]}
        eps[eid] = e
    return eps


H, V = cell(hf), cell(vl)
rows = {}
for eid in sorted(set(H) & set(V), key=int):
    row = {"hf": {k: v for k, v in H[eid].items() if not k.startswith("agent")},
           "vllm": {k: v for k, v in V[eid].items() if not k.startswith("agent")}}
    for a in (0, 1):
        ha, va = H[eid][f"agent{a}"], V[eid][f"agent{a}"]
        n = min(len(ha["actions"]), len(va["actions"]))
        div = next((i for i in range(n) if ha["actions"][i] != va["actions"][i]), n)
        row[f"agent{a}"] = {"hf": {k: ha[k] for k in ("turns", "err_turns", "mean_chars")},
                            "vllm": {k: va[k] for k in ("turns", "err_turns", "mean_chars")},
                            "first_divergence_turn": div, "same_prefix_actions": div}
    rows[eid] = row


def agg(E):
    t = sum(e[f"agent{a}"]["turns"] for e in E.values() for a in (0, 1))
    er = sum(e[f"agent{a}"]["err_turns"] for e in E.values() for a in (0, 1))
    ok = [e for e in E.values() if e["stats_ok"]]
    return {"episodes": len(E), "turns": t, "err_turns": er, "err_rate": round(er / t, 4) if t else None,
            "success": [e["success"] for e in ok], "pc_mean": round(sum(e["pc"] for e in ok) / len(ok), 4) if ok else None,
            "runtime_mean": round(sum(e["runtime"] for e in ok) / len(ok), 1) if ok else None}


common = set(rows)
summary = {"hf": agg({k: v for k, v in H.items() if k in common}), "vllm": agg({k: v for k, v in V.items() if k in common}),
           "episodes_compared": sorted(common, key=int), "per_episode": rows}
json.dump(summary, open(out, "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("hf", "vllm", "episodes_compared")}))
