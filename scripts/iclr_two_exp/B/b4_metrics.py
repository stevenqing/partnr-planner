"""Part B4: held-out H_R metrics from the B3 cells (zero model calls). Run on the remote box.

Per episode (pooled over folds, n = 197 per condition and seed):
  success     = stats task_state_success (the field Table 1/5 report as success); a `noneaction` crash (original
                code, partner action unparsed) counts 0; a missing stats file makes the cell incomplete (refused
                unless --allow-incomplete).
  completion  = task_percent_complete; noneaction -> 0.
Failure metrics (manuscript App. "Coordination Failure Metrics"), from planner-log/planner-log-episode_<id>_0.json.
An *action* of agent i is a step where replanned[i] is true and high_level_actions[i][0] is a tool name; its outcome
is the first later (or same-step) response of agent i that is not "still in progress". A_manip = Pick, Place, Open,
Close; target = first argument (the object).
  Conflict    : agent i issues a manip action on o while agent j's current action (its latest action, issued after
                agent i's previous action, i.e. in the same round) is a manip action on o. Counted once per event.
  FailPick    : Pick action whose outcome is not "Successful execution!".
  SelfConf    : per agent, action k and one of its previous 5 actions form (Pick(o), Place(o) back on the furniture
                o was picked from) or (Open(o), Close(o)) in either order. The source of o is o's location in the
                agent's world graph at the Pick step. SelfConf_literal (any Pick(o)/Place(o) pair) is also reported.
  NotClose    : manip action whose outcome says "Not close enough" (the skill's own distance check; positions of
                target objects are not logged, so d_thresh = 1.5 m cannot be applied directly).
Failure-metric means are over episodes that have a planner log (noneaction crashes write none); n is reported.
Retrieval (retrieval_dump): fallback rate = share of agent-episodes with memory whose mode is fallback_flat;
zero-overlap = no retrieved instance (resolved through the fold library's provenance.json) comes from the evaluated
episode's own fold.
Paired test: per seed, exact two-sided McNemar on success, a0-only vs both, over the 197 pooled episodes.
Usage: b4_metrics.py [--allow-incomplete]
"""
import csv, glob, json, math, os, re, statistics, sys

P = "/mnt/pfs/devs/pn5wp/shishuqing"
O = f"{P}/partnr-isambard-C/outputs/iclr_two_exp_B"
RES = f"{P}/partnr-planner/results/iclr_two_exp_2026-09-22"
LIBS = {k: f"{P}/partnr-isambard-C/data/hierarchical_skill_memory_iclr2exp/lib_{k}/hierarchical_heterogeneous_rerange"
        for k in "AB"}
ALLOW = "--allow-incomplete" in sys.argv
split = json.load(open(f"{RES}/B/split.json"))
FOLD = {e: k for k, v in split["folds"].items() for e in v}
MANIP = {"Pick", "Place", "Open", "Close"}
CONDS, SEEDS = ("both", "a0", "a1"), (0, 1, 2)
TABLE5 = {"both": dict(success=0.223, conflict=1.63, failpick=1.69, selfconf=0.27, notclose=15.99),
          "a0": dict(success=0.496, conflict=0.33, failpick=1.83, selfconf=1.36, notclose=9.82),
          "a1": dict(success=0.056, conflict=0.75, failpick=2.69, selfconf=1.51, notclose=13.81)}


def target(args):
    return (args or "").split(",")[0].strip() or None


def obj_locations(graph_text):
    locs, on = {}, False
    for line in (graph_text or "").splitlines():
        if line.startswith("Objects:"):
            on = True
            continue
        if on and ":" in line:
            o, l = line.split(":", 1)
            locs[o.strip()] = l.strip()
    return locs


def failure_metrics(path):
    steps = json.load(open(path))["steps"]
    acts = {0: [], 1: []}          # per agent: dicts (idx, tool, target, place_on, source, outcome)
    open_ = {0: None, 1: None}
    for n, s in enumerate(steps):
        for a in (0, 1):
            k = str(a)
            h = (s.get("high_level_actions") or {}).get(k) or [None]
            r = (s.get("responses") or {}).get(k) or ""
            if (s.get("replanned") or {}).get(k) and h[0]:
                args = h[1] or ""
                parts = [x.strip() for x in args.split(",")]
                rec = {"n": n, "tool": h[0], "target": target(args), "outcome": None,
                       "place_on": parts[2] if h[0] == "Place" and len(parts) > 2 else None, "source": None}
                if h[0] == "Pick":
                    rec["source"] = obj_locations((s.get("curr_graph") or {}).get(k)).get(rec["target"])
                acts[a].append(rec)
                open_[a] = rec
            if r and open_[a] is not None and open_[a]["outcome"] is None and "still in progress" not in r:
                open_[a]["outcome"] = r
    m = dict(conflict=0, failpick=0, selfconf=0, selfconf_literal=0, notclose=0, actions=len(acts[0]) + len(acts[1]))
    for i in (0, 1):
        j = 1 - i
        for k, x in enumerate(acts[i]):
            if x["tool"] == "Pick" and x["outcome"] is not None and not x["outcome"].startswith("Successful execution"):
                m["failpick"] += 1
            if x["tool"] in MANIP and x["outcome"] and "not close enough" in x["outcome"].lower():
                m["notclose"] += 1
            if x["tool"] in MANIP:
                prev_i = acts[i][k - 1]["n"] if k else -1
                cur_j = [y for y in acts[j] if y["n"] <= x["n"]]
                if cur_j:
                    y = cur_j[-1]
                    if y["n"] >= prev_i and y["tool"] in MANIP and y["target"] == x["target"]:
                        m["conflict"] += 1
            window = acts[i][max(0, k - 5):k]
            lit = rev = False
            for y in window:
                pair = {x["tool"], y["tool"]}
                if x["target"] != y["target"]:
                    continue
                if pair == {"Pick", "Place"}:
                    lit = True
                    pk, pl = (x, y) if x["tool"] == "Pick" else (y, x)
                    if pk["n"] < pl["n"] and pk["source"] and pl["place_on"] == pk["source"]:
                        rev = True
                if pair == {"Open", "Close"}:
                    lit = rev = True
            m["selfconf"] += rev
            m["selfconf_literal"] += lit
    return m


def load_bank_prov():
    out = {}
    for k, lib in LIBS.items():
        out[k] = json.load(open(f"{lib}/provenance.json"))
    return out


PROV = load_bank_prov()


def retrieval(cell_dir, lib):
    fb = tot = items = own = unresolved = 0
    own_eps = []
    for f in glob.glob(f"{cell_dir}/retrieval_dump/*.json"):
        d = json.load(open(f))
        if not d.get("rag_enabled"):
            continue
        tot += 1
        fb += d.get("mode") == "fallback_flat"
        ev_fold = FOLD[str(d["episode_id"])]
        for it in d.get("items", []):
            items += 1
            key, idx = it.get("entry_skill_key"), it.get("entry_instance_idx")
            src = None
            for f2 in ("L_coop_skills", "L_ind_skills"):
                if key in PROV[lib][f2] and idx is not None and idx < len(PROV[lib][f2][key]):
                    src = PROV[lib][f2][key][idx]
                    break
            if src is None:
                unresolved += 1
            elif FOLD.get(src) == ev_fold:
                own += 1
                own_eps.append((d["episode_id"], src))
    return dict(agent_episodes_with_memory=tot, fallback=fb, items=items, resolved=items - unresolved,
                own_fold_items=own, own_fold_examples=own_eps[:5])


def cell(seed, cond, fold):
    name = f"s{seed}_{cond}_{fold}"
    d = f"{O}/{name}"
    rd = glob.glob(f"{d}/hydra/results/*.json.gz")
    rows, missing = {}, []
    ids = split["folds"][fold]
    for e in ids:
        f = f"{rd[0]}/stats/{e}.json" if rd else None
        if not f or not os.path.exists(f):
            missing.append(e)
            continue
        s = json.load(open(f))
        row = {"episode_id": e, "fold": fold, "seed": seed, "cond": cond, "status": None}
        if "stats" in s:
            st = json.loads(s["stats"]) if isinstance(s["stats"], str) else s["stats"]
            if st.get("sim_step_count", 0) == 0:
                missing.append(e)
                continue
            row.update(status="ok", success=float(st.get("task_state_success", 0)),
                       completion=float(st.get("task_percent_complete", 0)))
        elif "object has no attribute 'lower'" in s.get("info", ""):
            row.update(status="noneaction", success=0.0, completion=0.0)
        elif "Episode over, call reset" in s.get("info", ""):
            row.update(status="steplimit", success=0.0, completion=0.0)
        else:
            missing.append(e)
            continue
        pl = f"{rd[0]}/planner-log/planner-log-episode_{e}_0.json"
        if os.path.exists(pl):
            row.update(failure_metrics(pl))
        rows[e] = row
    other = "B" if fold == "A" else "A"
    return name, rows, missing, retrieval(d, other)


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n10 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p), n01, n10


per_ep, cells, retr = [], {}, {}
incomplete = []
for s in SEEDS:
    for c in CONDS:
        for f in "AB":
            name, rows, miss, rt = cell(s, c, f)
            cells[name] = rows
            retr[name] = rt
            if miss:
                incomplete.append((name, miss[:5], len(miss)))
            per_ep.extend(rows.values())
if incomplete and not ALLOW:
    print(json.dumps({"refused_incomplete": incomplete}))
    sys.exit(2)

os.makedirs(f"{RES}/tables", exist_ok=True)
FIELDS = ["episode_id", "fold", "seed", "cond", "status", "success", "completion", "conflict", "failpick", "selfconf",
          "selfconf_literal", "notclose", "actions"]
with open(f"{RES}/B/b4_per_episode.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(per_ep)

METRICS = ["success", "completion", "conflict", "failpick", "selfconf", "selfconf_literal", "notclose"]
seedvals = {}   # (cond, seed) -> metric -> value
for c in CONDS:
    for s in SEEDS:
        rows = [r for f in "AB" for r in cells[f"s{s}_{c}_{f}"].values()]
        v = {"n": len(rows), "n_logs": sum("conflict" in r for r in rows),
             "noneaction": sum(r["status"] == "noneaction" for r in rows)}
        for m in METRICS:
            xs = [r[m] for r in rows if m in r]
            v[m] = sum(xs) / len(xs) if xs else None
        seedvals[(c, s)] = v


def ms(c, m):
    xs = [seedvals[(c, s)][m] for s in SEEDS if seedvals[(c, s)][m] is not None]
    if not xs:
        return None, None
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else None)


summary = {"conditions": {}, "per_seed": {f"{c}_s{s}": v for (c, s), v in seedvals.items()}}
with open(f"{RES}/tables/hr_heldout.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    head = ["protocol", "condition", "n_per_seed", "seeds"]
    for m in METRICS:
        head += [f"{m}_mean", f"{m}_std"]
    w.writerow(head + ["note"])
    for c in CONDS:
        row = ["held-out (this work)", c, seedvals[(c, 0)]["n"], len(SEEDS)]
        summary["conditions"][c] = {}
        for m in METRICS:
            mu, sd = ms(c, m)
            summary["conditions"][c][m] = {"mean": mu, "std": sd}
            row += [None if mu is None else round(mu, 4), None if sd is None else round(sd, 4)]
        w.writerow(row + ["Llama-3.1-8B, vLLM backend, run_ours_hr_mem_hr.sh config, fold library from the other fold"])
    for c in CONDS:
        t = TABLE5[c]
        row = ["seen (Table 5)", c, 197, "?"]
        for m in METRICS:
            row += [t.get(m), None]
        w.writerow(row + ["manuscript Table 5; H_R memory covers 196/197 evaluated episodes; HF backend; config not on disk"])

tests = {}
for s in SEEDS:
    ids = sorted(FOLD, key=int)
    a = [cells[f"s{s}_a0_{FOLD[e]}"].get(e, {}).get("success") for e in ids]
    b = [cells[f"s{s}_both_{FOLD[e]}"].get(e, {}).get("success") for e in ids]
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    p, n01, n10 = mcnemar([x for x, _ in pairs], [y for _, y in pairs])
    tests[f"s{s}"] = {"n": len(pairs), "a0_only_success": n01, "both_only_success": n10, "p_exact": p}
summary["mcnemar_a0_vs_both"] = tests
summary["retrieval"] = retr
summary["retrieval_totals"] = {
    k: sum(r[k] for r in retr.values()) for k in ("agent_episodes_with_memory", "fallback", "items", "resolved",
                                                   "own_fold_items")}

# pre-written endings
sa, sb = summary["conditions"]["a0"]["success"], summary["conditions"]["both"]["success"]
ca, cb = summary["conditions"]["a0"]["conflict"], summary["conditions"]["both"]["conflict"]
diff = sa["mean"] - sb["mean"]
big = max(x for x in (sa["std"], sb["std"]) if x is not None) if (sa["std"] or sb["std"]) else 0.0
per_seed_dir = [seedvals[("a0", s)]["success"] > seedvals[("both", s)]["success"] and
                seedvals[("a0", s)]["conflict"] < seedvals[("both", s)]["conflict"] for s in SEEDS]
if diff > big and all(per_seed_dir):
    ending = "B1"
elif diff > 0 and sum(seedvals[("a0", s)]["success"] > seedvals[("both", s)]["success"] for s in SEEDS) >= 1:
    ending = "B2"
else:
    ending = "B3"
summary["ending"] = {"ending": ending, "a0_minus_both_success": diff, "larger_std": big,
                     "per_seed_a0_higher_and_conflict_lower": per_seed_dir,
                     "rule": "B1: diff > larger std and, in all 3 seeds, a0 success > both and a0 Conflict/Ep < both; "
                             "B2: direction holds on the mean but not B1; B3: mean difference <= 0"}
summary["incomplete"] = incomplete
json.dump(summary, open(f"{RES}/B/b4_summary.json", "w"), indent=1)
print(json.dumps({"ending": summary["ending"], "conditions": summary["conditions"], "mcnemar": tests,
                  "retrieval_totals": summary["retrieval_totals"], "incomplete": incomplete}, default=str))
