"""Summarise one C cell. Each stats file is one of
  ok         {"success": true, "stats": ...}                       an evaluated episode
  steplimit  {"success": false, "info": "...Episode over, call reset before calling step"}
             the episode ran out of simulator steps; the harness raises instead of scoring. Counted as done,
             success 0, completion unknown (reported separately).
  noneaction {"success": false, "info": "...AttributeError: 'NoneType' object has no attribute 'lower'"}
             original code (dynamic_world_graph.py:1310) crashes when the partner's high-level action is None,
             i.e. the planner output did not parse. Deterministic under a fixed seed, so a rerun repeats it.
             Counted as done with success 0 and reported as a format-error episode.
  harness    any other failure (CUDA OOM from other users' processes, endpoint, ...): not done, rerun on resume.
Fake episodes (sim_step_count == 0) are not done either."""
import glob, json, os, re, sys
cell, expected = sys.argv[1], int(sys.argv[2])
ok, steplimit, noneaction, harness, fake = [], [], [], [], []
for f in glob.glob(f"{cell}/hydra/results/*.json.gz/stats/*.json"):
    eid = os.path.basename(f)[:-5]
    d = json.load(open(f))
    if "stats" in d:
        s = json.loads(d["stats"]) if isinstance(d["stats"], str) else d["stats"]
        (fake if s.get("sim_step_count", 0) == 0 else ok).append((eid, s))
    elif "Episode over, call reset before calling step" in d.get("info", ""):
        steplimit.append(eid)
    elif "object has no attribute 'lower'" in d.get("info", "") and "_update_gt_graph_by_other_agent_action" in d.get("info", ""):
        noneaction.append(eid)
    else:
        harness.append((eid, (re.findall(r"(\w+Error[^\n]{0,80})", d.get("info", "")) or ["?"])[-1]))
log = f"{cell}/run.log"
skipped = len(re.findall(r"Skipping evaluating episode", open(log, errors="ignore").read())) if os.path.exists(log) else -1
done = len(ok) + len(steplimit) + len(noneaction)
S = [s for _, s in ok]
out = dict(cell=os.path.basename(cell), done=done, expected=expected, complete=(done == expected),
           ok=len(ok), steplimit=len(steplimit), noneaction=len(noneaction), harness_fail=len(harness), fake=len(fake),
           harness_examples=harness[:3], skipped_log_lines=skipped,
           dumps=len(glob.glob(f"{cell}/retrieval_dump/*.json")),
           success_over_done=round(sum(s.get("task_state_success", 0) for s in S) / done, 4) if done else None,
           completion_over_ok=round(sum(s.get("task_percent_complete", 0) for s in S) / len(S), 4) if S else None,
           mean_runtime_s=round(sum(s.get("runtime", 0) for s in S) / len(S), 1) if S else None,
           max_runtime_s=round(max((s.get("runtime", 0) for s in S), default=0), 1))
print(json.dumps(out))
