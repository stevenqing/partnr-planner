"""Verdict for one cell: counts per-episode stats files against the dataset and flags fake episodes.

Each dataset episode falls in one class:
  ok         stats/<id>.json has metrics and sim_step_count > 0
  zero       metrics, sim_step_count == 0 and runtime < 60 s (fake episode signature: model/endpoint died).
             sim_step_count == 0 with a longer runtime is a real episode (e.g. the model says Done[] at once) -> ok
  env_over   stats/<id>.json is {"success": false} with "Episode over, call reset before calling step":
             the habitat env ended (end on task success or step limit) while the planner kept stepping.
             This is how the unchanged pipeline behaves; such episodes write no metrics and no planner-log.
             Since 09-22 03:10 every cell runs with +iclr_env_over_metrics=True (iclr_env_over.py), so an env_over
             episode is a leftover from a run without the switch: the cell is not done and resume reruns it.
  format_error {"success": false} raised inside the action parser (llm/instruct/utils.py actions_parser):
             the model's output could not be parsed. Final, reported as is (spec 9: no prompt fix, no rerun).
  noneaction {"success": false} with AttributeError 'NoneType' .lower() in
             dynamic_world_graph._update_gt_graph_by_other_agent_action: the partner's planner output did not parse
             (high-level action None) and the original code crashes on it. Same rule as part C: final, counted as
             done, success 0, completion 0 (no state at the crash is written), reported as a format error.
  error      {"success": false} with any other exception (OOM, crash): the cell is not done and is retried
  missing    no stats file (not run yet, or killed)
done = no missing, no zero, no error, no env_over. format_error and noneaction are final (spec 9).  Prints key: value lines (exit.txt).
"""
import gzip, json, pathlib, sys, re

out, ds, rc, why = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
B = pathlib.Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-isambard")
ids = [str(e["episode_id"]) for e in json.load(gzip.open(B / "task_classification_datasets" / ds))["episodes"]]
stats_dir = out / "hydra" / "results" / pathlib.Path(ds).name / "stats"
cls = {k: [] for k in ("ok", "zero", "env_over", "format_error", "noneaction", "error", "missing")}
ok = {}
for i in ids:
    f = stats_dir / f"{i}.json"
    if not f.exists():
        cls["missing"].append(i); continue
    d = json.loads(f.read_text())
    if "stats" in d:
        s = json.loads(d["stats"])
        fake = not s.get("sim_step_count") and s.get("runtime", 0) < 60
        (cls["zero"] if fake else cls["ok"]).append(i)
        if not fake:
            ok[i] = s
    elif "Episode over, call reset before calling step" in str(d.get("info", "")):
        cls["env_over"].append(i)
    elif "_update_gt_graph_by_other_agent_action" in str(d.get("info", "")) and "'NoneType' object has no attribute 'lower'" in str(d.get("info", "")):
        cls["noneaction"].append(i)
    elif "in actions_parser" in str(d.get("info", "")):
        cls["format_error"].append(i)
    else:
        cls["error"].append(i)
log = (out / "run.log").read_text(errors="replace") if (out / "run.log").exists() else ""
skipped = len(re.findall(r"Skipping evaluating episode", log))
seed = re.findall(r"ICLR_SEED (\d+)", log)
done = not (cls["missing"] or cls["zero"] or cls["error"] or cls["env_over"])
print(f"status: {'done' if done else 'incomplete'}")
print(f"rc: {rc}")
print(f"guard: {why or 'none'}")
print(f"episodes_with_stats: {len(ids) - len(cls['missing'])}/{len(ids)}")
for k, v in cls.items():
    print(f"{k}: {len(v)} {' '.join(v)}")
print(f"skipped_episodes_in_log: {skipped}")
print(f"iclr_seed_lines: {sorted(set(seed))}")
n_all = len(ids)
print(f"noneaction_rate: {len(cls['noneaction'])}/{n_all}")
print(f"format_error_rate: {len(cls['format_error']) + len(cls['noneaction'])}/{n_all} (format_error + noneaction)")
if done:  # all-episode denominator: format_error / noneaction count as success 0, completion 0
    print(f"all_success_mean: {sum(s['task_state_success'] for s in ok.values())/n_all:.4f}")
    print(f"all_pc_mean: {sum(s['task_percent_complete'] for s in ok.values())/n_all:.4f}")
if ok:
    n = len(ok)
    print(f"ok_success_mean: {sum(s['task_state_success'] for s in ok.values())/n:.4f}")
    print(f"ok_pc_mean: {sum(s['task_percent_complete'] for s in ok.values())/n:.4f}")
    print(f"ok_runtime_mean_s: {sum(s['runtime'] for s in ok.values())/n:.1f}")
