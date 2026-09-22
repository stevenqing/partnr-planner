#!/usr/bin/env python3
"""A2 re-ask part: send the rows whose re-ask fires through the UNCHANGED live runner.

For one model: writes, per (split, memory), a replay file holding only those rows' archived
first answers (taken from the same source file the published cell replayed), then runs
scripts/viki_eval_v2_intent_choice.py --replay <that file> --memory <replay-mined memory>
--out-dir. Rows absent from the replay file come back as NO_ARCHIVED_ANSWER without planning
or calling, so the only model calls are the re-asks. Jobs run concurrently against the one
endpoint; each has a hard timeout and a 20-minute stall guard on its partial file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
A11 = ROOT / "results/viki_memory_experiments/amendment11"
BASE = ROOT / "results/iclr_two_exp_2026-09-22/A"
PY = "/root/venvs/partnr/bin/python"
REPLAY = {("72B", "id"): "intent_crew_clean.jsonl", ("72B", "cg_text"): "recomb_text_agentic.jsonl",
          ("72B", "cg_image"): "recomb_imaged_agentic.jsonl", ("30B", "id"): "m30_id.jsonl",
          ("30B", "cg_text"): "m30_recomb_text.jsonl", ("30B", "cg_image"): "m30_recomb_imaged.jsonl",
          ("7B", "id"): "m7_id.jsonl", ("7B", "cg_text"): "m7_recomb_text.jsonl",
          ("7B", "cg_image"): "m7_recomb_imaged.jsonl"}
ENDPOINT = {"72B": ("http://127.0.0.1:8050/v1", "qwen2.5-vl-72b-amendment3-f2"),
            "30B": ("http://127.0.0.1:8062/v1", "qwen3-vl-30b"),
            "7B": ("http://127.0.0.1:8061/v1", "qwen2.5-vl-7b")}
RUNNER_SPLIT = {"id": "id", "ood": "id", "cg_text": "recombination-text", "cg_image": "recombination-imaged"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    url, served = ENDPOINT[args.model]
    work = BASE / "a2_reask" / args.model
    work.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split in ("id", "ood", "cg_image", "cg_text"):
        first = json.loads((BASE / "a2_first" / ("%s_%s_replay_first.json" % (args.model, split))).read_text())
        fire = [r for r in first["rows"] if r.get("reask_would_fire")]
        if not fire:
            continue
        source = A11 / REPLAY[(args.model, "id" if split == "ood" else split)]
        archived = {int(json.loads(l)["index"]): l for l in source.read_text().splitlines() if l.strip()}
        groups = {}
        for r in fire:
            key = r["memory"] if split == "ood" else "all"
            groups.setdefault(key, []).append(r)
        for key, rows in groups.items():
            family = rows[0]["task_name"] if split == "ood" else None
            tag = "%s_%s%s" % (args.model, split, "_" + family if family else "")
            replay_file = work / (tag + ".replay.jsonl")
            replay_file.write_text("".join(archived[r["index"]] + "\n" for r in rows))
            command = [PY, "scripts/viki_eval_v2_intent_choice.py", "--memory", str(ROOT / rows[0]["memory"]),
                       "--split", RUNNER_SPLIT[split], "--tag", tag, "--replay", str(replay_file),
                       "--model", served, "--base-url", url, "--workers", str(args.workers),
                       "--out-dir", str(work)]
            if family:
                command += ["--task-name", family]
            jobs.append({"tag": tag, "split": split, "rows": len(rows), "command": command})
    (work / "jobs.json").write_text(json.dumps(jobs, indent=1))
    print("%s: %d jobs, %d re-ask rows" % (args.model, len(jobs), sum(j["rows"] for j in jobs)), flush=True)
    if args.dry:
        return 0
    running = []
    for job in jobs:
        if (work / (job["tag"] + ".jsonl")).is_file():
            print("skip", job["tag"], flush=True)
            continue
        log = open(work / (job["tag"] + ".log"), "w")
        proc = subprocess.Popen(job["command"], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        running.append((job, proc, time.time()))
    while running:
        time.sleep(15)
        still = []
        for job, proc, started in running:
            partial = work / (job["tag"] + ".partial.jsonl")
            if proc.poll() is not None:
                print("done", job["tag"], "rc", proc.returncode, flush=True)
                continue
            age = time.time() - (partial.stat().st_mtime if partial.is_file() else started)
            if time.time() - started > args.timeout or age > 1200:
                proc.kill()
                print("KILLED", job["tag"], "stall" if age > 1200 else "timeout", flush=True)
                continue
            still.append((job, proc, started))
        running = still
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
