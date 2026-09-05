"""Pass rate on the abstraction rung: seed episodes x samples, one model.

A single run is a single draw. The arms in this project are not greedy and neither is this,
so the reportable quantity is a rate over several starting episodes and several samples,
not one transcript.

The framework is frozen; what varies here is the starting episode and the sampling seed.
"""
import argparse, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--base-url", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 3, 7, 9])
parser.add_argument("--samples", type=int, default=3)
parser.add_argument("--moves", type=int, default=18)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--holdout-pool", type=int, nargs="+", default=[4, 6, 8, 10, 12])
# Passed straight through. Unset, every command this builds is byte-identical to the frozen
# sweep's, so the old cell stays comparable; set, this is a new cell with its own label.
parser.add_argument("--target-key", default=None)
# Passed through to the rung. With it, an operator is accepted only if it makes a holdout
# episode the library cannot solve become solvable. Unset, commands are byte-identical to
# the frozen sweep's.
parser.add_argument("--library", default=None)
args = parser.parse_args()

root = Path("outputs/agentic_rung_sweep") / args.label
root.mkdir(parents=True, exist_ok=True)

jobs = []
for episode in args.episodes:
    holdout = [j for j in args.holdout_pool if j != episode][:4]
    for sample in range(args.samples):
        jobs.append((episode, sample, holdout))


def run(job):
    episode, sample, holdout = job
    tag = "%s/e%d_s%d" % (args.label, episode, sample)
    command = [sys.executable, "scripts/viki_agentic_rung_abstraction.py",
               "--tag", tag, "--moves", str(args.moves), "--base-url", args.base_url,
               "--model", args.model, "--seed-episode", str(episode),
               "--sample-seed", str(20260829 + sample), "--temperature", str(args.temperature),
               "--holdout"] + [str(j) for j in holdout]
    if args.target_key:
        command += ["--target-key", args.target_key]
    if args.library:
        command += ["--library", args.library]
    started = time.time()
    proc = subprocess.run(command, capture_output=True, text=True, timeout=5400)
    verdict_path = Path("outputs/agentic_rung") / tag / "verdict.json"
    verdict = json.loads(verdict_path.read_text()) if verdict_path.is_file() else {"passed": None}
    return {"episode": episode, "sample": sample, "holdout": holdout,
            "passed": verdict.get("passed"), "moves_used": verdict.get("moves_used"),
            "works_on": verdict.get("works_on"), "seconds": round(time.time() - started, 1),
            "returncode": proc.returncode, "stderr_tail": proc.stderr[-300:] if proc.returncode else ""}


results = []
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(run, job): job for job in jobs}
    for future in as_completed(futures):
        row = future.result()
        results.append(row)
        print("e%-2d s%d  passed=%-5s moves=%-3s %.0fs" %
              (row["episode"], row["sample"], row["passed"], row["moves_used"], row["seconds"]),
              flush=True)
        (root / "results.json").write_text(json.dumps(sorted(
            results, key=lambda r: (r["episode"], r["sample"])), indent=1))

passed = sum(1 for r in results if r["passed"])
by_episode = {}
for r in results:
    by_episode.setdefault(r["episode"], []).append(bool(r["passed"]))
summary = {"model": args.model, "label": args.label, "runs": len(results),
           "passed": passed, "rate": round(passed / len(results), 4) if results else 0.0,
           "temperature": args.temperature, "moves": args.moves,
           "by_episode": {str(k): "%d/%d" % (sum(v), len(v)) for k, v in sorted(by_episode.items())}}
(root / "summary.json").write_text(json.dumps(summary, indent=1))
print()
print(json.dumps(summary, indent=1))
