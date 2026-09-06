#!/usr/bin/env python3
"""P0: build the three agent libraries under the frozen sweep definition. 72B induction only.

One library per build seed. The three share the same seed episodes and holdouts exactly and
differ only in the sample seed, so the variance between them is LLM sampling and nothing
else. Rungs run in the definition's fixed order and acceptance is marginal: each rung is
given the library grown so far and an operator is admitted only if it makes a holdout
episode the memory cannot solve solvable.

Guards, per this repo's rules: one generation job on the endpoint, a hard timeout per run,
progress written to disk after every run so a dead session loses nothing, and waiting is by
PID rather than by any process-name pattern.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = "/root/venvs/partnr/bin/python"
RUNG = ROOT / "scripts/viki_agentic_rung_abstraction.py"
ASSEMBLE = ROOT / "scripts/viki_assemble_agentic_library.py"


def say(message: str) -> None:
    print("[%s] %s" % (datetime.now().strftime("%m-%d %H:%M:%S"), message), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, default=ROOT / "results/frozen_sweep.json")
    parser.add_argument("--base-url", default="http://192.168.32.40:8050/v1")
    parser.add_argument("--model", default="qwen2.5-vl-72b-amendment3-f2")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs/p0_libraries")
    parser.add_argument("--hard", type=int, default=3600, help="seconds per rung run")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--label-prefix", default="p0",
                        help="rung tag prefix, so a second arm does not collide with P0")
    parser.add_argument("--no-traces", action="store_true",
                        help="leakage control arm (d): the rung sees no episode data")
    args = parser.parse_args(argv)

    definition = json.loads(args.definition.read_text())
    rungs = definition["rungs"]
    seeds = args.seeds or definition["build_seeds"]
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"definition": str(args.definition), "definition_sha256": definition["definition_sha256"],
                "label_prefix": args.label_prefix, "no_traces": args.no_traces,
                "model": args.model, "base_url": args.base_url, "seeds": seeds,
                "temperature": definition["temperature"], "moves": definition["moves_per_run"],
                "started": datetime.now().isoformat(), "runs": []}
    manifest_path = args.out_root / "run_manifest.json"

    for seed in seeds:
        label = "%s_s%d" % (args.label_prefix, seed)
        working = args.out_root / ("working_%d.json" % seed)
        if not working.is_file():
            working.write_text(json.dumps({"operators": []}, indent=1))
        for position, rung in enumerate(rungs):
            key_slug = rung["target_key"].replace(".", "_")
            tag = "%s/e%d_%s" % (label, rung["seed_episode"], key_slug)
            verdict_path = ROOT / "outputs/agentic_rung" / tag / "verdict.json"
            if verdict_path.is_file():
                say("skip   %s (on disk)" % tag)
                continue
            held = json.loads(working.read_text())["operators"]
            command = [PY, str(RUNG), "--tag", tag,
                       "--seed-episode", str(rung["seed_episode"]),
                       "--sample-seed", str(seed),
                       "--target-key", rung["target_key"],
                       "--holdout", *[str(h) for h in rung["holdout"]],
                       "--temperature", str(definition["temperature"]),
                       "--moves", str(definition["moves_per_run"]),
                       "--model", args.model, "--base-url", args.base_url]
            if args.no_traces:
                command.append("--no-traces")
            if held:
                command += ["--library", str(working)]
            started = time.time()
            try:
                proc = subprocess.run(command, cwd=str(ROOT), capture_output=True,
                                      text=True, timeout=args.hard)
                status, err = proc.returncode, proc.stderr[-400:]
            except subprocess.TimeoutExpired:
                status, err = -1, "TIMEOUT after %ds" % args.hard
            elapsed = round(time.time() - started, 1)

            passed, moves_used = False, None
            if verdict_path.is_file():
                verdict = json.loads(verdict_path.read_text())
                passed, moves_used = bool(verdict.get("passed")), verdict.get("moves_used")
                if passed and verdict.get("operator"):
                    library = json.loads(working.read_text())
                    library["operators"].append(verdict["operator"])
                    working.write_text(json.dumps(library, indent=1))
            manifest["runs"].append({"seed": seed, "tag": tag, "position": position,
                                     "target_key": rung["target_key"],
                                     "seed_episode": rung["seed_episode"],
                                     "holdout": rung["holdout"], "status": status,
                                     "passed": passed, "moves_used": moves_used,
                                     "seconds": elapsed, "stderr_tail": err if status else ""})
            manifest_path.write_text(json.dumps(manifest, indent=1))     # disk before print
            say("%-42s %s  moves=%s  %ss  library=%d"
                % (tag, "PASS" if passed else "fail", moves_used, elapsed,
                   len(json.loads(working.read_text())["operators"])))

        out = args.out_root / ("agentic_library_%d.json" % seed)
        report = args.out_root / ("agentic_library_%d_assembly.json" % seed)
        say("assembling %s" % out.name)
        subprocess.run([PY, str(ASSEMBLE), "--rung-root",
                        str(ROOT / "outputs/agentic_rung" / label),
                        "--out", str(out), "--report", str(report)],
                       cwd=str(ROOT), timeout=7200)

    manifest["finished"] = datetime.now().isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=1))
    say("done; manifest at %s" % manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
