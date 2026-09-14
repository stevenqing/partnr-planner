#!/usr/bin/env python3
"""v2: one library per family, plus the no-trace control. 72B induction, one build seed.

Each family's sweep sees only that family's elementary episodes, so the leakage constraint
holds by construction -- a family library's seeds, unsolved list and support pool cannot
contain another family's rows, let alone a comp row. Libraries are combined by union at
evaluation time, never rebuilt per split.

Acceptance is marginal and now includes ordering coverage: an episode counts as unsolved
unless the ordering its own temporal constraints call for is actually emitted.

Guards: one generation job on the endpoint, a hard timeout per run, the manifest written
after every run so a dead session loses nothing, and waiting by PID rather than by pattern.
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


def endpoint_alive(base_url: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as response:
            return response.status == 200 and b'"id"' in response.read()
    except Exception:                                                # noqa: BLE001
        return False


def run_rungs(label, rungs, seed, args, manifest, manifest_path, working, extra=()):
    for rung in rungs:
        key_slug = rung["target_key"].replace(".", "_")
        tag = "%s/e%d_%s" % (label, rung["seed_episode"], key_slug)
        verdict_path = ROOT / "outputs/agentic_rung" / tag / "verdict.json"
        if verdict_path.is_file():
            say("skip   %s" % tag)
            continue
        if getattr(args, "probe_endpoint", False) and not endpoint_alive(args.base_url):
            # Stop rather than skip: rungs within a family build on each other's library, so
            # a hole left by a dead endpoint would change every later cell of the family.
            manifest["aborted"] = {"at": tag, "why": "endpoint %s not answering" % args.base_url,
                                   "when": datetime.now().isoformat()}
            manifest_path.write_text(json.dumps(manifest, indent=1))
            say("FATAL endpoint %s not answering before %s; stopping" % (args.base_url, tag))
            raise SystemExit(3)
        held = json.loads(working.read_text())["operators"]
        command = [PY, str(RUNG), "--tag", tag,
                   "--seed-episode", str(rung["seed_episode"]),
                   "--sample-seed", str(seed),
                   "--target-key", rung["target_key"],
                   "--holdout", *[str(h) for h in rung["holdout"]],
                   *(["--coverage-pool", *[str(c) for c in rung["coverage_pool"]]]
                     if rung.get("coverage_pool") else []),
                   "--temperature", str(args.temperature), "--moves", str(args.moves),
                   "--model", args.model, "--base-url", args.base_url, *extra]
        if held:
            command += ["--library", str(working)]
        started = time.time()
        try:
            proc = subprocess.run(command, cwd=str(ROOT), capture_output=True,
                                  text=True, timeout=args.hard)
            status, err = proc.returncode, proc.stderr[-400:]
        except subprocess.TimeoutExpired:
            status, err = -1, "TIMEOUT after %ds" % args.hard
        passed, moves_used = False, None
        if verdict_path.is_file():
            verdict = json.loads(verdict_path.read_text())
            passed, moves_used = bool(verdict.get("passed")), verdict.get("moves_used")
            if passed and verdict.get("operator"):
                library = json.loads(working.read_text())
                library["operators"].append(verdict["operator"])
                working.write_text(json.dumps(library, indent=1))
        manifest["runs"].append({"label": label, "tag": tag, "status": status,
                                 "target_key": rung["target_key"],
                                 "seed_episode": rung["seed_episode"],
                                 "passed": passed, "moves_used": moves_used,
                                 "seconds": round(time.time() - started, 1),
                                 "stderr_tail": err if status else ""})
        manifest_path.write_text(json.dumps(manifest, indent=1))      # disk before print
        say("%-46s %s moves=%s library=%d"
            % (tag, "PASS" if passed else "fail", moves_used,
               len(json.loads(working.read_text())["operators"])))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, default=ROOT / "results/frozen_sweep_v2.json")
    parser.add_argument("--notrace-definition", type=Path,
                        default=ROOT / "results/frozen_sweep.json")
    parser.add_argument("--base-url", default="http://192.168.32.40:8050/v1")
    parser.add_argument("--model", default="qwen2.5-vl-72b-amendment3-f2")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs/v2_libraries")
    parser.add_argument("--hard", type=int, default=3600)
    parser.add_argument("--skip-notrace", action="store_true")
    # RQ2 no_trace rerun (2026-09-14), all OFF by default so the v2 build is unchanged:
    # `--label-format` puts the cells under a new rung directory, `--no-traces` hands the
    # rung its leakage-control flag for every FAMILY cell (the same rungs, seeds, holdouts and
    # budget as round one), `--families` restricts the run, `--probe-endpoint` stops the run
    # when the endpoint is not answering before a cell.
    parser.add_argument("--label-format", default="v2_%s")
    parser.add_argument("--no-traces", action="store_true")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--probe-endpoint", action="store_true")
    args = parser.parse_args(argv)

    definition = json.loads(args.definition.read_text())
    args.temperature = definition["temperature"]
    args.moves = definition["moves_per_run"]
    seed = definition["build_seed"]
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "run_manifest.json"
    manifest = {"definition_sha256": definition["definition_sha256"], "build_seed": seed,
                "model": args.model, "ordering_gate": definition.get("ordering_gate"),
                "started": datetime.now().isoformat(), "runs": []}

    for entry in definition["libraries"]:
        if entry["status"] != "ok":
            say("%s -- %s" % (entry["family"], entry["status"]))
            continue
        family = entry["family"]
        if args.families and family not in args.families:
            continue
        label = args.label_format % family
        working = args.out_root / ("working_%s.json" % family)
        if not working.is_file():
            working.write_text(json.dumps({"operators": []}, indent=1))
        say("=== %s (%d rungs) ===" % (family, len(entry["rungs"])))
        run_rungs(label, entry["rungs"], seed, args, manifest, manifest_path, working,
                  extra=("--no-traces",) if args.no_traces else ())
        out = args.out_root / ("library_%s.json" % family)
        subprocess.run([PY, str(ASSEMBLE), "--rung-root",
                        str(ROOT / "outputs/agentic_rung" / label), "--out", str(out),
                        "--report", str(args.out_root / ("library_%s_assembly.json" % family))],
                       cwd=str(ROOT), timeout=7200)

    if not args.skip_notrace and args.notrace_definition.is_file():
        flat = json.loads(args.notrace_definition.read_text())
        label = "v2_notrace"
        working = args.out_root / "working_notrace.json"
        if not working.is_file():
            working.write_text(json.dumps({"operators": []}, indent=1))
        say("=== no-trace control, arm (d) (%d rungs) ===" % len(flat["rungs"]))
        run_rungs(label, flat["rungs"], seed, args, manifest, manifest_path, working,
                  extra=("--no-traces",))
        subprocess.run([PY, str(ASSEMBLE), "--rung-root",
                        str(ROOT / "outputs/agentic_rung" / label),
                        "--out", str(args.out_root / "library_notrace.json"),
                        "--report", str(args.out_root / "library_notrace_assembly.json")],
                       cwd=str(ROOT), timeout=7200)

    manifest["finished"] = datetime.now().isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=1))
    say("done; manifest at %s" % manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
