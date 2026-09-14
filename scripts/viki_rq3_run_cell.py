#!/usr/bin/env python3
"""One RQ3 evaluation run with provenance: probe, timeout, stall guard, resume, run.json.

Used by scripts/drivers/viki_rq3_cells.sh. Two modes:

  run       one invocation of viki_eval_v2_intent_choice.py --out-dir <dir>
  assemble  the ood_single_family column from 8 completed fold runs (924 rows)

A run directory holds <tag>.partial.jsonl (append-only, one row per line, flushed), the
final <tag>.jsonl, eval.log and run.json. Re-invoking on the same directory resumes: the
evaluator scores only indices missing from the partial file, rows already there are never
rewritten, and each attempt's partial-file SHA-256 before/after is logged in run.json.

A run is complete only if: exit 0, no timeout / stall kill, endpoint answered before and
after, library SHA unchanged, produced indices == expected indices with no duplicates, and
no row carries an endpoint error (REQUEST_FAILED / reask_error) or NO_ARCHIVED_ANSWER.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/root/venvs/partnr/bin/python"
EVAL = ROOT / "scripts/viki_eval_v2_intent_choice.py"
A11 = ROOT / "results/viki_memory_experiments/amendment11"
MANIFEST = ROOT / "results/viki_memory_experiments/amendment8b/interactive_manifest.jsonl"
A10 = ROOT / "results/viki_memory_experiments/amendment10"

CANONICAL = {"id": "id", "cg_image": "recombination-imaged", "pure_text": "recombination-text",
             "ood_single_family": "id"}
# Display names as the task spec requires them; served ids come from viki_v2_evaluate.ENDPOINT.
DISPLAY = {"72B": "Qwen2.5-72B", "30B": "Qwen3-8B", "7B": "Qwen2.5-7B-Instruct"}
DISPLAY_NOTE = {"30B": "spec display name 'Qwen3-8B'; the archived 30B column is served as "
                       "qwen3-vl-30b -- reconcile before publishing",
                "7B": "served id is qwen2.5-vl-7b"}
FLAG = {"full": None, "no_grounding": "--no-grounding", "no_order": "--no-order"}
STALL_SECONDS = 20 * 60
# Real replay runtimes (outputs/abl_*.log, ours_rep72b_v3.log, v3_chain_7B.log): every
# replay cell 7-110 s on 72B/30B/7B; the slowest, 72B CG text, 96 s. 1800 s is > 15x that,
# leaves room for a slow re-ask tail, and the stall guard catches a hang well before.
TIMEOUT = {"id": 1800, "ood_single_family": 1800, "cg_image": 1800, "pure_text": 1800}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def load_evaluate():
    spec = importlib.util.spec_from_file_location("viki_v2_evaluate", ROOT / "scripts/drivers/viki_v2_evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe(base_url: str, served: str):
    out = {"url": base_url + "/models", "at": now()}
    try:
        with urllib.request.urlopen(base_url + "/models", timeout=10) as response:
            body = response.read().decode()
            out["http_status"] = response.status
            data = json.loads(body)
            out["served_ids"] = [m.get("id") for m in data.get("data", [])]
            out["model_roots"] = [m.get("root") for m in data.get("data", [])]
            out["ok"] = response.status == 200 and served in out["served_ids"]
    except Exception as error:  # noqa: BLE001
        out["http_status"] = None
        out["error"] = "%s: %s" % (type(error).__name__, error)
        out["ok"] = False
    return out


def expected_indices(split: str, family: str | None):
    if split in ("id", "ood_single_family"):
        rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
        return sorted(int(r["index"]) for r in rows if family is None or r.get("task_name") == family)
    import pandas as pd
    name = "recombination.imaged.parquet" if split == "cg_image" else "recombination.text.parquet"
    return list(range(len(pd.read_parquet(A10 / name))))


def read_rows(path: Path):
    rows, bad = [], 0
    if not path.is_file():
        return rows, bad
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return rows, bad


def row_audit(rows, expected):
    indices = [int(r["index"]) for r in rows]
    return {
        "produced_n": len(rows),
        "unique_n": len(set(indices)),
        "duplicate_n": len(indices) - len(set(indices)),
        "missing_n": len(set(expected) - set(indices)),
        "unexpected_n": len(set(indices) - set(expected)),
        "indices_match_expected": sorted(set(indices)) == sorted(expected) and len(indices) == len(set(indices)),
        "endpoint_error_n": sum(1 for r in rows if str(r.get("reason", "")).startswith("REQUEST_FAILED")
                                or "reask_error" in r),
        "no_archived_answer_n": sum(1 for r in rows if r.get("reason") == "NO_ARCHIVED_ANSWER"),
        "unparseable_n": sum(1 for r in rows if r.get("reason") == "UNPARSEABLE"),
        "reask_n": sum(1 for r in rows if r.get("reask")),
        "success_count": sum(1 for r in rows if r.get("reason") == "SOLVED"),
    }


def git_info():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
        return {"commit": head or None, "dirty": bool(dirty)}
    except Exception:  # noqa: BLE001
        return {"commit": None, "dirty": None}


def run(args) -> int:
    evaluate = load_evaluate()
    split_src = CANONICAL[args.split]
    url, served = evaluate.ENDPOINT[args.model]
    replay = A11 / evaluate.REPLAY[(args.model, split_src)]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_json = out_dir / "run.json"
    partial = out_dir / ("%s.partial.jsonl" % args.tag)
    final = out_dir / ("%s.jsonl" % args.tag)
    log = out_dir / "eval.log"
    expected = expected_indices(args.split, args.family)

    previous = json.loads(run_json.read_text()) if run_json.is_file() else None
    if previous and previous.get("complete"):
        print("skip complete %s" % out_dir)
        return 0
    if previous and (previous.get("library_sha256") != sha_file(args.library)
                     or previous.get("flag") != args.flag or previous.get("tag") != args.tag):
        print("REFUSE resume: library/flag/tag differ from %s" % run_json)
        return 2

    record = previous or {
        "run_id": "rq3-%s-%s-%s%s-%s" % (args.model, args.condition, args.split,
                                         ("-" + args.family) if args.family else "",
                                         datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")),
        "experiment": "rq3",
        "attempts": [],
    }
    command = [PY, str(EVAL), "--memory", str(args.library), "--split", split_src,
               "--tag", args.tag, "--replay", str(replay), "--model", served, "--base-url", url,
               "--workers", str(args.workers), "--out-dir", str(out_dir)]
    if args.family:
        command += ["--task-name", args.family]
    if args.flag:
        command.append(args.flag)
    record.update({
        "tag": args.tag, "condition": args.condition, "model_key": args.model,
        "model_display_name": DISPLAY[args.model], "model_display_note": DISPLAY_NOTE.get(args.model),
        "model_id": served, "endpoint": url,
        "canonical_split": args.split, "source_split_name": split_src, "fold_family": args.family,
        "library_path": str(args.library), "library_sha256": sha_file(args.library),
        "base_library_path": str(args.base_library) if args.base_library else None,
        "base_library_sha256": sha_file(args.base_library) if args.base_library else None,
        "flag": args.flag, "replay_source": str(replay), "replay_sha256": sha_file(replay),
        "manifest": str(MANIFEST), "manifest_sha256": sha_file(MANIFEST),
        "evaluator": str(EVAL), "evaluator_sha256": sha_file(EVAL),
        "runner_sha256": sha_file(Path(__file__)),
        "decoding": {"temperature": 0, "max_tokens": 700, "think_condition": "think (VIKI_NO_THINK unset)",
                     "note": "replay: only the per-row re-ask is generated live"},
        "command": command, "config_path": str(args.library), "log_path": str(log),
        "row_file": str(final), "partial_file": str(partial),
        "expected_n": len(expected), "timeout_seconds": args.timeout, "stall_seconds": STALL_SECONDS,
        "git": git_info(), "note": args.note,
    })
    attempt = {"started_at": now(), "partial_sha256_before": sha_file(partial),
               "partial_rows_before": len(read_rows(partial)[0])}
    attempt["probe_before"] = probe(url, served)
    record["attempts"].append(attempt)

    def finish(exit_status, killed=None):
        attempt["finished_at"] = now()
        attempt["exit_status"] = exit_status
        attempt["killed"] = killed
        attempt["probe_after"] = probe(url, served)
        attempt["partial_sha256_after"] = sha_file(partial)
        attempt["partial_rows_after"] = len(read_rows(partial)[0])
        attempt["library_sha256_after"] = sha_file(args.library)
        rows, bad = read_rows(final) if exit_status == 0 else read_rows(partial)
        audit = row_audit(rows, expected)
        audit["corrupt_lines"] = bad
        record.update(audit)
        record["scored_n"] = audit["produced_n"] if exit_status == 0 else None
        record["started_at"] = record["attempts"][0]["started_at"]
        record["finished_at"] = attempt["finished_at"]
        record["exit_status"] = exit_status
        record["endpoint_status"] = {"before": attempt["probe_before"]["ok"], "after": attempt["probe_after"]["ok"]}
        record["row_file_sha256"] = sha_file(final) if exit_status == 0 else None
        reasons = []
        if exit_status != 0:
            reasons.append("exit_status=%s" % exit_status)
        if killed:
            reasons.append(killed)
        if not attempt["probe_after"]["ok"]:
            reasons.append("endpoint_dead_after")
        if attempt["library_sha256_after"] != record["library_sha256"]:
            reasons.append("library_hash_changed")
        if not audit["indices_match_expected"]:
            reasons.append("indices_mismatch")
        if bad:
            reasons.append("corrupt_lines")
        if audit["endpoint_error_n"]:
            reasons.append("endpoint_error_rows")
        if audit["no_archived_answer_n"]:
            reasons.append("no_archived_answer_rows")
        record["incomplete_reasons"] = reasons
        record["complete"] = not reasons
        record["success_rate"] = (audit["success_count"] / audit["produced_n"]) if record["complete"] else None
        run_json.write_text(json.dumps(record, indent=2) + "\n")
        print("%s complete=%s n=%d/%d solved=%d reasons=%s" % (
            record["run_id"], record["complete"], audit["produced_n"], len(expected),
            audit["success_count"], reasons))
        return 0 if record["complete"] else 1

    if not attempt["probe_before"]["ok"]:
        return finish(-1, "endpoint_unavailable_before_start")
    for path in (args.library, replay, EVAL):
        if not path.is_file():
            return finish(-1, "missing_input:%s" % path)

    env = {k: v for k, v in os.environ.items() if k != "VIKI_NO_THINK"}
    env["TOKENIZERS_PARALLELISM"] = "false"
    run_json.write_text(json.dumps(record, indent=2) + "\n")
    with log.open("a") as handle:
        handle.write("\n[%s] %s\n" % (now(), " ".join(command)))
        handle.flush()
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        attempt["pid"] = process.pid
        start = last_growth = time.time()
        last_size = partial.stat().st_size if partial.is_file() else -1
        killed = None
        while process.poll() is None:
            time.sleep(5)
            size = partial.stat().st_size if partial.is_file() else -1
            if size != last_size:
                last_size, last_growth = size, time.time()
            if time.time() - start > args.timeout:
                killed = "timeout_%ds" % args.timeout
            elif time.time() - last_growth > STALL_SECONDS:
                killed = "stall_%ds_no_growth" % STALL_SECONDS
            if killed:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
        status = process.returncode if process.returncode is not None else -9
    return finish(status if not killed else (status or -15), killed)


def assemble(args) -> int:
    """ood_single_family: family X's rows from the fold run whose library excluded X."""
    manifest = {int(json.loads(l)["index"]): json.loads(l)["task_name"]
                for l in MANIFEST.read_text().splitlines() if l.strip()}
    out_dir = args.out_dir
    folds, rows, reasons = [], [], []
    for fold_dir in args.fold_dirs:
        info = json.loads((fold_dir / "run.json").read_text()) if (fold_dir / "run.json").is_file() else None
        if not info:
            reasons.append("missing_fold:%s" % fold_dir.name)
            continue
        folds.append({k: info.get(k) for k in ("run_id", "fold_family", "complete", "produced_n", "expected_n",
                                               "success_count", "library_path", "library_sha256",
                                               "row_file", "row_file_sha256")})
        if not info.get("complete"):
            reasons.append("incomplete_fold:%s" % info.get("fold_family"))
            continue
        for row in read_rows(Path(info["row_file"]))[0]:
            if row.get("task_name") != info["fold_family"]:
                reasons.append("row_from_other_family_in_fold:%s" % info["fold_family"])
                break
            rows.append(row)
    rows.sort(key=lambda r: r["index"])
    expected = sorted(manifest)
    audit = row_audit(rows, expected)
    families_ok = all(manifest.get(int(r["index"])) == r.get("task_name") for r in rows)
    if len(folds) != 8:
        reasons.append("fold_count=%d" % len(folds))
    if not audit["indices_match_expected"]:
        reasons.append("indices_mismatch")
    if not families_ok:
        reasons.append("family_label_mismatch")
    final = out_dir / ("%s.jsonl" % args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": "rq3-%s-%s-ood_single_family-assembly-%s" % (
            args.model, args.condition, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")),
        "experiment": "rq3", "condition": args.condition, "model_key": args.model,
        "model_display_name": DISPLAY[args.model], "canonical_split": "ood_single_family",
        "source_split_name": "heldout (8 single-family folds on the id manifest)",
        "manifest": str(MANIFEST), "manifest_sha256": sha_file(MANIFEST),
        "folds": folds, "expected_n": len(expected), **audit,
        "family_labels_match_manifest": families_ok, "incomplete_reasons": reasons,
        "complete": not reasons, "finished_at": now(),
    }
    if not reasons:
        final.write_text("".join(json.dumps(r) + "\n" for r in rows))
        record["row_file"] = str(final)
        record["row_file_sha256"] = sha_file(final)
        record["scored_n"] = audit["produced_n"]
        record["success_rate"] = audit["success_count"] / audit["produced_n"]
    (out_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print("%s complete=%s n=%d unique=%d solved=%d reasons=%s" % (
        record["run_id"], record["complete"], audit["produced_n"], audit["unique_n"],
        audit["success_count"], reasons))
    return 0 if record["complete"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["run", "assemble"])
    parser.add_argument("--model", required=True, choices=sorted(DISPLAY))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--split", choices=sorted(CANONICAL), default="id")
    parser.add_argument("--family", default=None)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--base-library", type=Path, default=None)
    parser.add_argument("--flag", default=None, choices=[None, "--no-grounding", "--no-order"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--note", default=None)
    parser.add_argument("--fold-dirs", type=Path, nargs="*", default=[])
    args = parser.parse_args(argv)
    if args.timeout is None:
        args.timeout = TIMEOUT[args.split]
    return run(args) if args.mode == "run" else assemble(args)


if __name__ == "__main__":
    raise SystemExit(main())
