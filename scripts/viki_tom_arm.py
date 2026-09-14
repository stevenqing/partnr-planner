#!/usr/bin/env python3
"""Figure 2 ToM arm for VIKI-L2 (ICLR 2027 supplementary results).

ToM is the zero-shot arm plus one instruction and nothing else. Every part of the zero-shot
path is imported, not restated:

  messages   viki_amendment8b.interactive_messages (id / ood_single_family) and
             viki_amendment10_run.build_messages (cg_image / pure_text), called with an
             empty memory prompt -- exactly what the zero_shot arm sends.
  ToM        viki_amendment6.add_tom_to_messages, which appends
             viki_amendment6.TOM_REASONING_TEMPLATE to the official system message. The
             instruction text is read from prompts/viki_tom_prompt.txt; when it equals the
             canonical template the canonical function is called, otherwise the same
             insertion is done by hand and the run records `tom_insertion` accordingly.
  decoding   viki_amendment8b.PLAN_TEMPERATURE / PLAN_MAX_TOKENS / SEED, the served name
             from VIKI_BACKBONE, and the OpenAI client settings of the zero-shot runner
             for that split (8b: max_retries=5, timeout=3600; 10_run: library defaults).
  think      the benchmark's <think> rule stays in; the runner refuses VIKI_NO_THINK=1.
  scoring    viki_report_matrix.tolerant (JSON-tolerant) is the `success` field; the
             official bench.score_response is kept for the zero-shot row keys
             (score / task_score / format_score). Both run on the main thread, because
             both swap the same scorer module's `random` global.

Every request is asserted to differ from the zero-shot request only by the ToM suffix of
the system message before it is sent.

Subcommands
  run           one cell (or one OOD fold). Resume fills only missing example_ids.
  merge-folds   concatenate the 8 single-family folds into the ood_single_family cell.
  payload-diff  no model call: rebuild the zero-shot and ToM payloads for one example,
                print the diff, and check the zero-shot payload hash against the archived
                zero-shot row of the same model.

Model identity comes from VIKI_BACKBONE / VIKI_SERVED_MODEL, as for every other runner.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import difflib
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "our_method"))

import pandas as pd  # noqa: E402
from openai import OpenAI  # noqa: E402

import viki_fork_guard  # noqa: E402
from habitat_llm.evaluation import viki_bench as bench  # noqa: E402
import viki_amendment5 as a5  # noqa: E402
import viki_amendment6 as a6  # noqa: E402
import viki_amendment8b as a8b  # noqa: E402
import viki_amendment10_run as a10  # noqa: E402
import viki_amendment9_folds as foldmod  # noqa: E402
from viki_amendment9_diag102 import parse_plan  # noqa: E402

CONDITION = "tom"
OUT_ROOT = ROOT / "results/paper_viki_iclr2027/raw"
PROMPT_PATH = ROOT / "prompts/viki_tom_prompt.txt"
A8B_DIR = a8b.OUTPUT_DIR
A10_DIR = a10.SPLIT_DIR

SPLITS: Dict[str, Dict[str, Any]] = {
    "id": {"source_split_name": "id", "runner": "a8b", "expected_n": 924},
    "ood_single_family": {"source_split_name": "heldout", "runner": "a8b", "expected_n": 924},
    "cg_image": {"source_split_name": "imaged", "runner": "a10", "expected_n": 297},
    "pure_text": {"source_split_name": "text", "runner": "a10", "expected_n": 297},
}

# display name = the actual checkpoint; paper_column_label = the label the task spec uses
# for that column (its "Qwen3-8B" anchors are this repo's Qwen3-VL-30B-A3B numbers).
MODELS: Dict[str, Dict[str, str]] = {
    "qwen2_5_vl_72b": {
        "display": "Qwen2.5-VL-72B-Instruct",
        "slug": "qwen2.5-vl-72b-instruct",
        "paper_column_label": "Qwen2.5-72B",
        "zero_shot_tag": "",
    },
    "qwen3_vl_30b": {
        "display": "Qwen3-VL-30B-A3B-Instruct",
        "slug": "qwen3-vl-30b-a3b-instruct",
        "paper_column_label": "Qwen3-8B",
        "zero_shot_tag": "m30",
    },
    "qwen2_5_vl_7b": {
        "display": "Qwen2.5-VL-7B-Instruct",
        "slug": "qwen2.5-vl-7b-instruct",
        "paper_column_label": "Qwen2.5-7B-Instruct",
        "zero_shot_tag": "m7",
    },
}

SOURCE_FILES = {
    "tom_runner": HERE / "viki_tom_arm.py",
    "tom_canonical_source": HERE / "viki_amendment6.py",
    "messages_id_ood": HERE / "viki_amendment8b.py",
    "messages_cg": HERE / "viki_amendment10_run.py",
    "viki_bench": ROOT / "habitat_llm/evaluation/viki_bench.py",
    "memory_insertion": ROOT / "habitat_llm/evaluation/viki_memory_skill.py",
    "scorer_tolerant": HERE / "viki_report_matrix.py",
    "parser_tolerant": HERE / "viki_amendment9_diag102.py",
    "scorer_official": a8b.OFFICIAL_SCORER,
    "fold_construction": HERE / "viki_amendment9_folds.py",
}


class Refuse(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> Optional[str]:
    return a5.file_sha256(path) if Path(path).is_file() else None


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)


def model_entry() -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    backbone = a5.BACKBONE
    if backbone not in MODELS:
        raise Refuse(f"VIKI_BACKBONE={backbone} has no Figure-2 entry")
    return backbone, MODELS[backbone], a5.BACKBONES[backbone]


def git_state() -> Dict[str, Any]:
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--",
                                "scripts", "habitat_llm", "prompts", "our_method"],
                               capture_output=True, text=True, timeout=60).stdout
        return {"commit": head or None, "dirty_paths": [l for l in dirty.splitlines() if l]}
    except Exception as error:  # noqa: BLE001
        return {"commit": None, "error": repr(error)}


def cell_dir(out_root: Path, slug: str, split: str, fold: str = "") -> Path:
    base = out_root / "figure2_tom" / slug / split
    return base / "folds" / fold if fold else base


# ----------------------------------------------------------------------------- inputs


def load_split(split: str, fold: str) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]], Path]:
    runner = SPLITS[split]["runner"]
    if runner == "a8b":
        frame = pd.read_parquet(a8b.SOURCE_PARQUET)
        manifest = a8b.load_manifest()
        manifest_file = a8b.MANIFEST_PATH
        if split == "ood_single_family":
            if fold not in foldmod.folds():
                raise Refuse(f"unknown fold {fold!r}; folds are {foldmod.folds()}")
            if os.environ.get("A9_SIBLINGS") == "1":
                raise Refuse("A9_SIBLINGS=1 is the sibling-group construction, not single-family")
            keep = set(foldmod.rows_of(fold))
            manifest = {i: manifest[i] for i in sorted(keep)}
        return frame, manifest, manifest_file
    variant = SPLITS[split]["source_split_name"]
    return pd.read_parquet(a10.split_path(variant)), a10.manifest_for(variant), a10.split_path(variant)


def zero_shot_messages(split: str, sample: Dict[str, Any], record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What the zero_shot arm sends for this row (memory prompt is empty)."""
    if SPLITS[split]["runner"] == "a8b":
        return a8b.interactive_messages(sample, record["partner_prefix"], "")
    return a10.build_messages(sample, record["partner_prefix"], "", split == "cg_image")


def tom_messages(zero_shot: List[Dict[str, Any]], prompt_text: str) -> List[Dict[str, Any]]:
    if prompt_text == a6.TOM_REASONING_TEMPLATE:
        return a6.add_tom_to_messages(zero_shot)
    result = copy.deepcopy(zero_shot)
    system = next(m for m in result if m["role"] == "system")
    if not isinstance(system["content"], str):
        raise Refuse("Official VIKI system message is not text")
    system["content"] = f"{system['content']}\n\n{prompt_text}"
    return result


def only_tom_differs(zero_shot: List[Dict[str, Any]], tom: List[Dict[str, Any]], prompt_text: str) -> bool:
    if len(zero_shot) != len(tom):
        return False
    systems = 0
    for zs, tm in zip(zero_shot, tom):
        if zs["role"] != tm["role"]:
            return False
        if zs["role"] == "system":
            systems += 1
            if tm["content"] != f"{zs['content']}\n\n{prompt_text}":
                return False
        elif canonical(zs) != canonical(tm):
            return False
    return systems == 1


def make_client(split: str, base_url: str) -> OpenAI:
    if SPLITS[split]["runner"] == "a8b":
        return OpenAI(api_key="EMPTY", base_url=base_url, max_retries=5, timeout=3600)
    return OpenAI(base_url=base_url, api_key="EMPTY")


def probe(base_url: str) -> Tuple[int, Any]:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=10) as response:
            return response.status, json.loads(response.read().decode())
    except Exception as error:  # noqa: BLE001
        code = getattr(error, "code", 0) or 0
        return int(code), repr(error)


def read_rows(path: Path) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    rows: Dict[int, Dict[str, Any]] = {}
    duplicates: List[int] = []
    if not path.is_file():
        return rows, duplicates
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise Refuse(f"{path} ends in a partial line; inspect it by hand, nothing was changed")
    for number, line in enumerate(raw.decode().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Refuse(f"{path}:{number} is not JSON ({error}); nothing was changed") from error
        key = int(row["example_id"])
        if key in rows:
            duplicates.append(key)
        rows[key] = row
    return rows, duplicates


def json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=repr))
    except Exception:  # noqa: BLE001
        return repr(value)


# ----------------------------------------------------------------------------- run


def cmd_run(args: argparse.Namespace) -> int:
    if os.environ.get("VIKI_NO_THINK", "") == "1":
        raise Refuse("VIKI_NO_THINK=1 is the no-think condition; Figure 2 is the think condition")
    split, fold = args.split, args.fold or ""
    if split == "ood_single_family" and not fold:
        raise Refuse("ood_single_family runs per fold (--fold), then merge-folds")
    if fold and split != "ood_single_family":
        raise Refuse("--fold only applies to ood_single_family")
    backbone, model, registry = model_entry()
    out_root = Path(args.out_root)
    directory = cell_dir(out_root, model["slug"], split, fold)
    rows_path, run_path = directory / "rows.jsonl", directory / "run.json"

    prompt_file = Path(args.prompt_file).resolve()
    prompt_bytes = prompt_file.read_bytes()
    prompt_text = prompt_bytes.decode()
    prompt_sha = sha_bytes(prompt_bytes)

    # Endpoint gate: HTTP 200 on /models serving exactly this backbone, before any file.
    status, body = probe(args.base_url)
    if status != 200:
        raise Refuse(f"endpoint probe {args.base_url}/models returned {status}: {body}")
    served_ids = [m.get("id") for m in body.get("data", [])] if isinstance(body, dict) else []
    if served_ids != [a8b.SERVED_MODEL]:
        raise Refuse(f"endpoint serves {served_ids}, expected [{a8b.SERVED_MODEL}]")
    try:
        runtime = a5.validate_local_service(backbone, args.base_url)
    except a5.GateFailure as error:
        raise Refuse(f"service gate: {error}") from error

    frame, manifest, manifest_file = load_split(split, fold)
    if args.limit:
        manifest = {i: manifest[i] for i in sorted(manifest)[: args.limit]}
    expected_n = len(manifest)
    if not args.limit and not fold and expected_n != SPLITS[split]["expected_n"]:
        raise Refuse(f"{split} manifest has {expected_n} rows, expected {SPLITS[split]['expected_n']}")

    sources = {name: {"path": str(path), "sha256": sha_file(path)} for name, path in SOURCE_FILES.items()}
    config = {
        "experiment": "figure2_tom",
        "condition": CONDITION,
        "canonical_split": split,
        "source_split_name": SPLITS[split]["source_split_name"],
        "fold": fold or None,
        "limit": args.limit or None,
        "backbone": backbone,
        "model_id": registry["model_id"],
        "model_revision": registry["model_revision"],
        "served_model": a8b.SERVED_MODEL,
        "seed": a8b.SEED,
        "temperature": a8b.PLAN_TEMPERATURE,
        "max_tokens": a8b.PLAN_MAX_TOKENS,
        "think_condition": "think",
        "prompt_sha256": prompt_sha,
        "manifest_sha256": sha_file(manifest_file),
        "example_ids_sha256": sha_bytes(canonical(sorted(manifest)).encode()),
        "source_sha256": {name: item["sha256"] for name, item in sources.items()},
    }
    fingerprint = sha_bytes(canonical(config).encode())
    run_id = "__".join(
        [CONDITION, model["slug"], split]
        + ([fold] if fold else [])
        + ([f"smoke{args.limit}"] if args.limit else [])
        + [fingerprint[:12]]
    )

    previous: Dict[str, Any] = json.loads(run_path.read_text()) if run_path.is_file() else {}
    if previous and previous.get("run_id") != run_id:
        raise Refuse(
            f"{run_path} belongs to run {previous.get('run_id')}, this configuration is {run_id}; "
            "resume across a changed configuration or code version is not allowed"
        )
    done, duplicates = read_rows(rows_path)
    if duplicates:
        raise Refuse(f"{rows_path} already holds duplicate example_ids {sorted(set(duplicates))[:10]}")
    foreign = [i for i, r in done.items() if r.get("run_fingerprint") != fingerprint or i not in manifest]
    if foreign:
        raise Refuse(f"{rows_path} holds {len(foreign)} rows from another configuration or outside the manifest")
    pending = [i for i in sorted(manifest) if i not in done]

    git = git_state()  # forks: done before any worker thread exists
    attempt: Dict[str, Any] = {
        "started_at": now_iso(),
        "finished_at": None,
        "exit_status": "running",
        "command": " ".join([sys.executable] + sys.argv),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "log_path": args.log_path or None,
        "rows_before": len(done),
        "rows_sha256_before": sha_file(rows_path),
        "pending_at_start": len(pending),
        "git": git,
    }
    attempts = list(previous.get("attempts", [])) + [attempt]
    record_lock = threading.Lock()

    def write_run(exit_status: str) -> None:
        with record_lock:
            rows_now, dups_now = read_rows(rows_path) if rows_path.is_file() else ({}, [])
            attempt["finished_at"] = None if exit_status == "running" else now_iso()
            attempt["exit_status"] = exit_status
            attempt["rows_after"] = len(rows_now)
            attempt["rows_sha256_after"] = sha_file(rows_path)
            complete = (not dups_now) and set(rows_now) == set(manifest)
            atomic_json(run_path, {
                "run_id": run_id,
                "run_fingerprint": fingerprint,
                "experiment": "figure2_tom",
                "condition": CONDITION,
                "split": split,
                "canonical_split": split,
                "source_split_name": SPLITS[split]["source_split_name"],
                "fold": fold or None,
                "smoke": bool(args.limit),
                "started_at": attempts[0]["started_at"],
                "finished_at": attempt["finished_at"],
                "model_display_name": model["display"],
                "paper_column_label": model["paper_column_label"],
                "model_id": registry["model_id"],
                "model_revision": registry["model_revision"],
                "served_model": a8b.SERVED_MODEL,
                "endpoint": args.base_url,
                "endpoint_runtime": runtime,
                "expected_n": expected_n,
                "produced_n": len(rows_now),
                "duplicate_example_ids": sorted(set(dups_now)),
                "complete": complete,
                "exit_status": exit_status,
                "command": attempt["command"],
                "log_path": attempt["log_path"],
                "rows_path": str(rows_path),
                "rows_sha256": attempt["rows_sha256_after"],
                "prompt_path": str(prompt_file),
                "prompt_sha256": prompt_sha,
                "tom_insertion": (
                    "viki_amendment6.add_tom_to_messages"
                    if prompt_text == a6.TOM_REASONING_TEMPLATE
                    else "system message + '\\n\\n' + prompt file (same insertion, non-canonical text)"
                ),
                "prompt_equals_canonical_template": prompt_text == a6.TOM_REASONING_TEMPLATE,
                "seed": a8b.SEED,
                "temperature": a8b.PLAN_TEMPERATURE,
                "max_tokens": a8b.PLAN_MAX_TOKENS,
                "think_condition": "think (benchmark <think> rule kept; VIKI_NO_THINK unset)",
                "workers": args.workers,
                "client": ("OpenAI(max_retries=5, timeout=3600)" if SPLITS[split]["runner"] == "a8b"
                           else "OpenAI(library defaults)"),
                "scorer": "scripts/viki_report_matrix.py::tolerant",
                "parser": "scripts/viki_amendment9_diag102.py::parse_plan",
                "manifest_path": str(manifest_file),
                "manifest_sha256": config["manifest_sha256"],
                "sources": sources,
                "reask": "none: this arm has no re-ask path",
                "guards": {"hard_timeout_sec": args.hard_timeout_sec, "stall_min": args.stall_min},
                "attempts": attempts,
            })

    if not pending:
        write_run("ok" if set(done) == set(manifest) else "incomplete")
        print(f"{run_id}: nothing pending ({len(done)}/{expected_n})")
        return 0

    directory.mkdir(parents=True, exist_ok=True)
    write_run("running")
    print(f"{run_id}: {len(done)} done, {len(pending)} to run -> {rows_path}", flush=True)

    # Guards. The stall clock starts when the pool starts; a wedged GIL defeats this thread,
    # which is why the driver runs its own external stall watch as well.
    finished = threading.Event()
    started = time.monotonic()
    watch = {"size": rows_path.stat().st_size if rows_path.is_file() else 0, "changed": started}

    def die(status: str, code: int) -> None:
        try:
            write_run(status)
        finally:
            print(f"{run_id}: {status}", flush=True)
            os._exit(code)

    def watchdog() -> None:
        while not finished.wait(30):
            now = time.monotonic()
            size = rows_path.stat().st_size if rows_path.is_file() else 0
            if size != watch["size"]:
                watch["size"], watch["changed"] = size, now
            if now - started > args.hard_timeout_sec:
                die("timeout_hard", 4)
            if now - watch["changed"] > args.stall_min * 60:
                die(f"stalled_{args.stall_min}min", 3)

    threading.Thread(target=watchdog, daemon=True).start()

    scorer = bench.load_official_scorer(2, a5.BENCHMARK_ROOT)
    from our_method.skill_memory_v2 import Simulator
    from viki_report_matrix import tolerant

    sim = Simulator(a5.BENCHMARK_ROOT)
    viki_fork_guard.install()
    client = make_client(split, args.base_url)
    empty_sha = a8b.sha256_text("")

    def one(index: int):
        sample = a8b.native(frame.iloc[index].to_dict())
        record = manifest[index]
        zero_shot = zero_shot_messages(split, sample, record)
        messages = tom_messages(zero_shot, prompt_text)
        if not only_tom_differs(zero_shot, messages, prompt_text):
            raise Refuse(f"row {index}: ToM payload differs from zero-shot beyond the instruction")
        completion = client.chat.completions.create(
            model=a8b.SERVED_MODEL,
            messages=messages,
            temperature=a8b.PLAN_TEMPERATURE,
            max_tokens=a8b.PLAN_MAX_TOKENS,
            seed=a8b.SEED,
        )
        return index, sample, record, a5.messages_sha256(zero_shot), a5.messages_sha256(messages), completion

    try:
        with rows_path.open("a") as sink, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one, index): index for index in pending}
            written = set(done)
            for future in as_completed(futures):
                index, sample, record, zero_sha, tom_sha, completion = future.result()
                if index in written:
                    raise Refuse(f"row {index} produced twice")
                response = completion.choices[0].message.content or ""
                truth = sample["reward_model"]["ground_truth"]
                metrics = bench.score_response(scorer, 2, response, copy.deepcopy(truth), a8b.SEED)
                parsed = parse_plan(response)
                row = {
                    # the zero-shot row keys (viki_amendment8b.run_arm / viki_amendment10_run.run)
                    "index": index,
                    "arm": CONDITION,
                    "split": split,
                    "run_fingerprint": fingerprint,
                    "prompt_sha256": tom_sha,
                    "memory_prompt_sha256": empty_sha,
                    "memory_prompt_chars": 0,
                    "partner_prefix_sha256": record["partner_prefix_sha256"],
                    "response": response,
                    "response_sha256": a8b.sha256_text(response),
                    "score": metrics["score"],
                    "task_score": int(metrics["task_score"] == 1.0),
                    "format_score": int(metrics["format_score"] == 1.0),
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    # the paper export keys
                    "example_id": index,
                    "family": truth.get("task_name"),
                    "canonical_split": split,
                    "source_split_name": SPLITS[split]["source_split_name"],
                    "fold": fold or None,
                    "raw_output": response,
                    "parsed_output": json_safe(parsed),
                    "parse_success": int(parsed is not None),
                    "target": truth,
                    "success": int(tolerant(sim, response, copy.deepcopy(truth))),
                    "scorer": "viki_report_matrix.tolerant",
                    "run_id": run_id,
                    "model_id": registry["model_id"],
                    "model_display_name": model["display"],
                    "served_model": a8b.SERVED_MODEL,
                    "endpoint": args.base_url,
                    "condition": CONDITION,
                    "tom_prompt_sha256": prompt_sha,
                    "zero_shot_prompt_sha256": zero_sha,
                    "seed": a8b.SEED,
                    "temperature": a8b.PLAN_TEMPERATURE,
                    "max_tokens": a8b.PLAN_MAX_TOKENS,
                    "think_condition": "think",
                    "finish_reason": completion.choices[0].finish_reason,
                    "reask": False,
                    "created_at": now_iso(),
                }
                sink.write(json.dumps(row, sort_keys=True) + "\n")
                sink.flush()
                written.add(index)
    except BaseException as error:  # noqa: BLE001
        finished.set()
        write_run(f"error: {type(error).__name__}: {error}")
        raise
    finished.set()

    rows, duplicates = read_rows(rows_path)
    complete = not duplicates and set(rows) == set(manifest)
    write_run("ok" if complete else "incomplete")
    print(json.dumps({"run_id": run_id, "expected_n": expected_n, "produced_n": len(rows),
                      "complete": complete, "rows": str(rows_path)}, indent=2))
    return 0 if complete else 1


# ----------------------------------------------------------------------------- merge folds


def cmd_merge_folds(args: argparse.Namespace) -> int:
    _, model, _ = model_entry()
    out_root = Path(args.out_root)
    split = "ood_single_family"
    directory = cell_dir(out_root, model["slug"], split)
    manifest = a8b.load_manifest()
    fold_records, merged, duplicates = [], {}, []
    shared_keys = ("condition", "model_id", "model_revision", "served_model", "seed",
                   "temperature", "max_tokens", "prompt_sha256", "manifest_sha256", "smoke")
    reference: Dict[str, Any] = {}
    for family in foldmod.folds():
        fold_dir = cell_dir(out_root, model["slug"], split, family)
        run_path, rows_path = fold_dir / "run.json", fold_dir / "rows.jsonl"
        info = json.loads(run_path.read_text()) if run_path.is_file() else {}
        rows, dups = read_rows(rows_path)
        duplicates.extend(dups)
        for key in shared_keys:
            if info and key in reference and info.get(key) != reference[key]:
                raise Refuse(f"fold {family} disagrees with the other folds on {key}")
            if info:
                reference.setdefault(key, info.get(key))
        for index, row in rows.items():
            if index in merged:
                duplicates.append(index)
            merged[index] = row
        want = len(foldmod.rows_of(family))
        fold_records.append({
            "fold": family,
            "run_id": info.get("run_id"),
            "exit_status": info.get("exit_status", "absent"),
            "expected_n": want,
            "produced_n": len(rows),
            "complete": bool(info.get("complete")) and len(rows) == want and not dups,
            "rows_path": str(rows_path),
            "rows_sha256": sha_file(rows_path),
            "run_json_sha256": sha_file(run_path),
        })
    rows_path, run_path = directory / "rows.jsonl", directory / "run.json"
    directory.mkdir(parents=True, exist_ok=True)
    before = sha_file(rows_path)
    tmp = rows_path.with_name("rows.jsonl.tmp")
    with tmp.open("w") as sink:
        for index in sorted(merged):
            sink.write(json.dumps(merged[index], sort_keys=True) + "\n")
    tmp.replace(rows_path)
    complete = (all(f["complete"] for f in fold_records) and not duplicates
                and set(merged) == set(manifest) and len(merged) == SPLITS[split]["expected_n"])
    history = []
    if run_path.is_file():
        old = json.loads(run_path.read_text())
        history = old.get("merge_history", [])
    history.append({"merged_at": now_iso(), "rows_sha256_before": before, "rows_sha256_after": sha_file(rows_path)})
    atomic_json(run_path, {
        "run_id": "__".join([CONDITION, model["slug"], split, "merged",
                             sha_bytes(canonical([f["run_id"] for f in fold_records]).encode())[:12]]),
        "experiment": "figure2_tom",
        "condition": CONDITION,
        "split": split,
        "canonical_split": split,
        "source_split_name": "heldout",
        "construction": "8 single-family folds (viki_amendment9_folds), each its own run; "
                        "ToM holds no memory, so a fold only selects its rows",
        "model_display_name": model["display"],
        "paper_column_label": model["paper_column_label"],
        **{key: reference.get(key) for key in shared_keys},
        "expected_n": SPLITS[split]["expected_n"],
        "produced_n": len(merged),
        "duplicate_example_ids": sorted(set(duplicates)),
        "complete": complete,
        "exit_status": "ok" if complete else "incomplete",
        "command": " ".join([sys.executable] + sys.argv),
        "rows_path": str(rows_path),
        "rows_sha256": sha_file(rows_path),
        "folds": fold_records,
        "merge_history": history,
    })
    print(json.dumps({"complete": complete, "produced_n": len(merged), "folds": fold_records}, indent=2))
    return 0 if complete else 1


# ----------------------------------------------------------------------------- payload diff


def render(messages: List[Dict[str, Any]]) -> str:
    shown = copy.deepcopy(messages)
    for message in shown:
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    item["image_url"]["url"] = f"<data url, {len(url)} chars, sha256 {sha_bytes(url.encode())[:16]}>"
    return json.dumps(shown, indent=2, ensure_ascii=False)


def archived_zero_shot(split: str, fold: str, tag: str) -> Path:
    stem = f"zero_shot.{tag}.jsonl" if tag else "zero_shot.jsonl"
    if split == "id":
        return A8B_DIR / stem
    if split == "ood_single_family":
        return A8B_DIR / "folds" / fold / stem
    return A10_DIR / SPLITS[split]["source_split_name"] / stem


def cmd_payload_diff(args: argparse.Namespace) -> int:
    _, model, _ = model_entry()
    prompt_text = Path(args.prompt_file).read_bytes().decode()
    fold = args.fold or (foldmod.folds()[0] if args.split == "ood_single_family" else "")
    frame, manifest, _ = load_split(args.split, fold)
    indices = sorted(manifest)
    archive = archived_zero_shot(args.split, fold, model["zero_shot_tag"])
    archived = {}
    if archive.is_file():
        for line in archive.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                archived[int(row["index"])] = row["prompt_sha256"]
    tom_rows = {}
    if args.tom_rows:
        tom_rows, _ = read_rows(Path(args.tom_rows))
    checked = indices[: args.check_n] if args.check_n else [args.index if args.index is not None else indices[0]]
    report = {"split": args.split, "fold": fold or None, "archived_zero_shot": str(archive),
              "checked": 0, "only_tom_differs": 0, "zero_shot_sha_matches_archive": 0,
              "archive_missing": 0, "tom_sha_matches_rows": 0, "tom_rows_checked": 0}
    shown = False
    for index in checked:
        sample = a8b.native(frame.iloc[index].to_dict())
        zero_shot = zero_shot_messages(args.split, sample, manifest[index])
        tom = tom_messages(zero_shot, prompt_text)
        report["checked"] += 1
        report["only_tom_differs"] += int(only_tom_differs(zero_shot, tom, prompt_text))
        if index in archived:
            report["zero_shot_sha_matches_archive"] += int(archived[index] == a5.messages_sha256(zero_shot))
        else:
            report["archive_missing"] += 1
        if index in tom_rows:
            report["tom_rows_checked"] += 1
            report["tom_sha_matches_rows"] += int(tom_rows[index]["prompt_sha256"] == a5.messages_sha256(tom))
        if not shown:
            shown = True
            print(f"== example {index}: zero-shot sha {a5.messages_sha256(zero_shot)} "
                  f"(archived {archived.get(index)}), tom sha {a5.messages_sha256(tom)}")
            diff = difflib.unified_diff(render(zero_shot).splitlines(), render(tom).splitlines(),
                                        "zero_shot_payload", "tom_payload", lineterm="", n=1)
            print("\n".join(diff))
    print(json.dumps(report, indent=2))
    return 0


# ----------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--split", choices=list(SPLITS), required=True)
    run.add_argument("--fold", default="")
    run.add_argument("--base-url", required=True)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--limit", type=int, default=0, help="smoke only: first N manifest rows")
    run.add_argument("--out-root", default=str(OUT_ROOT))
    run.add_argument("--prompt-file", default=str(PROMPT_PATH))
    run.add_argument("--hard-timeout-sec", type=int, default=14400)
    run.add_argument("--stall-min", type=int, default=20)
    run.add_argument("--log-path", default="")
    merge = sub.add_parser("merge-folds")
    merge.add_argument("--out-root", default=str(OUT_ROOT))
    diff = sub.add_parser("payload-diff")
    diff.add_argument("--split", choices=list(SPLITS), required=True)
    diff.add_argument("--fold", default="")
    diff.add_argument("--index", type=int, default=None)
    diff.add_argument("--check-n", type=int, default=0)
    diff.add_argument("--tom-rows", default="")
    diff.add_argument("--prompt-file", default=str(PROMPT_PATH))
    args = parser.parse_args()
    try:
        if args.command == "run":
            code = cmd_run(args)
        elif args.command == "merge-folds":
            code = cmd_merge_folds(args)
        else:
            code = cmd_payload_diff(args)
    except Refuse as error:
        print(f"REFUSED: {error}", flush=True)
        raise SystemExit(2) from error
    raise SystemExit(code)


if __name__ == "__main__":
    main()
