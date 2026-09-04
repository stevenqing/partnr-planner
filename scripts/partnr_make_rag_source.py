#!/usr/bin/env python3
"""Build a retrieval pool for PARTNR's own trajectory RAG out of recorded rollouts.

The baseline's RAG wants a directory holding an `episode_result_log.csv` and a tree of
per-agent trace files, and it keeps the episodes whose logged `task_state_success` is 1.
The repository's shared result log is not that directory: it is appended to by every run
of every split, so an id that succeeded in `train_2k` would select a `train_mini` trace of
the same number. This writes a log containing one split's own episodes and nothing else.

The traces come from the privileged scripted planner, which makes this pool *stronger*
than the one the baseline was designed around -- optimal demonstrations rather than a
model's lucky episodes. That asymmetry favours the baseline, which is the direction an
asymmetry should run when the baseline is the thing you are arguing against.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

FIELDS = [
    "episode_id",
    "instruction",
    "run_id",
    "runtime",
    "sim_step_count",
    "task_percent_complete",
    "task_state_success",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="directory holding <split>.json.gz/")
    parser.add_argument("--split", default="train_mini")
    parser.add_argument("--out", default="results/rag_source_train_mini")
    parser.add_argument("--link", action="store_true", help="symlink the traces instead of copying")
    parser.add_argument(
        "--types", nargs="*", default=None,
        help="keep only episodes of these task types (e.g. R). Omit for every type.")
    parser.add_argument("--task-types", default="results/partnr_task_types.json",
                        help="the classification written by scripts/partnr_task_types.py")
    arguments = parser.parse_args()

    # A pool restricted by task type is what puts a prompt-shaped baseline on the same
    # compositional axis as skill memory v2. The operators were induced from the
    # rearrange-only half of this split, so a retrieval pool holding every type is a
    # baseline that has seen spatial and temporal work the operators never did -- the
    # unrestricted arm answers "is retrieval enough", the restricted one answers "does
    # this memory compose", and only the second is the paper's question.
    keep = None
    if arguments.types:
        classification = json.loads(Path(arguments.task_types).read_text())[arguments.split]
        keep = {str(episode) for kind in arguments.types
                for episode in classification.get(kind, [])}
        if not keep:
            raise SystemExit(f"no {arguments.split} episodes of type {arguments.types}")

    source = Path(arguments.results) / f"{arguments.split}.json.gz"
    out = Path(arguments.out)
    target = out / f"{arguments.split}.json.gz"
    out.mkdir(parents=True, exist_ok=True)

    rows, successes, read, unusable = [], 0, 0, 0
    for path in sorted((source / "stats").glob("*.json")):
        read += 1
        if keep is not None and path.stem not in keep:
            continue
        try:
            record = json.loads(path.read_text())
            if not record.get("success"):
                unusable += 1
                continue
            stats = json.loads(record["stats"])
        except Exception:
            unusable += 1
            continue
        episode = path.stem
        # Only a trace the loader can actually open is worth selecting.
        if not all(
            (source / "traces" / str(agent) / f"trace-episode_{episode}_0-{agent}.txt").is_file()
            for agent in (0, 1)
        ):
            unusable += 1
            continue
        success = float(stats.get("task_state_success", 0.0))
        successes += int(success == 1.0)
        rows.append(
            {
                "episode_id": episode,
                # The loader splits the line on spaces and reads the first and last
                # fields, so the instruction is replaced by a placeholder: the real one
                # is read out of the trace file itself, and commas in it would shift the
                # columns the loader counts on.
                "instruction": "-",
                "run_id": 0,
                "runtime": float(stats.get("runtime", 0.0)),
                "sim_step_count": float(stats.get("sim_step_count", 0.0)),
                "task_percent_complete": float(stats.get("task_percent_complete", 0.0)),
                "task_state_success": success,
            }
        )

    if target.exists() or target.is_symlink():
        if target.is_symlink():
            target.unlink()
        else:
            shutil.rmtree(target)
    if arguments.link:
        target.symlink_to(source.resolve(), target_is_directory=True)
    else:
        shutil.copytree(source, target)

    with (out / "episode_result_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    kept = "every type" if keep is None else f"types {arguments.types} ({len(keep)} episodes)"
    print(f"read {read} stats, kept {kept}, wrote {len(rows)} rows, {unusable} unusable")
    print(f"retrievable examples (state_success == 1): {successes}")
    print(f"pool at {out}  (rag_dataset_dir=['{out}/'], rag_data_source_name=['{arguments.split}.json.gz'])")


if __name__ == "__main__":
    main()
