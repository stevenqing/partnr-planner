#!/usr/bin/env python3
"""What the full condition's 896 proposal cells cost with the 72B proposer, and what the
no_trace rerun should therefore take. Read-only, zero calls.

Sources: outputs/v2_libraries/run_manifest.json (per-cell seconds for round one),
verdict mtimes (round one ran strictly sequentially, so gaps between consecutive verdicts are
cell durations; a gap far above the rest is a restart, not a cell), outputs/v3/round2.log (the
round-two start line with cell count and worker count) and round-two verdict mtimes, and the
transcripts for call and text volume. The rung does not record token usage, so tokens are an
estimate from characters (~4 chars/token), and completions are a lower bound because stored
answers are truncated.

    /root/venvs/partnr/bin/python scripts/viki_rq2_induction_cost.py
"""
from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNG = ROOT / "outputs/agentic_rung"
OUT = ROOT / "results/paper_viki_iclr2027/induction_cost_history.json"
CHARS_PER_TOKEN = 4.0


def volume(pattern: str):
    cells = moves = answer = feedback = context = 0
    for path in sorted(RUNG.glob(pattern)):
        verdict = json.loads((path.parent / "verdict.json").read_text())
        cells += 1
        moves += verdict.get("moves_used") or 0
        seen, running = set(), 0
        for record in json.loads(path.read_text()):
            if record.get("move") in seen:
                continue
            seen.add(record.get("move"))
            a = len(record.get("answer") or "")
            f = len(json.dumps(record.get("result"), default=str)[:4000])
            answer, feedback, running = answer + a, feedback + f, running + a + f
            context += running
    return {"cells": cells, "model_calls": moves, "stored_answer_chars": answer,
            "feedback_chars": feedback, "history_chars_summed_over_calls": context,
            "est_completion_tokens_lower_bound": round(answer / CHARS_PER_TOKEN),
            "est_prompt_tokens_excl_task_text": round(context / CHARS_PER_TOKEN)}


def summary(values):
    return {"n": len(values), "sum_s": round(sum(values), 1), "mean_s": round(st.mean(values), 1),
            "median_s": round(st.median(values), 1), "max_s": round(max(values), 1)} if values else None


def main() -> int:
    manifest = json.loads((ROOT / "outputs/v2_libraries/run_manifest.json").read_text())
    seconds = defaultdict(list)
    for run in manifest["runs"]:
        seconds[run["label"]].append(run["seconds"])
    family_seconds = [s for label, v in seconds.items() if label != "v2_notrace" for s in v]

    stamps = sorted(p.stat().st_mtime for p in RUNG.glob("v2_*/*/verdict.json")
                    if p.parent.parent.name != "v2_notrace")
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    restart_threshold = 20 * 60
    restarts = [g for g in gaps if g > restart_threshold]
    r1_cells = len(stamps)
    r1_mean = st.mean(family_seconds)

    log = (ROOT / "outputs/v3/round2.log").read_text()
    head = re.search(r"\[(\d\d)-(\d\d) (\d\d:\d\d:\d\d)\] (\d+) cells, (\d+) at a time, (\d+)s each", log)
    r2_stamps = sorted(p.stat().st_mtime for p in RUNG.glob("v3/*/*/verdict.json"))
    year = datetime.fromtimestamp(r2_stamps[0]).year
    r2_start = datetime.strptime("%d-%s-%s %s" % (year, head.group(1), head.group(2), head.group(3)),
                                 "%Y-%m-%d %H:%M:%S").timestamp()
    r2_cells, r2_workers = int(head.group(4)), int(head.group(5))
    r2_wall = r2_stamps[-1] - r2_start
    r2_cell_equiv = r2_wall * r2_workers / r2_cells

    notrace = seconds.get("v2_notrace", [])
    definition = json.loads((ROOT / "results/frozen_sweep_v2.json").read_text())
    nt_r1_cells = sum(len(e["rungs"]) for e in definition["libraries"] if e["status"] == "ok")
    nt_mean = st.mean(notrace) if notrace else r1_mean
    per_family = defaultdict(int)
    for e in definition["libraries"]:
        if e["status"] == "ok":
            per_family[e["family"]] += len(e["rungs"])
    report = {
        "round1": {
            "driver": "scripts/drivers/viki_v2_libraries.py (subprocess.run per cell: concurrency 1)",
            "concurrency": 1,
            "cells_with_verdict": r1_cells,
            "manifest_cell_seconds": summary(family_seconds),
            "manifest_covers_cells": len(family_seconds),
            "manifest_note": "the manifest is rewritten when the driver restarts; cells from the "
                             "earlier attempt are missing from it and are covered by verdict mtimes",
            "verdict_span_first_to_last_h": round((stamps[-1] - stamps[0]) / 3600, 2),
            "restart_gaps_over_20min_s": [round(g) for g in restarts],
            "active_wall_h": round((stamps[-1] - stamps[0] - sum(restarts)) / 3600, 2),
            "est_compute_h_at_manifest_mean": round(r1_cells * r1_mean / 3600, 2),
            "first_verdict": datetime.fromtimestamp(stamps[0]).isoformat(),
            "last_verdict": datetime.fromtimestamp(stamps[-1]).isoformat(),
            "volume": volume("v2_[!n]*/*/transcript.json"),
        },
        "round2": {
            "driver": "scripts/drivers/viki_v3_round2.sh",
            "cells": r2_cells, "concurrency": r2_workers, "cell_timeout_s": int(head.group(6)),
            "start": datetime.fromtimestamp(r2_start).isoformat(),
            "last_verdict": datetime.fromtimestamp(r2_stamps[-1]).isoformat(),
            "wall_h": round(r2_wall / 3600, 2),
            "cell_seconds_equivalent": round(r2_cell_equiv, 1),
            "volume": volume("v3/*/*/transcript.json"),
        },
        "old_no_trace_reference": {"cells": len(notrace), "cell_seconds": summary(notrace),
                                   "volume": volume("v2_notrace/*/transcript.json")},
        "no_trace_rerun_projection": {
            "basis": "old no-trace cells' mean seconds (72B, same endpoint, sequential, 18 moves); "
                     "round 2 scaled by 20/18 moves and by the round-2 worker count; "
                     "ranges use full's own per-cell seconds as the other bound",
            "round1_cells": nt_r1_cells,
            "round1_sequential_h": round(nt_r1_cells * nt_mean / 3600, 1),
            "round1_sequential_h_at_full_rate": round(nt_r1_cells * r1_mean / 3600, 1),
            "round1_with_4_family_drivers_h_if_no_contention": round(
                nt_r1_cells * nt_mean / 3600 / min(4, len(per_family)), 1),
            "round2_cells": r2_cells,
            "round2_wall_h_at_%d_workers" % r2_workers: round(r2_cells * nt_mean * 20 / 18 / r2_workers / 3600, 2),
            "round2_wall_h_at_full_rate": round(r2_wall / 3600, 2),
            "caveat": "4 concurrent requests on one endpoint were ~%.1fx slower per cell than "
                      "sequential in full (round-2 cell-equivalent %.0f s vs round-1 mean %.0f s); "
                      "some of that is round 2's longer interface cells"
                      % (r2_cell_equiv / r1_mean, r2_cell_equiv, r1_mean),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
