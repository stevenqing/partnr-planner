"""Replay a model's own intent answers against any operator library, on the ID split.

The point the user made, and it was right: an operator library is a data artefact and does
not belong to a model. What is model-specific is the *answer* the library is then asked to
execute -- and `m30_id.jsonl` / `m7_id.jsonl` already hold those, for 30B and 7B, over the
same 924 ID rows.

I first tried to score them with `viki_eval_skill_memory_v2.py`, got 0.00% for every library
including the reference, and concluded the sets were unusable and had to be regenerated.
That was wrong, and the tell was there: the reference scoring zero means the harness is
broken, not the memory. Those files are in the **intent** schema
(`{"work": [{"do", "X", "Y", "robots"}]}`), which a different evaluator reads. Nothing needs
generating; it needed the right reader.

So this replays: the model's answers come off disk unchanged, the library is swapped
underneath them, and Layers 2 and 3 are the reference's throughout. No endpoint is called,
and the same answers are scored for every library, so nothing here can move because a model
was sampled again.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from habitat_llm.evaluation import viki_bench as bench  # noqa: E402
from viki_amendment5 import BENCHMARK_ROOT  # noqa: E402
from viki_amendment11_goalparse import extract_json  # noqa: E402
from viki_eval_skill_memory_v2 import visits_of  # noqa: E402
from viki_eval_v2_intent_choice import to_requirement  # noqa: E402
from viki_intent_crew import collect, solve as solve_with_crew  # noqa: E402
from our_method.skill_memory_v2 import SEED, SkillMemoryV2, Simulator  # noqa: E402

OUT = ROOT / "results/viki_memory_experiments/amendment11"

# 72B's own intent answers. Which of `intent_clean` / `intent_full` / `intent_partner` is
# the archived 91.56% cell is settled by reproducing that number with the reference library,
# not by the filename -- the same discipline that caught two wrong readers today.
MODELS = [("72B", "intent_clean"), ("30B", "m30_id"), ("7B", "m7_id")]
LIBRARIES = [
    ("reference", OUT / "skill_memory_v2.json"),
    ("agentic", ROOT / "outputs/agentic_memory_runner.json"),
]


def solve(truth, record, memory, sim):
    blind = {k: v for k, v in truth.items() if k != "time_steps"}
    metadata = sim.metadata(blind, SEED)
    env = sim.world(metadata)
    parsed = extract_json(record.get("raw") or "")
    work = (parsed or {}).get("work") if isinstance(parsed, dict) else None
    if not isinstance(work, list):
        return 0.0, "UNPARSEABLE"
    requirements, crew = collect(work, to_requirement, memory,
                                 sorted(metadata["assets"]), metadata)
    if not requirements:
        return 0.0, "NO_USABLE_WORK"
    temporal = memory.order_for(requirements, visits_of(env, requirements, memory))
    plan, reason, _ = solve_with_crew(blind, memory, sim, SEED, requirements,
                                      crew, temporal, False)
    accuracy = sim.score(plan, truth, SEED) if plan else 0.0
    if plan and accuracy == 0.0:
        reason = "OVER_BUDGET" if len(plan) > len(truth["time_steps"]) else "GOAL_UNMET"
    return accuracy, ("SOLVED" if accuracy == 1.0 else reason)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/viki_intent_replay_matrix.json")
    arguments = parser.parse_args()

    sim = Simulator(BENCHMARK_ROOT)
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")

    table = {}
    for model, responses in MODELS:
        records = [json.loads(line) for line in
                   (OUT / f"{responses}.jsonl").read_text().splitlines() if line.strip()]
        if arguments.limit:
            records = records[: arguments.limit]
        for library, path in LIBRARIES:
            memory = SkillMemoryV2.load(path)
            solved, reasons, families = 0, Counter(), {}
            for record in records:
                truth = bench.get_ground_truth(
                    bench.to_native(frame.iloc[record["index"]].to_dict()))
                accuracy, reason = solve(truth, record, memory, sim)
                reasons[reason] += 1
                solved += int(accuracy == 1.0)
                seen, got = families.get(record["task_name"], (0, 0))
                families[record["task_name"]] = (seen + 1, got + int(accuracy == 1.0))
            table["%s/%s" % (library, model)] = {
                "model": model, "library": library, "responses": responses,
                "solved": solved, "total": len(records),
                "rate": round(solved / len(records), 4),
                "reasons": dict(reasons),
                "by_family": {k: list(v) for k, v in sorted(families.items())},
            }
            print("%-10s x %-4s  %d/%d = %.2f%%  %s"
                  % (library, model, solved, len(records), 100 * solved / len(records),
                     dict(reasons)), flush=True)

    arguments.out.write_text(json.dumps(table, indent=1))
    print("\n-> %s" % arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
