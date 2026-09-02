#!/usr/bin/env python3
"""The model plans in the memory's vocabulary: it chooses operators, the memory expands them.

Two ways of handing v2 to a language model have already been measured and one of them
fails. Written out as prose for the model to imitate, the library scores 5.8% -- worse
than v1's own trajectories -- because an abstract body is not something to copy and the
model fills the variables with objects that do not exist. Given to a planner, the same
library scores 98.2%, but then the model has said only what must become true and the
objection writes itself: the gain might be the planner.

This is the middle, and it is the one that keeps the model in the loop where the
thinking is. The memory offers what it knows how to do as a numbered vocabulary -- with
costs and the situation each suits, and without the bodies. The model reads the picture
and the instruction and answers with the work: which operators, bound to which objects,
run by which robots. The memory expands that into actions it knows are legal, and the
scheduler packs them into steps.

So the division is: the model decides what has to happen, with what, and by whom; the
memory supplies how, and guarantees the how is executable. Nothing about the plan's
content comes from the memory, and nothing about the actions comes from the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
from openai import OpenAI

from habitat_llm.evaluation import viki_bench as bench
from viki_amendment5 import BENCHMARK_ROOT
from viki_amendment11_goalparse import extract_json
from our_method.skill_memory_v2 import SEED, SkillMemoryV2, Simulator, planner

OUT = ROOT / "results/viki_memory_experiments/amendment11"
MANIFEST = ROOT / "results/viki_memory_experiments/amendment8b/interactive_manifest.jsonl"
SERVED_MODEL = "qwen2.5-vl-72b-amendment3-f2"
BASE_URL = "http://192.168.32.40:8050/v1"

INSTRUCTION = """\
Ignore the output format described above. Do NOT write out individual actions and do \
NOT use <think> or <answer> tags.

You have a memory of ways to get things done, each one already carried out successfully \
in earlier tasks. Choose from it.

{menu}

Reply with one JSON object and nothing else:

{{
  "work": [
    {{"op": "S5", "X": "<object>", "Y": "<where it must end up>", "robots": ["R1"]}},
    {{"op": "S4", "X": "<object to use>", "Z1": "<where it is used>", "robots": ["R2"]}}
  ]
}}

Rules:
- One entry per thing the task requires. Leave out anything already true in the picture.
- Pick the entry whose situation matches the scene, and among those the cheapest: the \
task is scored on the number of steps.
- "robots" lists the robots that carry out that entry, one name per role, in role order.
- Give every argument the entry asks for, naming objects and places exactly as a robot \
command would address them.
- Robots may work at the same time; two entries do not have to share a robot.
"""


def expand(memory, catalogue, work, env, robots):
    """Turn the model's choices into chains the scheduler can run."""
    chains, used = [], 0
    for item in work:
        if not isinstance(item, dict):
            continue
        operator = catalogue.get(str(item.get("op", "")).strip().upper())
        if operator is None:
            continue
        binding: Dict[str, str] = {}
        subject = memory.canonical_asset(item.get("X"), sorted(env.assets))
        if subject is None:
            continue
        binding["?x"] = subject
        if operator["effect"]["key"] == "pos.name":
            where = item.get("Y")
            if not isinstance(where, str):
                continue
            binding["?y"] = memory.canonical_place(where)
        for number, spare in enumerate(memory.spare_variables(operator), 1):
            value = item.get(f"Z{number}") or item.get(spare)
            bound = memory.canonical_asset(value, sorted(env.assets)) if value else None
            if bound is None:
                fits = [name for name, asset in env.assets.items()
                        if name not in binding.values()
                        and memory.suits(asset, operator.get("types", {}).get(spare, {}))]
                bound = fits[0] if fits else None
            if bound is None:
                continue
            binding[spare] = bound
        crew = [name for name in (item.get("robots") or []) if name in robots]
        if operator.get("coordinated"):
            if len(crew) < len(operator["roles"]):
                crew = crew + [name for name in robots if name not in crew]
            group = f"m{used}"
            used += 1
            for slot, role in enumerate(operator["roles"]):
                actions = [[entry["action"][0]] +
                           [t if t.startswith("?r") else binding.get(t, t) for t in entry["action"][1:]]
                           for entry in role["actions"]]
                if any(t.startswith("?") and not t.startswith("?r") for a in actions for t in a[1:]):
                    break
                chains.append({"actions": actions, "after": [e["after"] for e in role["actions"]],
                               "guard": [], "group": group, "role": slot,
                               "robot": crew[slot] if slot < len(crew) else robots[slot % len(robots)]})
        else:
            body = [[a[0]] + [binding.get(t, t) for t in a[1:]] for a in operator["body"]]
            if any(t.startswith("?") for a in body for t in a[1:]):
                continue
            chains.append({"actions": body, "guard": [],
                           "robot": crew[0] if crew else robots[0]})
    return chains


def run(memory, sim, metadata, chains, robots):
    """Schedule exactly what the model asked for; if that cannot run, only re-cast the crew."""
    def attempt(assignment):
        plans = defaultdict(list)
        for chain, robot in zip(chains, assignment):
            plans[robot].append({k: v for k, v in chain.items() if k != "robot"})
        return planner.schedule(metadata, dict(plans), sim)

    given = [chain["robot"] for chain in chains]
    steps = attempt(given)
    if steps is not None:
        return steps, False
    for assignment in permutations(robots, len(chains)) if len(chains) <= len(robots) else []:
        steps = attempt(list(assignment))
        if steps is not None:
            return steps, True
    for assignment in [[robot] * len(chains) for robot in robots]:
        steps = attempt(assignment)
        if steps is not None:
            return steps, True
    return None, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, default=OUT / "skill_memory_v2.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--tag", default="operator_choice")
    arguments = parser.parse_args()

    sim = Simulator(BENCHMARK_ROOT)
    memory = SkillMemoryV2.load(arguments.memory)
    menu, catalogue = memory.menu()
    manifest = {int(json.loads(l)["index"]): json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()}
    indices = sorted(manifest)
    if arguments.limit:
        indices = indices[: arguments.limit]
    frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
    client = OpenAI(api_key="EMPTY", base_url=BASE_URL, max_retries=5, timeout=3600)
    lock, done = threading.Lock(), [0]

    def work_on(index: int) -> Dict[str, Any]:
        sample = bench.to_native(frame.iloc[index].to_dict())
        truth = bench.get_ground_truth(sample)
        blind = {k: v for k, v in truth.items() if k != "time_steps"}
        metadata = sim.metadata(blind, SEED)
        env = sim.world(metadata)
        messages = bench.get_messages(sample)
        user = next(m for m in reversed(messages) if m["role"] == "user")
        block = INSTRUCTION.format(menu=menu)
        content = user["content"]
        if isinstance(content, list):
            item = next(i for i in content if i["type"] == "text")
            item["text"] = f"{item['text']}\n\n{block}"
        else:
            user["content"] = f"{content}\n\n{block}"
        record = {"index": index, "task_name": truth["task_name"], "accuracy": 0.0, "reason": ""}
        try:
            completion = client.chat.completions.create(
                model=SERVED_MODEL, messages=messages, temperature=0, max_tokens=900)
            text = completion.choices[0].message.content or ""
        except Exception as error:
            record["reason"] = f"REQUEST_FAILED:{type(error).__name__}"
            return record
        record["raw"] = text[-3000:]
        parsed = extract_json(text)
        if parsed is None or not isinstance(parsed.get("work"), list):
            record["reason"] = "UNPARSEABLE"
            return record
        chains = expand(memory, catalogue, parsed["work"], env, list(metadata["agents"]))
        if not chains:
            record["reason"] = "NO_USABLE_CHOICE"
            return record
        steps, recast = run(memory, sim, metadata, chains, list(metadata["agents"]))
        record["recast"] = recast
        if steps is None:
            record["reason"] = "NO_SCHEDULE"
            return record
        record["plan_len"] = len(steps)
        record["budget"] = len(truth["time_steps"])
        record["accuracy"] = sim.score(steps, truth, SEED)
        record["reason"] = ("SOLVED" if record["accuracy"] == 1.0
                            else "OVER_BUDGET" if len(steps) > record["budget"] else "GOAL_UNMET")
        return record

    records = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        futures = [pool.submit(work_on, index) for index in indices]
        for future in as_completed(futures):
            record = future.result()
            with lock:
                records.append(record)
                done[0] += 1
                if done[0] % 100 == 0:
                    hit = sum(r["accuracy"] for r in records)
                    print(f"  {done[0]}/{len(indices)}  solved={hit:.0f} "
                          f"({hit / len(records) * 100:.1f}%)", flush=True)

    records.sort(key=lambda r: r["index"])
    with (OUT / f"{arguments.tag}.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    hit = sum(r["accuracy"] for r in records)
    print(f"\n=== {arguments.tag}: the model chooses operators, {len(records)} rows ===")
    print(f"menu           {len(menu)} chars (~{len(menu)//4} tokens), {len(catalogue)} entries")
    print(f"accuracy       {hit:.0f}/{len(records)} = {hit / len(records) * 100:.2f}%")
    print(f"crew re-cast   {sum(1 for r in records if r.get('recast')) / len(records) * 100:.1f}% of rows")
    for reason, count in Counter(r["reason"] for r in records).most_common():
        print(f"  {reason:<20} {count:>5}  {count / len(records) * 100:5.1f}%")
    families = defaultdict(lambda: [0, 0])
    for record in records:
        families[record["task_name"]][0] += int(record["accuracy"] == 1.0)
        families[record["task_name"]][1] += 1
    print("\nby family:")
    for family, (won, total) in sorted(families.items(), key=lambda kv: -kv[1][1]):
        print(f"  {family:<48} {won:>4}/{total:<4} {won / total * 100:5.1f}%")
    print(f"\nwrote {OUT / (arguments.tag + '.jsonl')}")


if __name__ == "__main__":
    main()
