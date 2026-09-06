#!/usr/bin/env python3
"""The model says what to do and who does it; the memory decides how.

The first attempt at giving the model the memory's vocabulary listed the bodies with
the conditions each suits -- `use when the destination is a container that starts shut`
-- and asked it to choose. It chose sensibly and scored 32%, and reading the failures
says why: on clear_table it asked for the plain delivery twice, which is exactly right
except that the cupboard is shut, and there is no way to know that from the picture. The
applicability of a body depends on hidden state -- whether a container is closed, whether
an object is inside one, whether a robot can walk -- and asking a model to guess it is
asking for something the world does not show. G-Memory's arm never faces that choice: it
copies a trajectory from a nearly identical scene, with the right variant already baked
in.

So the vocabulary here is intents, not bodies. Put X at Y. Use X. Open X. Four entries,
no preconditions, nothing hidden to guess. The model still decides what work the task
needs, what to bind it to, and which robot runs it -- the decisions that need the picture
and the sentence -- and the memory chooses the body that fits the world it can actually
inspect.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import permutations, product
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
SPLITS = {
    "id": None,
    "recombination-imaged": "recombination.imaged.parquet",
    "recombination-text": "recombination.text.parquet",
}

# An earlier version of this block added "if it says to cut something, the knife has to
# be used; if it says to toast, the toaster has to be run." That sentence was written
# after reading which families failed on the test split, which makes any number it
# produces partly a report on the test set. It is gone. What it was compensating for --
# knowing that a verb in the instruction implies a predicate -- is Layer 2's job to learn
# from training episodes, and the cost of not yet having learned it shows up here.
REASK = """\
That assignment cannot be carried out. The planner reported: %s
The assignment you gave was: %s

Reassign the work. Keep the same entries; change only which robot does what, so that every entry can be carried out by the robot you name. Reply with one JSON object in the same format and nothing else."""


INSTRUCTION = """\
Ignore the output format described above. Do NOT write out individual actions and do \
NOT use <think> or <answer> tags.

You have a memory of how to carry things out. It already knows the action sequences, \
how to open what is shut, how to reach what is stowed away, and how robots hand things \
between them. You do not need to describe any of that. Say only what has to be done, to \
what, and by whom.

Things the memory knows how to do:
  put    move an object to a place    arguments: X (the object), Y (where it must end up)
  use    use, operate, switch on, or cut with an object    arguments: X (the object used),
         and optionally Y (the place it must be used at)
  open   open a container            arguments: X (the container)

Reply with one JSON object and nothing else:

{
  "work": [
    {"do": "put", "X": "<object>", "Y": "<place>", "robots": ["R1"]},
    {"do": "use", "X": "<object>", "Y": "<place it is used at>", "robots": ["R2"]}
  ]
}

Rules:
- One entry per thing the task requires to be true at the end, plus anything that must \
become true along the way.
- Leave out anything the picture already shows to be true.
- Do NOT add entries for opening containers or fetching things out of them unless the \
task itself asks for it; the memory handles that by itself.
- "robots" names the robots that carry the entry out. Robots work in parallel, so give \
different entries to different robots wherever you can -- the task is scored on the \
number of steps.
- Name objects and places exactly as a robot command would address them.
"""


def to_requirement(memory, item, scene_assets, ground: bool = True) -> Optional[Dict[str, Any]]:
    """`ground=False` is the `w/o Grounding` ablation: names are taken as written."""
    kind = str(item.get("do", "")).strip().lower()
    raw_subject = item.get("X")
    subject = (memory.canonical_asset(raw_subject, scene_assets) if ground
               else (raw_subject if raw_subject in scene_assets else None))
    if subject is None:
        return None
    where = item.get("Y")
    if kind == "put":
        if not isinstance(where, str):
            return None
        status = {"pos.name": memory.canonical_place(where) if ground else where.strip()}
    elif kind in ("use", "open"):
        status = {"is_activated": True} if kind == "use" else {"pos.name": subject}
    else:
        return None
    return {"type": "asset", "name": subject, "is_satisfied": True, "status": status}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, default=OUT / "skill_memory_v2.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--model", default=SERVED_MODEL)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--split", choices=sorted(SPLITS), default="id")
    parser.add_argument("--partner", action="store_true",
                        help="add the partner prefix the archived plan-writing arms were given")
    parser.add_argument("--free-crew", action="store_true",
                        help="let the memory cast the robots instead of the model")
    parser.add_argument("--tag", default="intent_choice")
    # The prompt is `user content + INSTRUCTION`, a module-level constant; the memory is
    # consulted only AFTER the answer arrives. So a model's answer does not depend on which
    # library is loaded, and one archived answer set can be scored against any number of
    # libraries with no further generation. The endpoint is still needed for the fallback
    # re-ask, which is per-row and rare.
    parser.add_argument("--replay", type=Path, default=None,
                        help="score these archived answers instead of generating new ones")
    parser.add_argument("--no-order", action="store_true",
                        help="ablation: do not apply Layer 2's ordering")
    parser.add_argument("--no-grounding", action="store_true",
                        help="ablation: do not ground names through Layer 3")
    arguments = parser.parse_args()

    sim = Simulator(BENCHMARK_ROOT)
    memory = SkillMemoryV2.load(arguments.memory)
    # The recombination split is a parquet of its own with no manifest and, in its text
    # form, no image; everything else about the pipeline is the same, which is the point
    # of running it at all.
    if arguments.split == "id":
        manifest = {int(json.loads(l)["index"]): json.loads(l)
                    for l in MANIFEST.read_text().splitlines() if l.strip()}
        frame = pd.read_parquet(BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet")
        indices = sorted(manifest)
    else:
        manifest = {}
        frame = pd.read_parquet(
            ROOT / "results/viki_memory_experiments/amendment10" / SPLITS[arguments.split]
        )
        indices = list(range(len(frame)))
    imaged = arguments.split != "recombination-text"
    if arguments.limit:
        indices = indices[: arguments.limit]
    client = OpenAI(api_key="EMPTY", base_url=arguments.base_url, max_retries=5, timeout=3600)
    lock, done = threading.Lock(), [0]

    replay: Dict[int, str] = {}
    if arguments.replay:
        for line in arguments.replay.read_text().splitlines():
            if not line.strip():
                continue
            archived = json.loads(line)
            if archived.get("raw") is not None:
                replay[int(archived["index"])] = archived["raw"]
        print("replaying %d archived answers from %s" % (len(replay), arguments.replay))

    def work_on(index: int) -> Dict[str, Any]:
        sample = bench.to_native(frame.iloc[index].to_dict())
        truth = bench.get_ground_truth(sample)
        blind = {k: v for k, v in truth.items() if k != "time_steps"}
        metadata = sim.metadata(blind, SEED)
        if imaged:
            messages = bench.get_messages(sample)
        else:
            messages = [{"role": m["role"], "content": m["content"]}
                        for m in bench.to_native(sample["prompt"])]
            if any("<image>" in str(m["content"]) for m in messages):
                record = {"index": index, "task_name": truth.get("task_name", "?"),
                          "accuracy": 0.0, "reason": "IMAGE_MARKER_IN_TEXT_SPLIT"}
                return record
        user = next(m for m in reversed(messages) if m["role"] == "user")
        # The archived plan-writing arms were told what a partner robot does at step 1,
        # which is a line of the reference plan. This arm was run without it; the flag
        # puts it back, so that whatever this arm scores against them is not scored with
        # less help than they had.
        extra = (manifest[index]["partner_prefix"] + "\n\n" + INSTRUCTION
                 if arguments.partner and index in manifest else INSTRUCTION)
        content = user["content"]
        if isinstance(content, list):
            item = next(i for i in content if i["type"] == "text")
            item["text"] = f"{item['text']}\n\n{extra}"
        else:
            user["content"] = f"{content}\n\n{extra}"
        record = {"index": index, "task_name": truth.get("task_name", "?"),
                  "accuracy": 0.0, "reason": ""}
        if arguments.replay is not None:
            if index not in replay:
                record["reason"] = "NO_ARCHIVED_ANSWER"
                return record
            text = replay[index]
            record["replayed"] = True
        else:
            try:
                completion = client.chat.completions.create(
                    model=arguments.model, messages=messages, temperature=0, max_tokens=700)
                text = completion.choices[0].message.content or ""
            except Exception as error:
                record["reason"] = f"REQUEST_FAILED:{type(error).__name__}"
                return record
        record["raw"] = text[-3000:]
        parsed = extract_json(text)
        work = (parsed or {}).get("work") if isinstance(parsed, dict) else None
        if not isinstance(work, list):
            record["reason"] = "UNPARSEABLE"
            return record

        scene = sorted(metadata["assets"])

        def read_work(items):
            requirements, crew = [], []
            for item in items:
                requirement = (to_requirement(memory, item, scene,
                                              ground=not arguments.no_grounding)
                               if isinstance(item, dict) else None)
                if requirement is None:
                    continue
                requirements.append(requirement)
                crew.append([name for name in (item.get("robots") or [])
                             if name in metadata["agents"]])
            return requirements, crew

        requirements, crew = read_work(work)
        if not requirements:
            record["reason"] = "NO_USABLE_WORK"
            return record
        record["stated"] = len(requirements)

        # The model's ordering is not asked for; Layer 2 supplies it.
        from viki_eval_skill_memory_v2 import visits_of
        env = sim.world(metadata)
        temporal = ([] if arguments.no_order
                    else memory.order_for(requirements, visits_of(env, requirements, memory)))
        blind["goal_constraints"] = [[requirement] for requirement in requirements]
        blind["temporal_constraints"] = temporal

        def cast_from(requirement_list, crew_list):
            if arguments.free_crew:
                return None
            out = {}
            for requirement, names in zip(requirement_list, crew_list):
                if names:
                    out[planner.predicate_key(requirement)] = names[0]
            return out

        casting = cast_from(requirements, crew)
        plan, reason = planner.plan(blind, memory, sim, SEED, crew=casting)
        record["cast_by_model"] = bool(casting)
        if plan is None and casting:
            # The fallback rule, fixed before the run. The model is told what made its
            # assignment infeasible and asked once more; if the second answer is still
            # infeasible the row is unsolved and recorded as such.
            #
            # This replaces two earlier steps, both removed. Dropping the temporal
            # constraints and retrying ran the `w/o Ordering` ablation inside the main
            # arm; falling back to the memory's own free search scored a memory-dispatch
            # row inside an LLM-dispatch arm.
            record["reask"] = True
            try:
                followup = list(messages) + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": REASK % (reason or "no feasible plan",
                                                         json.dumps(casting, sort_keys=True))},
                ]
                again = client.chat.completions.create(
                    model=arguments.model, messages=followup, temperature=0, max_tokens=700)
                second = again.choices[0].message.content or ""
                usage = getattr(again, "usage", None)
                record["reask_prompt_tokens"] = getattr(usage, "prompt_tokens", None)
                record["reask_completion_tokens"] = getattr(usage, "completion_tokens", None)
            except Exception as error:                               # noqa: BLE001
                record["reason"] = "infeasible_assignment"
                record["reask_error"] = type(error).__name__
                return record
            record["raw_reask"] = second[-3000:]
            reparsed = extract_json(second)
            rework = (reparsed or {}).get("work") if isinstance(reparsed, dict) else None
            if isinstance(rework, list):
                requirements2, crew2 = read_work(rework)
                if requirements2:
                    env2 = sim.world(metadata)
                    blind["goal_constraints"] = [[r] for r in requirements2]
                    blind["temporal_constraints"] = ([] if arguments.no_order
                        else memory.order_for(requirements2,
                                              visits_of(env2, requirements2, memory)))
                    plan, reason = planner.plan(blind, memory, sim, SEED,
                                                crew=cast_from(requirements2, crew2))
            if plan is None:
                record["reason"] = "infeasible_assignment"
                return record
        record["reason"] = reason
        if plan:
            record["plan_len"] = len(plan)
            record["budget"] = len(truth["time_steps"])
            record["accuracy"] = sim.score(plan, truth, SEED)
            if record["accuracy"] == 0.0:
                record["reason"] = ("OVER_BUDGET" if len(plan) > record["budget"] else "GOAL_UNMET")
            else:
                record["reason"] = "SOLVED"
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
    print(f"\n=== {arguments.tag}: the model names the work, {len(records)} rows ===")
    print(f"accuracy       {hit:.0f}/{len(records)} = {hit / len(records) * 100:.2f}%")
    print(f"crew           {'model' if not arguments.free_crew else 'memory'}; "
          f"re-asked on {sum(1 for r in records if r.get('reask')) / len(records) * 100:.1f}% "
          f"of rows; infeasible_assignment "
          f"{sum(1 for r in records if r.get('reason') == 'infeasible_assignment')}"
          f"/{len(records)}")
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
