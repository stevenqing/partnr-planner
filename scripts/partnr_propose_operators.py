#!/usr/bin/env python3
"""Propose operators from recorded PARTNR rollouts. Proposing only -- nothing is accepted here.

On VIKI-L2 the agent tests its own candidate inside the loop, because `run_operator`
replays an episode in microseconds. PARTNR has no such oracle: the only causal test is to
put the operator in the memory and run the benchmark, which takes minutes per episode. So
the loop is cut in two. This half reads rollouts and emits candidates; acceptance happens
outside, by execution, in `partnr_gate_batch.sh`.

That split is also a guard. The agent never sees an end-to-end score, so it cannot tune a
body against the set the gate measures on, and the gate set and the reported split are
different pools of different episodes (`results/partnr_pools/ledger.json`).

`predicts` remains available and remains advisory, and the prompt says so. Its precision
was calibrated on 2026-09-07 against deliberately broken operators and does not separate
them: 21 factory operators pass a 0.5 threshold 6 times, 42 broken ones 12 times. It still
answers one thing honestly -- whether a body ever matches anything -- and a body that
matches nothing is a body nobody has looked at.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from openai import OpenAI
from partnr_groundable import check as groundable
from partnr_induction_tools import Workbench
from viki_agentic_rung_abstraction import extract_request, normalise_request

TOOLS = """
{"tool": "list_traces", "args": {"effect": "is_in_room", "limit": 10}}
    the rollouts available, and which effects each one records becoming true.
{"tool": "show_trace", "args": {"index": 0, "max_steps": 30}}
    one rollout: every agent's action at each planner step, and for each proposition the
    step at which the recording says it became true.
{"tool": "held_by", "args": {"index": 0, "entity": "cup_0", "step": 4}}
    which agent was holding that entity at that step. For an effect nobody's action names
    -- an object is in a room because somebody carried it there -- this is how the actor
    is found at all.
{"tool": "actor_window", "args": {"index": 0, "actor": "0", "end": 12, "back": 12}}
    that agent's own actions leading up to that step. This is where a body comes from.
{"tool": "room_of", "args": {"index": 0, "entity": "stool_26"}}
    which room a piece of furniture is in, and {"tool": "rooms", "args": {"index": 0}}
    for the whole scene. The actions name furniture and the goal names a room; these two
    tools are the only bridge between them.
{"tool": "predicts", "args": {"operator": {...}, "index": 4}}
    ADVISORY. Matches your body against one agent's actions on a rollout you did not
    derive it from and asks the recording whether the effect held. It cannot tell a
    correct operator from a broken one -- that was measured -- so treat `matched` as the
    only information  it gives: a body that never matches is one nothing has been checked against.
{"submit": {...the operator...}}
    hands in a candidate. It is checked instantly for whether it can be instantiated in a
    scene at all, and refused with a reason if it cannot -- that check is free and says
    nothing about whether the operator is right. If it passes, it is queued for execution
    against the benchmark, which happens after this conversation ends; you will not see
    that result. Then propose a DIFFERENT one.
"""

TASK = """You are deriving reusable operators for a robot memory from replayed PARTNR rollouts.

An operator is a body of actions written over VARIABLES plus the effect it brings about:
  {"effect": {"key": "%s", "subject": "?x", "value": "?y"},
   "body": [["Verb", "arg", ""], ...],
   "preconditions": {fact_name: bool},
   "cost": int}
`?x` is the subject the effect is about and `?y` is its target. A body often needs a third
thing -- something to open, something to put `?x` down on, something to stand at -- and
every such variable must be written `?z1`, `?z2`, ...: the memory works out what each one
stands for by reading what the body does with it, and it only does that for variables under
those names. A variable named anything else can never be filled in a real scene.

Arguments are written exactly as the rollouts write them, and `Place` takes the compound
form they use.

The memory currently has no operator whose effect is `%s`, so every episode that asks for
it is one the planner cannot even begin: it has nothing to propose and stops. Your job is
to read what the recorded solver actually did on the steps where that effect became true,
and write the reusable form of it.

Emit exactly one JSON object per reply and nothing else. Available:
%s
You have %d moves. Submit up to %d DISTINCT candidates -- distinct means a different body,
not a renamed variable. Spend moves looking before the first submission; a body copied
from one rollout without checking a second is usually specialised to it.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://192.168.32.40:8050/v1")
    ap.add_argument("--model", default="qwen2.5-vl-72b-amendment3-f2")
    ap.add_argument("--rollouts", type=Path, default=ROOT / "results/partnr_rollouts/train_mini")
    ap.add_argument("--target-key", default="is_in_room")
    ap.add_argument("--moves", type=int, default=24)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1600)
    ap.add_argument("--sample-seed", type=int, default=20260907)
    ap.add_argument("--probe", type=int, default=120)
    ap.add_argument("--no-traces", action="store_true",
                    help="control arm: same agent, same task, tools that refuse to read")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    bench = Workbench(root=args.rollouts)
    args.out.mkdir(parents=True, exist_ok=True)

    KNOWN = {"list_traces", "show_trace", "held_by", "actor_window", "predicts", "room_of", "rooms"}

    def call(name: str, kwargs: Dict[str, Any]) -> Any:
        if args.no_traces:
            return {"error": "no rollouts are available in this condition"}
        if name == "list_traces":
            return bench.list_traces(kwargs.get("effect"), int(kwargs.get("limit", 10)))
        if name == "show_trace":
            return bench.show_trace(int(kwargs.get("index", 0)), int(kwargs.get("max_steps", 20)))
        if name == "held_by":
            return bench.held_by(int(kwargs.get("index", 0)), str(kwargs.get("entity", "")),
                                 int(kwargs.get("step", 0)))
        if name == "actor_window":
            return bench.actor_window(int(kwargs.get("index", 0)), str(kwargs.get("actor", "0")),
                                      int(kwargs.get("end", 0)), int(kwargs.get("back", 12)))
        if name == "room_of":
            return bench.room_of(int(kwargs.get("index", 0)), str(kwargs.get("entity", "")))
        if name == "rooms":
            return bench.rooms(int(kwargs.get("index", 0)))
        if name == "predicts":
            return bench.predicts(kwargs.get("operator") or {}, int(kwargs.get("index", 0)))
        return {"error": "unknown tool %r; available: %s" % (name, sorted(KNOWN))}

    task = TASK % (args.target_key, args.target_key, TOOLS, args.moves, args.candidates)
    client = OpenAI(api_key="EMPTY", base_url=args.base_url, max_retries=3, timeout=1800)
    messages = [{"role": "user", "content": task}]
    transcript: List[Dict[str, Any]] = []
    submitted: List[Dict[str, Any]] = []
    refused: List[Dict[str, Any]] = []

    def shape(operator: Dict[str, Any]) -> str:
        return json.dumps([list(a) for a in (operator.get("body") or [])], sort_keys=True)

    for move in range(1, args.moves + 1):
        completion = client.chat.completions.create(
            model=args.model, messages=messages, temperature=args.temperature,
            max_tokens=args.max_tokens, seed=args.sample_seed + move)
        answer = (completion.choices[0].message.content or "").strip()
        request, parse_error = extract_request(answer)
        request = normalise_request(request)
        record: Dict[str, Any] = {"move": move, "answer": answer[:1500]}

        if request is None:
            result: Any = {"error": parse_error or "emit one JSON object"}
        elif isinstance(request, dict) and "submit" in request:
            operator = request["submit"] or {}
            key = (operator.get("effect") or {}).get("key")
            record["tool"] = "submit"
            if key != args.target_key:
                result = {"refused": "the effect key must be %r, yours is %r" % (args.target_key, key)}
            elif not operator.get("body"):
                result = {"refused": "an operator with no body cannot be executed"}
            elif shape(operator) in {shape(s["operator"]) for s in submitted}:
                result = {"refused": "that is a body already submitted; propose a different one"}
            elif not (verdict := groundable(operator, args.target_key))["groundable"]:
                # Free and instant, so it belongs in the loop; it only rejects bodies that
                # could not run at all, never bodies that would run and be wrong.
                result = {"refused": verdict["why"]}
                refused.append({"move": move, "operator": operator, "why": verdict["why"]})
            else:
                # Advisory only, and recorded so the batch gate can be compared against it.
                matched = predicted = 0
                for index in range(min(args.probe, len(bench.paths))):
                    outcome = bench.predicts(operator, index)
                    matched += bool(outcome.get("matched"))
                    predicted += bool(outcome.get("matched") and outcome.get("predicted"))
                submitted.append({"move": move, "operator": operator,
                                  "advisory": {"matched": matched, "predicted": predicted}})
                result = {"queued": len(submitted), "advisory_matched": matched,
                          "note": "queued for execution against the benchmark; propose a different candidate"}
                if len(submitted) >= args.candidates:
                    record["result"] = result
                    transcript.append(record)
                    break
        else:
            name = request.get("tool") if isinstance(request, dict) else None
            kwargs = (request.get("args") or {}) if isinstance(request, dict) else {}
            record["tool"] = name
            result = call(name, kwargs) if name in KNOWN else {
                "error": "unknown tool %r; available: %s and {\"submit\": ...}" % (name, sorted(KNOWN))}

        record["result"] = result
        transcript.append(record)
        # The budget was stated once, in the opening task, and never again: a 30B run on
        # 2026-09-16 spent all 24 moves reading traces and never submitted at all. Carry it
        # in every observation, and say plainly when it is time to stop looking. This adds
        # information only -- nothing here accepts a candidate that would otherwise be refused.
        left = args.moves - move
        budget = {"moves_left": left, "candidates_queued": len(submitted),
                  "candidates_wanted": args.candidates}
        nudge = ""
        if left > 0 and len(submitted) < args.candidates and left <= max(4, args.moves // 4):
            nudge = (" Only %d moves remain and you have queued %d of %d candidates. Stop"
                     " looking and submit now: {\"submit\": {...the operator...}}."
                     % (left, len(submitted), args.candidates))
        messages += [{"role": "assistant", "content": answer},
                     {"role": "user", "content": json.dumps(result, default=str)[:4000]
                      + "\n" + json.dumps(budget) + nudge}]

    payload = {"target_key": args.target_key, "rollouts": str(args.rollouts),
               "model": args.model, "no_traces": args.no_traces,
               "moves_used": len(transcript), "candidates": submitted,
               "refused_not_groundable": refused}
    (args.out / "candidates.json").write_text(json.dumps(payload, indent=1, default=str))
    (args.out / "transcript.json").write_text(json.dumps(transcript, indent=1, default=str))
    print(json.dumps({"candidates": len(submitted), "moves_used": len(transcript),
                      "advisory": [s["advisory"] for s in submitted]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
