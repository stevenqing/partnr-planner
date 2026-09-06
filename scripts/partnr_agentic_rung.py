#!/usr/bin/env python3
"""One agentic rung on PARTNR: derive one operator from a replayed rollout.

PARTNR is where the induction headroom is. The shipped inducer's attribution rule -- the
action at the satisfying step must carry a completing verb AND name one of the
proposition's entities -- discards 39.0% of what the rearrange-only library is shown
(180 of 462), and 100% of `is_in_room`, which is achieved by carrying an object while
navigating and so can never be learned by a rule keyed on completing verbs. The machinery
to execute such an operator already exists (`_resolve` implements `is_in_room`); only the
entry is missing. Writing one by hand would be authoring memory rather than inducing it.

**What is different here, and it is a limitation not a choice.** VIKI-L2 rollouts carry
world state, so an operator can be executed and the effect read back. PARTNR traces carry
actions only, so there is no counterfactual replay: the strongest available criterion is
whether the operator, matched against a DIFFERENT rollout's actions, predicts a
satisfaction the recording actually contains. That is weaker than execution, and the
workbench's own two-way calibration is what makes it usable at all -- factory operators
score 0.92/0.94 precision where a reversed body scores 0.17 and a swapped effect key
matches nothing.

Acceptance therefore has two parts, both mechanical:
  precision  the operator must predict correctly on traces it was not derived from
  coverage   it must predict satisfactions the library already held does not
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from openai import OpenAI
from partnr_induction_tools import Workbench
from viki_agentic_rung_abstraction import extract_request, normalise_request

TOOLS = """
{"tool": "list_traces", "args": {"effect": null, "limit": 10}}
    the rollouts available, with the effects each one records becoming true.
{"tool": "show_trace", "args": {"index": 0, "max_steps": 20}}
    one rollout: each planner step's action per agent, and for every proposition the
    episode is judged on, the step at which the recording says it became true.
{"tool": "held_by", "args": {"index": 0, "entity": "cup_0", "step": 4}}
    which agent was holding that entity at that step, read off the recording.
{"tool": "actor_window", "args": {"index": 0, "actor": "0", "end": 12, "back": 12}}
    that agent's own actions leading up to that step, which is where a body comes from.
{"tool": "predicts", "args": {"operator": {...}, "index": 4}}
    THE TEST. Matches your body's verb sequence against one agent's recorded actions on a
    DIFFERENT rollout, binds your variables to the entities those actions name, and asks
    the recording whether your effect over those entities actually became true. Returns
    `matched` and `predicted` separately: never matching means untested, which is not the
    same as matching and being wrong.
{"submit": {...the operator...}}
"""

TASK = """You are deriving one reusable operator from a replayed PARTNR rollout.

An operator is: a body of actions written over VARIABLES, and the effect it brings about.
It is reusable only if, on a DIFFERENT rollout with different objects, its body matches
some agent's actions and the effect over the bound entities is one the recording says
became true.

Operator shape:
  {"effect": {"key": "%s", "subject": "?x", "value": "?y"},
   "body": [["Verb", "arg", ""], ...],
   "preconditions": {fact_name: bool},
   "cost": int}

The arguments are written exactly as the recording writes them; `Place` takes the compound
form the rollouts use. Variables are `?x` for the subject and `?y` for the target.

There is no simulator here. You cannot execute a candidate; you can only ask `predicts`
whether it matches and is right on rollouts you did not derive it from. Use it before
submitting -- an operator that matches nothing has not been tested.

Emit exactly one JSON object per reply and nothing else. Available:
%s
You pass when a submitted operator matches at least %d rollouts you were not shown, is
correct on at least %.0f%% of them, and predicts at least one satisfaction the memory
already held does not. You have %d moves.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://192.168.32.40:8050/v1")
    parser.add_argument("--model", default="qwen2.5-vl-72b-amendment3-f2")
    parser.add_argument("--moves", type=int, default=18)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--sample-seed", type=int, default=20260901)
    parser.add_argument("--tag", default="partnr")
    parser.add_argument("--seed-trace", type=int, default=0)
    parser.add_argument("--target-key", default="is_in_room")
    parser.add_argument("--probe", type=int, default=120)
    parser.add_argument("--min-matched", type=int, default=5)
    parser.add_argument("--min-precision", type=float, default=0.8)
    parser.add_argument("--library", type=Path, default=None)
    args = parser.parse_args()

    bench = Workbench()
    outdir = ROOT / "outputs/partnr_rung" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    library: List[Dict[str, Any]] = []
    if args.library and args.library.is_file():
        record = json.loads(args.library.read_text())
        library = record["operators"] if isinstance(record, dict) else record

    def covered(operators) -> set:
        """Rollouts on which the library already predicts a satisfaction correctly."""
        out = set()
        for operator in operators:
            for index in range(min(args.probe, len(bench.paths))):
                if index == args.seed_trace:
                    continue
                outcome = bench.predicts(operator, index)
                if outcome.get("matched") and outcome.get("predicted"):
                    out.add(index)
        return out

    covered_before = covered(library) if library else set()

    task = TASK % (args.target_key, TOOLS, args.min_matched,
                   args.min_precision * 100, args.moves)
    task += "\n\nStart from rollout index %d." % args.seed_trace
    if library:
        task += ("\n\nA memory of %d operators already exists and already predicts %d "
                 "rollouts correctly. An operator is accepted only if it predicts at least "
                 "one rollout that memory does not." % (len(library), len(covered_before)))

    client = OpenAI(api_key="EMPTY", base_url=args.base_url, max_retries=3, timeout=1800)
    messages = [{"role": "user", "content": task}]
    transcript, verdict = [], {"passed": False, "moves_used": 0}

    KNOWN = {"list_traces", "show_trace", "held_by", "actor_window", "predicts"}

    def call(name, kwargs):
        if name == "list_traces":
            return bench.list_traces(kwargs.get("effect"), int(kwargs.get("limit", 10)))
        if name == "show_trace":
            return bench.show_trace(int(kwargs.get("index", 0)),
                                    int(kwargs.get("max_steps", 20)))
        if name == "held_by":
            return bench.held_by(int(kwargs.get("index", 0)), str(kwargs.get("entity", "")),
                                 int(kwargs.get("step", 0)))
        if name == "actor_window":
            return bench.actor_window(int(kwargs.get("index", 0)), str(kwargs.get("actor", "0")),
                                      int(kwargs.get("end", 0)), int(kwargs.get("back", 12)))
        if name == "predicts":
            return bench.predicts(kwargs.get("operator") or {}, int(kwargs.get("index", 0)))
        return {"error": "unknown tool %r; available: %s" % (name, sorted(KNOWN))}

    for move in range(1, args.moves + 1):
        completion = client.chat.completions.create(
            model=args.model, messages=messages, temperature=args.temperature,
            max_tokens=args.max_tokens, seed=args.sample_seed)
        answer = (completion.choices[0].message.content or "").strip()
        request, parse_error = extract_request(answer)
        request = normalise_request(request)
        record = {"move": move, "answer": answer[:1500]}
        verdict["moves_used"] = move

        if request is None:
            result = {"error": parse_error or "emit one JSON object"}
        elif isinstance(request, dict) and "submit" in request:
            operator = request["submit"]
            key = (operator.get("effect") or {}).get("key")
            if key != args.target_key:
                result = {"submitted": True, "refused": "the operator must have effect key %r,"
                          " yours is %r" % (args.target_key, key)}
            else:
                score = bench.score_operator(operator, exclude=[args.seed_trace],
                                             probe=args.probe)
                gained = []
                if (score["matched"] >= args.min_matched
                        and (score["precision"] or 0) >= args.min_precision):
                    after = covered(library + [operator])
                    gained = sorted(after - covered_before)
                result = {"submitted": True, "score": score,
                          "rollouts_newly_predicted": gained[:20],
                          "n_newly_predicted": len(gained)}
                if (score["matched"] >= args.min_matched
                        and (score["precision"] or 0) >= args.min_precision and gained):
                    verdict = {"passed": True, "moves_used": move, "operator": operator,
                               "score": score, "n_newly_predicted": len(gained)}
                    record["tool"], record["result"] = "submit", result
                    transcript.append(record)
                    break
                if score["matched"] < args.min_matched:
                    result["why"] = ("matched only %d rollouts; an operator that matches "
                                     "nothing has not been tested" % score["matched"])
                elif (score["precision"] or 0) < args.min_precision:
                    result["why"] = ("precision %.2f is below %.2f -- it matches but the "
                                     "recording disagrees" % (score["precision"] or 0,
                                                              args.min_precision))
                else:
                    result["why"] = "adds no rollout the memory does not already predict"
            record["tool"] = "submit"
        else:
            name = request.get("tool") if isinstance(request, dict) else None
            kwargs = (request.get("args") or {}) if isinstance(request, dict) else {}
            record["tool"] = name
            result = call(name, kwargs) if name in KNOWN else {
                "error": "unknown tool %r; available: %s and {\"submit\": ...}"
                         % (name, sorted(KNOWN))}
        record["result"] = result
        transcript.append(record)
        messages += [{"role": "assistant", "content": answer},
                     {"role": "user", "content": json.dumps(result, default=str)[:4000]}]

    (outdir / "transcript.json").write_text(json.dumps(transcript, indent=1, default=str))
    (outdir / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
    print(json.dumps({k: v for k, v in verdict.items() if k != "operator"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
