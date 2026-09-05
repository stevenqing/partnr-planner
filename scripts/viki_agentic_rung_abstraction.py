"""The abstraction rung: one sub-problem, one gate, tools instead of a scalar.

The model is asked for a single operator and may call the workbench's oracles while it
works. It passes when an operator it submits binds and achieves its effect on episodes it
was not shown -- which is the whole content of "abstraction": an operator induced from one
episode is only an operator if it still works on another.

This exists because the whole-inducer task gives one bit of feedback for six decisions. Here
the model can ask `try_bind` why a body was rejected instead of inferring it from a 0.0.
"""
import argparse, json, re, sys, time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import viki_fork_guard
from openai import OpenAI
from viki_induction_tools import Workbench
from our_method.skill_memory_v2.simulator import SEED

TOOLS = """
{"tool": "list_episodes", "args": {"family": null, "limit": 10}}
{"tool": "show_trace", "args": {"index": 0, "max_steps": 12}}
    the replayed episode: each step's actions per robot, what each robot carried, and the
    completions -- for every predicate the episode is judged on, the step it became true,
    a name-match guess at who did it (often None or wrong), and the world facts at the start.
{"tool": "contrast_actors", "args": {"index": 0}}
    every robot's own decision sequence in that episode, side by side, repeats collapsed.
    Two robots in one episode routinely do different things; a body only one of them
    performed is still an operator.
{"tool": "check_actor", "args": {"index": 0, "completion": 0, "actor": "R1", "start": null}}
    replays ONLY that robot's actions over the segment and reports whether the predicate
    then holds, plus the actions themselves. This is the causal test.
{"tool": "try_bind", "args": {"operator": {...}, "index": 4}}
    runs the planner's own binding on another episode and, when it refuses, names the
    tokens that were left unbound.
{"tool": "run_operator", "args": {"operator": {...}, "index": 4}}
    binds on that episode, executes the bound body in the simulator, and reports whether
    the effect then holds.
{"submit": {...the operator...}}
"""

TASK = """You are deriving one reusable operator from a replayed robot demonstration.

An operator is: a body of primitive actions written over VARIABLES, the effect it brings
about, the world facts that held when it was seen to work, and what it costs. It is reusable
only if, on a DIFFERENT episode with different objects, the planner can bind its variables
and the executed body then makes the effect true.

Operator shape:
  {"kind": "achievement",
   "effect": {"key": "pos.name" | "is_activated", "subject": "?x", "value": "?y" | true},
   "body": [["Verb", "arg"], ...],
   "preconditions": {fact_name: bool},
   "types": {var: {property: bool}},
   "requires": [verb_lowercase, ...], "carries": bool,
   "cost": int, "support": 1, "families": [task_name], "runner_types": [robot_type]}

Work with the tools below. Look at a trace, decide who achieved which predicate and over
which actions, then write the body so it transfers. Use the tools to check your answer
before submitting -- `try_bind` and `run_operator` will tell you exactly why something was
refused.

Emit exactly one JSON object per reply and nothing else. Available:
%s
You pass when a submitted operator binds AND achieves its effect on at least two episodes
you were not shown. You have %d moves.
"""



def extract_request(answer: str):
    """Find the model's JSON object.

    Models wrap the object in prose, in a fenced block, or emit it bare. Refusing any of
    those would score presentation rather than the work, so all three are accepted: fenced
    blocks first, then the first balanced object anywhere in the reply.
    """
    for block in re.findall(r"```(?:json)?\s*\n(.*?)```", answer, re.S):
        try:
            return json.loads(block), None
        except Exception:
            continue
    depth, start = 0, None
    for position, character in enumerate(answer):
        if character == "{":
            if depth == 0:
                start = position
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(answer[start:position + 1]), None
                except Exception:
                    start = None
    return None, "no parseable JSON object found; emit one JSON object"


def normalise_request(request):
    """Accept the obvious ways of saying "here is my operator".

    `extract_request` already takes fenced, bare and prose-wrapped JSON, on the stated
    grounds that refusing any of them would score presentation rather than the work. The
    dispatcher then insisted on the key `submit` and answered anything else with
    "unknown tool None" -- which named neither the problem nor the fix. 152 of 524
    transcripts hit that, and in the worst of them a model produced exactly the operator
    the memory was missing, on move 4, wrapped as {"operator": ...}, and resubmitted it
    fifteen times against the same unhelpful error. That is the same mistake one level up,
    so the same principle applies: the key a correct answer arrives under is presentation.

    A bare operator object -- one carrying `effect` and `body` -- is taken as a submission
    too. Nothing here relaxes what an operator must DO to be accepted.
    """
    if not isinstance(request, dict):
        return request
    if "submit" in request or "tool" in request:
        return request
    for alias in ("operator", "submission", "answer", "result"):
        if isinstance(request.get(alias), dict):
            return {"submit": request[alias]}
    if "effect" in request and ("body" in request or "roles" in request):
        return {"submit": request}
    return request

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.32.40:8050/v1")
    parser.add_argument("--model", default="qwen2.5-vl-72b-amendment3-f2")
    parser.add_argument("--moves", type=int, default=18)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--tag", default="72b")
    parser.add_argument("--seed-episode", type=int, default=0)
    # Sampling seed only. The framework -- tools, oracles, task text, protocol, move
    # budget -- is frozen; this varies the draw so a pass rate can be measured.
    parser.add_argument("--sample-seed", type=int, default=SEED)
    parser.add_argument("--holdout", type=int, nargs="+", default=[4, 6, 8, 10])
    # Asking for a specific effect key makes this a DIFFERENT cell, not a tweak to the old
    # one: the task text the model reads changes, and the frozen rung's pass rate was
    # measured without it. Left unset the prompt and the run are byte-identical to the
    # frozen cell, so the two can coexist; set, the result must be reported under its own
    # label and never pooled with the 40% that was measured before it existed.
    parser.add_argument("--target-key", default=None,
                        help="require the submitted operator to have this effect key")
    # Marginal contribution. Without it the rung asks only "does this operator achieve its
    # effect on two held-out episodes", which an operator already in the library passes
    # trivially: 40 submissions across six families deduplicated to 3 operators, and 15 of
    # them were the same `Move Reach Grasp Move Place` body re-derived, accepted on episodes
    # the planner still cannot solve. Given a library, an operator must now make at least
    # one holdout episode go from unsolved to solved -- which is the thing the library is
    # actually short of. Unset, the run is byte-identical to the frozen cell.
    parser.add_argument("--library", type=Path, default=None,
                        help="admit only operators that add coverage to this library")
    args = parser.parse_args()

    viki_fork_guard.install()
    reference = json.loads(Path("results/viki_memory_experiments/amendment11/skill_memory_v2.json").read_text())
    bench = Workbench("/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet",
                      "/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R", SEED,
                      {"layer2": reference["layer2"], "layer3": reference["layer3"]})
    client = OpenAI(api_key="EMPTY", base_url=args.base_url, max_retries=3, timeout=1800)

    outdir = Path("outputs/agentic_rung") / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    KNOWN_TOOLS = {"list_episodes", "show_trace", "contrast_actors", "check_actor",
                   "try_bind", "run_operator"}

    def call(name, kwargs):
        if name == "list_episodes":
            return bench.list_episodes(kwargs.get("family"), int(kwargs.get("limit", 10)))
        if name == "show_trace":
            return bench.show_trace(int(kwargs["index"]), int(kwargs.get("max_steps", 12)))
        if name == "contrast_actors":
            return bench.contrast_actors(int(kwargs["index"]))
        if name == "check_actor":
            return bench.check_actor(int(kwargs["index"]), int(kwargs["completion"]),
                                     kwargs["actor"], kwargs.get("start"))
        if name == "try_bind":
            return bench.try_bind(kwargs["operator"], int(kwargs["index"]))
        if name == "run_operator":
            return bench.run_operator(kwargs["operator"], int(kwargs["index"]))
        return {"error": "unknown tool %r. Emit either {\"tool\": <one of %s>, "
                         "\"args\": {...}} or {\"submit\": {...the operator...}}."
                         % (name, ", ".join(sorted(KNOWN_TOOLS)))}

    library_operators = []
    if args.library:
        library_operators = json.loads(Path(args.library).read_text())["operators"]
    solved_before = {}
    if library_operators:
        for j in args.holdout:
            try:
                solved_before[j] = bench.plan_with(library_operators, j)["official_score"] >= 1.0
            except Exception:
                solved_before[j] = False

    task = (TASK % (TOOLS, args.moves)) + "\n\nStart from episode index %d." % args.seed_episode
    if library_operators:
        unsolved = [j for j, ok in solved_before.items() if not ok]
        task += ("\n\nA memory of %d operators already exists and it CANNOT solve episodes %s."
                 " Achieving your effect somewhere is not enough: your operator is accepted"
                 " only if adding it to that memory makes at least one of those episodes"
                 " solvable. An operator that repeats what the memory already does will be"
                 " refused, however well it works." % (len(library_operators), unsolved))
    if args.target_key:
        task += ("\n\nThe operator you submit must have effect key %r. Episodes that "
                 "demonstrate it are the ones you were given; a submission with any other "
                 "effect key is refused without being run." % args.target_key)
    messages = [{"role": "user", "content": task}]
    transcript, verdict = [], {"passed": False, "moves_used": 0}
    submitted_bodies = []

    def body_of(operator):
        return "/".join(action[0] for action in (operator.get("body") or []))

    for move in range(1, args.moves + 1):
        completion = client.chat.completions.create(
            model=args.model, messages=messages, temperature=args.temperature,
            max_tokens=args.max_tokens, seed=args.sample_seed)
        answer = (completion.choices[0].message.content or "").strip()
        request, parse_error = extract_request(answer)
        request = normalise_request(request)
        record = {"move": move, "answer": answer[:1500]}
        if request is None:
            result = {"error": parse_error}
        else:
            pass
        if True:
            if request is not None and "submit" in request:
                operator = request["submit"]
                submitted_bodies.append(body_of(operator))
                submitted_key = (operator.get("effect") or {}).get("key")
                if args.target_key and submitted_key != args.target_key:
                    # Refused before execution, so a wrong-key operator cannot pass by
                    # happening to work: this cell is measuring coverage of one key.
                    result = {"submitted": True, "refused": True,
                              "why": "effect key %r is not the requested %r"
                                     % (submitted_key, args.target_key)}
                    record["result"] = result
                    transcript.append(record)
                    verdict["moves_used"] = move
                    print("move %2d  submit         -> refused (%s)" % (move, submitted_key), flush=True)
                    messages += [{"role": "assistant", "content": answer},
                                 {"role": "user", "content": json.dumps(result)[:4000]}]
                    continue
                checks = [bench.run_operator(operator, j) for j in args.holdout]
                works = [j for j, c in zip(args.holdout, checks)
                         if c.get("bound") and c.get("effect_holds")]
                result = {"submitted": True, "episodes_it_works_on": works,
                          "detail": [{k: c.get(k) for k in ("bound", "effect_holds", "failure")}
                                     for c in checks]}
                record["result"] = result
                transcript.append(record)
                gained = None
                if library_operators and len(works) >= 2:
                    gained = []
                    for j in args.holdout:
                        if solved_before.get(j):
                            continue
                        try:
                            after = bench.plan_with(library_operators + [operator], j)
                        except Exception:
                            continue
                        if after["official_score"] >= 1.0:
                            gained.append(j)
                    result["episodes_newly_solved"] = gained
                    if not gained:
                        # Naming what the memory already holds, and naming a repeat as a
                        # repeat. Without this the refusal said only "adds no coverage",
                        # and models answered it by re-reading the same trace and
                        # resubmitting the identical body -- observed four times in one run.
                        # This is the model's own library, so telling it is not a leak.
                        held = sorted({body_of(op) for op in library_operators if op.get("body")})
                        result["memory_already_contains"] = held
                        result["your_body"] = body_of(operator)
                        if body_of(operator) in submitted_bodies[:-1]:
                            result["note"] = (
                                "you have submitted this exact body before and it was refused "
                                "for the same reason. Resubmitting it cannot succeed. Look at "
                                "another actor in the episode -- `contrast_actors` shows every "
                                "robot's sequence -- or at an episode where this body fails.")
                        else:
                            result["note"] = (
                                "this achieves its effect but adds no coverage: the memory "
                                "already contains %s. A different body is needed, not a better "
                                "argument for this one. `contrast_actors` shows what each robot "
                                "in the episode did; they are often not the same."
                                % (held or "an equivalent operator"))
                        record["result"] = result
                        transcript.append(record)
                        verdict["moves_used"] = move
                        print("move %2d  submit         -> works on %s but adds nothing"
                              % (move, works), flush=True)
                        messages += [{"role": "assistant", "content": answer},
                                     {"role": "user", "content": json.dumps(result, default=str)[:4000]}]
                        continue
                if len(works) >= 2 and (gained is None or gained):
                    verdict = {"passed": True, "moves_used": move, "operator": operator,
                               "works_on": works, "newly_solved": gained}
                    print("PASSED on move %d, works on episodes %s" % (move, works), flush=True)
                    break
                result["note"] = "needs to work on at least two; revise and resubmit"
            elif request is not None:
                name = request.get("tool")
                try:
                    result = call(name, request.get("args") or {})
                except Exception as error:
                    result = {"error": "%s: %s" % (type(error).__name__, error)}
                record["tool"] = name
        record.setdefault("result", result)
        transcript.append(record)
        verdict["moves_used"] = move
        print("move %2d  %-14s -> %s" % (move, record.get("tool", "submit"),
              json.dumps(result, default=str)[:160]), flush=True)
        messages += [{"role": "assistant", "content": answer},
                     {"role": "user", "content": json.dumps(result, default=str)[:4000]}]

    (outdir / "transcript.json").write_text(json.dumps(transcript, indent=1, default=str))
    (outdir / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
    print(json.dumps({k: v for k, v in verdict.items() if k != "operator"}, indent=1))


if __name__ == "__main__":
    main()
