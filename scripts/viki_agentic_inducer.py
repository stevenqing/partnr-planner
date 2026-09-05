"""Ask a model to write the operator inducer, then score what it wrote.

The model never sees `induction.py`. It is given the data it may read, the primitives it
may call, the artefact contract it must produce, and the test its output will face -- and
nothing about how the shipped inducer solves any of it. That restriction is the experiment:
a library transcribed from the reference would say nothing about whether the rules can be
derived from the traces.

Two primitives are handed over rather than delegated, for the reason argued in
docs/AGENTIC-OPERATOR-INDUCTION.md: `replay` is simulator execution, and `runs_alone` is
the counterfactual that decides whether one robot's actions suffice. Everything between
them -- attribution, segmentation, abstraction, preconditions, repair, provenance -- is the
model's to write.

Scoring is `scripts/viki_inducer_bench.py`: the memory's own self-check, planning held-out
training episodes with the symbolic planner and scoring them with the official judge. The
model does not score itself and never sees the held-out episodes.
"""
import argparse, json, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import viki_fork_guard
from openai import OpenAI

from our_method.skill_memory_v2 import induction
from our_method.skill_memory_v2.build import load_episodes
from our_method.skill_memory_v2.simulator import SEED, Simulator

CONTRACT = '''
Return a dict:
  {"operators": [operator, ...], "episodes_replayed": int,
   "replay_outcomes": {status: count}, "families_seen": {family: count}}

Every operator is read by a symbolic planner and must carry:
  effect         {"key": str, "subject": "?x", "value": "?y" | True}
  preconditions  {fact_name: bool}          -- matched against world facts at plan time
  cost           int
  support        int                        -- how many episodes demonstrated it
  families       [str]                      -- task_name of each episode that demonstrated it
  types          {var: {property: bool}}    -- optional but used to bind spare variables
  kind           "achievement" | "repair" | "coordination"

A single-runner operator also carries:
  body           [[Verb, arg, ...], ...]    -- args are variables: "?x", "?y", "?z1", ...
  requires       [verb.lower(), ...]
  carries        bool
  runner_types   [str]

A multi-robot operator instead carries:
  coordinated    True
  roles          [{"variable": "?r0",
                   "actions": [{"action": [Verb, arg, ...],
                                "offset": int,
                                "after": [[other_role_index, how_many_of_its_actions_first], ...]}]}]
  role_types     {role_variable: [runner type, ...]}

The planner looks up a repair operator by effect key "unsealed".
'''

PRIMITIVES = '''
from our_method.skill_memory_v2.induction import replay, requirements_of
from our_method.skill_memory_v2.induction import _runs_alone as runs_alone
from our_method.skill_memory_v2.simulator import (
    holds, state_facts, object_properties, predicate_status, flatten_predicates)

replay(truth, sim, seed) -> (trace, status)
    Executes the episode's reference plan in the simulator. status is "OK" or a failure
    name; trace is None on failure. On success trace is:
      {"metadata": {...,"agents": {robot: {"type": str}}},
       "history": [{"actions": {robot: [Verb, arg, ...]}, "carried_before": {robot: [name]}}],
       "states":  [world state before each history step],
       "completions": [(history_index, actor_or_None, predicate)],
       "unmet": [predicate]}
    `completions` records, for each predicate the episode is judged on, the step at which
    it became true and a guess at which robot did it -- the guess is a name match and is
    often None or wrong; you do not have to use it.

runs_alone(state, history, start, index, actor, predicate, sim) -> bool
    Replays ONLY `actor`'s actions in history[start..index] from `state` and reports
    whether `predicate` then holds. This is the causal test; use it, do not reimplement it.

holds(state, predicate) -> bool
state_facts(state, predicate) -> {fact_name: bool}      -- world facts about the predicate
object_properties(asset) -> {property: bool}
predicate_status(predicate) -> {key: value}             -- what the predicate asserts
'''

TASK = '''You are writing the operator-induction step of an executable robot memory.

Training episodes come with a reference plan that solves them. Replaying a plan shows which
required predicates became true, when, and what each robot did. Your job: turn those
replays into a library of reusable operators -- an operator being a body of primitive
actions over variables, the effect it brings about, the world facts that held when it was
seen to work, and what it costs.

The library is then used like this: the planner is given a goal predicate, looks up
operators whose effect key matches, prefers ones whose preconditions match the current
world, binds the variables to real objects and runs the body. So an operator is only useful
if its body, with its variables substituted, actually achieves its effect in a situation
the operator was not induced from.

Write a Python module with exactly this entry point:

    def induce(episodes, sim, seed, per_family=250, exclude_family=None) -> dict

`episodes` is a list of ground-truth dicts; each has "task_name", "time_steps" (the
reference plan) and the goal/temporal constraints. Skip an episode with no "time_steps".
Do not exceed `per_family` episodes of any one task_name. Skip `exclude_family` entirely.

Available to import (nothing else from our_method):
%s
Your module must return:
%s
Think about what makes an operator transfer to an episode it was not induced from, and what
would make it fail to. Output one ```python code block and nothing else.
''' % (PRIMITIVES, CONTRACT)


def sample_traces(sim, episodes, n=2):
    """Real replayed traces, abbreviated -- the model reads data, not a description of it."""
    out = []
    for truth in episodes:
        if not isinstance(truth, dict) or not truth.get("time_steps"):
            continue
        trace, status = induction.replay(truth, sim, SEED)
        if trace is None:
            continue
        out.append({
            "task_name": truth.get("task_name"),
            "history_first_3": [
                {"actions": s["actions"], "carried_before": s["carried_before"]}
                for s in trace["history"][:3]
            ],
            "history_len": len(trace["history"]),
            "completions": [
                {"history_index": i, "actor": a, "predicate_status": induction.predicate_status(p),
                 "predicate_name": p.get("name")}
                for i, a, p in trace["completions"]
            ],
            "state_facts_at_first_completion": (
                induction.state_facts(trace["states"][0], trace["completions"][0][2])
                if trace["completions"] else {}),
        })
        if len(out) >= n:
            break
    return out


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return blocks[0] if blocks else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.32.40:8050/v1")
    parser.add_argument("--model", default="qwen2.5-vl-72b-amendment3-f2")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--tag", default="72b")
    parser.add_argument("--train", type=Path,
                        default=Path("/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet"))
    parser.add_argument("--benchmark-root", type=Path,
                        default=Path("/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R"))
    args = parser.parse_args()

    viki_fork_guard.install()
    client = OpenAI(api_key="EMPTY", base_url=args.base_url, max_retries=3, timeout=1800)

    sim = Simulator(args.benchmark_root)
    episodes = load_episodes(args.train)
    samples = sample_traces(sim, episodes[::2], 2)

    outdir = Path("outputs/agentic_inducer") / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    messages = [
        {"role": "user", "content": TASK + "\n\nTwo real replayed traces, abbreviated:\n"
         + json.dumps(samples, indent=1, default=str)[:9000]},
    ]
    history = []
    for round_index in range(1, args.rounds + 1):
        print("=" * 70)
        print("round", round_index, flush=True)
        started = time.time()
        completion = client.chat.completions.create(
            model=args.model, messages=messages, temperature=args.temperature,
            max_tokens=args.max_tokens, seed=SEED)
        answer = completion.choices[0].message.content or ""
        print("  %.0fs, %d completion tokens" % (time.time() - started,
              completion.usage.completion_tokens), flush=True)
        code = extract_code(answer)
        (outdir / ("round%d_answer.md" % round_index)).write_text(answer)
        if not code:
            messages += [{"role": "assistant", "content": answer},
                         {"role": "user", "content": "No python code block found. Output exactly one."}]
            continue
        module_path = outdir / ("round%d_inducer.py" % round_index)
        module_path.write_text(code)
        bench_out = outdir / ("round%d_bench.json" % round_index)
        proc = subprocess.run(
            [sys.executable, "scripts/viki_inducer_bench.py", "--inducer", str(module_path),
             "--out", str(bench_out)],
            capture_output=True, text=True, timeout=3600)
        verdict = json.loads(bench_out.read_text()) if bench_out.is_file() else {"ran": False}
        rate = (verdict.get("self_check") or {}).get("rate")
        print("  operators=%s  support=%s  self_check=%s  violations=%s" % (
            verdict.get("operators"), verdict.get("support_sum"), rate,
            verdict.get("contract_violations")), flush=True)
        history.append({"round": round_index, "operators": verdict.get("operators"),
                        "by_kind": verdict.get("by_kind"), "support_sum": verdict.get("support_sum"),
                        "self_check": verdict.get("self_check"),
                        "contract_violations": verdict.get("contract_violations"),
                        "ran": verdict.get("ran")})
        (outdir / "history.json").write_text(json.dumps(history, indent=1))
        if rate == 1.0 and not verdict.get("contract_violations"):
            print("  self-check perfect; stopping", flush=True)
            break

        if not verdict.get("ran"):
            feedback = ("Your module failed to run. Traceback tail:\n\n"
                        + (verdict.get("error") or proc.stderr[-2000:]))
        else:
            feedback = (
                "Your library was built and scored. Results:\n"
                + json.dumps({k: verdict.get(k) for k in
                              ("operators", "by_kind", "support_sum", "contract_violations",
                               "bodies_containing_a_variable", "support", "effect_keys_offered",
                               "effect_keys_the_holdout_goals_need",
                               "self_check", "replay_outcomes")}, indent=1)
                + "\n\n`self_check` planned held-out training episodes with your operators and "
                  "scored them with the official judge. `outcomes` names why each unsolved "
                  "episode failed. `bodies_containing_a_variable` counts operators whose "
                  "body still names concrete objects rather than variables, "
                  "`effect_keys_offered` is what your library can bring about and "
                  "`effect_keys_the_holdout_goals_need` is what it is asked for. "
                  "Revise the module and output one python code block.")
        messages += [{"role": "assistant", "content": answer}, {"role": "user", "content": feedback}]

    print()
    print(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
