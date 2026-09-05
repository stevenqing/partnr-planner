# Agentic operator induction — protocol

Status: design, 2026-09-04. Nothing implemented. Written before any code so the controls
are fixed in advance rather than chosen after seeing results.

## The claim this is meant to support

> Replay induction can only learn what its author anticipated. An agent can propose what
> the traces suggest, and the simulator decides which proposals survive.

The second half is the part that makes this publishable. **The agent only ever proposes.
Acceptance is always a mechanical test against the simulator or the official checker. No
operator enters a library on the model's say-so.** The existing inducer already has this
shape -- `_runs_alone` in `our_method/skill_memory_v2/induction.py` is a counterfactual
verifier -- but its proposal step is eight lines of hard-coded pattern matching.

## Why: measured headroom, not intuition

`scripts/partnr_induction_headroom.py` and `..._bytype.py` count where the shipped inducer
gives up. Over the 161 rearrange traces the R-only library actually reads:

    propositions satisfied            462
    discarded by the inducer          180   = 39.0%
      is_on_top      425 seen, 144 lost   33.9%
      is_in_room      36 seen,  36 lost  100.0%

Over all 399 train_mini traces, 820 of 1711 satisfied propositions (47.9%) are discarded;
674 of those are attribution failures. Broken out by predicate:

    is_next_to     342 seen, 294 lost   86.0%   (R_S / R_S_T traces only)
    is_in_room     127 seen, 127 lost  100.0%

The cause is `induce_from_trace`'s attribution rule: the action at the satisfying step must
have a verb on the `COMPLETING` whitelist *and* name one of the proposition's entities. At
the failures the step's action is `Navigate` (883) or `Wait` (437). So:

* `is_in_room` is achieved by **carrying an object while navigating**. `Navigate` is not a
  completing verb, so this predicate can never be learned. 100%, every split.
* `is_next_to` is achieved as a **side effect** of a `Place`; the placed object ends up
  beside a reference object the action never names, so `touches` fails.

### VIKI-L2: no headroom at all

`scripts/viki_induction_headroom.py` over the same half `build.py` induces from:

    episodes replayed        2698    replay failures: none
    completions found        5147
    discarded                   0    = 0.0%
      kinds: 4808 achievement, 339 coordination
    predicate shapes present in the whole training set:
      pos.name       4183
      is_activated    964

**The inducer loses nothing on VIKI-L2.** `_bind`'s two hard-coded predicate shapes cost
nothing because the benchmark contains exactly those two shapes; attribution never fails
and no replay fails. This is a negative result and it should be respected: an agentic
proposal step on VIKI-L2 has nothing to recover. Whatever it did there it would have to
justify differently -- proposing operators for a family no trace demonstrates (`dog_push`,
whose `Push` primitive occurs in one family only) is exactly the leakage-risky case the
control below exists to catch, and it is not a good place to start.

The contrast is itself worth reporting: VIKI-L2 has 14 task families, 92.5% verbatim
train/test instruction overlap and two predicate shapes. PARTNR has nine predicates,
side-effect satisfaction and navigation-satisfied goals. **The induction problem only
exists on the second one.**

## Two consequences worth separating, because only one of them is a defect


1. **Not a defect.** `is_next_to` never occurs in the rearrange traces. The R-only library
   genuinely cannot learn spatial operators, and the compositional generalisation test is
   sound on that axis.
2. **A defect.** 39% of what the R-only library *was shown* it threw away, and the
   all-types library -- the coverage ceiling the R-only one is measured against -- was
   built while discarding 86% of the spatial demonstrations. The reported ceiling is itself
   depressed.

## What is delegated, and what is never delegated

| step | today | agentic? | acceptance test |
|---|---|---|---|
| replay | simulator execution | **no**; only failure recovery is delegated | the repaired trace replays to completion |
| attribution | verb whitelist + name match | **yes** | counterfactual replay of the proposed actor's window |
| segmentation | actor's actions since its last completion | **yes** | replaying exactly the proposed body makes the effect hold |
| minimality | *absent* | agent orders candidates | leave-one-out: drop an action, replay, keep it dropped if the effect still holds |
| independence (achievement vs coordination) | counterfactual replay | **no** | already mechanical; it is the causal criterion itself |
| abstraction / effect schema | `_bind`, two predicate shapes | **yes** -- the largest gain | apply the abstracted operator on a *different* episode whose types match; official checker judges |
| preconditions | whatever held at segment start | agent orders candidates | drop a fact, replay, keep it dropped if the effect still holds |
| repair mining | hard-coded to `Open` | **yes** | find a state where the blocked operator fails, run the repair, check it then succeeds |
| dedup / provenance | mechanical | **no** | unchanged |

Delegating independence would be the one change that destroys the method: it is the test
that distinguishes an operator one robot can run from one that needs two, and it must stay
an execution, not a judgement.

## Leakage control

An LLM asked to propose a spatial operator may produce one from pretraining rather than
from our traces. If that happens, "generalises from rearrange-only data" is false and the
result is worthless. Four arms, fixed now:

| arm | memory | purpose |
|---|---|---|
| (a) | replay induction, R-only | current baseline |
| (b) | **agentic, sees only the 161 R traces** | the claim |
| (c) | replay induction, all types | coverage ceiling |
| (d) | **agentic, given the predicate menu and no traces** | the control that decides everything |

**If (d) ≈ (b), the model was not using our data and the word "memory" does not apply.**

Mechanical support, not a promise: every operator records the traces that proposed it and
the held-out episodes that verified it. **An effect schema with no support in any replayed
trace's proposition set is flagged as ungrounded.** That check is automatic and reported.

The agent may see: replay traces, world states, the predicate vocabulary of the *training*
episodes, the type system. It may not see: test episodes, the evaluation split's
propositions, or any example of a task type outside the library's declared scope.

## Anti-gaming

* **Degenerate bodies.** An operator whose body is the whole reference plan passes any
  effect test. Minimality is therefore mandatory, not optional, and cost is recorded.
* **Single-episode operators.** An operator must fire on **at least two distinct episodes**
  to enter the library; `build.py` already splits the training set in half for this.
* **Non-determinism.** The library becomes a random variable. Build under three seeds and
  report the spread, the way the evaluation arms already do.

## Order of work

0. **Fix the executor first — done 2026-09-05, and the estimate here was wrong.** 19% of
   PARTNR episodes scored exactly 0 because both agents reported `Done` at step 0. The
   cause was not the `is_done` line itself: `requirements_from_propositions` marks every
   requirement `bound`, so `_bind` returns early and the explore fallback inside it is
   unreachable in every privileged arm — an agent whose objects were all still unseen could
   ground nothing, found no work, and stood down. (`Explore` fires 671,399 times across the
   296 survivors and **0** times in the 70; the survivors bootstrap exploration through
   `_repair`'s "unseen" path, which needs a first action to have been issued.) `_claim` now
   explores while work remains.

   This section estimated that 45 of the 70 had nothing to do with operator coverage.
   **Measured, it is 30.** The other 40 ask only for effects no library here holds, and
   `is_in_room` — which the headroom counts below already show is lost 100% of the time,
   every split — is in 61 of the 70. Post-fix those episodes explore the whole house for
   thousands of steps and never emit a `Navigate`, because `operators_for("is_in_room")` is
   empty. So the step-0 death was *masking* the attribution gap this document is about,
   and the recoverable half was smaller than assumed: paired on the 70, 0.0100 → 0.1275
   (Wilcoxon p=0.0033), and on the 30 coverable ones 0.0067 → 0.2809.

   The floor is gone either way, which is what step 0 was for: on the full split
   `v2_memory_R` moves 0.6748 → 0.7319, and the recovery lands on the compositional axis
   (`R_T` 0.621 → 0.742 normalised). `ceiling` re-runs bit-identical at 0.9515, which is
   the control that says these deltas are the fix and not run-to-run noise.
1. Headroom counts -- **done for both**. PARTNR: 39% of the R-only demonstrations
   discarded. VIKI-L2: 0%.
2. Build the agentic loop on **PARTNR, and only PARTNR**. The counts, not convenience,
   decide this: there is nothing on VIKI-L2 for it to recover.
3. Port to VIKI **only if** a capability emerges that is not "recover discarded
   demonstrations" -- and then say plainly what it is buying there, because the induction
   step is already lossless on that benchmark.

The first capability to build is **attribution**, because its entire headroom -- 39% of the
R-only demonstrations -- lies inside data the library has already been given. It needs no
new knowledge, so it carries no leakage risk, and it is measurable against (a) alone.

---

## The framework: local oracles, not a scalar

*Added 2026-09-04 after the first two runs.*

The first harness was not an agentic framework. It asked a model to write a six-stage
inducer -- attribution, segmentation, abstraction, preconditions, repair, dedup -- and
returned **one number**. Six decisions, one bit. A candidate that wrote `?apple_0` where a
variable belonged learned nothing from its `0.0`, because the binding failure never reached
it. That is a credit-assignment problem, and no amount of prompt work fixes it.

Each induction decision has a cheap mechanical oracle of its own:

| decision | its oracle | cost |
|---|---|---|
| attribution | `check_actor` -- counterfactual replay of that robot's actions | one replay |
| segmentation | replaying just the proposed body achieves the effect | one replay |
| **abstraction** | `run_operator` -- bind on a **different** episode and execute | two replays |
| preconditions | drop a fact, replay, see if it still holds | one per fact |
| repair | the blocked operator succeeds after it | two replays |
| dedup | two operators achieve the same effects on the same states | batch |

The enabling fact is measured, not assumed: replaying the whole induction half takes 3.5
seconds, so a single episode is milliseconds and these can be called tens of thousands of
times. `scripts/viki_induction_tools.py` implements them. `try_bind` calls the planner's own
`chains_for` rather than a paraphrase of it, and when the planner is silent it names the
tokens left unbound.

**The workbench reads `episodes[::2]` only.** The half the self-check plans is never
exposed, so tool use cannot fit to the gate.

### What the rung changed, measured

`scripts/viki_agentic_rung_abstraction.py` asks for one operator and grants tool access.
Qwen2.5-VL-72B, same model both times:

    blind loop, scalar feedback     every submitted body used pseudo-variables (`?apple_0`);
                                    0 of 25542 operators could be bound; self-check 0.0
    tool loop, local oracles        every submitted body bound successfully; the failure
                                    moved to one token

The remaining error is worth recording because it is not a structural one. The model
submitted

    [["Move","?x"],["Reach","?x"],["Grasp","?x"],["Move","?y"],["Reach","?y"],["Place","?x"]]

against a demonstration whose own final action was `["Place", "plate"]` -- the target. The
body executes cleanly and the effect does not hold: it places the subject rather than
placing at the target. The information was in the trace; a prior about what "place the
apple" means overrode it. (The stray `Reach ?y` is the second error: `Reach` cannot resolve
a place, only an asset.)

So the framework converted an opaque `0.0` into a single wrong token, which is the entire
point of the ladder. What it has not yet done is get a model over the rung.

### Next, and the line for a fair model comparison

One further oracle is warranted and is not a hint: `run_operator` should report **what the
effect predicate actually became** after execution, not just that it is false. "the object
ended on the robot, not on the bowl" is an observation about the world; "you wrote `Place
?x` and the demonstration wrote `Place ?y`" would be the answer and must not be given.

After that the framework freezes -- tools, oracles, ladder, protocol -- and the model is the
only thing that varies. Otherwise the sweep measures prompt engineering rather than the
models, and the capability curve is worthless.

## FROZEN 2026-09-04

`viki_induction_tools.py` (oracles, including `run_operator`'s `effect_state`),
`viki_agentic_rung_abstraction.py` (protocol, task text, 18 moves, pass = binds and
achieves its effect on >=2 unseen episodes) and `viki_inducer_bench.py` (the global gate)
are frozen from here. The model is the only thing that varies across the sweep. Any change
to the framework after this point invalidates the comparison and must re-run every model.

### Freeze broken again, 2026-09-05, four times, and why

Four changes to the frozen framework. Each was made because the harness was found to be
measuring itself rather than the model, and each is recorded here because **the capability
curve of 2026-09-04 (72B 40% / 30B 40% / 7B 0%) was collected under the broken version and
is therefore a lower bound, not a measurement.** Every model must be re-run before that
curve is quoted again.

1. **`normalise_request` — submissions were being thrown away on the key they arrived
   under.** The dispatcher required `{"submit": ...}`; anything else fell through to
   `request.get("tool") -> None` and was answered `"unknown tool None"`, which named neither
   the problem nor the fix. **152 of 524 transcripts hit it**, the 7B cell worst (10 of 14
   runs, 125 rejections). In the clearest case a model produced exactly the operator the
   memory was missing, on move 4, wrapped as `{"operator": ...}`, and resubmitted it fifteen
   times against the same error. `extract_request` already accepts fenced, bare and
   prose-wrapped JSON on the stated grounds that refusing them would score presentation
   rather than the work; this is the same principle one level up.

2. **`run_operator` tries every robot, not `sorted(env.agents)[0]`.** The reference
   library's own sealed-target operator -- which takes all four `clear_table` holdout
   episodes from 0.00 to 1.00 under `plan_with` -- was refused by this test on every one of
   them (`checker refused ['Open', 'cabinet']`), because its `runner_types` are
   `['unitree_h1', 'stompy']` and R1 is neither. A model submitted a byte-identical copy of
   that operator and was told it did not work. The verifier was asking a question the
   library never asks: the planner assigns an agent, it does not insist on the first one.

3. **`contrast_actors`, a new oracle.** In every seed episode of every failing family, one
   robot performs the plain body and the other performs exactly the missing variant. Traces
   are 7-8 steps and `show_trace` returns all of them, so the evidence was always on screen;
   the transcripts show a model calling `check_actor` on the first robot, being refused,
   re-reading the same trace four times and resubmitting the identical body, never looking
   at the second. Faithful and unreadable is still unreadable.

4. **Marginal-contribution acceptance (`--library`), and an informative refusal.** The old
   test asked only whether an operator achieved its effect on two unseen episodes, which an
   operator already in the library passes trivially: 40 submissions deduplicated to 3
   operators, 15 of them the same body re-derived and accepted on episodes the planner still
   could not solve. Opt-in; unset, the run is byte-identical to the frozen cell.

None of these relaxes what an operator must ACHIEVE. They stop the harness from failing
correct answers. The regression check for (2) is recorded: the frozen base operator still
verifies on exactly the episodes it did before.

**Consequence, stated plainly.** Every pass rate in this document collected before
2026-09-05 is a lower bound. The 2026-09-04 capability curve must be re-measured on all
three models under the repaired framework before it is used to compare them, and until then
the sentence "15 runs is enough to separate 0% from 40%" is not supported -- the 7B cell's
zero is the one most affected by (1).

### Freeze broken once, 2026-09-04, and why

The first frozen sweep scored Qwen2.5-VL-7B as failing its protocol on 13 of 18 moves. It
was not: the model emitted a well-formed JSON object inside a fenced block, wrapped in
prose, and the harness's greedy `\{.*\}` refused it. Scoring presentation as capability
would have made the sweep worthless, so the extractor now accepts a fenced block, or the
first balanced object anywhere in the reply, or a bare object.

Per the rule above, **every model was re-run after this change**. No result from before it
is reported.

## First sweep on the frozen framework, 2026-09-04

Abstraction rung, 18 moves, temperature 0.2, seed episode 0, one sample per model.

| model | outcome | how it failed |
|---|---|---|
| Qwen2.5-VL-72B | not passed | every submitted body binds -- the pseudo-variable defect is gone -- but it loops on `Reach <place>`, which cannot resolve, and never removes the action the oracle keeps refusing |
| Qwen2.5-VL-7B | not passed | stops emitting a valid tool request after move 5 and repeats a malformed one for the remaining 14 |
| Qwen3-VL-30B | deferred | its endpoint is running the P0 chain, and this project does not put two generation jobs on one endpoint |

Context exhaustion is ruled out: the conversations reached ~6.7K and ~8.2K tokens against a
16,384 window.

The shared failure is worth naming because it is not about operators at all: **neither model
uses the oracle's answer to change course.** 72B re-issues an action the simulator has
refused four times; 7B re-issues a request the protocol has rejected fourteen times. Tool
access removed the structural error and exposed a behavioural one.

**This is one draw per model.** The evaluation arms in this project are not greedy and a
single run is a single sample; the same caution applies here. Before any of this is quoted
as a capability, the rung has to be run over several seed episodes and several samples per
model, and reported as a rate.

## Pass rate on the abstraction rung, 2026-09-04

5 seed episodes x 3 samples = 15 runs per model. 18 moves, temperature 0.7, sampling seeds
20260829+k. The framework is frozen; only the starting episode and the sampling seed vary.
Pass = a submitted operator binds and achieves its effect on at least two episodes the run
was not shown.

| model | pass rate | by seed episode (0 / 1 / 3 / 7 / 9) |
|---|---|---|
| Qwen2.5-VL-72B | **6/15 = 40%** | 1/3, 2/3, 0/3, 1/3, 2/3 |
| Qwen2.5-VL-7B | **0/15 = 0%** | 0/3 throughout |
| Qwen3-VL-30B | **6/15 = 40%** | 0/3, 1/3, 2/3, 0/3, 3/3 |

The rung is passable and not a trick gate: when 72B passes it usually passes quickly --
6, 6, 7, 7, 12 and 15 moves of the 18 allowed.

Difficulty is episode-dependent: 72B is 0/3 from episode 3 and 2/3 from episodes 1 and 9,
which is the reason for measuring a rate rather than quoting one transcript.

One of the fifteen 7B runs did not fail the task -- it died with a 400 from the server, its
own prose having filled the 16,384-token window (15,870 input tokens with 1,500 reserved for
the reply). The other fourteen are genuine failures. Verbosity is a property of the model,
but a context crash is not a wrong answer and is reported separately rather than folded in.

### The curve is a threshold, not a gradient

    Qwen2.5-VL-72B    6/15 = 40%
    Qwen3-VL-30B      6/15 = 40%
    Qwen2.5-VL-7B     0/15 =  0%

30B ties 72B on the rate while failing on a different set of starting episodes (72B is 0/3
on episode 3 and 1/3 on 7; 30B is 2/3 on episode 3 and 0/3 on 7). Whatever the rung needs,
30B has as much of it as 72B, and 7B has none of it.

This is worth reporting against the inference-side delegation curve on the same three
models, which is graded rather than stepped: 91.56% / 79.00% / 21.75% on VIKI-L2 ID. **The
ability to derive an operator under verification and the ability to use one are not the same
capability and do not scale together.** Fifteen runs per model is enough to separate 0% from
40% and not enough to separate 40% from 40%; a difference between 72B and 30B, if there is
one, would need more runs.
