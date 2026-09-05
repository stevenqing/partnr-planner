# Agentic Operator Induction — Method

**读法（3 行）.** 这份是**论文用的方法描述**，英文，可直接搬进 submission 的 Method /
Validity 两节。它**不含结果**（结果在 `VIKI-L2-PAPER-RESULTS.md`），也**不含实验日志**
（过程、破冻结的经过、坑，在 `AGENTIC-OPERATOR-INDUCTION.md` 与各 `HANDOVER-*.md`）。
每一条主张后面都跟着它的控制；没有控制的话不写进这里。

---

## 1. Problem statement

A skill memory is only as general as the procedure that built it. Our prior artefact is
induced by *replay induction*: training episodes are replayed in the simulator, and a
hard-coded rule decides, at each step where a goal predicate becomes true, which agent
achieved it, which of its actions constitute the achieving body, and how that body
abstracts into a reusable operator. The rule is eight lines of verb-whitelist and
name-matching. It is fast, deterministic, and auditable — and it can only ever recover the
demonstrations its author anticipated.

This is not a rhetorical concern; it is measurable, and we measure it before proposing
anything (§3). The method below replaces the *proposal* step with an agent while leaving
*acceptance* entirely mechanical. The one-sentence statement of the contribution is:

> **The agent only ever proposes. Every operator that enters the library is admitted by an
> execution in the simulator or a judgement by the benchmark's official checker. No operator
> enters on the model's say-so.**

## 2. Preliminaries

**Episodes and traces.** An episode carries an initial world state, a natural-language
instruction, a set of goal predicates, and optional temporal constraints. Replaying a
reference plan in the simulator yields a *trace*: a per-step record of each agent's action
and the resulting world state, together with the steps at which each goal predicate first
becomes satisfied (its *completions*).

**Operators.** An operator is a typed record `(effect, body, preconditions, runner_types,
cost)`. The `effect` is a predicate schema over variables (e.g. `?x` at `pos.name = ?y`);
the `body` is a sequence of parameterised primitive actions; `preconditions` are world facts
that must hold at binding time; `runner_types` restricts which embodiments may execute it.
Operators are of three kinds — *achievement* (one agent suffices), *coordination* (two or
more agents with role variables and cross-role ordering counts), and *repair* (unblocks a
precondition, e.g. opening a sealed container).

**The memory artefact has three layers.**

| layer | content | built by |
|---|---|---|
| 1 | the operator library | induction over replayed traces |
| 2 | which pairs of requirements must be ordered | pattern mining over episodes' own temporal constraints |
| 3 | the domain's canonical asset and place names | counting over training episodes |

**Only Layer 1 is the object of this method.** Layers 2 and 3 are held fixed across every
arm; §5.4 states precisely what that does and does not concede, and gives the mechanical
check that backs the statement.

**Two halves, fixed in advance.** Training episodes are split deterministically:
`episodes[::2]` is the *induction half*, `episodes[1::2]` the *self-check half*. Nothing in
the induction procedure — including every tool the agent may call — is given access to the
self-check half, so tool use cannot fit the gate.

## 3. Why the proposal step is the binding constraint — measured, not assumed

We instrument the shipped inducer to count what it is shown and discards. On PARTNR, over
the 161 rearrange traces the rearrange-only library reads:

```
propositions satisfied            462
discarded by the inducer          180   = 39.0%
  is_on_top      425 seen, 144 lost   33.9%
  is_in_room      36 seen,  36 lost  100.0%
```

Over all 399 `train_mini` traces, 820 of 1711 satisfied propositions (47.9%) are discarded;
674 of those are attribution failures. The cause is uniform: the inducer requires the action
at the satisfying step to carry a verb on a `COMPLETING` whitelist *and* to name one of the
proposition's entities. At the failures the action is `Navigate` (883) or `Wait` (437).
Two consequences differ in kind and must be separated:

* `is_in_room` is achieved by *carrying an object while navigating*. `Navigate` is not a
  completing verb, so this predicate **can never be learned, in any split**. The machinery
  that would execute such an operator already exists downstream; only the entry is missing.
* `is_next_to` is achieved as a *side effect* of a placement: the placed object ends up
  beside a reference object the action never names, so the name-match fails (86% lost).

On VIKI-L2 the same instrument returns **zero**: 2698 episodes replayed with no replay
failures, 5147 completions found, 0 discarded (4808 achievement, 339 coordination). The
whole training set contains exactly two predicate shapes (`pos.name` 4183, `is_activated`
964), which are exactly the two the hard-coded rule handles.

**This negative result determines the experimental design and we report it as such.**
Induction on VIKI-L2 is lossless, so an agentic proposal step has nothing to *recover*
there. VIKI-L2 is nonetheless the primary testbed, for the opposite reason: because a
known-correct target library exists, the question "did the agent derive the right
operators?" is *falsifiable* there and is not falsifiable where no such target exists.
PARTNR is the setting in which the method can exceed the rule, and `is_in_room` — lost 100%
of the time in every split, and demanded by 61 of the 70 episodes that our rearrange-only
memory cannot score — is the concrete target. The two benchmarks therefore answer different
questions: **VIKI-L2 asks whether the induced library is right; PARTNR asks whether it
reaches what no hand-written rule reaches.**

## 4. The framework

### 4.1 Induction is a sequence of decisions, not one output

The first harness we built asked a model to write a six-stage inducer — attribution,
segmentation, abstraction, preconditions, repair mining, deduplication — and returned a
single scalar: the resulting library's self-check rate. Six decisions, one bit of feedback.
The failure mode this produces is not a prompting problem but a credit-assignment problem,
and it is stark: the model wrote pseudo-variables (`?apple_0`) where schema variables
belonged; the planner silently discards unbindable operators; **25,542 operators were
produced, 0 could be bound, and the self-check was 0.0 every time.** The signal never
reached the decision that was wrong.

### 4.2 One mechanical oracle per decision

The framework's core is that every induction decision admits a cheap, mechanical oracle,
exposed to the agent as a tool. The enabling fact is measured rather than assumed: replaying
the entire induction half takes **3.5 seconds**, so a single episode costs milliseconds and
these oracles can be called tens of thousands of times.

| decision | oracle exposed to the agent | cost |
|---|---|---|
| attribution | `check_actor` — counterfactual replay of one agent's actions | one replay |
| segmentation | replay the proposed body alone; does the effect hold? | one replay |
| **abstraction** | `run_operator` — bind on a **different** episode and execute | two replays |
| preconditions | drop a fact, replay; if the effect still holds, drop it permanently | one per fact |
| repair | does the blocked operator succeed after the proposed repair? | two replays |
| deduplication | do two operators achieve the same effects on the same states? | batch |
| legibility | `contrast_actors` — every agent's collapsed decision sequence, side by side | none |

Two design rules govern what an oracle may say. It reports **what the world became**, never
what the agent should have written: `run_operator` returns the post-execution state of the
effect predicate ("the object ended on the robot, not on the bowl"), because that is an
observation; it never reports "you wrote `Place ?x` where the demonstration wrote
`Place ?y`", because that is the answer. And `try_bind` calls the planner's own `chains_for`
rather than a reimplementation of it, so a binding that the tool accepts is a binding the
planner accepts; when binding fails it names the tokens left unbound.

The measured effect of the decomposition, with the model held fixed, is that structural
failure disappears and the residual error becomes a single token: under scalar feedback no
submitted body could be bound; under local oracles **every** submitted body bound, and the
remaining failure was one wrong argument in one action. Converting an opaque `0.0` into a
localised error is the entire purpose of the decomposition.

### 4.3 What is never delegated

`_runs_alone` — the counterfactual replay that decides whether a segment is achievable by
one agent or genuinely requires several — is **not** delegated. It is the criterion that
distinguishes an achievement operator from a coordination operator, and it must remain an
execution rather than a judgement. Replay itself, deduplication, and provenance recording
are likewise mechanical throughout. Delegating the independence test would make the method's
central distinction a matter of the model's opinion, and would invalidate it.

### 4.4 Acceptance: marginal contribution, not effect achievement

The naive acceptance test — *does this operator achieve its stated effect on at least two
held-out episodes?* — is inadequate, and we establish this empirically rather than by
argument: an operator already present in the library passes it trivially. Under that test,
40 submissions deduplicated to 3 distinct operators, 15 of them re-derivations of a body
already held, each accepted on holdout episodes the planner still could not solve. The
library did not move.

Acceptance therefore requires a **coverage gain**: adding the operator must take at least
one holdout episode from *unsolved* to *solved* under the planner. The agent is told which
episodes the current memory cannot solve; a resubmission of a held body is refused with the
reason "adds no coverage" rather than accepted. Two further admission rules bound gaming:

* **Minimality.** An operator whose body is the entire reference plan passes any effect
  test, so minimality is mandatory: leave-one-out over the body, dropping any action whose
  removal leaves the effect holding. Cost is recorded.
* **Support ≥ 2.** An operator must fire on at least two distinct episodes to enter the
  library.

### 4.5 The rung protocol

Induction is decomposed into *rungs*, one per decision, run independently so that a failure
localises. A rung presents one seed episode, grants tool access, allows **18 moves**, and
ends when the agent submits an operator or exhausts its moves. Submissions are parsed
permissively — fenced JSON, a bare object, or an object embedded in prose, under any of the
accepted request keys — on the stated ground that refusing a well-formed answer for its
presentation scores presentation rather than work (§5.6 records what enforcing this cost us
before it was fixed). A capability measurement over a rung is reported as a **rate** over
seed episodes × sampling seeds, never as a single transcript, because difficulty is strongly
episode-dependent.

### 4.6 Library assembly

Accepted operators are deduplicated by effect schema and body signature, and each is
assigned a **measured** support: the operator is executed over 60 induction-half episodes
and support is the count of episodes on which it binds and achieves its effect. The model's
self-reported support is discarded. Provenance is recorded per operator: the traces that
proposed it and the held-out episodes that verified it.

## 5. Validity

### 5.1 Information barrier

Every tool reads `episodes[::2]` only. The half on which the build-time gate plans is never
exposed to the agent, so no amount of tool use can fit the gate. The end-to-end benchmark
arm scores the *test* split, which no part of the induction loop can see.

### 5.2 Anti-gaming

Minimality and support ≥ 2 are stated in §4.4. In addition, because the library is a random
variable under sampling, it is built under three seeds and the spread is reported, matching
the treatment the evaluation arms already receive.

### 5.3 Leakage control and provenance

An LLM asked to propose an operator may produce one from pretraining rather than from our
traces. If that happens, the word *memory* does not apply and the result is worthless. Four
arms, fixed before any run:

| arm | Layer 1 source | purpose |
|---|---|---|
| (a) | replay induction, rearrange-only | the current baseline |
| (b) | **agentic, shown only the rearrange traces** | the claim |
| (c) | replay induction, all task types | coverage ceiling |
| (d) | **agentic, given the predicate vocabulary and no traces** | the decisive control |

**If (d) ≈ (b), the agent was not using our data.** Alongside the arms, a mechanical
provenance check runs automatically on every build: an effect schema with no support in the
proposition set of any replayed trace is flagged as *ungrounded*, and the count is reported.

The agent may see replayed traces, world states, the predicate vocabulary of the *training*
episodes, and the type system. It may not see test episodes, the evaluation split's
propositions, or any example of a task type outside the library's declared scope.

### 5.4 Scope of "agent-built": Layer 1, and shared inductive bias

The end-to-end evaluation requires all three layers, so an agent-built Layer 1 is grafted
onto a reference artefact's Layers 2 and 3 (`scripts/viki_memory_from_library.py`). The
obvious objection is that the borrowed layers import knowledge derived from the reference's
hand-written operators, making the result not the agent library's. **The objection is
answerable, and we answer it mechanically rather than by assertion**
(`scripts/viki_layer23_equivalence.py`, four checks, report written to disk):

1. **Signature independence.** `dependencies.mine(episodes, sim, seed, per_family,
   exclude_family)` and `vocabulary.harvest(episodes, exclude_family)` accept no library
   argument, so no Layer 1 can reach either builder. Layer 2's mining reads its `visits`
   feature off the *replayed training trace*, not off any operator body.
2. **Graft invariance.** Running the shipped grafting script with two deliberately unlike
   Layer 1 libraries yields byte-identical Layer 2 and Layer 3.
3. **Fold reference.** Held-out-family cells must borrow *that fold's* layers, since the
   fold rebuilds all three with the family excluded. The check quantifies what would
   otherwise leak.
4. **Rebuild identity.** Layers 2 and 3 are rebuilt from the artefact's own recorded seed,
   `per_family` and `excluded_family` and compared byte for byte with what is stored.

Borrowing is therefore a computational shortcut, not an import of hand-authored entries.
**What is genuinely shared and authored** must still be stated: Layer 2's hypothesis space
(six structural features relating a pair of requirements, plus thresholds `MIN_SUPPORT = 30`
and `MIN_PRECISION = 0.9`) and Layer 3's harvesting and name-normalisation rules. These are
*inductive bias*, held fixed and identical across every arm including all baselines; they
are not memory entries. The honest scope of the claim is: **the operators are agent-derived
under mechanical verification; the ordering patterns and vocabulary are mined, not authored;
the mining procedures themselves are authored and shared.**

One coupling must be reported because it is load-bearing for the results. Layer 2's *rules*
are Layer-1-independent, but their *application* is not: at planning time the ordering
constraint for a requirement pair is computed from the bodies of the operators the memory
would consider (`visits_of`). An operator body that is too short causes the pattern not to
match and **no ordering constraint to be emitted at all**. A deficient Layer 1 therefore
manifests as an ordering failure rather than as a coverage failure, and diagnosis must
distinguish the two.

### 5.5 Framework freeze

After the oracles, the rung protocol and the gate are settled, the framework is **frozen**:
the model becomes the only variable across the sweep. Any subsequent change to the framework
invalidates cross-model comparison and obliges a re-run of **every** model. We enforce this
literally, and we report every break together with the re-run (§5.6). No pass rate collected
under a superseded framework version is reported.

### 5.6 Instrument validation, reported as a first-class result

Verification harnesses can fail correct answers, and a harness that does so is
indistinguishable, from the outside, from a model that cannot answer. We therefore adopt a
standing rule and report its consequences rather than only its conclusion:

> **When a rung scores zero, run the reference library's own correct answer through the same
> acceptance test before drawing any conclusion about the model.**

Applied retrospectively, this rule overturned four of five readings of "these models cannot
produce the missing operator variants". The defects were: submissions discarded on the key
they arrived under (152 of 524 transcripts, worst on the smallest model); an acceptance test
that executed only the first agent, which refuses the reference library's own operator on
every episode it provably solves; evidence that was faithfully returned but not legible (the
needed variant is performed by the *other* agent in every seed episode); and an
effect-only acceptance test that admitted duplicates (§4.4). Repairing them moved the
end-to-end score of the induced library from 32.47% to 74.24% **with no change to any
model, seed, holdout or acceptance standard** — none of the repairs relaxes what an operator
must achieve; they stop the harness from failing correct answers.

The consequence for reporting is stated plainly: every pass rate collected before the
repairs is a **lower bound**, and the capability curve was re-measured on all three models
under the repaired framework before being used. The earlier curve suggested a threshold
(the smallest model at exactly 0%); the re-measured curve does not, and the apparent
threshold was an artefact of the discarded-submission defect, which affected that model's
cell most.

## 6. Evaluation protocol

### 6.1 Two arms, and why the build-time gate must never be quoted as performance

| arm | what it plans | what it scores |
|---|---|---|
| **gate** (`viki_inducer_bench.py`) | held-out **training** episodes, from oracle goal predicates, with the symbolic planner | self-check rate |
| **end-to-end** (`viki_eval_skill_memory_v2.py`) | the **test** split, from model-parsed goals | benchmark accuracy |

The two agree closely for a strong library and diverge enormously for a weak one. **The gate
is an inner-loop signal only; no claim about what the framework achieves is stated without
the end-to-end number.** Equally: the end-to-end arm scores the test split, so using it to
select operators, seeds or hyper-parameters would be fitting to test. It is a reporting
instrument, never a selection instrument; selection stays on the induction half.

### 6.2 Measurement axes

A library is model-agnostic, so it is judged on more than one axis:

1. **The library alone** — oracle goal predicates, the model removed entirely. This isolates
   coverage and executability from any reasoning ability.
2. **In-distribution × model scale** — the benchmark's evaluation manifest, model responses
   replayed from disk so that no difference between configurations can be a resampling
   artefact.
3. **Held-out family** — each family is planned by a memory rebuilt as if that family had
   never existed: operators re-induced, ordering re-mined, vocabulary re-collected (§5.4,
   check 3).
4. **Recombination** — each instance demands two capability modes that never co-occur in any
   single training trajectory.

Axes 2–4 are run offline from archived responses wherever possible, so that a cell costs no
tokens and cannot move under resampling. Every reported cell names the split, the model, the
dispatch arm and the scoring convention.

### 6.3 Statistics and scoring conventions

Binary per-instance outcomes are compared with the exact McNemar test over paired instances.
Continuous completion scores are compared with the paired Wilcoxon signed-rank test or a
bootstrap null band; McNemar is not applicable to them. Where a configuration is repeated
under several sampling seeds, the standard deviation is reported alongside the mean, and
where the standard deviation is of the same order as the mean the cells are explicitly
declared unsuitable for ranking. Parsing tolerance is fixed and stated once, applied
identically to every arm including all baselines.

## 7. Limitations

1. **Coordination cannot be verified by the current workbench.** `run_operator` executes with
   a single runner, so a genuinely multi-agent operator can never be admitted. This is a
   property of the verifier, not of the models, and it caps the attainable library
   independently of any model. Coordination is the sole family the induced library fails
   under oracle goals.
2. **Layers 2 and 3 are borrowed** at evaluation time, with the equivalence argument and its
   four checks in §5.4. The authored inductive bias in those layers is shared with, not
   exclusive to, the proposed method.
3. **VIKI-L2 has zero induction headroom** (§3). The framework cannot demonstrate *recovery*
   there; it demonstrates *derivation against a known-correct target*. The recovery claim
   belongs to PARTNR and must be evidenced there.
4. **PARTNR admits a weaker acceptance criterion.** Its traces record actions without world
   state, so counterfactual replay is unavailable, and the target of `is_in_room` is a room
   identifier that cannot be resolved offline to a furniture-to-room mapping. Acceptance on
   that predicate reduces to subject-plus-temporal agreement, weaker than the criterion
   available for placement predicates. The decisive adjudication is consequently the outer
   gate: rebuild the library, run the privileged sweep, and compare against a clean baseline.
5. **Capability rates are measured at n = 15 per model** (5 seed episodes × 3 samples). This
   separates a near-zero rate from a moderate one; it does not separate two moderate rates,
   and no such separation is claimed.

## 8. Reproduction

| purpose | entry point |
|---|---|
| oracles / workbench | `scripts/viki_induction_tools.py` |
| abstraction rung | `scripts/viki_agentic_rung_abstraction.py` (`--library`, `--target-key` opt-in) |
| build-time gate | `scripts/viki_inducer_bench.py` |
| library → memory artefact | `scripts/viki_memory_from_library.py` (`--reference` for fold cells) |
| Layer 2/3 independence checks | `scripts/viki_layer23_equivalence.py` |
| end-to-end benchmark arm | `scripts/viki_eval_skill_memory_v2.py` |
| held-out-family folds | `scripts/viki_eval_v2_intent_folds.py` |
| headroom instruments | `scripts/partnr_induction_headroom.py`, `scripts/viki_induction_headroom.py` |
| passability control (reads the reference; must never gate seed selection) | `scripts/viki_family_passability.py` |
| PARTNR workbench and its two-way calibration | `scripts/partnr_induction_tools{,_selftest}.py` |

Results, with their splits and conventions, are reported in `VIKI-L2-PAPER-RESULTS.md`.
