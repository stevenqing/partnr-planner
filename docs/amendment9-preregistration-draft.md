# Amendment 9 (draft) — Grounded Skill Memory, and an evaluation that can distinguish it

Status: DRAFT. Nothing here is frozen. Written 2026-08-30, before any Amendment 9
run exists. The Amendment 8b results it reasons from are final and unmodified.

## 1. What Amendment 8b established

Official VIKI-L2-Interactive scorer, 924 paired rows, natural stage:

| arm | task | format | McNemar vs skill_memory |
|---|---|---|---|
| gmemory | 9.85% | 20.5% | skill_memory loses, p = 1.9e-05 |
| skill_memory (K=4) | 7.58% | 99.7% | — |
| zero_shot | 2.81% | 100% | skill_memory wins, p = 3.7e-11 |
| trajectory_rag | 2.06% | 54.9% | skill_memory wins, p = 4.3e-11 |

Three measured facts drive this draft.

**F1. The official gap understates G-Memory.** All 735 of its format failures are
one cause: it emits JSON `null` for idle robots and the scorer parses `<answer>`
with `ast.literal_eval`, which has no `null`. Re-parsing with `json.loads` and
dropping null actions recovers 380 correct plans: 9.85% -> 50.97%. Verified not to
be an artifact of the normalisation — all 380 recoveries occur on rows whose step
count is unchanged, and the 13 rows containing an all-null step recover none.
skill_memory is unaffected (3 format failures, 0 recovered).

**F2. Our differentiator is not reaching the prompt.** Cooperation skills dominate
retrieval (80.2% of top-4; 0 of 116 sampled rows had a top-4 without one) and score
higher than individual skills (abstract_score median 0.4255 vs 0.3961). But the
coordination structure lives in 5 of 412 coop skills:

| skill | instances | with roles |
|---|---|---|
| division_of_labor | 6682 | 6682 |
| synchronization | 3121 | 3121 |
| sequential_handoff | 2592 | 2592 |
| complementary_work | 101 | 101 |
| theory_of_mind | 17 | 17 |
| the other 407 skills | 3466 | 0 |

Those 5 are never retrieved. `_context_to_text` builds an instance embedding from
objects / locations / rooms / object_locations / action_sequence only. A role
instance has none of those, embeds near-empty text, scores last, and is cut by
`instance_top_k`. Measured: 0 of 705 retrieved coop skills carry role fields.
So the 407 task-specific coop skills that *are* retrieved are multi-agent action
sequences — structurally the same kind of object G-Memory retrieves, and G-Memory
retrieves it better.

**F3. The benchmark is in-distribution by construction.** VIKI-L2 has 14 task
families. All 14 test families appear in train; 1800 of 1800 test rows have a
same-family neighbour in the memory bank. Nearest-trajectory retrieval is close to
optimal here, and abstraction can only lose information that was not needed.

## 2. The claim worth testing

Skill abstraction should pay off when there is no near-duplicate to copy — when the
test task is phrased differently, or composes familiar sub-tasks in an unfamiliar
way. On in-distribution data it cannot pay off, because copying is available and
better. Amendment 8b therefore did not test the claim; it tested the regime where
the claim predicts no advantage.

H1. On held-out task families, grounded skill memory beats G-Memory.
H2. On in-distribution rows, it does not, and we say so.

H2 is stated in advance so that an in-distribution loss is not later reported as a
limitation discovered after the fact.

## 3. Method: grounded skill memory

Three changes. Each fixes a defect measured above; none is a free parameter chosen
after seeing an outcome.

**C1 — role-aware instance embedding.** Extend the instance text function so agent
roles and coordination mechanism contribute to the embedding, making the 12,513
role instances retrievable. This changes the method, not the harness.

**C2 — provenance grounding.** Record `source_episode_id` on every instance at build
time; at render time attach that episode's concrete multi-robot step sequence. The
skill says which pattern applies and why; the grounded trace shows one concrete,
step-aligned realisation. This is the step structure the scorer measures and the
thing our abstraction currently discards. Retrieved fragments today come from
different episodes and contradict each other — for one pear-to-sink query the top
four coop skills proposed `Push[cardboardbox,R1]`, `R3`, `R2` and no push at all.

**C3 — LLM relevance rescoring, adopted from G-Memory.** G-Memory scores two
candidates with the LLM and keeps the better one. Adopt the same step over our top
candidates and render one grounded skill instead of eight fragments. This is
explicitly borrowed; it is reported as borrowed.

What stays ours: the retrieval index is the skill hierarchy — preconditions,
effects, roles — not instruction similarity. That is the part the held-out-family
experiment is designed to test.

## 4. Evaluation

Splits are constructed, because VIKI ships none:

- **ID**: current 924-row manifest, memory bank unchanged. Expect no win (H2).
- **Held-out family**: for each of the 14 families, rebuild the bank with that
  family's episodes removed and evaluate that family's rows. Reported as the mean
  over folds and per fold, not as a single favourable fold.
- **Compositional**: rows whose ground-truth plan uses an action pair present in
  the bank only in separate families.

Arms: zero_shot, trajectory_rag, gmemory, skill_memory (K=4, as filed), grounded
skill memory. Same 6699-episode source, same partner prefix, same seed.

Scoring: official scorer as primary. The JSON-tolerant variant from F1 reported
alongside for every arm, since the official number understates any arm that emits
`null`. Neither replaces the other, and the scorer is not modified.

Token budget: the Amendment 8b protocol stands — natural stage first, then trimmed
to the per-row minimum across memory arms. Note for the record that the K=8 rich
render measured 712 memory tokens against G-Memory's 577, i.e. +23%, because the
chars-per-token ratio of the old renderer (4.48) does not hold for the new one
(3.6). Any budget-matched claim uses K=6, not K=8.

## 5. Failure conditions

- If C1 does not raise retrieved role coverage well above the measured 0 of 705,
  C2 and C3 are not run and the negative result is reported.
- If the held-out-family split shows no advantage, the conclusion is that skill
  abstraction does not transfer better than trajectory retrieval on this benchmark.
- Fold results are reported for all 14 folds.
