# VIKI Amendment 1: Branch Memory and Strict Composition

## Status

**GATE A1 passed as a diagnosis of the output flips, but the proposed mechanism
did not hold and the redesigned method failed GATE A5.** Wrong branch or object
selection explains 62 of the 73 legal OOD regressions. However, A0 found no
all-present placement branch in train memory: all 7,024 applicable train rows
were already labeled some-absent. The claim that OOD failure was caused by
retrieving all-present train demonstrations is therefore contradicted.

The branch-indexed grounded arm then lost 9.00 percentage points on the frozen
300-row ID safety sample, and the graded arm lost 5.67 points. Both violated
the preregistered -1 point safety floor, so **Track A halted at GATE A5**. No
Track A OOD calls were made. Branch-complete memory did not improve VIKI and
must not be described as doing so.

Track C built and completed three deterministic 400-row channels. C0 and C2
passed. The two strict channels have zero full-signature, exact-plan, and
plan-trigram overlap, but both degenerate to the same 172 observation-only
`dog_check_environment` memory rows. C3 found no strict skill-memory advantage.
These results characterize the frozen Track A method after its rejection; they
do not reverse the A5 decision.

C4 was not performed. The PartNR memory subsets and generation logs behind
Table 1 and Figure 3 are not present locally, and the paper text provides no
artifact URL from which to recover them. Inventing or reconstructing different
subsets would not be the identical audit required by the amendment.

This remains a static, single-robot VIKI evaluation. It does not measure
decentralized cooperation or cooperation memory.

## Frozen Protocol

**The frozen-artifact and no-post-result-tuning requirements held.** The
released Qwen2.5-VL 7B L2 checkpoint, VIKI-R data, published baseline logs,
7,196-row train memory, MPNet embeddings, and original top-k 5 retrieval were
not modified. Generation used vLLM 0.8.4, Transformers 4.51.3, BF16, a 4,096
token context, at most 2,000 output tokens, greedy decoding, and scorer seed
equal to the original dataset row index.

Amendment outputs are under
`results/viki_memory_experiments/amendment1/`. New implementation is in:

- `habitat_llm/evaluation/viki_branch_conditions.py`
- `habitat_llm/evaluation/viki_branch_memory.py`
- `habitat_llm/evaluation/viki_composition.py`
- `scripts/viki_amendment1_analysis.py`
- `scripts/viki_branch_manifest.py`
- `scripts/viki_amendment1_a5.py`
- `scripts/viki_composition_splits.py`
- `scripts/viki_composition_audit.py`
- `scripts/viki_composition_c3.py`

## A0: Availability Predicate

**GATE A0 passed 50/50, while the preregistered train-branch diagnosis was
contradicted.** The parser uses only each row's instruction and `init_pos`.
Asset, portable-asset, and region vocabularies are frozen from train, and the
same code path is applied to train, ID test, and OOD val. It never reads an
evaluation row's reference plan, `time_steps`, or goal constraints.

The census was:

| Split | Rows | All present | Some absent | Not applicable |
| --- | ---: | ---: | ---: | ---: |
| Train | 7,196 | 0 | 7,024 | 172 |
| ID | 1,800 | 0 | 1,750 | 50 |
| OOD | 1,218 | 0 | 1,218 | 0 |

All OOD rows were some-absent, consistent with split construction. There were
no OOD exceptions. The only not-applicable rows were observation-only
`dog_check_environment` examples.

Every OOD-routed placement skill already had far more than five usable
some-absent train instances:

| Skill | Some-absent train instances |
| --- | ---: |
| `ensure_all_fruits_on_table` | 1,232 |
| `single_move_asset_to_target` | 824 |
| `sequential_pick_two_and_place` | 402 |
| `serve_bread_after_checking_cabinet` | 304 |

The decisive contradiction is not subtle: the assumed missing branch was
already abundant in train memory.

Artifacts: `a0_branch_census.parquet`, `a0_branch_census.csv`, and
`a0_branch_census.summary.json`.

## A1: Regression Autopsy

**GATE A1 passed: 62/73 legal goal misses, or 84.93%, were branch decisions,
above the required 37/73 threshold.** This confirms the local behavioral
account that memory often changes the selected branch or object. It does not
confirm the proposed all-present contamination mechanism, which A0 and the
retrieved-demo census directly refute.

| Classification | Count |
| --- | ---: |
| Uses a present asset where baseline fetches the absent asset | 62 |
| Copies injected-demo structure | 8 |
| Swaps the manipulated object | 3 |
| **Legal goal misses** | **73** |

The broader non-exclusive flags found 65 object swaps, 62 omitted required
fetches, and 17 cases with copied demo structure. For the 54 legal fixes, all
54 memory outputs selected the required absent asset.

The plate-missing contrast rules out the amendment's specific contamination
story. Regressions and fixes both received five some-absent demos on average,
zero all-present demos, and nearly identical retrieval similarities:

| Plate-missing group | N | Mean max similarity | Median max similarity |
| --- | ---: | ---: | ---: |
| All regressions | 78 | 0.6124 | 0.5971 |
| All fixes | 17 | 0.6138 | 0.6202 |

The evidence therefore supports a narrower result: injected memory can change
the branch/object decision, but the broad some-absent label and instruction
similarity do not distinguish beneficial from harmful changes.

Artifacts: `a1_legal_regression_autopsy.csv`,
`a1_legal_fix_autopsy.csv`, and `a1_autopsy.summary.json`.

## A2: Counterfactual Synthesis

**A2 was not triggered, exactly as required by the preregistered trigger.** No
OOD-routed placement skill had fewer than five some-absent train examples.
`dog_check_environment` had no placement conditions and was an OOD routing
error, not a placement skill eligible for synthesis.

No counterfactual rows were produced, no simulator or LLM fallback calls were
made, and no evaluation row contributed to memory. This is important because
creating synthetic examples despite the failed trigger would have been a
post-hoc change to the protocol.

Artifact: `a2_a4.summary.json`.

## A3: Branch-Indexed Retrieval

**GATE A3 passed 50/50: disabling branch indexing reproduced the V0 demo IDs
exactly and in order.** With indexing enabled, retrieval first routes to a
skill, retains executable instances with the row's branch label, and ranks
those candidates by context similarity. It does not fall back to grounded
cross-branch examples.

On the 1,218 OOD rows, the ungraded grounded configuration produced 1,193
grounded tiers and 25 abstract tiers. Exactly 25 rows changed from V0, the
misrouted `dog_check_environment` cases with no compatible placement memory.

Artifacts: `a3_a4_ood_manifest.parquet`,
`a3_a4_ood_manifest.csv`, and `a2_a4.summary.json`.

## A4: Graded Injection

**The A4 train-only calibration was applied as preregistered, but it rejected
grounded injection for every OOD row.** The similarity bar was the 10th
percentile of each train row's nearest different-context neighbor within the
same skill and branch:

```text
train_similarity_bar = 0.9034629464149475
```

All 1,218 OOD rows fell below that bar and received the abstract tier. Abstract
descriptions passed the object-token prohibition and contained no grounded
plan. The calibration therefore exposed a substantial train-to-OOD similarity
shift rather than providing a useful graded mixture on this split.

Artifact: `a2_a4.summary.json`.

## A5: ID Safety Gate

**GATE A5 failed for both arms, so the three preregistered OOD predictions were
not tested.** The seed-20260814 sample contained 300 ID rows and excluded every
calibration row. Both arms were paired against the same frozen baseline rows.

| Arm | Baseline | Arm | Delta | Fixes | Regressions | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Branch-indexed grounded | 278/300 (92.67%) | 251/300 (83.67%) | -9.00 pp | 3 | 30 | 0.0000014012 |
| Branch-indexed graded | 278/300 (92.67%) | 261/300 (87.00%) | -5.67 pp | 3 | 20 | 0.0004882813 |

The allowed floor was -1.00 point for each arm. The degradation was present
for both one-robot and two-robot rows; three-robot rows were 0/9 in all arms.
This is not a marginal gate miss. The grounded arm created ten regressions on
`clear_table_with_two_robots_and_put_in_cabinet` alone, while the graded arm
still created nine.

Per the mandatory halt, the following were not run:

- branch-indexed grounded OOD evaluation;
- branch-indexed plus graded OOD evaluation;
- per-subset A5 OOD deltas and McNemar tests;
- optional P1 full-ID unconditional-memory arm.

Consequently, the predictions that plate-missing would recover to within two
points of 92.58% and bowl-missing would exceed 11.49% remain untested. There is
also no observed both-missing result for the redesigned method.

Artifacts: `a5_id_safety_manifest.parquet`, `a5_id_safety.parquet`,
`a5_id_safety.jsonl`, and `a5_id_safety.summary.json`.

## C0: Composition Census

**GATE C0 passed 50/50.** Deterministic parsing decomposed reference plans into
`fetch`, `relocate`, `open_container_then_retrieve`, `check_then_act`,
`state_change`, and counted `multi_object_sequence` units. A full signature is
the ordered multiset of units plus reference length and relocated payload
object count. Reference plans were used only to define and audit splits; they
never entered prompts or memory.

| Split | Rows | Unique full signatures |
| --- | ---: | ---: |
| Train | 7,196 | 341 |
| ID | 1,800 | 304 |

Artifacts: `c0_composition_census.parquet`,
`c0_composition_census.csv`, and `c0_composition_census.summary.json`.

## C1: Dual-Channel Splits

**C1 produced all three seeded 400-row channels, but both strict constructions
degenerated to observation-only memory.** The recorded seed was 20260814 and
the systematicity pair threshold was at least 30 ID rows.

| Channel | Test rows | Allowed memory rows | Full-signature overlap |
| --- | ---: | ---: | ---: |
| Episode-heldout | 400 | 172 to 1,232 per row | 400 |
| Task-heldout productivity | 400 | 172 per row | 0 |
| Task-heldout systematicity | 400 | 172 per row | 0 |

Episode-heldout uses other train episodes from the same task category. The
productivity pool permits only single-unit signatures. The systematicity pool
excludes every row containing a held-out unit pair. In this VIKI task grammar,
both strict rules leave exactly the 172 `dog_check_environment` train rows.

This satisfies the literal strict definitions, but it means the strict arms
mostly test whether an observation-only memory can help multi-action planning,
not whether a rich library recombines familiar elementary manipulation skills.
That limitation is reported rather than repaired after seeing the split.

Artifacts: `c1_split_manifest.parquet`, `c1_split_manifest.csv`, and
`c1_split_manifest.summary.json`.

## C2: Leakage Audit

**GATE C2 passed for both strict channels, while the episode-heldout channel
was entirely near-duplicate replay.** Metrics are maximum MPNet instruction
cosine, minimum Levenshtein distance over grounded action tokens, and maximum
grounded-plan trigram Jaccard.

| Channel | Mean max cosine | Mean min edit | Exact plans | Exact trigrams | Full signatures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Episode-heldout | 0.9945 | 0.000 | 400/400 | 400/400 | 400/400 |
| Productivity | 0.3474 | 9.370 | 0/400 | 0/400 | 0/400 |
| Systematicity | 0.3508 | 9.255 | 0/400 | 0/400 | 0/400 |
| Published OOD memory | 0.6029 | 5.089 | 0/1,218 | 0/1,218 | 0/1,218 |

For both strict channels, maximum trigram Jaccard was exactly 0. The published
OOD arm also had no exact plan, trigram, or full-signature overlap among its
five retrieved demos. Its negative aggregate result therefore was not caused
by exact near-duplicate plan replay.

The audit sharply changes the interpretation of episode-heldout performance:
all 400 rows have an exact reference-plan and exact trigram match in allowed
memory. That channel measures near-duplicate instance transfer, not strict
composition.

Artifacts: `c2_leakage_audit.parquet`, `c2_leakage_audit.csv`, and
`c2_leakage_audit.summary.json`.

## C3: Arms and Predictions

**Neither numerical C3 prediction held; the near-duplicate diagnostic held for
episode-heldout, but the audited strict splits supplied no positive
re-anchoring result.** Prompt preflight and final artifact validation passed.
The result set has 1,200 unique `(channel, index)` rows, 400 per channel, zero
endpoint errors, zero strict full-signature leakage, and scorer seed equal to
the original ID index for every row.

| Channel | Skill tiers | Flat control k | Max token difference |
| --- | --- | --- | ---: |
| Episode-heldout | 397 grounded, 3 abstract | 397 at k=5, 3 at k=1 | 1.92% |
| Productivity | 211 none, 189 abstract | 211 at k=0, 189 at k=1 | 4.97% |
| Systematicity | 211 none, 189 abstract | 211 at k=0, 189 at k=1 | 5.00% |

The global token difference is at most 5.00%; 156 flat prompts required
token-level truncation. When a strict row has tier `none`, both memory arms
replay the baseline output. This occurs on 211/400 rows in each strict channel
and necessarily limits the measured difference between skill and flat memory.

### Accuracy

| Channel | Baseline | Skill | Flat |
| --- | ---: | ---: | ---: |
| Episode-heldout | 378/400 (94.50%) | 358/400 (89.50%) | 356/400 (89.00%) |
| Productivity | 374/400 (93.50%) | 374/400 (93.50%) | 374/400 (93.50%) |
| Systematicity | 373/400 (93.25%) | 373/400 (93.25%) | 373/400 (93.25%) |

The strict equalities are paired equalities, not merely equal aggregate
accuracies: baseline, skill, and flat have the same success/failure vector in
both strict channels. Thus every strict pair has zero discordant rows and exact
McNemar `p = 1.0`. This does not establish equivalence of the generated plans.

### Paired Tests

| Channel | Comparison | Right minus left | Left only | Right only | Exact McNemar p |
| --- | --- | ---: | ---: | ---: | ---: |
| Episode-heldout | Baseline vs skill | -5.00 pp | 23 | 3 | 0.0000879765 |
| Episode-heldout | Baseline vs flat | -5.50 pp | 24 | 2 | 0.0000104904 |
| Episode-heldout | Flat vs skill | +0.50 pp | 3 | 5 | 0.7265625 |
| Productivity | Baseline vs skill | 0.00 pp | 0 | 0 | 1.0 |
| Productivity | Baseline vs flat | 0.00 pp | 0 | 0 | 1.0 |
| Productivity | Flat vs skill | 0.00 pp | 0 | 0 | 1.0 |
| Systematicity | Baseline vs skill | 0.00 pp | 0 | 0 | 1.0 |
| Systematicity | Baseline vs flat | 0.00 pp | 0 | 0 | 1.0 |
| Systematicity | Flat vs skill | 0.00 pp | 0 | 0 | 1.0 |

Prediction 1 said every method would drop from episode-heldout to both strict
channels. It held only for baseline: 94.50% fell to 93.50% and 93.25%. Skill
rose from 89.50% to 93.50% and 93.25%, while flat rose from 89.00% to the same
values. The prediction therefore failed for skill and flat.

Prediction 2 said the skill-over-flat advantage would be larger under strict
composition. The advantage was +0.50 points episode-heldout and exactly 0.00
in each strict channel. It failed in both comparisons.

Prediction 3 was conditional: if ID-style gains relied on near-duplicates, the
claim would be re-anchored on audited strict splits. C2 confirmed exact
near-duplicate replay for every episode-heldout row, but C3 found no strict
memory advantage and both strict pools were degenerate. The correct re-anchor
is therefore a limitation, not a positive compositional result: these VIKI
experiments do not support a structured-memory compositional advantage.

Preflight artifacts: `c3_prompt_preflight.parquet`,
`c3_prompt_preflight.csv`, and `c3_prompt_preflight.summary.json`. Final
artifacts: `c3_results.jsonl`, `c3_results.parquet`, and
`c3_results.summary.json`.

## C4: PartNR Audit

**C4 is blocked by unavailable artifacts and was not performed.** The required
inputs are the exact PartNR memory subsets behind Table 1 and Figure 3, plus
the provenance needed to identify the single-constraint elementary episodes
for the requested strict Figure 3 row. Searches of this workspace and the user
home directory found the paper text and newly generated VIKI memory artifacts,
but not those PartNR banks or generation logs. No artifact URL was available.

The audit cannot be substituted with the PARTNR dataset or an independently
sampled memory bank: that would no longer test whether the published gains use
near-duplicate retrieval. Once the frozen banks are supplied, each PartNR test
row must be compared only with its actual allowed memory subset using the same
instruction-similarity, grounded-plan edit-distance, trigram-overlap, and
full-signature checks as C2. The Figure 3 rerun must then add a separately
labeled strict row whose memory contains only single-constraint elementary
episodes, with its audit attached.

## Conclusions

**The amendment's proposed VIKI improvement did not hold.** The experiments
support four narrower conclusions:

1. Most legal OOD regressions are real branch/object decision flips.
2. Those flips are not explained by all-present train-memory contamination;
   applicable train memory was already entirely some-absent.
3. Branch indexing and abstract fallback were unsafe in ID, so the mandatory
   gate correctly prevented an OOD efficacy claim.
4. VIKI episode-heldout transfer is exact near-duplicate replay, while the two
   literal strict splits remove leakage but collapse to observation-only
   memory.

The result is diagnostic rather than positive: memory structure can steer the
planner's decision, but this branch representation and train-only similarity
calibration do not identify when that steering is reliable.
