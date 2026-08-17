# VIKI-L2-OOD Memory-as-Skill Improvement Experiments

## Status

**HALTED at GATE V4.** The preregistered train-only validator passed only
`391 / 403 = 97.02%` of published baseline successes. The required tolerance was
99%, so the miss is `-1.98` percentage points and eight additional baseline
successes would have needed to pass. No pass-2 calls were made.

No new LLM inference calls were made in this experiment sequence. Two vLLM
servers began loading for V2 and V3 after their free gates passed, but both were
stopped before either endpoint received an evaluation request when GATE V4
failed.

The frozen VIKI-R checkout, checkpoints, train memory, published baseline logs,
and published memory-arm logs were not modified. New analysis and wrapper code
lives outside the VIKI-R checkout.

This benchmark split is static and single-robot. It evaluates only the
individual-skill branch under domain shift and directly addresses the paper's
"transfer beyond PartNR is unverified" limitation. It is not an evaluation of
cooperation memory or the decentralized cooperation method.

## Frozen Configuration

| Item | Value |
| --- | --- |
| VIKI-R code | `a0f13ed1ffe2cc509639fcd34d3f2ecbf4a2e5c5` |
| Dataset | `0160418e37518a99cb3f67d9c04521e651f29834` |
| 7B L2 checkpoint | `dd3b6a42aea5dfad42607bd538a68474e9b7f9c2` |
| Runtime | vLLM 0.8.4, Transformers 4.51.3, BF16 |
| Context / output budget | 4,096 / 2,000 tokens |
| Decoding | greedy, temperature 0 |
| Published memory settings | MPNet, top-k 5, threshold 0.3, routing budget 768 |
| Scorer seed | `run_seed + index`, with `run_seed = 0` in both arms |

## V0: Instrumentation And Log Extraction

**Both preregistered predictions held: fallback was exactly 0%, and there were
no context truncation events. GATE V0 passed.**

Outputs:

- `results/viki_memory_experiments/viki_ood_samples.parquet`
- `results/viki_memory_experiments/viki_ood_samples.csv`
- `results/viki_memory_experiments/viki_ood_samples.summary.json`

Implementation:

- `scripts/viki_memory_experiments.py`

The tidy table contains all 1,218 indices exactly once and all requested fields:

```text
index, ood_subset, baseline_success, memory_success,
routed_skill, injected_demo_ids, n_demos_injected,
retrieval_sim_max, retrieval_sim_mean, fallback_fired,
input_tokens_total, injected_tokens, truncation_flag,
plan_len_baseline, plan_len_memory, scorer_seed
```

### GATE V0

| Subset | N | Baseline | Memory | Fixes | Regressions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bowl missing | 409 | 13 (3.18%) | 47 (11.49%) | 38 | 4 |
| Plate missing | 418 | 387 (92.58%) | 326 (77.99%) | 17 | 78 |
| Both missing | 391 | 3 (0.77%) | 0 (0.00%) | 0 | 3 |
| **Overall** | **1,218** | **403 (33.09%)** | **373 (30.62%)** | **55** | **85** |

The exact two-sided McNemar p-value is `0.0139573341`. Both arms use the same
reconstructed scorer seed at every index. These values exactly reproduce all
published anchors.

### Fallback

Prediction: 0.00%.

Observed: `0 / 1,218 = 0.00%`. Every published sample received five demos. The
prediction held and confirms the gate-never-fires pathology for this run.

### Retrieval Similarity

These are distributions of the maximum instance-level cosine similarity among
the five injected demos.

| Subset | Median | p10 | Minimum |
| --- | ---: | ---: | ---: |
| Overall | 0.5904 | 0.5503 | 0.3821 |
| Bowl missing | 0.5971 | 0.5503 | 0.3821 |
| Plate missing | 0.5904 | 0.5503 | 0.3821 |
| Both missing | 0.5904 | 0.5503 | 0.3821 |

The recorded `skill_similarity` minimum was 0.4727. Exact VLM skill-name
matches are assigned similarity 1.0, so the configured 0.3 threshold primarily
constrains centroid fallback rather than individual retrieved demos.

### Context Headroom

The exact frozen Qwen processor from the released checkpoint was used with
Transformers 4.51.3. vLLM 0.8.4 uses the same `Qwen2_5_VLProcessor` and expands
image placeholders from `image_grid_thw`, so the reconstruction includes visual
tokens.

- Input budget after reserving 2,000 output tokens: 2,096
- Maximum memory-arm input: 1,775 tokens
- Minimum remaining headroom: 321 tokens
- Truncation events: 0

vLLM rejects `input_tokens + max_tokens > max_model_len`; it does not silently
truncate these chat requests. Context truncation is therefore not a confound in
the published regression.

## V1: Discordant-Pair Taxonomy

**The demo-anchoring/length-inflation mechanism did not explain most
regressions. GATE V1 passed, and the contradiction is the finding.**

Outputs:

- `results/viki_memory_experiments/regression_taxonomy.csv`
- `results/viki_memory_experiments/fix_taxonomy.csv`
- `results/viki_memory_experiments/viki_v1_taxonomy.summary.json`

Implementation:

- `scripts/viki_memory_v1.py`
- Official scorer tracing in `scripts/viki_memory_experiments.py`

The wrapper dynamically observes the official evaluator's error code and
reapplies the official reference-length rule without modifying the VIKI-R
checkout. It reproduced all 1,218 frozen task scores in each arm with zero
mismatches.

### GATE V1

The required first 50 baseline successes had zero grounding false positives.
As a stronger check, all remaining 353 baseline successes were also checked;
all 403 accepted plans had zero grounding false positives.

Grounding uses exact model targets against the official scorer's type-level
asset vocabulary. Numeric suffixes are never stripped from model output.
`Move` and `Place` position literals follow the official action signatures and
are not incorrectly treated as asset references.

### Regressions

| Primary category | Count |
| --- | ---: |
| Absent-object reference | 12 |
| Length-bound violation | 0 |
| Infeasible action | 0 |
| Legal but wrong / goal miss | 73 |
| **Total** | **85** |

Official failure reasons agree: 73 `FAILED_GOAL_CONSTRAINT` and 12
`NOT_FOUND_ENTITY`. There were no length-only failures and no dynamic
`ACTION_NOT_FEASIBLE` failures among the 85 regressions.

Regression counts by subset are 4 bowl-missing, 78 plate-missing, and 3
both-missing.

Mean `plan_len_memory - plan_len_baseline` among regressions:

| Subset | Mean | Median | Range |
| --- | ---: | ---: | ---: |
| Bowl missing | 0.000 | 0 | 0 to 0 |
| Plate missing | 0.346 | 0 | 0 to 3 |
| Both missing | -3.000 | -3 | -3 to -3 |

Across all paired rows, the corresponding means are only 0.037, 0.065, and
0.031. Demo-induced plan length is therefore not the mechanical source of the
published regression.

### Demo Anchoring

Anchoring is the fraction of injected demos manipulating at least one exact
asset type absent from the current official scorer scene.

Within the 403 baseline-success rows, regressions had a lower mean anchoring
fraction than retained successes:

- Regressions: 0.7647
- Retained successes: 0.8226
- Point-biserial correlation with memory failure: -0.0614
- Spearman correlation: -0.0704

Across all 1,218 rows the point-biserial correlation was -0.0512. The proposed
positive association was not observed. Absent demo assets are common in both
successful and unsuccessful rows, so this fraction alone does not identify the
harmful examples.

### Fixes And Retention Ceiling

Of the 55 fixes, 54 baseline plans were legal simulator plans that missed the
goal, and one contained an ungrounded entity.

**`n_static_fix = 1`.**

This is the preregistered V4 static-repair retention ceiling. A repair trigger
limited to illegal actions and ungrounded entities can recover at most one of
the 55 published fixes.

## V2: L2-ID Control

**The ID prediction was not tested because GATE V4 later halted all new model
calls. The free baseline gate passed.**

Existing frozen baseline output:

- `results/viki_official_7b_l2_id.jsonl`
- `results/viki_official_7b_l2_id.summary.json`

### GATE V2

The baseline has exact coverage of all 1,800 ID rows, zero endpoint errors, and
100% format compliance:

- Successes: 1,695 / 1,800
- `mean_task_score`: 94.1667%, reproducing 94.17%

Baseline robot-count breakdown:

| Active robots | N | Successes | Accuracy |
| --- | ---: | ---: | ---: |
| 1 | 876 | 869 | 99.20% |
| 2 | 878 | 826 | 94.08% |
| 3 | 46 | 0 | 0.00% |

The ID memory arm was not run, so no `viki_id_samples.parquet` or paired ID
delta exists. The prediction that ID is near-zero or positive remains untested.

## V3: Object-Availability Filter

**The preregistered filtering mechanism was contradicted before generation; the
disabled-equivalence gate passed, but the end-to-end prediction was not tested
because GATE V4 halted all model calls.**

Implementation:

- `habitat_llm/evaluation/viki_memory_replay.py`
- `scripts/viki_memory_v3_run.py`

### GATE V3

With filtering disabled, replay reproduced the V0 injected demo IDs exactly,
in the same order, on all 50 gate samples: `50 / 50 PASS`.

The filter follows the row's initial state at type level. It does not condition
on which object is missing from the table. An asset with a non-null `init_pos`
exists in the scene even when it is absent from the target location.

This distinction directly contradicts preregistered prediction 1: in a
plate-missing row, the plate generally exists elsewhere in `init_pos`, so a demo
manipulating a plate is not removed as absent from the scene. For example, V0
index 0 retained all five bowl/plate demos.

The zero-call filter statistics were:

| Subset | Demos dropped | Mean surviving k | Fallback |
| --- | ---: | ---: | ---: |
| Overall | 80.23% | 0.989 | 960 / 1,218 (78.82%) |
| Bowl missing | 77.60% | 1.120 | 313 / 409 (76.53%) |
| Plate missing | 81.39% | 0.931 | 335 / 418 (80.14%) |
| Both missing | 81.74% | 0.913 | 312 / 391 (79.80%) |

The most frequently dropped asset targets were banana, pear, bread, tomato,
apple, cabinet, and meat, not plate. This filter mostly removes out-of-domain
training assets and collapses the method to baseline on about four-fifths of
rows. No plan calls were made, so there is no
`viki_ood_filtered.parquet`, paired delta, or McNemar result.

## V4: Two-Pass Repair

**The preregistered positive predictions were not tested because the mandatory
monotonicity gate failed.**

Implementation:

- `habitat_llm/evaluation/viki_memory_repair.py`

The validator used exactly:

1. action names from the current official prompt APIs;
2. exact type-level entities from the current row's non-null `init_pos`;
3. a train-only cap of `ceil(1.5 * maximum train plan length)` within the V0
   routed skill.

It did not receive the current row's reference plan or reference length.

### GATE V4 Failure

| Field | Required | Observed | Miss |
| --- | ---: | ---: | ---: |
| Baseline-success validator pass rate | at least 99.00% | 97.02% | -1.98 pp |
| Baseline successes passing | at least 399 / 403 | 391 / 403 | -8 rows |
| Ideal target | 403 / 403 | 391 / 403 | -12 rows |

All 12 false triggers were otherwise valid five-step plans. Their V0 router had
selected `dog_check_environment`, whose train-only maximum plan length is 2 and
whose resulting cap is 3. The failing indices are:

```text
111, 126, 148, 161, 195, 203, 222, 289, 298, 306, 310, 380
```

Eleven are plate-missing rows and one is bowl-missing. This is a router/cap
interaction: a valid five-step substitution plan is rejected because the
abstract routed skill is too short, not because the plan is statically invalid.

Train-only caps for routed skills observed in OOD were:

| Routed skill | Cap |
| --- | ---: |
| `dog_check_environment` | 3 |
| `single_move_asset_to_target` | 8 |
| `ensure_all_fruits_on_table` | 12 |
| `sequential_pick_two_and_place` | 12 |
| `serve_bread_after_checking_cabinet` | 12 |

Across all rows, the failed validator would have triggered pass 2 on 27 / 1,218
(2.22%): 25 due to the routed-skill length cap and two due to an ungrounded
entity. Triggering on this set would replace 12 known baseline successes, so the
method is not non-inferior by construction.

No pass-2 calls were made. Changing the cap after observing these OOD failures
would be post-hoc tuning. Resuming V4 requires a separately preregistered
protocol amendment, for example removing the routed-skill cap or defining a
misroute-robust train-only cap before examining any new outputs.

## V5: Instruction-Similarity Sweep

**Prediction not tested.** No sweep calls were made because GATE V4 required a
global halt. There are no threshold-arm outputs.

## V6: Base-Model Control

**Prediction not tested.** The stock base-model control was not started because
GATE V4 required a global halt.

## V7: Substitution-Augmented Memory

Optional task not run. No synthetic demos were created.

## V8: Router Accuracy

Optional task not run. The V4 failure nevertheless supplies direct evidence
that some OOD routes are semantically unsuitable for downstream validation:
12 accepted five-step plans were routed to a skill whose train-only cap was 3.
This is not a substitute for the requested held-out train accuracy and manual
100-row OOD adjudication.

## Current Interpretation

The completed evidence does not support a simple "memory demos mention an
absent object, make plans longer, and therefore fail" account:

- context truncation: absent;
- threshold fallback: absent;
- length-bound regressions: zero;
- infeasible-action regressions: zero;
- absent-object regressions: 12 / 85;
- legal goal misses: 73 / 85;
- demo-anchoring correlation: weakly negative, not positive;
- static-fix ceiling: only one published fix;
- routed-skill cap: falsely rejects 12 baseline successes.

The strongest supported statement is narrower: unconditional demonstrations
change visual/entity selection enough to turn many previously correct
plate-missing plans into legal but goal-missing plans. A state-availability
filter removes many obviously unrelated training assets, but the specified
filter is not targeted to the observed legal goal misses and collapses to the
baseline on 78.82% of rows. End-to-end evidence for benefit remains unavailable
because the preregistered V4 safety gate failed before new inference.
