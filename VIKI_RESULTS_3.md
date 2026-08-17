# VIKI Amendment 2: Segment Memory on Stock Qwen

## Status

**GATE M0 and GATE B0 passed; B1 prediction 1 held only at the floor, prediction
2's literal inequality held but was vacuous, prediction 3 failed, and prediction
4 measured little recovery; GATE C-prime-0 failed and halted Track C-prime.**

The stock `Qwen/Qwen2.5-VL-7B-Instruct` backbone had high format compliance but
almost no official task success: 1/300 on ID, 0/400 on episode-heldout, and
0/1,218 on OOD. Segment memory changed those results to 5/300, 4/400, and
0/1,218. The ID and episode gains were not significant, and every episode gain
plus four of five ID fixes came from `dog_check_environment`.

The OOD comparison was exactly tied at zero task successes, while memory format
compliance was lower by 1.31 percentage points. Thus, Amendment 2 does not
establish that paper-style segment memory improves VIKI L2-OOD. It instead
shows that the stock backbone is below the task-competence regime measured by
the released VIKI-R checkpoint.

C-prime generated no model outputs. Its frozen systematicity construction
narrowed from 1,750 eligible multi-unit rows to zero covered rows, so the
preregistered gate stopped all three arms before generation. C4 remains blocked
on the original PartNR banks and logs.

This is still a static VIKI evaluation. It does not measure decentralized
cooperation or cooperation memory.

## Frozen Protocol

**The frozen-artifact, determinism, paired-seed, and no-post-result-tuning rules
held.** Generation used:

- model: `Qwen/Qwen2.5-VL-7B-Instruct`;
- revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`;
- vLLM 0.8.4, Transformers 4.51.3, BF16, context length 4,096, seed 0;
- greedy decoding at temperature 0 and at most 2,000 output tokens;
- official VIKI L2 scoring with dataset index as the seed, paired across arms;
- `all-mpnet-base-v2` retrieval, similarity threshold 0.3;
- at most two instances per skill and six instances total;
- prompt fitting by removing instances from the final active group until the
  4,096-token input-plus-output limit was met.

M1 first predicted an ordered subgoal sequence, matched each subgoal to an
abstract skill, retrieved grounded instances by context, and injected the
surviving groups in predicted order. Similarity distributions were diagnostic
only and did not alter the frozen threshold after results were observed.

Implementation and orchestration are in:

- `habitat_llm/evaluation/viki_stage1_memory.py`;
- `habitat_llm/evaluation/viki_segment_memory.py`;
- `scripts/viki_amendment2_pipeline.py`;
- `scripts/viki_amendment2_cprime.py`.

All generated artifacts are under
`results/viki_memory_experiments/amendment2/`.

## M0: Stage 1 Memory

**GATE M0 passed: all 7,196 rows were accounted for, every retained instance was
contiguous, every valid row reconstructed losslessly, and retrieval smoke passed
100/100.** The deterministic extraction call used the same stock model at
temperature 0 with a 768-token output limit. Prompts, responses, model revision,
and the prompt hash were persisted.

| Measure | Result |
| --- | ---: |
| Train rows | 7,196 |
| Valid source rows | 6,699 |
| Excluded invalid extractions | 497 |
| Grounded instances | 19,499 |
| Contiguous instances | 19,499 |
| Raw skill names | 116 |
| Merged abstract skills | 109 |
| Single-unit instances | 10,359 |
| Zero-unit instances | 780 |
| Retrieval smoke | 100/100 |

The 497 invalid extraction rows were excluded and recorded rather than repaired
after seeing downstream results. Retained instances had 0 to 4 parsed units,
mean 1.525, median 1, and p90 2. Instances per abstract skill ranged from 1 to
8,192, with mean 178.89 and median 7.

| Single-unit kind | Instances |
| --- | ---: |
| `fetch` | 3,887 |
| `relocate` | 4,612 |
| `state_change` | 1,860 |

Skill names were merged with deterministic single-link connected components on
MPNet name embeddings at cosine similarity 0.90; each component's medoid was
its canonical name. The scope was individual skills only and no cooperation
skills were fabricated.

## B0: Serving Smoke

**GATE B0 passed: oracle wiring passed 100/100, both format rates exceeded 90%,
and there were no endpoint errors.** The 100 ID rows were sampled from the
frozen A5 manifest with seed 20260814.

| N | Base task | Memory task | Delta | F->S | S->F | Discordant | Exact p | Base format | Memory format |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0/100 (0.00%) | 2/100 (2.00%) | +2.00 pp | 2 | 0 | 2 | 0.5 | 99/100 (99.00%) | 100/100 (100.00%) |

This gate checked wiring and format stability, not task competence.

## B1: Preregistered Runs

**Prediction 1 held only in the literal no-asymmetry sense; prediction 2's
non-negative inequality held but had no competence-region subset to test;
prediction 3 failed; prediction 4 found only 1.44% ID and 1.06% episode recovery
of the RL-versus-stock gap.**

Prediction accounting:

1. OOD bowl-missing and plate-missing base results were both 0%. This has no
   extreme asymmetry and both are far from the RL checkpoint's 92.58% and
   3.18%, but it is floor equality rather than balanced competence.
2. Memory minus base was non-negative overall on ID, episode-heldout, and OOD.
   However, base exceeded 50% in no reported subset, so the competence-region
   clause was vacuous and does not establish absence of competence harm.
3. Episode-heldout memory was 1.00%, not near the ceiling. This prediction
   failed decisively; the frozen RL checkpoint memory result was 89.50%.
4. On identical frozen rows, M-bank recovered 1.33 of the 92.33-point overall
   ID RL-versus-stock gap and 1.00 of the 94.50-point episode gap.

### Overall Paired Results

`Task` below is official `mean_task_score`; format is reported separately.

| Channel | N | Base task | Memory task | Delta | F->S | S->F | Discordant | Exact p | Base format | Memory format |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen ID | 300 | 1/300 (0.33%) | 5/300 (1.67%) | +1.33 pp | 5 | 1 | 6 | 0.21875 | 299/300 (99.67%) | 299/300 (99.67%) |
| Episode-heldout | 400 | 0/400 (0.00%) | 4/400 (1.00%) | +1.00 pp | 4 | 0 | 4 | 0.125 | 398/400 (99.50%) | 399/400 (99.75%) |
| OOD | 1,218 | 0/1,218 (0.00%) | 0/1,218 (0.00%) | +0.00 pp | 0 | 0 | 0 | 1 | 1,209/1,218 (99.26%) | 1,193/1,218 (97.95%) |

No B1 comparison reached p < 0.05. Exact p is 1 when there are no discordant
pairs.

### Robot-Count Subsets

**The robot-count breakdown found no gain outside one-robot rows, and none of
the one-robot gains was significant.**

| Channel | Robots | N | Base | Memory | Delta | F->S | S->F | Exact p | Base format | Memory format |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID | 1 | 144 | 0.69% | 3.47% | +2.78 pp | 5 | 1 | 0.21875 | 100.00% | 100.00% |
| ID | 2 | 147 | 0.00% | 0.00% | +0.00 pp | 0 | 0 | 1 | 100.00% | 99.32% |
| ID | 3 | 9 | 0.00% | 0.00% | +0.00 pp | 0 | 0 | 1 | 88.89% | 100.00% |
| Episode | 1 | 196 | 0.00% | 2.04% | +2.04 pp | 4 | 0 | 0.125 | 99.49% | 99.49% |
| Episode | 2 | 196 | 0.00% | 0.00% | +0.00 pp | 0 | 0 | 1 | 100.00% | 100.00% |
| Episode | 3 | 8 | 0.00% | 0.00% | +0.00 pp | 0 | 0 | 1 | 87.50% | 100.00% |
| OOD | 1 | 1,218 | 0.00% | 0.00% | +0.00 pp | 0 | 0 | 1 | 99.26% | 97.95% |

### ID Task Families and RL-Gap Recovery

**Only `dog_check_environment` and `serve_bread_from_counter` recovered any ID
RL-versus-stock gap; every other positive-gap task family recovered 0%.**

`Recovered` is `(stock memory - stock base) / (RL zero-shot - stock base)` on
the same rows. It is descriptive, not an additional significance test.

| Task | N | RL zero-shot | Stock base | Stock memory | Recovered | F->S | S->F | Exact p | Base fmt | Memory fmt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clear_table_with_two_robots_and_put_in_cabinet` | 30 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `cut_fruit_on_board` | 25 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 96.00% |
| `cut_two_fruits_on_board` | 22 | 40.91% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `dog_check_environment` | 8 | 100.00% | 12.50% | 50.00% | 42.86% | 4 | 1 | 0.375 | 100.00% | 100.00% |
| `dog_push_box_for_two_panda_transport` | 9 | 0.00% | 0.00% | 0.00% | n/a | 0 | 0 | 1 | 88.89% | 100.00% |
| `ensure_all_fruits_on_table` | 46 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `parallel_human_dual_asset_to_plate_or_bowl` | 19 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `sequential_pick_two_and_place` | 13 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `serve_bread_after_checking_cabinet` | 14 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `serve_bread_from_counter` | 16 | 100.00% | 0.00% | 6.25% | 6.25% | 1 | 0 | 1 | 100.00% | 100.00% |
| `set_plate_and_fork_on_table` | 20 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `single_move_asset_to_target` | 38 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `toast_bread_and_set_plate` | 15 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `wash_fruit_and_serve` | 25 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| **Overall** | **300** | **92.67%** | **0.33%** | **1.67%** | **1.44%** | **5** | **1** | **0.21875** | **99.67%** | **99.67%** |

### Episode Task Families and RL-Gap Recovery

**Only `dog_check_environment` recovered any episode RL-versus-stock gap; every
other positive-gap task family recovered 0%.**

| Task | N | RL zero-shot | Stock base | Stock memory | Recovered | F->S | S->F | Exact p | Base fmt | Memory fmt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clear_table_with_two_robots_and_put_in_cabinet` | 40 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `cut_fruit_on_board` | 44 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `cut_two_fruits_on_board` | 30 | 56.67% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `dog_check_environment` | 7 | 100.00% | 0.00% | 57.14% | 57.14% | 4 | 0 | 0.125 | 100.00% | 100.00% |
| `dog_push_box_for_two_panda_transport` | 8 | 0.00% | 0.00% | 0.00% | n/a | 0 | 0 | 1 | 87.50% | 100.00% |
| `ensure_all_fruits_on_table` | 64 | 98.44% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `parallel_human_dual_asset_to_plate_or_bowl` | 17 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `sequential_pick_two_and_place` | 21 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 95.24% | 95.24% |
| `serve_bread_after_checking_cabinet` | 19 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `serve_bread_from_counter` | 23 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `set_plate_and_fork_on_table` | 21 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `single_move_asset_to_target` | 44 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `toast_bread_and_set_plate` | 23 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| `wash_fruit_and_serve` | 39 | 100.00% | 0.00% | 0.00% | 0.00% | 0 | 0 | 1 | 100.00% | 100.00% |
| **Overall** | **400** | **94.50%** | **0.00%** | **1.00%** | **1.06%** | **4** | **0** | **0.125** | **99.50%** | **99.75%** |

### OOD Task Families

**Prediction 1's no-asymmetry pattern held, but all three OOD task families were
at the task-score floor in both arms.**

| Task | N | Base | Memory | Delta | F->S | S->F | Exact p | Base fmt | Memory fmt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bring_bowl_to_table_plate_already_there` | 409 | 0.00% | 0.00% | +0.00 pp | 0 | 0 | 1 | 100.00% | 97.07% |
| `bring_plate_and_bowl_to_table` | 391 | 0.00% | 0.00% | +0.00 pp | 0 | 0 | 1 | 97.95% | 98.47% |
| `bring_plate_to_table_bowl_already_there` | 418 | 0.00% | 0.00% | +0.00 pp | 0 | 0 | 1 | 99.76% | 98.33% |

### Routing Diagnostics

**Routing completed without endpoint errors; route parsing failed on only four
of the 700 ID and episode B1 rows and on no OOD rows.**

| Channel | Route parse errors | Mean retained instances | Instances removed by budget | Endpoint errors |
| --- | ---: | ---: | ---: | ---: |
| B0 | 1 | 1.92 | 275 | 0 |
| ID | 2 | 1.953 | 816 | 0 |
| Episode | 2 | 0.94 | 245 | 0 |
| OOD | 0 | 2.00 | 4,653 | 0 |

The large OOD removal count reflects the frozen 4,096-token fitting rule; it
was reported diagnostically and did not trigger threshold or k tuning.

## C-prime: Compositional Generalization

**GATE C-prime-0 failed, so all three compositional predictions are unmeasured
and no zero-shot, skill-memory, or flat-control generation was run.** The formal
failure artifact records `generation_halted: true`.

| Channel | Candidates before coverage | Covered candidates | Selected | Status |
| --- | ---: | ---: | ---: | --- |
| Instance | n/a | at least 400 | 400 | Ready before joint gate |
| Productivity | 1,750 | 722 | 400 | Ready before joint gate |
| Systematicity | 1,750 | 0 | 0 | Gate failure |

The productivity pool covered `fetch`, `relocate`, and `state_change` with
3,887, 4,612, and 1,860 single-unit instances. Under the frozen systematicity
rule, only 171 `state_change` single-unit instances survived; neither `fetch`
nor `relocate` had 30 usable instances. In particular, `fetch+relocate` was a
held-out pair in all 1,750 eligible multi-unit test rows. Applying the
preregistered covered-signature narrowing therefore left zero systematicity
rows.

Prediction accounting:

1. The single-unit versus multi-unit base comparison was not generated.
2. The strict-channel skill-versus-flat inequality was not generated.
3. The instance-level leakage audit was not reached after the gate failure, so
   zero full-signature coverage is not claimed as a measured result.

Changing held-out-pair semantics or admitting composite source rows after this
failure would be a new amendment, not completion of Amendment 2.

## C4: PartNR Audit

**C4 remains blocked and unmeasured.** The workspace does not contain the
frozen PartNR memory banks or generation logs behind Table 1 and Figure 3.
Reconstructing different subsets would not be the identical audit specified by
the amendment.

The required handoff is unchanged: place the original banks and logs in the
workspace, after which C4 can run without further design work.

## Validation

**The completed B1 artifacts passed independent consistency checks.** The OOD
audit recomputed all metrics from raw JSONL rather than calling the pipeline's
summary function:

- 1,218 rows, indices 0 through 1,217 exactly, no missing or duplicate rows;
- one run fingerprint:
  `46dd2314fe548a68034e3bc3c8d71a235a3e218d1966c00e37c4b25ac0d82bd0`;
- zero endpoint errors and zero route parse errors;
- 2,436 retained instances, mean 2.0, and 4,653 budget removals;
- raw success counts 0 versus 0 and format counts 1,209 versus 1,193;
- zero discordant pairs and independently computed exact p = 1;
- exact agreement for every overall, task, and robot-count summary field;
- exact row-and-value agreement between JSONL and the 1,218-row parquet.

M0, B0, ID, episode, OOD, and C-prime summaries are durable JSON artifacts.
No unmeasured C-prime or C4 result is inferred from a prepared candidate pool.
