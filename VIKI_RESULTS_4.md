# VIKI Amendment 3: Local 72B Window Round

## Status

This round is a post-F1 local-model amendment. The preregistered F1 pick selected Gemini-2.5-Flash after no open model cleared every threshold; the user then explicitly authorized Qwen2.5-VL-72B for F2. The original pick artifact remains unchanged.

The primary OOD prediction held. Zero-shot was 10/1218 and segment memory was 23/1218 (delta +1.07 pp, exact McNemar p=0.035082).

## Paired Results

| Regime | N | Zero-shot | Memory | Delta [paired bootstrap 95%] | F->S | S->F | Exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ID** | | | | | | | |
| Stock 7B full-bank segment | 300 | 1/300 (0.33%) | 10/300 (3.33%) | +3.00 pp [+1.00, +5.33] | 10 | 1 | 0.0117188 |
| Local 72B full-bank segment | 300 | 22/300 (7.33%) | 69/300 (23.00%) | +15.67 pp [+11.00, +20.33] | 54 | 7 | 4.32211e-10 |
| RL 7B full-bank segment | 300 | 278/300 (92.67%) | not estimated: raised-context smoke failed | - | - | - | - |
| **OOD** | | | | | | | |
| Stock 7B segment memory | 1218 | 0/1218 (0.00%) | 0/1218 (0.00%) | +0.00 pp [+0.00, +0.00] | 0 | 0 | 1 |
| Local 72B segment memory | 1218 | 10/1218 (0.82%) | 23/1218 (1.89%) | +1.07 pp [+0.16, +1.97] | 23 | 10 | 0.035082 |
| RL 7B legacy skill memory | 1218 | 403/1218 (33.09%) | 373/1218 (30.62%) | -2.46 pp [-4.35, -0.57] | 55 | 85 | 0.0139573 |

The three regimes are descriptive rather than treatment-controlled: the stock and local-72B arms use the Amendment 2 segment bank, while the RL checkpoint uses earlier branch/legacy skill memories and a different context-budget policy.

## Local 72B OOD Families

| Family | N | Zero-shot | Memory | Delta [95%] | F->S | S->F | Exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bring_bowl_to_table_plate_already_there | 409 | 0/409 (0.00%) | 10/409 (2.44%) | +2.44 pp [+0.98, +4.16] | 10 | 0 | 0.00195312 |
| bring_plate_and_bowl_to_table | 391 | 0/391 (0.00%) | 0/391 (0.00%) | +0.00 pp [+0.00, +0.00] | 0 | 0 | 1 |
| bring_plate_to_table_bowl_already_there | 418 | 10/418 (2.39%) | 13/418 (3.11%) | +0.72 pp [-1.44, +2.87] | 13 | 10 | 0.677639 |

## C-prime

Systematicity remains excluded as grammar-impossible. Instance and productivity each use 400 frozen rows; flat prompts are token-matched to skill-memory prompts within 5%.

| Channel | Arm | Success | Format |
| --- | --- | ---: | ---: |
| instance | zero_shot | 23/400 (5.75%) | 100.00% |
| instance | skill_memory | 28/400 (7.00%) | 96.75% |
| instance | flat_memory | 47/400 (11.75%) | 94.00% |
| productivity | zero_shot | 5/400 (1.25%) | 100.00% |
| productivity | skill_memory | 36/400 (9.00%) | 99.50% |
| productivity | flat_memory | 57/400 (14.25%) | 98.75% |

| Channel | Comparison | Delta [paired bootstrap 95%] | F->S | S->F | Exact p |
| --- | --- | ---: | ---: | ---: | ---: |
| instance | zero_shot_to_skill | +1.25 pp [-0.50, +3.25] | 10 | 5 | 0.301758 |
| instance | zero_shot_to_flat | +6.00 pp [+3.25, +8.75] | 28 | 4 | 1.93012e-05 |
| instance | flat_to_skill | -4.75 pp [-7.25, -2.25] | 4 | 23 | 0.000310749 |
| instance | flat token match | max 0.07%; prefix-trimmed 209 | | | |
| productivity | zero_shot_to_skill | +7.75 pp [+4.75, +10.75] | 35 | 4 | 3.35316e-07 |
| productivity | zero_shot_to_flat | +13.00 pp [+9.50, +16.50] | 55 | 3 | 2.25986e-13 |
| productivity | flat_to_skill | -5.25 pp [-9.50, -1.00] | 28 | 49 | 0.0220335 |
| productivity | flat token match | max 0.08%; prefix-trimmed 355 | | | |

## Diagnostics

On local-72B ID, zero-shot was 22/300 and memory was 31/300. No skill-memory instance was removed for the 16K budget. Detailed retained-instance and token-headroom distributions are stored in `f2_ood.summary.json`, `f2_id.summary.json`, and both C-prime summaries.

![Three measured regimes](results/viki_memory_experiments/amendment3/f2_three_regime_curve.png)

## Amendment 4: Flat-Pool Attribution

The final equal-pool segment-flat control matched skill memory: segment flat reached 46/400 and skill memory 36/400, a skill-minus-segment difference of -2.50 percentage points (paired 95% interval [-6.75, +1.50], exact McNemar p=0.288784). The active ingredient on VIKI is segment-granular content with pool breadth; the organization claim on VIKI is withdrawn and rests on PartNR, stated plainly.

### G0: Flat-Pool Leakage

The manifest fields establish that neither channel used all 6,699 M0 source train rows. The instance rows used task-restricted `allowed_instance_ids`, with `allowed_source_count` ranging from 170 to 1,141. Every productivity row used 10,359 `allowed_instance_ids` from 4,993 `allowed_source_indices`. The skill retriever and flat builder received the same allowed segment IDs, but the flat builder regrouped selected segments by `source_train_index` and injected whole source rows; those source rows were not necessarily single-unit.

The audit counts only source rows whose complete formatted segment blocks remained after token-level prefix trimming. Coverage uses the C2 definition: a contiguous ordered-unit subsequence covers the test row's full C0 signature.

| Channel | Rows with covering exemplar | Flat success / skill failure with cover | Injected multi-unit exemplars | Maximum grounded-plan trigram overlap |
| --- | ---: | ---: | ---: | ---: |
| instance | 89/400 (22.25%) | 18/23 (78.26%) | 198 | 1.00 |
| productivity | 143/400 (35.75%) | 38/49 (77.55%) | 245 | 1.00 |

The preregistered G0 branch therefore declared productivity confounded and triggered G2. Instance triggered no rerun by design; its original flat result is a memorization ceiling under permitted near-duplicate and whole-answer replay.

### G1: Route Quality

Route alignment projects C0 onto primitive `fetch`, `relocate`, and `state_change` units, excludes derived composite units to avoid double counting, and leaves navigation-only subgoals unscored. Strict route correctness requires exact primitive-kind multiset and order agreement plus a retrieval hit for every scorable predicted subgoal.

| Channel | Strict route-correct | Multiset agreement | Order agreement | Mean multiset F1 | Mean order LCS | Mean retrieval hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| instance | 1/400 | 22.75% | 11.25% | 0.711 | 0.674 | 11.96% |
| productivity | 0/400 | 29.25% | 22.50% | 0.718 | 0.777 | 19.90% |

The strict preregistered route-correct comparison is not estimable at useful sample size: productivity has no strict-correct rows, while the sole instance strict-correct row was solved by both skill and flat. On route-incorrect rows, skill trailed the original flat by 4.76 points in instance (27/399 versus 46/399) and 5.25 points in productivity (36/400 versus 57/400). These strata do not distinguish routing-stage loss from composed-segment-format loss at this model scale.

For OOD, all 23 fixes and 10 regressions were strict-route-incorrect. Mean per-subgoal retrieval hit was 89.57% for fixes and 75.00% for regressions; mean primitive multiset F1 was 0.523 and 0.400, respectively. This is descriptive evidence that retrieval alignment was better on fixes, not a route-correct causal contrast.

### G2: Clean Flat

The primary clean pool contained the 171 M0 source rows whose C0 full signature is single-unit, represented by 319 segment instances. C0 source row 6075 was single-unit but absent from M0 and therefore not injectable. The optional original-pool-minus-covering arm was not run.

Preflight passed all 400 rows with zero symmetric drops, zero covering exemplars, 100% single-unit sources, and maximum token mismatch 0.0754% against the unchanged per-row 5% gate. All 400 zero-shot and skill prompts matched their original hashes. The rerun made exactly 400 generation calls, reused all zero-shot and skill outputs unchanged, discarded all 400 original flat outputs, and recorded a new run fingerprint.

| Arm | Success | Format |
| --- | ---: | ---: |
| zero-shot (reused) | 5/400 (1.25%) | 100.00% |
| skill memory (reused) | 36/400 (9.00%) | 99.50% |
| clean flat | 4/400 (1.00%) | 98.75% |

| Comparison | Paired delta | Left only | Right only | Exact p |
| --- | ---: | ---: | ---: | ---: |
| clean flat to skill | skill +8.00 pp | 3 | 35 | 6.67787e-08 |
| zero-shot to clean flat | clean flat -0.25 pp | 4 | 3 | 1.0 |
| zero-shot to skill | skill +7.75 pp | 4 | 35 | 3.35316e-07 |

The source-row-restricted clean control closes and reverses the original gap. On this intermediate control, skill memory beats clean flat by 8.00 points. Amendment 4.1 tests whether that difference is organization or the much broader segment pool.

### Amendment 4.1 / G2b: Equal-Pool Segment Flat

G2b is the final VIKI control. Per row, segment flat uses the exact 10,359 `allowed_instance_ids` supplied to the skill arm, representing 4,993 source rows. All candidate segments are single-unit and none covers the test full signature under C2. Retrieval is instruction-only MPNet cosine over segment contexts, with no subgoal prediction, skill matching, or grouping. Each segment uses the exact singleton skill-rendered block after its grouping header is removed.

The zero-call pool gate passed on all 400 manifest rows with one canonical pool hash. The full 10,359-segment renderer had zero template mismatches. On 20 deterministic sampled rows, 120/120 top-ranked pre-trim segments were byte-identical; after token trimming, all 20 final prompts were exact byte prefixes of the headerless segment rendering, with 35/35 complete segment blocks byte-identical and at most one partial token-trim suffix per row.

Preflight passed all 400 rows with zero symmetric drops, zero covering segments, 100% single-unit segments, and maximum token mismatch 0.0750% against the unchanged per-row 5% gate. All 400 zero-shot, clean-flat, and skill prompts matched their original hashes. The run made exactly 400 new calls, reused those three prior outputs unchanged, added only segment flat, and recorded a new fingerprint.

| Arm | Success | Format |
| --- | ---: | ---: |
| zero-shot (reused) | 5/400 (1.25%) | 100.00% |
| clean flat (reused) | 4/400 (1.00%) | 98.75% |
| segment flat | 46/400 (11.50%) | 96.00% |
| skill memory (reused) | 36/400 (9.00%) | 99.50% |

| Comparison | Paired delta [95%] | Left only | Right only | Exact p |
| --- | ---: | ---: | ---: | ---: |
| zero-shot to clean flat | -0.25 pp [-1.50, +1.00] | 4 | 3 | 1.0 |
| zero-shot to segment flat | +10.25 pp [+7.25, +13.50] | 3 | 44 | 2.46473e-10 |
| zero-shot to skill | +7.75 pp [+4.75, +10.75] | 4 | 35 | 3.35316e-07 |
| clean flat to segment flat | +10.50 pp [+7.50, +13.75] | 2 | 44 | 3.07523e-11 |
| clean flat to skill | +8.00 pp [+5.25, +11.00] | 3 | 35 | 6.67787e-08 |
| segment flat to skill | -2.50 pp [-6.75, +1.50] | 41 | 31 | 0.288784 |

The decomposition is therefore content-and-pool +10.25 points for segment flat over zero-shot, while organization is -2.50 points for skill over segment flat and is not significant. Segment flat statistically matches skill; the VIKI organization claim is withdrawn. The positive VIKI result is that segment-granular memory content with broad retrieval materially improves the local 72B model.

Organization here means the skill arm's routing, grouping, and ordering jointly, G2b does not separate those three, and routing is part of the structure being claimed. A further routed-but-ungrouped arm is possible at another 400 calls and is not required for the paper's claim.

The three-regime curve remains descriptive rather than treatment-controlled: memory variants and context-budget policies differ across regimes. VIKI generation is now closed under Amendment 4.1; no further VIKI arm is authorized. C4 remains blocked on the original PartNR memory banks and generation logs, and that user-side handoff is unchanged. The next VIKI deliverable is the chapter draft.

## Amendment 5: Cross-Backbone Deployment

Amendment 5 superseded the Amendment 4.1 generation closure once, solely for frozen-H0 deployment and cross-model replication. H0 uses the full 19,499-segment M0 bank, instruction-only MPNet cosine retrieval, top 6 segments with bare fallback below 0.3, headerless singleton rendering, greedy decoding, 2,000 output tokens, and at least 16,384 tokens of context. No per-backbone tuning is allowed.

| Backbone | Zero-shot | Composed skill | Segment | Segment minus zero-shot [95%] | Exact p | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-VL-72B | 10/1218 (0.82%) | 23/1218 (1.89%) | 83/1218 (6.81%) | +5.99 pp [+4.52, +7.47] | 7.08891e-16 | PASS |
| Qwen2.5-VL-7B stock | 0/1218 (0.00%) | - | 0/1218 (0.00%) | +0.00 pp [+0.00, +0.00] | 1.0 | PASS |
| Qwen3-VL-30B-A3B | 2/1218 (0.16%) | - | 6/1218 (0.49%) | +0.33 pp [-0.08, +0.82] | 0.2890625 | **FORMAT GATE FAIL** |
| GPT-4o-2024-08-06, optional | 137/1218 (11.25%) | - | 19/1218 (1.56%) | -9.69 pp [-11.58, -7.88] | 3.01707e-25 | PASS |

For Qwen2.5-VL-72B, zero-shot and composed outputs were reused unchanged from the certified source run; only the segment arm required 1,218 new calls. Segment also exceeded composed skill by 4.93 points (paired exact p=2.71882e-11). The run completed with zero endpoint errors and fingerprint `723e57892c0ce7b3f8fe5ebda6bde5e09ef9c6adc55573ebb3eea49c8987f80b`.

For stock Qwen2.5-VL-7B, zero-shot was reused unchanged and only segment was generated. Both arms scored zero; format compliance was 99.26% for zero-shot and 99.43% for segment. The run completed with zero endpoint errors and fingerprint `ea7a3f655454e302ed90094a2f0a484cda7bbb885ee80c6bd731cbe4323f0bd3`.

Qwen3 generated both arms, making exactly 2,436 calls. All 1,218 rows are present exactly once, both arms are present on every row, endpoint errors are zero, and the single run fingerprint is `a98b4b14021c1dfea01d8c1e4e8b2784f5f533675e07b819014b73ed22b5e68f`. The live runtime identity matched the frozen preflight: vLLM 0.11.2, `Qwen/Qwen3-VL-30B-A3B-Instruct`, and 16,384-token context.

The Qwen3 completion gate failed for a specific protocol reason rather than artifact corruption. Every response began with `<reasoning>` instead of the scorer-required `<think>` tag; `<think>` occurred in 0/1,218 responses in both arms, so official format compliance was 0% in both arms. Zero-shot accuracy remained consistent with the preregistered 0/200 probe interval [0.00%, 1.83%], but the frozen 90% format threshold failed. No settings were changed and no calls were repeated.

Independent certification re-scored all 2,436 saved Qwen3 responses with the official VIKI-L2 scorer and original-row-index seeds. Stored and recomputed `score`, `format_score`, and `task_score` had zero mismatches; all family summaries, the paired interval, and exact McNemar result matched. The JSONL SHA256 is `b1afc0741fcb8fc77b33c4e8474c3ea02d9419b3199ca260bef6b4cf48562863`, and all H0, manifest, preflight, runtime, and run-fingerprint bindings passed.

### Qwen3 Offline Format Repair

The saved responses were re-evaluated offline under two deterministic post-hoc adapters. Neither adapter made model or API calls, and neither changed a task outcome. The raw artifacts remain unchanged and remain the primary preregistered result.

| Offline policy | Zero-shot format | Segment format | Zero-shot task | Segment task | Task-score changes | Format gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| tag-only | 878/1218 (72.09%) | 1079/1218 (88.59%) | 2/1218 | 6/1218 | 0 | FAIL |
| canonical-null | 1218/1218 (100.00%) | 1218/1218 (100.00%) | 2/1218 | 6/1218 | 0 | PASS |

Tag-only replaces the first `<reasoning>` / `</reasoning>` pair with `<think>` / `</think>` and leaves every other byte, including the answer, unchanged. It does not clear the gate because incomplete or truncated envelopes remain. Canonical-null preserves each complete, parseable generated answer exactly and wraps it in the required envelope; when no complete parseable answer exists, it emits the semantically empty plan `[]`. It preserved 1,029 zero-shot answers and 1,164 segment answers, with null fallback on 189 and 54 responses respectively.

Under canonical-null, the completion gate passes with 100% format compliance and unchanged zero-shot probe consistency. Accuracy, paired delta (+0.33 points), paired interval [-0.08, +0.82], and exact McNemar p=0.2890625 are unchanged. This establishes that deployment-format compatibility can be repaired entirely offline, but the repaired run is explicitly `post_hoc=true` and `primary_inference_eligible=false`; it does not convert the nonsignificant Qwen3 task result into positive evidence.

Optional GPT-4o was run through CloudGPT deployment `gpt-4o-20240806` with Azure CLI cached authentication and no static token. All 1,218 pairs completed in 2,436 calls with zero endpoint errors. Its zero-shot score, 137/1,218, passed the preregistered replication interval of 102-143 successes; both arms had 99.92% format compliance. Segment memory caused a large, significant OOD loss of 9.69 points rather than a gain. The run fingerprint is `6e3ba0d1e4820c07e8b1163d2ce78e73ae89d9c5f7d525acc2dbb3bad20c4cba`.

CloudGPT authentication is operational, but Gemini-2.5-Flash is unavailable on the endpoint. The direct deployment name and four bounded official version aliases all returned `DeploymentNotFound`; the models-list route is unsupported with HTTP 404. Gemini OOD+ID was therefore not started. Amendment 5 is not yet the permanent VIKI closure.

## Amendment 5.1: ID Deployment Column

Amendment 5.1 replaces the main table's prior ID memory variants with the recommended full-bank segment variant on the exact frozen 300-row A5 slice. The ID manifest uses seed 20260814 and has SHA256 `d9ebe66966003bbd5776cac4defc947718a6b136bf11210e526e6a9dba58f580`. The generated deployment manifest contains the same 300 unique indices, the full 19,499-segment bank, zero fallback rows, H0 SHA256 `69912ff9d85a7c2690018fbd3792787fbff43c774c15ec3a7288cabd63713090`, and manifest SHA256 `d2ed9765f050309879287de13b33f371df4ae71e932b6974f4cd06e2d59d1808`.

| Backbone | Zero-shot | Segment | Delta [paired bootstrap 95%] | F->S | S->F | Exact p | Backbone gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-VL-72B | 22/300 (7.33%) | 69/300 (23.00%) | +15.67 pp [+11.00, +20.33] | 54 | 7 | 4.32211e-10 | **FAIL: segment format 81.67%** |
| Qwen2.5-VL-7B stock | 1/300 (0.33%) | 10/300 (3.33%) | +3.00 pp [+1.00, +5.33] | 10 | 1 | 0.0117188 | PASS |
| VIKI-R RL 7B | 278/300 (92.67%) | not run | not estimated | - | - | - | **FAIL: byte smoke** |
| Qwen3-VL-30B-A3B, optional | 13/300 (4.33%) | 71/300 (23.67%) | +19.33 pp [+14.67, +24.00] | 61 | 3 | 4.74284e-15 | PASS under preregistered format exemption |
| GPT-4o-2024-08-06, optional | 54/300 (18.00%) | 70/300 (23.33%) | +5.33 pp [+0.33, +10.33] | 39 | 23 | 0.0558972 | **FAIL: segment format 72.00%** |

The 72B zero-shot arm was reused byte-for-byte from certified `f2_id`; all source token counts and current official prompt hashes matched. The 300-call segment arm produced a much larger task gain than the preregistered directional threshold of +3 points, but only 245/300 responses passed the official format scorer. The backbone therefore fails its preregistered 90% format gate. Its task result is reported as landed but is not promoted to a gate-passing deployment result, and no repair or rerun was made.

The stock 7B zero-shot arm was reused byte-for-byte from certified B1. B1 used the same official `bench.get_messages(sample)` path, but its older serving stack reported different usage-token counts; prompt provenance is therefore bound by the B1 runner, official bench, test parquet, source-result SHA, and current prompt hashes, while legacy token accounting is diagnostic. Contrary to the preregistered floor prediction, segment improved by 3.00 points and passed both the exact test and format gate.

The RL checkpoint ran on its original vLLM 0.8.4 graph-mode stack with bfloat16, 0.65 GPU utilization, four maximum sequences, and the certified model name. The only sanctioned change was `max_model_len=16384`. The deterministic 20-row smoke halted on its first index, 46: source response SHA256 `e59708e9499a5a14f8ba186c48fa20d4b97032c6a3c6277ace5a727125b25d3a` differed from observed SHA256 `943497f61f4f26175e096cba010b309681d6816c59b5bd5f8d4a530334573878`. Exactly one smoke call and zero segment calls were made. The predicted specialization harm is therefore not measured under the recommended segment variant.

Optional Qwen3 generated both raw arms in exactly 600 calls. Raw official format compliance was 0% in both arms because the model used its incompatible reasoning envelope, so the preregistered Qwen3 exemption is displayed rather than hidden. The raw official task result strongly falsified the predicted floor: segment gained 19.33 points with exact p=4.74284e-15. The post-hoc canonical-null appendix preserved 281 zero-shot and 283 segment answers, used empty-plan fallback for 19 and 17 unrecoverable responses, reached 100% format in both arms, and changed zero task scores. It made zero model calls and remains `primary_inference_eligible=false`.

Optional GPT-4o generated both ID arms in exactly 600 CloudGPT calls. The raw task result rose by 5.33 points, with paired-bootstrap interval [+0.33, +10.33], but exact McNemar p=0.0558972 did not cross 0.05. Zero-shot format was 296/300 (98.67%); segment format was only 216/300 (72.00%), so the preregistered per-arm 90% gate failed. The result is preserved raw with no repair or rerun.

Predictions were recorded before the local runs. The 72B directional task prediction was confirmed but its backbone gate failed; the stock-floor and Qwen3-floor predictions were falsified; the RL sign prediction was not tested because its mandatory smoke failed. Across 5.1, 1,201 local and 600 CloudGPT generation calls were made, for 1,801 total.

ID-with-memory measures deployment effect, not compositional ability, since the bank contains near-duplicate in-distribution episodes, and attribution remains with the C-prime controls.

Amendment 5.1 is complete with backbone gate failures, including the optional GPT-4o fold-in. Gemini's OOD+ID pair is blocked on a confirmed CloudGPT deployment alias or access. Permanent closure is not reinstated and the chapter draft is not yet authorized.

## Amendment 6: Deployment Baselines

Amendment 6 is preregistered and supersedes generation closure once for its complete fixed scope. It adds trajectory-level RAG and a VIKI-native explicit counterpart/state-reasoning ToM port on Qwen2.5-VL-72B, first on all 1,218 OOD rows and then on the frozen 300-row ID slice. Zero-shot and segment outputs are reused with prompt, response, source-artifact, and run-fingerprint verification. Required new generation is 3,036 calls. GPT-4o is an optional 3,036-call tier only after all required 72B arms complete; Gemini remains excluded because it is not deployed on the available CloudGPT endpoint.

Trajectory RAG retrieves from all 6,699 valid M0 source rows using instruction-only MPNet cosine. Each retrieval unit is a complete source row rendered as the exact source-order concatenation of its headerless segment blocks. Ranked rows are extended and prefix-trimmed to the frozen segment arm's per-row input-token budget within 5%; bare fallback remains fixed at cosine below 0.3. The offline retrieval cache is complete and certified at 6,699 rows, 19,499 segments, and embedding shape 6,699 by 768. Full token-band, zero-symmetric-drop, and 20-row rendering gates await the certified 72B service.

The repository contains the paper-level PartNR ToM definition and reported comparator numbers, but no runnable ToM module, exact prompt, config, or result artifact. The frozen VIKI port therefore does not claim exact implementation reproduction. Its filed template explicitly reasons over active robots, capabilities, shared visual scene state, and complementary assignments before emitting the unchanged VIKI-L2 executable-plan format; single-robot rows reduce to explicit capability/state reasoning. OOD contains 1,218 single-robot rows. The frozen ID slice contains 144 one-robot, 147 two-robot, and 9 three-robot rows, so the ToM claim is carried by the ID robot-count breakdown.

Status is `WAITING_FOR_72B_CAPACITY`, with zero Amendment 6 generation calls made. The certified endpoint on port 8050 is stopped, and all eight A100 GPUs are currently occupied by an active multi-GPU training job; the service was not co-located and no user process was interrupted. Once four unshared GPUs are available, execution order is trajectory RAG OOD, ToM OOD, trajectory RAG ID, ToM ID, then the optional GPT-4o tier. The four-arm analysis is already fail-closed and will produce zero-shot / trajectory-RAG / ToM / segment tables, all six exact McNemar comparisons with paired bootstrap intervals, OOD family breakdowns, ID robot-count breakdowns, and per-arm input, injected, and generated token counts.

## Artifacts

- `results/viki_memory_experiments/amendment3/f2_local_override.json`
- `results/viki_memory_experiments/amendment3/f2_final_results.json`
- `results/viki_memory_experiments/amendment3/f2_final_tables.csv`
- `results/viki_memory_experiments/amendment3/f2_three_regime_curve.png`
- `results/viki_memory_experiments/amendment4/audit_summary.json`
- `results/viki_memory_experiments/amendment4/g0_flat_prompt_rows.parquet`
- `results/viki_memory_experiments/amendment4/g0_flat_prompt_exemplars.parquet`
- `results/viki_memory_experiments/amendment4/g1_cprime_route_rows.parquet`
- `results/viki_memory_experiments/amendment4/g1_ood_route_rows.parquet`
- `results/viki_memory_experiments/amendment4/g2_clean_flat.preflight.summary.json`
- `results/viki_memory_experiments/amendment4/g2_clean_flat.summary.json`
- `results/viki_memory_experiments/amendment4/g2_clean_flat.parquet`
- `results/viki_memory_experiments/amendment4/g2b_gates.summary.json`
- `results/viki_memory_experiments/amendment4/g2b_posttrim_render_gate.summary.json`
- `results/viki_memory_experiments/amendment4/g2b_segment_flat.preflight.summary.json`
- `results/viki_memory_experiments/amendment4/g2b_segment_flat.summary.json`
- `results/viki_memory_experiments/amendment4/g2b_segment_flat.parquet`
- `results/viki_memory_experiments/amendment4/VIKI_GENERATION_CLOSED.json`
- `results/viki_memory_experiments/amendment5/h0_frozen.json`
- `results/viki_memory_experiments/amendment5/deployment_manifest.summary.json`
- `results/viki_memory_experiments/amendment5/qwen2_5_vl_72b.summary.json`
- `results/viki_memory_experiments/amendment5/qwen2_5_vl_7b_stock.summary.json`
- `results/viki_memory_experiments/amendment5/qwen3_vl_30b_a3b.summary.json`
- `results/viki_memory_experiments/amendment5/qwen3_vl_30b_a3b.jsonl`
- `results/viki_memory_experiments/amendment5/qwen3_vl_30b_a3b.offline_tag_only.summary.json`
- `results/viki_memory_experiments/amendment5/qwen3_vl_30b_a3b.offline_tag_only.jsonl`
- `results/viki_memory_experiments/amendment5/qwen3_vl_30b_a3b.offline_format_repair.summary.json`
- `results/viki_memory_experiments/amendment5/qwen3_vl_30b_a3b.offline_format_repair.jsonl`
- `results/viki_memory_experiments/amendment5/gpt_4o_optional.summary.json`
- `results/viki_memory_experiments/amendment5/gpt_4o_optional.jsonl`
- `results/viki_memory_experiments/amendment5/cloudgpt_route_probe.json`
- `results/viki_memory_experiments/amendment5_1/preregistration.json`
- `results/viki_memory_experiments/amendment5_1/id_deployment_manifest.summary.json`
- `results/viki_memory_experiments/amendment5_1/qwen2_5_vl_72b.summary.json`
- `results/viki_memory_experiments/amendment5_1/qwen2_5_vl_7b_stock.summary.json`
- `results/viki_memory_experiments/amendment5_1/rl_7b.smoke.summary.json`
- `results/viki_memory_experiments/amendment5_1/qwen3_vl_30b_a3b.summary.json`
- `results/viki_memory_experiments/amendment5_1/qwen3_vl_30b_a3b.offline_format_repair.summary.json`
- `results/viki_memory_experiments/amendment5_1/gpt_4o_optional.summary.json`
- `results/viki_memory_experiments/amendment5_1/gpt_4o_optional.jsonl`
- `results/viki_memory_experiments/amendment5_1/final_summary.json`
- `results/viki_memory_experiments/amendment6/preregistration.json`
- `results/viki_memory_experiments/amendment6/closure_supersession.json`
- `results/viki_memory_experiments/amendment6/trajectory_row_embeddings.summary.json`
- `scripts/viki_amendment6.py`
