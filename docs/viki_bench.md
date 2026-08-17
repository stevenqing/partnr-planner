# VIKI-Bench Evaluation

The adapter in `habitat_llm.evaluation.viki_bench` evaluates model responses
with the official scorer from a separate VIKI-R checkout. It passes through the
system prompt, user prompt, and image stored in each published Parquet row.

## Scope

- VIKI-L1 evaluates visual agent selection and maps directly to PARTNR's
  multi-agent allocation capability.
- VIKI-L2 evaluates executable multi-agent plans and is the primary comparison
  for PARTNR planners. The VIKI prompt defines its robot and action vocabulary.
- VIKI-L3 predicts two pixel-space trajectories. It is useful as a diagnostic
  for the underlying vision model, but does not measure PARTNR's symbolic
  planner directly.

The downloaded public test sets contain 2,043 L1, 1,800 L2, and 442 L3 samples.
L2 scoring is seeded per sample because its official simulator samples an
initial position when several are valid.

For paper-compatible reporting, use `mean_task_score`, not `mean_score`. The
former is exact set accuracy for L1 and feasible, reference-length-bounded plan
success for L2. The latter adds the training-oriented format reward with weights
0.1 and 0.9. L2 `test.parquet` is the ID split; `val.parquet` is the 1,218-sample
held-out OOD split.

All 1,218 released L2-OOD rows have exactly one active robot. Treat OOD as a
scene/task generalization metric, not a multi-agent generalization metric. Use
the L2-ID one-, two-, and three-agent breakdown to analyze coordination scaling.

## Wiring Check

Run one oracle sample from each level before evaluating a model:

```bash
for level in 1 2 3; do
  python -m habitat_llm.evaluation.viki_bench \
    --benchmark-root ../VIKI-R \
    --level "$level" \
    --provider oracle \
    --limit 1
done
```

Each summary should report `mean_score: 1.0`.

## OpenAI-Compatible Endpoint

Expose the method through a multimodal OpenAI-compatible server, then run:

```bash
python -m habitat_llm.evaluation.viki_bench \
  --benchmark-root ../VIKI-R \
  --level 2 \
  --provider endpoint \
  --base-url http://127.0.0.1:8000/v1 \
  --model YOUR_SERVED_MODEL \
  --workers 8 \
  --resume
```

The API key is read from `OPENAI_API_KEY` by default. An unauthenticated local
server can use any non-empty value or the adapter's `EMPTY` fallback.

## PARTNR Multimodal Model

Run the configured `MultiModalLlama` implementation directly:

```bash
python -m habitat_llm.evaluation.viki_bench \
  --benchmark-root ../VIKI-R \
  --level 2 \
  --provider partnr \
  --llm-name multimodal_llama \
  --engine /path/to/vision-model \
  --workers 1
```

The local provider is intentionally single-worker because it owns one model
instance. A text-only PARTNR planner is not a valid VIKI baseline because all
three levels require image input.

## Official VIKI-R Baseline Reproduction

VIKI-R publishes a separate checkpoint for each level and model size. Reproduce
these baselines independently of any PARTNR integration:

- `henggg/Qwen2.5VL-3B-Instruct-VIKI-R-1`
- `henggg/Qwen2.5VL-3B-Instruct-VIKI-R-2`
- `henggg/Qwen2.5VL-3B-Instruct-VIKI-R-3`
- `henggg/Qwen2.5VL-7B-Instruct-VIKI-R-1`
- `henggg/Qwen2.5VL-7B-Instruct-VIKI-R-2`
- `henggg/Qwen2.5VL-7B-Instruct-VIKI-R-3`

They are Qwen2.5-VL models, so they must be loaded through a Qwen-compatible
runtime such as vLLM rather than PARTNR's Mllama-specific `MultiModalLlama`.
Serve each checkpoint under a distinct model name and evaluate it only on its
matching level. These scores validate the benchmark setup; they are not scores
for a new PARTNR method.

Published checkpoint scores should always include the VIKI code, data, and
model revisions plus inference settings. The public checkpoints differ from
the snapshots used for the paper and therefore do not exactly reproduce the
reported values. Evaluating all six released checkpoints produced:

| Split / metric | Released 3B | Paper 3B | Released 7B | Paper 7B |
| --- | ---: | ---: | ---: | ---: |
| L1 test accuracy | 78.32% | 74.10% | 90.70% | 93.00% |
| L2 ID test accuracy | 95.61% | 93.61% | 94.17% | 95.22% |
| L2 OOD accuracy | 33.42% | 32.11% | 33.09% | 33.25% |
| L3 RMSE | 81.81 | 75.69 | 68.67 | 64.87 |
| L3 Hausdorff | 98.64 | 90.25 | 86.63 | 79.23 |
| L3 discrete Frechet | 105.56 | 103.65 | 92.79 | 89.36 |

L1 uses all 2,043 test rows, L2-ID all 1,800 test rows, L2-OOD all
1,218 validation rows, and L3 all 442 test rows. Lower is better for all L3
distances. The L3 release script samples with `temperature=0.1` and retries a
zero-scoring response up to three total attempts; the other reported runs use
greedy decoding.

The measured sanity baseline used:

- VIKI-R code `a0f13ed1ffe2cc509639fcd34d3f2ecbf4a2e5c5`
- VIKI-R dataset `0160418e37518a99cb3f67d9c04521e651f29834`
- 3B L1 checkpoint `527c94d50013c2aa433ba284dc3697eec28a0b6d`
- 3B L2 checkpoint `de059f06417757c6195e9a344cc6c25bb285f743`
- 3B L3 checkpoint `301afdd69a3f5e75d703209ad664d29cdacab98b`
- 7B L1 checkpoint `cc274aafa5b81e546128d6cfb8be0d02d29455a7`
- 7B L2 checkpoint `dd3b6a42aea5dfad42607bd538a68474e9b7f9c2`
- 7B L3 checkpoint `d1709bd1e52759f292bb97dc2eb3b8118da592a5`
- vLLM 0.8.4, Transformers 4.51.3, BF16, greedy decoding, 4,096-token model
  context, and 2,000 maximum generated tokens (except L3 sampling noted above)

For a fair method comparison, use the same published test rows, official
prompts, deterministic generation (`temperature=0`), maximum output length,
and scorer seed. Report L1 activation accuracy and L2 executable-plan success
as the main PARTNR-facing metrics. Report format compliance separately so a
planning error is not conflated with malformed output.

## Memory-as-Skill L2-OOD Adaptation

For a step-by-step Chinese description of the implementation, data boundary,
prompt flow, failed alternatives, and paired analysis, see
[VIKI L2-OOD Memory-as-Skill 实现说明](viki_memory_as_skill_zh.md).

The `memory-endpoint` provider adapts the individual-skill branch of
Memory-as-Skill to VIKI-L2. The full cooperation branch cannot be evaluated on
this split: every one of the 1,218 L2-OOD rows has one active robot and one
static image, so there is no alternating partner execution or observable
environment delta for effect-based partner inference.

The adapted pipeline uses only the 7,196 L2 training rows to build memory:

1. Group successful train plans into 14 abstract skills using their train task
  categories.
2. Ask the VLM to route the complete task instruction to one API-executable
  abstract skill. The router never receives validation ground truth or images;
  images remain available to the final plan generator for current-state
  grounding.
3. Within that skill, retrieve five unique train demonstrations by cosine
  similarity with `all-mpnet-base-v2`.
4. Filter train instances by active robot count and available action APIs, then
  inject the executable demonstrations before the current multimodal task.
5. Fall back to the unaugmented prompt below cosine similarity 0.3.

With the released 7B L2 checkpoint, greedy decoding, a 768-token skill-routing
budget, and the same 2,000-token plan budget as the baseline, the full paired
OOD evaluation produced:

| OOD task | Samples | 7B baseline | Memory-as-Skill | Delta |
| --- | ---: | ---: | ---: | ---: |
| Bowl missing, plate present | 409 | 3.18% | 11.49% | +8.31 pp |
| Plate missing, bowl present | 418 | 92.58% | 77.99% | -14.59 pp |
| Bowl and plate both missing | 391 | 0.77% | 0.00% | -0.77 pp |
| **All L2-OOD** | **1,218** | **33.09%** | **30.62%** | **-2.46 pp** |

The memory method changed 55 baseline failures into successes but regressed 85
baseline successes. The two-sided exact McNemar test gives `p=0.014`, so this
configuration is a statistically significant regression, not an improvement.
Both methods have 100% format compliance and zero endpoint errors. The result
also exposes the main transfer limitation: train memory helps the checkpoint's
weak bowl-missing cases, but demonstration conditioning disrupts its already
strong plate-missing behavior and does not solve two-object plans. Do not report
this static single-agent adaptation as the paper's decentralized cooperation
method.

Run the provider with the frozen experiment settings:

```bash
python -m habitat_llm.evaluation.viki_bench \
  --benchmark-root ../VIKI-R \
  --level 2 \
  --split val \
  --provider memory-endpoint \
  --base-url http://127.0.0.1:8000/v1 \
  --model viki-r-7b-l2-memory \
  --workers 4 \
  --temperature 0 \
  --max-tokens 2000 \
  --memory-top-k 5 \
  --memory-similarity-threshold 0.3 \
  --memory-prediction-max-tokens 768
```

## Evaluating the Full PARTNR Planner

The providers above evaluate a multimodal model as a direct VIKI policy. They do
not by themselves evaluate PARTNR's full Habitat planning loop. A
`CentralizedLLMPlanner` expects a populated `WorldGraph`, agent tools, and
environment feedback, while VIKI-Bench provides one static image.

Use three explicitly named experiment tracks:

1. **Direct VLM policy:** pass the official image and prompt directly to the
  model. This is the existing endpoint or local-provider path and is directly
  comparable to the published VIKI-R checkpoints.
2. **Oracle-state planner ablation:** construct a PARTNR world graph using only
  VIKI initial-state fields (`robots` and `init_pos`), never `time_steps`, goal
  constraints, or temporal constraints. This measures planning independently
  of visual perception and must be labeled as privileged-state input.
3. **Full method:** infer entities, locations, and robot identities from the
  image, populate a world graph, run the planner, and convert its tool plan to
  VIKI actions. This is the end-to-end result but requires an image-to-world-
  graph perception adapter that PARTNR does not currently provide for VIKI.

The oracle-state planner ablation is executable through a local
OpenAI-compatible endpoint. It builds a PARTNR `WorldGraph` from the current
row's `robots`/`init_pos` plus training-vocabulary locations explicitly named
in the task, and never exposes test `goal_constraints`,
`temporal_constraints`, or `time_steps` to the planner:

```bash
python -m habitat_llm.evaluation.viki_bench \
  --benchmark-root ../VIKI-R \
  --level 2 \
  --provider partnr-planner-oracle-state \
  --base-url http://127.0.0.1:8000/v1 \
  --model local-planner \
  --limit 1
```

Each result stores the PARTNR prompt trace, high-level actions, entity mapping,
and converted VIKI plan under `provider_metadata`. This track is privileged-state
input and must not be reported as the end-to-end visual PARTNR method.

A deterministic L2 action converter should expand PARTNR tools as follows:

| PARTNR tool | VIKI action sequence |
| --- | --- |
| `Navigate(target)` | `Move(target)` |
| `Pick(object)` | `Reach(object)`, then `Grasp(object)` |
| `Place(..., destination, ...)` | `Place(destination)` after navigation |
| `Open(container)` / `Close(container)` | same action and target |
| `Explore(target)` or object-state tools | `Interact(target)` when supported |

Preserve agent assignments, emit at most one action per robot per time step,
and compact independent actions into parallel steps. Reject any conversion that
uses an action outside the robot APIs embedded in the official L2 prompt.

Export one PARTNR trace per line to score planner output directly:

```json
{"index": 0, "agent_map": {"0": "R1"}, "entity_map": {"apple_0": "apple"}, "trace": [{"0": ["Navigate", "apple_0", ""]}], "available_actions": {"R1": ["Move", "Reach", "Grasp", "Place"]}}
```

`entity_map` is required because PARTNR uses instance names such as `apple_0`
while VIKI's simulator addresses type-level entities such as `apple`. Build this
mapping from the current sample's initial state; never infer it by blindly
stripping numeric suffixes.

```bash
python -m habitat_llm.evaluation.viki_bench \
  --benchmark-root ../VIKI-R \
  --level 2 \
  --provider partnr-traces \
  --predictions partnr_traces.jsonl
```

## Offline Predictions

To decouple generation from scoring, provide JSONL records containing an index
and the raw tagged response:

```json
{"index": 0, "response": "<think>...</think><answer>...</answer>"}
```

Score them with:

```bash
python -m habitat_llm.evaluation.viki_bench \
  --benchmark-root ../VIKI-R \
  --level 2 \
  --provider predictions \
  --predictions predictions.jsonl
```

The evaluator writes one JSONL record per sample and a sibling
`.summary.json`. `mean_score` is the official combined reward;
`mean_format_score` isolates output compliance; `mean_task_score` is L1 exact
activation accuracy, L2 executable-plan success, or L3 normalized trajectory
quality.

Results are flushed after every sample. Re-run with `--resume` to skip completed
indices and retry records that contain an endpoint error.
