# Deviations from SPEC.md

> 注意：2026-09-22 15:44 Part A 的 agent 写入本文件时，可能覆盖了 Part B 已写入的内容（写入前未读到）。Part B 的偏差请重新追加到下面的「Part B」一节。Part A 的完整副本在 `A/deviations_A.md`。

# Part A

Written before each affected run started. The runs continued under user instruction of 2026-09-22 (run at full speed, no approval wait; record deviations, take the closest-to-SPEC route).

## A0

- A0-1. Figure 3 is `fig:viki_rq2` (third figure: `fig:workflow`, `fig:viki_main`, `fig:viki_rq2`). No plotting script for `figs/viki_rq2_ablation.pdf` is on disk. Its provenance is taken from the registry `results/paper_viki_iclr2027/` (`cells.json`, `rows/rq2/full/<model>/<split>.jsonl`, `SHA256SUMS.txt`, `rq2_summary.csv`). The byte-identity check compares the first-turn answer of every row (`raw` in the amendment11 cell vs `raw_output` in the registry row) and checks the registry file against `SHA256SUMS.txt`.
- A0-2. The archived first answer is stored truncated to its last 3,000 characters (`record["raw"] = text[-3000:]` in the runner). That stored text is what every published cell was scored from, so it is the frozen first-turn output here too.

## A1 mining

- A1-1. `our_method/skill_memory_v2/build.py` (the builder of the 19-operator reference library) is not used as is. It caps each family at 250 episodes, deduplicates by (effect, body, preconditions), keeps non-asset place names literal, and mines a `repair` kind. `scripts/iclr_two_exp/A/mine_replay.py` keeps its replay and segmentation functions (`induction.replay`, `_segment_start`, `_runs_alone`, `_bind`) and changes exactly those four points.
- A1-2. Step 2 segmentation is per actor, as in the builder: when a judged predicate (goal or temporal stage) becomes true, the segment is the completing robot's own actions since its previous completion. If those actions alone do not reproduce the predicate in the simulator, the whole window becomes one coordination candidate with one role per robot (`?r0`, `?r1`, ...).
- A1-3. Preconditions are the planner's fact vocabulary (`state_facts`: subject/target sealed, in container, on agent) at segment start. The merged operator keeps the facts that held with the same value in every supporting occurrence. SPEC's "predicates read by the segment" is approximated by these subject/target facts.
- A1-4. Lifting: subject `?x`, target `?y`, every other asset or place `?z1, ?z2, ...`. A place that is not an asset gets an empty type. A robot named as a target inside a single-robot body would become `?agent:<name>` as in the builder, and the planner cannot bind it. No mined operator had one.
- A1-5. Support counts distinct episodes. All 3,598 episodes replayed OK.

## A1 certification

- A1-6. The paper names an execution check and three criteria: (i) solve a validation episode, (ii) cover the segment's necessary actions, (iii) recur across traces. In code they map as follows. Execution check: `Workbench.run_operator` holds on at least 2 of the family's 4 holdout episodes (rung rule). (i): the rung's marginal test, where adding the operator to the family library turns at least one coverage-pool episode from unsolved to solved, with solved meaning the official score plus the ordering gate. (ii) has no separate code path in the admitted pipeline. The ordering gate inside the rung's `episode_solved` is the nearest thing and is applied through (i). (iii): the assembler's measured support, at least 2 over induction-half episodes 0..59. The union script then re-probes support on each column pool, which is unchanged.
- A1-7. Validation episodes per family are the admitted library's round-2 holdout (4) and coverage pool (8) from `outputs/v3/targets.json`, all induction-half indices.
- A1-8. Round structure. The admitted library had a first round with an empty family library, where any operator passing the execution check was accepted, and a second round with the marginal test. Mined operators are processed per family in descending support. The first operator of each (family, effect key) that passes the execution check is admitted as in round one. Every later one needs the marginal gain. The admitted round one was run per (family, seed episode, sample, effect key) cell, so this ordering rule is the closest mechanical analogue, not an identical procedure.
- A1-9. Column memories are built by the unchanged `scripts/viki_union_library.py`, as the admitted ones were: all 14 families for ID and CG, and the other 13 for each OOD fold. Layers 2 and 3 are then replaced by the admitted column memory's own. They were byte-identical before the swap in all 9 memories (`A/memories/swap_report.json`).

## A2

- A2-1. Re-ask rows are sent through the unchanged live runner `scripts/viki_eval_v2_intent_choice.py` (`--replay` with a file holding only those rows' archived first answers, `--memory` the replay-mined column memory, `--out-dir`). OOD rows use `--task-name <family>` with that fold's memory.
- A2-2. Services: 72B uses the archived command (`serve-qwen72b.sh`, the `unchanged` block of `amendment7/service_resource_reuse.json`) on GPUs 2,3,4,5. 30B and 7B use the archived `scripts/drivers/a11_serve.sh` arguments on GPUs 2–5. `max_num_seqs` is raised to 64 and the runner uses 64 workers, on user instruction. This changes only batching. Temperature-0 vLLM is already not row-deterministic under concurrency (see memory `viki-incontext-library-control`).
- A2-3. Qwen3-VL-30B-A3B-Instruct: the archived 30B cells were generated without `enable_thinking`. It is an Instruct (non-thinking) model, and the runner is kept unchanged, so no `extra_body` is passed. The chat template is checked for an `enable_thinking` branch before launch, and the result is recorded in report_A0.md.
- A2-4. "Which operator was selected" is logged by matching each chain of the schedule that `compose` returned against the memory's operator bodies (consistent variable binding). The CG "same operator sequence" count compares the multiset of selected operator bodies per row.
- A2-5. 30B and 7B services use `--gpu-memory-utilization 0.80` instead of the archived 0.90. After the 72B service stopped, another user's Isaac jobs held 8–11 GB on GPUs 2 and 4, so 0.90 would not fit. This is a resource-only change.
- A2-6. The Qwen3-VL-30B-A3B-Instruct chat template (`tokenizer_config.json`, `chat_template.json`, snapshot 9c4b90e1) has no `enable_thinking` branch. Passing `enable_thinking=False` would change nothing, and the archived 30B cells did not pass it. The unchanged runner is used.
- A2-7. For the CG same-sequence count, each chain is compared as the set of operator bodies it matches. The admitted library holds two `is_activated` operators with the identical body `Move ?x, Interact ?x` (they differ only in types and preconditions), so a chain can match that body twice. Operator counts in `replay_gap_by_family.csv` count chains (a row using three chains counts three).

# Part B (copy of B/deviations_B.md; edit there, both kept in sync)

B-D1..B-D9 were first written 2026-09-22 15:36, before B2 started (15:40); a concurrent Part A write replaced the
shared `deviations.md`, so this section was restored at 15:5x and is kept here as well. B-D1 was revised and
B-D10..B-D12 added before any B3 cell started. User instruction of 2026-09-22: do not stop at Gate B0 / Gate B2;
record the deviation, continue with the closest procedure, and mark it in the report. Inventory files: `B/`.

## B-D1. Gate B0 fails: the Table 5 run configs are not on disk
The three configs of `tab:hr_failure` (both / agent 0 only / agent 1 only) are in none of: repo A, repo B
(`all_scripts`, `old`, `our_method`, `logs`, git history, `freeze_2026-09-13`), `~/restore_isambard`, the
`~/Downloads` zips and tex files, the remote box. The manuscript's own `% TODO[hr-leak]` says the one-agent runs
are only in the Isambard outputs. No repo has a script with memory on only one agent. Gate B0 as written is not met.
Replacement (closest on disk): repo B `all_scripts/bash_files/run_ours_hr_mem_hr.sh`, the only "ours" script with
H_R data and the H_R memory slot (both agents with memory; template `rag_prompt_sequential_cooperation_skills`,
max_tokens 1500, rag_top_k 5, hierarchical retrieval, ablation switches at defaults). Changed: dataset = one fold,
`rag_dataset_dir` = the other fold's library, and per-agent `enable_rag` for the condition. An agent with
`enable_rag=False` gets the same template with `rag_examples = ""` (`llm_planner.py`), so the three conditions
differ only in which agent receives `rag_examples`. Agent 0 = leader, agent 1 = follower (manuscript
App. hr_failure). (The 09-22 part C used the Table 1 R_S config with the v4 template instead; every H_R "ours"
script on disk uses the non-v4 template with max_tokens 1500, so that one is taken here.)

## B-D2. The build manifest has no gate
`our_method/build_hierarchical_skill_memory.py` (archived H_R command `--include-failed --use-llm --use-api
--patch-failed`, Llama-3.3-70B-Instruct) has no execution check and no admission criteria: every LLM-extracted
skill is merged by name and saved. There are no gate settings to copy. B2 uses the same builder unchanged, i.e.
trace-conditioned proposal only, as the archived H_R library and the Table 1/2 libraries were built. "Proposals"
and "admitted" are the same set; the report gives skill and instance counts per fold.

## B-D3. Retrieval settings (a record, unchanged)
Effective theta = 0.3 (`habitat_llm/planner/rag.py:468`, `abstract_threshold=0.3`, hard-coded), instance_top_k = 5
(same call), k = `rag_top_k=5` on the command line. The templates' `rag_retrieval` blocks (`similarity_threshold:
0.7` in v4) are never instantiated: no code reads `rag_retrieval`.

## B-D4. Fold sizes
`sha256(episode_id) mod 2` on the dataset's string ids gives 89 / 108; one exact-instruction group (1987, 1988,
1989: "NOTE: JSON INCORRECT. OBJECT INITIALIZATION INCORRECT 'cabinet_0' not in kitchen.") straddles the folds,
so 1988 moves B -> A. Final 90 / 107, not "about 99 / 98". Episode 438 has no heuristic trajectory, so the fold-B
library is built from 106 episodes. The three "JSON INCORRECT" episodes are kept and evaluated.

## B-D5. Proposing-model resources
Llama-3.3-70B-Instruct (ModelScope weights, sha256-checked 09-22), vLLM TP 2 on GPUs 0,1 instead of the archived
TP 4, gpu-memory-utilization 0.80, `--enforce-eager`, max-model-len 8192, bf16 (as the 09-22 route-Q build). The
folds are built one after the other on one endpoint.

## B-D6. Library file format
The builder writes a directory (`L_ind_skills.json.gz`, `L_coop_skills.json.gz`, `episodic_memory.json.gz`,
`failure_analyses.json.gz`, `memory_summary.json`), which the retriever loads. `hr_fold<k>_library.json` is a
manifest of that directory (sha256 per file, counts, source episodes). `build_with_provenance.py` records the
source episode of every instance in `provenance.json`; it changes no builder argument, prompt or call.

## B-D7. Seeds
`planner_demo.py` hard-codes seed 47668090 and decoding is greedy; no run script varies a seed, so "the same seed
semantics as Table 1" is not recoverable from disk. Seeds 0/1/2 enter via `+seed_override` (python, numpy, torch
RNG and `habitat.seed`) plus `PYTHONHASHSEED=<seed>`. Decoding stays greedy. All cells use the same num_proc, so
the episode-to-process assignment (and hence the RNG stream per episode) is the same across conditions.

## B-D8. Harness switches (verified 09-22)
`+iclr_env_over_metrics=True` (score at env end instead of dropping the episode; denominator = all episodes),
`+retrieval_dump_dir` (per episode and agent: the bank entries whose text enters the prompt; joined with
`provenance.json` for source episode ids). Episodes that crash on the partner's unparsed action (`noneaction`,
original code) count as done with success 0.

## B-D9. Run order
Seed-major (the 6 condition x fold cells of seed 0, then seed 1, then seed 2); within a seed the single-sided cells
(agent 0 only, agent 1 only) first.

## B-D10. Decoding backend (user instruction 09-22)
Table 5 was decoded in-process with HF transformers + transformers-CFG (fp16, batch 1). B3 decodes with vLLM
(same snapshot 0e9e39f, fp16, temperature 0, same max_tokens and stopword) through an opt-in
`inference_mode=vllm_openai` (`scripts/iclr_two_exp/B/apply_vllm_backend.py`); the planner's per-call grammar is
passed unchanged as an xgrammar structured-output grammar. All conditions in this experiment use the same backend.
Validation against the HF smoke cells of 09-22 (same 5 episodes, seed 0, same config): see
`B/backend_validation.json` and the note appended below once it finishes.

### B-D10 validation result (16:15, before any B3 cell)
Same 5 H_R episodes (`hr_eval_smoke5`), seed 0, same config as the 09-22 HF smoke cells (C1 both-memory, C2
leader-only; v4 template, route-Q library), vLLM backend vs the HF cells on disk (`B/backend_validation_C{1,2}.json`,
episodes both sides finished: C1 439/443/445/447, C2 439/445/447):
- parse-error turns (next user message reports a syntax/parse error): C1 HF 21/250 (8.4%) vs vLLM 0/192 (0.0%);
  C2 HF 11/95 (11.6%) vs vLLM 11/219 (5.0%). Not higher on vLLM.
- percent complete (mean over episodes with stats): C1 0.625 vs 0.639; C2 0.708 vs 0.750. Success 0/4 vs 0/3 (C1),
  0/2 vs 1/3 (C2).
- runtime per episode: C1 1,077 s vs 249 s; C2 1,407 s vs 553 s.
- Per-turn identity is not testable: the first prompt of the same episode already differs between the two runs
  (agent start room, retrieval scores in the second decimal), independent of the backend. The odd tokens between
  turns ("datingsider", "assistantinely") appear in both backends' transcripts (prompt construction, not decoding).
- One vLLM episode (C1 443) crashed on a 31,498-token prompt against a 32,768 context; all endpoints now use
  max-model-len 65,536. The HF path has no context cap.

## B-D11. Failure metrics
No script that computes Conflict/Ep, FailPick/Ep, SelfConf/Ep, NotClose/Ep is on disk (searched repo A, repo B,
zips, remote). B4 will need a new counter written from the manuscript's definitions; its definitions will be
recorded here before it is run on B3 output.

## B-D12. Listing 3
Listing 3 (`lst:ours`, "Observation Diff / Skill Prediction ...") matches no template file on disk; neither the
v4 template nor `rag_prompt_sequential_cooperation_skills.yaml` (used here) contains that text.

## B-D13. Run resources
num_proc = 8 in every B3 cell (the H_R script used 14), habitat processes on the slot's GPU, LLM calls to an 8B vLLM
endpoint (GPU 6 port 8206 at first, GPUs 0/1 ports 8200/8201 after B2). The backend validation (B-D10) ran on the
v4-template config of the HF smoke cells, not on the B3 template; the backend change is the same code path for both.

## B-D14. Concurrent fold builds (user instruction 09-22 16:19)
From 16:19 fold B is built concurrently with fold A on the same 70B endpoint (supersedes the "one after the other" in B-D5). Within a fold the builder stays sequential: skills are merged by name in episode order, so parallel episodes would change which episode names a skill. Per-fold call counts are no longer separable in the vLLM log; the driver reports the total.
