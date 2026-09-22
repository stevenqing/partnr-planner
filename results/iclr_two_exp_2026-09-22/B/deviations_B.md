# Deviations from SPEC.md (Part B)

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

## B-D15. Failure-metric operationalisation (written 09-23 01:00, before the final B4 run)
Definitions from the manuscript appendix "Coordination Failure Metrics"; computed by
`scripts/iclr_two_exp/B/b4_metrics.py` from `planner-log-episode_<id>_0.json`. Where the text is not operational:
- Time unit: one high-level action of an agent (a replan step with a tool call), not a simulator step; outcome = the
  agent's next response that is not "still in progress". Replans whose output did not parse (SyntaxError, no tool)
  are not actions.
- Conflict "within one coordination round": agent i issues Pick/Place/Open/Close on o while agent j's latest
  action, issued after agent i's previous action, is a manip action on o. Counted once per such event.
- FailPick: Pick whose outcome is not "Successful execution!" (inventory is not logged; outcome text is used).
- SelfConf: the formula counts every Pick(o)/Place(o) pair within tau=5, which fires on every ordinary transport;
  the text restricts it to placing o back where it was picked. Reported value = the restricted form (source of o =
  its location in the agent's world graph at the Pick step; plus Open(o)/Close(o) pairs); the literal form is
  reported as SelfConf_literal.
- NotClose: target-object positions are not logged, so d_thresh = 1.5 m cannot be applied; counted as manip actions
  whose outcome is the skill's "Not close enough" failure.
- Episodes that crash with `noneaction` write no planner log: success and completion count 0 (n = 197), failure
  metrics are averaged over the episodes with a log (n reported per seed).
Consequence: the four failure metrics are not on the same scale as Table 5 (whose computation is not on disk);
only within-experiment comparisons between conditions are meaningful.
