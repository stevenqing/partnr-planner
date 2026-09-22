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

## B-D11. Failure metrics
No script that computes Conflict/Ep, FailPick/Ep, SelfConf/Ep, NotClose/Ep is on disk (searched repo A, repo B,
zips, remote). B4 will need a new counter written from the manuscript's definitions; its definitions will be
recorded here before it is run on B3 output.

## B-D12. Listing 3
Listing 3 (`lst:ours`, "Observation Diff / Skill Prediction ...") matches no template file on disk; neither the
v4 template nor `rag_prompt_sequential_cooperation_skills.yaml` (used here) contains that text.
