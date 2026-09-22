# Spec: two experiments for the ICLR 2027 Memory-as-Skill submission

Date: 2026-09-22. Results go to `results/iclr_two_exp_2026-09-22/`. Each part has an inventory step with zero model calls, a frozen configuration, and pre-written endings. Run the inventory, report, and wait for approval before any generation.

## Shared rules

1. Frozen artifacts are read-only. That covers the admitted VIKI-L2 library (8 operators, 3 ordering rules, vocabulary), the RQ2 first-turn outputs, every PARTNR library behind Tables 1 and 2, and every logged baseline run. Nothing here reruns a baseline.
2. Every number in a report comes from a file on disk that the report names. No number is typed from memory.
3. The two parts together add at most 10 lines to the main text. Everything else goes to the appendix.
4. Any deviation from this spec is recorded in `deviations.md` in the results directory before the affected run starts.

## Part A. VIKI-L2 same-planner control with replay-mined operators

Question: how far is the admitted 8-operator library, proposed by the LLM from traces, from a library mined mechanically from the same induction episodes, on the same first-turn model outputs and the same planner. Nothing in this part is written by hand. The word "hand-written" does not appear in any report or in the paper.

### A0. Inventory (zero calls)

Locate and list, with paths and hashes:
- the first-turn model outputs of the RQ2 full arm for Qwen2.5-VL-72B, Qwen3-VL-30B-A3B, Qwen2.5-VL-7B on ID (924), single-family OOD (924), CG w/ Image (297), CG w/o Image (297);
- the planner entry point, the re-ask policy (confirm one re-ask at most), the ordering rules file, the vocabulary file;
- the operator schema (fields, variable syntax, how a demo is bound);
- the list of the 14 training families and, for each, the effects that appear in its reference plans, computed from the induction half only (the 3,598 even-indexed episodes).

Gate A0: the four split files, the planner, and the schema all exist and the RQ2 full-arm outputs are byte-identical to the ones behind Figure 3. Stop and report if not.

### A1. Replay-mined operator set

Mine operators from the induction half only (the 3,598 even-indexed episodes), with no LLM call and no manual editing:
1. For each episode, replay the reference plan in the simulator and record the state before and after every primitive action.
2. Segment the plan at every point where a goal predicate of the episode becomes true. Each segment is a candidate body; its effect is the predicate that became true; its preconditions are the predicates that held at the segment start and were read by the segment.
3. Lift the candidate to variables by replacing every asset and place name with a typed variable, using the same variable syntax as the admitted operators.
4. Deduplicate by (effect, variable-lifted body). Record support as the number of episodes whose replay produced the body.

Store as `replay_library.json` with the mining script hash. Record the count of operators and the families each covers.

Certification: run every mined operator through the same execution check and the same three admission criteria used for the admitted library, on validation episodes from the induction half. Log pass/fail per operator and the reason for each failure. The certified set is frozen before A2. Report both counts, mined and certified, and use only the certified set in A2.

### A2. Planner runs

For each of the 12 cells (3 models x 4 splits): feed the frozen RQ2 first-turn outputs to the planner with the certified `replay_library.json` in place of the admitted library, keeping the same ordering rules and vocabulary and the same re-ask policy. Log per row: solved (strict SOLVED), reason code, whether a re-ask happened, which operator was selected.

Re-ask turns are the only model calls in Part A. Cap: 2,000 calls total. Log the re-ask rate per cell next to the full arm's rate.

### A3. Scoring

Per cell: strict SOLVED rate for the replay-mined arm; paired exact McNemar replay-mined vs ours (rows solved by exactly one arm). Per family on ID and OOD: rows solved by the replay-mined arm and missed by ours, with the operator it used. On CG: the same, plus the count of rows where the two arms select the same operator sequence.

### A4. Outputs

`tables/replay_vs_ours.csv` with columns model, split, n, ours, replay, ours_only, replay_only, p. `tables/replay_gap_by_family.csv`. `report_A.md` with the ending that fired.

### Pre-written endings for Part A

Sanity condition first: at 72B ID the replay-mined arm must be at least as high as ours. If it is lower, the mining or certification lost operators the LLM found; stop, diagnose, and report which effects are missing before anything is written.

- Ending A1 (ours within 5 points of the replay-mined arm on both CG cells at 72B): main-text sentence, "Operators mined by replaying the reference plans of the same induction episodes, certified by the same gate and consumed by the same planner, reach X% and Y% on CG at 72B, so the eight admitted operators are within Z points of a library that does not depend on LLM proposal."
- Ending A2 (replay-mined arm above ours by more than 5 points on CG, or by more than 15 on single-family OOD): main-text sentence, "The same planner with operators mined by replay from the same induction episodes reaches X% on single-family OOD against 54.2% for the admitted library; the gap lies in families F1, F2, F3, for which trace-conditioned proposal admits no operator." Add the family table to the appendix.
- Ending A3 (mixed): report both sentences with the cells they apply to.

In all endings the appendix gets `replay_vs_ours.csv` as a table and a paragraph stating the mining procedure, the mined and certified counts, and that the pool is the induction half. The arm is named "replay-mined" everywhere. It is a control on proposal, not an upper bound claim, because certification can also drop operators.

## Part B. Held-out H_R for symmetric deployment

Question: does the leader-only inversion (49.6 vs 22.3 success, Conflict/Ep 1.63 to 0.33) hold when the H_R memory is built on episodes disjoint from the evaluated ones.

### B0. Inventory (zero calls)

List the 197 H_R episode ids and their instruction texts. Locate the existing H_R library and its build manifest (which episodes, which proposing model, gate settings). Locate the run configs of the Table 5 conditions (both, agent 0 only, agent 1 only) and confirm what the seed controls. Confirm the deployed template is Listing 3, retrieval config theta=0.3 and k=5, planner Llama-3.1-8B.

Gate B0: the three run configs exist and differ only in which agent gets `rag_examples`. Stop and report if they differ elsewhere.

### B1. Split

Deterministic two-fold split of the 197 ids by `sha256(episode_id) mod 2`: fold A (about 99) and fold B (about 98). Save `split.json`. Check that no instruction text appears in both folds; if one does, move that episode so both copies sit in the same fold and log it. Every episode is evaluated exactly once, with a memory built from the other fold, so the pooled evaluation has n = 197.

### B2. Libraries

Build two H_R libraries, one from fold A and one from fold B, each with the full pipeline: trace-conditioned proposal, execution check, and the three admission criteria, same proposing model and gate settings as the manifest in B0. Log proposals, admitted, and the individual/cooperation split for each. Freeze as `hr_foldA_library.json` and `hr_foldB_library.json`.

Gate B2: each fold library must admit at least one cooperation skill and at least one individual skill for each of Pickup, Place, and the state-changing actions that H_R goals require. If a fold library misses one of these, report the yield and stop; the experiment would then measure coverage, not deployment.

Retrieval for fold-A episodes reads only `hr_foldB_library.json` and the reverse. Verify by listing each bank's source episode ids against `split.json` before any run.

### B3. Runs

Three conditions on each fold with Llama-3.1-8B: both agents with memory, agent 0 only, agent 1 only. Three seeds each, seeds 0, 1, 2, same seed semantics as Table 1. About 197 x 3 x 3 = 1,770 episodes. Cap: 2,000 episodes. Prompt, theta, k, and instance filter unchanged. Log every retrieval call with the source episode ids of the returned instances.

### B4. Metrics

Per condition, pooled over both folds: success, completion, Conflict/Ep, FailPick/Ep, SelfConf/Ep, NotClose/Ep, each as mean and sample std over the three seeds. Paired comparison agent-0-only vs both: per seed, exact McNemar on the 197 pooled episodes; report all three p values. Also report both-memory success on this held-out protocol next to the covered-episode value 22.3 from Table 5, so the size of the memorization effect is visible. Fallback rate and a zero-overlap check (no returned instance from the episode's own fold) from the retrieval log.

### B5. Optional, only if B3 completes under cap

N=3 on the same two folds with the same libraries: all-mem, leader-only, followers-only, one seed. Reported as descriptive.

### Outputs

`tables/hr_heldout.csv` mirroring Table 5 with mean and std columns. `report_B.md` with the ending that fired and the library yield.

### Pre-written endings for Part B

- Ending B1 (agent-0-only success exceeds both-memory success by more than the larger std, and Conflict/Ep is lower, in all three seeds): replace the seen-episode qualifiers. Abstract: delete "on heterogeneous rearrangement episodes that the memory already covers" and state the held-out numbers. Section 5.4: replace "The H_R memory covers the evaluated episodes ... not as a measure of transfer" with "On a held-out half of H_R (Appendix ref), leader-only reaches X% against Y% for both, and Conflict/Ep falls from A to B." Table 5 becomes the held-out table; the seen-episode table moves below it with one sentence.
- Ending B2 (direction holds but not by more than one std, or holds in fewer than three seeds): keep the seen-episode numbers in Section 5.4, add "On a held-out half the ordering is the same but the gap is within run-to-run variation (Appendix ref)." Abstract keeps the qualifier.
- Ending B3 (direction reverses or vanishes): Section 5.4 reports both, Limitations adds "The leader-only advantage on H_R is measured on episodes the memory covers and does not appear on a held-out half." The abstract sentence is removed.

## What not to do

Do not tune theta or k. Do not rebuild the Table 1 or Table 2 libraries. Do not rerun any baseline. Do not touch the CG split. Do not use test-split rows to mine operators. Do not edit any mined operator by hand; a mined operator that fails certification is dropped, not repaired. Do not merge seen-episode and held-out numbers in one table without a column that says which is which.
