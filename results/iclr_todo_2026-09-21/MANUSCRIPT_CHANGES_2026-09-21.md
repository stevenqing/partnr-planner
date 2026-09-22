# ICLR 2027 稿件修改清单（2026-09-21）

给改稿 agent 用。基准是带 TODO 的原稿（`main.orig.tex`，行号以它为准）。下面 18 处改动已在工作副本 `results/iclr_todo_2026-09-21/src/main.tex` 上做完，逐处给出**原文**与**替换后全文**，请在你手里那份 main.tex 上逐条查找替换。完整 unified diff 在 `results/iclr_todo_2026-09-21/main_patch.diff`。

## 一、结论摘要（改稿时要守住的事实）

1. **PARTNR 泄漏范围**：已按每个结果实际用的记忆配置核对，同时按 episode id 和指令原文匹配。Table 1、Table 2 的目标 episode 都不在各自的源记忆里（ours、RAG、MEMENTO 都是 0 重叠）。只有「H_R 记忆 → H_R」重叠：ours 196/197、RAG 196/197、MEMENTO 源集 197/197。检索不排除当前 episode。G-Memory 没有代码，查不了。**不要写成「Table 1/2 泄漏」。**
2. **PARTNR 库没有执行门**：Table 1/2 的库由 Llama-3.3-70B 提议后直接入库。执行门只在 VIKI-L2 和一个 PARTNR 技能（is_in_room）上评估过。
3. **19 算子参考库不是手写的**：它由仿真器重放 VIKI-L2 train 偶数行挖出，没有 LLM 提议，并带 8 个逐折版本。单族 OOD 必须用逐折版本：72B 0.656、30B 0.460、7B 0.145（不是 0.674 / 0.510 / 0.160）。在三个规模的 OOD 上它都高于我们（0.542 / 0.354 / 0.117，配对 105/0、98/0、26/0）。ID 上我们在 72B、30B 领先（0.780 vs 0.674，0.574 vs 0.510），7B 落后（0.139 vs 0.160）。
4. **VIKI 格数**：72 个主比较格 + 36 个 RQ2 格 + 36 个 RQ3 格 = 144，另有 18 个 sibling 格。30B 组合泛化的 4 个消融格 09-14 已补跑，数值：CG w/ Image 下 no_grounding 0.024、no_order 0.552（full 0.650）；CG w/o Image 下 no_grounding 0.724、no_order 0.643（full 0.724）。
5. **检索实现**：测试时没有按任务类型路由（每次运行用哪个库由配置写死），也没有 object-type boost（日志里最高检索分 0.965）。θ=0.3 作用于原始 cosine。embedding 是 all-mpnet-base-v2。
6. **MEMENTO H_R_T**：2026-09-21 在全部 11 个 H_R_T episode 上重跑，记忆 H_R+H_T，Llama-3.1-8B，旧流水线代码不变。结果 **1/11 = 9.1%**，平均完成度 0.393。稿件原来的 14.5% 对不上任何一次运行。重跑后 MEMENTO 排到 RAG（11.1）和 ToM（9.2）之后。
7. **判分口径**：统一成 JSON-tolerant 或官方 strict 任一种，60 对比较的胜负方向与显著性都零翻转（最大 p=7.2e-6）。现在的混合口径其实偏向 baseline。表在 `tables/scorer_parity.{csv,tex}`，是否写进附录由作者定。

## 二、逐处修改（按原稿行号顺序）

### 1. [library-provenance] 原稿第 123 行

Introduction 贡献二收窄：执行门只在 VIKI-L2 和一个 PARTNR 技能上评估过。

**原文**

```latex
We evaluate on two complementary benchmarks. PARTNR measures end-to-end cooperation and transfer across constraint compositions, whereas VIKI-L2 isolates task-family and text/image shifts. Memory-as-Skill reaches 88.2\% success on PARTNR with Llama-3.3-70B. Under sibling-group withholding on VIKI-L2, it is ahead of G-Memory at 72B and 30B and statistically tied at 7B. This paper makes three contributions. First, we introduce a hierarchy that separates abstract skills from grounded instances and individual from cooperation knowledge. Second, we develop a construction pipeline that couples trace-conditioned proposal with execution-based admission. Third, we provide a cross-benchmark evaluation that connects task performance and held-out-family transfer to construction and coordination mechanisms.
```

**替换为**

```latex
We evaluate on two complementary benchmarks. PARTNR measures end-to-end cooperation and transfer across constraint compositions, whereas VIKI-L2 isolates task-family and text/image shifts. Memory-as-Skill reaches 88.2\% success on PARTNR with Llama-3.3-70B. Under sibling-group withholding on VIKI-L2, it is ahead of G-Memory at 72B and 30B and statistically tied at 7B. This paper makes three contributions. First, we introduce a hierarchy that separates abstract skills from grounded instances and individual from cooperation knowledge. Second, we develop a construction pipeline that couples trace-conditioned proposal with execution-based admission, and we evaluate the admission gate on VIKI-L2 and on one targeted PARTNR skill. Third, we provide a cross-benchmark evaluation that connects task performance and held-out-family transfer to construction and coordination mechanisms.
```

### 2. [library-provenance] 原稿第 171–178 行

删掉 TODO 注释（对应的声明已写进 5.1 Evaluation protocol）。

**删除这段**

```latex
% TODO[library-provenance] The PARTNR banks behind Table 1 hold several hundred
% skills each, and cooperation skills on HT/HRST average 1.0 instance, which
% conflicts with criterion (iii). If those banks did not pass this gate, say so
% in Section 5.1, for example: "The PARTNR libraries in Tables 1 and 2 are built
% by trace-conditioned proposal without the execution gate. The gate is
% evaluated on VIKI-L2 and on one targeted PARTNR skill (Section 5.3)."
% The alternative is to rebuild the PARTNR bank with the gate and rerun Table 1
% on Llama-3.1-8B.
```

### 3. [retrieval-impl] 原稿第 202–206 行

Eq.1 后两句改为代码事实：库由每次运行的配置固定，测试时没有分类器选库；没有 object-type boost，两级检索都用原始 cosine。同时删掉 TODO。

**原文**

```latex
Here, $\tau_t^j$ is agent $j$'s local history, $\text{sim}(\cdot,\cdot)$ is cosine similarity over \texttt{all-mpnet-base-v2} sentence embeddings, and $\mathcal{L}=\mathcal{L}_{\text{ind}}\cup\mathcal{L}_{\text{coop}}$. The first retrieval level matches the predicted behavior to a skill name (e.g., \texttt{Pickup} or \texttt{Handover}). The second level ranks demonstrations within that skill by current context. The PARTNR implementation adds two steps. A proposition-based task-type classifier first routes the query to the skill bank of the predicted type (Appendix~\ref{sec:task_classification_appendix}). The skill-level score then adds a fixed boost of 0.15 when an object type named in the instruction also appears in the skill, so reported scores can exceed 1 (Appendix~\ref{subsec:retrieval_examples}).
% TODO[retrieval-impl] The two sentences above are reconstructed from the
% appendix (routing figure, "object-type boost 0.150", post-boost medians
% above 1). Check them against hierarchical_retrieval.py, in particular
% whether the boost applies at the skill level or the instance level.
```

**替换为**

```latex
Here, $\tau_t^j$ is agent $j$'s local history, $\text{sim}(\cdot,\cdot)$ is cosine similarity over \texttt{all-mpnet-base-v2} sentence embeddings, and $\mathcal{L}=\mathcal{L}_{\text{ind}}\cup\mathcal{L}_{\text{coop}}$. The first retrieval level matches the predicted behavior to a skill name (e.g., \texttt{Pickup} or \texttt{Handover}). The second level ranks demonstrations within that skill by current context. In the PARTNR runs, the skill banks searched are fixed by each run's configuration and stated with each result, and no classifier selects a bank at test time. Both retrieval levels score by raw cosine similarity with no additional term.
```

### 4. [viki-instantiation] 原稿第 247–252 行

重写 VIKI-L2 实例化段落，原第 1、3、4 句不准确（CG w/o Image 没有图；有一个双角色接力算子；模型输出的是 requirement，不是技能名；planner 负责选算子、分配接力角色、套用顺序规则）。同时删掉 TODO。

**原文**

```latex
A VIKI-L2 episode gives an instruction and a scene image and asks for one complete multi-robot plan. There are no alternating rounds, so no partner effect is observed and the cooperation branch of the library is empty on this benchmark. We call an admitted VIKI-L2 skill an operator. Our method outputs a sequence of skill calls with robot assignments chosen by the LLM, and a deterministic planner expands each call into primitive actions using the ordered \texttt{demo} of the admitted operator. All comparison baselines output primitive plans directly. Each CG instruction combines a two-robot cutting subtask with an independent delivery subtask. No training episode contains this combination, although each part appears in training.
% TODO[viki-instantiation] Drafted from the method notes and the Sep 5 leakage
% audit. Verify every sentence against the code. The audit allows the claim
% "two-robot cut plus independent delivery is unseen" and does not allow
% "activation plus delivery is unseen". The 297 CG rows hold 295 distinct
% task_ids (4307_8-1 and 1426_9-1 repeat once). Decide whether to report that.
```

**替换为**

```latex
A VIKI-L2 episode gives an instruction, with a scene image on every split except CG w/o Image, and asks for one complete multi-robot plan. There are no alternating rounds, so no partner effect is observed and the library holds no effect-conditioned cooperation skill on this benchmark. Its one multi-robot operator is a fixed two-role relay. We call an admitted VIKI-L2 skill an operator. For each requirement our method states the effect, the object, the target place, and the robots that carry it out. A deterministic planner selects an admitted operator with a matching effect, binds its ordered \texttt{demo} to the named entities, keeps the robot named by the LLM for single-robot operators, assigns the roles of the relay itself, and orders the resulting primitive actions with the ordering rules stored in the library. All comparison baselines output primitive plans directly. Each CG instruction combines a two-robot cutting subtask with an independent delivery subtask. No training episode contains this combination, although each part appears in training.
```

### 5. [library-provenance + partnr-leak] 原稿第 266 行

Evaluation protocol：声明 PARTNR 库没有过执行门；声明记忆来源与泄漏范围（Table 1/2 的目标 episode 与源记忆不相交，已逐配置核对；H_R 记忆含 197 个 H_R episode 中的 196 个，从它检索的 H_R 结果属于已见 episode 上的复用）。

**原文**

```latex
Memory is frozen during evaluation. PARTNR comprises more than 14{,}000 runs across 205 configurations, with per-task sample sizes between 5 and 197 (Appendix~\ref{sec:skill_memory_analysis}). VIKI-L2 induction sees only the 3{,}598 even-indexed episodes from its 7{,}196-episode training pool, and the other half is unavailable for proposal. The search over 14 families contains 896 combinations of family and effect. We use paired McNemar tests for binary outcomes and paired bootstrap intervals for completion. Each VIKI-L2 configuration uses one library built once with a fixed sampling seed, so the variance from library construction is not estimated. Evaluation decodes at temperature zero, and three repeats of the 72B and 7B full arms are identical row by row. Significance on VIKI-L2 therefore rests on the paired tests alone.
```

**替换为**

```latex
Memory is frozen during evaluation. The PARTNR libraries in Tables~\ref{tab:main_results_task} and~\ref{tab:partnr_composition} are built by trace-conditioned proposal without the execution gate. The gate is evaluated on VIKI-L2 and on one targeted PARTNR skill (Section~\ref{subsec:induction_validation}). Every PARTNR memory, including those of RAG and MEMENTO, is built offline from heuristic-agent trajectories on the episodes of its source task group, and retrieval does not exclude the current episode. No target episode in Tables~\ref{tab:main_results_task} and~\ref{tab:partnr_composition} occurs in its source memory. The H\_R memory, however, is built from 196 of the 197 H\_R episodes, so any H\_R result that retrieves from it (Appendices~\ref{sec:hr_failure_appendix} and~\ref{sec:n_agent_extension}) measures reuse on seen episodes rather than transfer. PARTNR comprises more than 14{,}000 runs across 205 configurations, with per-task sample sizes between 5 and 197 (Appendix~\ref{sec:skill_memory_analysis}). VIKI-L2 induction sees only the 3{,}598 even-indexed episodes from its 7{,}196-episode training pool, and the other half is unavailable for proposal. The search over 14 families contains 896 combinations of family and effect. We use paired McNemar tests for binary outcomes and paired bootstrap intervals for completion. Each VIKI-L2 configuration uses one library built once with a fixed sampling seed, so the variance from library construction is not estimated. Evaluation decodes at temperature zero, and three repeats of the 72B and 7B full arms are identical row by row. Significance on VIKI-L2 therefore rests on the paired tests alone.
```

### 6. [memento-hrt] 原稿第 349 行

正文：MEMENTO H_R_T 14.5 → 9.1（2026-09-21 重跑，1/11），排序随之改变。

**原文**

```latex
Constraint composition remains difficult. Memory-as-Skill reaches 27.7\% success on H\_R\_T, compared with 19.8\% for G-Memory, 14.5\% for MEMENTO, and 11.1\% for RAG. On H\_R\_S\_T it reaches 40.0\%, compared with 31.5\% for G-Memory and 20.0\% for MEMENTO. The methods in the H\_R\_S\_T row do not share one matched episode count, so we treat that row as descriptive. Table~\ref{tab:partnr_composition} reports these transfers.
```

**替换为**

```latex
Constraint composition remains difficult. Memory-as-Skill reaches 27.7\% success on H\_R\_T, compared with 19.8\% for G-Memory, 11.1\% for RAG, and 9.1\% for MEMENTO. On H\_R\_S\_T it reaches 40.0\%, compared with 31.5\% for G-Memory and 20.0\% for MEMENTO. The methods in the H\_R\_S\_T row do not share one matched episode count, so we treat that row as descriptive. Table~\ref{tab:partnr_composition} reports these transfers.
```

### 7. [memento-hrt] 原稿第 362 行

Table 2：MEMENTO H_R_T 格 14.5 → 9.1。

**原文**

```latex
H\_R\_T & H\_R+H\_T & \textbf{27.7} & 11.1 & 14.5 & 0.0 & 9.2 & 19.8 \\
```

**替换为**

```latex
H\_R\_T & H\_R+H\_T & \textbf{27.7} & 11.1 & 9.1 & 0.0 & 9.2 & 19.8 \\
```

### 8. [memento-hrt] 原稿第 369–371 行

「earlier run」那句改为如实说明重跑，删掉 TODO。

**原文**

```latex
The MEMENTO H\_R\_T value of 14.5\% comes from an earlier run and is not repeated in the final evaluation pass.
% TODO[memento-hrt] Rerun this cell or confirm it from logs, then delete the
% sentence above.
```

**替换为**

```latex
The MEMENTO H\_R\_T cell is a rerun of the same code and memory on all 11 H\_R\_T episodes (1/11 successes, mean completion 39.3\%); the earlier value of 14.5\% could not be matched to any run.
```

### 9. [handwritten-7b] 原稿第 382 行

RQ2 正文：参考库不是手写的（由仿真器重放训练集挖出，没有 LLM 提议）；72B OOD 改用逐折参考库 65.6%（原 67.4% 用的是见过留出族的完整库）。

**原文**

```latex
ToM and zero-shot both emit primitive plans, and ToM differs significantly from zero-shot in none of the 12 paired cells (all $p\geq.296$). In its highest-scoring cell, 72B OOD, ToM raises in-scene-object validity from 52.7\% to 64.4\%, yet success moves from 3.5\% to 2.7\% (Table~\ref{tab:tom_funnel}, Figure~\ref{fig:tom_failure}). Our method instead outputs skill calls, so its gap to the primitive-plan baselines measures the skill-call interface and the library together. A hand-written 19-operator library with the same interface exceeds the learned library on 72B OOD (67.4\% vs.\ 54.2\%), while the learned library leads on 72B ID (Appendix~\ref{sec:viki_audit}).
```

**替换为**

```latex
ToM and zero-shot both emit primitive plans, and ToM differs significantly from zero-shot in none of the 12 paired cells (all $p\geq.296$). In its highest-scoring cell, 72B OOD, ToM raises in-scene-object validity from 52.7\% to 64.4\%, yet success moves from 3.5\% to 2.7\% (Table~\ref{tab:tom_funnel}, Figure~\ref{fig:tom_failure}). Our method instead outputs skill calls, so its gap to the primitive-plan baselines measures the skill-call interface and the library together. A 19-operator reference library with the same interface, mined by simulator replay of the same even-indexed training episodes without LLM proposal, exceeds the learned library on 72B OOD even when the held-out family is withheld from it too (65.6\% vs.\ 54.2\%), while the learned library leads on 72B ID (Appendix~\ref{sec:viki_audit}).
```

### 10. [cells] 原稿第 412 行

消融图图注：删掉「30B 只有 ID/OOD」，这四格 09-14 已补跑。加一条 TODO[cells-fig]：图 PDF 要重画。

**原文**

```latex
    \caption{VIKI-L2 representation ablation. OOD is single-family held-out and CG denotes compositional generalization. The 30B ablation covers ID and OOD only, so its compositional cells are marked as not run.}
```

**替换为**

```latex
    \caption{VIKI-L2 representation ablation. OOD is single-family held-out and CG denotes compositional generalization.}
% TODO[cells-fig] figs/ for this figure still marks the four 30B CG cells as not run.
% They exist (2026-09-14): w/ Image no_grounding 0.024, no_order 0.552 (full 0.650);
% w/o Image no_grounding 0.724, no_order 0.643 (full 0.724). Regenerate the figure.
```

### 11. [retrieval-impl] 原稿第 418 行

RQ3：删掉指向路由诊断的那句（路由消融图已删）。

**原文**

```latex
Figure~\ref{fig:ablation} removes cooperation memory, hierarchy, or individual skills under matched Llama-3.1-8B evaluation. Removing cooperation skills causes the largest losses and reaches zero success on the hardest transfers. Flattening the hierarchy likewise collapses 4-constraint transfer. Additional routing diagnostics appear in Appendix~\ref{sec:detailed_partnr_diagnostics}.
```

**替换为**

```latex
Figure~\ref{fig:ablation} removes cooperation memory, hierarchy, or individual skills under matched Llama-3.1-8B evaluation. Removing cooperation skills causes the largest losses and reaches zero success on the hardest transfers. Flattening the hierarchy likewise collapses 4-constraint transfer.
```

### 12. [retrieval-impl] 原稿第 453–462 行

附录：删掉路由消融图（figs/tasktype_ablation.pdf）及其段落，开头那句改为只引 fig:one_agent。

**原文**

```latex
The main text contains the cross-benchmark evidence for each research question. The appendix provides additional PARTNR analyses of failure scenarios, classifier fidelity, retrieval sensitivity, and induced skill libraries. Two supplementary diagnostics help interpret RQ3. Figure~\ref{fig:tasktype} evaluates task-aware routing, and Figure~\ref{fig:one_agent} isolates the effect of assigning the learned library to only one role.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.72\linewidth]{figs/tasktype_ablation.pdf}
    \caption{PARTNR task-type routing ablation. ``No cls'' retrieves from the merged skill pool, and task-aware routing selects the corresponding skill bank.}
    \label{fig:tasktype}
\end{figure}

Task-aware routing is most useful when the task composition is heterogeneous because the routed library avoids irrelevant candidates, whereas the merged-pool condition exposes retrieval to cross-type interference. This diagnostic is supplementary because the main component ablation already shows that the larger gains come from cooperation-specific skills and hierarchical organization, not from classifier headroom alone.
```

**替换为**

```latex
The main text contains the cross-benchmark evidence for each research question. The appendix provides additional PARTNR analyses of failure scenarios, classifier fidelity, retrieval sensitivity, and induced skill libraries. A supplementary diagnostic helps interpret RQ3. Figure~\ref{fig:one_agent} isolates the effect of assigning the learned library to only one role.
```

### 13. [cells] 原稿第 474–478 行

附录格数：72 + 36 + 36 = 144，另加 18 个 sibling 格；删掉「四个 30B 格没跑」和 TODO。

**原文**

```latex
This section records the controls underlying the main-text claims. All reported VIKI-L2 cells are computed from per-episode outputs. They comprise 72 main-comparison cells, 36 RQ2 cells, and 32 RQ3 cells. Four 30B compositional ablation cells are not run. Our method and the RQ2/RQ3 ablations use simulator \texttt{SOLVED}. Comparison baselines use the JSON-tolerant parser because the official scorer rejects some otherwise valid percentage-form outputs.
% TODO[cells] This paragraph counts 72 main-comparison cells, and the
% "Pairing" paragraph below counts 90 combinations of model, split, and method.
% 72 = 3 models x 4 splits x 6 methods and 90 adds the sibling-group split.
% Use one count, or say which split the 72 leaves out.
```

**替换为**

```latex
This section records the controls underlying the main-text claims. All reported VIKI-L2 cells are computed from per-episode outputs. They comprise 72 main-comparison cells, 36 RQ2 cells, and 36 RQ3 cells, 144 in total, plus 18 sibling-group comparison cells. Our method and the RQ2/RQ3 ablations use simulator \texttt{SOLVED}. Comparison baselines use the JSON-tolerant parser because the official scorer rejects some otherwise valid percentage-form outputs.
```

### 14. [cells + handwritten-7b] 原稿第 504–506 行

Pairing 段：90 改为 72 + sibling 18（sibling 上没跑 ToM，第六臂是早期 skill-memory 变体）；参考库一句改写，OOD 用逐折值 0.656/0.460/0.145，补上 30B OOD，ID 与 OOD 分开写。同时删掉 TODO。

**原文**

```latex
The main VIKI comparison contains all 90 expected combinations of model, split, and method. Our method exceeds ToM in all 12 model and split comparisons, and ToM is not significantly different from zero-shot in any of its 12 paired cells. Three temperature-zero repeats of the 72B and 7B full arms are identical row by row. This confirms that decoding is deterministic and does not estimate the variance from library construction, since each configuration uses one library. The 30B full arm has one run. The 30B and 7B comparison baselines use the think condition. No matched no-think variant exists for our method, and baseline-only no-think results vary by method, so we draw no cross-condition conclusion. A 19-operator hand-written reference library further cautions against attributing every score to induction. It outperforms the learned eight-operator library on 72B OOD (0.674 vs.\ 0.542) and 7B ID/OOD (0.160 vs.\ 0.139/0.117), whereas the learned library leads on 72B ID and 30B ID.
% TODO[handwritten-7b] "0.160 vs. 0.139/0.117" gives one reference value for
% two splits. Write the reference ID and OOD values separately.
```

**替换为**

```latex
The main VIKI comparison contains all 72 expected combinations of three models, four splits, and six methods. The sibling-group split adds 18 cells for the same three models. ToM is not run on that split, and its sixth arm is an earlier skill-memory variant of our method. Our method exceeds ToM in all 12 model and split comparisons, and ToM is not significantly different from zero-shot in any of its 12 paired cells. Three temperature-zero repeats of the 72B and 7B full arms are identical row by row. This confirms that decoding is deterministic and does not estimate the variance from library construction, since each configuration uses one library. The 30B full arm has one run. The 30B and 7B comparison baselines use the think condition. No matched no-think variant exists for our method, and baseline-only no-think results vary by method, so we draw no cross-condition conclusion. A 19-operator reference library, mined by simulator replay of the same even-indexed training episodes without LLM proposal, further cautions against attributing every score to induction. On single-family OOD we use its per-fold versions, which withhold the held-out family. It outperforms the learned eight-operator library on OOD at all three scales (72B 0.656 vs.\ 0.542, 30B 0.460 vs.\ 0.354, 7B 0.145 vs.\ 0.117) and on 7B ID (0.160 vs.\ 0.139), whereas the learned library leads on 72B ID (0.780 vs.\ 0.674) and 30B ID (0.574 vs.\ 0.510).
```

### 15. [hr-leak（新 TODO）] 原稿第 555 行之后插入

hr_failure 附录开头加 TODO 注释：单 agent 和 N=3 的 H_R run 用的是哪个记忆，盘上没有记录。

**插入位置：紧接在这一行之后**

```latex
\label{sec:hr_failure_appendix}
```

**插入内容**

```latex
% TODO[hr-leak] Which memory the one-agent and N=3 H_R runs retrieve from is not on disk
% (Isambard outputs). If it is the H_R memory, say so here; the protocol paragraph
% already states that such runs reuse seen episodes.
```

### 16. [retrieval-impl-gt（新 TODO）] 原稿第 625 行之后插入

分类器验证附录加 TODO 注释：代码里测试时没有路由，classifier vs GT 那张表要说清实际变的是什么，否则删掉。

**插入位置：紧接在这一行之后**

```latex
We compare proposition-based task-type predictions with ground-truth labels on 2{,}000 episodes from the validated PARTNR training set~\citep{PARTNR}. Figure~\ref{fig:task_classification} reports per-type F1. Elementary types (R, S, H, T) average 0.97, and composite types (R\_S, H\_R, H\_R\_T, H\_R\_S\_T) average 0.90. H\_R\_S\_T is lower and contains only five episodes. Overall exact-match accuracy is 90.9\%, supporting the use of predicted types for routing while leaving measurable classifier headroom.
```

**插入内容**

```latex
% TODO[retrieval-impl-gt] No test-time routing exists in the code: the bank is fixed
% per run. Say what the classifier-vs-GT runs below actually varied, or remove them.
```

### 17. [retrieval-impl-fallback（新 TODO）] 原稿第 826 行之后插入

fallback 率段加 TODO 注释：「post-boost」中位数 1.162、1.226 在没有 boost 的情况下不可能出现。

**插入位置：紧接在这一行之后**

```latex
To quantify how often the fallback path activates during evaluation, we instrument the retrieval pipeline and log every retrieval call in our main experiments. Across $8{,}855$ retrieval calls under the matched-memory configurations used for the main results, the fallback to retrieval-free generation triggered in $0$ cases ($0.00\%$ fallback rate, equivalently $100\%$ memory utilization). Composite similarity scores remained well above the deployment threshold $\theta=0.3$ across all task types, with median post-boost scores of $0.836$ on R\_S, $0.813$ on H\_R, $1.162$ on H\_R\_T, and $1.226$ on H\_R\_S\_T. The respective minima were $0.488$, $0.343$, $0.845$, and $0.580$, all above $\theta=0.3$. On a broader evaluation set spanning more diverse memory configurations ($13{,}795$ retrieval calls), the overall fallback rate remained at $0.55\%$, concentrating on the most challenging 4-constraint H\_R\_S\_T tasks ($2.65\%$, $27/1{,}020$) where library coverage is most stretched, while staying at $0\%$ on the H\_R and H\_R\_T groups.
```

**插入内容**

```latex
% TODO[retrieval-impl-fallback] "post-boost" medians of 1.162 and 1.226 are impossible
% without a boost; the code has none and the highest logged score is 0.965.
% Source of these medians not found (Isambard outputs). Fix or remove.
```

### 18. [retrieval-impl + effect-match] 原稿第 942–944 行

附录检索示例：删掉例 (1)（boost 不存在）和例 (3)（日志里没有这类激活），只保留例 (2) 并去掉编号；删掉 TODO[effect-match]。

**原文**

```latex
To complement the aggregate statistics, we extract concrete retrieval cases from evaluation logs that illustrate how the hierarchical retrieval mechanism operates in practice. \textbf{(1) Task-aware boosting.} For the instruction ``move kettle from living room to kitchen, fill and turn on,'' the retrieval system returns \texttt{kettle\_operation} with combined score 1.117 (cosine similarity 0.967 plus object-type boost 0.150 from matching the ``kettle'' token). Lexical and contextual signals supplement raw embedding distance, providing discriminative power when multiple skills have similar descriptions but apply to different object classes. \textbf{(2) Decomposition into complementary skills.} For multi-object instructions such as ``move jug, kettle, and teapot to kitchen,'' top-$k$ retrieval ($k{=}5$) surfaces three complementary individual skills covering each object class plus cooperation skills capturing the handover patterns required for joint transport, rather than returning redundant variants of a single skill. \textbf{(3) Effect-based cooperation skill activation.} When the observation difference $\Delta o_t^j$ matches a stored \texttt{partner\_cond} pattern (e.g., ``object no longer on table, partner approaching kitchen''), the retrieval mechanism elevates handover-related cooperation skills above individual skills with comparable embedding similarity. Cooperation skill retrieval is thus gated by partner-induced environmental changes rather than instruction-level lexical similarity alone, realizing the effect-based partner inference mechanism described in Section~\ref{subsec:effect_inference}.
% TODO[effect-match] Example (3) has the same status as Section 4.2. Keep it
% only if the logs of the reported runs contain such an activation.
```

**替换为**

```latex
To complement the aggregate statistics, we describe a retrieval case that illustrates how the hierarchical retrieval mechanism operates in practice. \textbf{Decomposition into complementary skills.} For multi-object instructions such as ``move jug, kettle, and teapot to kitchen,'' top-$k$ retrieval ($k{=}5$) surfaces three complementary individual skills covering each object class plus cooperation skills capturing the handover patterns required for joint transport, rather than returning redundant variants of a single skill.
```

## 三、还没做完、需要作者或改稿 agent 处理的

### 3.1 本轮新加的 TODO（超出授权，只标注、没有改写）

| tag | 位置 | 要做的事 |
|---|---|---|
| `cells-fig` | 表示消融图 | `figs/viki_representation_ablation.pdf` 里 30B CG 四格还标着 not run，按摘要第 4 条的数重画 |
| `retrieval-impl-gt` | 分类器验证附录（Classifier headroom 段与 GT 路由表） | 代码里没有测试时路由。要么说清这组 run 实际变的是什么（例如按预测类型还是按真值类型给每组配库），要么删掉这段和表 |
| `retrieval-impl-fallback` | fallback 率段 | 「median post-boost scores」里的 1.162、1.226 不可能出现。要么从 Isambard 原始输出重算原始 cosine 中位数，要么删掉这几个数 |
| `hr-leak` | hr_failure 附录 | 查清单 agent 和 N=3 的 H_R run 用哪个记忆。如果是 H_R 记忆，在这里写明是已见 episode 上的复用 |

### 3.2 原稿里原样保留的 TODO（缺输入，无法完成）

| tag | 卡在哪 |
|---|---|
| `effect-match`（第 213、836 行附近） | 等 `partnr_effect_schema_spec.md` 与五臂结果，两样都不存在。0.43 ms 等 latency 数没有埋点，来源找不到 |
| `gmemory-run` | PARTNR 的 G-Memory 代码、配置、日志都不在盘上 |
| `table1-ci` | Table 1 逐 episode 输出在 Isambard（`/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner/outputs/habitat_llm/`），没拷回来；`wilson_ci.py` 也不存在 |
| `hr-main` | 同上缺输出；而且每个方法的 H_R 记忆都含 H_R 评测 episode，补跑也只能算已见 episode 上的复用 |
| `skill-call-baseline` | 需要新跑 72B 的 zero-shot / G-Memory skill-call 臂（B1 的 P/G 臂，约 2,442 次 call），没开 |
| `topk-40` | 没有任何 k=10/20 的运行；40.0% = 2/5，15.4% 的分母是 13，都凑不出来 |

### 3.3 没有 TODO、但发现有问题的地方

- **Table 2 H_R_T 行**：除了 MEMENTO 重跑格，27.7、19.8、9.2 都不是 k/11，说明这一行混用了不同 episode 数，和 H_R_S_T 行同病。要么从 Isambard 输出查清各格的 n 并在图注写明，要么把这一行也标为描述性。
- **附录检索示例 (2)**：日志里同样找不到这个例子。它不在授权范围内，没动。
- **VIKI CG 的 task_id**：297 行 CG 只有 295 个不同的 task_id（`4307_8-1`、`1426_9-1` 各重复一次）。写不写进稿件由作者定。
- **判分口径对照表**：`tables/scorer_parity.tex` 可直接放进附录，用来回应「混合口径」的质疑。

## 四、证据文件

| 内容 | 路径（partnr-planner 仓库内） |
|---|---|
| 完整报告 / 结构化答案 / diff / B 层调用记录 | `results/iclr_todo_2026-09-21/{report.md,answers.json,main_patch.diff,calls_log.jsonl}` |
| 打完 patch 的 main.tex / 原稿 | `results/iclr_todo_2026-09-21/src/{main.tex,main.orig.tex}` |
| MEMENTO H_R_T 重跑逐 episode | `results/iclr_todo_2026-09-21/b2/episode_result_log.csv` |
| 逐折参考库配对 | 远端 `outputs/viki_ablation/v3_libfold_fold_{72B,30B,7B}.json` |
| 参考库 15 格 / 判分口径对照 | `results/iclr_todo_2026-09-21/tables/` |
| 打 patch 的脚本 | `scripts/iclr_todo/patch_decisions_0921.py`、`patch_b2_0921.py` |
