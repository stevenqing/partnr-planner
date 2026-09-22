# A 部分 main.tex 改动清单（A1、A4）

pass10 `main.tex` 不在盘上，`main_patch.diff` 暂缺。下面每条给出定位、原句、替换句。等 pass10 放到盘上后再生成 diff。
所有数字出自 `answers.json` 的 `A1.*` 与 `A4.*` 条目。

文风自检（spec §1 第 7 条），每一条都查过：没有破折号，句中没有冒号或分号，现在时，没有 in which，没有加粗或斜体，没有比喻，也没有 audit、provenance、preregistered、row-level、replay 这些词。`\ref{sec:viki_audit}` 是稿件里已有的 label，不属于正文用词。

---

## P1  Method → "Planner use for complete plans" 段（A1，删除 `TODO[reask]`）

定位：该段里写着 72B ID 46/924 的那一句（pass10 原文不在盘上，按 `46/924` 查找）。

替换为：

```latex
When the robot assignment in the answer admits no plan, the planner reports the reason and asks the model once more, and the row is unsolved if the second answer also fails. This second turn occurs on 46 of 924 ID rows at 72B and adds no solved row there. Across the 15 model and split cells it adds 14 solved rows, all on 30B ID.
```

净增行数：+1（原句约 1 行，新句约 2 行，按两栏排版估算）。

如果该句附近有 `% TODO[reask]` 注释，一并删除。

---

## P2  Setup → "VIKI-L2 comparison and ablations" 段末尾（A1 第 3 点：基线没有第二轮）

插入：

```latex
Comparison baselines answer once. Without the re-asked turn our success changes by at most 1.5 points in any cell (Appendix~\ref{sec:viki_audit}).
```

净增行数：+2。

说明（不进正文）：G-Memory 在出计划前会做两次相关性打分调用，MEMENTO 会做一次抽取调用。但出计划的调用都只有一次，没有任何基线拿到失败反馈后再答一次（代码行号见 report.md 的 A1.3）。所以 "answer once" 指的是计划只写一次，这句话是成立的。

---

## P3  附录 `sec:viki_audit` 加表（A1）

```latex
\begin{table}[h]
\centering
\small
\caption{Second turn on VIKI-L2. Share of rows on which the model is asked a second time, success with both turns, and success with the first answer only. All three use the same answers, library, and planner.}
\label{tab:viki_reask}
\begin{tabular}{llrrr}
\toprule
Model & Split & Second turn (\%) & Success (\%) & First answer only (\%) \\
\midrule
72B & ID            & 5.0  & 78.0 & 78.0 \\
72B & OOD           & 28.8 & 54.2 & 54.2 \\
72B & OOD sibling   & 28.8 & 31.9 & 31.9 \\
72B & CG w/ Image   & 0.0  & 79.8 & 79.8 \\
72B & CG w/o Image  & 0.0  & 86.2 & 86.2 \\
30B & ID            & 7.7  & 57.4 & 55.8 \\
30B & OOD           & 29.0 & 35.4 & 35.4 \\
30B & OOD sibling   & 28.9 & 15.9 & 15.9 \\
30B & CG w/ Image   & 0.0  & 65.0 & 65.0 \\
30B & CG w/o Image  & 0.0  & 72.4 & 72.4 \\
7B  & ID            & 14.6 & 13.9 & 13.9 \\
7B  & OOD           & 26.4 & 11.7 & 11.7 \\
7B  & OOD sibling   & 26.4 & 3.7  & 3.7  \\
7B  & CG w/ Image   & 0.0  & 15.2 & 15.2 \\
7B  & CG w/o Image  & 0.3  & 25.6 & 25.6 \\
\bottomrule
\end{tabular}
\end{table}
```

表下一句：

```latex
The second turn never turns a solved row into an unsolved one, because it is asked only when the first answer yields no plan.
```

净增行数：正文 0，附录约 +22。

---

## P4  附录，CG 组合曲线（A4）

盘上有按库的来源族分层的结果，但 CG 测试集只有一种组合（297 行全是 `recombine_cut_and_deliver`），所以没有按测试组合深度分层的结果。这里的曲线是「库从几个族归纳」，不是「测试任务组合了几个技能」。附录段落：

```latex
\paragraph{Library coverage on CG.} We rebuild the library from subsets of the training families and evaluate each on the 297 CG rows with the same answers and planner. A library from the two cutting families has four operators and reaches 78.1\% (72B), 63.0\% (30B), and 26.6\% (7B) on CG w/o Image. Adding the single delivery family adds no operator and leaves success within one row of these values. The library from all 14 families has eight operators and reaches 86.2\%, 72.4\%, and 25.6\%. The gain from the full library is 24 rows at 72B and 28 rows at 30B with no row lost ($p<10^{-6}$, exact McNemar), and at 7B it is two rows lower ($p=0.5$). On CG w/ Image the full library adds 34 rows at 30B with no row lost ($p<10^{-9}$) and changes at most one row at 72B and 7B.
```

核对：72B 232/297 = 78.1，30B 187/297 = 63.0，7B 79/297 = 26.6；full 库 256/215/76。72B 文本 24 胜 0 负，p = 1.2e-7；30B 文本 28 胜 0 负，p = 7.5e-9；7B 文本 comp(cut+delivery) 到 comp(all) 是 0 胜 2 负，p = 0.5（comp(cut) 79 → comp(all) 76）。带图：30B 34 胜 0 负，p = 1.2e-10；72B comp(cut) 到 comp(cut+delivery) 1 胜 0 负，之后 0/0；7B 三点都是 45。

注意 7B 那半句：成对检验是 comp(cut+delivery) 对 comp(all)，78 → 76，0 胜 2 负。和 comp(cut) 的 79 比，差 3 行。正文写成「two rows lower」，指的是相邻两点的差。

净增行数：正文 0，附录约 +8。

## P5  附录 Listing `lst:ours` 换成出数时实际用的模板（A3，删除 `TODO[ours-prompt]`）

查找：Listing `lst:ours` 的全部正文（pass6 原文见 `A3/listing_lst_ours_pass6.txt`，包括四步 Decision Process、Rules 节、`=====` 分隔、`[Diff][Skill][Ref][Plan]` 输出格式）。

替换为（`rag_prompt_sequential_cooperation_skills_v4.yaml` 的 prompt 字段原文，去掉 chat tag 占位）：

```
You are an agent solving rearrangement tasks with another agent. You take turns acting.

{agent_role_description}

Action format: ActionName[parameter]
Examples: Navigate[kitchen_1], Pick[cup_0], Place[cup_0, on, counter_29, None, None], Explore[living_room_1], Done[]

{rag_examples}
Task: {input}

{world_description}

{state_comparison}

Actions: {tool_descriptions}

Respond with:
Thought: <brief reason>
Action[parameter]
Assigned!
```

caption 补一句：The prompt is built once per agent at the start of an episode, and each later turn appends the result of the previous action. `{rag_examples}` holds the top five retrieved demonstrations, `{state_comparison}` is a fixed first-observation sentence, and `{world_description}` is filled once.

Method 的 Instance filter and action selection 段按以下事实核对（改法等 pass10 原文）：每个 agent 每个 episode 开局检索一次，query 由模板拼成，不经过 LLM；之后每轮不重新检索、`state_comparison` 不刷新，只追加动作结果。

正文净增：0 行（附录内替换）。附录净变化约 −25 行（Listing 由 59 行缩到约 20 行）。

待核：yaml 里 `similarity_threshold: 0.7`，稿件写的是 θ=0.3。需确认出数运行是否在命令行覆盖了这个值（Isambard 原始命令不在盘上）。
