# A 部分全部结果（VIKI-L2 公平性补齐与零调用清理）

2026-09-22 汇总。本文件由 A/ 下各交付物拼合，数字以 answers.json（163 个数，带分子/分母/路径）为准。全程零 LLM call。

目录：1 报告正文　2 正文与附录 patch（P1–P5）　3 参考库 patch（留档，用户定不进论文）　4 A3 实际模板原文　5 交付物清单

---

# 1 报告正文


2026-09-22。A1、A4 完成；A2 数字已算，但用户定参考库不作为对比、不进论文（对比只对 LLM 方法），patch 不合入；A3 在 `A3/`（另报）。`calls_log.jsonl` 为空。
口径核对：归档首轮答案经同库同 planner 重规划，15 个主表格逐行复现，0 行不一致（`replay/replay_*_ours_full.json`）。

## DISCREPANCIES

1. A2 预期值（72B 参考库 ID 86.4、OOD 60.3、CG w/ Image 70.4、CG w/o Image 73.7）在盘上无来源，对应分子 798/924、557/924、209/297、219/297 均搜不到。盘上同口径（首轮、同 planner、strict SOLVED）值为 ID 67.4（623/924）、OOD 65.6（606/924，逐折库）、CG w/ Image 79.8（237/297）、CG w/o Image 86.2（256/297），与 `results/viki_memory_experiments/amendment11/appendix_reference_72B_*.jsonl` 及 `outputs/viki_ablation/v3_lib*_72B.json` 一致。A2 标 HALTED，数未改。
2. 「手写」前提不成立：`amendment11/skill_memory_v2.json` 的 `built_from` 为 train 偶数行，由模拟器执行训练 episode 挖出。

## A1 re-ask（TODO[reask]）

- 触发行数：72B ID 46/924（5.0%）、OOD 266、sibling 266、CG 0/0；30B ID 71（7.7%）、OOD 268、sibling 267、CG 0/0；7B ID 135（14.6%）、OOD 244、sibling 244、CG w/o Image 1、CG w/ Image 0。
- 每行最多一次：`scripts/viki_eval_v2_intent_choice.py:319` 判定触发，`:335` 为唯一一次再问。
- 基线无第二轮：`viki_amendment8b.py:630`（ID/OOD 的 zero-shot、G-Memory、trajectory RAG、skill memory v1）、`viki_amendment10_run.py:178`（CG）、`viki_tom_arm.py:486`（ToM）、MEMENTO `viki_eval_memento.py:130` 抽取 + `:162` 出计划。G-Memory 出计划前两次相关性打分（`viki_amendment8_memory.py:129`）属记忆侧。
- 逐行差异：由败转胜 14 行，全在 30B ID（57.4 对 55.8）；由胜转败 0 行（结构性，第二轮只在首答规划失败时触发）。X = 1.5 个点。
- patch：`main_patch.md` P1–P3，正文净增 +3 行。

## A2 参考库对照（用户定：参考库不作对比、不进论文；数字仅留档）

成功数，本方法 vs 参考库（本方法胜/参考库胜，exact McNemar），见 `a2_per_row.csv`：

| 模型 | ID | OOD | CG w/ Image | CG w/o Image |
|---|---|---|---|---|
| 72B | 721 vs 623（115/17，p=4.6e-19） | 501 vs 606（0/105） | 逐行相同 | 逐行相同 |
| 30B | 516 vs 471（92/47，p=1.7e-4） | 327 vs 425（0/98） | 193 vs 196（0/3，p=.25） | 215 vs 218（0/3，p=.25） |
| 7B | 128 vs 148（10/30，p=.002） | 108 vs 134（0/26） | 45 vs 47（0/2，p=.5） | 76 vs 78（0/2，p=.5） |

sibling-group（零 call 补算，首轮对首轮）：

| 模型 | 本方法 | 参考库 | 本方法胜/参考库胜 | p |
|---|---|---|---|---|
| 72B | 295/924（31.9） | 400/924（43.3） | 0/105 | 4.9e-32 |
| 30B | 147/924（15.9） | 244/924（26.4） | 0/97 | 1.3e-29 |
| 7B | 34/924（3.7） | 44/924（4.8） | 0/10 | 0.0020 |

来源 `replay/replay_<model>_heldout_sibgrp_ref_first.json`。构建：`our_method/skill_memory_v2/build.py` 未改，包装脚本 `scripts/iclr_master/A/a2_build_ref_sibgrp.py` 调同样三层函数（seed、per_family=250 不变），跳过整个兄弟组；等价性核对为只排除 `cut_fruit_on_board` 时与归档 `amendment11/skill_memory_v2.fold_cut_fruit_on_board.json` 逐字段相同（仅缺 `self_check` 字段）。新库 `ref_sibgrp/skill_memory_v2.foldgrp_cut.json`（17 算子）、`foldgrp_parallel.json`（19 算子），其余五族沿用单族逐折库。

预写方向只有 OOD 半句成立（单族与兄弟组两个 OOD 都低于参考库）；`main_patch_reference_library.md` 按实测方向改写（below it on both OOD splits）并把 hand-written 改为 mined，未合入。

## A4 CG 组合曲线

CG 只有一种组合（297 行全是 `recombine_cut_and_deliver`），不存在按测试任务组合深度分层的结果（该维度 NOT_FOUND）。盘上有按库来源族分层的三点：

| 模型 | split | comp(cut) 2 族 4 算子 | comp(cut+delivery) 3 族 4 算子 | comp(all) 14 族 8 算子 |
|---|---|---|---|---|
| 72B | CG w/o Image | 232 | 232 | 256（+24/−0，p=1.2e-7） |
| 30B | CG w/o Image | 187 | 187 | 215（+28/−0） |
| 30B | CG w/ Image | 159 | 159 | 193（+34/−0） |
| 7B | CG w/o Image | 79 | 78 | 76（不显著） |

附录段草稿：`main_patch.md` P4。

## 需要用户决定

1. pass10 `main.tex` 放到盘上后生成 `main_patch.diff`（P1–P5）。
2. A3 待核：模板 yaml 的 similarity_threshold 为 0.7，稿件写 θ=0.3。

## Future（不跑）

无。

脚本（未提交）：`scripts/iclr_master/A/a_replay.py`、`a_report.py`。

## A3 PARTNR 的 prompt 列表（TODO[ours-prompt]）

- 出数时实际用的模板：B 仓库 `habitat_llm/conf/instruct/rag_prompt_sequential_cooperation_skills_v4.yaml`（sha 5bebc20d…，未进 git），副本 `A3/actual_template_rag_prompt_sequential_cooperation_skills_v4.yaml`；pass6 的 Listing 原文在 `A3/listing_lst_ours_pass6.txt`。
- 与 Listing 的差别：B 仓库任何 instruct yaml 里都没有 Listing `lst:ours` 的文字。Listing 里的四步 Decision Process（Diff、Skill Prediction、Matching、Action）、Rules 节、`=====` 分隔、`[Diff][Skill][Ref][Plan]` 输出格式，v4 一样都没有。v4 只有一句 system 说明、动作格式与示例、若干槽位，以及 "Thought: <brief reason> / Action[parameter] / Assigned!"。
- prompt 每个 agent 每个 episode 只构建一次，之后每轮只追加动作结果。槽位怎么填：
  - `rag_examples`：开局检索一次，top-5。每条带 skill 名、类型、相关度、trace、objects、action pattern；只要有 cooperation skill 就加一句 ToM。
  - `state_comparison`：始终是写死的 "This is your first observation…"。
  - `world_description`：只填一次，从不刷新。
  - `agent_role_description`：固定文字，uid 1 是任务发出方，uid 0 是接收方。
- patch：Listing 换成 v4 原文，Method 的 Instance filter and action selection 段按上面几点核对（待 pass10 `main.tex` 出 diff）。

---

# 2 正文与附录 patch（main_patch.md）

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

---

# 3 参考库 patch（main_patch_reference_library.md，留档，不合入）

# A2 参考库 patch（单独成文件，合不合入由用户决定）

**状态：HALTED。** 原因有两条，都列在 report.md 的 DISCREPANCIES：
1. spec 预期的 72B 数值（ID 86.4、OOD 60.3、CG w/ Image 70.4、CG w/o Image 73.7）在盘上找不到来源。同一口径下盘上的值是 67.4 / 65.6 / 79.8 / 86.2。
2. 这个库不是手写的。`results/viki_memory_experiments/amendment11/skill_memory_v2.json` 写着 `built_from: VIKI-L2 train.parquet, even-indexed episodes`，是模拟器重放训练 episode 挖出来的，没有经过 LLM 提议。所以 "hand-written" 这个词不能用。

下面的草稿用盘上的数，措辞按实测方向写。**没有用户确认不合入。**

pass10 不在盘上，这里只给插入句，不给 diff。

---

## R1  RQ1 → "What the VIKI-L2 gap measures" 段末尾

spec 预写句的第二句是 "below it on ID and OOD and above it on both CG splits"。实测只有 OOD 这半句成立：ID 上我们更高，两个 CG split 上逐行完全相同。按 spec §1 第 5 条，只改和数据矛盾的部分：

```latex
A 19-operator library mined by simulator execution of the same training episodes, used with the same answers and planner, reaches 67.4/65.6/43.3/79.8/86.2\% on ID, OOD, sibling-group OOD, CG w/ Image, and CG w/o Image at 72B. The learned library of eight operators is above it on ID, below it on both OOD splits, and solves the same rows on both CG splits.
```

净增行数：+3。兄弟组的参考库由 `scripts/iclr_master/A/a2_build_ref_sibgrp.py` 零 call 重建（同一构建函数，排除单位从单族换成兄弟组）。

文风自检：没有破折号、冒号、分号，没有加粗，没有 replay 这个词（原来的 "simulator replay" 改成了 "simulator execution"）。

## R2  附录表

```latex
\begin{table}[h]
\centering
\small
\caption{Learned library and 19-operator mined library under the same first answers and planner. Success in \%, with exact McNemar counts (learned only / mined only). OOD sib.\ withholds the whole sibling group from both libraries.}
\label{tab:viki_mined_library}
\begin{tabular}{llrrrr}
\toprule
Model & Split & Learned & Mined & Counts & $p$ \\
\midrule
72B & ID           & 78.0 & 67.4 & 115/17 & $<10^{-18}$ \\
72B & OOD          & 54.2 & 65.6 & 0/105  & $<10^{-31}$ \\
72B & OOD sib.\     & 31.9 & 43.3 & 0/105  & $<10^{-31}$ \\
72B & CG w/ Image  & 79.8 & 79.8 & 0/0    & 1 \\
72B & CG w/o Image & 86.2 & 86.2 & 0/0    & 1 \\
30B & ID           & 55.8 & 51.0 & 92/47  & $2\times10^{-4}$ \\
30B & OOD          & 35.4 & 46.0 & 0/98   & $<10^{-29}$ \\
30B & OOD sib.\     & 15.9 & 26.4 & 0/97   & $<10^{-28}$ \\
30B & CG w/ Image  & 65.0 & 66.0 & 0/3    & 0.25 \\
30B & CG w/o Image & 72.4 & 73.4 & 0/3    & 0.25 \\
7B  & ID           & 13.9 & 16.0 & 10/30  & 0.002 \\
7B  & OOD          & 11.7 & 14.5 & 0/26   & $3\times10^{-8}$ \\
7B  & OOD sib.\     & 3.7  & 4.8  & 0/10   & 0.002 \\
7B  & CG w/ Image  & 15.2 & 15.8 & 0/2    & 0.5 \\
7B  & CG w/o Image & 25.6 & 26.3 & 0/2    & 0.5 \\
\bottomrule
\end{tabular}
\end{table}
```

"Learned" 这一列是 first-turn-only（spec A2.2 要求同一口径）。只有 30B ID 和主表不同：主表带第二轮是 57.4，对 Mined 是 106/47，p = 2.1e-6。

---

# 4 A3：Table 1 出数时实际用的本方法模板（rag_prompt_sequential_cooperation_skills_v4.yaml）

```yaml
prompt: |-
    {system_tag}You are an agent solving rearrangement tasks with another agent. You take turns acting.

    {agent_role_description}

    Action format: ActionName[parameter]
    Examples: Navigate[kitchen_1], Pick[cup_0], Place[cup_0, on, counter_29, None, None], Explore[living_room_1], Done[]

    {eot_tag}{rag_examples}{user_tag}Task: {input}

    {world_description}

    {state_comparison}

    Actions: {tool_descriptions}

    Respond with:
    Thought: <brief reason>
    Action[parameter]
    Assigned!{eot_tag}{assistant_tag}

stopword       : "Assigned!"
end_expression : "Done[]"

actions_parser:
  _target_     : habitat_llm.llm.instruct.utils.zero_shot_action_parser
  _partial_    : true

rag_retrieval:
  _target_     : habitat_llm.llm.instruct.utils.rag_retriever
  _partial_    : true
  top_k: 5
  similarity_threshold: 0.7
  trajectory_path: "data/rag_datasets/cooperation_skills_org_re_sp/react_trajectories"
  retrieval_strategy: "task_similarity"
```

附录 Listing lst:ours（pass6 原文）见 `A3/listing_lst_ours_pass6.txt`。

---

# 5 交付物清单

| 文件 | 内容 |
|---|---|
| `answers.json` | 全部数字，每个带 value/numerator/denominator/source_path/computation |
| `a1_per_row.csv` | A1 逐行：full 与 first-turn-only |
| `a2_per_row.csv` | A2 逐行：本方法 vs 参考库（含 sibling-group） |
| `a4_per_row.csv` | A4 逐行：comp(cut)/comp(cut+delivery)/comp(all) |
| `tables.json` | 各表汇总 |
| `replay/` | 零 call 重放结果（含口径复现 ours_full、first-turn、参考库） |
| `ref_sibgrp/` | 兄弟组排除的参考库（17、19 算子） |
| `config_diff.json` | 配置差异 |
| `calls_log.jsonl` | 空（0 次 call） |
| 脚本 | `scripts/iclr_master/A/{a_replay.py,a_report.py,a2_build_ref_sibgrp.py}`（未提交） |
