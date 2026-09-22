# A 部分报告：VIKI-L2 公平性补齐（零 LLM call）

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
2. A3 已核（09-22 Part B0）：θ=0.3 写死在 `rag.py:468`，模板 yaml 的 0.7 没有代码读取，稿件的 θ=0.3 正确。

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
