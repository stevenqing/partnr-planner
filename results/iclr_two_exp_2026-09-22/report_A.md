# Part A 报告：replay-mined 算子的同 planner 对照

2026-09-22。由 Part A 子代理的回报整理落盘（子代理写不了报告文件）。偏差见 `deviations.md` 的 Part A 节，以及 `A/deviations_A.md`。

## 结局：Ending A1

72B 的两个 CG 格上，replay-mined 与 ours 都是 237/297（79.8%）和 256/297（86.2%），逐行完全相同，差 0 个点。

## A0 盘点（零调用）：Gate A0 通过

- 12 个首轮格都在 `results/viki_memory_experiments/amendment11/`（`v3_ours_<M>_{id,heldout}.jsonl`、`v3_oursall_<M>_{imaged,text}.jsonl`），行数 924/924/297/297 全部对得上。
- Figure 3 即 `fig:viki_rq2`，盘上没有它的画图脚本。它背后的登记行文件是 `results/paper_viki_iclr2027/rows/rq2/full/*`，12 个文件与 `SHA256SUMS.txt` 全部一致；逐行比较首答，0 行不同、0 行缺失。
- 训练集 7,196 行，偶数位确为 3,598 个 episode，14 个族。
- 各项路径、sha、族×effect 表见 `A/a0_first_turn_cells.csv`、`A/a0_artifacts.csv`、`A/a0_family_effects.csv`、`A/a0_inventory.json`。
- re-ask 每行最多一次：`viki_eval_v2_intent_choice.py` 第 319 行判定触发，第 332 行发出。

## 盘上现有的 19 算子库不是 SPEC 的 replay-mined

逐条对照 `our_method/skill_memory_v2/build.py`：

- 每族最多只用 250 个 episode，不是全部 3,598 个；
- 去重键多含 preconditions；
- 非物体的地点名保持字面，不提升为变量；
- 另外挖了一类 repair 算子；
- 没有经过执行检查和三条准入判据（`self_check` 只是奇数位 episode 上的端到端打分）。

所以它在 72B ID 上 623 < 721，不构成 sanity 失败。本实验另行挖掘。

## A1 挖掘与认证（零 LLM、零手改）

- 挖掘：`scripts/iclr_two_exp/A/mine_replay.py`（sha f68e4700…），沿用 `induction.py` 的重放与切段函数。3,598 个 episode 全部重放成功，挖出 **17** 个算子（13 个 pos.name，4 个 is_activated），存于 `A/replay_library.json`。
- 认证：`certify_replay.py`，与 admitted 库用同一套代码。
  - 执行检查：该族 4 个 holdout episode 中至少 2 个成立。
  - 判据 (i)：边际增益，带排序门。
  - 判据 (iii)：在归纳半前 60 个 episode 上实测 support ≥ 2。
- 结果 **9 个通过**（m0000/1/3/4/5/6/8/10/11）。未通过的：
  - `Interact ?x` 缺 Move，在 holdout 上全部执行失败；
  - 7 个协调变体没有边际增益。
- 列库由原封不动的 `viki_union_library.py` 构建，Layer 2/3 与 admitted 列库逐字节相同。

## A2/A3 结果（strict SOLVED，`tables/replay_vs_ours.csv`）

| 模型 | split | n | ours | replay-mined | ours 独赢 | replay 独赢 | p（exact McNemar） |
|---|---|---|---|---|---|---|---|
| 72B | ID | 924 | 721 | 738 | 0 | 17 | 1.5e-5 |
| 72B | 单族 OOD | 924 | 501 | 501 | 0 | 0 | 1 |
| 72B | CG w/ Image | 297 | 237 | 237 | 0 | 0 | 1 |
| 72B | CG w/o Image | 297 | 256 | 256 | 0 | 0 | 1 |
| 30B | ID | 924 | 530 | 568 | 10 | 48 | 4.5e-7 |
| 30B | 单族 OOD | 924 | 327 | 328 | 0 | 1 | 1 |
| 30B | CG w/ Image | 297 | 193 | 196 | 0 | 3 | 0.25 |
| 30B | CG w/o Image | 297 | 215 | 218 | 0 | 3 | 0.25 |
| 7B | ID | 924 | 128 | 165 | 0 | 37 | 1.5e-11 |
| 7B | 单族 OOD | 924 | 108 | 124 | 0 | 16 | 3.1e-5 |
| 7B | CG w/ Image | 297 | 45 | 47 | 0 | 2 | 0.5 |
| 7B | CG w/o Image | 297 | 76 | 78 | 0 | 2 | 0.5 |

- **Sanity 成立**：72B ID 上 738 ≥ 721。
- ID 上多出的行主要在 `dog_push_box_for_two_panda_transport`（72B：ours 0/46，replay-mined 17/46）。replay 库有一个三角色接力的协调算子，admitted 库没有。逐族结果见 `tables/replay_gap_by_family.csv`。
- 30B ID 上 ours 独赢的 10 行全在 `clear_table`，两臂都走了 re-ask：归档的第二次回答解开了，这次现场重跑的第二次回答超步数或不可行。这是温度 0 并发下的非确定性（见 memory `viki-incontext-library-control`），不是库的差别。
- CG 上两臂选中同一组算子的行数：72B 297/297；30B 283 和 286；7B 262 和 277（`A/tables/cg_same_operator_sequence.csv`）。
- re-ask 模型调用共 952 次（上限 2,000）：72B 295，30B 293，7B 364。触发率与原臂对照见 `A/tables/reask_rates.csv`，例如 72B ID 为 3.1% 对 5.0%，OOD 相同。三个服务都已按 PID 停掉。

## 按 Ending A1 的稿件改动

正文（+2 行，放在 RQ2 或 RQ1 讨论 VIKI 缺口处）：

> Operators mined by replaying the reference plans of the same induction episodes, certified by the same gate and consumed by the same planner, reach 79.8% and 86.2% on CG at 72B, the same rows as the eight admitted operators, so on CG the admitted library matches a library that does not depend on LLM proposal.

预写句是 "within Z points"，Z 实测为 0。这里按 §1 第 5 条只改与数据矛盾的半句，把 "within 0 points" 改写为 "the same rows"，属偏离，待用户确认。

附录：`replay_vs_ours.csv` 作表，加一段写明挖掘程序、mined 17 / certified 9、池子是归纳半（3,598 个偶数位 episode）；全文统一称该臂为 replay-mined，并说明它是对 proposal 的对照、不是上界。附录可再加一句 ID 上的差距来源（dog_push 的三角色协调算子）。

## 脚本（未提交）

`scripts/iclr_two_exp/A/`：`mine_replay.py`、`certify_replay.py` 及重放/打分脚本。
