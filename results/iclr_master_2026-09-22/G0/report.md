# G0 报告：PARTNR 环境与产物盘点

2026-09-22。由 G0 子代理的回报整理落盘（子代理无法写 .md）。逐日志 sha 见 `isambard_logs_per_log.csv`。

## DISCREPANCIES

1. Table 1 Llama-8B Ours 0.722 与任何日志都对不上；盘上四份部分日志分别为 0.538、0.724、0.746、0.708。
2. Table 2 R+S 行 MEMENTO 稿件为 79.1；日志 168088 给出 58/76 = 76.3，恰是稿件的 RAG 值。
3. Llama-8B MEMENTO R-only 稿件为 0.717；日志 168086 给出 56/74 = 0.757。
4. 附录 Listing `lst:ours` 不是出数时用的模板（见 A3）。
5. 协议问题：R_S 冒烟 ep 153 在 step 8,006 已到 task_complete 100%，env 结束后 planner 继续 step，得到 "Episode over, call reset"，该 episode 被丢弃，metrics 与 planner-log 都没写出。归档四份 Llama-8B Ours R-only 日志分别只覆盖 65、58、67、48 个 episode（共 76 个），稿件的数也不是 1/76 的倍数。推断原 Table 1 只平均了没崩的 episode。处理：加 env_over 开关，按 env 结束时状态记分，分母统一为全部 episode；同时报原口径作对照。

## 1. 冒烟（G0.1）

Llama-3.1-8B zero-shot，2 个 R_S + 2 个 H_R：
- R_S ep 25：13,394 sim 步，success 0。ep 153：env_over（见上）。
- H_R ep 436：6,139 步，pc 0.667。ep 433：CUDA OOM（与 B 队列同卡并发），台子失败。

流水线可跑。

## 2. 产物（G0.2）

找到（本地 `~/restore_isambard/partnr-planner`，已同步到远端 `partnr-isambard/data`）：

| 产物 | 内容 |
|---|---|
| R 库 `hierarchical_rerange_only` | 223 individual + 227 cooperation = 450，tree sha 903a57b9… |
| S 库 `spatial_only` | 14 + 6 = 20 |
| H_R 库 `heterogeneous_rerange` | 280 + 175 = 455，由 196 个 episode 建成 |
| H_T 库 `heterogeneous_temporal` | 7 + 5 = 12 |
| H_R_S_T 库 | 20 + 12 = 32 |
| 基线记忆 | 5 份 MEMENTO memory，5 份 rag_datasets |
| 建库代码与日志 | `our_method/build_hierarchical_skill_memory.py`，`logs/build_memory_*` |
| episode 集 | R_S 76（`rerange+spatial_matched_subtasks`），H_R_T 11，H_R_S_T 5，H_R 197 |
| 出数 prompt 模板 | `habitat_llm/conf/instruct/rag_prompt_sequential_cooperation_skills_v4.yaml`（sha 5bebc20d…） |

只有部分日志：`isambard_logs_per_{log,episode}.csv`。24 个 Table 1 格没有一个有完整的 76 episode × 3 seed 运行，全部重跑。

NOT_FOUND：
- Table 1/2 逐 episode 结果；
- 全部 zero-shot、ToM、G-Memory 运行；
- `tab:hr_failure` 三个条件的配置；
- 70B、72B、Claude、GPT 的配置。

## 3. TODO[gmemory-run]

B 流水线里没有 G-Memory 实现。PARTNR 上唯一的 G-Memory 移植在 A 仓库 `our_method/partnr_baselines/gmemory.py`：为 09-03 的组合泛化对照写的，图加 insights，来自 R-only 轨迹，train_mini/val_mini。它不是 Table 1 背后那次运行。Table 1/2 的 G-Memory 列标 BLOCKED。

## 4. TODO[hr-leak]

一方用记忆那两个条件的配置 NOT_FOUND。盘上 Both-memory 的 H_R 日志检索的是 `hierarchical_heterogeneous_rerange`，即 H_R memory，覆盖 197 个 H_R episode 里的 196 个，所以那批运行跑在见过的 episode 上。C.2 改用 Table 1 的 Llama-8B 配置。

## 5. 需要用户交接

- Isambard `/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner/outputs/habitat_llm/`（Table 1/2 与 hr_failure 的原始输出）
- Qwen 的 Table 1 配置
- G-Memory 的 PARTNR 配置与代码
- Llama-70B（原用 OpenRouter `llama-3.3-70b-instruct:free`）与 Qwen2.5-72B 的运行方式：本机无权重，4–7 号卡被别人占用
- pass10 `main.tex`（`partnr_meanstd_reference.tex` 与汇总脚本 `partnr_mean_std.py` 已在 `~/Downloads/mas_iclr_rev.zip` 的 `reference/` 里找到）
