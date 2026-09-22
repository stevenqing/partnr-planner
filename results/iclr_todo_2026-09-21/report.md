# ICLR 2027 TODO spec, A 层报告（2026-09-21）

A 层全部零 LLM call。B 层只跑了 B2（788 次 planner call，记在 `calls_log.jsonl`）；B1、B3 没动。
稿件是用户贴进来的带 TODO 的版本，存为 `src/main.tex`，patch 基于它，见 `main_patch.diff`。
没找到的输入有两个。`wilson_ci.py` 本地和远端都没有，A6 被 G0 卡住，所以没用上。`partnr_effect_schema_spec.md` 也不存在，五臂结果同样不存在。

## 09-21 拍板之后（用户：1 选一、2 先查 MEMENTO、3 同意、4 全做）

**泄漏范围比下面原文写的窄，Table 1/2 本身是干净的。** 这次按实际配置逐行核对，同时按 episode id 和指令原文两种方式匹配：

| 方法 | 源记忆 → 目标 | 重叠 |
|---|---|---|
| ours | R → R_S（Table 1、Table 2 第 1 行） | 0/76 |
| ours | R+S → R_S | 0/76 |
| ours | H_R+H_T → H_R_T | 0/11 |
| ours | R+S+H_R+H_T → H_R_S_T | 0/5 |
| RAG | 以上各配置实际用的轨迹库 → 对应目标 | 全 0 |
| MEMENTO | 所有记忆 → R_S / H_R_T / 实际用的 H_R_S_T 四库 | 0/76、0/11、0/5 |
| ours / RAG / MEMENTO | H_R 记忆 → H_R | 196/197、196/197、197/197（MEMENTO 的指令原文命中 123/197） |

- HRST 库覆盖 H_R_S_T 4/5 是真的，但 Table 2 的 H_R_S_T 行用的是四个基本库的并集，不是 HRST 库。
- G-Memory 没有代码，查不了。
- MEMENTO 是离线建库，评测时只读，检索没有 episode 过滤（`memento_retriever.py:160-230`、`rag.py:1428-1498`），所以不属于「边评测边积累」的合规协议。
- **B2（MEMENTO H_R_T）不受泄漏影响，可以跑**；B3（H_R）仍然不能跑，因为每个方法的 H_R 记忆都含 H_R 评测 episode。

**已落进 `src/main.tex`**（脚本 `scripts/iclr_todo/patch_decisions_0921.py`，打 patch 之前的版本存为 `src/main.pre_decisions.tex`；`main_patch.diff` 已对 `main.orig.tex` 重新生成）：

1. **泄漏声明**，写在 Evaluation protocol 段：所有 PARTNR 记忆（含 RAG、MEMENTO）都从本任务组的 heuristic 轨迹离线建，检索不排除当前 episode；Table 1/2 的目标 episode 不在其源记忆里；H_R 记忆含 197 个 H_R episode 中的 196 个，所以凡是从它检索的 H_R 结果（附录 hr_failure、N=3）测的是已见 episode 上的复用，不是迁移。
2. **cells**：写成 72 + 36 + 36 = 144，另加 18 个 sibling 格；Pairing 段的 90 改为「72 个主比较格，sibling split 另加 18 格，没跑 ToM，第六臂是我们的早期 skill memory 变体」；图注里「30B 只有 ID/OOD」删掉。
3. **handwritten-7b**：两处都去掉 "hand-written"，改为「用仿真器重放同一批偶数行训练 episode 挖出来的，没有 LLM 提议」（`our_method/skill_memory_v2/induction.py`）。OOD 改用逐折版本 0.656 / 0.460 / 0.145；30B OOD 参考库同样赢我们，原稿漏了这一格，现在写上。
4. **retrieval-impl**：Eq.1 后那两句改为代码事实（库由每次运行的配置固定，测试时没有分类器选库；两级检索都是原始 cosine，没有附加项）；删掉附录例 (1)、路由消融图及其段落，以及 RQ3 里指向它的那句。

**新增的 TODO 注释（授权范围外，只标注、没有改写）**：

| tag | 位置 | 问题 |
|---|---|---|
| `cells-fig` | 表示消融图 | 图 PDF 里 30B CG 四格仍标着「not run」，要重画。四格的数：w/ Image no_grounding 0.024、no_order 0.552（full 0.650）；w/o Image 0.724、0.643（full 0.724） |
| `retrieval-impl-gt` | 分类器验证附录 | 「classifier vs GT routing」表：代码里测试时没有路由，要说清这组 run 实际变的是什么，否则删掉 |
| `retrieval-impl-fallback` | fallback 率段 | 「post-boost」中位数 1.162、1.226 在没有 boost 的情况下不可能出现，来源找不到 |
| `hr-leak` | hr_failure 附录 | 单 agent 和 N=3 的 H_R run 用的是哪个记忆，盘上没有记录 |

## B2 MEMENTO H_R_T：已跑（09-21，用户点头）

- **1/11 = 9.1% 成功，平均完成度 0.393**。配置：
  - 记忆 H_R+H_T，按 Table 2 所写，不含旧 rht 脚本多带的 R。
  - Llama-3.1-8B-Instruct，HF 进程内贪心解码。
  - 旧流水线代码原样，放在远端 `/mnt/pfs/devs/pn5wp/shishuqing/partnr-isambard`。
  - 规模：788 次 planner call，11/11 个 episode 全部真跑，零跳过。
- 逐 episode 结果在 `b2/episode_result_log.csv`，调用记录在 `calls_log.jsonl`。
- 旧 Isambard 的 rht 运行（多带 R 库）也是 1/11、0.414。稿件的 14.5% 仍然对不上任何一次运行。
- 稿件已改（`scripts/iclr_todo/patch_b2_0921.py`）：表里 14.5 → 9.1；正文改为「G-Memory 19.8、RAG 11.1、MEMENTO 9.1」；「earlier run」那句改为如实说明重跑；删掉 `TODO[memento-hrt]`。
- **排名变了**：MEMENTO 从 H_R_T 第三名掉到 RAG 和 ToM 之后。
- **仍然存在的问题**：这一行其它格（27.7、19.8、9.2）也都不是 k/11，说明这一行混用了不同的 episode 数，与 H_R_S_T 那一行同病，原始输出在 Isambard 上。
- 前两次尝试都是台子问题，都在第一次 LLM 调用之前就挂了，零 call，目录留作证据：
  - A 仓库的 transformers-CFG 不认 Llama-3.1 的 tokenizer，改用 B 仓库打过补丁的版本；
  - 离线环境缺 gpt2 tokenizer，已补下载。

## 先看这里：比任何一个 TODO 都重

> **09-21 更正**：下面「Table 1、Table 2 的记忆库从评测 episode 建」这个说法把范围说大了。按实际配置，Table 1/2 的源记忆与目标不相交，见上一节。



**Table 1、Table 2 的 PARTNR 记忆库是从评测 episode 建出来的。**
库的 `episodic_memory` 键和评测集的 episode id 取交集：

| 库 | 评测集 | 重叠 |
|---|---|---|
| HR | H_R | 196 / 197 |
| HRST | H_R_S_T | 4 / 5 |
| R | rerange_only | 578 / 588 |

`run_ours_hr_mem_hr.sh` 就是用 HR 库评 H_R。检索代码（`rag.py`、`hierarchical_retrieval.py`、`llm_planner.py`）里没有任何排除当前 episode 的逻辑。
这件事影响 A7 和 B3：在 H_R 上补跑 ours，等于让它检索测试 episode 本身。**B3 在这个问题解决之前不应该开跑。**

## DISCREPANCIES（稿件值与盘上值不符，相关 tag 标 HALTED，没有改数）

| tag | 稿件 | 盘上 | 路径 |
|---|---|---|---|
| handwritten-7b，以及第 383 行正文 | "hand-written 19-operator library" | 这个库是从 VIKI-L2 train 偶数行归纳出来的（`built_from`），并且带 8 个逐折版本，每个都排除了留出的那一族 | `results/viki_memory_experiments/amendment11/skill_memory_v2.json`、`skill_memory_v2.fold_*.json` |
| 同上 | 72B OOD 67.4% | 用逐折版本是 **65.6%**（606/924）。67.4 用的是见过留出族的完整库 | `work/ref_fold_72B.json`；sanity 用完整库复现 0.6742，见 `work/sanity_ref_full_fold_72B.json` |
| 同上 | 7B "0.160 vs 0.139/0.117" | ID 0.160（148/924）；OOD 用逐折版本是 **0.145**（134/924） | `outputs/viki_ablation/v3_lib_id_7B.json`、`work/ref_fold_7B.json` |
| cells | 32 个 RQ3 格；30B 组合泛化 4 格没跑；总数 140 | RQ3 **36/36**，那 4 格在 2026-09-14 已补跑；总数 **144** | `results/paper_viki_iclr2027/cells.json`、README |
| cells，以及第 412 行图注 | "30B ablation covers ID and OOD only" | 30B 的 RQ3 组合泛化格存在 | 同上 |
| retrieval-impl，附录例 (1) | object-type boost 0.150，综合分 1.117 | 没有任何 boost 代码。5,174 个检索块里最高分是 0.965 | B 仓库 `logs/`；`hierarchical_retrieval.py`（sha 5b864172） |

注：方向没变。逐折参考库仍然在三个模型的单族 OOD 上高于我们：105/0、98/0、26/0 个不一致格，p 都小于 3e-8。
另外我自己 09-18 写进 `RESULTS-2026-09-18.md` §3c 的"参考库不随留出折变化"是错的，同样的 leak。还没改，等你确认口径。

## G0 PARTNR 产物定位

| 项 | 结局 | 路径 / 说明 |
|---|---|---|
| Table 1 六 backbone × 六方法逐 episode | **NOT_FOUND** | 只有 Llama-3.1-8B 的部分日志，在 `~/restore_isambard/partnr-planner/logs/ours_rs_mem_*`、`memento_rs_mem_*`：48–76 个 episode，0.538–0.763，没有一个等于表中值 |
| Table 2 四行迁移 | **NOT_FOUND** | 有部分 HRT/HRST 日志，数都对不上；G-Memory、zero-shot、ToM 没有 |
| 五个 memory bank | **FOUND，计数完全一致** | `~/restore_isambard/partnr-planner/data/hierarchical_skill_memory/`，24 个文件，文件列表 sha `db545157…58b5`<br>技能数 450 / 20 / 455 / 12 / 32<br>建库日志 `all_scripts/slurm_files/logs/rebuild_memory_168228.log` |
| H_R 全方法逐 episode | **NOT_FOUND** | ours 最大 n=50，MEMENTO 55；其余方法没有 |
| G-Memory / MEMENTO 配置与日志 | G-Memory **NOT_FOUND**；MEMENTO 只有部分 | MEMENTO 代码在 `methods/MEMENTO/`，16 个 slurm/sh 配置，88 份日志 |
| top-k 扫描 | **NOT_FOUND** | 只有渲染好的 `~/Downloads/topk_ablation_2x2_v3.pdf`；脚本里只出现 k=3 和 5 |

**需要你交接的清单**：评测输出写在 Isambard 集群 `/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner/outputs/habitat_llm/`（建库日志里写的路径），从来没拷回来。缺的就是它：Table 1、Table 2、H_R、G-Memory 全部运行、top-k 扫描，以及 latency 测量。

## A1 retrieval-impl：HALTED

对照 B 仓库，也就是 `run_hires_viz.sh` 实际配置的那条代码路径：

1. **task-type 路由**：检索时没有路由。每次运行用哪个库，由 `plan_config.rag_dataset_dir` 固定（`run_hires_viz.sh:37-38`）；运行时没有 classifier 选库。
2. **object-type boost**：B 仓库、ICLR 冻结 diff、A 仓库里都没有 boost 代码。
3. **θ**：`match_abstract_skills` 用 θ=0.3 比较原始 cosine，因为没有 boost，所以不存在 boost 前后之分。
4. **embedding**：`all-mpnet-base-v2`，这一点对（`rag.py:145`、`:466-483`）。

原计划的 patch 是把 Eq.1 后那两句改成只陈述代码事实。但附录例 (1) 和路由消融图用的数在盘上没有依据，只改正文两句会让正文和附录互相矛盾。所以按停止规则标 HALTED，TODO 原样保留。

## A2 library-provenance：结局 B（没过 gate），已出 patch

1. **建库链路**：`rebuild_hierarchical_memory.slurm` → `rebuild_all_memories.sh` → `build_hierarchical_skill_memory.py --include-failed --use-llm --patch-failed`。
   - proposing model：Llama-3.3-70B-Instruct（vLLM，TP4，T=0.1）
   - 日期：2026-01-05
   - 源轨迹：2025-12-30 的 heuristic agent 运行
2. **没有执行门**：LLM 返回的每个技能都以 `success=True` 入库（`:939`、`:963`、`:1123-1143`）。建库过程不 import 仿真器，没有 validation episode，也不检查跨 trace 复现。
3. **instance 数为 1 的技能占比**（ind / coop）：

   | 库 | ind | coop |
   |---|---|---|
   | R | 69.5% | 69.6% |
   | S | 71.4% | 66.7% |
   | HR | 61.1% | 78.3% |
   | HT | 100% | 100% |
   | HRST | 95.0% | 100% |

4. **不是同一个库**：`is_in_room` 那个库是 A 仓库的 `results/partnr_operators_iir1.json`，22 个 JSON 算子，2026-09 生成。Table 1 用的是 B 仓库的 `L_ind`/`L_coop` 自然语言技能，2026-01 生成。

**patch**（净增约 +3 行）：
- 5.1 Evaluation protocol 加 spec 给的那两句
- Introduction 贡献二收窄为"gate 在 VIKI-L2 和一个 PARTNR skill 上评估"
- 删掉这条 TODO

重建库并重跑 Table 1 Llama-8B 的 call 数估算：每 episode 两个 agent，按日志约 20–40 次 planner call，76 个 episode 约 3,000 次；建库本身另计。这个量就把 8000 的 CALL_CAP 用掉一大截，而且会撞上面的泄漏问题。

## A3 effect-match：部分完成

1. **0.43 ms**：如果真测过，量的会是 `HierarchicalRetriever.filter_executable`，纯 Python 子串匹配，**不含 LLM 调用**。但 B 和 A 仓库里都没有 `perf_counter` 埋点，也没有 latency 数据；54 / 28.4 / 24.8 / 0.43 / 0.11 ms 和 9.4±9.0 s 的来源都是 NOT_FOUND。第 836 行的 TODO 保留，因为它要等五臂结果。
2. **例 (3)**：带检索打印的日志有 136 份，里面没有任何这类激活记录，代码里也没有提升 cooperation 技能排名的机制。**patch 删掉例 (3)**，净减约 −3 行。例 (1)、(2) 同样找不到，(1) 已列入 DISCREPANCIES；(2) 在 spec 授权之外，只报告，没有动。
3. **五臂结果**：不存在，第 213 行和第 836 行的 TODO 保留。

## A4 viki-instantiation：DONE，已出 patch（净增约 +2 行）

| 句 | 判定 | 依据 |
|---|---|---|
| 1 一条 instruction 加一张场景图 | IMPRECISE | CG w/o Image 那个 split 没有图（`viki_eval_v2_intent_choice.py:178`） |
| 2 cooperation 分支为空 | IMPRECISE | 没有 partner_cond 技能，这点对；但 8 个算子里有 1 个双角色接力算子 |
| 3 输出带机器人指派的 skill call | IMPRECISE | 模型输出的是 `{do, X, Y, robots}` 形式的需求（提示词 `:71-101`），不点技能名 |
| 4 planner 按 demo 展开 | IMPRECISE | 算子是 planner 按 effect 选的；它还会在候选之间搜索、给接力算子分配角色、套用库里的顺序规则 |
| 5 baseline 直接出 primitive plan | TRUE | 所有 baseline 的 `parsed_output` 都是逐步动作表 |
| 6 CG 组合 | TRUE | 审计的判定器在训练集命中 0 行，且只用了允许的表述；不带双机器人条件的版本命中 1,033 行，所以"激活加搬运未见过"不能写 |

另外，297 行 CG 只有 **295** 个不同的 `task_id`（`4307_8-1`、`1426_9-1` 各重复一次），来自 `audit/comp_leakage_2026-09-05/AUDIT_REPORT.md` L0。写不写进正文由你决定，patch 里没写。

## A5 gmemory-run：BLOCKED

B 仓库、A 仓库、远端都没有 PARTNR 的 G-Memory 代码、配置或日志。一句也写不出能被产物支持的话。Claude 和 GPT-4o 两格属于"没跑、跑了未完成、跑了没通过核对"里的哪一种，同样无法判定。

## A6 table1-ci：BLOCKED（G0）

没有逐 episode 结果，Wilson 区间和 McNemar 都算不了。

## A7 hr-main：BLOCKED，本可转 B3，但见文首的泄漏问题

各方法在 H_R 上找到的最大 n：ours 50、ours-rule 33、MEMENTO 55；RAG、zero-shot、ToM、G-Memory 都没有。

## A8 memento-hrt：NOT_FOUND，可转 B2

找到 12 次 MEMENTO H_R_T 运行，n 在 10 到 68 之间，没有一个等于 14.5，而且对这些 n 来说 14.5 都不是 k/n 能取到的值。最接近的是 9/61 = 14.8，用的是只含 H_T 的记忆。现行 H_R_T 集有 11 个 episode，14.5 同样取不到。

## A9 cells：HALTED

- **72 格**：figure2，3 模型 × 4 split × {ours, ToM, zero-shot, RAG, G-Memory, MEMENTO}。
- **90 格**：`results/agent_library_v3/baseline_comparison.json`，3 × 5 split × {ours, G-Memory, MEMENTO, **skill memory v1**, RAG, zero-shot}。
- 所以 90 − 72 不只是 18 个 sibling 格，**方法集合也不一样**：90 里没有 ToM，多了 skill memory v1。
- `scripts/viki_report_matrix.py` 是 amendment 时期的脚本，这两个数都不是它出的。
- RQ3 的 32 格与 140 总数和盘上不符，见 DISCREPANCIES。
- 建议的口径：正文写 72 + 36 + 36 = 144；sibling 另写 18 格；Pairing 段落的 90 改成"72 个主比较格加 18 个 sibling 格"。等你拍板再出 patch。

## A10 handwritten-7b：HALTED

15 格见 `tables/reference_library_15cells.csv`：

| split | 72B | 30B | 7B |
|---|---|---|---|
| ID | 0.674 盘上 | 0.510 盘上 | 0.160 盘上 |
| OOD 单族（逐折参考库） | 0.656 新算 | 0.460 新算 | 0.145 新算 |
| OOD sibling | NOT_FOUND | NOT_FOUND | NOT_FOUND |
| CG w/o Image | 0.862 盘上 | 不配对 | NOT_FOUND |
| CG w/ Image | 0.798 盘上 | 不配对 | NOT_FOUND |

- 72B 的两个 CG 格和我们 297 行逐行同分。
- 30B 的 `recomb_*_reference` 是另一批答案，只有 256/297 和 203/297 行的原始答案与我们相同，所以不能配对。
- "新算"的三格是零 call 算的：同一批归档答案，换成逐折参考库重放。脚本 `scripts/iclr_todo/viki_reference_fold_replay.py` 直接复用现成的重放代码，没有改动。
- B1 的 R 臂：72B 四个 split 在盘上或已零 call 算出，**不需要任何 call**。

## A11 topk-40：BLOCKED

- 没有任何 k=10 或 k=20 的运行。
- H_R_S_T 只有 5 个 episode，40.0% 等于 2/5；R-only 和全并集下都没有一次运行达到 2/5。
- 15.4% 意味着分母是 13，任何单次运行都凑不出来。

## A12 scorer-parity：DONE，没有翻转

- 72 格每格都在 JSON-tolerant 和官方 strict 两种判分下各算一遍。
- 60 对 ours 与 baseline 的比较，在两种统一口径下胜负方向和显著性都没有变化，最大 p = 7.2e-6。
- 严格判分只会让 baseline 变低，例如 72B G-Memory ID 从 51.0 降到 9.8。也就是说，正文现在的混合口径是偏向 baseline 的。
- ours 在两种口径下按代码逐位相同。
- 文件：`tables/scorer_parity.{csv,tex,json}`、`scorer_parity_mcnemar.csv`。

## B 层

没有开始，需要你决定：

- **B1**：R 臂零 call 已齐。P 臂和 G 臂要 call，72B 四个 split 共 2,442 行，G 臂另算。
- **B2**：可以跑，11 个 episode。但 MEMENTO 的记忆源同样要先查有没有泄漏。
- **B3**：建议先解决泄漏再说。
