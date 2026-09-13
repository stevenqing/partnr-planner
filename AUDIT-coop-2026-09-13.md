# AUDIT — effect-indexed cooperation 在 VIKI-R 上的盘上证据

日期 2026-09-13。只读审计：仅 `grep` / `ls` / `json` 解析 / `sha256`，**未修改任何代码、数据或产物，未发起任何 LLM 调用**。
本文件本身是唯一新增物。

审计对象是论文 `docs/memory-as-skill` 中 **Effect-based Partner Inference** 一段（`:238-239`）及其执行期对应物（`:261`）。
被审机制的描述：cooperation instance 带一个建库时用自然语言写下的 `partner_cond`，使用时由 LLM 判定当前观测差分是否匹配，
匹配则该 instance 进入候选。

---

## 摘要

| 问 | 结论 |
|---|---|
| Q0 cooperation 分支在 VIKI-R 可用吗 | **主表所报的库里为零，且不在主表运行时代码路径上**。`partner_cond` 字段在全仓库没有实现 |
| Q1 跨物体/跨房间激活 | **盘上无证据，且不可测**（无激活点记录；产物里没有房间字段）。已穷尽全部 6 份 gate 产物 |
| Q2 匹配是否只是字面匹配 | **无激活点可算重合度**；但机制本身可判定——盘上唯一的 cooperation 门是子串与关键词匹配，**不是语义匹配，也没有 LLM** |
| Q3 未覆盖时走哪条路 | **盘上无证据**；代码中的实际逻辑与提示描述不符，该门近乎恒开 |
| Q4 `w/o Coop` 消融 | **存在，但属于 PartNR、Llama-3.1-8B、n=5 的图 2b，不是 VIKI-R**；其图与 `no_coop` 运行产物**不在本仓库** |

外加一条贯穿性事实：**这版论文根本没有提到 VIKI**（`grep -c "VIKI" docs/memory-as-skill` = 0），
`:346` 写明「We evaluate on the PartNR benchmark」，`:449` 自列局限「domain transfer beyond PartNR's
household environments is unverified」。所以"在 VIKI-R 上是否成立"目前是一个**尚未写下的**主张。

---

## Q0 — cooperation 分支在 VIKI-R 上是否可用

### 主表所报的库（v3，8 算子）

`outputs/v3_memories/memory_all.json` 的 `layer1.operators` 共 8 条，`kind` 只有两种取值：

| kind | 条数 |
|---|---|
| achievement | 7 |
| coordination | 1 |
| **cooperation** | **0** |

该文件内字面量出现次数：`cooperation` **0**、`partner_cond` **0**、`partner` **0**、`coop` **0**。

扩到盘上**全部 63 个库/记忆产物**（`outputs/v*_libraries/library_*.json` + `outputs/v*_memories/memory_*.json`）：
achievement 130、coordination 11、**cooperation 0**。

另注口径不一致：论文定义 `type ∈ {individual, cooperation}`（`docs/memory-as-skill:236`），
而 VIKI 线实际出库用的是 `kind ∈ {achievement, coordination}`，**两套分类法不可直接对应**。

### 运行时

主表评测入口 `scripts/viki_eval_v2_intent_choice.py` 中**没有任何 cooperation 代码路径**。
全文唯一与 partner 相关的是 `--partner` 开关（`:136-137`、`:205-210`），作用是把归档 plan-writing 臂
用过的一段 partner 前缀文本拼进 prompt——它不索引、不筛选、不激活任何技能。

### `partner_cond` 在本仓库没有实现

本地与远端仓库全量 grep（`*.py` / `*.json` / `*.md` / `*.sh`，排除 `third_party/`）命中 **1 处**，
且是一张说明表里的字段名引用：

> `docs/viki_memory_as_skill_zh.md:50`
> `| 合作技能 | 根据 partner_cond 检索 | 没有可定义的 partner_cond | 明确关闭 |`

`outputs/` 下全部产物中 `partner_cond` 零命中。

### 盘上确实存在 cooperation 类型的地方（基线臂，且不是 effect-indexed）

`results/viki_memory_experiments/amendment8b/skill_memory_bank/L_coop_skills.json.gz` ——
这是主表里 **`skill memory v1` 基线**那一列的库，不是 ours。

- 412 条，全部 `skill_type: "cooperation"`。另有 8 个逐族 fold 库各 326–400 条，及一个
  `INVALID-3episode-bank` 4 条。
- 字段集合（412/412 一致）：`name, skill_type, description, preconditions, effects,
  precond_joint{trigger}, partner_context_pattern{belief_formation, intention_inference,
  action_selection}, trigger_conditions, instances[{context, demo, e_src, success}], instance_count`。
- 全文 6,790,355 字符中：**`partner_cond` 0 次**，**`room` 0 次**（所以"跨房间"在这个产物里不可表示）。
- 决定一条 cooperation 技能是否可用的门是
  `our_method/hierarchical_retrieval.py:442 _check_cooperation_preconditions`，逻辑为**确定性字符串匹配**：

  1. `trigger_conditions` 为空 → 直接 `True`（`:465-467`）；
  2. 否则在四个硬编码 token 上做子串匹配：`object_moved_to_target_location` /
     `partner_action_in_progress` / `heterogeneous_task_requirements` /
     `task_decomposable_into_subtasks`（`:469-477`）；
  3. 再退到目标串关键词 `["together","both","each","while","and","then","after"]` → `True`
     （`:479-484`，注意 `"and"` 几乎匹配所有指令）；
  4. 再退到 `abstract_score > 0.5` → `True`（`:486-488`）。

  **这条路径上没有任何 LLM 调用**；该文件里 LLM 只出现在 `:609` 把技能渲染进 prompt。
- 候选排序是 embedding 相似度，向量只由 `objects / locations / rooms / object_locations /
  action_sequence` 构成（`_context_to_text`，另见 `docs/amendment9-preregistration-draft.md:42-46`）。

**Q0 结论**：提示中描述的机制在本仓库盘上**没有任何实现**。最接近的现存物是一个确定性关键词门，
位于一条基线臂上。按既定规则，Q1–Q3 只做到"报告无证据 + 写明原因"，不作推断。

---

## Q1 — 跨物体/跨房间激活是否真实发生

**盘上无证据，且不可测。** 主表的库里 cooperation 算子为 0、运行时无该分支，
故不存在"cooperation instance 被激活的决策点"；基线臂那 412 条技能的实例 `context` 不含房间字段。

已穷尽盘上**全部 6 份 gate 产物**，没有任何一份记录 instance 级激活点、`partner_cond` 原文、
或建库 context 与激活时 context 的对照：

| 产物 | 测的是什么 | 与 cooperation 的关系 |
|---|---|---|
| `amendment4/g2b_gates.summary.json` | 零模型调用的 prompt 池/渲染保真：池 400 行、10,359 个允许实例、`byte_identical_rows` 400/400、渲染 120/120 逐字节一致、模板不匹配 0 | **无关** |
| `amendment4/g2b_posttrim_render_gate.summary.json` | 截断后渲染保真：20 行、完整片段 35/35 逐字节一致、11 行有且仅有一个被截断尾 token | **无关** |
| `amendment8b/stepaligned_gate.json` | 并行/空闲结构有没有渲染进 prompt：24 行，coordination block 20/24、可见空闲机器人 20/24、同一步两个机器人 20/24、重写 82 / 保持压平 110（`rewrite_rate` 0.427）、`row_coverage` 0.833 | 有关，但测**渲染**不测激活 |
| `amendment8b/gate_patternslot.json` | `coop_skills_selected` 151、`partner_in_query` 24/24、`rows_with_role_text_in_prompt` 23/24 | 有关，但是"进了 prompt 多少条"，非按观测差分激活 |
| `amendment8b/gate_grounded.json` | 同上（151 条、partner_in_query 24/24） | 同上 |
| `amendment8b/gate_rescore.json` | 同上（151 条、partner_in_query 24/24） | 同上 |

`amendment9_ledger.json` 另记：rung `queryfix` 与 `roleaware` 下 `coop_skills_seen` 140（24 行抽样）、
**`coop_skills_with_roles` 0**、门 `roles_retrieved: false`。
与之一致，`docs/amendment9-preregistration-draft.md:29-46` 记载承载协作结构的只有 412 条中的 **5 条**，
且「**0 of 705 retrieved coop skills carry role fields**」。

### 主表那批格里这套机制是关的

主表 `skill memory v1` 那一列的归档格 tag 是 `skill_memory.fullactions_k8`
（`scripts/viki_p0_report.py:51`、`scripts/viki_v2_baseline_comparison.py:119`）。逐个读 run 元数据：

| 格 | 模型 | `a9_mode` | `a9_role_aware` | `A9_PATTERN_SLOTS` | `A9_STEP_ALIGNED` | `skill_render_top_k` |
|---|---|---|---|---|---|---|
| `amendment8b/…k8`（ID / 留出） | 72B | `""` | `"0"` | **未记录** | 未记录 | 8 |
| `…k8_m30`、`…k8_m7` 及 r2/r3、nt 变体 | 30B / 7B | `""` | `"0"` | 未记录 | 未记录 | 8 |
| `amendment10/text`、`amendment10/imaged` | — | `""` | `"0"` | **`"0"`** | **`""`（关）** | 8 |

comp 两格明确记下 `A9_PATTERN_SLOTS=0`、`A9_STEP_ALIGNED` 空；ID/留出那批的 run.json
**没有 pattern_slots 这个键**（全 `amendment8b/*.run.json` 零命中），只能记为"未记录"，
代码默认值为 0（`scripts/viki_amendment8_memory.py:42`、`:46` 均注明 off by default）。

含义：预留槽没开，所以「承载协作结构的 5 条技能从不被检索到」在主表那一列里**依然成立**；
`gate_patternslot.json` 的 151 条是**开了预留槽之后的侧支实验**，不是主表格的性质。

---

## Q2 — 匹配是否只是字面匹配

**盘上无激活点，重合度分布无法计算。** 但机制本身无需统计即可判定：
Q0 引用的 `_check_cooperation_preconditions` 是子串与关键词匹配，**不是语义匹配，也不是 LLM 判定**。
据此**不能**写"匹配是语义的而非字面的"。

论文 `:261` 明确写的是「LLM-based verification (rather than rule-based) provides the semantic
flexibility needed under distribution shift」——**这与本仓库代码相反**，且本仓库里
**根本没有任何 LLM 版的可执行性过滤实现**（不只是 VIKI 路径上没有）。

---

## Q3 — 未覆盖的情况走了哪条路

**盘上无证据**（无决策点级记录）。代码中的实际逻辑与提示描述不同，照实记录：
cooperation 门在 `trigger_conditions` 为空、目标串含 `and` 等关键词、或 `abstract_score > 0.5` 时
**一律返回 True**（`our_method/hierarchical_retrieval.py:465-488`），
即该门在实践中近乎恒开，**不存在"候选为空 → 回退"的分支**。
主表 ours 臂的回退逻辑与 cooperation 无关，见 `scripts/viki_eval_v2_intent_choice.py`。

---

## Q4 — `w/o Coop` 消融的实际数字

`w/o Coop` 确实存在，但**不在 VIKI-R 上**。出处是论文源文件的图 2b caption：

> `docs/memory-as-skill:396`
> `\caption{\textbf{Component ablation (RQ2).} Full: complete method; w/o Coop: no cooperation memory;`
> `w/o Hier: no hierarchical retrieval; w/o Ind: no individual skills.}`

其归属（`:382` 消融定义、`:399` 共用 caption）：

- benchmark：**PartNR**（任务组 2-constraint / H_R_T / H_R_S_T）
- backbone：**Llama-3.1-8B**，`:663` 注明 H_R_T 11 集、H_R_S_T 5 集
- 载体：图 `figs/ablation_combined_3panel.pdf`，正文无数字
- 该图**不在本仓库**（`find` 零命中）；`no_coop` 的任何运行产物也**不在**
  （本地 `*.json` / `*.csv` / `*.py` 与远端 `results/`、`outputs/` 全量搜索均为零）

只能说"不在这里"，**不能推断别处也没有**。

### VIKI-R 侧实际存在的消融

- 运行时开关只有 `--no-grounding`、`--no-order`（`scripts/viki_eval_v2_intent_choice.py:148-150`），
  **没有任何 coop 相关开关**。
- 盘上消融格共 24 个：`v3_abl{,full}_{noground,noorder}_{72B,7B}_{id,text,imaged}`。
  **只有 72B 与 7B，没有 30B。**
- 报在 `RESULTS-2026-09-13.md` §3，分「全份记忆 `memory_all.json`」与「半份记忆 `comp_cd`」两套；
  对照列取**同一份记忆**的未消融格，与 §1 主表是同一批库（v3）、同一批 replay 归档。

### 三处与之一致的既有文档陈述

- `docs/viki_bench.md:146-150`：「The full cooperation branch cannot be evaluated on this split:
  every one of the 1,218 L2-OOD rows has one active robot and one static image, so there is no
  alternating partner execution or observable environment delta for effect-based partner inference.」
- `VIKI_RESULTS.md:19-22`：「It is not an evaluation of cooperation memory or the decentralized
  cooperation method.」
- `docs/viki_memory_as_skill_zh.md:50`：合作技能一行标注为「**明确关闭**」。

---

## 论文三条可检验主张 vs 盘上情况

| # | 论文主张（出处） | VIKI-R 盘上情况 |
|---|---|---|
| A | cooperation instance 比 individual 多一个 `partner_cond`（`docs/memory-as-skill:236`） | **不成立**。v3 库 cooperation 为 0；v1 库 412 条的字段是 `precond_joint.trigger` / `trigger_conditions` / `partner_context_pattern`，`partner_cond` 全文 0 次 |
| B | 「the same cooperation skill activates across unseen object–location combinations」（`:239`） | **不可测**。无激活点记录；实例 context 无房间字段 |
| C | 可执行性过滤是 **LLM-based verification**，且明确「rather than rule-based」（`:261`） | **与代码相反**。`filter_executable`（`:383`）→ `_check_cooperation_preconditions`（`:442`）全程无模型调用 |

---

## 引用产物与 hash

远端 `/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner/`，sha256 前 16 位：

| 路径 | sha256[:16] | 字节 |
|---|---|---|
| `outputs/v3_memories/memory_all.json` | `d580b15c90b539ae` | 20,785 |
| `results/viki_memory_experiments/amendment8b/skill_memory_bank/L_coop_skills.json.gz` | `e1534b60e52690bb` | 561,655 |
| `results/viki_memory_experiments/amendment8b/skill_memory_bank/memory_summary.json` | `22beeb8c25e7a68f` | 34,166 |
| `results/viki_memory_experiments/amendment8b/amendment9_ledger.json` | `1b63fbc53eac6189` | 3,563 |
| `results/viki_memory_experiments/amendment8b/gate_patternslot.json` | `ea996d2ac3be004a` | 649 |
| `results/viki_memory_experiments/amendment8b/gate_grounded.json` | `0a7e4297a0e5f6a6` | 671 |
| `results/viki_memory_experiments/amendment8b/gate_rescore.json` | `9f22b6824f176086` | 669 |
| `results/viki_memory_experiments/amendment8b/stepaligned_gate.json` | `8d4004f9577366be` | 524 |
| `results/viki_memory_experiments/amendment4/g2b_gates.summary.json` | `79201c5f831080c8` | 1,292 |
| `results/viki_memory_experiments/amendment4/g2b_posttrim_render_gate.summary.json` | `d9981be8de57327d` | 536 |
| `results/viki_memory_experiments/amendment8b/skill_memory.fullactions_k8.jsonl.run.json` | `af1a2e6a7d8428ca` | 72B ID 格元数据 |
| `results/viki_memory_experiments/amendment8b/skill_memory.fullactions_k8_m30.jsonl.run.json` | `0e1fe3ed99d27d36` | 30B |
| `results/viki_memory_experiments/amendment8b/skill_memory.fullactions_k8_m7.jsonl.run.json` | `d8a6c68d59cef954` | 7B |
| `results/viki_memory_experiments/amendment10/text/skill_memory.fullactions_k8.jsonl.run.json` | `da5cd6c47003a5d6` | comp 文本格 |
| `results/viki_memory_experiments/amendment10/imaged/skill_memory.fullactions_k8.jsonl.run.json` | `4684a0f70783a42b` | comp 带图格 |
| `results/agent_library_v3/baseline_comparison.json` | `3c5502be40fd2125` | 24,693 |
| `RESULTS-2026-09-13.md` | `ddf43dab8ad91da0` | 29,045 |

源码与文档（本地，与远端经 mutagen 同步）：

| 路径 | sha256[:16] |
|---|---|
| `docs/memory-as-skill` | `6c56dc3daad50c12` |
| `scripts/viki_eval_v2_intent_choice.py` | `f3502c16aa083e25` |
| `our_method/hierarchical_retrieval.py` | `6106ee517367b20e` |
| `our_method/viki_adapter.py` | `4d421cf306ffd3ec` |
| `scripts/viki_amendment8_memory.py` | `4f48e9061265d8ea` |
| `docs/viki_bench.md` | `5608b43f39cd6ebe` |
| `docs/viki_memory_as_skill_zh.md` | `657b489f5288c206` |
| `docs/amendment9-preregistration-draft.md` | `e8bf182998bc400c` |
| `VIKI_RESULTS.md` | `041c4a3728231c92` |

### 一处现场备注

`amendment4` 两份 gate 的 `artifacts` 路径写的是 `/home/aiscuser/partnr-planner/...`，
**是另一台机器的路径**（不是当前的 aibox）。对应的三个 parquet 在本箱上均**存在**
（`g2b_pool_gate.parquet` / `g2b_render_gate.parquet` / `g2b_posttrim_render_gate.parquet`），
产物没丢，只是元数据记的是产生它们的那台机器。

---

## 一句话

论文若要在 VIKI-R 上陈述 effect-indexed cooperation（`partner_cond` + LLM 判定观测差分），
**盘上没有任何支撑**：主表的库里 cooperation 算子为 0，运行时无该分支，`partner_cond` 字段在整个
仓库没有实现，唯一的 cooperation 门是基线臂里的确定性关键词匹配且其配套机制在主表格里是关的；
`w/o Coop` 消融属于 PartNR 而非 VIKI-R，其数据不在本仓库。
而这版论文本身**没有任何 VIKI 陈述**——这是一个尚未写下的主张，不是一个被夸大的主张。
