# AUDIT — PartNR `partner_cond` 与 VIKI-R `ordering` 能否写成同一形式化条件

日期 2026-09-13。只读审计：仅 `grep` / `ls` / `git log` / `json`·`gzip` 解析 / `sha256`，
**未修改任何代码、数据或产物，未发起任何 LLM 调用**。本文件是唯一新增物。

目的：判断两侧条件能否写成同一形式化条件的两种实例化，还是必须写成两套独立机制。
结论全部来自代码与产物，附路径、hash、行号。不给设计建议。

---

## 两份代码库

| | 路径 | commit | 说明 |
|---|---|---|---|
| **A** | `/Users/shishuqing/partnr-planner` | `dd343f6a023ac63671c1e6b9f83a52957ecaddfa`（2026-09-13） | 被审仓库。VIKI-R 线 + 用同一方法做的 PartNR 线 |
| **B** | `/Users/shishuqing/restore_isambard/partnr-planner` | `e9abe2bb2ed18b44a3abf49f580b70a434300e9e`（2025-12-27，"Add cooperation skills extraction and RAG enhancement features"） | PartNR 侧 `partner_cond` 语义的实现所在 |

**B 的工作树有 229 个未提交改动**，commit 不能锚定内容，因此本报告引用的每个文件单独给 hash（见末节）。

定位过程：`partner_cond` 这个**字段名**在 A、B 以及 `PRISM` / `aris_repo` / `care-gem` /
`robotwin-composition` / `LW-Compositional-Bench`（本地）与 `VIKI-R` / `AutoDataGen` /
`LW-Compositional-Bench`（远端 `/mnt/pfs/devs/pn5wp/shishuqing/`）中**全部 0 命中**。
论文里的 `partner_cond` 在代码中的实际载体是 **`trigger_conditions`（+ `precond_joint`）**，
以下按这个载体审。

---

## Q1 — PartNR 侧 `partner_cond` 的实际形式

### 1.1 schema

`data/hierarchical_skill_memory/*/L_coop_skills.json.gz`（B），每条 cooperation skill
固定 10 个字段（227/227 一致）：

```
name, skill_type("cooperation"), description, preconditions, effects,
precond_joint{...},
partner_context_pattern{belief_formation, intention_inference, action_selection},
trigger_conditions[str], instances[{context, demo, e_src, success}], instance_count
```

条件本体是 **`trigger_conditions`：一个自然语言字符串列表**。
`partner_cond` 字面量在该文件 679,300 字符中 **0 次**。

### 1.2 取值来源：建库时由 LLM 写

`our_method/llm_skill_extractor.py:278 extract_cooperation_skills()`，
prompt 在 `:297`（system）与 `:308`（user）。要求模型输出的 JSON 模板（`:344-356`）：

```
"cooperation_patterns": [{
   "pattern_name": "division_of_labor|sequential_handoff|synchronization|complementary_work",
   "description": "how this pattern manifests",
   "agent_0_role": "...", "agent_1_role": "...",
   "trigger_condition": "when this pattern is used",
   "coordination_mechanism": "how coordination happens (waiting, sequencing, etc.)",
   "tom_reasoning": {"belief_formation": "...", "intention_inference": "...",
                     "action_selection": "..."}
}]
```

另有一条**手写**来源：`our_method/build_rule_based_memory.py` 里硬编码的
`precond_joint` / `triggers`（`:578`、`:630`、`:672`、`:720`、`:771`、`:822`；
装配在 `:525-527`、`:1066-1068`），产出落在 `data/hierarchical_skill_memory_rule*/`。

### 1.3 运行时判定函数与逐行逻辑

**`our_method/hierarchical_retrieval.py:423 _check_cooperation_preconditions`（B）**
（A 仓库同名函数在 `:442`，逻辑逐行相同）。
调用链：`retrieve()` → `:515 filter_executable()`（A `:534`）→ `:395` 分派至该函数。

| 行（B） | 逻辑 | 类型 |
|---|---|---|
| `:436-437` | 取 `skill_data["trigger_conditions"]` 与 `query.partner_effects` | — |
| `:444-445` | `len(trigger_conditions) == 0` → `return True` | 无条件放行 |
| `:449-461` | 对每条 trigger 做 **Python 子串匹配**：`"object_moved_to_target_location" in trigger` → 看 `partner_effects["moved_objects"]`；`"partner_action_in_progress"` → 看 `["action"]`；`"heterogeneous_task_requirements"` / `"task_decomposable_into_subtasks"` → 直接 `True` | **字符串匹配** |
| `:463-466` | 目标串含 `["together","both","each","while","and","then","after"]` 任一 → `True`（`"and"` 几乎匹配所有指令） | 关键词兜底 |
| `:470-471` | `abstract_score > 0.5` → `True` | 相似度兜底 |
| `:473` | 否则 `False` | — |

**不是结构化谓词匹配，也不是 LLM 调用。** 该文件中 LLM 仅出现在把技能渲染进 prompt
（A `:609`）。整个 B 仓库里**没有**任何 LLM 版可执行性过滤实现。

### 1.4 使这条分支形同虚设的实测事实

四个被匹配的 token 与库里实际写下的字符串**不相交**：

| 库（B, `data/hierarchical_skill_memory*/`） | coop 技能数 | 有 trigger 的 | 不同 trigger 串 | 含四 token 的 |
|---|---|---|---|---|
| `/hierarchical_rerange_only/` | 227 | **3** | 3 | **0** |
| `/hierarchical_heterogeneous_rerange/` | 175 | 3 | 3 | **0** |
| `/hierarchical_heterogeneous_temporal/` | 5 | 2 | 2 | **0** |
| `/hierarchical_spatial_only/` | 6 | 2 | 2 | **0** |
| `_llama31_8b/hierarchical_rerange_only/` | 294 | 11 | 11 | **0** |
| `_llama31_8b/hierarchical_heterogeneous_rerange/` | 91 | 9 | 9 | **0** |
| `_location/*`（5 个） | 5–227 | 2–3 | 2–3 | **0** |
| `_rule_location/hierarchical_heterogeneous_rerange/` | 5 | 5 | 5 | 4 |
| `_rule_location/hierarchical_rerange_only/` | 4 | 4 | 4 | 3 |

即：**LLM 建的库里一条都不含可被匹配的 token**，只有 rule-based 库含。
而运行脚本 `run_hires_viz.sh:37-38` 配的是
`data/hierarchical_skill_memory/hierarchical_rerange_only/`（LLM 库）。
在该配置下 `:449-461` **永远不会命中**，每条技能都从 `:444`（空 trigger）
或 `:463` / `:470` 兜底放行。

### 1.5 10 条真实 `trigger_conditions` 原文

LLM 库（全部 50 条不同串中取 10）：

```
 1. 'Multiple tasks with different objects'
 2. 'Dependent tasks with a specific order'
 3. 'Interdependent tasks requiring coordination'
 4. 'When Agent 0 finishes setting the table'
 5. 'When Agent 0 has finished placing cup and teapot and Agent 1 needs to fill them with water'
 6. 'When Agent 1 needs to bring an object to a specific location'
 7. 'when Agent 0 picks up the plant saucer and Agent 1 is ready to place the laptop stand'
 8. 'when an agent is ready to hand off an object to the other agent'
 9. 'when objects are on shelves and table is available'
10. 'kettle task needs to be completed before table and teapot tasks'
```

rule-based 库里则是 token 形式：`object_moved_to_target_location`、
`partner_action_in_progress`、`heterogeneous_task_requirements`、
`task_decomposable_into_subtasks`、`agents_have_specializations` 等。

### 1.6 判定输入 `partner_effects` 的来源

`habitat_llm/planner/llm_planner.py:1108 _extract_partner_effects()`：

- `:1125-1126` 无 `self.previous_objects_state` → 返回空字典
- `:1128` **`if len(self._agents) == 1:`** —— 只在每个 planner 管一个 agent（去中心化）时计算
- `:1131-1136` `extract_object_states_from_world_graph(..., centralized=self.planner_config.centralized)` 取当前状态
- `:1140-1146` 与 `previous_objects_state` 比 `location`，变化者记入 `moved_objects`，
  并把 `action` 设为**常量字面量 `"Rearrange"`**
- `:1149-1152` 有移动则写一条 `completed_subtasks`
- `:1153-1154` 整段包在 `try/except` 里，异常被吞掉只打印警告

`previous_objects_state` 在 `:391` / `:906` 更新为当前快照。
传入点：`:299` 构造 → `:310` `retrieve_top_k_given_query(..., partner_effects=...)`
→ `habitat_llm/planner/rag.py:1221` → `rag.py:1370` `hierarchical_retriever.retrieve(...)`。

---

## Q2 — VIKI-R 侧 ordering 的实际形式

### 2.1 存储形式

`outputs/v3_memories/memory_all.json`（A，远端）的 **`layer2`**，与 layer1 算子并列：

```
layer2 = { rules: [...], kept_patterns: [json 字符串], min_support: 30,
           min_precision: 0.9, orderings_recalled: 1277, orderings_total: 1277,
           recall: 1.0, false_orderings: 0 }
```

单条 rule：

```json
{"pattern": {"a_key":"pos","b_key":"act",
             "a_subject_is_b_target":false,"a_target_is_b_subject":false,
             "a_target_is_b_target":false,"b_visits_a_target":true},
 "ordered":630,"seen":630,"precision":1.0,
 "families":["cut_fruit_on_board","cut_two_fruits_on_board"],"kept":true}
```

即：**两个 effect key（`pos` / `act`）之间的结构共指模式**，外加统计量。
**没有任何自然语言字段。**

### 2.2 挖掘代码

`our_method/skill_memory_v2/dependencies.py:71 mine()`：遍历训练 episode 的
`time_steps`（`:85`），`replay()` 后取 `_ordered_pairs(truth)`（`:95`）与
`requirements_of(truth)`，按 `_describe(a, b, …)` 归并计 `[ordered, seen]`，
按 `MIN_PRECISION`（`:144`，实测 0.9）与 support（实测 30）过滤，
输出 `kept_patterns`（`:142`）。

### 2.3 运行时消费与逐行影响

1. `scripts/viki_eval_v2_intent_choice.py:263-266`
   `temporal = memory.order_for(requirements, visits_of(env, requirements, memory))`，
   写入 `blind["temporal_constraints"]`；`--no-order`（`:148`）时置空。
   重规划路径同理（`:315-317`）。
2. `our_method/skill_memory_v2/memory.py:127 order_for` →
   `dependencies.py:152 order_for`：对每对 (i, j) 算 `_describe`，
   **若该 pattern 在 `kept` 集合中**，产出 `[[a],[b]]`（`:168-171`）。
3. `our_method/skill_memory_v2/planner.py:59-65`：把每条 temporal constraint 展开成
   `register(predicate, earlier)`，即给后者挂一个 **`guard` 谓词列表**（`:52`、`:57`）。
4. `planner.py:197`：
   **`if len(chain["actions"]) == 1 and any(not holds(env, p) for p in chain["guard"]): continue`**

**性质：硬约束，但有范围限定。** 不满足 guard 的候选链被 `continue` 直接跳过
（不是降权、不是改 prompt 呈现顺序——ordering 根本不进 prompt，它进的是 `blind` 任务规格）；
该检查**只对单动作链生效**（`len(chain["actions"]) == 1`）。
多动作链的 guard 挂在最后一格（`:135`、`:142`）但不走这条检查。

### 2.4 10 条真实 ordering 规则（该库全部 10 条）

| # | a_key → b_key | 结构条件 | ordered/seen | precision | kept |
|---|---|---|---|---|---|
| 1 | pos → act | `b_visits_a_target` | 630/630 | 1.000 | **是** |
| 2 | pos → act | `a_target_is_b_subject, b_visits_a_target` | 441/441 | 1.000 | **是** |
| 3 | pos → pos | `a_subject_is_b_target` | 206/206 | 1.000 | **是** |
| 4 | pos → pos | `a_target_is_b_target, b_visits_a_target` | 0/2008 | 0.000 | 否 |
| 5 | pos → act | （无） | 0/441 | 0.000 | 否 |
| 6 | pos → pos | （无） | 0/882 | 0.000 | 否 |
| 7 | act → pos | （无） | 0/1071 | 0.000 | 否 |
| 8 | act → pos | `a_subject_is_b_target` | 0/441 | 0.000 | 否 |
| 9 | pos → pos | `a_target_is_b_target` | 0/166 | 0.000 | 否 |
| 10 | pos → pos | `a_target_is_b_subject, b_visits_a_target` | 0/206 | 0.000 | 否 |

`false_orderings: 0`，`orderings_recalled / total = 1277/1277`。

---

## Q3 — 两者能否对齐

### a) 条件的语义载体 —— **不可对齐**

| | 指向什么 | 依据 |
|---|---|---|
| PartNR | 条件串是**自然语言任务描述**（"When Agent 0 finishes setting the table"）；判定时读的是**观测到的位置差分**；但其中一条 token 路径读 `partner_effects["action"]`，而该值是硬编码常量 `"Rearrange"`，**不是可观测状态** | `llm_planner.py:1141-1146`；`hierarchical_retrieval.py:455-457`(B) |
| VIKI-R | pattern 指向**两个 effect key 之间的结构共指关系**（`a_target_is_b_subject`、`b_visits_a_target`），是算子间的类型级关系，**不描述任何"已发生的变化"** | `memory_all.json` layer2.rules；`dependencies.py:152-171` |

一个是"伙伴造成的状态差分（外加一个常量动作标签）"，一个是"两个算子的参数共指"。
**不是同一条件的两种实例化。**

### b) 判定时刻 —— **不可对齐**

- PartNR：**候选筛选阶段**。`retrieve()` 的 Stage 4，
  `hierarchical_retrieval.py:515`（B）/ `:534`（A）调用 `filter_executable`，
  `:395` 分派至 coop 检查。此时尚未规划。
- VIKI-R：**跨两个阶段**。约束**生成**在规划之前、检索之外
  （`viki_eval_v2_intent_choice.py:263-266`）；约束**执行**在规划器的调度循环内
  （`planner.py:197`）。VIKI-R 侧不存在"cooperation 候选筛选"这一步。

### c) 判定输入 —— **不可对齐**

- PartNR 读：`self.previous_objects_state` 与当前 `world_graph` 的**跨步位置差分**
  （`llm_planner.py:1131-1146`），即"上一时刻 → 此刻"。
- VIKI-R 读两样，都不是跨步差分：
  1. 约束生成读 **初始静态世界** `env = sim.world(metadata)` 与**算子 body 的 visits 集合**
     （`viki_eval_v2_intent_choice.py:263`；`scripts/viki_eval_skill_memory_v2.py:85-115`，
     其中 `candidates = memory.operators_for(...)`、遍历 `operator["body"]` 取 `?x/?y/?z` 目标）；
  2. 约束执行读 `holds(env, p)`，而该 `env` 由规划器自己的
     `env.sim_step(step_commands)` 推进（`planner.py:150`、`:235`、`:197`）。

一侧输入是"另一个 agent 造成的差分"，另一侧是"初始世界 + 算子签名"和"自己刚排的步骤"。

### d) 不满足时的行为 —— **不可对齐**（表面同为"排除"，作用对象不同）

- PartNR：返回 `False` → `skill.is_executable = False`（`:401`），该技能**不进入 executable 列表**。
  但实测该分支近乎恒开：`:444` 空 trigger 放行、`:463-466` 关键词放行、`:470` 分数放行，
  且四个 token 在实际所用库中 0 命中（§1.4）。
- VIKI-R：guard 不成立 → `continue`（`planner.py:197`），**该候选链被排除出规划**，
  且仅限单动作链。不降权，不影响 prompt 呈现顺序。

**四点全部不可对齐。**

---

## Q4 — 中心化退化是否成立

### 运行时是否存在"已执行动作 / 世界状态变化"的表示被传给条件判定

**存在一个，但不是伙伴动作，而是规划器自己推进的模拟状态。**

- 数据结构：`chain["guard"]` —— 谓词列表，建于 `planner.py:52`（`entry["guard"]`），
  挂载于 `:135` / `:142`。
- 传入点：`planner.py:197` `holds(env, p)`，其中 `env` 建于 `:150` `sim.world(metadata)`，
  并在调度循环中被 `:235` `env.sim_step(step_commands)` 逐步推进。
- **不存在**任何"上一步伙伴动作"或"跨 episode 步的观测差分"表示。
  VIKI-R 侧全仓库无 `partner_effects` 构造（该名字只出现在 `our_method/viki_adapter.py`，
  那是把 VIKI 行喂给旧 v1 检索器的适配层，不在 v3 主表运行时路径上）。

### 结论（两句都必须写，单写任一句都是错的）

1. **决定"哪些序约束存在"的那一步完全不读执行状态。**
   `order_for` 只看 requirement 两两之间的结构共指与 `visits`，而 `visits` 来自初始世界与算子 body
   （`viki_eval_v2_intent_choice.py:263`、`viki_eval_skill_memory_v2.py:85-115`、
   `dependencies.py:152-171`）。这一步是**静态的算子间顺序规则**。
2. **决定"某条链此刻能否落地"的那一步读的是规划器自己已排步骤造成的状态。**
   `holds(env, guard)` 对着被 `sim_step` 推进过的 `env` 判定（`planner.py:197` / `:235`）。

所以：VIKI-R 的 ordering **不能**整体表述为"条件由规划器自身的已执行步骤提供"的特例
——那只覆盖第 2 步；也**不能**说它"根本不读任何已执行状态"——那忽略了第 2 步。
**约束的选取是静态的，约束的兑现读自推进状态。**
而 PartNR 侧没有与第 1 步对应的东西（它没有 mined pattern 这一层），
也没有与第 2 步对应的东西（它的判定发生在检索阶段，不在调度循环内）。

---

## Q5 — 命名与口径对照

| PartNR（B） | VIKI-R（A） | 判定 |
|---|---|---|
| `skill_type` ∈ {`individual`, `cooperation`} | `kind` ∈ {`achievement`, `coordination`} | **不同的东西，名字相近**。前者按"几个 agent 参与"分；后者按"效果是达成一个谓词还是协调两个算子"分。`coordination` 不是 `cooperation` 的改名：v3 库 8 个算子里 7 个 `achievement` + 1 个 `coordination`，cooperation 类算子为 0 |
| `trigger_conditions: [str]`（自然语言） | `layer2.rules[].pattern`（结构化 6 字段） | **不同的东西**。论文的 `partner_cond` 对应前者；后者无自然语言载体 |
| `precond_joint: {…}`（如 `{"both_agents_available": true}`） | 无对应物 | 盘上无对应 |
| `preconditions`（列表，实测多为空） | `preconditions`（dict，如 `{"subject_on_agent": false, "target_on_agent": false}`） | **同名异物**。前者是技能级前提字符串；后者是算子对世界的类型级前提，参与 `operators_for` 匹配 |
| `effects`（列表，实测多为空） | `effect: {key, subject, value}` + `effect_key` | **同名异物**。前者自由文本；后者是算子的索引键（`pos.name` / `is_activated`） |
| `partner_context_pattern{belief_formation, intention_inference, action_selection}` | 无对应物 | 盘上无对应 |
| `instances[{context, demo, e_src, success}]` | `body`（动作序列）+ `families` + `provenance{proposed_by, sources, verified_on}` | **不同的东西**。前者存自然语言 demo；后者存可执行动作体与出处 |
| `partner_effects{action, moved_objects, completed_subtasks}` | 无对应物（v3 路径） | 盘上无对应 |
| `abstract_score`（检索相似度） | `support` / `precision` / `cost` | **不同的东西**。前者查询相关度；后者是建库统计与规划代价 |
| `filter_executable` → `is_executable`（候选筛选） | `chain["guard"]` → `holds(env, p)`（规划内） | **同名异物**：都叫"可执行性"，一个筛检索候选、一个筛规划链 |

同名异物有 `preconditions`、`effects`、"可执行性" 三处；
**真正同物异名的对子为零**。

---

## 总判定

两侧**必须写成两套独立机制**。依据是 Q3 四点全部不可对齐，且 Q5 中没有一对同物异名：

- 语义载体不同（自然语言伙伴条件 vs 算子间结构共指）；
- 判定时刻不同（检索候选筛选 vs 规划约束生成 + 调度内兑现）；
- 判定输入不同（跨步观测差分 vs 初始世界 + 算子签名 + 自推进状态）；
- 作用对象不同（检索候选 vs 规划链）。

附带两条对论文陈述直接相关的实测事实：

- PartNR 侧那条"效果匹配"分支在**实际所用的 LLM 库配置下永远不会命中**
  （四个被匹配 token 在 LLM 库中 0 出现，`run_hires_viz.sh:37-38` 配的正是 LLM 库），
  所有 cooperation 技能由空 trigger / 关键词 / 相似度三条兜底放行。
- PartNR 侧的判定**不是 LLM 判定**（`hierarchical_retrieval.py:423-473`），
  整个 B 仓库无 LLM 版可执行性过滤实现。

---

## 引用清单与 hash

**仓库 B**（`/Users/shishuqing/restore_isambard/partnr-planner` @ `e9abe2b`，
工作树 229 处未提交，故逐文件 hash）：

| 路径 | sha256[:16] |
|---|---|
| `our_method/hierarchical_retrieval.py` | `5b864172f6b30659` |
| `our_method/planner_integration.py` | `5feb8e1379186a27` |
| `our_method/llm_skill_extractor.py` | `dfd74fbeb83aae94` |
| `our_method/build_rule_based_memory.py` | `aff13094a60ff221` |
| `habitat_llm/planner/rag.py` | `e7c0c0a3e0a22a08` |
| `habitat_llm/planner/llm_planner.py` | `e08517a3a81dd27e` |
| `run_hires_viz.sh` | `bd3bbf2ba9b7e647` |
| `data/hierarchical_skill_memory/hierarchical_rerange_only/L_coop_skills.json.gz` | `4f307d4f24ec4d54` |

**仓库 A**（`/Users/shishuqing/partnr-planner` @ `dd343f6`，工作树除两个 submodule 外干净）：

| 路径 | sha256[:16] |
|---|---|
| `our_method/hierarchical_retrieval.py` | `6106ee517367b20e` |
| `our_method/viki_adapter.py` | `4d421cf306ffd3ec` |
| `our_method/skill_memory_v2/planner.py` | `c84fee072b4ba528` |
| `our_method/skill_memory_v2/dependencies.py` | `7caeff9459fd3c2e` |
| `our_method/skill_memory_v2/memory.py` | `ba8c65749b2a485c` |
| `scripts/viki_eval_v2_intent_choice.py` | `f3502c16aa083e25` |
| `scripts/viki_eval_skill_memory_v2.py` | `cb70003120a9343b` |
| `outputs/v3_memories/memory_all.json`（远端） | `d580b15c90b539ae` |

未单独取 hash 的：B 仓库其余 24 个 `L_coop_skills.json.gz`
（§1.4 表中的计数由脚本一次性读出，未逐个取 hash）。
