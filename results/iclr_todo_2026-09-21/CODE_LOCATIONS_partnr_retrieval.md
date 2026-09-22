# PARTNR 出数版本：skill 预测与检索的代码位置（2026-09-21 核对）

## 版本

- 仓库：B 仓库，本地副本 `~/restore_isambard/partnr-planner`。git 基点 `e9abe2b`（2025-12-27），另有未提交的工作区改动。
- 与 09-13 冻结记录 `freeze_2026-09-13/hashes.txt` 逐字节一致：
  - `habitat_llm/planner/llm_planner.py`：sha256 `e08517a3…`
  - `habitat_llm/planner/rag.py`：`e7c0c0a3…`
  - `our_method/hierarchical_retrieval.py`：`5b864172…`
  - 这三个文件的 `git diff HEAD` 与 `freeze_2026-09-13/worktree.diff` 相同。
- 运行配置（以 Table 1 R_S 为例）：`all_scripts/bash_files/run_ours_rs_mem_r.sh`。
  - `example_type=hierarchical`（:46-47）
  - `rag_top_k=5`（:40-41）
  - prompt 模板：`instruct=rag_prompt_sequential_cooperation_skills_v4`
- 限定：Table 1/2 的原始输出在 Isambard 上，没有拷回来，所以无法逐字证明出数时用的就是这份代码。能佐证的是：本地同期的部分日志里，「HIERARCHICAL RETRIEVAL」出现次数正好是 episode 数 × 2，和下面「每个 agent 每个 episode 只检索一次」一致。例如 `logs/ours_hr_mem_h_167956.log`：71 个 episode，142 次检索。

## 结论

**评测路径上不存在「每步做 skill 预测」，也不存在「每步检索」。** 检索只在每个 episode 开局、每个 agent 各做一次，query 用模板拼接，不调用 LLM。之后每一步只做 LLM 生成动作，外加状态变化触发 replan。

## 实际调用链（全部在评测路径上）

| 步骤 | 位置 | 说明 |
|---|---|---|
| 每步入口 | `habitat_llm/planner/llm_planner.py:696` `get_next_action` | 每个仿真步都会进来 |
| **只在开局构造 prompt** | `llm_planner.py:732-735` | `if self.curr_prompt == "":` 才调用 `prepare_prompt` |
| `curr_prompt` 清空 | `llm_planner.py:121-135` `reset()`（:132） | 每个 episode 开始时调用一次，其它地方不会清空 |
| prompt 构造 | `llm_planner.py:264` `prepare_prompt` | |
| ├ 取 agent 状态、环境状态 | `llm_planner.py:292-296` | 开局时手里什么都没拿 |
| ├ 取伙伴效应 | `llm_planner.py:299` → `:1108-1156` `_extract_partner_effects` | 开局没有上一步状态，走 `:1125-1126` 返回 `{"action": None, "moved_objects": [], "completed_subtasks": []}` |
| ├ **检索调用（唯一一次）** | `llm_planner.py:304-311` `self.rag.retrieve_top_k_given_query(input_instruction, …)` | 第一个参数是 task instruction |
| ├ 检索结果写入 prompt | `llm_planner.py:313-318` `_format_rag_examples` → `params["rag_examples"]` | |
| └ state comparison 写入 prompt | `llm_planner.py:355-404` | 开局时 `compute_state_differences_from_world_graphs` 返回写死的提示语（`habitat_llm/llm/instruct/utils.py:809-810`） |
| RAG 分派 | `habitat_llm/planner/rag.py:1190` `retrieve_top_k_given_query` → `:1220-1227` | `example_type == "hierarchical"` 时进入分层检索 |
| 分层检索包装 | `rag.py:1296` `_retrieve_hierarchical_top_k` → `:1370` `self.hierarchical_retriever.retrieve(…, goal=query)` | 结果为空时退回 `:1383` `_flat_retrieval` |
| 检索主流程 | `our_method/hierarchical_retrieval.py:476-533` `retrieve` | 四个阶段依次调用，最终按「技能分 × 实例分均值」排序（:533） |
| 阶段 1：生成 query | `hierarchical_retrieval.py:179-258` `generate_query` | **模板拼接**：`"Goal: <instruction> \| Holding \| At \| Seen objects(≤5) \| Known rooms(≤5) \| Partner …"`，不调用 LLM |
| 阶段 2：抽象技能匹配 | `hierarchical_retrieval.py:261-290` `match_abstract_skills` | query 与 `name + description` 的 embedding 算 cosine，阈值 0.3。技能向量在 :129-140 构建 |
| 阶段 3：实例检索 | `hierarchical_retrieval.py:292-362` `retrieve_instances` | 状态上下文与 `context + demo` 算 cosine。实例向量在 :144-151 构建 |
| 阶段 4：可执行性过滤 | `hierarchical_retrieval.py:364-403` `filter_executable` | |
| ├ individual | `:405-420` | 只检查 `agent_hands_empty` / `agent_holding_object` 两个字符串 |
| └ cooperation | `:422-475` | `trigger_conditions` 为空就放行（各库 97–99% 的 cooperation 技能都是空的）；否则 instruction 含 `and/then/…` 子串或 `abstract_score>0.5` 也放行。`preconditions` 和 `partner_context_pattern` 完全不读取 |

## 之后每一步做什么

| 步骤 | 位置 | 说明 |
|---|---|---|
| 需要 replan 时调用 LLM | `llm_planner.py:751-757` → `:666-693` `replan` | `self.llm.generate(self.curr_prompt, …)`（:677 或 :687）。prompt 里的 `rag_examples` 始终是开局那一批 |
| LLM 输出格式 | `habitat_llm/conf/instruct/rag_prompt_sequential_cooperation_skills_v4.yaml:17-20` | `Thought: … / Action[parameter]`，没有「先预测 skill」这一步 |
| 每轮追加进 prompt 的内容 | `llm_planner.py:566-620` `_add_responses_to_prompt` | 只追加 `Agent_i_Observation: <动作执行回执>`。物体状态更新只在 `centralized=True` 时追加，而本配置为 False（`conf/planner/llm_zero_shot_react_planner.yaml:15`） |
| 状态变化只用来触发 replan | `llm_planner.py:848-906` | `previous_objects_state` 每步都更新，但变化内容不会写进 prompt，也不会重新检索 |

## 看起来像、但不在评测路径上的代码（别引用）

| 位置 | 为什么不算 |
|---|---|
| `our_method/planner_integration.py:225-254` `HierarchicalSkillPlanner.step`（每步 perceive → decide/retrieve → act，:64-147） | 这正是稿件 Method 描述的循环，但 `habitat_llm` 里没有任何地方导入它，只在 `our_method/__init__.py:54` 被 re-export，并在 `run_build_memory.py:168-174` 的使用说明里打印出来 |
| `habitat_llm/planner/rag.py:852` `retrieve_hierarchical` | 没有任何调用方 |
| v4 prompt 的 `rag_retrieval:` 块（`rag_prompt_sequential_cooperation_skills_v4.yaml:29-35`） | 指向的 `utils.rag_retriever` 函数不存在，也没有代码读取这个配置 |
