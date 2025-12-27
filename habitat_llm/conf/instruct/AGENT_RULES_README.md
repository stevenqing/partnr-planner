# 代理规则提示模板使用指南

本目录包含用于定义代理行为规则的提示模板文件。

## 文件说明

### 1. `agent_rules_template.yaml`
这是一个模板文件，包含可自定义的规则占位符。你可以根据需要填充这些占位符来创建自定义的规则集。

**主要规则类别：**
- `{allowed_rooms}` - **硬约束**：允许访问的房间列表
- `{allowed_actions}` - **硬约束**：允许执行的动作列表
- `{prohibited_rooms}` - **硬约束**：禁止访问的房间列表
- `{prohibited_actions}` - **硬约束**：禁止执行的动作列表
- `{behavioral_rules}` - 核心行为规则
- `{collaboration_rules}` - 协作规则
- `{error_handling_rules}` - 错误处理规则
- `{task_completion_rules}` - 任务完成规则

### 2. `agent_rules_example.yaml`
这是一个完整的示例文件，展示了如何填充模板中的规则占位符。你可以参考这个文件来创建自己的规则配置。

## 使用方法

### 方法 1: 使用模板文件创建自定义规则

1. 复制 `agent_rules_template.yaml` 并重命名为你的配置文件名（例如：`my_custom_rules.yaml`）

2. 填充各个规则占位符，**特别注意硬约束部分**：

```yaml
prompt: |-
  {system_tag}You are an expert multi-agent planning agent...

  ### ⚠️ HARD CONSTRAINTS - YOU MUST STRICTLY FOLLOW THESE RULES ⚠️

  **Allowed Rooms:**
  {allowed_rooms}
  # 例如：
  # - kitchen_1
  # - living_room_1
  # - dining_room_1

  **Allowed Actions:**
  {allowed_actions}
  # 例如：
  # - Explore[room_name]
  # - Navigate[target]
  # - Pick[object_name]
  # - Place[object_name, relation, receptacle, ...]

  **Prohibited Rooms:**
  {prohibited_rooms}
  # 例如：
  # - bedroom_1
  # - bathroom_1

  **Prohibited Actions:**
  {prohibited_actions}
  # 例如：
  # - Open[object_name]
  # - Close[object_name]

  ### Core Behavioral Rules:
  {behavioral_rules}
  # 替换为你的具体规则，例如：
  # 1. Always explore unexplored rooms first
  # 2. Never repeat the same action more than twice
  # ...
```

3. 在实验配置中引用你的新提示模板

### 方法 2: 直接使用示例文件

如果你觉得示例文件中的规则适合你的需求，可以直接使用 `agent_rules_example.yaml`。

## 规则定义建议

### ⚠️ 硬约束 (Hard Constraints) - 最重要
这些是**绝对限制**，代理在任何情况下都不能违反。

#### 允许的房间 (Allowed Rooms)
定义代理**只能**访问的房间列表：
- 列出所有允许的房间名称（例如：`kitchen_1`, `living_room_1`）
- 代理不能探索、导航到或与不在列表中的房间内的对象交互
- 如果任务需要访问不允许的房间，代理必须与其他代理协调或寻找替代方案

#### 允许的动作 (Allowed Actions)
定义代理**只能**执行的动作列表：
- 列出所有允许的动作（例如：`Explore`, `Navigate`, `Pick`, `Place`）
- 代理不能使用不在列表中的任何动作
- 如果任务需要不允许的动作，代理必须与其他代理协调

#### 禁止的房间 (Prohibited Rooms)
定义代理**绝对不能**访问的房间列表：
- 即使任务似乎需要，代理也不能进入这些房间
- 用于明确限制某些区域

#### 禁止的动作 (Prohibited Actions)
定义代理**绝对不能**执行的动作列表：
- 即使任务似乎需要，代理也不能使用这些动作
- 用于限制代理的能力范围

**示例：创建一个只能探索和导航的代理**
```yaml
**Allowed Rooms:**
- kitchen_1
- living_room_1

**Allowed Actions:**
- Explore[room_name]
- Navigate[target]

**Prohibited Actions:**
- Pick[object_name]
- Place[object_name, relation, receptacle, ...]
```

### 行为规则 (Behavioral Rules)
定义代理的基本行为模式：
- 探索策略
- 动作选择优先级
- 状态感知要求

### 协作规则 (Collaboration Rules)
定义多代理协作的行为：
- 如何与其他代理协调
- 如何避免冲突
- 如何分工合作

### 动作选择规则 (Action Selection Rules)
定义动作选择的逻辑：
- 动作优先级
- 动作序列要求
- 动作有效性检查

### 错误处理规则 (Error Handling Rules)
定义如何处理失败和错误：
- 失败分析要求
- 恢复策略
- 自适应学习机制

### 任务完成规则 (Task Completion Rules)
定义任务完成的判断标准：
- 验证要求
- 完成条件
- 何时使用 Done[]

### 安全和约束 (Safety and Constraints)
定义安全限制和约束：
- 动作有效性检查
- 资源管理
- 物理约束

## 在实验中使用

在你的实验配置文件中，可以通过覆盖提示配置来使用这些规则模板：

```yaml
# 在你的实验配置中
defaults:
  - /planner@evaluation.planner: llm_planner
  - /planner/llm@evaluation.planner.llm: gpt4o
  - /planner/instruct@evaluation.planner.llm.instruct: agent_rules_example  # 使用你的规则模板
```

## 注意事项

1. **占位符格式**：确保使用正确的占位符格式，如 `{system_tag}`, `{agent_role_description}` 等
2. **格式要求**：严格遵守输出格式要求，特别是动作格式必须使用方括号 `[]`
3. **规则一致性**：确保定义的规则之间没有冲突
4. **测试验证**：在使用新规则前，建议进行小规模测试验证

## 自定义规则示例

### 示例 1: 限制代理只能访问特定房间

```yaml
### ⚠️ HARD CONSTRAINTS ⚠️

**Allowed Rooms:**
- kitchen_1
- dining_room_1

**Allowed Actions:**
- Explore[room_name]
- Navigate[target]
- Pick[object_name]
- Place[object_name, relation, receptacle, ...]

**Prohibited Rooms:**
- bedroom_1
- bathroom_1
- living_room_1
```

### 示例 2: 限制代理只能执行探索和导航动作

```yaml
### ⚠️ HARD CONSTRAINTS ⚠️

**Allowed Actions:**
- Explore[room_name]
- Navigate[target]

**Prohibited Actions:**
- Pick[object_name]
- Place[object_name, relation, receptacle, ...]
- Open[object_name]
- Close[object_name]
```

### 示例 3: 完整的行为规则示例

```yaml
### Core Behavioral Rules:
1. Always prioritize safety over speed
2. Verify object locations before manipulation
3. Never explore the same room twice

### Collaboration Rules:
1. Communicate your intentions clearly in Thought section
2. Avoid working on objects the other agent is handling
3. Coordinate task division to maximize efficiency
```

## 更多信息

有关如何扩展提示模板的更多信息，请参考项目文档中的 `docs/extending.md`。
