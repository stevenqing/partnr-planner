# 代理类型总结

本文档总结了所有可用的代理配置类型及其特点。

## 代理类型列表

### 1. **Scout Agent (侦察兵)** - `agent_rules_explore_only.yaml`
**角色描述：** 只能探索和导航，不能操作对象

**能力限制：**
- ✅ 允许动作：`Explore[room_name]`, `Navigate[target]`
- ❌ 禁止动作：`Pick`, `Place`, `Open`, `Close` 等所有操作动作
- ✅ 允许房间：所有房间（可探索所有区域）

**适用场景：**
- 需要快速定位对象位置
- 与其他操作型代理配合，分工明确
- 作为"侦察兵"角色，为其他代理提供信息

**协作方式：**
- 通过探索为其他代理提供对象位置信息
- 其他代理观察状态变化来了解探索结果

---

### 2. **Worker Agent (工人)** - `agent_rules_worker.yaml`
**角色描述：** 只能操作对象，不能探索房间

**能力限制：**
- ✅ 允许动作：`Navigate[target]`, `Pick[object_name]`, `Place[object_name, ...]`
- ❌ 禁止动作：`Explore[room_name]`（不能探索）
- ✅ 允许房间：所有房间（可导航到任何位置）

**适用场景：**
- 对象位置已知或由其他代理提供
- 专注于高效的对象操作
- 与侦察型代理配合使用

**协作方式：**
- 依赖状态信息或其他代理的探索结果来定位对象
- 专注于对象拾取和放置任务

---

### 3. **Manipulation Only (纯操作型)** - `agent_rules_manipulation_only.yaml`
**角色描述：** 只能执行导航和对象操作，不能探索

**能力限制：**
- ✅ 允许动作：`Navigate[target]`, `Pick[object_name]`, `Place[object_name, ...]`
- ❌ 禁止动作：`Explore[room_name]`, `Open`, `Close` 等
- ✅ 允许房间：所有房间

**与 Worker Agent 的区别：**
- 更强调纯操作，不进行探索
- 完全依赖状态信息进行对象操作

**适用场景：**
- 对象位置完全已知
- 需要快速执行大量对象操作任务

---

### 4. **Kitchen Specialist (厨房专家)** - `agent_rules_kitchen_only.yaml`
**角色描述：** 只能访问厨房区域

**能力限制：**
- ✅ 允许房间：`kitchen_1`（仅厨房）
- ❌ 禁止房间：所有其他房间
- ✅ 允许动作：`Explore`, `Navigate`, `Pick`, `Place`
- ❌ 禁止动作：`Open`, `Close`

**适用场景：**
- 任务主要涉及厨房区域
- 需要专门的厨房操作代理
- 与其他区域专家代理分工协作

**协作方式：**
- 处理所有厨房相关的任务
- 与其他区域专家协调对象传递

---

### 5. **Living Room Specialist (客厅专家)** - `agent_rules_living_room_specialist.yaml`
**角色描述：** 只能访问客厅区域

**能力限制：**
- ✅ 允许房间：`living_room_1`（仅客厅）
- ❌ 禁止房间：所有其他房间
- ✅ 允许动作：`Explore`, `Navigate`, `Pick`, `Place`
- ❌ 禁止动作：`Open`, `Close`

**适用场景：**
- 任务主要涉及客厅区域
- 需要专门的客厅操作代理
- 区域分工明确的多代理协作

---

### 6. **Public Areas Agent (公共区域代理)** - `agent_rules_public_areas.yaml`
**角色描述：** 只能访问公共/共享区域

**能力限制：**
- ✅ 允许房间：`kitchen_1`, `living_room_1`, `dining_room_1`, `hallway_1`（公共区域）
- ❌ 禁止房间：`bedroom_1`, `bathroom_1`（私人区域）
- ✅ 允许动作：`Explore`, `Navigate`, `Pick`, `Place`
- ❌ 禁止动作：`Open`, `Close`

**适用场景：**
- 模拟隐私限制（不能进入私人房间）
- 公共区域任务分工
- 与可访问私人区域的代理协作

**协作方式：**
- 处理公共区域的所有任务
- 与可访问私人区域的代理协调对象传递

---

## 代理组合建议

### 组合 1: 侦察兵 + 工人
- **Scout Agent** 负责探索和定位对象
- **Worker Agent** 负责对象操作
- **优势：** 分工明确，效率高

### 组合 2: 区域专家组合
- **Kitchen Specialist** + **Living Room Specialist**
- 每个代理负责自己的区域
- **优势：** 区域专业化，减少冲突

### 组合 3: 公共区域 + 全权限代理
- **Public Areas Agent** 处理公共区域
- 另一个代理（无限制）处理私人区域
- **优势：** 模拟隐私限制场景

### 组合 4: 探索型 + 操作型
- **Scout Agent** 探索所有区域
- **Manipulation Only Agent** 执行对象操作
- **优势：** 快速定位 + 高效操作

## 使用示例

在实验配置中使用这些代理：

```yaml
# 使用侦察兵代理
defaults:
  - /planner@evaluation.planner: llm_planner
  - /planner/llm@evaluation.planner.llm: gpt4o
  - /planner/instruct@evaluation.planner.llm.instruct: agent_rules_explore_only

# 使用厨房专家代理
defaults:
  - /planner@evaluation.planner: llm_planner
  - /planner/llm@evaluation.planner.llm: gpt4o
  - /planner/instruct@evaluation.planner.llm.instruct: agent_rules_kitchen_only
```

## 自定义代理

如果需要创建自定义代理，可以：
1. 复制 `agent_rules_template.yaml`
2. 根据需求设置 `allowed_rooms` 和 `allowed_actions`
3. 设置 `prohibited_rooms` 和 `prohibited_actions`
4. 调整行为规则以适应特定角色

参考 `AGENT_RULES_README.md` 获取详细说明。
