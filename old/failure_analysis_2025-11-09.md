# 失败案例分析报告 - 2025-11-09_23-25-34

## 整体性能指标

- **总episodes**: 57
- **失败episodes**: 35 (61.4%)
- **平均成功率**: 38.6%
- **平均完成度**: 59.0%
- **平均重规划次数**: Agent 0 = 36.7次, Agent 1 = 39.4次

## 关键问题统计

1. **达到重规划阈值(51次)**: 31/35 (88.6%)
2. **Agent 1更容易失败**: 27/35 (77.1%) 达到阈值 vs Agent 0的54.3%
3. **任务类型分布**:
   - 涉及cabinet: 4个
   - 涉及shelves: 3个
   - 复杂任务(多对象): 12个

## 失败模式深度分析

### 模式1: 状态理解错误 (最严重)

**案例**: Episode 1007, Agent 0

**问题描述**:
- Place操作失败: "No valid placements found for entity cabinet_28"
- Agent的Thought一直说"Now that the lamp is placed in the cabinet"
- 但实际状态显示: `lamp_0: held by the agent` (持续51次)
- Agent陷入循环: Navigate[cabinet_28] -> Navigate[table_12] -> Pick[vase_1] (失败，因为还拿着lamp_0) -> 重复

**根本原因**:
1. Agent没有正确理解Place失败后的状态
2. Agent忽略了错误信息，继续假设操作成功
3. Agent没有检查Objects状态来验证操作是否成功

**Thought示例**:
```
Thought: Now that the lamp is placed in the cabinet, I can navigate back to the table to pick up the vase.
```
但实际状态: `lamp_0: held by the agent`

### 模式2: 对象位置理解错误

**案例**: Episode 1007, Agent 1

**问题描述**:
- vase_1实际在`table_12`，但Agent一直尝试Navigate[table_19]
- Pick[vase_1]失败: "Not close enough to the object"
- 陷入循环: Navigate[table_19] -> Pick[vase_1] -> 失败 -> 重复

**根本原因**:
1. Agent混淆了对象位置（table_12 vs table_19）
2. Agent没有仔细读取Objects状态信息
3. Agent没有使用FindObjectTool来确认位置

**Thought示例**:
```
Thought: I will navigate to table_19 to pick up the vase.
```
但实际状态: `vase_1: table_12 in living_room_1`

### 模式3: 重复动作检测缺失

**案例**: Episode 153

**Agent 0问题**:
- 步骤10-36: Explore[living_room_1] -> Navigate[table_21] 重复20多次
- 没有意识到toy_construction_set_1已经被Agent 1处理了

**Agent 1问题**:
- 步骤20-50: Pick[toy_bee_0]重复30多次
- toy_bee_0已经被Agent 0放置到shelves_19，但Agent 1还在尝试Pick

**根本原因**:
1. 没有检测重复动作的机制
2. Agent没有意识到对象已经被其他agent处理
3. 状态更新可能不及时

### 模式4: 容器操作失败

**案例**: Episode 1007, 1065, 153

**问题描述**:
- Place操作在cabinet/shelves上经常失败
- "No valid placements found for entity cabinet_28"
- Agent不知道如何正确处理容器操作

**根本原因**:
1. 容器操作需要特殊处理（within vs on）
2. Agent可能使用了错误的空间关系
3. 缺少容器操作的明确指导

## 具体失败案例详细分析

### Episode 1007 - Agent 0

**任务**: Move the lamp and vase from the living room table to the kitchen cabinet

**失败流程**:
1. 步骤51: Place[lamp_0, within, cabinet_28] → **失败**: "No valid placements found"
2. 步骤52-102: 陷入循环
   - Navigate[cabinet_28] → 成功
   - Navigate[table_12] → 成功
   - Pick[vase_1] → **失败**: "The arm is currently grasping lamp_0"
   - 重复51次

**关键问题**:
- Agent的Thought一直说"lamp is placed"，但状态显示"held by agent"
- Agent没有意识到Place操作失败了
- Agent没有尝试不同的Place方式（如改为"on"而不是"within"）

### Episode 1007 - Agent 1

**失败流程**:
1. 步骤30-346: 陷入循环
   - Navigate[table_19] → 成功
   - Pick[vase_1] → **失败**: "Not close enough"
   - 尝试Navigate[left, table_19] → **失败**: "Node not present"
   - Navigate[right, table_19] → **失败**: "Node not present"
   - 重复

**关键问题**:
- vase_1实际在table_12，但Agent一直尝试table_19
- Agent没有读取Objects状态来确认位置
- Agent尝试了无效的导航命令（left/right修饰符）

### Episode 153

**任务**: Move the toy bee and toy construction set from the living room table to the closet shelves

**Agent 0失败**:
- 步骤10-36: Explore[living_room_1] -> Navigate[table_21] 重复26次
- 没有意识到toy_construction_set_1已经被处理

**Agent 1失败**:
- 步骤20-50: Pick[toy_bee_0]重复30次
- toy_bee_0已经在shelves_19，但Agent 1还在尝试Pick

## 失败原因总结

### 1. 状态理解问题 (最严重)
- **现象**: Agent没有正确理解操作失败后的状态
- **表现**:
  - Place失败后仍认为对象已放置
  - 忽略错误信息和状态更新
- **影响**: 导致无限循环，88.6%的失败都达到重规划阈值

### 2. 对象位置混淆
- **现象**: Agent混淆对象位置或使用错误的位置信息
- **表现**:
  - 尝试Navigate到错误的位置
  - 没有使用FindObjectTool确认位置
- **影响**: 导致Pick操作反复失败

### 3. 重复动作检测缺失
- **现象**: Agent重复执行相同动作而不自知
- **表现**:
  - 没有检测重复动作的机制
  - 没有意识到对象已被其他agent处理
- **影响**: 浪费大量重规划次数

### 4. 容器操作处理不当
- **现象**: 在cabinet/shelves等容器上Place操作失败率高
- **表现**:
  - 使用错误的空间关系（within vs on）
  - 不知道如何正确处理容器
- **影响**: 导致任务无法完成

### 5. Agent 1更容易失败
- **现象**: Agent 1达到重规划阈值的比例(77.1%)远高于Agent 0(54.3%)
- **可能原因**:
  - Agent 1在sequential模式下看到的是更新后的状态
  - 状态更新可能有延迟或错误
  - Agent 1需要推断Agent 0的动作，可能理解错误

## 改进建议

### 1. 强化状态验证
- 在Thought中强制要求验证操作结果
- 要求Agent明确检查Objects状态
- 如果操作失败，必须尝试不同策略

### 2. 增强重复检测
- 在代码层面检测重复动作
- 在prompt中明确禁止重复相同动作
- 如果检测到重复，强制改变策略

### 3. 改进错误处理
- 操作失败后，必须分析失败原因
- 不能简单重复相同动作
- 需要尝试替代方案

### 4. 优化容器操作
- 在prompt中明确容器操作的正确方式
- 提供容器操作的示例
- 如果within失败，尝试on

### 5. 改进Agent 1的状态理解
- 强化状态比较逻辑
- 确保Agent 1能正确理解Agent 0的动作结果
- 可能需要改进状态更新机制
