# D 部分零 call 层报告（D.1.1、D.1.2）2026-09-22

## DISCREPANCIES

1. 稿件把 gate 写成三条判据 (i) effect 复现、(ii) necessary actions、(iii) 跨轨迹复现，加上 privileged state restoration。验证 is_in_room 时用的 PARTNR gate（A 仓库）不是这个形式。它是配对执行门：在 60 个 train episode 的 gate 池上，带候选算子的库和不带的库各跑一格，比较 `task_percent_complete`。验收要四条同时成立：格完整、bootstrap CI 排除 0、按增益降序取时有边际覆盖、压的格不多于抬的格。见 D.1.1。
2. 结论：**D.1.2 的可行性关没过**（spec 口径 0%，宽口径 53.6%，门槛 80%）。按 spec 走 **结局 D-E3**，不跑 D.1.3 和 D.2。

## D.1.1 现有 gate 的盘点（A 仓库，`/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner`）

产物位置：
- 库：`results/partnr_operators.json`（21 个算子，base）、`results/partnr_operators_iir1.json`（22 个，`accepted: [0]`，`adjudicated_on: gate_iir`）
- 池：`results/partnr_pools/ledger.json`，`gate_iir` 是 60 个 train R episode，`require_key: is_in_room`，seed 20260907；确认池是 `conf_iir`（60 个）
- 脚本：`scripts/partnr_gate_batch.py`、`scripts/partnr_gate_compare.py`、`scripts/partnr_gate_adjudicate.py`、`scripts/partnr_groundable.py`（免费落地关）、`scripts/drivers/partnr_gate_cell.sh`
- 配置：`habitat_llm/conf/baselines/skill_memory_v2_oracle_goals.yaml`（`goal_source: oracle`，partial_obs 为 True）
- 产物：`outputs/gate/iir1/{SUMMARY,ADJUDICATION}.json`、`outputs/confirm/iir1/compare.json`

六个问题：
1. **claimed transition 怎么声明**：算子自带 `effect = {key: <PARTNR 评测谓词>, subject: ?x, value: ?y}`，例如 `is_in_room(?x, ?y)`。这是评测函数里的一个命题，不是世界图的差。body 是带变量的 tool call 列表，例如 `[Navigate ?x][Pick ?x][Navigate ?z1][Place ?x, on, ?z1, none, none]`。
2. **怎么重绑定实体**：`?x`/`?y` 取 validation episode 评测函数里对应命题的参数（privileged 臂直接读 episode 自己的命题）。spare 变量 `?zN` 的角色从 body 里读出来（container / beside / receptacle / station，`our_method/skill_memory_v2/partnr_memory.py` 的 `_roles_in`），再到该 agent 的世界图里按角色找实体（`WorldView.container_of`、`furniture_in_room` 等）。按角色和类型自动做，不按名字，也不人工指定。
3. **判据 (i) 的通过条件**：没有「n 个 validation episode 里成功 k 个」这种逐算子判据。通过条件是整池配对比较，即上面四条同时成立。is_in_room 的 cand0 在 gate_iir 上 Δ +0.6014，CI [0.456, 0.738]，抬 41 格、压 7 格。
4. **判据 (ii) 怎么检查 necessary actions**：没有逐动作的必要性检查。免费落地关（`partnr_groundable.py`）只查 body 在一个桩场景里能否实例化出动作。`normalize()` 拒掉 body 里 Pick/Place 了 `?x` 以外物体的算子。多余或缺失的动作只能间接反映在执行增益上：09-07 双向校准里，多一步 Open 的变体 +0.2583，无 Pick 为 0，只 Pick 为 −0.0375。
5. **判据 (iii) 的跨轨迹复现怎么计数**：不按轨迹计数。复现体现在 gate 池里 60 个互不相同的 train episode 上的配对增益，外加「边际覆盖」：按增益降序取候选时，后面的候选必须抬起前面没抬起的格。另有一个不相交的确认池 conf_iir 只跑一次（Δ +0.6717）。
6. **privileged state restoration 恢复了什么**：PARTNR 这套门里没有这个步骤。每个 validation episode 都从数据集的初始状态 reset。这里的「privileged」指 planner 直接拿 episode 的评测命题当目标（oracle goals），不经过模型。

## D.1.2 R 库和 S 库的可执行性（可行性关）

库：B 仓库本地副本 `~/restore_isambard/partnr-planner/data/hierarchical_skill_memory/`，`hierarchical_rerange_only`（223 个 individual + 227 个 cooperation = 450）和 `hierarchical_spatial_only`（14 + 6 = 20）。条数与稿件一致。sha256 见 `answers.json`。

脚本 `scripts/iclr_master/D/executability.py`，逐 instance 结果在 `per_instance.csv`，汇总在 `executability_summary.json`。

| 量 | R（6,125 个 instance） | S（36 个） | 合计 |
|---|---|---|---|
| `demo` 能确定性解析成 tool call | 0 | 0 | 0 / 6,161 |
| `context.action_sequence` 全部可解析（动词合法、参数具体） | 4,029 | 24 | 4,053 / 6,161 (65.8%) |
| 记录了来源 episode | 145（`episode_<id>_partial/patched`） | 0 | 145 / 6,161 (2.4%) |
| 记录了步区间 | 0 | 0 | 0 |
| 能从自身字段推出 transition（Place→is_on_top/is_inside，Clean/Fill/PowerOn→对应谓词） | 1,606 | 10 | 1,616 / 6,161 (26.2%) |
| 只操作一个物体（能放进 gate 的单主语算子形式） | 2,338 | 14 | 2,352 |

skill 层（至少一个 instance 同时满足）：
- spec 口径（demo 可解析且可导出 transition）：**0 / 470 = 0%**
- 宽口径（action_sequence 可解析且可导出 transition）：**252 / 470 = 53.6%**
- 再加「只操作一个物体」：138 / 470 = 29.4%

门槛 80%，两个口径都没过。原因：
- `demo` 是自然语言（例如 "Navigate to the toy fire truck"），不调 LLM 解析不了；
- 来源 episode 和步区间基本没记（只有失败 episode 打补丁时记了 id），所以不能从来源轨迹前后的世界图之差导出 transition；
- 能从自身动作序列推出谓词的 instance 只占 26%，其余是纯 Navigate/Pick 片段，或者是没有动作序列的 cooperation pattern；
- 格式也对不上：这些 instance 是具体实体 id、多物体交错的片段，gate 要的是带变量、单主语、带 effect key 的算子。

## 结局

**D-E3**。不跑 B 层。Limitations 的预写句要填 B%，两个口径给出的数不同，请用户选：
- spec 口径是 0%。「fewer than 0%」这句话不成立，只能改写，例如 None of the PARTNR demonstrations can be turned into tool calls without a language model, so the gate is not applied to those libraries.
- 宽口径是 54%，照预写句填：The PARTNR skills are stored as text demonstrations, and fewer than 54\% of them can be executed automatically, so the gate is not applied to those libraries.

main.tex pass10 不在盘上，`main_patch.diff` 只给出要插入的句子和位置说明（Limitations 段，净增 1 行）。

## Future（不跑）

- 把有动作序列、只操作一个物体的 138 个 skill 抬升成带变量的算子，再过 gate。这需要新写一个抬升器，spec 没有定义，所以没做。
