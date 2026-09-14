# HANDOVER 2026-09-14b — PARTNR 类型化要求接口落地；VIKI-L2 ICLR 补实验待开工

接 `HANDOVER-2026-09-14.md`。那份的「下一步：改要求接口」已经做完，它的 §2 现场已过时。

## 0. 为什么交接

会话从 09-14 11:20 跑到 17:55，未压缩，上下文远超 400K；用户要开新会话做 VIKI-L2 的 ICLR 补实验。

## 1. 移动了结论的读数

结论本体在 memory **`partnr-typed-candidate-interface`**（数字、口径、投影各版本收益都在那里），这里只列会影响下一步判断的：

- **旧意图打分器的 exact 口径结构性低估**（`is_in_room` 的 GT 目标是区域名、只认第一个可接受目标/实例）。
  `partnr-reask-does-not-help-7b` 里的 0.061/0.055 是这个口径；公平口径下 7B 自由生成 consistent 召回约 0.09。已在该 memory 里加注。
- **组合轴上 typed 臂比自由生成臂衰减得多**（train 配对 comp/R 0.65 [0.54,0.80] 对 0.84 [0.46,1.55]）；
  自由生成臂平是因为它贴地板。**state_success 在 R_T / R_S_T 上两臂都约为 0**：时序没解决，typed 臂的阶段顺序只来自行序，planner 不读 DAG。
- train_mini 120 集仿真配对（7B，iir1，全部 120/120，各崩 1 集为基准感知 bug `floor_closet_1`）：
  typed v7b 对 自由生成 percent_complete 0.577 vs 0.210（+0.367 [+0.293,+0.442]），**state_success 0.383 vs 0.067（+0.317 [+0.233,+0.400]）**；
  v7b 对 v5 投影 +0.032 [+0.004,+0.064]，0 步集 15→2。**这是 train，而且投影在同一池子上调过。**
- 组合泛化协议（用户 17:00 提出）：iir1 库纯 R；inside 先验用纯 R 训练集重算判定完全不变；但 train 调出的提示词第三条示例是两阶段（T）。
  已改为开关 `typed_examples` R / RS / RST；train_mini 离线 R 0.666 / RS 0.659 / RST 0.651，去掉 T 不掉分；只在 R 集上挑投影规则也选出同一套 v7b。

## 2. 现场（17:54 实测）

- **val_mini 两格在跑**（提交 `f96ba0f`，369 集，32 进程，8h 硬超时 + 30 分钟 stall 守卫）：
  - `outputs/cand_iface_0914/val_mini/typed_v7b_R_7b`：:8063 / GPU 1，runner PID 2374217，**253/369**，近 20 集每 25 s 一集 → 预计 **18:45–19:00**。
  - `outputs/cand_iface_0914/val_mini/typed_v7b_RS_7b`：:8064 / GPU 2，runner PID 2374211，**310/369**，每 22 s 一集 → 预计 **18:20–18:30**。
  - 两格 0 Traceback；示例集已核实（R 格无 next-to 示例，RS 格有；两格都无两阶段示例与阶段规则）。
  - 驱动收尾写 `CELL.json`（含 fake_episodes、commit）。**判死活看 PID 和 stats mtime，不看 CELL.json。**
- **本会话里挂的报表轮询和 VIKI 审计代理会随会话结束消失**，新会话要自己出报表、自己重做审计。
- 端点：:8061 GPU0（PID 1794004）、:8063 GPU1 / :8064 GPU2 / :8065 GPU4（PID 1903234/5/6），全是 qwen2.5-vl-7b，16k。
  val_mini 两格结束后 :8061/:8065 一直空闲。**不用就按 PID 停掉还卡**——VIKI 任务要 72B（TP=4）。
- 箱子负载 175–200/180（有别人的作业），GPU 6/7 是别人的。

## 3. 下一步

### 3.1 先收 PARTNR val_mini（两格都有 CELL.json 后，便宜）

```bash
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner
V=outputs/cand_iface_0914/val_mini; B=outputs/headtohead_0913/val_mini_fixed
/root/venvs/partnr/bin/python scripts/partnr_typed_simpair_report.py --pool val_mini \
  --cell typed_R=$V/typed_v7b_R_7b typed_RS=$V/typed_v7b_RS_7b react_7b=$B/react_7b \
         react_rag_R_7b=$B/react_rag_R_7b gmemory_7b=$B/gmemory_7b memento_7b=$B/memento_7b v2_intent_7b=$B/v2_accepted_7b \
  --compare typed_R:react_7b typed_R:react_rag_R_7b typed_R:gmemory_7b typed_R:memento_7b typed_R:v2_intent_7b \
            typed_RS:react_7b typed_RS:react_rag_R_7b typed_RS:gmemory_7b typed_RS:memento_7b typed_RS:v2_intent_7b typed_RS:typed_R \
  --json outputs/cand_iface_0914/val_mini_reports/val_mini_typed_R_RS.json
```
先看两格 369/369、fake_episodes、崩溃数；**只报一次，不在 val_mini 上调**。基线在组合轴上的对照表已由同脚本自检算过
（`val_mini_reports/baselines_selftest.json`，旧 7B intent 0.223，基线 0.155–0.179）。先验：train 上 typed 约 0.58，val_mini 应明显高于 0.22，但 R_S_T/R_T 的 state_success 仍接近 0。
结果进 memory `partnr-typed-candidate-interface`（替换「val_mini 在跑」那句）。

### 3.2 VIKI-L2 ICLR 2027 补实验（用户交代的主任务）

**完整要求逐字保存在 `TASK-viki-iclr2027-2026-09-14.md`，开工前通读，别凭本文摘要做。**
要点：v3 库（8 skills）、JSON-tolerant scorer、四个 canonical split（id 924 / ood_single_family 924 / cg_image 297 / pure_text 297）、
Figure 2 ToM 12 格（P0）、RQ2 full/no_trace/no_execution_admission、RQ3 full/no_grounding/no_order；
输出到 `results/paper_viki_iclr2027/`，所有数从行级文件自动算，不改论文 LaTeX。

**第一步是只读审计，不动 GPU**（本会话起过但结果到不了新会话）。审计清单：
1. v3 库文件与 SHA；896/23/8 与 no_trace 的归纳产物；归纳脚本有无「不给轨迹」「跳过执行准入」「按族留出 fold 归纳」的钩子；**准入前的 proposal 记录是否落盘**（no_execution_admission 必须复用同一批）。
2. 四个 split 的 manifest 与 SHA，OOD 8 fold 如何拼成 924，fold 库在哪。
3. JSON-tolerant scorer / parser 路径与 SHA。
4. Ours 与各基线的行级文件、字段、行数；**用行级文件重算 72B、7B 的 Ours 是否等于锚点**（不等就停下报告）；「8B」列究竟是 Qwen3-8B 还是旧 30B，7B 是 Instruct 还是 VL。
5. harness 的解码参数、think 开关、检索阈值/top-k、有无 re-ask。
6. 仓库里有无 VIKI ToM 实现或 ToM 指令。
7. no_grounding / no_order 实现与现有 72B v3 消融行级文件（区分 v2 与 comp_cd 半份记忆）。
8. 算力：72B / Qwen3-8B 启动配置、历史每格耗时。
然后给用户缺口矩阵（可复用 / 需写代码 / 需跑）与 P0→P1→P2 的 GPU 排程，再开跑。

相关 memory：`viki-l2-skill-memory-v2`、`viki-l2-scorer-null-artifact`、`viki-l2-crossmodel-baselines`、`viki-l2-fork-per-request-hang`、`viki-harness-was-the-bottleneck`、`viki-172-row-gap-is-interface-not-coordination`（v3 由来）；表格口径看 `RESULTS-2026-09-13.md`。

## 4. 文件与代码

| 路径 | 状态 |
|---|---|
| `29a20f5` | typed 接口：`our_method/skill_memory_v2/partnr_typed_goals.py`、planner `goal_source: typed`、两份 conf、`scripts/partnr_typed_intent.py`（离线内环，`--merge --reproject-graphs` 零模型重打分）、train 配对与 v7b 单格驱动、`scripts/partnr_typed_simpair_report.py` |
| `f96ba0f` | `typed_examples` 开关（R/RS/RST）、val_mini 驱动 `scripts/drivers/partnr_typed_val_mini.sh`（`EXAMPLES=R|RS`）、报表加 `--cell` 与组合轴 |
| 本文件、`TASK-viki-iclr2027-2026-09-14.md`、`CLAUDE.md` | **未提交** |
| `third_party/GMemory` 删除、`third_party/semantic_exploration` | 不要提交 |
| 远端 `results/partnr_object_kinds_train.json`（106 类 = HSSD 类别表）、`results/partnr_inside_prior_train{,_R_only}.json` | planner 运行时读，不在 git |
| 远端 `outputs/cand_iface_0914/` | `train_mini/ceiling`（403 集全观测图）、`train_mini/r1/{p1_*,p2_*,reproj*,examples,full403_v5}`、`simpair_train_mini/{intent_7b,typed_7b,typed_v7b_7b,report_three_way.json,compositional_axis.json}`、`val_mini/`、`val_mini_reports/` |

已知未修：两件同类物体时重复行靠 `copies_allowed` 从指令数词推断，不认识的量词会少一条；typed 臂没有时序 DAG；inside 先验来自 rec/gate 录制池而非全 train。

## 5. 坑（本轮新增）

- **ssh 单引号命令里的 python f-string 用 `r['x']` 会把引号吃掉**（本轮三次 NameError）。用双引号或先赋变量。
- **vLLM temperature 0 也不确定**：同提示词换端点 free 臂差 ~0.03；typed 臂恰好一致。别把 <0.03 当信号。
- **`typed4` 那种带表头行的提示词撞 planner 的 `"\n\n"` stop**（221/342 只出了表头）。
- **`--shard k/n` 依赖文件列表**：ceiling 还在写图时各片起跑时间不同会错位；先做符号链接快照再分片。
- **`kill` 背景 ssh 等待器 ≠ 远端作业死**：本机 ssh 断线时等待器报 255，远端作业照跑；轮询要每次单独 ssh。
- **别在 bash 驱动运行中改驱动文件**（bash 边执行边读）；改动写新文件。
- `partnr_gate_compare.py` 只算 percent_complete；要 state_success 和组合轴用 `partnr_typed_simpair_report.py`。
- 基准感知 bug：`perception_sim.py:792` 找 `floor_<room>` 不存在时整集崩（train 152 集，三臂都崩）。

## 6. 本次改过的 memory

- 新增 `partnr-typed-candidate-interface`（含 v7b、train 仿真配对、组合协议与 R/RS/RST）。
- `partnr-reask-does-not-help-7b`：加打分口径警告（0.061 是旧 exact 口径）。
- `partnr-7b-h2h-crashfix`：注明它是自由生成臂，typed 臂 val_mini 结果待收。
- `partnr-achieve-tool-parity`：「待解决的是模型侧接口」改为指向 typed 接口。
