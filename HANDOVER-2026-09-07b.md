# HANDOVER 2026-09-07b — PARTNR：归纳出 `is_in_room` 算子，执行门验收成立

接 `HANDOVER-2026-09-07.md`。那份里「**PARTNR 的单算子验收门经校准证明不存在**」这句
**只对 trace-matching 那个门成立**；它自己提议的替代路线（外层执行门）今天建起来了、
双向校准通过、并且成本比当时估的低两个数量级。见下。

## 〇、这个仓库在做什么（新会话先读这一节）

**要证的一件事：技能记忆里的算子可以由 agent 从轨迹里推导出来，而验收全部是机械的。**
目的是消掉「你们的记忆是针对这个 benchmark 手工调出来的归纳器」这条批评。
协议文档 `docs/AGENTIC-OPERATOR-INDUCTION.md`，投稿用 Method/Validity 在
`docs/AGENTIC-FRAMEWORK-METHOD.md`。纪律只有一条总纲：**agent 只提议，模拟器永远说了算。**

两条 benchmark 是分工，不是重复（`docs/COMPOSITIONAL-GENERALISATION.md`：
**PARTNR 扣维度、VIKI 扣共现**）：

- **VIKI-L2 = 受控试验台。** 归纳无损（2698 episode / 5147 completion，丢弃 0.0%），
  手写归纳器是**已知正确的靶子**（19 算子 / support 5833 / 自检 200/200），
  所以「agent 只给轨迹能不能推出等价的」是可证伪的。**这半边已经跑完，见 §7。**
- **PARTNR = headroom 在哪。** 规则归纳器读 161 条轨迹丢掉 **39.0%** 的满足命题
  （`is_in_room` 丢 100%、`is_next_to` 丢 86%）。**当前会话做的就是把 `is_in_room`
  这个丢掉的键用 agent 推回来，并用执行门机械验收**——这是本文档 §1–§4 的全部内容。

## 为什么换会话

上一个会话（`207aa021`）上下文约 **409K**（末轮 `cache_creation 382,927 + cache_read 26,448`），
29 轮、中位 2.8 min/轮、最长 23.9 min，无 compaction。缓存整块失效后每轮都要重算全量 prefill——
13:49 的一句「完成了吗」到 13:54:27 才发出第一个 Bash 调用。**不是挂死，是每轮 5 分钟。**

## 一、先说要更正的（这四条都推翻了会话中途报过的数）

1. **诊断引错了产物。** 最初报的 `is_in_room` 基线 **0.0699**、「61/68 step-0 猝死」、其余
   **0.8133**，全部来自 `outputs/sweep_remeasured/`（09-03，**step-0 修复之前**）。**作废。**
   正确的既有基线是 `outputs/sweep_postfix/`（09-05，修复后）**0.7323**，今天干净重跑 **0.7391**，
   step-0 猝死数 **0**。缺口的实质不变（68 格对着 ceiling 0.96 近乎全崩）。
   → **今后一律用 `sweep_postfix`，`sweep_remeasured` 不要再引。**

2. ~~「privileged 臂配对噪声带在 val_mini 上不是 0」~~ —— **band 格 23:52 跑完，band = 0.0**
   （`outputs/report/priv_iir1/band_val_mini.json`，`complete: true`，367 格计分，
   `mean_delta 0.0`、bootstrap [0.0, 0.0]、**0 格移动**；两格同库同配置、同一天、
   `base` 5582 s / `base_rep` 6181 s）。
   → **`+0.1093` CI [0.079, 0.141] 不需要任何噪声修正，站得住。**
   → **但 09-05 与今天那 38/366 格的差异因此不是跑间不确定性**（当天自比重复是逐格一致的），
   而是 09-05 到今天之间**有东西变了**。这条**未结**：要用就查，不用就别引 09-05 那份。

3. **privileged 臂在带 `is_in_room` 的格上不是上界。** 同 19 格：privileged **0.1316** vs
   7B **0.2982**。机制：privileged 照字面读命题，没算子就不动；7B 菜单里也没这个词，
   于是改写成 `is_on_top(物体, 目标房间里的家具)`——而那本来就满足 `is_in_room`。
   → **门上的 +0.60 不能外推到模型臂那一列，必须实测**（[[viki-gate-overstates-benchmark]] 的形状）。

4. **`predicts`（trace-matching 顾问量）对 `is_in_room` 原本恒等于 0**：它要求命题 target 出现在
   动作命名的实体里，而房间永远不被动作命名。补了房间映射才有数（候选 0：matched 109 / predicted 89）。
   仍然只当顾问，不参与验收。

## 二、已定的结果

**缺口定位**——`is_in_room` 一个键。R-only 库 21 算子（`is_on_top`×20、`is_inside`×1），
all-type 库 134 算子 8 个键，**两份都是 0 个 `is_in_room`**。规则归纳器的归因规则要求
「满足步的动作带完成动词且命名命题实体」，而 `is_in_room` 是抱着东西走出来的，没有完成动词。
仓库自己的 `partnr_task_types.classify` 把它归为 **R**，所以这是**归因失败，不是词表外推**
（词表外推的是 `is_next_to`）——材料就在训练池里。

**执行门双向校准**（40 格 train 池，先清空整个 `is_on_top` 键做地板 0.1667，再放回一个）。
**是分级的，不是二值的**：

| 变体 | Δ vs 地板 | 动了几格 |
|---|---|---|
| 出厂算子（正确） | **+0.7375** | 32/40 |
| 正确＋一步冗余 Open（部分正确） | **+0.2583** | 11/40 |
| 倒序体（加载即被免费关拒） | 0.0000 | 0/40 |
| 换 effect key（挂在别的键上、不会被调用） | 0.0000 | 0/40 |
| 无 Pick（**能跑、手是空的**） | 0.0000 | 0/40 |
| 只 Pick（**能跑、只归因一半**） | **−0.0375** | 3/40（都变差） |

后两行是真正「能执行且错」的算子——这正是 trace-matching 那个门做不到的事（那边正确与坏
算子的 precision 分布完全重叠）。

**提议**：72B 在 24 步里交 6 个候选；**no-trace 对照臂 30 步 0 个候选**。
加上「spare 变量必须写 `?z1/?z2`」这一句接口约定 + 环内免费落地关后，72B 第一次就交出正确形状。

**裁决**（`scripts/partnr_gate_adjudicate.py`，四条同时成立：格完整 / bootstrap 区间排除 0 /
按增益降序取时有边际覆盖 / 压的格不多于抬的格）：**只有 cand0 进库，21 → 22 算子**。

    cand0  Δ +0.6014  CI [0.456, 0.738]  抬 41 压 7   ← 收
    cand1  Δ +0.1736  CI [0.086, 0.271]  抬 12 压 0   ← 拒：抬的 12 格 cand0 全抬了
    cand2  Δ +0.0228  CI 含 0            抬  4 压 2   ← 拒
    cand3  Δ +0.0394  CI 含 0            抬 17 压 14  ← 拒（纯噪声搅动）
    cand4/5 同 cand2                                  ← 拒

    进库的算子： [Navigate ?x][Pick ?x][Navigate ?z1][Place ?x, on, ?z1, none, none]

**确认池 `conf_iir`**（60 格，预登记、全程只开这一次、从未参与选择）：

    base 0.2333 → accepted 0.9050    Δ +0.6717    CI [0.542, 0.788]    51/60 格移动

**比门上的 +0.6014 还高，所以没有对门集过拟合。**

**privileged 臂 val_mini 上报**（369 格全覆盖，`complete: true`，成对同时刻同代码）：

| | base（21 算子） | accepted（22 算子） | Δ | 区间 / 移动 |
|---|---|---|---|---|
| 全部（366 计分） | 0.7384 | 0.8477 | **+0.1093** | CI [0.079, 0.141]，71 格 |
| └ 带 `is_in_room`（68） | 0.1987 | 0.8031 | **+0.6044** | 61 格 |
| └ 其余（298） | 0.8616 | 0.8579 | −0.0037 | 10 格 |

**一个归纳出来的算子，在上报 split 上 +0.109 绝对分，增益 100% 落在该落的地方。**
与按命题占比预估的上界 +0.1086 几乎完全吻合——这个键的可回收部分基本被拿满了。

## 三、现场状态（09-08 00:00；**PARTNR 已按指示全部停掉**）

**PARTNR：停了，一个进程都没有。**

- band 格在被停之前自己跑完了（369/369），我补算了比较 → **band = 0.0**，见 §1.2。
  这是这条线今晚唯一还需要的那个数，已落盘。
- 22:08 那一分钟坏掉的两条模型臂（7B 被 4h 硬超时砍在 334/340；8B 对着死端点写了 90 个
  零分假格）**没有重跑**：重跑链 `partnr_model_rerun.sh` 已杀，8B 的污染目录在
  `outputs/stale/m8b_*_deadendpoint_0907_2220`。要恢复就跑
  `scripts/drivers/partnr_model_rerun.sh`（已写好，一次一对、`HARD_TIMEOUT=28800`、
  `partnr_model_cell.sh` 现在带端点心跳）。
- 7B 那半份数（329 共享格 Δ+0.0551）是**有偏子集**，别用，见 §3b。

**VIKI：两个批次在跑，补 §7c 那张表的窟窿。**

| 批次 | 内容 | 端点 | 状态 |
|---|---|---|---|
| `viki_p0_7b.sh` | 7B 四基线 × 三 split = **12 格** | 8061 | zero_shot 924/924 ✅；trajectory_rag 在跑 |
| `viki_memento_gaps.sh` | MEMENTO 30B×2 + 7B×2 = **4 格** | 8062 → 8061 | 30B/text 在跑（该脚本收尾时才落盘，跑中显示 0 行是正常的） |

`viki_memento_gaps.sh` 的 7B 两格会**等 `viki_p0_7b.sh` 退出**再启动（一个端点同时只跑一个
生成作业）。两个批次都带 20 分钟 stall 守卫与硬超时。

## 四、下一步

1. **等两个 VIKI 批次跑完**（7B 十二格 ~2–4 h；MEMENTO 四格与之串行）。
   看进度：`grep '^\[' outputs/p0_7b.log outputs/memento_gaps.log`。
2. **重新生成报表**（生成器已改，见 §7d）：

   ```
   /root/venvs/partnr/bin/python scripts/viki_v2_baseline_comparison.py --models 72B 30B 7B
   ```
   然后核 `missing` 是不是空的。**7B 那一列的旧数字全部作废**，重跑前写过的
   「7B ID 对 G-Memory 90/101」等句子要一并删掉。
3. PARTNR 那两条模型臂什么时候补，看你的优先级——现在完全停着，不占机器。

## 五、文件与代码

- 库：`results/partnr_operators.json`（21，base）/ `results/partnr_operators_iir1.json`
  （22，`accepted: [0]`，`adjudicated_on: "gate_iir"`）
- 池子**预登记**：`results/partnr_pools/ledger.json`——8 池两两不交，全部从 train 切。
  铁律：**`val_mini` 365/365 全在 `val` 里**，门一步不能踩 val/val_mini。
- 门与裁决：`outputs/gate/iir1/{SUMMARY,ADJUDICATION}.json`
- 确认池：`outputs/confirm/iir1/compare.{json,txt}`
- 上报：`outputs/report/priv_iir1/compare_val_mini.{json,txt}`（已定）、`outputs/report/model_iir1/`（在跑）
- 既有基线：`outputs/sweep_postfix/`。**`outputs/sweep_remeasured/` 是修复前的，别用。**
- **git：HEAD 仍是 `0c58f46`（VIKI 那次提交），这条线一行都没提交。**
  已改 tracked：`CLAUDE.md`、`docs/AGENTIC-OPERATOR-INDUCTION.md`、`habitat_llm/llm/vllm_chat.py`
  （加 `extra_body` 透传）、`our_method/skill_memory_v2/partnr_recorder.py`
  （补 `name_to_room`/`room_id_to_name`，工作台加 `room_of`/`rooms` oracle——**这是 PARTNR 侧的一次破冻结**）、
  `scripts/partnr_induction_tools.py`。
  未跟踪新脚本：`scripts/partnr_{make_pool,propose_operators,groundable,gate_batch,gate_compare,`
  `gate_adjudicate,calib_libraries,calib_adversarial,is_in_room_readability}.py`，
  `scripts/drivers/partnr_{gate_cell,model_cell,model_chain,record_pool,band_val,calib_chain,`
  `induction_chain,report_chain,report_now,model_rerun,status}.sh`。

## 六、坑（都真中过）

1. **`CELL.json` 会撒谎。** m8b 两格的 `CELL.json`（mtime 18:31、`status: 137`、`episodes: 0`）
   是**被杀那一次**留下的；重启的跑用同一个目录，要到收尾才覆盖它。
   **判活看 `ps` 里的 `hydra.run.dir=` 和 `stats/` 目录 mtime，别看 `CELL.json`。**
   （我这次就照它误报了一次「8B 死了」。）
2. **端点起飞前检查。** 8B 那两格第一次从头到尾对着死端点跑：418 次 `APIConnectionError` 被
   `VLLMChat` 吞成空串、183 格零分，**看起来完全像「模型不会规划」**。驱动已加：不返 200 就拒绝启动。
3. **Qwen3 必须关思考**（`enable_thinking=False`，经 `extra_body` 透传）。默认输出 `<think>`，
   planner 在空行处停止生成，回来的是半截思考、**可解析 requirement 0 条**——那一列会得零，
   但原因是台子不是模型。实测：7B 解析 2 条 / 8B 思考开 **0 条** / 8B 关思考 3 条。
4. **不要在带残留的目录上重跑。** 第一次 `rm -rf` 报 "Directory not empty" 而没核实，
   残留文件让「9 格里 8 格零分」看起来像模型不行。
5. **放 habitat 进程前先看那块卡上有什么。** GPU 1 上有你起的 Qwen3-8B（8101，0.88 显存），
   把它当空闲导致两次 CUDA OOM。仿真格现在只用 GPU 0/6，GPU 1 留给服务。
6. **不要在脚本运行中改它**：bash 按字节偏移继续读，会读到新代码（表现为 `unbound variable`）。
7. **scp 与 mutagen 会撞车**：一次 scp 上去的脚本被 mutagen 抹掉、作业从没启动。
   改成**写本地让 mutagen 推、核对远端存在后再发**。
8. **比较器默认在交集上比。** cand1 那半格（CUDA OOM，只跑 32/60）因此给出像模像样的 +0.2177，
   基线均值还悄悄从 0.2417 变成 0.3065。已加完整性核对（`complete: false` 时 `gate_batch` 直接拒绝），
   但 `partnr_gate_compare.py` 本身仍会产出残表——**每次读 compare 先看 `complete`**。
10. **起飞前检查挡不住「跑到一半死」**——8B 那次服务 22:08 关掉后两格又烧了 12 分钟、
   写出 90 个零分假格。**假格判据：`sim_step_count == 0`（runtime ≈ 13 s）**，
   别数 `APIConnectionError`（46 个假格只留 20 条错误，差一倍）。**已修**：
   `partnr_model_cell.sh` 现在有运行中的端点心跳 + `fake_episodes` / `endpoint_dead` 两个字段。
11. **vLLM 服务一定要 `setsid nohup` 起。** 22:08:08 那次是**优雅关闭**（日志里 `Shutting down`），
   不是崩、不是 GPU OOM、不是 OOM killer——最可能是被父进程的信号带走的。
12. **模型臂的硬超时不能抄 privileged 臂。** 四格挤在 180 核上时 7B 一格跑满 369 需要 >4h，
   而默认 `HARD_TIMEOUT=14400` 正好在 4h，两格都被砍在 334/340。重跑已改 28800 + 一次只跑一对。

13. **`pkill -f "port 8101"` 会杀掉你自己的 ssh 会话——第七次中招。** CLAUDE.md 里那条
   「`pgrep -f` 自匹配」对 `pkill` 一样成立，而且模式出现在**自己的命令行**里时括号写法也没用。
   **先 `ps -eo pid,cmd | grep "[a]pi_server"` 拿 PID，再 `kill` 那个 PID。**

9. **验收要边际，不能只写「增益 > 噪声带」。** 第一版规则把 6 个候选全收了；查逐格才看清
   cand2–cand5 抬起的每一格 cand0 都已抬起。这是 [[viki-rung-acceptance-must-be-marginal]] 又犯一次。

## 七、这次改了用户 memory

- `viki-agentic-operator-induction`：09-06 那条「PARTNR 承载不了单算子验收门」按今天的校准**限定
  到 trace-matching 门**，并补上执行门成立 + 实测成本（60 格一格 11–15 min，不是当时估的 55 小时）。
- `viki-partnr-v2-port`：里面的 0.6752 / 斜率 0.736 是 **step-0 修复前**（`sweep_remeasured`）的数，
  已标注作废并指向 `sweep_postfix` 0.7323。

## 七、VIKI 那半边：已定，不用重跑

全部数字与判定在 `HANDOVER-2026-09-07.md` 与 `results/agent_library_2026-09-07/`
（`BASELINE_COMPARISON.md` / `FINDINGS_AND_CORRECTIONS.md` / `baseline_comparison.json`）。
盘上产物已核：**54 个 v2 评测格** `results/viki_memory_experiments/amendment11/v2_*.jsonl`、
**11 份并集记忆** `outputs/v2_memories/`（09-07 那份写的是 12，实际 11）。

**主表已经全部落在 agent 库上**——14 族库并集去重后 **4 个算子**（11 进 4 出）；
arm (d) 无轨迹对照 **2016 次提交、0 个算子**（记忆二字成立的关键对照）。

```
72B      ours     G-Memory  MEMENTO  v1      trajRAG  zero-shot
ID       0.6126   0.5097    0.1991   0.1677  0.0671   0.0281    最窄一格 240/145 p=1.5e-06
comp文本  0.8620   0.0337    0.1010   0.0438  0.0000   0.0034    246/0
comp带图  0.7980   0.0471    0.0168   0.0303  0.0067   0.0000    225/2
```

- **comp 两格 4 个算子打平 19 算子的手写参考库**（文本 0.8620 = 0.8620，带图 0.7980 vs 0.7946）
  ——这就是「agent 能推出等价物」的正面证据。
- 30B 五个基线全胜；**7B 的 ID 对 G-Memory 与 v1 不显著且方向偏基线**（90/101 p=0.47、
  90/97 p=0.66），**这一格必须如实写**。
- 组合泄漏审计结局 **A**，19 个产物附 sha256（`audit/comp_leakage_2026-09-05/`）。

**三条不通过 / 要如实写负结果的**：

1. **留出族输给 G-Memory**：ours 168/924 = **0.1818** vs G-Memory 192/924 = **0.2078**，
   配对 ours 胜 70 / G-Memory 胜 94，**p = 0.072** → 按 spec v2 §3 判定条件**这一列不通过**。
   （MEMENTO 折次 6.28% 我们赢，但 G-Memory 是 spec 里唯一的判定对象。）
2. **真实缺口只有 172 行**（set_plate 85、ensure_fruits 79、dog_push 46 中的部分），
   **全部是 `infeasible_assignment`、全部指向 coordination**，而工作台结构上验不了
   coordination（`run_operator` 单执行者）。**放大池子或取消边际验收都修不了它**，(b)/(c) 重建已取消。
3. **comp 半份→全份那条曲线（cut 0.0000 → cut+delivery 0.6869 → all 0.8620）有限定**：
   三格的 Layer 2/3 是按各自池子重挖的（1 规则/24 地点 对 3 规则/41 地点），**不是干净的
   Layer 1 归因**，不能当作「加算子就涨」来写。

**知情不修**：`results/frozen_sweep_v3.json` 已提交但它定义的 (b) coverage pool 已被已知答案
检验否掉，**别用**；`scripts/partnr_agentic_rung.py` 的 precision 门是坏的，PARTNR 现在走的是
`partnr_gate_*.py` 那一套执行门，那个文件里的验收段要重写或删掉。

**机器现状（22:16 实测）**：8050 (`qwen2.5-vl-72b-amendment3-f2`, 192.168.32.40) / 8061
(`qwen2.5-vl-7b`) / 8062 (`qwen3-vl-30b`) 三个端点全 200；**8101 (`Qwen3-8B`) 已死，见 §3b**。
GPU 0 和 6 被 PARTNR 的仿真格占满（97%），GPU 1 空，2–5/7 空闲。

### 7b、09-07 22:50 我从原始格重算了一遍，五条是文档里没有的

用的是 `reason == "SOLVED"` 逐行、McNemar exact，跟 `viki_v2_baseline_comparison.py` 同一口径。

**(1) ID 上 4 算子库输给 19 算子参考库，而且显著。**

    72B/id      ours 0.6126 (566/924)   ref19 0.6742 (623/924)
                配对 ours 胜 115 / ref 胜 172   p = 0.00092   方向：参考库

`FINDINGS_AND_CORRECTIONS.md` 只给了 566 vs 623 这两个数、没做配对检验。
**所以「打平参考库」这句只在 comp 两格成立，ID 那格我们是输的，写论文不能含糊过去。**

**(2) comp 两格不是「打平」，是逐行完全相同。**

    72B/text    ours 0.8620   ref19 0.8620   297 行**逐行一致**，0 胜 0 负
    72B/imaged  ours 0.7980   ref19 0.7980   297 行**逐行一致**，0 胜 0 负

**含义要说清楚**：这两格分不开 4 算子库和 19 算子库——一个完全不同的库给出**每一行都相同**
的结果。所以 0.8620 / 0.7980 不能当作「我们这个库好」的证据，只能说「这两格上两个库等价」。
另：09-07 交接写的 imaged「0.7980 vs 0.7946」与盘上对不上，盘上 ref19 就是 0.7980。

**(3) 两个消融跑在半份记忆上，而且在 text 那格是零效应。**
`viki_v2_evaluate.py:152-161`：`id` 用 `id_memory`，`text`/`imaged` 用 **`comp_cd`（半份）**。
所以正确对照是 `v2_ours_72B_*`（0.6869 / 0.6801），不是主表的 0.8620 / 0.7980：

    消融          split    对照     消融后    配对
    no-grounding  id      0.6126   0.3312   260/0  p=1.1e-78   ← 真效应
    no-grounding  imaged  0.6801   0.0000   202/0  p=3.1e-61   ← 整格塌掉
    no-grounding  text    0.6869   0.6869     0/0  p=1    **297/297 逐行一致**
    no-order      id      0.6126   0.4199   178/0  p=5.2e-54   ← 真效应
    no-order      text    0.6869   0.6869     0/0  p=1    **297/297 逐行一致**
    no-order      imaged  0.6801   0.6801     0/0  p=1    **297/297 逐行一致**

**两条限定**：消融只在 ID（和 grounding 在 imaged）上咬得动，**在 text 组合泛化格上两个消融
都一行都不动**；而且它们**根本没测主表报的那份全份记忆**。

**(4) 留一族消融：8 族里 6 族对 ID 完全冗余，三个模型同形。**

    去掉哪一族                72B              30B              7B
    clear_table            0.374 (346/924)  0.290 (268/924)  0.110 (102/924)
    cut_fruit              0.420 (388/924)  0.355 (328/924)  0.063 ( 58/924)
    其余六族各自去掉          0.613 = 全量     0.499 = 全量     0.140 = 全量

只有 `clear_table`（72B −0.239）和 `cut_fruit`（−0.193）在扛 ID 的分。

**(5) 留出族那一列是两族在扛，六族恰好为 0。**

    72B (0.1818)   parallel_human 86/86、cut_two_fruits 82/108，其余六族 0/220 0/189 0/46 0/82 0/85 0/108
    30B (0.1461)   parallel_human 84/86、cut_two_fruits 48/108、toast_bread 3/108，其余五族全 0
    7B  (0.0335)   parallel_human 28/86、cut_two_fruits 3/108，其余六族全 0

**另外两条**：`v2_notrace_*` 评测格**不存在**——arm (d) 归纳出 0 个算子，没有记忆可评，
所以「2016 次提交 0 算子」是**建库侧的对照，不是评测侧的对照**，措辞要区分。
`baseline_comparison.json` 自己声明缺 4 格：MEMENTO-style 的 30B/text、30B/imaged、7B/text、7B/imaged。

### 7c、只对基线（不含 19 算子参考库）：三个模型 × 四个 split 到底齐没齐

**结论：没齐。ours 12 格全在；基线只有 72B 是完整的，7B 那一列的基线几乎全是假的。**

|  | ID (924) | 留出族 OOD (924) | comp 文本 (297) | comp 带图 (297) |
|---|---|---|---|---|
| **72B** | ours + 5 基线 ✅ | ours + 5 基线 ✅ | ours + 5 基线 ✅ | ours + 5 基线 ✅ |
| **30B** | ours + 5 基线 ✅ | **只有 ours** ❌ | ours + 4（缺 MEMENTO） | ours + 4（缺 MEMENTO） |
| **7B** | ours + **只有 MEMENTO 是真的** ❌ | **只有 ours** ❌ | ours + **0 个真基线** ❌ | ours + **0 个真基线** ❌ |

**7B 的基线是 30B 的存档。** `viki_p0_report.py:61` 写死 `TAG = "m30"`，而 `cell_path` 对
非 72B 一律拼 `.m30`，从不代入模型名。所以 `baseline_comparison.json` 里 30B 与 7B 的
G-Memory / v1 / trajRAG / zero-shot **逐格逐行完全相同**（141 / 137 / 76 / 4）。
`amendment8b` 下**没有任何 `*m7*` 基线存档**，即这些跑从来没做过。
→ **「7B 的 ID 对 G-Memory 不显著且方向偏基线（90/101）」实际是 7B-ours 对 30B-G-Memory，
这个比较不成立，必须撤下或补跑。**（MEMENTO 是例外，它有 `memento_id_m7.jsonl`，
7B/id = 0.0400 是真的。）

**OOD 那一列 `viki_v2_baseline_comparison.py` 从没算过**：`OURS` 字典里写了 heldout，
但主循环只跑 `("id","text","imaged")`。基线的逐族存档其实在盘上
（`results/viki_memory_experiments/amendment8b/folds/<族>/{gmemory,gmemory.shuffled,
skill_memory.fullactions_k8,trajectory_rag,zero_shot}.jsonl`，**只有 72B**），
我按与 ours 相同的拼法 + JSON-tolerant 重算了（09-07 23:20）：

    留出族 OOD，72B                          配对（ours 胜 / 基线胜 / p）
    ours (agent library)   168/924 = 0.1818
    G-Memory               192/924 = 0.2078   70 / 94 / p=0.072    ← 这一列我们输
    skill memory v1        109/924 = 0.1180   89 / 30 / p=5.7e-08
    MEMENTO-style           58/924 = 0.0628  130 / 20 / p=6.0e-21
    zero-shot               32/924 = 0.0346  136 /  0 / p=2.3e-41
    trajectory RAG          30/924 = 0.0325  149 / 11 / p=4.6e-32
    G-Memory shuffled 对照   88/924 = 0.0952   89 /  9 / p=1.1e-17

**shuffled 对照值得单列**：把兄弟族打乱后 G-Memory 从 0.2078 掉到 0.0952，**它在 OOD 上
赢我们的那 0.026 有一半以上来自兄弟族泄漏**（对上 [[viki-l2-ood-folds-negative-results]]）。

ours 的 OOD 三个模型都有：72B **0.1818**、30B **0.1461**、7B **0.0335**。

**要补齐还差什么**（按代价从小到大）：
1. 30B 的 MEMENTO comp 两格（文本 / 带图）—— 2 个格。
2. 7B 的四个基线 × 三个 split —— **12 个格**，这是最要紧的一块，不补就不能报 7B 那一行的基线。
   外加 MEMENTO 的 7B comp 两格。
3. 30B / 7B 的 OOD 基线 —— 5 方法 × 8 折 × 2 模型 = **80 格**。要么补，要么在论文里
   **明写 OOD 的基线对比只在 72B 上做**。

### 7d、09-07 23:40–00:00 为补这张表改了什么

**(1) `viki_p0_report.py` 的 7B 解析已修。** 原来 `cell_path` 对任何非 72B 都拼模块级
`TAG = "m30"`，所以 7B 读的是 30B 的存档。现在有 `MODEL_TAG = {"7B": "m7"}` 与 `tag_for()`；
**30B 的路径一个字节都没变**（它仍走 `TAG`，含 `--tag` 覆盖与 `m30r2`/`m30nt` 那些变体）。

**(2) 新驱动 `scripts/drivers/viki_p0_7b.sh`。** 从 `viki_p0_30b.sh` 派生，只改端点
（8061）、backbone（`qwen2_5_vl_7b` / `qwen2.5-vl-7b`，registry 里本来就有）、tag（`m7`）
与起飞前的端点门。其余条件逐条保留 72B 批次的口径：partner prefix 开着、v1 用
`fullactions_k8`、seed 20260829、**脚本内不打分**。

**(3) 新驱动 `scripts/drivers/viki_memento_gaps.sh`** 跑 `baseline_comparison.json` 自己
声明缺的那四格（MEMENTO 30B/7B × text/imaged）。30B 走 8062，7B 那两格**等 8061 空出来**
再跑。注意 `viki_eval_memento.py` 是**收尾时一次性写文件**的，跑到一半查行数是 0 属正常。

**(4) 报表生成器现在自己出 OOD 那一列。** 原来 `viki_v2_baseline_comparison.py` 的主循环只跑
`id/text/imaged`，OOD 从来没进过产物。现在循环是 `("id","heldout","text","imaged")`，
基线的 heldout 列**按与 ours 完全相同的方式**从 `amendment8b/folds/<族>/<arm>.jsonl` 拼
（MEMENTO 从 `memento_fold_<族>.jsonl`）。**30B/7B 没有 fold 跑，所以显式报 missing，
绝不拿别的模型的存档去填**——那正是当初造出 30B/7B 两列逐行相同的那个错。
冒烟跑过（`--models 72B`，`missing: []`），七行与我手算的逐位一致。

**(5) `G-Memory (shuffled control)` 只在 OOD 那一列作为**对照**输出，代码注释里写死了
「它是检索安慰剂，不是基线，永远不能替代 G-Memory 本身」。** 因为 `A9_GMEM_SHUFFLE=1`
做的是 `gmemory.py:137` 把检索结果随机打乱，测的是「检索本身值多少分」（约 11 点），
**不是「去掉兄弟族泄漏的 G-Memory」**。要真的做无泄漏 OOD，得把兄弟族一起留出
（`cut_fruit_on_board` 与 `cut_two_fruits_on_board` 归一组）**并且两条臂都重跑**——
这是另一件事，没做。

### 7e、09-08 00:15 起：去掉兄弟族泄漏的 OOD split

**为什么要做**：单族留出只拿掉一个**标签**、不是一个**分布**——近似重复的族还留在库里，
arm 检索到它就等于没留出。盘上的证据：`amendment8b/folds/sibling_retrieval.json` 显示
G-Memory 在 `cut_fruit_on_board` 那折的 189 行里有 **171 行检索到 `cut_two_fruits_on_board`**，
而把检索随机打乱的安慰剂让它从 0.2078 掉到 0.0952。

**分组判据（预登记，`results/sibling_folds_preregistration.json`，sha256 3c17384dcfc3eba0）**：
每族 40 条**训练指令**的 MiniLM 均值向量余弦，阈值 **0.85**。
**只用指令、不用任何 arm 的检索或分数**——照着 G-Memory 检索到什么去定分组，就是拿被审的
arm 来设计审它的 split。阈值取 0.85 是因为排序后有可见断层：

    0.9567  serve_bread_after_checking_cabinet  <-> serve_bread_from_counter
    0.9124  cut_fruit_on_board                  <-> cut_two_fruits_on_board          ← 都是折
    0.8588  parallel_human_dual_asset           <-> sequential_pick_two_and_place     ← 折
    ------------------------------- 断层 -------------------------------
    0.8182  serve_bread_after_checking_cabinet  <-> set_plate_and_fork_on_table

所以 **8 折里只有 3 折有兄弟**，另外 5 折的分组就是它自己、旧结果按构造可直接复用
（对没有兄弟的族，两种留出是同一个实验，重跑只会加噪声）。

**必须两条臂一起做**：只给 G-Memory 拿掉兄弟，就是把偏向反过来。所以 ours / G-Memory /
skill memory v1 / trajectory RAG 都重跑；zero-shot 没有记忆、不受影响，行直接复用。
**已知缺口**：MEMENTO 也是记忆臂但没重跑——`viki_eval_memento.py` 的 `--exclude-family`
只吃一个族、没有分组形式，它那一行必须标注为「单族留出」。

**我们这条臂（已出，replay 无模型调用）**：

    族                                          单族留出    兄弟组留出
    cut_two_fruits_on_board                     82/108      54/108
    parallel_human_dual_asset_to_plate_or_bowl  86/ 86      86/ 86   ← 拿掉兄弟一分不掉
    cut_fruit_on_board                           0/189       0/189
    其余五族                                       0           0
    ------------------------------------------------------------------
    72B 合计                                168/924=0.1818  140/924=0.1515
                                            配对 28 胜 0 负 p=7.5e-09
    30B 合计                                135/924=0.1461   88/924=0.0952
    7B  合计                                 31/924=0.0335   28/924=0.0303

**读法**：我们掉 0.030，损失**全部集中在 cut_two_fruits 一族且只掉一部分**（54/108 还在）；
`parallel_human` 的 86/86 与兄弟族无关。**这不是「我们没事」**——30B 掉了 0.051，
而且真正的判据是 G-Memory 掉多少，还在 8050 上跑。

**代码**：`viki_amendment9_folds.py` 加了 `SIBLING_GROUPS` / `sibling_group()` /
`masked_families()`，由 `A9_SIBLINGS=1` 开启；**不开时每个 id 集合与既有产物逐字节不变**。
驱动 `scripts/drivers/viki_sibling_folds.py`（我们，replay）与
`viki_sibling_folds_baselines.sh`（基线，8050，写成 `<arm>.sibgrp.jsonl`，绝不覆盖原档）。
报表生成器新增 `heldout_sibgrp` 这一列，拼法与 ours 完全一致。

### 7f、**兄弟组留出的结果：G-Memory 的 OOD 优势没了，但我们也没赢**

09-08 01:25，72B，JSON-tolerant + McNemar exact：

| arm | 单族留出 | 兄弟组留出 | Δ | 配对（单族赢 / 组赢 / p） |
|---|---|---|---|---|
| ours (agent library) | 0.1818 | **0.1515** | −0.0303 | 28 / 0 / 7.5e-09 |
| **G-Memory** | 0.2078 | **0.1439** | **−0.0639** | 66 / 7 / 3.9e-13 |
| skill memory v1 | 0.1180 | 跑完待评（文件名是 `skill_memory.sibgrp`） | | |
| trajectory RAG | 0.0325 | 最后一折在跑 | | |
| zero-shot | 0.0346 | 无记忆，按构造复用 | — | — |

**这一列上 ours 对 G-Memory：0.1515 vs 0.1439，配对 61 胜 54 负，p = 0.576。**

**措辞纪律**：这不是「我们赢了」。**拿掉兄弟族泄漏后，G-Memory 掉的（−0.064）是我们的两倍
（−0.030），它原来 0.026 的领先消失了——但结果是打平（p=0.576），不是反超。**
能说的是：**单族留出那一列上 G-Memory 的优势主要由兄弟族泄漏构成**，
这与它的 shuffled 安慰剂对照（0.2078 → 0.0952）指向同一件事。
不能说的是我们在 OOD 上强于 G-Memory。

另外要如实写的：**两条臂在这一列都很低**（0.15 上下，对 ID 的 0.61）。
兄弟组留出上我们唯一还剩的非零族是 `parallel_human`（86/86，拿掉兄弟一分不掉）与
`cut_two_fruits`（54/108），其余六族仍是 0。**这一列衡量的东西对两条臂都基本没被解决。**

### 7g、09-08 09:27 发出的最后一块：30B / 7B 的 OOD 基线

`viki_v2_baseline_comparison.py` 在 09-08 01:34 那版里把 23 格判为 missing，其中 22 格是
**30B / 7B 的 OOD 基线**（两个口径都缺，因为盘上只有 72B 有 fold 跑）。现在补：

`scripts/drivers/viki_folds_crossmodel.sh`，`MODEL=30B`（8062）与 `MODEL=7B`（8061）**并行**，
每个模型两遍：

    pass 1  单族留出   4 个 prompt 臂 × 8 折 + MEMENTO × 8 折     每臂 924 行
    pass 2  兄弟组留出 3 个记忆臂 × 3 折 + MEMENTO × 3 折         每臂 383 行
            （zero-shot 无记忆，兄弟组这一列复用它 pass 1 的行，不重跑）

**每一格都用 `--variant <tag>` 命名空间隔开**（`folds/<族>/<arm>.m30.jsonl` /
`.m30sibgrp.jsonl`），所以**绝不会覆盖 72B 的存档**——已实测确认路径分开。
MEMENTO 走 `--exclude-family <族> --only-family <族>`，写成
`memento_fold_m30_<族>.jsonl` / `memento_foldgrp_m30_<族>.jsonl`。

报表生成器相应加了 `fold_path(model, method, family, grouped)`：72B 走既有档名
（v1 是 `skill_memory.fullactions_k8`），非 72B 走 `<arm>.<tag>[sibgrp]`。
**回归已验：只跑 `--models 72B` 重算，30 格与 09-08 那版逐格一致。**

预计 4 小时左右（按昨晚 7B 批次实测 ~28 行/分钟，每模型约 6500 行，两个模型并行）。
跑完自动重出全表到 `results/agent_library_<日期>/`。

**这一批跑完后，表上仍会剩的唯一一格**：MEMENTO 72B 的兄弟组——`viki_eval_memento.py`
的 `--exclude-family` 只吃一个族，72B 那次没有分组跑。它那行必须标注为「单族留出」。
（30B/7B 的 MEMENTO 兄弟组这次跑了，因为是新建的格，直接按分组条件建。）

### 7h、**全表完成（09-08 13:47）——OOD 那一列的结论随模型反转**

`results/agent_library_2026-09-08/`。ID / OOD 单族 / OOD 兄弟组 / comp 文本 / comp 带图
五个 split × 三个模型 × 五个基线，除三格外全齐。

**ID 与 comp：三个模型五个基线全胜，无一例外。**

    ID          72B 0.6126 (次高 G-Memory 0.5097, p=1.5e-06)
                30B 0.4989 (次高 MEMENTO   0.2078, p=4.1e-48)
                7B  0.1407 (次高 G-Memory  0.0768, p=3.8e-06)
    comp 文本    0.8620 / 0.7239 / 0.2559   最强基线分别 0.1010 / 0.1145 / 0.0000
    comp 带图    0.7980 / 0.6498 / 0.1515   最强基线分别 0.0471 / 0.0640 / 0.0000
    7B 的 comp 两格**所有基线整格为 0**。

**OOD：我们一格都没赢，而且「G-Memory 赢在兄弟族泄漏」只在 72B 成立。**

    OOD 单族留出          ours      G-Memory    配对              判定
      72B               0.1818    0.2078      70/94  p=0.072   打平（偏 G-Memory）
      30B               0.1461    0.1494      52/55  p=0.847   打平
      7B                0.0335    0.0455      17/28  p=0.135   打平

    OOD 兄弟组留出        ours      G-Memory    配对              判定
      72B               0.1515    0.1439      61/54  p=0.576   打平
      30B               0.0952    0.1136      12/29  p=0.0115  **G-Memory 显著赢**
      7B                0.0303    0.0325      20/22  p=0.878   打平

**关键的机制反转**——拿掉兄弟族之后，谁掉得多随模型反过来：

    72B    ours −28 行   G-Memory −59 行     ← G-Memory 更依赖兄弟族
    30B    ours −47 行   G-Memory −33 行     ← **我们更依赖兄弟族**
    7B     ours − 3 行   G-Memory −12 行

所以 **「G-Memory 的 OOD 优势由兄弟族泄漏构成」这句话必须限定为 72B**。在 30B 上恰恰相反：
去掉泄漏后我们从打平变成显著落后（p=0.0115）。这与 [[viki-l2-dispatch-reverses]] 是同一
形状的教训——**一个机制结论在一个模型上成立，不等于它是方法的性质**。

还要一起写的：30B 上 MEMENTO 在两个 OOD 口径都不比我们差（单族 0.1126、兄弟组 0.1147，
后者对我们 p=0.114 不显著）。而 v1 / trajRAG / zero-shot 在所有模型的两个 OOD 口径上都被
我们显著击败。

**能说的**：ID 与组合泛化上，一个 4 算子的 agent 库在三个模型上全面且显著胜过五个基线。
**不能说的**：我们在留出族 OOD 上更强——**三个模型 × 两个口径六格里，我们赢 0 格、
打平 5 格、输 1 格**。留出族这一列对所有臂都基本没被解决（最高 0.15，对 ID 的 0.61）。

**剩下的 3 格 missing，都是已知且非必需**：
1. `MEMENTO-style 72B/heldout_sibgrp` —— `viki_eval_memento.py` 的 `--exclude-family` 只吃
   一个族，72B 那批历史存档没有分组跑；30B/7B 的是这次新建的所以有。要补就是给那个 runner
   加分组参数再跑 3 折。
2/3. `G-Memory (shuffled control)` 的 30B/7B —— shuffled 是**安慰剂对照不是基线**，
   只在 72B 跑过，用来解释 72B 单族那一列。30B/7B 不跑它不影响任何一格结论。
