# HANDOVER 2026-09-13b — PARTNR：我们输给基线的原因、重问无用、ReAct+Achieve 原型没被调用

接 `HANDOVER-2026-09-13.md`（spatial 阴性结果、两份论文审计）。那份写于本会话开头，§2 的「零作业」已过时。

## 0. 为什么交接

本会话从 09-13 20:30 跑到 23:35，上下文已远超仓库的 300–400K 线，未压缩过。按规矩换会话。

## 1. 移动了结论的读数（否定的先说）

### 1.1 模型臂上我们输给所有基线，而且差距在接口，不在记忆

09-05 的 head-to-head（`outputs/headtohead/val_mini/report.txt`，30B，val_mini）：
ReAct 0.896/0.885/0.838/0.747/0.611（R/R_S/R_T/R_S_T/H_R），G-Memory、MEMENTO、轨迹检索、v2_prompt 在 R 上都是 0.87–0.92；
**我们的模型臂 `v2_intent` 是 0.443/0.370/0.203/0.227/0.087**。privileged 臂也在 R_T/R_S_T/H_R 上低于 ReAct。
所以 **+0.1093 只是库内增益，不是赢基线**（已在 memory `partnr-is-in-room-wall` 里补上这条限定）。

归因见 **`DIAG-why-v2-loses-2026-09-13.md`**（只读分支写，核心机制我在代码里核过：`partnr_planner.py:942` 每 agent 每集只抽一次要求）。
五臂共有 332 集上：react 0.843、priv 0.739、intent 0.320。**差距 0.523 = 抽取接口 0.419（80%）+ 词表覆盖 0.104（20%）**。
关键参照：库完全覆盖且无 DAG 的 107 集，**给对要求时 priv 0.967 > ReAct 0.884**——执行不是问题。
重复劳动、步数预算、台子不对等都量过并排除；崩溃剔除偏向基线。1a（矛盾终态）/1b（两 agent 表不一致）仍只是分组相关。

### 1.2 「探索后重问」在 7B 上无效

memory **`partnr-reask-does-not-help-7b`**。7B 严格召回：开局 0.061 / 自己看到的物体 0.055 / 全观测 0.056；
参照答案过同一打分器 0.675。255 集任务物体全已看见时仍 0.042。错在谓词混用、选错目标实例（「白色的桌子」在文本世界描述里不存在）、`is_inside` 顶替 `is_in_room`。**30B 没测**（开局 0.206，可能不同）。

### 1.3 `is_next_to` 在 7B 模型臂上：四个候选都涨，阴性对照也涨

`gate_S3` 60 集，7B（`outputs/gate/nxt2_7b/`）：A +0.0836 CI[−0.004,+0.169]、**B +0.0836 CI[+0.011,+0.158]**、C +0.0715（59 集时）、**D（阴性对照）+0.0869 CI[+0.014,+0.165]**；
base 同配置自比 Δ −0.010，58 集动 3 集。A 与 D 有 19 集变化量完全相同。
**增益来自 menu 里出现 `is_next_to` 这个谓词，不是算子体写法**。D 的空槽**没有**被 `with_beside` 补（那只对折叠进放置的 next_to 生效），是「放到参照物所在家具上」本身就常满足 next_to。
gate_S3 上 7B base 0.5458 远高于 val_mini R_S 的 0.16–0.18（三次独立运行一致），更像是池子偏易（train 切出），不是台子压低了旧数。
正式裁决 `outputs/gate/nxt2_7b/ADJUDICATION.json` 在 22:05 写出，**B/C 那两行我没读**。

### 1.4 25857 崩溃是台子 bug，已修

`Object handbag_0 has no parent`（两 agent 各搬一次同一物体后世界图丢边）。`world_graph.py:211` 改为描述成 `unknown` 不再 raise。
论证：raise 会终止整集，所以任何没崩的集从没走过这行，修复只影响崩的集。oracle A/C/D 与 7B D 的 25857 已重跑拼接，原崩溃文件在 `outputs/gate/fix25857/crashed/`。
oracle 臂重裁决仍 0 收（四条 CI 全含 0）。有无 25857 八行结论全不变。

### 1.5 原型 ReAct + `Achieve` 宏工具：7B 不调用它

两次冒烟（val_mini 0/101/102，1 进程）都是 **`Achieve` 调用 0 次**；第二次已加 Example 0 示例和工具自动探索，episode 101 仍 0 分，
两个 agent 都在 `Explore[living_room_0]` / `Explore[living_room]` 循环（场景里是 `living_room_1`）。**工具的执行路径至今没被真正跑过**。

## 2. 现场（09-13 23:32 实测）

- **只剩 GPU 0 是我们的**：GPU 1 22:12 起被别人的 LightNav habitat 评测占（`tanshenghan`），2–7 也全是别人的。
- **端点**：7B `qwen2.5-vl-7b` :8061 在 GPU 0（16k、0.30，归档参数），200。8B :8101 已停。30B 没有。
- **7B 同台三格**（`scripts/drivers/partnr_h2h7b_0913.sh`，22:43 起，`outputs/headtohead_0913/val_mini/`，OOM 0）：

  | 格 | 进程 | 23:32 进度 | 按当前速率的估计 |
  |---|---|---|---|
  | react_7b | 24 | 81/369 | ~02:15 |
  | v2_accepted_7b（22 算子） | 24 | 47/369 | ~05:05 |
  | react_rag_R_7b | 12 | 33/369 | **会在 06:43 撞 8h 硬超时，约 300/369** |

  速率只有 48 分钟的样本，前段偏慢，估计不可靠。**rag 格若被砍：同代码带 `+resume=True` 补跑是安全的**，别在带残留的目录外另起。
- **overnight 队列**（`partnr_h2h7b_night_0913.sh`，PID 1868421）：stage 0 已完成（MEMENTO 抽取用 7B 重做，368 条，`results/partnr_memento_extractions_val_mini_7b.json`）；
  等 react 与 ours 两个 runner 退出且 GPU 0 < 55G 后起 `gmemory_7b` / `memento_7b`（各 24 进程），再等 h2h 驱动结束写 `report_all.txt`。按上表估计约 09:00 后才有全表。
- **冒烟 b**（`partnr_memskill_smoke.sh`，PID 1885969）还在跑剩下两集，`timeout 3600`，00:22 前自动结束。
- 09-13 20:30 那版 val_mini 7B/8B base-vs-accepted 四格（`partnr_model_valmini_0913.sh`）**从未起飞**（端点 8061 在 22:04 死过一次，已重拉）。

## 3. 下一步（最便宜的在前）

1. **早上先读 `report_all.txt`**，逐格看 `CELL.json` 的 `episodes` / `fake_episodes` / `killed`。rag 格不满 369 就 resume 补，补完再重出报表。先验：7B 上我们（22 算子）与 react_7b 同量级，09-05 是 0.198 对 0.154（R）。
2. **不靠 LLM 验证 `Achieve` 的执行路径**：写一个脚本化调用（固定 episode、直接调工具跑到 `Successful execution!`），先确认工具本身能把东西搬到位。它从没被真正执行过，别在它身上花模型格。
3. **再决定 7B 能不能用这个接口**：两次冒烟说明 7B 连示例里的工具都不调、只会抄示例房间名。可选：把示例改成真实场景名风格、或在 30B/72B 上冒烟（需要四张卡）。**先验：7B 大概率仍不调用**，这条原型的主张要在更大模型上验。
4. 1a/1b 因果（重放 LLM 要求表进 privileged 组合器，三格零 LLM）——DIAG §5.1，要一块 GPU。

## 4. 文件与代码

| 路径 | 状态 |
|---|---|
| `DIAG-why-v2-loses-2026-09-13.md` | **未提交** |
| `habitat_llm/world_model/world_graph.py`（:211 修复） | **未提交** |
| `scripts/partnr_intent_diagnostic.py`（`--intent`）、`scripts/partnr_reask_intent.py` | **未提交** |
| `our_method/skill_memory_v2/skill_tool.py`、`conf/tools/motor_skills/memory_skill.yaml`、`conf/agent/oracle_rearrange{,_object_states}_memory_agent.yaml`、`conf/baselines/react_memskill_vllm.yaml`、`conf/instruct/*_memskill_instruct.yaml` | **未提交**（原型） |
| `scripts/drivers/partnr_{nxt2_7b_pergpu,nxt2_7b_followup,fix25857,h2h7b_0913,h2h7b_night_0913,memskill_smoke,model_valmini_0913}.sh` | 除 pergpu 外**未提交** |
| 上一次提交 | `9b090db`（is_next_to 门参数化、挖掘器、pergpu 驱动、HANDOVER-09-13、两份 AUDIT） |
| 远端产物 | `outputs/reask_0913/`、`outputs/gate/nxt2_7b{,_r2,_retry}/`、`outputs/gate/fix25857/`、`outputs/headtohead_0913/` |

`.claude/`（worktree 残留）与 `third_party/GMemory` 删除不要提交。

## 5. 坑（本轮新增）

- **RAG 基线每个 worker 往 GPU 上装一份 all-mpnet-base-v2（`rag.py:52`），1.75G/worker**，是 ReAct worker 的 2.4 倍。三格 × 36、三格 × 24 都 OOM，最后 rag 格降到 12。G-Memory / MEMENTO 是 `memory_device: cpu`，不受影响。
- **vLLM `--max-model-len 32768` 配 `--gpu-memory-utilization 0.30` 起不来**（KV cache −9.6G）。
- **别的会话写的交接会过时得很快**：09-13 那份说「零作业」，半小时后就有它自己起的波次在跑；它说新增了 memory `partnr-next-to-negative`，**盘上没有这个文件**，`partnr-execution-gate` 里指向它的链接是悬空的。
- **grep `Agent_0_Action:` 会先命中 prompt 里的 few-shot 示例**，看起来像模型在行动。模型真实动作只在 `traces/` 里，而且整集结束才写盘。
- **ssh 单引号里的 Python f-string 不能带反斜杠（py3.9）**，远端分析一律用 `<<"EOF"` + 双引号。
- **MEMENTO 的逐指令抽取表是 30B 做的**（builder 默认 `qwen3-vl-30b`），7B 臂必须重做，否则白送一个 30B 的第一步。

## 6. 本次改过的 memory

- 新增 `partnr-reask-does-not-help-7b`。
- `partnr-is-in-room-wall`：补「+0.1093 是库内增益，模型臂 30B 输所有基线」。
- `partnr-execution-gate`：7B base 更正为 60/60 0.5458，补「模型臂上也分不出算子体，A≈D」，标注 `partnr-next-to-negative` 链接悬空。
