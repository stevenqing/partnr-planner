# HANDOVER 2026-09-05b

## 为什么换会话

接着 `HANDOVER-2026-09-05.md` 又跑了一整天:PARTNR 修 executor + 六格重测,VIKI 从零建起
agent 算子库的完整测量链条。轮次多、工具调用密集。(精确 token 数与压缩次数无可靠读数,**未核实**。)

**这份取代 09-05 那份的结论 1 和结论 3。** 方法学结论一律在 memory 里,这里不重复。

---

## 一、改结论的结果

**1. PARTNR 的「差 0.181」是 0.113。** step-0 猝死修掉之后六格全部重测:
`priv:v2_memory_R - v2_prompt` 由 −0.181 变 **−0.113**(null band 0.076,仍 REAL),
四条主基线的差距整体缩小约 0.068。**方向不变,旧表上每一个幅度都不能再引。**
详见 `partnr-postfix-headtohead`。

**2. 一条消融结论死了。** `priv:v2_memory_R - priv:v2_R_noorder` 由 +0.080 (REAL) 变
+0.065(**落进噪声带**)。"用 episode 自带时序约束能改善组合泛化"不再是 finding ——
诚实说法是它从来没被稳健分开过。

**3. 09-05 那份的结论 3(能力曲线是阈值不是梯度)被推翻。** 修好 harness 之后:

        Qwen2.5-VL-72B   6/15 = 40%  ->  8/15 = 53.3%
        Qwen3-VL-30B     6/15 = 40%  ->  7/15 = 46.7%
        Qwen2.5-VL-7B    0/15 =  0%  ->  1/15 =  6.7%

**7B 不再是 0,三点看起来是梯度不是阈值。** 那个"阈值"是 harness 把 7B 的正确提交按键名
丢掉造出来的(它 14 个 transcript 里 10 个中招)。72B 与 30B 仍分不开,n=15 的限定照旧。

**4. VIKI agent 算子库:模型没换,端到端 32.47% -> 74.24%,全部来自修 harness。**
五次"模型不行"的读法里**四次是台子的问题**。详见 `viki-harness-was-the-bottleneck`
(附一条以后必用的检查:一格得零,先把参照库自己的正确答案送进同一个判据)。

**5. 那 19% 死 episode 只有 30/70 可恢复**,其余 40 个是 `is_in_room` 词汇边界。
详见 `partnr-step0-death-and-vocabulary-boundary`。

---

## 二、现场(本次实测)

- **没有任何作业在跑。** 三个端点都在线:8050 `qwen2.5-vl-72b-amendment3-f2`(外部机)、
  8061 `qwen2.5-vl-7b`、8062 `qwen3-vl-30b`。
- **GPU 只剩 0 空着。** 1 和 6 今天被别的项目占了(09-05 早上还是空的,**已过期**)。
  2-5 是 30B(TP4),7 是 7B。
- PARTNR postfix 六格全部 `DONE`,各 369 episodes。
- git 仍在 `dee67ea`,**45 处未提交**。`third_party/GMemory` 的 `D` 仍是本地假象,**别提交**。

落盘产物(全部实测存在):

    outputs/headtohead/val_mini/compositional.postfix.txt   新的 head-to-head(0.113)
    outputs/sweep_postfix/val_mini/                         PARTNR 六格 post-fix
    outputs/partnr_dead_fix.json                            70 个死 episode 的配对检验
    outputs/viki_parity.json / outputs/parity/bench_*.json  VIKI 全部格
    outputs/capability_curve_v4.json                        修复后的能力曲线
    outputs/agentic_library_runner.json                     当前最好的 agent 库(7 条)
    outputs/viki_family_passability.json                    可通过性对照
    outputs/partnr_induction_tools_selftest.json            PARTNR 工作台双向校准

`outputs/sweep_remeasured/` 是 pre-fix 留档,是 bug 代价的证据,**别覆盖**。

---

## 三、下一步(cheapest-diagnostic-first)

1. **提交。** 45 处未提交,其中四处改了框架语义。再跑任何东西之前先落 commit,
   否则下一次"框架冻结"没有基准点。
2. **VIKI 还差 74.24% -> 93.83%**(可验证形状的天花板)。缺口是两个家族:
   `cut_fruit_on_board` 0/189、`dog_push_box` 0/46。**先读 transcript 再加算力**——
   今天五次里没有一次是靠加采样找到的。`dog_push_box` 要 coordination,
   `run_operator` 现在会逐个机器人试但仍是单执行者,**验不了双机器人算子**,是已知结构缺口。
3. **PARTNR 归因 rung 可以开工。** 工作台已建成并双向校准通过
   (`partnr_induction_tools{,_selftest}.py`,出厂算子 0.92/0.94/0.74 对弄坏的 0.17/0.56/0.24,
   SEPARATED)。**已知限制**:PARTNR 没有反事实重放(trace 只有动作、零世界状态),
   `is_in_room` 的 target 是房间 id、离线拿不到家具→房间归属,所以它的判据只能是
   "subject + 时序",比 `is_on_top` 弱。真正的裁决是外层 gate:重建库跑特权 sweep,
   和干净基线比(`v2_memory_R` 0.7319 / mean comp 0.804)。
4. 论文侧:`docs/VIKI-L2-PAPER-RESULTS.md` 的效度威胁一节仍然只在 memory 里,没写。

---

## 四、代码

**改了框架语义(四处,已记进 `docs/AGENTIC-OPERATOR-INDUCTION.md` 的"破冻结"一节):**

    viki_induction_tools.py       run_operator 逐个机器人试;新增 contrast_actors
    viki_agentic_rung_abstraction.py   normalise_request(按别名收提交);错误信息列出合法形式;
                                       --library 边际贡献判据;--target-key(均为 opt-in)

不传 `--library` / `--target-key` 时命令与冻结版逐字节相同,但 TOOLS 文本和 harness 行为已变,
**所以曲线是重跑过的,只报 v4 那组。**

新增:

    scripts/viki_library_shim.py / viki_assemble_agentic_library.py / viki_memory_from_library.py
    scripts/viki_reference_ablation.py / viki_family_passability.py / viki_parity_report.py
    scripts/viki_rung_targets.py / viki_rung_residual_targets.py / viki_rung_family_targets.py
    scripts/partnr_induction_tools{,_selftest}.py / partnr_dead_fix_report.py
    scripts/drivers/viki_{parity,diversity,family,marginal,interface,format,runner}_overnight.sh
    scripts/drivers/viki_curve_rerun.sh / verify_dead_fix.sh / rerun_priv_postfix.sh
    habitat_llm/examples/planner_demo.py   +episode_id_filter(多进程切分前按 id 过滤)
    our_method/skill_memory_v2/partnr_planner.py   _claim 的 explore fallback(step-0 猝死修复)

**知情未修:** `run_operator` 仍是单执行者,验不了 coordination(对 gate 值 +6,
端到端值 46 个 episode);PARTNR 侧 `is_in_room` 判据偏弱(见上)。

---

## 五、坑

- **一格得零,先把参照库自己的正确答案送进同一个判据。** 今天第 5 条就是这么抓到的:
  判据在拒绝正确算子。这一步应该成为默认动作。
- **读 transcript,别先加算力。** 五个诊断全部来自读 transcript / 读原始 answer,零个来自采样。
- **"忠实"不等于"可读"。** `show_trace` 逐步返回全部内容,模型读了四遍也没看出两个机器人不同。
- **别按键名丢弃答案。** `extract_request` 早就写了"拒绝任何一种形式都是在给表达打分",
  派发器却在下一层违反了它,吃掉 152/524 个 transcript。
- **报表脚本先落盘再打印**、**等进程按 PID 别用 `pgrep -f`**、**每类格都要超时+stall 守卫**、
  **一端点一个生成作业** —— 09-05 那份的坑全部仍然有效,今天都照做了。
- **`sed` 全局替换会误伤**:改 driver 时把"产出的库"也一起换成了"输入的库",查 grep 才发现。

(更早的坑见 `HANDOVER-2026-09-05.md` 及更前,仍然有效。)

---

## 六、09-05 20:40 复核（另一个会话追加）

对上面「二、现场」的复核。**这一节里有一条是我自己先搞错又更正的，连过程一起留着。**

### 1. 三次重复与 no-think 已经打过分了（我一度以为没有）

我先查了 09-04 交接点名的 `outputs/p0_30b_report.json`，发现它只覆盖 `m30`
（`m30r2`/`m30r3`/`m30nt*` 各 0 次），就下了「48 格生成了从没打过分」的结论。**错的。**
`outputs/p0_30b_repeats.json`（**09-05 00:19**，`scripts/viki_p0_repeats.py`）
早就把五个 tag 全打完了，think 和 nothink 两块都在。

**我犯的正是这个仓库记了三次的那个错**：在一个地方没找到信号，就当成到处都没有。
正确动作是先 `ls outputs/` 找同类产物，再说「缺」。

两条路径独立算出的数**逐格一致**，算是一次意外的交叉验证。清点之后：

    scripts/viki_p0_repeats.py   outputs/p0_30b_repeats.json      ← 00:19，正本汇总表
    outputs/p0_30b_report.<tag>.json  ×5                          ← 20:28，**保留**
    scripts/viki_p0_repeats_table.py + outputs/p0_repeats_table.* ← 20:32，**已删**

**保留那五份不是冗余**：正本只有汇总（`cells`/`think`/`nothink`），而它们带
`per_index` 逐行 0/1 明细（12 格 × 924/297 行），**跨重复做配对检验必须用这个**。
真正重复的只有汇总表那一份，已删掉。

`scripts/viki_p0_report.py` 新增了 opt-in `--tag`（默认 `m30`，
**已自证默认路径与打补丁前逐字节相同**）：主报表脚本原先把 `m30` 写死，
现在能直接指向任一 tag，不必再另写脚本。

### 2. 60 格的完整性是干净的（这一条是新验的）

行数 924/297 全对，**索引唯一性 0 格不合格** —— 没有 09-03 那种 594 行覆盖 297 索引的事故。

### 3. no-think 的结论（与另一会话独立得出、数值一致）

    method (ID)       think          no-think        parseable
    G-Memory          15.55 ± 0.50   30.34 ± 0.49    99.78 -> 100.00
    skill memory v1   14.61 ± 0.29   20.71 ± 0.49    95.20 ->  99.93
    trajectory RAG     7.58 ± 0.57    3.75 ± 0.23    96.39 -> 100.00
    zero-shot          0.76 ± 0.29    0.87 ± 0.11    90.66 ->  99.60

方向不齐（G-Memory 翻倍、trajectory RAG 砍半），所以不是"去掉思考块之后格式变干净"。
**「G-Memory 掉 36 点」必须限定为 think 条件**，memory 已收窄。详见
`viki-l2-crossmodel-baselines`。

### 4. 「二、现场」有三处已过期

- **git 不再是 `dee67ea` / 45 处未提交**，现在是 `4c689f0`（"Docs: the induction protocol,
  two handovers, and the freeze broken four times"），**10 处未提交**。「下一步」第 1 条已完成。
- **8050 端点已经掉了**（这份写"三个端点都在线"）。现在只有 8061（7B）、8062（30B）。
- `viki_ordering_overnight` **20:13:34 已自行跑完**，PARTNR 侧现在没有作业在跑。
- GPU 仍是「只剩 0 空着」，这条没变。

### 5. `pgrep -f` 自匹配第 6 次，两个看守空转了 1 天 4 小时

    until ! pgrep -f "viki_agentic_inducer.py" > /dev/null 2>&1; do sleep 20; done

`pgrep -af` 实测把这两个看守**自己的 `bash -c` 命令行**列为匹配，而真正的
`viki_agentic_inducer.py` 进程一个都没有 → 循环永不退出。PID 935054 等了 **1-04:48**，
971528 等了 **1-04:34**。所幸它们后面只挂了两条打印状态的 `grep`，没排作业，损失为零。
**两个都已按 PID kill 掉**（没用任何 `pgrep` 模式），已复核退出。

第 6 次，而且就写在这份交接自己的坑清单里。**写进文档挡不住它 ——
等进程只用 `while kill -0 <pid>`，别写模式。**

### 6. 两个会话在并发改同一条 memory

`viki-l2-crossmodel-baselines` 今晚被两个会话各写了一遍，出现了讲同一件事的重复小节
（已合并去重）。**同一时间只让一个会话写 memory**，否则合并成本比写还高。
