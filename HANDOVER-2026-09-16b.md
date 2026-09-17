# HANDOVER 2026-09-16b — LLM 生成 memory 在 R 与 H 两族跑通并经留出池确认；组合泛化的 claim 已收窄

接 `HANDOVER-2026-09-16.md`（上午由另一个会话写的，其 §3 的优先级 1 与 2 都已在本轮做完）。
**结论本体在 memory `partnr-llm-memory-direction`（本轮大幅重写），这里不重复。**

## 0. 为什么交接

会话 09-16 12:22 → 21:20，约 9 小时，未压缩，跨了 6 轮仿真格（共 27 个格）。任务收口干净：
两族的门与确认都已交付，远端无作业。

## 1. 移动了结论的读数

### 1a. 先说阴性

- **`is_next_to` 在特权执行门下不可验收。** 归纳出来的 body 是对的
  （`Navigate ?x / Pick ?x / Navigate ?z1 / Place "?x, on, ?z1, next_to, ?y"`），但两个池子都被拒：
  `gate_nxt` pc +0.018 [−0.032,+0.079]；专门新切的 `gate_nxt2` pc +0.063 [−0.024,+0.156]、
  按谓词 +0.067 [−0.050,+0.183]（8 升 4 降）。
  **根因不是池子**：`gate_nxt2` 上开局已成立只占 29.1%（按 `rigid_objs` 实算距离），base 却满足 83.5%，
  即 ~54% 由组合器自己的放置路径达成。这是 `partnr-execution-gate` 记过的特权臂结构性盲区。
  **不要再切第三版池子。**
- **`is_filled` 没学会**：8 条纯 H 轨迹，body 满足 0/40 且让 `is_on_top` 掉 0.038（0 升 3 降），被拒。
- **`is_inside` / `is_on_floor` 各只有 2 条轨迹**，候选 advisory `matched 0`，没起格。材料边界，不是失败。
- **7B 不能当归纳器**（上午已记）：17 次提交全因变量名被拒。30B 可以。

### 1b. 成立的

- **H 族**：`gate_H` 上 `is_clean` 0→0.980、`is_powered_on` 0→0.864，**三条天然阴性对照全部被拒**；
  库 22→24（`results/partnr_operators_h30b.json`）。
  **`conf_H` 确认（只跑一次，两格 60/60）：pc 0.4985→0.7547，+0.2563 [+0.2197,+0.2933]，39 升 1 降**；
  `is_clean` 0→0.981、`is_powered_on` 0.037→0.926、`is_on_top` 一条没动。
- **R 族 — 本轮最干净的一条**：30B 从 118 条 `rec_R_plain` 轨迹只提出一条 body
  （`Navigate ?x / Pick ?x / Navigate ?y / Place "?x, on, ?y, none, none"`，advisory 90/73）。
  按 `partnr-execution-gate` 的清空-该键协议，在专用池 `gate_ontop` 上：

  ```
  floor (4 算子, is_on_top 清空)   is_on_top 6/127 = 0.047   pc 0.0500
  rule20 (+20 条规则算子)          is_on_top 124/127 = 0.976 pc 0.9708
  llm1   (+1 条 LLM 算子)          is_on_top 124/127 = 0.976 pc 0.9708
  llm1 − rule20 = 0.0000 [0,0]，0 条 episode 移动
  ```

  **`conf_ontop` 确认同样是 0.0000 [0,0]、0 条移动**（双方 125/132 = 0.947）。
  即 **20 条规则归纳的算子是同一条通用 body 的冗余特化**，LLM 一条完全等价。
- **机制（归纳自己发现的）**：`Navigate ?x` 前缀决定一切。`Clean ?x` 单独 0.122、加前缀 0.980；
  `PowerOn ?x` 0.273 → 0.864。

### 1c. 组合泛化 claim 的更正（零 GPU，优先级 1）

上午的交接 §3 写「v2_prompt 0.783 比我们高 ⇒ 表示没问题、开环执行是瓶颈」。**两句都要改：**

- `v2_prompt` 是**我们的算子当文本喂进同一个闭环 ReAct**（`scripts/drivers/partnr_overnight.sh:8`）。
  `DIAG-why-v2-loses-2026-09-13.md` §2.8 量过：n=307 上 0.819 对 react 0.844，
  **Δ −0.025 [−0.059,+0.008]，在噪声线内**。记忆内容既没帮忙也没伤害，不能当「表示是好的」的证据。
- **开环组合器不是瓶颈。** 同一开环组合器，只把要求换成真命题、库用 iir1，val_mini **0.8477**
  （memory `partnr-is-in-room-wall`），react 在 DIAG 的同一口径（n=332）是 **0.843**——打平；
  按类型也不输（priv R_T 0.846 对 0.792，R_S_T 0.722 对 0.695）。

  ```
  v2_intent 0.330 → 冻结 typed 臂 0.651 → react 0.787 → priv 真要求+同一开环组合器 0.848 → ceiling 0.959
                    └────────── 缺口仍全在「LLM → 要求」这一步 ──────────┘
  ```
  口径提醒：priv 的 0.848 来自 09-09 `sweep_postfix`（366 池 / DIAG 的 332 交集），
  和 369 crash-as-zero 那张表不同分母，**只能说「同量级、打平」，不能排名**。

- **用户已定：claim 收窄成「零 token、确定性的组合器把复合任务的衰减补回来」，如实写出 30B 上输给闭环基线；
  不花卡做闭环化组合器。**

## 2. 现场（09-16 21:20 实测）

- **远端无本人作业、无本人端点**（8061 / 8063 / 8071 全部不通）。
- GPU：0=18.5 G、2=53.7 G、4=97 G、5=65 G、7=73.9 G 都是别人的；1、3、6 基本空。
- **根分区 9.0 G / 96%，依旧吃紧**；`/mnt/pfs` 40 T 可用。
- 产物：`outputs/induct_0916/`（全部归纳 transcript 与候选）、
  `outputs/gate/{nxt30b,h30b,nxt2_30b,ontop30b}/`、`outputs/confirm/{h30b,ontop30b}/`。
- 库：`results/partnr_operators_h30b.json`（24 算子，当前库）、
  `results/partnr_gate/ontop30b/{floor,rule20,llm1}.json`（R 对照三件套）。
- 池子从 20 个增加到 **24 个**（`results/partnr_pools/ledger.json`）：
  本轮新增 `gate_nxt2` / `conf_nxt2` / `gate_ontop` / `conf_ontop`，都已核对互不重叠。
- git：HEAD `a294bc4`（分支 `partnr-skill-memory-v2`，**未推送**）。未提交：`CLAUDE.md`（本次改动）、
  `HANDOVER-2026-09-15.md` / `-16.md` / 本文件未跟踪、`third_party` 照旧。

## 3. 下一步

**优先级 1：装一个纯 LLM 库并端到端比一次。** 现在库里 `is_on_top`(20) 仍是规则归纳的，
虽然已证明与 LLM 那条等价。把它换成 `llm1` 的一条，得到 **5 算子的纯 LLM 库**
（`is_on_top` / `is_inside` / `is_in_room` / `is_clean` / `is_powered_on`），
在特权臂和冻结的 typed 臂上对 `iir1` 比一次。
**先验：特权臂上应当完全打平**（两个池子的 0.0000 已经说明这点），
所以真正的看点是 **typed 模型臂**——库变小、菜单变短，会不会改善模型选谓词。这是唯一还没测过的一段。

**优先级 2（如果要补 S）：** 唯一可能验收 `is_next_to` 的路是改在**模型臂**上验收，
但那正是 memory 里记着的混淆来源（增益来自菜单里出现该谓词，故意写坏的算子涨幅一样）。
**我的建议是不做，把它写成方法边界。**

**优先级 3：** `is_filled` 若要救，得先补录纯 H 轨迹（`train` 里纯 H 只有 22 条，已全用完），
或者放宽到 H_T / H_R_S 的轨迹——但那就不再是「只从基本任务归纳」。

## 4. 文件与代码

| 提交 | 内容 |
|---|---|
| `89886f6` | `scripts/partnr_key_admission.py` 按谓词验收（S/H 必需，整体完成度分不出算子体） |
| `af12173` `dc16398` | 归纳台子：`actor_window` 信息化拒绝；接受 `{"tool":"submit","args":{...}}` 信封 |
| `227a268` | 每步观测带 `moves_left` / `candidates_queued`，预算将尽时明说提交 |
| `d40914b` | distinct 判定忽略补位参数位（`PowerOn ?x ?z1 ?z2 ?z3` 这类孪生塌成一条） |
| `5b50088` | **门补上两条早已写明却没实现的判据**：bootstrap 区间必须排除 0；压的格不多于抬的格 |
| `a294bc4` | `--next-to-starts-apart` 池子判据，写进 ledger |

已知未修：`is_inside`(2 条) / `is_on_floor`(2 条) / `is_powered_off`(3 条) / `is_empty`(1 条) 无材料；
归纳循环仍缺「目标选择 → 批量验收 → 汇总成库」的 driver 与 assembler
（`partnr_gate_batch.py` / `partnr_gate_adjudicate.py` 可直接复用，后者四条判据本来就是全的）。

## 5. 坑（本轮新增）

- **`partnr_gate_batch.py` 从来没读它自己算的 bootstrap 区间**，只判「增益 > 噪声带」。
  这是 `partnr-execution-gate` 记过的第一版规则重现，它把 `is_next_to` 判成了 accepted。已修（`5b50088`）。
  **判决以 `partnr_gate_adjudicate.py` 为准，它的四条判据一直是全的。**
- **`pkill -f` 又中招一次**（第八次）：模式出现在自己的 ssh 命令行里，把远端 shell 一起杀了，
  那一批格没起来。**停端点按 PID，不要按模式。**
- **端点加载时间不稳定**：同一个 30B，早上 5 分钟、晚上 18 分钟（PFS 读竞争，82 s/shard）。
  **探端点的等待器超时后必须硬失败**，我那次只等 6.7 分钟就往下跑，三个归纳全撞上死端口。
- **并行两格一波要 36–41 分钟，不是单格 22 分钟打个折。** 60 进程一格时 load 已到 61（共 180 核）。
  本轮我连续三次把时间估乐观了。
- **合并多个键跑一次 `gate_batch` 时，`--key` 只能给一个**，SUMMARY 里其它键候选的
  「落在该谓词上」那条判据是按错键算的。**以 `partnr_key_admission.py` 的按谓词读数为准。**
- **`partnr_key_admission.py` 读的是 `steps[-1]["stats"]["task_explanation"]`**（嵌在 `stats` 里）。
  我按 step 顶层取，得到「全部 episode 都没有 explanation」的假象，差点得出「打分器把空 explanation
  当全满足」的错误结论。**动它的输出前先按它自己的路径取字段。**
- **候选为 0 且没有落地拒绝 ⇒ 读 transcript**（上午已记，本轮又救回一次：30B 24 步一次没提交，
  是因为预算只在开头说过一次）。
- 远端 `python3` 的 f-string 嵌套引号会炸；写远端小脚本一律用 `%` 格式化。

## 6. 本次改过的 memory

- `partnr-llm-memory-direction`：大幅重写，加入 30B 归纳可用、三个台子 bug、H 族门与确认、
  `is_next_to` 的实测根因与「不可验收」结论、R 族一条顶二十条及其确认。索引行同步更新。
- `partnr-typed-candidate-interface`：补上「本条目里的 7B 基线优势在 30B 上不成立」的更正与指针。
