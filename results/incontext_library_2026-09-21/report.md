# VIKI-L2 in-context-library 臂：执行报告（2026-09-22）

对象是 pass8 `main.tex` 里的 `TODO[in-context-library]`。按 spec 做完 A 层（零 call）和 B 层（P 臂，两个 CG 格）。

## 0. 先看这里

### 0.1 主格结果不落在预写的三种结局里

主格是 CG w/o Image，JSON-tolerant 口径，p < 0.05。

| 比较 | 行数 | 单方独赢 | p |
|---|---|---|---|
| P vs G-Memory | 9/297 vs 10/297 | 9 vs 10 | 1.0 |
| P vs zero-shot | 9/297 vs 1/297 | 8 vs 0 | 0.0078 |

- E1 要求 P 显著高于 G-Memory **和** zero-shot：对 G-Memory 不成立。
- E2 要求 P 与两者**都**无显著差异：对 zero-shot 不成立。
- E3 要求 P 显著低于 zero-shot：不成立。

**patch 的处理**：采用 E2 的结构，只改掉与数据矛盾的半句。
- E2 原句是 "not separable from the primitive-plan baselines"。这对 zero-shot 不成立，改成 "not above G-Memory under the same output format"。这句在两个 CG 格上都成立：无图格 9 vs 10，p = 1.0；有图格 0 vs 14，p = 1.2e-4。
- E2 第二句 "On VIKI-L2 the gain therefore requires the planner that consumes the library, and we treat the planner as part of the method." 原样保留。本方法对 P 在两格上分别独赢 247 行和 237 行，0 行输掉。
- Limitations 里 E2 那句 "the library helps only when the deterministic planner consumes it" 也和「P 显著高于 zero-shot」相矛盾，改成 "the library gains over G-Memory only when the deterministic planner consumes it"。
- **这是偏离预写措辞的地方，需要你确认。** 如果你想严格照 E2 原文写，就得接受第一句有一半与数据不符，我不建议。

### 0.2 单次运行的波动有几行那么大

前 20 行先冒烟跑了一次，全量时又跑了一次。两次里 19/20 行的回答不同，成功数从 0 变成 3。温度 0 下仍不确定，最可能的原因是 vLLM 在 8 路并发下批次组成不同。A1 在 RQ2 的 re-ask 上看到的跨次运行差异是同一类现象。所以 P 对 zero-shot 的 8 vs 0 有可能在重跑时变动；它对 G-Memory「分不开」、以及本方法对 P 的 247/0、237/0，不会因为几行波动而改变。附录那段已写明这一点。

### 0.3 服务端无法与基线逐字节对齐

G-Memory 臂是 2026-08-31 在另一台机器（`192.168.32.40:8050`，已下线）上跑的，那台机器的 vLLM 版本无从核对。P 臂按 `amendment7/service_resource_reuse.json` 的 unchanged 块逐项复现服务配置：模型、修订号 89c86200、served name、bf16、TP=4、max_model_len 16384、max_num_seqs 16、async scheduling、seed 0。只有资源字段和主机不同。本机 vLLM 是 0.11.2。prompt 组装经零 call 自检与基线逐字节相同，见 §4.1。

## 1. A1 同输出核对：S2（范围很窄）

- 首轮模型回答：12 个 model×split 格、三条臂逐字节相同（ID/OOD 各 924/924，CG 各 297/297），而且都等于回放源里的归档答案。
- prompt 不含库：从 27 个库文件里抽了 163 个特征串，在 INSTRUCTION、REASK 模板和前 5 条重建的实际 prompt 里都是 0 命中。构建位置在 `scripts/viki_eval_v2_intent_choice.py:234,247,252-254`，库在 `:283` 才第一次被用到，此时回答已经拿到。
- 差异只来自 re-ask：
  - 是否触发由库决定，例如 72B ID 上 full 46 次、no_trace 924 次、no_adm 123 次；
  - re-ask prompt 里带着 planner 的原因码，no_trace 全是 `UNSUPPORTED_PREDICATE`，no_adm 全是 `NO_SCHEDULE`；
  - 两臂 re-ask prompt 相同的行里，也有跨次运行的输出差异（30B OOD 51/229）。
- 结局 S2：RQ1 那句 "hold the interface and the planner fixed and change only the library" 不动。
- 如果以后想改，数据支持的说法是：the RQ2 arms share identical first-turn model outputs and the same planner and differ only in the library, which also decides whether a second, re-asked turn is needed.
- 证据：`work/A1/a1_summary.json`，脚本 `scripts/incontext_library/a1_same_output.py`。

## 2. A2 `cg-audit`：PASS（用户按撞号解释改判；字面规则给出 HALTED）

Figure 2 的 CG 格用的是 `outputs/v3_memories/memory_all.json`，即 14 族并集那次构建。它的全部输入池如下：
- 第一轮 784 个 run（14 族 × 56）；
- 第二轮 112 个 run（14 族 × 8）；
- 支持度探针（workbench 0–59）；
- layer 2/3 的挖掘池（偶数半 3,598 行）。

| 附录里的数 | 旧 | 新 |
|---|---|---|
| 工具点名的训练 episode | 180 | **1,242**（只算 13 个有库的族是 1,081；连工具返回结果里列出的一起算是 2,496） |
| transcript / run 目录 | 884 | **896** |
| actor-blind 敏感性行数 | 1,033 | 1,033（索引集合与 09-05 逐条相同） |
| CG 不同 task_id | 295 | 295 |

- 所有池子的索引都在归纳半之内，越界 0；transcript 里精确出现 CG task_id 0 次。
- 按 task_id 求交集：295 个 CG id 里有 4 个也是训练 episode 的 id，其中 2 个在归纳半（350_9-1、2395_9-1）。VIKI-L2 的 task_id 在 train 和 test 之间本来就会重复（共 24 个）。4 对都在 instruction、初始位置、目标、参考计划四项上与对应的 CG 任务不同，同一 episode 0/4；决定性检测器在训练集上命中 0。
- patch 已按新数改写附录段落，并删除 TODO。
- 顺带发现（没改）：附录把 1,033 行的结构写成 "an activation step followed by a transport step"。审计里的敏感性检测器定义是「不带双机器人条件的 cut 单元（放置在激活之前）加一次独立递送」，两者描述不一致，建议作者核对措辞。
- 证据：`work/A2/{a2_summary.json,a2_taskid_collisions.json,a2_runs.json}`，脚本 `scripts/incontext_library/a2_cg_audit_v3.py`、`a2_taskid_collisions.py`。

## 3. A3 数字核对：全部 MATCH → 删除 `TODO[gmemory-72b]`

来源是 `results/paper_viki_iclr2027/figure2_tom_summary.csv`，并按 `rows/figure2/*/*.jsonl` 的 `success` 重数过。

| 项 | 分子/分母 | 值 |
|---|---|---|
| G-Memory 72B ID / OOD | 471/924，192/924 | 51.0，20.8 |
| G-Memory 72B CG w/ / w/o Image | 14/297，10/297 | 4.7，3.4 |
| Ours 30B CG w/ / w/o | 193/297，215/297 | 65.0，72.4 |
| Ours 7B CG w/ / w/o | 45/297，76/297 | 15.2，25.6 |
| ToM 72B ID / OOD | 23/924，25/924 | 2.5，2.7 |

- 9 月 8 日的文档对得上（`RESULTS-2026-09-08.md:19-23`），9 月 2 日的文档里找不到这些数。
- G-Memory 的 CG 在官方 strict 口径下两格都是 0/297。

## 4. A4 盘点：没有可复用的产物，所以跑了 B 层

唯一把库放进 prompt 的运行是 `amendment11/inprompt_v2_full.jsonl`。它只有 ID，用的是 19 算子参考库，不是 Figure 2 的库，不能用。

## 5. B 层 P 臂

### 5.1 配置与 prompt

- runner：`scripts/incontext_library/run_p_arm.py`。它原样导入 `scripts/viki_amendment10_run.py`（72B 基线的 CG runner），只换记忆 provider；输出经软链目录写到 `runs/`，不碰 amendment10。
- **模板自检（零 call）**：用今天的代码、空记忆块重建两个 split 各 297 行的 prompt，与归档 zero-shot 运行的 `prompt_sha256` 相比，**297/297 相同**。P 臂在同一位置插入库块（`viki_memory_skill.py:420-434`）。
- 配置 diff：`work/B/config_diff.json`。除记忆块外，只有服务的资源字段和主机不同，见 §0.3。
- 记忆块：`scripts/incontext_library/render.py`（写完即冻结）→ `prompts/library_block.txt`，2,252 字符，sha256 `13e33587…`；插入时去掉末尾换行，运行记录里的 sha256 是 `e8d9028c…`，594 行全部相同。
  - 内容：8 个算子，每个一段，写 name、effect、preconditions、带变量的 demo；接力算子两个角色分开列；3 条保留的顺序规则各写一句；没有词表。
  - 库里的算子没有 name 字段，名字按「序号 + 类型」确定性生成，例如 "Skill 2 (achievement)"。第 4 个算子的效果变量是 `?tool`、demo 里是 `?x`，这是库本身的写法，照原样保留。
- 前 5 行的完整 prompt：`prompts/{text,imaged}_first5.txt`。导出时断言了它们与运行记录的 sha256 相同；图片以 sha256 代替。
- prompt token 数（只报，不要求对齐）：

  | 格 | P 均值 / 最大 | G-Memory 均值 / 最大 |
  |---|---|---|
  | w/o Image | 1,622 / 1,640 | 1,411 / 1,685 |
  | w/ Image | 1,813 / 1,825 | 1,709 / 1,870 |

### 5.2 流程

- 4.3 第 1 步，冒烟 20 行：官方格式合规 20/20，最长 completion 992 token，没有截断，于是继续。
- 两个 split 都跑完：各 297/297，退出码 0，没有 stall。全量的格式合规率：w/o 296/297，w/ 297/297；截断 0。
- 共 614 次 call（20 冒烟 + 594），CALL_CAP 700。冒烟那 20 行在全量里重跑，因为 runner 的元数据守卫不允许跨行数续跑。
- 配对自检：用同一个 `tolerant()` 从归档回答重算 G-Memory 和 zero-shot，两个 split 都是 297/297 与 Figure 2 逐行一致。

### 5.3 结果（逐行在 `per_row/{text,imaged}.csv`，汇总在 `work/B/score_summary.json`）

| 格 | P（JSON-tolerant / 官方） | G-Memory（tolerant / 官方） | zero-shot（tolerant / 官方） | 本方法 SOLVED |
|---|---|---|---|---|
| CG w/o Image | **9/297（3.0%）** / 9 | 10 / 0 | 1 / 1 | 256 |
| CG w/ Image | **0/297（0.0%）** / 0 | 14 / 0 | 0 / 0 | 237 |

McNemar（精确，双侧；297 行，括号里是去掉两个重复 task_id 后的 295 行）：

| 格 | 比较 | 单方独赢 | p |
|---|---|---|---|
| w/o Image | P vs G-Memory | 9 vs 10（9 vs 10） | 1.0（1.0） |
| w/o Image | P vs zero-shot | 8 vs 0（8 vs 0） | .0078（.0078） |
| w/o Image | 本方法 vs P | 247 vs 0（245 vs 0） | 8.8e-75（3.5e-74） |
| w/ Image | P vs G-Memory | 0 vs 14（0 vs 14） | 1.2e-4（1.2e-4） |
| w/ Image | P vs zero-shot | 0 vs 0 | 1.0 |
| w/ Image | 本方法 vs P | 237 vs 0（235 vs 0） | 9.1e-72（3.6e-71） |

- 附加项（descriptive，只报 w/ Image）：事后用库的 `vocabulary.canonical` 做名字对齐后，P-canon 10/297，G-canon 24/297。名字对齐让两臂都有所上升。
- 描述性观察：P 的回答大多写出了 `Interact`（w/o 262/297，w/ 257/297），所以失败不是因为漏写切这一步。更细的失败归因没有做。
- P 的官方分和 tolerant 分逐格相同；G-Memory 的官方分是 0，这是百分比格式的问题，与 A12 的结论一致。

## 6. 稿件改动（`main_patch.diff`）

基准是用户贴的 pass8。`src/pass8_excerpts.tex` 是逐字摘出的受影响段落；diff 在摘录上生成，已验证能准确应用；打到完整 main.tex 上时 GNU patch 会带偏移定位。脚本：`scripts/incontext_library/patch_main.py`。

1. 删除 `TODO[gmemory-72b]`（A3）。
2. RQ1 末句 "Whether a baseline memory would gain from the same interface is not tested." 替换为下面两句，并删除 `TODO[in-context-library]`：
   > With the admitted library placed in the prompt and primitive plans as output, success on the two CG splits is 0.0\% with images and 3.0\% without, not above G-Memory under the same output format (Appendix~\ref{sec:viki_audit}). On VIKI-L2 the gain therefore requires the planner that consumes the library, and we treat the planner as part of the method.
3. Limitations 加一句："On VIKI-L2 the library gains over G-Memory only when the deterministic planner consumes it."
4. 附录「CG split audit」段按 A2 的新数改写，删除 `TODO[cg-audit]`。
5. 附录新增 `\paragraph{In-context library control.}`，放在 sibling-group 段之前。内容包括：配置、两种口径的四个数、三组 McNemar 的 p 值和独赢行数、295 行复核，以及单次运行波动的说明。
6. RQ1 那句 "hold the interface and the planner fixed …" 不动（A1 结局 S2）。

**页数**：正文净增约 4 行（RQ1 约 +2.9，Limitations 约 +1.0，按每行约 95 字符估；删掉的 TODO 是注释，不占版面），在第 9 页剩余的约 12 行以内。附录增加约 16 行，不计正文页数。实际页数要用真实的样式和图重新编译确认。

## 7. 范围外的顺带发现（没改）

- 附录的 "Ours Prompt" 列表（Observation Diff / Skill Prediction / Skill Matching / Action Selection 四步那版）与出数代码实际用的 `habitat_llm/conf/instruct/rag_prompt_sequential_cooperation_skills_v4.yaml` 不是同一个模板。代码里那版很短，只要求 `Thought: <brief reason>` 加一个动作，见 `CODE_LOCATIONS_partnr_retrieval.md`。建议作者换成实际模板。
- 1,033 行那句的结构描述与审计的检测器定义不一致，见 §2。

## 8. 现场

- 远端的 72B 服务已按 PID 停掉，GPU 0–3 已释放。本任务在远端没有残留进程。
- 可选扩展（ID 和单族 OOD，共 1,848 次 call）没跑，等你看过这份结果再定。
