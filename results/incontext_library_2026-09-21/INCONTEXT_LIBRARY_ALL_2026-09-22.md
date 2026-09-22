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


---

# 附录 A：逐处修改全文（在 pass8 main.tex 上逐条查找替换）

以下 4 条覆盖报告 §6 列出的全部改动（相邻改动合并成一块）。每条都已自检：在 pass8 的逐字摘录上依次应用，结果与 `main_patch.diff` 打完后的文本逐字节相同。


## A1. [gmemory-72b（A3）]

删除 TODO 注释：10 个数全部对上。


**删除这段**

```latex
% TODO[gmemory-72b] The G-Memory 72B values 51.0/20.8/4.7/3.4 come from the Sep 2 and Sep 8 result
% documents (JSON-tolerant scoring). Check them against the data behind Figure 2.
```


## A2. [in-context-library（B）]

RQ1 末句替换为 P 臂结果，删除 TODO 注释。


**原文**

```latex
ToM and zero-shot both emit primitive plans, and ToM differs significantly from zero-shot in none of the 12 paired cells (all $p\geq.296$). In its highest-scoring cell, 72B OOD, ToM raises in-scene-object validity from 52.7\% to 64.4\%, yet success moves from 3.5\% to 2.7\% (Table~\ref{tab:tom_funnel}, Figure~\ref{fig:tom_failure}). Our method instead outputs skill calls, so its gap to the primitive-plan baselines measures the skill-call interface and the library together. The RQ2 arms hold the interface and the planner fixed and change only the library. The no-trace library yields zero successes in every cell, the library without admission averages 21.9\% at 72B, and the admitted library averages 74.6\% (Section~\ref{subsec:induction_validation}). The interface alone therefore produces no success, and within it the library accounts for the gain. Whether a baseline memory would gain from the same interface is not tested.
% TODO[in-context-library] (formerly skill-call-baseline) Put the admitted library in the prompt,
% let the model write primitive plans, score like the baselines. 72B, two CG splits, 594 calls.
% See specs/incontext_library_spec.md. The result replaces the last sentence of this paragraph.
```

**替换为**

```latex
ToM and zero-shot both emit primitive plans, and ToM differs significantly from zero-shot in none of the 12 paired cells (all $p\geq.296$). In its highest-scoring cell, 72B OOD, ToM raises in-scene-object validity from 52.7\% to 64.4\%, yet success moves from 3.5\% to 2.7\% (Table~\ref{tab:tom_funnel}, Figure~\ref{fig:tom_failure}). Our method instead outputs skill calls, so its gap to the primitive-plan baselines measures the skill-call interface and the library together. The RQ2 arms hold the interface and the planner fixed and change only the library. The no-trace library yields zero successes in every cell, the library without admission averages 21.9\% at 72B, and the admitted library averages 74.6\% (Section~\ref{subsec:induction_validation}). The interface alone therefore produces no success, and within it the library accounts for the gain. With the admitted library placed in the prompt and primitive plans as output, success on the two CG splits is 0.0\% with images and 3.0\% without, not above G-Memory under the same output format (Appendix~\ref{sec:viki_audit}). On VIKI-L2 the gain therefore requires the planner that consumes the library, and we treat the planner as part of the method.
```


## A3. [in-context-library（B）]

Limitations 加一句。


**原文**

```latex
\paragraph{Limitations.} We study alternating rather than concurrent actions. Each VIKI-L2 library is built once, and the 30B full arm has one run and no matched no-think arm. The PARTNR libraries behind Tables~\ref{tab:main_results_task} and~\ref{tab:partnr_composition} are built without the execution gate, and the PARTNR gate validation uses privileged state. On PARTNR the partner condition of a cooperation skill is free text that only the planner reads, and observed state changes trigger replanning but are not written into the prompt. The VIKI-L2 CG split has one composition pattern, and no single setting combines decentralized cooperation with a well-powered compositional test. Low 7B sibling-group success precludes transfer claims for smaller backbones.
```

**替换为**

```latex
\paragraph{Limitations.} We study alternating rather than concurrent actions. Each VIKI-L2 library is built once, and the 30B full arm has one run and no matched no-think arm. The PARTNR libraries behind Tables~\ref{tab:main_results_task} and~\ref{tab:partnr_composition} are built without the execution gate, and the PARTNR gate validation uses privileged state. On PARTNR the partner condition of a cooperation skill is free text that only the planner reads, and observed state changes trigger replanning but are not written into the prompt. The VIKI-L2 CG split has one composition pattern, and no single setting combines decentralized cooperation with a well-powered compositional test. On VIKI-L2 the library gains over G-Memory only when the deterministic planner consumes it. Low 7B sibling-group success precludes transfer claims for smaller backbones.
```


## A4. [cg-audit（A2，改判 PASS）+ in-context-library（B）]

这一块合并了三处相邻改动：按 Figure 2 所用库的审计新数改写 CG split audit 段的前两句；删除 `TODO[cg-audit]` 注释；在 sibling-group 段之前新增 In-context library control 段。CG split audit 段的后三句内容未改，但它们和前两句在同一行，所以也出现在这一块的原文和替换文本里。


**原文**

```latex
A read-only audit compares the CG task identifiers with the nine episode pools used to build the library and finds no overlap. The 180 episodes that the proposing agent opens all lie in the induction half, and none of its 884 transcripts contains a CG task identifier. The 297 CG rows hold 295 distinct tasks because two tasks occur twice. The unseen property is the combination of a two-robot cutting subtask with an independent delivery subtask. A weaker structure, an activation step followed by a transport step, does occur in training (1{,}033 rows), so we do not claim that this structure is unseen.
% TODO[cg-audit] Numbers are from the 2026-09-05 audit. Confirm that the nine pools are those of the
% library evaluated in Figure 2 (the 14-family build) and rerun the id intersection if they differ.
```

**替换为**

```latex
A read-only audit compares the CG tasks with every episode pool used to build the library evaluated in Figure~\ref{fig:viki_main}, namely the seeds, holdouts, and tool calls of its 896 induction runs, the support probe, and the 3{,}598-episode pool from which the ordering rules are mined. All 1{,}242 training episodes that the proposing agent addresses through tools lie in the induction half, and none of the 896 transcripts contains a CG task identifier. Four of the 295 CG task identifiers also label training episodes, two of them in the induction half, because VIKI-L2 reuses identifiers across its training and test files. Each paired episode differs from its CG task in instruction, initial positions, goals, and reference plan, and none contains the held-out combination. The 297 CG rows hold 295 distinct tasks because two tasks occur twice. The unseen property is the combination of a two-robot cutting subtask with an independent delivery subtask. A weaker structure, an activation step followed by a transport step, does occur in training (1{,}033 rows), so we do not claim that this structure is unseen.

\paragraph{In-context library control.}
This arm places the admitted library in the prompt of Qwen2.5-VL-72B-Instruct and asks for a primitive plan. It keeps the G-Memory prompt, model, decoding, and scoring of Figure~\ref{fig:viki_main} and replaces only the memory block with one fixed rendering of the eight operators and the three ordering rules, identical for every row. The G-Memory and zero-shot responses are those of Figure~\ref{fig:viki_main}. On CG w/o Image the arm succeeds on 9 of 297 rows under both the JSON-tolerant parser and the official scorer, against 10 for G-Memory (9 vs.\ 10 rows solved by only one arm, $p=1.0$) and 1 for zero-shot (8 vs.\ 0, $p=.0078$). On CG w/ Image it succeeds on no row under either criterion, against 14 for G-Memory (0 vs.\ 14, $p=1.2\times10^{-4}$) and none for zero-shot. Our method solves 247 and 237 rows that this arm misses and loses none ($p<10^{-70}$ on each split). Dropping the second occurrence of the two repeated tasks changes neither direction nor significance. Rerunning the first 20 rows of CG w/o Image changes 19 of the 20 responses and their success count from 0 to 3, so single-run differences of a few rows are within run-to-run variation.
```


---

# 附录 B：B 层判分汇总原文（`work/B/score_summary.json`）

```json
{
 "text": {
  "complete": true,
  "rows": 297,
  "manifest": 297,
  "P_json_tolerant": [
   9,
   297
  ],
  "P_official": [
   9,
   297
  ],
  "GMemory_json_tolerant": [
   10,
   297
  ],
  "GMemory_official": [
   0,
   297
  ],
  "zero_shot_json_tolerant": [
   1,
   297
  ],
  "zero_shot_official": [
   1,
   297
  ],
  "ours_SOLVED": [
   256,
   297
  ],
  "pairing_check": {
   "g_memory_tolerant_recomputed_equals_fig2": 297,
   "zero_shot_tolerant_recomputed_equals_fig2": 297,
   "rows": 297
  },
  "mcnemar_297": {
   "P_vs_GMemory": {
    "first_only": 9,
    "second_only": 10,
    "p": 1.0
   },
   "P_vs_zero_shot": {
    "first_only": 8,
    "second_only": 0,
    "p": 0.0078125
   },
   "ours_vs_P": {
    "first_only": 247,
    "second_only": 0,
    "p": 8.843436600416711e-75
   }
  },
  "mcnemar_295": {
   "P_vs_GMemory": {
    "first_only": 9,
    "second_only": 10,
    "p": 1.0
   },
   "P_vs_zero_shot": {
    "first_only": 8,
    "second_only": 0,
    "p": 0.0078125
   },
   "ours_vs_P": {
    "first_only": 245,
    "second_only": 0,
    "p": 3.5373746401666845e-74
   }
  },
  "rows_295": 295,
  "format_compliance_official": [
   296,
   297
  ],
  "truncated_ge_2000_tokens": 0,
  "prompt_tokens": {
   "P_mean": 1621.962962962963,
   "P_max": 1640,
   "GMemory_mean": 1410.6599326599326,
   "GMemory_max": 1685
  },
  "memory_prompt_sha256_distinct_P": [
   "e8d9028c88578f80d4c2784c1368fb1c237680ca01d4851a1c43592586ad6b77"
  ]
 },
 "imaged": {
  "complete": true,
  "rows": 297,
  "manifest": 297,
  "P_json_tolerant": [
   0,
   297
  ],
  "P_official": [
   0,
   297
  ],
  "GMemory_json_tolerant": [
   14,
   297
  ],
  "GMemory_official": [
   0,
   297
  ],
  "zero_shot_json_tolerant": [
   0,
   297
  ],
  "zero_shot_official": [
   0,
   297
  ],
  "ours_SOLVED": [
   237,
   297
  ],
  "pairing_check": {
   "g_memory_tolerant_recomputed_equals_fig2": 297,
   "zero_shot_tolerant_recomputed_equals_fig2": 297,
   "rows": 297
  },
  "mcnemar_297": {
   "P_vs_GMemory": {
    "first_only": 0,
    "second_only": 14,
    "p": 0.0001220703125
   },
   "P_vs_zero_shot": {
    "first_only": 0,
    "second_only": 0,
    "p": 1.0
   },
   "ours_vs_P": {
    "first_only": 237,
    "second_only": 0,
    "p": 9.055679078826712e-72
   }
  },
  "mcnemar_295": {
   "P_vs_GMemory": {
    "first_only": 0,
    "second_only": 14,
    "p": 0.0001220703125
   },
   "P_vs_zero_shot": {
    "first_only": 0,
    "second_only": 0,
    "p": 1.0
   },
   "ours_vs_P": {
    "first_only": 235,
    "second_only": 0,
    "p": 3.622271631530685e-71
   }
  },
  "rows_295": 295,
  "format_compliance_official": [
   297,
   297
  ],
  "truncated_ge_2000_tokens": 0,
  "prompt_tokens": {
   "P_mean": 1812.6835016835016,
   "P_max": 1825,
   "GMemory_mean": 1709.2491582491582,
   "GMemory_max": 1870
  },
  "memory_prompt_sha256_distinct_P": [
   "e8d9028c88578f80d4c2784c1368fb1c237680ca01d4851a1c43592586ad6b77"
  ],
  "descriptive_canon": {
   "P_canon": [
    10,
    297
   ],
   "G_canon": [
    24,
    297
   ]
  }
 }
}
```


# 附录 C：库渲染块全文（`prompts/library_block.txt`，sha256 `13e33587b2a7123ba6ff8e36e290abb86153d76fbe13129a09a5e23b11d712a1`）

```text
These are reusable skills and ordering rules. Use them to write the plan.

Skill 1 (achievement)
  Effect: ?x is at ?y
  Preconditions: subject_on_agent = false; target_on_agent = false; ?x.subject_in_container = false; ?x.subject_sealed = false; ?y.target_on_agent = false; ?y.target_sealed = false
  Demo: Move(?x) -> Reach(?x) -> Grasp(?x) -> Move(?y) -> Place(?y)

Skill 2 (achievement)
  Effect: ?x is activated
  Preconditions: subject_in_container = false; subject_on_agent = false; subject_sealed = false; ?x.is_device = true
  Demo: Move(?x) -> Interact(?x)

Skill 3 (achievement)
  Effect: ?x is activated
  Preconditions: ?x.is_oven = true
  Demo: Move(?x) -> Interact(?x)

Skill 4 (achievement)
  Effect: ?tool is activated
  Preconditions: ?x.is_cutting_tool = true; ?z1.is_surface = true
  Demo: Move(?x) -> Reach(?x) -> Grasp(?x) -> Move(?z1) -> Reach(?z1) -> Interact(?x)

Skill 5 (achievement)
  Effect: ?x is activated
  Preconditions: ?x.is_knife = true; ?z1.is_cutting_surface = true
  Demo: Move(?x) -> Reach(?x) -> Grasp(?x) -> Move(?z1) -> Interact(?x)

Skill 6 (coordination)
  Effect: ?x is at ?y
  Preconditions: none
  Role ?r0 demo: Move(?x) -> Reach(?x) -> Grasp(?x) -> Move(?y) -> Place(?y)
  Role ?r1 demo: Move(?y) [after role 0 step 5]

Skill 7 (achievement)
  Effect: ?x is at ?y
  Preconditions: subject_in_container = true; subject_on_agent = false; subject_sealed = true; target_on_agent = false; target_sealed = false; ?x.in_container = true; ?y.on_agent = false
  Demo: Move(?z1) -> Reach(?z1) -> Open(?z1) -> Move(?x) -> Reach(?x) -> Grasp(?x) -> Move(?y) -> Place(?y)

Skill 8 (achievement)
  Effect: ?x is at ?y
  Preconditions: subject_in_container = false; subject_on_agent = false; ?x.is_graspable = true; ?x.is_reachable = true; ?y.is_container = true; ?y.is_openable = true
  Demo: Move(?x) -> Reach(?x) -> Grasp(?x) -> Move(?y) -> Reach(?y) -> Open(?y) -> Place(?y)

Ordering rules:
- Do placing an object A at a place P before activating an object B when the skill for B visits P.
- Do placing an object A at a place P before activating an object B when P is the object B and the skill for B visits P.
- Do placing an object A at a place P before placing an object B at a place Q when A is the place Q.
```


# 附录 D：配置 diff（`work/B/config_diff.json`）

```json
{
 "compared": "72B G-Memory arm on CG (amendment10/{text,imaged}/gmemory.jsonl, run 2026-08-31) vs P arm (scripts/incontext_library/run_p_arm.py)",
 "identical_by_construction": {
  "runner": "scripts/viki_amendment10_run.py run() and build_messages(), imported unchanged",
  "served_model": "qwen2.5-vl-72b-amendment3-f2",
  "temperature": 0,
  "max_tokens": 2000,
  "seed": 20260829,
  "extra_body": "none (no think switch at API level; benchmark system prompt asks for <think>/<answer>; VIKI_NO_THINK unset in both)",
  "images": "imaged: bench.get_messages -> PNG base64 image_url; text: no image, same as baseline",
  "memory_insertion": "habitat_llm/evaluation/viki_memory_skill.py:420-434 add_memory_to_messages",
  "scorer": "official scorer via bench.score_response (task_score) + JSON-tolerant viki_report_matrix.tolerant, same as Figure 2",
  "env": {
   "A8B_SKILL_TOPK": "8",
   "A9_ROLE_AWARE": "0",
   "A9_PATTERN_SLOTS": "0",
   "A9_ACTION_CAP": "",
   "A9_STEP_ALIGNED": "",
   "A9_MODE": ""
  }
 },
 "template_check_zero_calls": {
  "text": "297/297 zero-shot prompt_sha256 identical to archived",
  "imaged": "297/297 identical"
 },
 "service": {
  "unchanged_block_reproduced": {
   "model": "Qwen/Qwen2.5-VL-72B-Instruct",
   "revision": "89c86200743eec961a297729e7990e8f2ddbc4c5",
   "served_model": "qwen2.5-vl-72b-amendment3-f2",
   "dtype": "bfloat16",
   "tensor_parallel_size": 4,
   "max_model_len": 16384,
   "max_num_seqs": 16,
   "async_scheduling": true,
   "generation_config": "vllm",
   "seed": 0
  },
  "resource_only_differs": {
   "host": "baseline 192.168.32.40:8050 (other machine, gone) -> this box 127.0.0.1:8050",
   "cuda_visible_devices": "0,1,2,3",
   "gpu_memory_utilization": 0.72
  },
  "not_verifiable": "vLLM build on 192.168.32.40 at 2026-08-31; this box runs vllm 0.11.2 (installed 2026-08-27). The baseline runner was uncommitted at run time (committed 2026-09-02, 89044d1)."
 },
 "differs": {
  "memory_block": "G-Memory: per-row retrieved trajectory+insights (text 1950 chars on row 4, imaged 2029 on row 5); P: fixed library block, 2251 chars (+ newline stripped), sha256 13e33587b2a7123ba6ff8e36e290abb86153d76fbe13129a09a5e23b11d712a1",
  "arm_name": "gmemory -> incontext_library"
 }
}
```
