# ICLR 2027 Memory-as-Skill 稿件 TODO 执行 spec

日期 2026-09-21。对象是本目录下的 `main.tex`，里面有 14 处 `% TODO[tag]` 注释，共 12 个 tag。本 spec 逐个 tag 说明要查什么、去哪查、交什么、查到之后稿子怎么改。

## 0. 总则

1. 数字只能来自盘上产物。每个交付的数字都要带文件路径、行或键、计算方式。找不到产物就报 `NOT_FOUND`，不允许用重建的库或重写的脚本替代原始产物。
2. 先查后跑。A 层任务全部零 LLM call、只读。B 层任务要发 call，必须等 A 层报告交完并且本 spec 写明的前置条件满足才开始。
3. 不改冻结产物。不改已有结果文件、库文件、评测脚本的判分逻辑。新脚本放 `scripts/iclr_todo/`。
4. `main.tex` 的改动只以 patch 形式交付，不直接覆盖。patch 只允许动 TODO 注释附近的句子、对应的表格单元和 caption。某个 TODO 解决后，patch 里同时删掉那条注释。没解决的 TODO 原样保留。
5. 正文文风约束，写 patch 时遵守。不用破折号，句中不用冒号和分号，方法和结果用现在时，不用 in which 这类间接从句，不加粗不加斜体，不用 audit、supplied、provenance、preregistered、row-level、replay 这类内部流程用语。
6. 正文 9 页是硬上限，目前第 9 页约剩三分之一页。每条 patch 注明净增行数的估计。

## 1. 交付物

全部放在 `results/iclr_todo_2026-09-21/`。

- `report.md`，按 tag 分节，每节写结局编号、证据、建议的稿件改动。
- `answers.json`，每个 tag 一个对象，字段为 `status`（DONE / NOT_FOUND / BLOCKED / HALTED）、`ending`、`numbers`（每个数字带 `value`、`numerator`、`denominator`、`source_path`）、`notes`。
- `main_patch.diff`，对 `main.tex` 的统一 diff。
- `tables/`，新表的 LaTeX 片段与对应 CSV。
- `calls_log.jsonl`，B 层每次 LLM call 一行。A 层结束时该文件必须为空。

## 2. Gate G0，PARTNR 产物定位

8 月那轮 C4 被卡住的原因是 Table 1 和 Fig 3 背后的冻结 PARTNR memory bank 与生成日志不在 agent 工作区。先解决这件事。

查找并列出以下产物的路径、文件数、hash。

- Table 1 六个 backbone、六种方法的逐 episode 结果
- Table 2 四行迁移配置的逐 episode 结果
- 五个 PARTNR memory bank（R 450、S 20、HR 455、HT 12、HRST 32）及其建库日志
- H_R 任务组所有方法的逐 episode 结果
- G-Memory 与 MEMENTO 的运行配置和日志
- top-k 扫描（k 取 3、5、10、20）的逐 episode 结果

找不到的项写进 `report.md` 的 G0 节，作为需要用户交接的清单。依赖该项的 tag 标 `BLOCKED`，其余照常做。

## 3. A 层，零 call 只读任务

### A1 `retrieval-impl`（main.tex 第 203 行附近）

问题。正文 Eq. 1 后面补的两句是从附录反推的，需要对代码。

查 `hierarchical_retrieval.py` 及其调用方，回答四件事。
1. task-type classifier 是否在检索前把 query 路由到对应类型的 bank，出 Table 1 的运行是否开启。
2. object-type boost 的数值、触发条件、加在 skill 层分数上还是 instance 层分数上。
3. 阈值 θ 比较的是 boost 前还是 boost 后的分数。
4. embedding 模型是否为 `all-mpnet-base-v2`。

结局。与正文一致则删 TODO。不一致则 patch 改正文那两句，只陈述代码事实。

### A2 `library-provenance`（第 171 行附近）

问题。Table 1 和 Table 2 用的 PARTNR bank 有没有过 Stage 1 的 execution gate。

查建库日志与库文件。回答。
1. 每个 bank 的建库脚本、proposing model、日期。
2. 建库过程是否对每个 proposal 在独立 validation episode 上执行并按三条判据筛选。
3. 每个 bank 里 instance 数为 1 的 skill 占比（判据 iii 要求跨 trace 复现）。
4. 第 5.3 节那个 `is_in_room` operator 所在的库（21→22）与 Table 1 的 bank 是不是同一个库。

结局。
- A 过了 gate。删 TODO，在 Setup 加一句说明。
- B 没过 gate。patch 在 Section 5.1 加注释里那句声明，把 Introduction 的贡献二收窄为 gate 在 VIKI-L2 与一个 PARTNR skill 上验证。
- 重建 bank 并重跑 Table 1 不在本 spec 授权范围内，只在报告里给出 call 数估算。

### A3 `effect-match` 的只读部分（第 213、836、943 行）

4.2 节的重写取决于 `partnr_effect_schema_spec.md` 的五臂结果，那份 spec 单独执行，这里不重复。本任务只做三件只读的事。
1. 附录 latency 一节的 executability filtering 0.43 ms 量的是哪个函数，是否含 LLM 调用。
2. 附录 Qualitative Retrieval Examples 的例 (3)，在出数运行的日志里是否存在这样的激活记录。存在则给出 episode 与 step。不存在则 patch 删掉例 (3)。
3. 如果五臂实验已有结果，按那份 spec 的预写结局给出 4.2 节两个段落的 patch。C 胜 E 则写成 `{key, subject, value}` schema 上的确定性绑定。C 平 E 则删去 effect indexing 的主张，abstract 与 Introduction 同步改。没有结果则这三处 TODO 保留。

### A4 `viki-instantiation`（第 248 行附近）

逐句核对 Setup 里 Instantiation on VIKI-L2 这一段。每句给出 TRUE / FALSE / IMPRECISE 和代码或数据依据。
1. 每个 episode 给一条 instruction 和一张场景图，要求一次输出完整多机器人计划。
2. 没有 alternating round，库的 cooperation 分支在该 benchmark 上为空。
3. 本方法输出带机器人指派的 skill call 序列，指派由 LLM 决定。
4. 确定性 planner 用 operator 的有序 demo 把每个 call 展开为 primitive action。
5. 所有 baseline 直接输出 primitive plan。
6. 每条 CG instruction 由一个双机器人切的子任务和一个独立搬运子任务组成，训练集里没有这个组合，各部分单独出现过。

第 6 句只能用 9 月 5 日泄漏审计允许的表述，不得写成激活加搬运未见过。另报 297 行里不同 `task_id` 的个数（审计结果是 295），由用户决定是否写进正文。

### A5 `gmemory-run`（第 257 行附近）

回答。G-Memory 用的是哪份代码和 commit，memory 由哪个 backbone 在哪批轨迹上构建，是否与其他 memory 方法共用 heuristic-agent 轨迹，在线更新是否关闭。Claude 与 GPT-4o 两格为什么没有结果（没跑、跑了未完成、跑了没通过核对，三选一）。

交付一句可以直接放进 Baselines 段的英文，以及 Table 1 caption 里 N/A 的准确说法。

### A6 `table1-ci`（第 272 行）

对 Table 1 的 36 个 success 单元和 Table 2 的 24 个单元，给出分子、分母、运行次数。显示值不是 1/76 的整数倍，要说明显示值是怎么得到的（多次运行平均，还是分母不是 76）。

用本目录的 `wilson_ci.py` 算 95% Wilson 区间。多次运行平均的格子改用 episode 级 paired bootstrap，10,000 次重采样，seed 20260921。

另对每个 backbone 做 ours 对最强 baseline 的 paired McNemar exact。要求两臂 episode 集合相同，不同则报告交集大小并在交集上算。

交付。带 n 的 Table 1 新版 LaTeX（区间放附录新表，正文表只加 n 一行），以及 Table 2 每格的分子分母。H_R_S_T 一行各方法分母不一致的情况照实列出。

### A7 `hr-main`（第 275 行）

H_R 是最大的任务组，197 个 episode。从盘上汇总六种方法在 H_R 上的 success 与 completion，backbone 至少含 Llama-3.1-8B，其余有多少报多少。格式与 Table 1 相同，带 n 和 Wilson 区间。

结局。日志齐全则交附录新表和正文一句话。缺方法或缺 backbone 则列出缺格，转入 B3。

### A8 `memento-hrt`（第 370 行）

在日志里找 Table 2 的 MEMENTO H_R_T 14.5%。找到则给分子分母和运行配置，patch 删掉正文那句说明。找不到转入 B2。

### A9 `cells`（第 475 行）

附录同一节里一处写 72 个 main-comparison cell，另一处写 90 个组合。用 `scripts/viki_report_matrix.py` 的输出核对模型、split、方法三个维度的实际格数，说明 72 与 90 的差是不是 sibling-group split 的 18 格。patch 统一成一个口径，140 这个总数相应更新。

### A10 `handwritten-7b`（第 505 行）

附录写手写 19-operator 库在 7B ID/OOD 上是 0.160 对 0.139/0.117，一个参考值对两个 split。给出手写库在三个模型、五个 split 上的全部 15 格，标明哪些格盘上已有。patch 把那句改成 ID 与 OOD 分开写。缺格转入 B1。

### A11 `topk-40`（第 857 行）

H_R_S_T 在 R-only memory、k=5 下是 40.0%，Table 2 里完整 source union 下也是 40.0%。确认这是两次不同的运行，给出各自的配置文件、分子分母。如果其实是同一次运行被标错，指出正确的标签，patch 改 top-k 一节的正文和 caption。

### A12 `scorer-parity`（无对应 TODO，新增）

正文承认 ours 用 strict `SOLVED`，baseline 用 JSON-tolerant parser。用已有的逐行输出，把 VIKI-L2 主对比的全部格子在两种判分下各算一遍，零 call。交一张同判分口径的附录表。如果同口径下任何一格的胜负方向翻转，在报告开头单独标出。

## 4. B 层，需要 LLM call 的任务

开始条件。A 层报告已交，`calls_log.jsonl` 为空已核对。总 CALL_CAP 为 8000，超出即停。温度 0，prompt 与参数逐 call 落盘。

### B1 `skill-call-baseline`（第 383 行）

目的。把 skill-call 接口的收益和 learned library 的收益分开。只跑 Qwen2.5-VL-72B，四个 split（ID 924、single-family OOD 924、CG w/ Image 297、CG w/o Image 297）。

臂。
- R 参考臂，手写 19-operator 库加同一套 skill-call 接口。先用 A10 确认盘上已有哪些格，只补缺格。
- P 本方法的 primitive 输出臂。把 learned library 的 operator demo 作为 in-context 示例放进 prompt，模型直接输出 primitive plan，判分与 baseline 同口径。每个 split 一次 call 一行。
- G 给 G-Memory 配 skill-call 接口，检索内容不变，输出 skill call，用手写库展开。可选，CALL_CAP 有余量才跑。

分析。全部 paired McNemar exact，报 ours 对 R、ours 对 P、P 对 zero-shot、P 对 G-Memory。

预写结局。
- E1，P 显著高于 G-Memory 与 zero-shot。memory 内容本身有贡献，正文写明。
- E2，P 与 baseline 无显著差异。增益来自接口与 planner，正文如实写，RQ1 的 VIKI 段改成接口加库的联合效果。
- E3，P 低于 zero-shot。同 E2，并在 Limitations 加一句。

三种结局都进论文，不因结局不利而不报。

### B2 `memento-hrt` 重跑

仅当 A8 为 NOT_FOUND。用 Table 2 同配置重跑 MEMENTO 在 H_R_T（H_R+H_T memory，Llama-3.1-8B）上的 11 个 episode。结果替换表中 14.5，patch 删掉正文那句说明。

### B3 `hr-main` 补格

仅当 A7 有缺格。只补 Llama-3.1-8B 这一列的缺失方法，197 个 episode。其他 backbone 不补。

## 5. Halt 规则

- 任一 A 层任务发现盘上数字与 `main.tex` 里的数字不一致，不自行改数。在报告开头的 DISCREPANCIES 节列出稿件值、盘上值、路径，该 tag 标 `HALTED`，其余任务继续。
- B 层任一臂的格式合规率低于 90%，保留原始结果并照报，不修复，不重跑。
- 端点错误或超时导致某个 split 未跑完，该 split 标 incomplete，不报部分结果。
- 不新增本 spec 之外的实验臂。
