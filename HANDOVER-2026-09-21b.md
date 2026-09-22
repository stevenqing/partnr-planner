# HANDOVER 2026-09-21b — ICLR 2027 稿件 TODO spec：A 层交完，B 层未动，四件事等用户拍板

**与 `HANDOVER-2026-09-21.md` 并行**：那份是另一个会话的 PARTNR 线（`is_clean` 30B 确认格，GPU 7 / 端点 8071 是它的），
**别动它的格和端点**。本份只管 ICLR 稿件 TODO spec。结论本体在 memory `iclr-todo-0921-findings`，这里不重复推导。

## 0. 为什么交接

会话 `8c5c1a35` 从 09-17 跑到 09-21，跨了 VIKI 消融、RESULTS 两版、四次提交和整个 ICLR spec 的 A 层，上下文已过 CLAUDE.md 的阈值。

## 09-21 晚：四件事已拍板并执行（先读这一节）

用户的决定：**1 选一（稿件如实声明）、2 先查 MEMENTO、3 同意、4 全做**。执行结果：

- **泄漏范围更正：Table 1/2 是干净的**。按实际运行配置，同时用 episode id 和指令原文两种方式匹配，结果如下：
  - ours：R→R_S 0/76、R+S→R_S 0/76、H_R+H_T→H_R_T 0/11、R+S+H_R+H_T→H_R_S_T 0/5。
  - RAG、MEMENTO：同样全 0。
  - 只有「H_R 记忆 → H_R」重叠：ours 196/197、RAG 196/197、MEMENTO 源集 197/197。
  - 下面 §1 第 1 条「影响 Table 1、Table 2」是错的。
- **B2 已跑完**（用户点头）：MEMENTO H_R_T **1/11 = 9.1%**，完成度 0.393，788 次 call。稿件表格和正文已改（`patch_b2_0921.py`）。详见 report.md「B2」节。远端旧流水线装在 `/mnt/pfs/devs/pn5wp/shishuqing/partnr-isambard`（代码 rsync 自 `~/restore_isambard`，场景数据软链到 A 仓库；Llama-3.1-8B 已下到 `$PFS/hf`；必须用 B 仓库自己的 transformers-CFG）。B3 仍然不跑。
- **main.tex**：`scripts/iclr_todo/patch_decisions_0921.py` 已跑。patch 前的版本存为 `src/main.pre_decisions.tex`，`main_patch.diff` 已重新生成。改了四处：泄漏声明、cells 144+18、参考库去掉 hand-written 并改用逐折数、retrieval-impl 改写并删掉例 (1) 和路由图。
- **新增四个 TODO 注释（超出授权，只标注，没改写）**：
  - `cells-fig`：消融图 PDF 还标着 30B CG「not run」，要重画。
  - `retrieval-impl-gt`：classifier vs GT「routing」表。
  - `retrieval-impl-fallback`：「post-boost」中位数大于 1。
  - `hr-leak`：H_R 单 agent 和 N=3 用的是哪个记忆，盘上没有。
- **RESULTS §3c 已修**：单族留出读 `outputs/viki_ablation/v3_libfold_fold_<M>.json`，由 `scripts/iclr_todo/build_libfold.py` 生成，零 call，ours 复现主表；配对 105/0、98/0、26/0。`RESULTS-2026-09-21.md` 已重新生成，也带上了另一个会话在 17:13 之后改的脚本散文。
- `report.md` 顶部加了「09-21 拍板之后」一节，`answers.json` 新增 `partnr-leak` 项，另外三个 tag 改为 DONE。

## 1. 这一轮改变结论的东西（坏消息在前）

1. **旧 PARTNR 记忆库是从评测 episode 建的**：
   - 库的 `episodic_memory` 键与评测集重叠：HR 库覆盖 H_R 197 个里的 196 个，HRST 库覆盖 H_R_S_T 5 个里的 4 个，R 库覆盖 rerange_only 588 个里的 578 个。
   - 检索时不排除当前 episode。
   - 建库**没有执行门**：LLM 返回的技能都以 `success=True` 入库。
   - 影响范围：Table 1、Table 2，以及任何 H_R 补跑。
2. **"手写 19 算子参考库"不是手写的**：
   - `amendment11/skill_memory_v2.json` 的 `built_from` 是 train 偶数行，另有 8 个逐折版本。
   - 稿件 72B OOD 的 67.4%，以及我 09-18 写进 RESULTS §3c 的单族留出列（0.6742 / 0.5097 / 0.1602），都用了见过留出族的完整库。
   - 公平值（逐折版本，零 call 重放）是 **0.6558 / 0.4600 / 0.1450**，仍高于我们的 0.5422 / 0.3539 / 0.1169（105/0、98/0、26/0，p<3e-8）。**方向没变，数要改**。
3. **Table 1、Table 2、H_R、G-Memory、top-k、latency 的原始输出不在任何盘上**：
   - 它们留在 Isambard 集群 `/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner/outputs/habitat_llm/`。
   - 本地 `~/restore_isambard/partnr-planner` 只有 Llama-3.1-8B 的部分日志，没有一个数对上稿件。
   - 五个 bank 在，计数对得上（450/20/455/12/32），sha 等细节见 report 的 G0 节。
4. **稿件里没有代码支撑的说法**：
   - object-type boost（0.150，综合分 1.117）：没有任何 boost 代码，日志里最高检索分 0.965。
   - 检索时按任务类型选库：运行时没有这一步，库是每次运行的配置写死的。
   - 附录例 (1)、(2)、(3)：日志里都找不到。
   - latency 那组数：没有埋点。
5. **稿件的格数错了**：
   - RQ3 实为 36/36，30B 组合泛化 4 格已于 09-14 补跑；总数是 144，不是 140。
   - 90 格那套矩阵用 skill memory v1 换掉了 ToM，所以 90 − 72 不只是 sibling 那 18 格。
6. **好消息**：判分口径统一以后，60 对胜负方向与显著性零翻转（最大 p=7.2e-6）。现在的混合口径其实偏向 baseline。

## 2. 现场（09-21 17:15 实测）

- **本会话在远端没有任何进程**。`ps` 里的 30B vLLM（:8071，GPU 7）是另一个会话的 PARTNR 格。
- GPU：0–3 各约 10–11G，4 满，5 67G，6 56G，7 63G。
- `calls_log.jsonl` 为 0 字节，B 层零 call。
- git HEAD `f18a5df`。未提交的：
  - `scripts/iclr_todo/`（本会话，未跟踪）
  - `CLAUDE.md`（两个会话都改过）
  - `scripts/drivers/partnr_state_fix_0917.sh`（另一个会话改的，**别替它提交**）
- `results/` 被 ignore，所以交付物不进 git，这是设计如此。

## 3. 下一步

**先等用户拍板四件事**（report.md 末尾也列了）：

1. **PARTNR 泄漏怎么处理**：
   - 选项一：稿件如实声明。
   - 选项二：从非评测 episode 重建 bank 再重跑 Table 1 Llama-8B，约 3,000 次 planner call，还没算建库。这会吃掉 CALL_CAP 8000 的大头。
   - **B3（H_R 补格）在此之前不跑**，否则等于检索测试 episode 本身。
2. **B 层开不开**：
   - B1 的 R 臂在 72B 四个 split 上已零 call 齐了。
   - B1 的 P 臂 72B 共 2,442 行，一行一 call；G 臂可选。
   - B2（MEMENTO H_R_T）11 个 episode。**但先查 MEMENTO 记忆源有没有同样的泄漏。**
3. **cells 口径**：建议写 72 + 36 + 36 = 144，sibling 另写 18 格，Pairing 段的 90 改说法。定了再出 patch。
4. **HALTED 三项的改数要不要做**：
   - handwritten-7b：65.6 / 0.145，去掉"hand-written"。
   - cells：见第 3 条。
   - retrieval-impl：正文两句改成代码事实，同时删附录例 (1) 和路由消融；这超出 spec 授权。

**拍板之后的零 GPU 活**：

- **修 RESULTS §3c 的泄漏**：让 `scripts/viki_partnr_results_md.py` 的 §3c 读 `results/iclr_todo_2026-09-21/work/ref_fold_*.json`，并删掉"参考库不随折变化"那句，然后重新生成 RESULTS。
- **如果要把逐折参考库的格作为正式产物**：把 `work/` 下的三个 json 移到 `outputs/viki_ablation/`（命名如 `v3_libfold_fold_<M>.json`）。
- **main.tex 的 patch**：只在工作副本 `results/iclr_todo_2026-09-21/src/main.tex` 上改，再和 `src/main.orig.tex` 重新 diff。辅助函数在 `scripts/iclr_todo/patch_util.py`（`drop_todo` 会删掉整段 `% TODO[tag]` 注释）。

**要从用户那里拿的输入**：

- `wilson_ci.py`、`partnr_effect_schema_spec.md`：本机和远端都没有。A6 被 G0 卡住，暂时用不上；A3 第 3 条要等五臂结果。
- Isambard 上的评测输出：A5、A6、A7、A11 都靠它。

## 4. 各 tag 状态（细节在 `results/iclr_todo_2026-09-21/answers.json` 和 `report.md`）

| tag | 状态 | patch |
|---|---|---|
| G0 | bank 找到，其余 NOT_FOUND | — |
| retrieval-impl | HALTED | 无 |
| library-provenance | DONE，结局 B | 5.1 两句 + Intro 贡献二收窄 |
| effect-match | 例 (3) 删；213、836 保留等五臂 | 删例 (3) |
| viki-instantiation | DONE | 重写那一段，第 1、3、4 句不准；295 个 task_id 未写入 |
| gmemory-run | BLOCKED | — |
| table1-ci | BLOCKED | — |
| hr-main | BLOCKED（泄漏） | — |
| memento-hrt | NOT_FOUND → B2 | — |
| cells | HALTED | — |
| handwritten-7b | HALTED | — |
| topk-40 | BLOCKED | — |
| scorer-parity | DONE，零翻转 | 表在 `tables/` |

## 5. 文件

| 路径 | 内容 |
|---|---|
| `results/iclr_todo_2026-09-21/{report.md,answers.json,main_patch.diff,calls_log.jsonl}` | 四件交付物 |
| `results/iclr_todo_2026-09-21/src/main.tex` / `main.orig.tex` | 已打 patch 的工作副本 / 用户贴进来的原稿（原件只在对话里，zip 里那版**没有** TODO） |
| `results/iclr_todo_2026-09-21/tables/` | `scorer_parity.{csv,tex,json}`、`scorer_parity_mcnemar.csv`、`reference_library_15cells.csv` |
| `results/iclr_todo_2026-09-21/work/ref_fold_{72B,30B,7B}.json`、`sanity_ref_full_fold_72B.json` | 逐折参考库重放，带 `solved_rows`；sanity 用完整库复现 0.6742 |
| `scripts/iclr_todo/viki_reference_fold_replay.py` | 包一层 `viki_ablation_replay.py`，把 ROOT 指到软链目录（远端 `/tmp/iclr_refroot_<M>`） |
| `scripts/iclr_todo/a12_scorer_parity.py` | 判分口径对齐，零 call，约 1 分钟 |
| `scripts/iclr_todo/patch_util.py` | 改 main.tex 用 |
| `AUDIT-align-2026-09-13.md`（仓库根） | 09-13 对 B 仓库 cooperation 过滤的审计，A1/A3 的依据 |
| `~/restore_isambard/partnr-planner/` | 旧 PARTNR 流水线（B 仓库），bank 与部分日志都在这里 |

## 6. 坑

- **用户说的"本目录下的 main.tex"不在任何盘上**。`~/Downloads/_ICLR2027_Memory_as_Skill.zip` 里那版没有 TODO。带 TODO 的原稿是用户在对话里贴的，已存为 `src/main.orig.tex`。行号以它为准。
- **"参考库不随留出折变化"是错的**：参考库有逐折版本。任何拿参考库和我们比留出列的地方，都要用 `fold_<family>` 版本。
- **同一天有两个会话在写交接**：`HANDOVER-2026-09-21.md` 是 PARTNR 会话的，本份是 `-21b`。改 `CLAUDE.md` 时两行都要留。
- `scripts/viki_report_matrix.py` 是 amendment 时期的脚本，72 和 90 这两个数都不是它出的。72 出自 `results/paper_viki_iclr2027/cells.json`，90 出自 `results/agent_library_v3/baseline_comparison.json`。
- baseline 的 `official_format_score` 是格式分，不是官方任务成功。官方 strict 成功是源文件里的 `task_score`（`viki_amendment8b.py` 里 `bench.score_response`）。
