# partnr-planner

Memory-as-Skill / skill memory v2 的实验仓库。VIKI-L2 和 PARTNR 两条线。

## 开工先读

**`HANDOVER-2026-09-17.md`** ← 当前交接（**两条线**。PARTNR：5 算子纯 LLM 库在冻结 typed 模型臂上对 iir1
**完全打平**（30B +0.002、7B −0.001，四格 369/369），第三臂 h30b 把两个效应拆开、**两个都是零**；
根因是 **typed 要求接口没有 H 谓词这个词**（`RELATIONS` 只有三个放置关系），已加 `typed_state` 开关（默认关闭）；
接口修好后 `is_powered_on` 0→0.727/0.778，`is_clean` 仍 0——又是判据（`project()` 把家具主语一律丢弃，
而池里每条 `is_clean` 都指家具），修好后 **7B 上 0.000→0.327 [+0.204,+0.449]、16 升 0 降，且只有该动的键动了**；
**30B 确认欠着**（八张卡全被占）。VIKI：ToM 无效的机制查清（zero-shot 本来就 100% 在推理伙伴、
ToM 修好 181 格幻觉但 0 格得分、94% 死在模拟器那一关），四臂零 GPU 消融跑出
**拿掉模型的机器人分配反而涨 11.7 分（108 格变好 0 格变坏）**、**re-ask 零贡献**、**ordering 扛结果**。
**现场：远端无本人作业与端点；八张卡全部被别人占；根分区回到 44G。**）

`HANDOVER-2026-09-16b.md`（历史：LLM 生成 memory 在 R 与 H 两族跑通并经留出池确认——30B 归纳 → 按谓词执行门
→ 装库 → 不相交确认池；`conf_H` 上 `is_clean` 0→0.981、`is_powered_on` 0.037→0.926、pc **+0.2563**；
**`is_on_top` 一条 LLM 算子与 20 条规则算子逐条 episode 完全一致**。阴性：`is_next_to` 归纳正确但在特权门下不可验收
（base 无该算子仍满足 83.5%），别再切第三版池子；`is_filled`/`is_inside`/`is_on_floor` 材料只有个位数。
其 §3 优先级 1 已在 09-17 做完，**且它预期的「库变小会改善模型选谓词」被证否——见 09-17 §1a**；优先级 2/3 未动）

`HANDOVER-2026-09-16.md`（历史：组合泛化 7B 上成立、30B 上输给所有强基线；冻结配置
tag `partnr-comp-freeze-0916`、val_mini 7B ss 0.358→0.485、组合轴 0.427→0.799；执行器诊断
typed 0.158 → 特权臂真 DAG 0.451 → ceiling 0.835。其 §3 的优先级 1 与 2 已在 09-16b 做完，
**其中 §3 对 `v2_prompt` 的解读是错的，见 09-16b §1c**）

`HANDOVER-2026-09-15.md`（历史：VIKI-L2 ICLR 2027 补实验 144/144 格交付、PARTNR typed 接口首次 val_mini 0.583/0.590；
其 §3 下一步已全部完成）

`HANDOVER-2026-09-14b.md`（历史：PARTNR 类型化要求接口落地与 train_mini 配对；其 §2 现场、§3 下一步均已完成）。

`HANDOVER-2026-09-14.md`（历史：7B 基线空解析崩溃修复、Achieve 执行对齐；其 §2 现场与「下一步」已过时）。

`HANDOVER-2026-09-13b.md`（历史：30B 输基线的归因、7B 重问无效、Achieve 原型；其 §2 现场已过时）。

`HANDOVER-2026-09-13.md`（历史：spatial 阴性结果、两份论文审计；其 §2「零作业」已过时）。

`HANDOVER-2026-09-09.md`（历史，PARTNR 的结论已定：执行门验收成立、库 21→22、
确认池 +0.6717、privileged val_mini **+0.1093** 且 band 实测 = 0；**唯一没拿到的是 7B/8B 两列，
两次重跑都 `complete: false`，死因是 4h 硬超时与端点中途死亡，不是模型也不是方法**。
现场四条：无作业在跑、**7B 端点 8061 已不在**、箱子上多了别人的两个服务只剩 GPU 0 空、
**这条线的代码一行都没提交**）。
**`RESULTS-2026-09-18.md`**：VIKI 与 PARTNR 的全部表格，由 `scripts/viki_partnr_results_md.py` 从盘上产物直读，**表格不要手改**（§6/§7 是脚本里的手写散文，要改改脚本里的字面量）。重生成：远端 `--tag-prefix v3 --comparison results/agent_library_v3/baseline_comparison.json --out RESULTS-<日期>.md`。09-18 版比 09-13 多四处：**§3b** 零 GPU 重放消融（三模型 × ID/单族，报 no-grounding / no-order / no-reask；**用户定 `no_casting` 不报**）、**§3c** 19 算子参考库对照（**7B 上参考库在 ID 也赢我们**）、**§5c** ToM 失效机制（12 格无一显著）、**§5d** ICLR 补实验 144 格（figure2 / rq2 / rq3 / 配对 / 归纳预算）。同时更正了三处过时散文：no-trace 交出的是 2 个算子不是 0、§6 第 4 条、§7.4。**PARTNR 侧仍只到 09-09 的 `is_in_room`**：typed 模型臂、LLM 生成 memory、H 谓词接口都还没进文档。
这一版是 **v3 库（8 算子）**：72B ID 0.7803、单族留出 0.5422、兄弟组 0.3193，留出列已翻盘。
09-13 补上了 **72B 三轮重复、7B 的消融/重复/no-think**；**还缺 30B 那三组**（要四张卡才能按归档的
TP=4 起，用两卡凑合会把 TP 的数值差异混进重复表的 sd）。**抢卡守卫 `/tmp/serve_30b_when_free.sh`
已于 09-13 03:24 随"让卡"一起停掉，现在没有任何东西在等卡。**09-10 及更早是同一套表的旧版本，**不要混表**。
**`HANDOVER-2026-09-07b.md`**：PARTNR 这条线的方法与全部推导（缺口定位、造轨迹、提议、
免费落地关、执行门双向校准、四条验收判据）。**它更正了 09-07 里「PARTNR 单算子验收门不存在」
那句——死掉的只是 trace-matching 门。**
**`HANDOVER-2026-09-07.md`**：VIKI 主表全部落在 agent 库上（4 个算子、72B ID 0.6126 /
comp 0.8620·0.7980，留出族 0.1818 输给 G-Memory 0.2078），三条自己的诊断被推翻。
**`HANDOVER-2026-09-05b.md`**：PARTNR step-0 猝死已修。注意其中「74.24%」是 memory RA 的数，已作废。
**`HANDOVER-2026-09-05.md`**：挂死根因、P0 72 格、agentic framework 的由来。
更早的 09-02 / 09-03 有 PARTNR 组合泛化对照与意图接口诊断；08-31 / 09-01 是历史。

方法学结论在用户 memory 里（会话启动时自动加载索引），别在交接文档里重复找：
`viki-l2-skill-memory-v2`（方法本体）、`viki-l2-dispatch-reverses`（委派方向反转）、
`partnr-execution-gate`（PARTNR 的验收门与四条判据）、`partnr-is-in-room-wall`（缺口定位）、
`viki-l2-scorer-null-artifact`（打分口径）、`viki-l2-crossmodel-baselines`（30B 那一列 + 三轮 sd
+ no-think 的限定）、`viki-l2-fork-per-request-hang`（三次挂死的真根因）、
`viki-agentic-operator-induction`（agent 推导算子的 framework 与能力曲线，**能力曲线数已过期，
见 `viki-harness-was-the-bottleneck`**）。

## 硬规矩

- **报 VIKI-L2 结果一律用 JSON-tolerant 口径**，官方 scorer 会把 79.5% 的行误判 0 分。
- **PARTNR 的 percent_complete 是连续量**，配对检验用 Wilcoxon 或 bootstrap，不是 McNemar。
- 远端 `ssh aibox-root`，仓库在 `/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner`，mutagen 双向同步。
  评测/归纳脚本的路径常量是远端绝对路径，**run/report 必须在远端跑**。
- `pgrep -f <模式>` / `pkill -f <模式>` 会匹配到自己的 ssh 命令行、误杀远端 shell（**已中招七次**）。
  括号写法 `ps -eo pid,cmd | grep '[d]rivers'` 只防 grep 匹配自己，**防不了模式出现在自己
  父进程命令行里**（09-04 的看守脚本就是这样死锁了整夜）。**等进程按 PID
  （`while kill -0 <pid>`），杀进程按完整二进制路径。**
- **用 heredoc 往远端写脚本，脚本正文会整段出现在写它那个 shell 的命令行里**——09-13
  `viki_wave2.sh` 的 `wait_free` 就被 `bash -c cat > /tmp/x.sh <<EOS ... viki_ours_repeats.sh ...`
  挡了十几分钟。按名字等作业的看守对它无能为力。**跨端点的作业不要互相等**：
  `wait_free` 按脚本名匹配，72B 那台机器上的重复作业会挡住本机 7B 的格。
- **每一类评测格都要有超时守卫**，不能只给上次出事的那一类加。挂死的判据是
  **端点 `num_requests_running` / 连接数为 0**，不是进程忙不忙——挂死时进程照样烧 CPU、开几百线程。
- **报表脚本不带 `--json`（或不重定向）就什么都不留**，会话一死结论就没了。要写进论文的数，落盘再说完成。
  **落盘要放在打印之前**——09-05 一份报表打完全部表格后崩在写 JSON 那一步，等于白跑。
- **超时守卫按真实耗时定，不能照抄别的格。** ID 格最慢 85 分钟，抄 recomb 的 3600 会每轮误杀。
  除了硬超时，还要有 stall 守卫：**输出文件 20 分钟不增长即判死**（runner 逐行 append+flush）。
- **一个端点同时只跑一个生成作业。**
- **驱动缺输入时只打一行就跳过，跑完要数格。** `viki_ours_repeats.sh` 把 comp 的 replay
  拼成 `7b_recomb_*.jsonl`（盘上叫 `m7_*`），于是 30B / 7B 的 comp 重复格**从 v2 起就一直是空的**，
  报表里那两行不见了也没人发现。已修（从 `REPLAY` 取前缀）。
- **某一格得零时，先把参照/已知正确的答案送进同一个判据再下结论。** 09-05 有四次
  「模型不行」最后都是台子的问题（判据只试第一个机器人、按键名丢弃提交、信息不可读）。
  **先读 transcript，别先加采样。**
- **判一个格死活看 `ps` 的 `hydra.run.dir=` 和 `stats/` 目录 mtime，不看 `CELL.json`**——
  被杀那一次留下的 `CELL.json`（`status: 137`）会一直躺到重跑收尾才被覆盖。
- **每次读 `compare_*.json` 先看 `complete` 字段。** 比较器默认在两格的交集上比，
  半格能给出像模像样的增益、基线均值还会悄悄漂。
- **模型格起飞前先探端点（不返 200 就拒绝启动）；Qwen3 系必须 `enable_thinking=False`**
  （经 `extra_body` 透传），否则 planner 在空行处停止生成、可解析 requirement 0 条，
  整列得零而原因是台子。
- **放 habitat 进程前先看那块卡上有什么**（GPU 1 常驻 Qwen3-8B:8101），否则 CUDA OOM。
- **拉 vLLM 一律 `CUDA_VISIBLE_DEVICES=<卡> setsid nohup ...`。** 不写卡号它会去抢 GPU 0；
  不写 `setsid` 它会被父进程的信号带走（09-07 22:08 的 8B 服务就是这样没的）。
- **端点起飞前探一次挡不住「跑到一半死」。** 假格的签名是 `sim_step_count == 0` +
  `runtime ≈ 13 s`，按这个数，别数日志里的 `APIConnectionError`（会少数一个量级）。
- **模型格必须带 `+resume=True`**：一格 5.5 小时，端点死一次就整格作废（已赔约 10 小时）。
  同库同配置逐格确定（band = 0），所以 resume 安全——**但跨代码版本 resume 不安全**。
- **`scp` 往远端仓库塞脚本会和 mutagen 抢**（文件会被抹掉、排的队从没启动）。
  **写在本地让 mutagen 推，核对远端存在后再发。**
- **不要在带残留的目录上重跑**；`rm -rf` 报 "Directory not empty" 之后要核实到空，否则会拿
  上一次的假格做诊断。
- **箱子不再独占**（GPU 6/7 上是别人的服务）。起格前 `nvidia-smi` 看卡，别用两天前的印象排。
- 每个评测格必须独立 `hydra.run.dir`（`paths.results_dir` 挂在它下面，否则会覆盖别的 run）。

## 会话卫生

这个仓库的会话会跑很长、上下文涨得快。**到 300–400K token 就写新的
`HANDOVER-<日期>.md` 并换会话**，同时把上面"开工先读"那一行指向新文件。
超过 500K 之后每轮要几十分钟，700K 之后连 `/compact` 都跑不动。
