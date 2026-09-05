# partnr-planner

Memory-as-Skill / skill memory v2 的实验仓库。VIKI-L2 和 PARTNR 两条线。

## 开工先读

**`HANDOVER-2026-09-05b.md`** ← 当前交接（PARTNR step-0 猝死已修、六格重测完，差距
0.181→0.113、ordering 消融不再成立；VIKI agent 算子库 32.47%→74.24%，全部来自修 harness；
能力曲线重测后 7B 不再是 0，「阈值」说法被推翻）。**它取代 09-05 那份的结论 1 和 3。**
**`HANDOVER-2026-09-05.md`** 配着读：挂死根因、P0 72 格、agentic framework 的由来。
更早的 09-02 / 09-03 有 PARTNR 组合泛化对照与意图接口诊断；08-31 / 09-01 是历史。

方法学结论在用户 memory 里（会话启动时自动加载索引），别在交接文档里重复找：
`viki-l2-skill-memory-v2`（方法本体）、`viki-l2-dispatch-reverses`（委派方向反转）、
`viki-partnr-v2-port`（PARTNR 移植，负结果）、`viki-l2-scorer-null-artifact`（打分口径）、
`viki-l2-crossmodel-baselines`（30B 那一列 + 三轮 sd + no-think 的限定）、
`viki-l2-fork-per-request-hang`（三次挂死的真根因）、
`viki-agentic-operator-induction`（agent 推导算子的 framework 与能力曲线，**能力曲线数已过期，
见 `viki-harness-was-the-bottleneck`**）。

## 硬规矩

- **报 VIKI-L2 结果一律用 JSON-tolerant 口径**，官方 scorer 会把 79.5% 的行误判 0 分。
- **PARTNR 的 percent_complete 是连续量**，配对检验用 Wilcoxon 或 bootstrap，不是 McNemar。
- 远端 `ssh aibox-root`，仓库在 `/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner`，mutagen 双向同步。
  评测/归纳脚本的路径常量是远端绝对路径，**run/report 必须在远端跑**。
- `pgrep -f <模式>` 会匹配到自己的 ssh 命令行、误杀远端 shell（**已中招六次**）。
  括号写法 `ps -eo pid,cmd | grep '[d]rivers'` 只防 grep 匹配自己，**防不了模式出现在自己
  父进程命令行里**（09-04 的看守脚本就是这样死锁了整夜）。**等进程按 PID
  （`while kill -0 <pid>`），杀进程按完整二进制路径。**
- **每一类评测格都要有超时守卫**，不能只给上次出事的那一类加。挂死的判据是
  **端点 `num_requests_running` / 连接数为 0**，不是进程忙不忙——挂死时进程照样烧 CPU、开几百线程。
- **报表脚本不带 `--json`（或不重定向）就什么都不留**，会话一死结论就没了。要写进论文的数，落盘再说完成。
  **落盘要放在打印之前**——09-05 一份报表打完全部表格后崩在写 JSON 那一步，等于白跑。
- **超时守卫按真实耗时定，不能照抄别的格。** ID 格最慢 85 分钟，抄 recomb 的 3600 会每轮误杀。
  除了硬超时，还要有 stall 守卫：**输出文件 20 分钟不增长即判死**（runner 逐行 append+flush）。
- **一个端点同时只跑一个生成作业。**
- **某一格得零时，先把参照/已知正确的答案送进同一个判据再下结论。** 09-05 有四次
  「模型不行」最后都是台子的问题（判据只试第一个机器人、按键名丢弃提交、信息不可读）。
  **先读 transcript，别先加采样。**
- 每个评测格必须独立 `hydra.run.dir`（`paths.results_dir` 挂在它下面，否则会覆盖别的 run）。

## 会话卫生

这个仓库的会话会跑很长、上下文涨得快。**到 300–400K token 就写新的
`HANDOVER-<日期>.md` 并换会话**，同时把上面"开工先读"那一行指向新文件。
超过 500K 之后每轮要几十分钟，700K 之后连 `/compact` 都跑不动。
