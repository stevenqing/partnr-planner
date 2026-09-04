# partnr-planner

Memory-as-Skill / skill memory v2 的实验仓库。VIKI-L2 和 PARTNR 两条线。

## 开工先读

**`HANDOVER-2026-09-04.md`** ← 当前交接（30B 跨模型基线链条挂死的现场、13/60 的真实进度、
两处「结论有了但盘上没有」、机器上只剩一张空卡）。
**`HANDOVER-2026-09-03.md`** / **`HANDOVER-2026-09-02.md`** 配着读：PARTNR 组合泛化对照的
结论、意图接口诊断，以及一批仍然会踩的坑在那里。
更早的 `HANDOVER-2026-08-31.md` / `HANDOVER-2026-09-01.md` 是历史，除非要追溯否则不必读。

方法学结论在用户 memory 里（会话启动时自动加载索引），别在交接文档里重复找：
`viki-l2-skill-memory-v2`（方法本体）、`viki-l2-dispatch-reverses`（委派方向反转）、
`viki-partnr-v2-port`（PARTNR 移植，负结果）、`viki-l2-scorer-null-artifact`（打分口径）、
`viki-l2-crossmodel-baselines`（30B 那一列，G-Memory 换模型就崩）。

## 硬规矩

- **报 VIKI-L2 结果一律用 JSON-tolerant 口径**，官方 scorer 会把 79.5% 的行误判 0 分。
- **PARTNR 的 percent_complete 是连续量**，配对检验用 Wilcoxon 或 bootstrap，不是 McNemar。
- 远端 `ssh aibox-root`，仓库在 `/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner`，mutagen 双向同步。
  评测/归纳脚本的路径常量是远端绝对路径，**run/report 必须在远端跑**。
- `pgrep -f <模式>` 会匹配到自己的 ssh 命令行、误杀远端 shell（**已中招四次**）。
  括号写法 `ps -eo pid,cmd | grep '[d]rivers'` 只防 grep 匹配自己，**防不了模式出现在自己
  父进程命令行里**（09-04 的看守脚本就是这样死锁了整夜）。**等进程按 PID
  （`while kill -0 <pid>`），杀进程按完整二进制路径。**
- **每一类评测格都要有超时守卫**，不能只给上次出事的那一类加。挂死的判据是
  **端点 `num_requests_running` / 连接数为 0**，不是进程忙不忙——挂死时进程照样烧 CPU、开几百线程。
- **报表脚本不带 `--json`（或不重定向）就什么都不留**，会话一死结论就没了。要写进论文的数，落盘再说完成。
- 每个评测格必须独立 `hydra.run.dir`（`paths.results_dir` 挂在它下面，否则会覆盖别的 run）。

## 会话卫生

这个仓库的会话会跑很长、上下文涨得快。**到 300–400K token 就写新的
`HANDOVER-<日期>.md` 并换会话**，同时把上面"开工先读"那一行指向新文件。
超过 500K 之后每轮要几十分钟，700K 之后连 `/compact` 都跑不动。
