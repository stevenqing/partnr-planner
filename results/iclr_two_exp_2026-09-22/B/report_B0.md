# Part B0/B1 盘点报告（零调用）

2026-09-22，由 B 子代理的回报整理落盘。偏差清单见 `deviations_B.md`（B-D1 至 B-D13）。

## B0

- **Gate B0：不过**，按用户指示没有停。
  - Table 5 三个条件的运行配置在以下位置都没有：A 仓库、B 仓库（含 git 历史）、`~/restore_isambard`、Downloads 里的 zip 和 tex、远端。稿件自己的 `% TODO[hr-leak]` 注释也写明这些运行只在 Isambard 上。
  - 没有任何仓库里有「只给一个 agent 开 memory」的脚本。
  - Conflict/Ep、FailPick/Ep、SelfConf/Ep、NotClose/Ep 的计算脚本也不在盘上，B4 需要新写。
- **替代配置**：B 仓库的 `run_ours_hr_mem_hr.sh`。它是唯一一个用 H_R 数据加 H_R 库的 ours 脚本（非 v4 模板，max_tokens 1500，k=5）。三个条件只改每个 agent 的 `enable_rag`，关掉的一方拿到空的 `rag_examples`。agent0 是 leader，agent1 是 follower。
- **manifest 里没有 gate**：builder 对成功 episode 调 3 次 LLM，对失败 episode 调 1 次打补丁，然后按 skill 名合并，没有执行检查，也没有三条准入判据。原库是 455 个 skill、1,743 个 instance，来自 196 个 episode（缺 438），其中只有 419 个 instance 记了来源。
- **θ 实际是 0.3**，写死在 `rag.py:468`；instance_top_k 是 5，k=5 走命令行。模板 yaml 里的 `similarity_threshold: 0.7` 没有代码读取。
- **Listing 3 就是 lst:ours**，与盘上任何模板文件都对不上。
- **seed**：`planner_demo.py` 里写死为 47668090，解码贪心。本实验用 `seed_override` 并设 `PYTHONHASHSEED=seed`。所有格固定 NP=8。

## B1

- episode_id 取数据集字段的原样十进制字符串（如 `"433"`）。sha256 mod 2 直接切出 89 / 108。
- 1987、1988、1989 三个 episode 的指令是同一条数据集占位文本（"NOTE: JSON INCORRECT…"），跨了两折。按规则把 1988 从 B 挪到 A，最终 A=90、B=107，移动 1 个。
- 438 没有 heuristic 轨迹，所以 B 折库只用 106 个 episode 建。

## B2（进行中）

- 预估 call：A 折 208，B 折 234，共 442。公式是成功 episode ×3 加失败 episode ×1，这个公式复现了上一轮的 214。
- Llama-3.3-70B，TP=2，GPU 0、1，实测约 77 s/episode。
- 每折建完后自动做覆盖检查（Gate B2）和来源核对，核对不过不装库。

## vLLM 后端验证（偏差：B3 的解码后端与 Table 5 原运行不同）

与上一轮 HF smoke 用同样的 5 个 episode、seed 0、同一配置对比：

| 条件 | 格式错误轮次（HF → vLLM） | 完成度（HF → vLLM） | 单 episode 耗时（HF → vLLM） |
|---|---|---|---|
| 两人都有记忆 | 8.4% → 0% | 0.625 → 0.639 | 1,077 s → 249 s |
| 只有 leader 有记忆 | 11.6% → 5.0% | 0.708 → 0.750 | 1,407 s → 553 s |

- 两次运行的首个 prompt 本身就不同（起始房间不同，检索分数在第二位小数上有差），与后端无关，所以没法逐轮对比。
- 有一个 episode 的 prompt 到了 31.5k token，撞上 32k 上限崩掉，所有端点已改为 64k。
- 原始数据：`backend_validation_C{1,2}.json`。
