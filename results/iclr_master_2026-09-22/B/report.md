# B 部分报告：PARTNR 主表 mean ± std（进行中）

2026-09-22，状态 running。零 call 层的结论见 `../G0/report.md`。

## B.1 零 call 层

- Table 1 的 24 个格：没有一个已有完整运行，全部重跑。盘上只有 Llama-8B 的部分日志：Ours 4 份，MEMENTO 2 份（6 个和 74 个 episode）。
- `TODO[table2-zeroshot]`（50.0 vs 57.3）：盘上没有任何 zero-shot 或 ToM 的运行输出，判定不了差异来自哪里（NOT_FOUND）。pass6 的 TODO 写的是两次不同运行、09-21 对调过。用 seed 0/1/2 重跑后以新值取代。
- 汇总脚本：`~/Downloads/mas_iclr_rev.zip` 里的 `reference/partnr_mean_std.py` 和 `partnr_meanstd_reference.tex`。
- zero-shot 和 ToM 的配置由 `old/run_planner_demo_openrouter_*_{org,tom}.sh`（`zero_shot_prompt_org` / `zero_shot_prompt_tom`，不带 RAG）重建，只换 LLM 块。

## B.2 队列

- 脚本在 `scripts/iclr_master/B/`：`cells.tsv`（69 格，3,204 个 episode）、`queue.sh`、`run_cell.sh`、`cell_verdict.py`、`args/*.args`（归档 Isambard 脚本的 hydra 参数逐字搬来，`MANIFEST.json` 记源 sha256）。
- 状态：远端跑 `bash scripts/iclr_master/B/status.sh`。停队列：`touch partnr-isambard/outputs/iclr_master/B/STOP`。
- seed 控制的是 Python、NumPy、torch 的 RNG 和 `habitat.seed`。新文件 `planner_demo_seeded.py` 只多一个 `+iclr_seed`，原 `planner_demo.py` 未改（sha b5a0f4a2）。解码是贪心：HF 下 `do_sample=False`，vLLM 下 temperature 0。
- 覆盖范围：
  - 跑：Llama-3.1-8B（进程内 HF + transformers-CFG）和 Qwen2.5-7B（vLLM，fp16，与归档的 Isambard Qwen 跑法一致）。
  - BLOCKED：Llama-70B、Qwen2.5-72B（无权重、无卡）；G-Memory（B 流水线里没有实现）。
- 正在处理的台子问题：
  - env_over 开关（见 G0 DISCREPANCIES 第 5 条）；
  - Qwen 的 format_error：Qwen 照模板字面写出 `Action[Explore[bedroom_1]]`，`utils.py:580` 的 `split("[")` 遇到嵌套括号就抛异常，76/76 个 episode 死在第一步。正在查归档的 Qwen 原跑法；
  - GPU 2 上的 OOM。
- 预计耗时：Llama 部分约 412 lane-小时，两条 lane 约 8.6 天。D.2 要用的三格 Llama Ours 约 30 小时出齐。

## 05:0x 更新：队列 04:53 重启

- **跨进程不确定**：同 config、同 seed、同 episode，两次运行的首轮 prompt 就不同。原因是 "Known Rooms" 和 "Seen objects" 由 set 生成，顺序随 PYTHONHASHSEED 变，进而改变检索 query 和检到的 skill。现在每格都设 `PYTHONHASHSEED=<seed>`。第一次跑出的 7 个 episode 移到 `B_evidence/hashseed_nondeterminism/`，不计入结果。
- **env_over 开关**：`+iclr_env_over_metrics=True`（`scripts/iclr_master/B/iclr_env_over.py`），env 报 over 时停止调用 env，按 env 结束时的状态写出 metrics。所有格都开，分母是目标的全部 episode。`planner_demo.py` 未改，`planner_demo_seeded.py`（sha d2087fe3）只多了 seed 和开关两个分支。开关开启时的行为，等第一个被 env 结束的 episode 出现后再核对。
- **偏离 §9（已接受，需用户知悉）**：开关关闭的 5 episode 核对没有做到逐字节一致，停在 4/5，结果见 `B_verify/{switch_off_compare,first_divergence}.txt`。不一致出现在模型生成的文本里（ep 1461 在 `Assigned!<|eot_id|>` 之后的续写处分岔，ep 217 在第一段 thought 里就分岔），是不同 GPU 上 fp16 贪心解码造成的。老文件和它自己比也不一致。结论：差异来自流水线本身的噪声，与开关无关。后果是同 seed 的运行不能逐字节复现，B.3 的 std 里包含生成噪声。
- **Qwen**：归档的 Qwen 脚本（`old/*qwen*.sh`）都是 HF 加 `constrained_generation=False`，并配 Qwen 专用 prompt（`*_qwen.yaml`），这版 prompt 写明了 `Explore[x]` 格式。现在 Qwen 各格用与方法对应的 Qwen prompt。MEMENTO 没有 Qwen 版 prompt，保留 `rag_prompt_enhanced`，格式错误照实报。之前「HF 输出退化」的诊断是错的，vLLM 端点已撤掉。
- **吞吐**：每个进程约 37–41 GB，GPU 2、3 各跑 2 个进程，共 4 个并发。3,204 个 episode，每个约 20 分钟，总共约 270 小时，**约 11 天**。
- **队列顺序**：D 走 D-E3，不再优先跑 Ours 格。改为按 seed 排：所有 Table 1 格先跑 s0，再 s1，再 s2；Table 2 各行同样按 seed 排在其后。

## 06:0x 更新：Qwen Ours s0 前 3 个 episode 全 0（判为模型失败，不是台子问题，格继续跑）

- 读了 1118、1215、202 三个 episode：
  - 动作确实在执行，日志里满是 "Successful execution!"；parse 错误很少。
  - 三个都跑到 20,000 sim 步上限，success 和 pc 都是 0。
  - Agent 1 大部分时间在 `Wait[]`（12–16 次），理由是等 Agent 0 把东西递过来。
  - Agent 0 自己出错：Place 语法写错、去不存在的节点、放它手里没有的物体。
- Qwen 有时输出 `<|im_end|>` 却不带 "Assigned!"，之后继续生成乱码。原因是 `qwen.py` 把停止词设成 "Assigned!" 的最后一个 token。这是归档 Isambard 代码原样的行为，parser 仍能取到正确的动作行。
- prompt、parser、停止词与归档的 Qwen 配置一致（`rag_prompt_sequential_cooperation_skills_qwen`、HF、不开语法约束）。
- DISCREPANCY：稿件里 Qwen2.5-7B Ours 是 0.574。原 Table 1 Qwen 格的配置 NOT_FOUND，可能用了盘上没有的配置或别的 prompt。等各方法 s0 跑完再判断整列。

## 09:38 更新

- Llama Ours s0 缺 metrics 的 6 个 episode（165、174、202、858、1098、1317）全是 GPU 2 上的 CUDA OOM。每个 Llama 进程峰值 44–48 GB，同卡两个进程超过 95 GB。
- 修法：Llama 格改为每卡 1 个进程（`queue.sh` 里 llama8b 设 `NP=1`），Qwen 仍是每卡 2 个。该格按 PID 停掉后以 num_proc=1 resume，重跑 6 个 OOM episode 和尚未跑的 45 个；已完成的 19 个保留。
- env_over 开关在两条路径上都生效：Llama 日志里 3 次 `ICLR_ENV_OVER`，对应 episode 都写出了 metrics；`B_verify/switch_on_check.txt` 核对了 Qwen ep 1215（VERDICT: metrics written）。
- 代价：Llama 部分吞吐减半，总预计从约 11 天增加到约 16 天。

## 13:20 用户决定：B 停止，只做 C

- 13:20 按进程组停掉 B 全部 lane，并放了 `outputs/iclr_master/B/STOP`。
- 盘上保留的部分结果，均未跑完，不报：
  - Llama Ours s0 34/76
  - Qwen Ours s0 64/76
  - Llama zs s0 15/76
- GPU 2、3、4 转给 C。
