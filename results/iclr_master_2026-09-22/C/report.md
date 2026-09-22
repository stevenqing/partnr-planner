# C 部分报告：H_R held-out 上的角色区分检索（进行中）

2026-09-22。C.1 完成，C.2 从 06:00 开始跑。由 C 子代理的回报整理落盘（子代理写不了 .md）。

## DISCREPANCIES

1. H_R 库 1,743 个 instance 里只有 419 个（24%）记了来源 episode，而且都来自打过补丁的失败 episode，没有一个记步区间。路线 P 不成立，按 spec 走路线 Q：
   - 用归档的 builder（`--include-failed --patch-failed`），模型是 Llama-3.3-70B，在 build 半的 98 个 episode 上重建。这 98 个里有 40 个失败、打了补丁。
   - 共 214 次 call，0 次失败，记在 `calls_log.jsonl`，不计入 EPISODE_CAP。
   - 与原配置只在资源项上不同：TP 从 4 改为 2，util 0.8，enforce-eager。权重来自 ModelScope，41 个文件逐个核对过 sha256。
   - 另外做了一个零 call 的 P' 库（事后归因，只保留来源全在 build 半的 instance，共 261 个 skill），放在 `heldout_bank_Pprime_record_only/`，只作记录，不参与任何运行。
2. `tab:hr_failure` 的配置盘上没有。C.2 改用 Table 1 的 Llama-8B ours 配置（`run_ours_rs_mem_r.sh`，v4 模板，max_tokens 2000，top-k 5）。B 仓库里 H_R 脚本与它的差别是用了非 v4 模板、max_tokens 1500。
3. `planner_demo.py` 里 seed 写死为 47668090，解码是贪心。seed 0/1/2 通过新开关 `seed_override` 传入，控制 python、numpy、torch 的 RNG 和 `habitat.seed`，同时把 PYTHONHASHSEED 设为 seed，解码不变。
4. 不固定 PYTHONHASHSEED 时，原代码在两个进程里的首个 prompt 就不同（房间列表来自 set）。固定后比了 5 个 episode：
   - 开关关闭的路径在 4 个可比 episode 上一致。其中 3 个的 prompt、trace、动作序列、stats 逐字节相同；443 两边都在同一处因 Episode over 崩掉，但它没有 prompt 文件，所以逐字节比较只覆盖前 3 个。
   - 445 在 ref 侧被其他用户的进程挤到 CUDA OOM，不可比。
5. C.1.2 跳过：盘上没有 Both memory 条件下每个 agent 进 prompt 的 instance 日志。这组数据由 C.2 的 `retrieval_dump` 补上。

## C.0.3 与 C.1.1

- H_R 库在 B 仓库 `data/hierarchical_skill_memory/hierarchical_heterogeneous_rerange`：280 个 individual + 175 个 cooperation = 455 个 skill，1,743 个 instance。
- 切分：197 个 H_R episode 按 id 升序，偶数位 99 个进 build（sha `1327ca31…`），奇数位 98 个进 eval（sha `9b8a87de…`）。438 号不在 heuristic 日志里，所以 build 实际是 98 个。
- 路线 Q 库（`heldout_bank_q/`）：

| 类型 | skill 前→后 | instance 前→后 |
|---|---|---|
| individual | 280 → 185 | 1,189 → 601 |
| cooperation | 175 → 111 | 554 → 273 |

  库里的 98 个来源 episode 全部在 build 半。
- 路线 Q 是重新生成的，skill 名字和原库不同，所以不存在 spec 所说的「名字可能是看着 eval 轨迹提出来的」那类 skill。写论文时这个限定改成：库由同一 builder 只在 build 半上重建。

## C.2 队列

- 条件：
  - C3 用 B 仓库已有的开关 `include_ind_skills=False`，follower 只从 cooperation 分支检索。
  - 新开关都默认关闭：`seed_override`、`role_hint`（C4）、`retrieval_dump_dir`，外加和 B 部分相同的 `iclr_env_over_metrics`。
- 两条 lane（GPU 0、1），每条 num_proc=2：
  - 先对每个条件冒烟 5 个 episode，冒烟门不过（出现假格、dump 少于 2 份/episode、或完成不到 3 个）就停掉整条 lane；
  - 通过后接 15 格正式运行，每格带 resume，最多重试 3 次，硬超时 30 小时，另有 20 分钟 stall 守卫，1,700 个 episode 前自动停。
- 06:29 第一个真 episode 跑完（smoke_C1_s0）：sim 步数 > 0，耗时 1,692 秒，completion 0.667，retrieval_dump 正常写出。
- 耗时：每个 episode 实测 10–55 分钟，均值约 25 分钟。4 个并发，1,470 个 episode 约 150 小时，约 6 天。
- 风险：两张卡已用到 86–88G/97G，GPU0 上别人的进程会周期性占显存。OOM 的 episode 记为 harness 失败，resume 时重跑。
- 状态：远端 `bash scripts/iclr_master/C/status.sh`。输出在 `partnr-isambard-C/outputs/iclr_master_C/`。

## 结局

数据还没出，暂不判定。

## 09:0x 更新

- C1、C3 的冒烟都过了（5/5）。C0、C2、C4 还在跑。
- **noneaction**：smoke_C0 的 ep 439 崩在 `dynamic_world_graph.py:1310`。搭档的 high-level action 是 None（planner 输出没解析出来），对它调 `.lower()` 就报错。未打补丁的 Cref 里是同一行，所以这是原代码本来就会崩，不是我们的开关造成的。
  - 固定 seed 下它是确定性的。
  - 单列为 noneaction：计入 done，success 记 0，按格式错误报告，不当 harness 失败反复重试。
  - 每格报 noneaction 率，超过 10% 按 spec 照报，不修。
  - 无记忆的 agent 手里没有检索块里的动作格式示例，所以 C0、C2 这类条件预计 noneaction 会更多。
  - B 部分统一用这个口径。
- **OOM**：GPU 0 和 GPU 1 都出现过，都是别人的进程周期性占用显存。num_proc 保持 2，因为降到 1 队列要约 12 天。OOM 的 episode 记为 harness 失败，resume 时重跑，每格最多 3 次，3 次后仍不完整的格再单独降。

## 13:22 全速

- B 停止后，C 的 worker 扩到 GPU 0、1、2、3、4、5 共六张卡，每卡 NP=2，全部开 `+iclr_share_llm=True`（开关验证见 `B_verify/share_llm/VERDICT.txt`：Llama、Qwen 各 2 个 episode 逐字节一致）。
- 15 格共用一个队列，按格领取。

## 15:13 耗时修正与队列重排

- 冒烟实测，每个 episode 的平均/最长耗时：C0 为 3,014 s / 8,647 s，C2 为 3,230 s / 8,810 s，C1 为 1,200 s / 1,825 s。有 zero-shot agent 的条件（C0 两边都是，C2 的 follower 是）约慢 2.5 倍。C0_s0 正式格里单步要 270–440 s，GPU3 利用率 99%，是生成本身慢，不是卡死。
- 按最长优先重排 `queue.txt`（原顺序存为 `queue.txt.orig_0922`）：已领的 6 格之后，先排 C0_s1、C2_s1、C0_s2、C2_s2，再排 C3_s1、C4_s1、C1_s2、C3_s2、C4_s2。
- 六个 worker 按 PID 重起为 adopt 模式，接管各自正在跑的格，没有打断任何 episode。
- 修正后的估计：剩余约 410 格·小时，6 张卡，按格粒度排完约 70–80 小时，预计 09-25 下午到晚上完成。

## 15:3x 用户决定：C 停止

- 按 PID 停掉全部 worker 和在跑的格，各卡都放了 STOP_gpu 文件。已完成的 episode 留在盘上，但没有一格跑完，不报。
- 改做新 spec `results/iclr_two_exp_2026-09-22/SPEC.md`，其 Part B 取代本部分。
