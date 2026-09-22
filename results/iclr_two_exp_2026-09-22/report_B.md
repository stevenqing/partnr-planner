# Part B 报告：H_R held-out 上的单边记忆

2026-09-23。由 Part B 子代理的回报整理落盘。数据：`tables/hr_heldout.csv`、`B/b4_summary.json`、`B/b4_per_episode.csv`、`B/gate_b2_fold{A,B}.json`、`B/sources_check_fold{A,B}.json`、`B/hr_fold{A,B}_library.json`。偏差见 `deviations.md` 的 Part B 节（B-D1 到 B-D15）。

## 设置

- 切分 `sha256(episode_id) mod 2`：A 折 90、B 折 107（1988 因与 1987/1989 指令相同，从 B 挪到 A）。每折用另一折的库评测，合计 n=197。
- 两个折库用归档 builder 与同一提议模型 Llama-3.3-70B 重建，共 442 次 call，0 次失败。fold A 库 168 individual + 96 cooperation（825 instance，来源 90 个 episode），fold B 库 193 + 102（933 instance，来源 105 个）。
- **Gate B2 通过**：Pick、Place、Clean、Fill、PowerOn、PowerOff 六个动作在两折的 individual 与 cooperation 分支都各有 ≥3 个 skill（最少的是 fold B 的 cooperation:PowerOff = 3）。原库按同一判据也通过。
- **来源核对通过**：两折库的 provenance、episodic memory、显式 e_src 全部落在本折内，与被评测折交集为空。
- **Gate B0 没过**（Table 5 的运行配置不在盘上），改用 `run_ours_hr_mem_hr.sh`，三个条件只差每个 agent 的 `enable_rag`。解码后端从 HF 换成 vLLM（B-D10，格式错误率没有上升）。

## 结果（三 seed 均值 ± 样本 std，每 seed n=197）

| 条件 | Succ. | Comp. | Conflict/Ep | FailPick/Ep | SelfConf/Ep | NotClose/Ep |
|---|---|---|---|---|---|---|
| Both（held-out） | **0.225 ± 0.016** | 0.551 ± 0.014 | 0.172 ± 0.056 | 4.99 ± 0.15 | 0.416 ± 0.069 | 6.46 ± 0.56 |
| Agent 0 only | 0.135 ± 0.028 | 0.494 ± 0.029 | 0.346 ± 0.071 | 5.04 ± 0.13 | 0.330 ± 0.060 | 6.32 ± 0.30 |
| Agent 1 only | 0.184 ± 0.021 | 0.523 ± 0.015 | **0.061 ± 0.033** | 4.27 ± 0.07 | 0.254 ± 0.065 | 6.21 ± 0.03 |
| Both（seen，Table 5） | 0.223 | — | 1.63 | 1.69 | 0.27 | 15.99 |
| Agent 0 only（seen） | 0.496 | — | 0.33 | 1.83 | 1.36 | 9.82 |
| Agent 1 only（seen） | 0.056 | — | 0.75 | 2.69 | 1.51 | 13.81 |

- 配对 exact McNemar（a0-only vs both，每 seed 197 格）：s0 p=0.0139（11 vs 27），s1 p=0.0259（15 vs 31），s2 p=0.00075（8 vs 29）。三个 seed 都显著，方向都是 both 更好。
- 每 seed 的 both 成功率 0.244 / 0.218 / 0.213，a0-only 0.162 / 0.137 / 0.107，没有一个 seed 出现反向。
- fallback 率 0/2,364 个带记忆的 agent-episode（0.00%）；检索到的 11,820 条 instance 全部可追溯来源，**0 条来自被评测 episode 自己那一折**。
- 20 个 episode 以 `noneaction` 崩溃（原代码，搭档动作没解析出来），按 success 0 计入分母；失败指标只在有 planner log 的 episode 上平均（每 seed 192–197）。
- 四个失败指标与 Table 5 不同量纲（B-D15：Table 5 的计算脚本不在盘上），只能在本实验内部横向比。

## 结局：B3（方向反转）

- a0-only 比 both 低 0.090，远超较大的 std（0.028），且 Conflict/Ep 反而更高（0.346 vs 0.172）。
- **both 在 held-out 上是 0.225，与稿件 seen 口径的 0.223 几乎相同**；反转掉的是 leader-only 那一列（0.496 → 0.135）。也就是说 49.6% 那个数不是「对称记忆过度协调」的证据，而是 leader-only 在自己见过的 episode 上的记忆复用。
- 按预写规则：5.4 节两个口径都报；Limitations 加一句 "The leader-only advantage on H\_R is measured on episodes the memory covers and does not appear on a held-out half."；摘要里那句删掉。
- 附带：本实验里真正让 Conflict/Ep 下降的是 follower-only（0.061），但它的成功率也低于 both，所以不支持任何单边部署的结论。

## 现场

远端作业全停，GPU 0–5 已清空（6、7 上是别人的进程）。`b4_final.json` 在远端 `outputs/iclr_two_exp_B/`。脚本在 `scripts/iclr_two_exp/B/`，未提交。
