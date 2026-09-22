# A2 参考库 patch（单独成文件，合不合入由用户决定）

**状态：HALTED。** 原因有两条，都列在 report.md 的 DISCREPANCIES：
1. spec 预期的 72B 数值（ID 86.4、OOD 60.3、CG w/ Image 70.4、CG w/o Image 73.7）在盘上找不到来源。同一口径下盘上的值是 67.4 / 65.6 / 79.8 / 86.2。
2. 这个库不是手写的。`results/viki_memory_experiments/amendment11/skill_memory_v2.json` 写着 `built_from: VIKI-L2 train.parquet, even-indexed episodes`，是模拟器重放训练 episode 挖出来的，没有经过 LLM 提议。所以 "hand-written" 这个词不能用。

下面的草稿用盘上的数，措辞按实测方向写。**没有用户确认不合入。**

pass10 不在盘上，这里只给插入句，不给 diff。

---

## R1  RQ1 → "What the VIKI-L2 gap measures" 段末尾

spec 预写句的第二句是 "below it on ID and OOD and above it on both CG splits"。实测只有 OOD 这半句成立：ID 上我们更高，两个 CG split 上逐行完全相同。按 spec §1 第 5 条，只改和数据矛盾的部分：

```latex
A 19-operator library mined by simulator execution of the same training episodes, used with the same answers and planner, reaches 67.4/65.6/43.3/79.8/86.2\% on ID, OOD, sibling-group OOD, CG w/ Image, and CG w/o Image at 72B. The learned library of eight operators is above it on ID, below it on both OOD splits, and solves the same rows on both CG splits.
```

净增行数：+3。兄弟组的参考库由 `scripts/iclr_master/A/a2_build_ref_sibgrp.py` 零 call 重建（同一构建函数，排除单位从单族换成兄弟组）。

文风自检：没有破折号、冒号、分号，没有加粗，没有 replay 这个词（原来的 "simulator replay" 改成了 "simulator execution"）。

## R2  附录表

```latex
\begin{table}[h]
\centering
\small
\caption{Learned library and 19-operator mined library under the same first answers and planner. Success in \%, with exact McNemar counts (learned only / mined only). OOD sib.\ withholds the whole sibling group from both libraries.}
\label{tab:viki_mined_library}
\begin{tabular}{llrrrr}
\toprule
Model & Split & Learned & Mined & Counts & $p$ \\
\midrule
72B & ID           & 78.0 & 67.4 & 115/17 & $<10^{-18}$ \\
72B & OOD          & 54.2 & 65.6 & 0/105  & $<10^{-31}$ \\
72B & OOD sib.\     & 31.9 & 43.3 & 0/105  & $<10^{-31}$ \\
72B & CG w/ Image  & 79.8 & 79.8 & 0/0    & 1 \\
72B & CG w/o Image & 86.2 & 86.2 & 0/0    & 1 \\
30B & ID           & 55.8 & 51.0 & 92/47  & $2\times10^{-4}$ \\
30B & OOD          & 35.4 & 46.0 & 0/98   & $<10^{-29}$ \\
30B & OOD sib.\     & 15.9 & 26.4 & 0/97   & $<10^{-28}$ \\
30B & CG w/ Image  & 65.0 & 66.0 & 0/3    & 0.25 \\
30B & CG w/o Image & 72.4 & 73.4 & 0/3    & 0.25 \\
7B  & ID           & 13.9 & 16.0 & 10/30  & 0.002 \\
7B  & OOD          & 11.7 & 14.5 & 0/26   & $3\times10^{-8}$ \\
7B  & OOD sib.\     & 3.7  & 4.8  & 0/10   & 0.002 \\
7B  & CG w/ Image  & 15.2 & 15.8 & 0/2    & 0.5 \\
7B  & CG w/o Image & 25.6 & 26.3 & 0/2    & 0.5 \\
\bottomrule
\end{tabular}
\end{table}
```

"Learned" 这一列是 first-turn-only（spec A2.2 要求同一口径）。只有 30B ID 和主表不同：主表带第二轮是 57.4，对 Mined 是 106/47，p = 2.1e-6。
