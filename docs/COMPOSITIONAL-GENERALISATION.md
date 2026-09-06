# 组合泛化：VIKI-L2 与 PARTNR 的分工

设计笔记，2026-09-06。**不是论文正文**（英文方法描述在 `AGENTIC-FRAMEWORK-METHOD.md`，
结果台账在 `VIKI-L2-PAPER-RESULTS.md`，过程与破冻结记录在 `AGENTIC-OPERATOR-INDUCTION.md`）。
这份写的是两条线为什么都要做、各自测什么、以及库的组织方式为什么长成现在这样。

---

## 1. 两条线问的是同一个问题

**训练里有各个零件，但从没见过它们拼在一起，记忆能不能拼？**

差别在于「零件」和「拼」各指什么。

## 2. 差别一：扣掉的是新维度，还是新组合

**PARTNR 扣一整个约束维度。** 记忆只从 R（纯重排）轨迹归纳，评测加上空间 / 时序 /
状态变更约束：

    任务类型   n     ceiling   v2 R-only 记忆
    R         260    0.978      0.897
    R_S       261    0.939      0.841     加空间
    R_T        68    0.984      0.708     加时序
    R_S_T     186    0.879      0.651     都加
    H_R       207    0.918      0.325     加状态变更

关键：**`is_next_to` 在 R 轨迹里根本不出现**。不是「见过但没组合过」，是这个谓词的演示
一次都没有。PARTNR 测的是**词表外推**。

**VIKI 扣一个共现。** comp 的每一行要求「切割（双机器人：梨→砧板，再用刀）+ 一次独立搬运
（单机器人）」。两种模式在训练里都大量存在，只是**从未在同一条轨迹里同时出现**。谓词还是
那两个（`pos.name` / `is_activated`），一个新的都没有。VIKI 测的是**已知词表的重组**。

审计把这件事量化了（`audit/comp_leakage_2026-09-05/`）：

    行                              双机器人 cut   独立 delivery
    comp 297 行（295 个 task_id）       True          True     ← 留出的组合
    cut_fruit / cut_two_fruits          True         False
    toast_bread / wash_fruit           False          True
    训练集中同时具备两者的行：0

## 3. 差别二：坏在哪不一样，而且互补

**PARTNR 坏在覆盖。** R-only 记忆在 207 个异质任务上 `task_state_success` **精确为
0.0000**，而 `task_percent_complete` 是 0.325 —— 一份只见过重排的记忆完不成任何一个异质
任务，却仍能满足其中三分之一的单项要求。补上状态变更算子补回缺口的 45%（0.325 → 0.592）。
**缺的是条目。**

**VIKI 坏在组合。** comp 上所有失败都是 `GOAL_UNMET`，**从来不是
`UNSUPPORTED_PREDICATE`** —— 覆盖是够的。坏在 Layer 2 从算子体的访问集推不出时序：

    参考库 is_activated 体  Move Reach Grasp Move Reach Interact  visits = {刀, 砧板}
    短体                    Move Reach Interact                   visits = {刀}

Layer 2 的规则是「A(放置) → B(激活)，当 B 的体去了 A 放置的地点」。短体的 visits 里没有
砧板，规则匹配不上，**一条约束都不发**，规划器于是并行安排，还没放梨就先切。
**缺的是体的长度，不是条目。**

一个方法只解决其中一种，都说明不了问题。

## 4. 差别三：口径与检验

    PARTNR   task_percent_complete   连续量   配对 Wilcoxon / bootstrap null band
    VIKI     每行 0/1                二值     McNemar exact

PARTNR 上用 McNemar 是错的。这是硬规矩。

## 5. 差别四：归纳 headroom 决定了两边的角色

    PARTNR   R-only 库读的 161 条轨迹，462 个满足的命题丢掉 180 = 39.0%
             is_in_room 丢 100%（靠 Navigate 达成，不在完成动词白名单）
             is_next_to 丢 86%（Place 的副作用，参数里没有参照物）
    VIKI     2698 个 episode、5147 个 completion，丢弃 0.0%

- **VIKI 是验真台。** 归纳无损 → 存在一个已知正确的靶子（19 算子手写库），agent 只给轨迹
  能不能推出等价物，**可证伪**。
- **PARTNR 是出新的地方。** 丢 39% → agent 有可能拿到手写归纳器拿不到的东西。`is_in_room`
  是最好的靶子：丢 100%、每个 split 都丢，61/70 条猝死 episode 要它，`_resolve` 已经实现了
  它 —— **机器在，条目不在**。手写一个等于在编写记忆而不是归纳记忆。

## 6. 一条真正跨 benchmark 共通的结论

**迁移边界落在算子词表上**，两边同构：

    VIKI    Push 只出现在 dog_push 一族。留出该族 = 把原语整个抽走，任何归纳都恢复不了
            → 8 折的折次代价 1.84 点全部来自这一族，其余七折 fold 与 full 逐行相同。
    PARTNR  is_in_room 只能靠 Navigate 达成，而 Navigate 不在完成动词白名单 → 100% 丢失。

**留出一个族几乎不花钱，除非那个族独占某个原语。**

## 7. 一条必须收回的「共通结论」

早先文档写过「PARTNR 的时序 DAG 值 17–21 点，与 VIKI 上 Layer 2 的 −20.8 量级一致」，
当成两边互证。**这条不成立了。** step-0 猝死修完重测后：

    priv:v2_memory_R − priv:v2_R_noorder    +0.080 (REAL) → +0.065（落进 0.076 噪声带）

诚实的说法是**它从来没被稳健分开过** —— 修前刚过带、修后刚不过。所以现在是**不对称的**：

- **VIKI 上排序是决定性的**，机制验过（0.862 对 0.327；gate 对照表；`w/o Ordering` 把三个
  库全部压到 0.327/0.276，与缺变体的库逐位相同）。
- **PARTNR 上排序的贡献不是一个 finding，不能引。**

## 8. 记号：`Task(Memory)`

沿用 PARTNR 的写法，两边对齐：

    PARTNR   R_T(R)                 时序任务，用纯重排记忆
             H_R(R) vs H_R(all)     异质任务，R-only 对全类型
    VIKI     comp(cut+delivery)     组合行，用切割族 + 搬运族的并集
             comp(cut)              同样的行，只给切割族        ← 归因格
             ID(8 families)         ID 行，八族并集
             heldout_X(7 others)    族 X 的行，用不含 X 的并集

`comp(cut)` 是对着 `H_R(R)` 设计的：**故意只给一半，看掉多少**。若 cut-only 掉得惨，说明
组合确实需要两边的算子，成绩不是切割那一族单独撑起来的。

---

## 9. 为什么建这么多库

**建的库和评测用的记忆不是一回事。** 14 个族库是积木，评测用的记忆是它们的并集。

    评测列                        记忆份数   由哪些族库并出
    ID (924)                        1        全部 14 族
    comp(all) 带图/文本             -        与 ID 同一份记忆，只换 split
    comp(cut+delivery) 带图/文本    1        cut_fruit + cut_two_fruits + single_move
    comp(cut) 带图/文本             1        cut_fruit + cut_two_fruits
    留出族 × 8                      8        每份 = 除该族外的 13 个族库
    no-trace 对照                   1        它自己

库要花 call 建，记忆是并集出来的，零成本。

**为什么建满 14 族，而不是评测涉及的 8 族。** 论文那边的规矩是所有 memory 方法用同一批
轨迹，而基线确实是从整个归纳半建的：`viki_eval_memento.py:93` 用 `episodes[::2]` of
`train.parquet`，`exclude_family` 只在折次格设；G-Memory 与 RAG 臂读同一个文件。我们只用
8 族就是**自己少看了数据**，审稿人不会替我们算这个让步，只会看到数字低。

`comp(cut)` → `comp(cut+delivery)` → `comp(all)` 三格是**从半份到全份的曲线**，后两格零成本。

**留出族用 13 族并集，不是 7 族。** 否则 ID 用 14 族、留出族用 7 族，ID 高于留出族就分不清
是留出的代价还是记忆本身小，下面「各列之间只剩哪些族可用一个变量」这句就不成立了。

真正的收益不在省 run，在三件事：

1. **泄漏变成结构上不可能。** 族库在物理上只读过本族的归纳半 episode，没有哪一步能看到
   comp。不需要事后审计来证明。
2. **留出族本来就需要 8 份不同的记忆。** 「该族从未存在过」意味着评 `cut_fruit` 的 189 行
   时记忆里不能有任何来自该族的东西。用积木是 8 次并集，直接建是 8 次建库。
3. **各列之间只剩「哪些族可用」一个变量。** 各建各的话，ID 高于留出族可能只是两个库碰巧
   不同；用积木，两份记忆里是同一批算子对象，ID 是 14 族、留出族是 13 族，差的就是被留出
   的那一族。

### 每个库对应什么

**八个折次族** —— 924 行 ID manifest 全部落在这八族里，所以留出族只能对它们建：

    F_clear_table       220 行   收进柜子（含开封容器）
    F_cut_fruit         189 行   水果上砧板 + 用刀（需要时序）
    F_cut_two_fruits    108 行   同上，两个水果
    F_dog_push           46 行   推箱子接力（唯一需要 coordination）
    F_ensure_all_fruits   82 行   水果都弄上桌
    F_parallel_human     86 行   双物体分头送
    F_set_plate          85 行   摆盘子和叉子
    F_toast_bread       108 行   面包进烤箱 + 开烤箱（需要时序）

**另外六个族不在 ID manifest 里**（test split 本身有全部 14 族、1800 行，是 manifest 只取
了 8 族的 924 行）。它们各自提供 ID 上缺的体变体：

    single_move_asset_to_target        Move Reach Grasp Move Place
                                       最朴素的递送。comp 的搬运那一半按审计 L1 就是它
    wash_fruit_and_serve               Move Reach Grasp Move Place Interact
                                        Move Reach Grasp Move Place
                                       放下→激活→再拿起来放到别处。单机器人版的
                                       「放置门控激活」，11 步最长
    sequential_pick_two_and_place      Move Reach Grasp Move Reach Grasp Move Place
                                       一次抓两个再放
    serve_bread_after_checking_cabinet Move Reach Open Move Reach Grasp Move Place
                                       先开柜再取 —— 开的是「东西所在的容器」，
                                       与 clear_table 的「开目标容器」是两个变体
    serve_bread_from_counter           Move Reach Grasp Move Place
    dog_check_environment              Move Interact                最短的激活

ID 上 22.7% 的 `infeasible_assignment` 就是可行指派太少造成的，这些变体直接对症。

第十个 **no-trace 对照（arm d）** —— 不给轨迹，只给谓词表与原语表，让模型凭先验写。它回答
的是最要命的质疑：**算子是不是模型从预训练里背出来的？** 若它的分与正常库相当，「记忆」二字
不成立。上一轮它建出 1 个算子（`Move Reach Grasp Interact`，实测 support 20 —— 模型确实能
猜出可用的激活体），但端到端全 0，正常库 0.45–0.51。对照通过。

### 三层的并法不同

    Layer 1  并集，按 effect schema + body signature 去重，provenance 合并，
             support 在并集自己的池子上重算（不沿用、不相加族内的数）
    Layer 2  不并集，在该列的 episode 池上重新 mine，阈值不变
    Layer 3  不并集，在同一池上重新 harvest

Layer 2/3 可以这样做，依据是机械核实过的：`dependencies.mine(episodes, ...)` 与
`vocabulary.harvest(episodes, ...)` **签名里没有库参数**，Layer 1 到不了它们 —— 那两层是
**池子的属性**，不是算子的属性。所以留出族 X 那一列，X 在三层里都不存在。

### 按族建的动机，和它不再有的代价

按族建是「留出族那一列必须能整族拿掉」逼出来的：要能拿掉一个族，就得知道每个算子来自哪个
族。早先版本只建了评测涉及的 8 族 + 搬运族，那等于在 ID 上自缚一手 —— 手写参考库是从全部
14 族归纳的，它 support 最高的递送算子 support 2500，来自 8 个不同族。

现在建满 14 族，这个代价消失：**ID 与基线同池，留出族与 ID 只差一个族。**
规模 14 × 56 + 56(no-trace) = 840 run，上限 15,120 call。

## 10. 为什么评测几乎不花 call

已核实，不是假设：`viki_eval_v2_intent_choice.py` 的 prompt 是 `user content +
INSTRUCTION`，`INSTRUCTION` 是模块级常量（第 62 行）；`memory` 只在收到回复**之后**才参与
（`to_requirement` 接地、planner 展开）。**算子库不进 prompt** —— 19 个算子的库和 4 个算子
的库，给模型看的字一模一样。

所以同一个 (模型, split) 的归档回复可以对任意库离线重打分。已知答案检验：拿参考格自己的
归档回复走新 replay 路径、用参考库重打分 → **257/297 = 0.8653，与记录在案的逐位相同**。
