# 方法本体，用 Python 写一遍

这份文档把两条线（VIKI-L2 与 PARTNR）的方法压成一段可读的 Python。**代码是真的**——每个函数
都对应仓库里的实现，函数名后面注明了出处；省略的只有工程细节（重试、超时、日志、并行）。
要跑的是那些文件，不是这一份；这一份是给人读的。

纪律只有一条总纲，其余全部是它的推论：

> **agent 只提议，模拟器永远说了算。**

---

## 0. 数据结构

```python
from dataclasses import dataclass, field
from typing import Any, Literal

Action = list[str]                      # ["Move", "?z1"] / ["Place", "?x", "on", "?z1", ...]

@dataclass
class Operator:
    """一条技能。变量的含义是固定的，不是提议者能改的。

    ?x  效果所关于的主体        ?y  效果的目标
    ?z1 ?z2 ...  备用变量：规划器读"body 拿它做什么"来决定它绑到什么
    ?r0 ?r1 ...  角色变量：多机器人算子里每个角色一台机器人，由规划器指派
    """
    effect: dict          # {"key": "pos.name" | "is_activated" | "is_in_room", "subject": "?x", "value": "?y"}
    body: list[Action] = field(default_factory=list)     # 单执行者
    roles: list[dict] = field(default_factory=list)      # 多执行者：[{"variable": "?r0", "actions": [{"action", "offset", "after"}]}]
    preconditions: dict[str, bool] = field(default_factory=dict)   # subject_sealed / target_sealed / ...
    types: dict[str, dict] = field(default_factory=dict)           # 每个变量必须具备的属性
    cost: int = 0
    coordinated: bool = False
```

对应 `our_method/skill_memory_v2/memory.py`（Layer 1 的记录格式）。

---

## 1. 归纳：agent 提议，工作台是机械的

### 1.1 工作台（`scripts/viki_induction_tools.py::Workbench`）

工作台只回答问题，**从不下判断**。它读的永远是归纳半 `episodes[::2]`，另一半永不暴露，
所以任何工具使用都不可能对验收集过拟合。

```python
class Workbench:
    def list_episodes(self, family, limit, start) -> dict: ...
    def show_trace(self, index, max_steps) -> dict:
        """一次重放：每步每台机器人做了什么，每个谓词在第几步变真。"""

    def contrast_actors(self, index) -> dict:
        """同一 episode 里每台机器人各自的动作序列，并排放。
        —— 加这个工具之前，模型只看第一台机器人，0/144；加完 32.47% → 50.43%。"""

    def check_actor(self, index, completion, actor, start) -> dict:
        """反事实重放：只跑这台机器人这一段，谓词还成立吗。这是因果检验。"""

    def try_bind(self, operator, index) -> dict:
        """跑规划器自己的 chains_for；绑不上时点名哪些 token 没被绑住。"""

    def run_operator(self, operator, index) -> dict:
        """在另一个 episode 上绑定 → 执行 → 问效果是否成立。这就是抽象检验。

        单执行者：逐个机器人试（不能只试第一个，否则会枪毙掉正确的算子）。
        多执行者：交给 planner.schedule 按角色排程，再回放观察效果 —— 2026-09-09 补。
        返回里带 bindings：每个变量实际绑到了什么（观察，不是提示）。"""
```

### 1.2 提议循环（`scripts/viki_agentic_rung_abstraction.py`）

```python
def induce_one_operator(model, bench, family, seed_episode, holdout, library, moves=18):
    """一次 rung：agent 用工具看轨迹，提交算子，台子机械地判。"""
    messages = [task_text(family, seed_episode)]      # 含变量约定与算子形状，见 §1.3
    for move in range(moves):
        request = parse_one_json(model.reply(messages))

        if "submit" not in request:                   # 用工具
            messages += [request, bench.call(request["tool"], request["args"])]
            continue

        operator = request["submit"]

        # 判据一：在没给它看过的 episode 上，绑得上且效果成立 —— 至少两个。
        works = [j for j in holdout
                 if (r := bench.run_operator(operator, j))["bound"] and r["effect_holds"]]
        if len(works) < 2:
            messages += [refusal(works)]              # 告诉它失败在哪一步、变量绑到了什么
            continue

        # 判据二：边际。必须让一个"当前库解不了"的 episode 变成解得了。
        # 只写"增益 > 噪声"会把 6 个候选全收进来——查逐格才发现后 5 个抬起的每一格，
        # 第 1 个都已经抬起了。[[viki-rung-acceptance-must-be-marginal]]
        gained = [j for j in coverage_pool
                  if not solved(library, j) and solved(library + [operator], j)]
        if library and not gained:
            messages += [refusal_no_coverage(library)]   # 并告诉它库里已经有什么
            continue

        return operator            # 通过
    return None
```

**这里没有任何一步看得到端到端分数。** agent 看不到 benchmark 的成绩，所以它没法把 body
调到验收集上；验收集与上报 split 也是不相交的两批 episode。

### 1.3 接口约定（2026-09-09 补，`--interface-v2`）

任务文本必须说清楚**变量不是提议者的**：

```text
The planner binds ?x to the subject the effect is about and ?y to its target, and you
cannot repurpose either. Every other variable is a SPARE: write it ?z1, ?z2, ... and the
planner fills it by reading what the body does with it. So the body that fetches an object
out of a shut cupboard opens the spare and grasps ?x, never the other way round.
```

**为什么值得单列一节**：不说这句话的代价是可测量的。v2 那轮里模型两次写出了正确形状的
"开柜取物"算子，但把 `?x` 当成柜子 → 台子回 `checker refused ['Open', 'plate']` → 它再没
恢复过来。那一个算子单独值 ID 列 **155 行**。同一句话在 PARTNR 侧加上之后，72B 第一次
就交出了正确形状。这是 [[viki-interface-not-capability]] 的第二次。

### 1.4 按族建库，并集去重

```python
def build_library(model, bench, families):
    per_family = {f: induce_family(model, bench, f) for f in families}   # 每族只读本族轨迹
    return dedup(union(per_family.values()))        # scripts/viki_assemble_agentic_library.py
```

- **泄漏在结构上不可能**：族库只读本族的归纳半 episode，所以"留一族"折出来的库
  = 其余 13 族族库的并集。
- **support 不信模型的话，重算**：把算子拿去归纳半上跑，数它在哪些 episode 上绑得上且
  效果成立。`"support": 1` 是模型自己写的，丢掉。
- **代价**：一个算子只有一个供体族时，那一折就会把它折没——这正是留出列 −398 行的机制，
  见 `results/viki_ood_gap_2026-09-09.json`。

---

## 2. 消费：库 + 规划器

`our_method/skill_memory_v2/planner.py`。**规划器自己的贡献是刻意做小的**：它只挑 body、
派机器人、逐步推进；可行性由 benchmark 自己的 checker 在它自己的世界模型上判，
所以规划器不可能和裁判对"什么是合法的"有分歧。

```python
def plan(truth, memory, sim, crew=None):
    """先试单执行者的 body，只有它们做不到时才付多执行者搜索的代价。"""
    steps, why = compose(truth, memory, sim, coordinated=False, crew=crew)
    return (steps, why) if steps else compose(truth, memory, sim, coordinated=True, crew=crew)


def compose(truth, memory, sim, coordinated, crew):
    env = sim.world(sim.metadata(truth))
    choices = []
    for requirement in collect_requirements(env.metadata):       # 目标谓词 + 时序阶段
        if holds(env, requirement.predicate):
            continue
        facts = state_facts(env, requirement.predicate)          # subject_sealed / target_sealed / ...
        # 变体的选择不问模型："柜子是关着的吗"是世界能看、图片看不出的隐藏状态。
        chains = chains_for(env, requirement, memory, coordinated)
        if not chains:
            return None, "UNSUPPORTED_PREDICATE"
        choices.append(chains)

    for selection in product(*choices):                # 每个需求选一条链
        for order in permutations(selection):          # 链之间的先后
            for assignment in castings(order, env.agents, crew):
                steps = schedule(env.metadata, assignment, sim)
                if steps: return best(steps), "OK"
    return None, "NO_SCHEDULE"


def schedule(metadata, plans, sim, cap=16):
    """每台机器人每步推进一个动作，合法性由裁判的 checker 说了算。

    两个没写成规则、自己长出来的行为：
      * 下一个动作还不合法的机器人原地重试 —— "等待"于是自己出现了：
        去关着的柜子的机器人正好等到有人把它打开。
      * 一台机器人同时持有多条链、推进任何一条能推进的 —— 于是"开柜"变成
        搬运行程里的一次插入，而不是单独跑一趟。步数预算就是这么省下来的。
    """
```

`crew` 是外部指派：**谁命名工作的人也可以命名机器人**。同一批需求带/不带它各规划一次，
差值就是"分工"值多少——这不是便利，这是让分工可测量。

---

## 3. PARTNR：环切成两半

PARTNR 没有微秒级 replay oracle（唯一的因果检验是把算子放进记忆、跑 benchmark，
一个 episode 几分钟），所以 VIKI 那个"环内自测"的循环在这里切成两半：
**提议只读轨迹，验收在环外用执行做。**

```python
# 半一：提议（scripts/partnr_propose_operators.py）—— 看不到任何端到端分数
def propose(model, rollouts, target_key, moves=24, candidates=6):
    bench = Workbench(rollouts)          # list_traces / show_trace / held_by / actor_window
                                         # / room_of / rooms / predicts(仅顾问)
    submitted = []
    for move in range(moves):
        request = parse_one_json(model.reply(...))
        if "submit" not in request:
            continue_with(bench.call(request))
        operator = request["submit"]
        if operator["effect"]["key"] != target_key:      # 拒
            continue
        if not groundable(operator)["groundable"]:        # 免费落地关：只拒"根本跑不起来"的
            continue                                      # 从不拒"跑得起来但是错的"
        submitted.append(operator)
        if len(submitted) >= candidates:
            break
    return submitted


# 半二：验收（scripts/partnr_gate_batch.py + partnr_gate_adjudicate.py）
def accept(candidates, gate_pool, base_library):
    """四条判据必须同时成立，缺一不可。"""
    results = []
    for cand in candidates:
        cell = run_benchmark(base_library + [cand], gate_pool)   # 成对、同代码、同时刻
        results.append((cand, compare(cell, run_benchmark(base_library, gate_pool))))

    accepted = []
    for cand, r in sorted(results, key=lambda x: -x[1].delta):
        if not r.complete:                       # 一：格必须跑完（半格会给出像模像样的假增益）
            continue
        if r.ci_low <= 0 <= r.ci_high:           # 二：bootstrap 区间排除 0
            continue
        if not marginal_gain(accepted, cand, r): # 三：按增益降序取时仍有边际覆盖
            continue
        if r.pushed_down > r.pushed_up:          # 四：压下去的格不多于抬起来的
            continue
        accepted.append(cand)
    return accepted
```

**门必须先双向校准**，否则"门"这个词没有意义。做法是：清空整个 `is_on_top` 键做地板，
再放回一个算子，看门给出什么分数——

| 变体 | Δ vs 地板 | 结论 |
|---|---|---|
| 出厂算子（正确） | **+0.7375** | 门认得出对的 |
| 正确 + 一步冗余 Open | **+0.2583** | **分级**，不是二值 |
| 倒序体 / 换 effect key | 0.0000 | 加载即被免费关拒 |
| 无 Pick（能跑、手是空的） | 0.0000 | ← trace-matching 门做不到的那一类 |
| 只 Pick（能跑、只归因一半） | **−0.0375** | ← 同上 |

---

## 4. 评测协议：池子预登记，报表由脚本直读

```python
POOLS = {                    # results/partnr_pools/ledger.json，八池两两不交，全部从 train 切
    "rec_R_iir":   240,      # 录轨迹给 agent 读（agent 实际只打开了 5 条）
    "gate_iir":     60,      # 验收门
    "conf_iir":     60,      # 确认池：预登记，全程只开这一次
    "calib_ontop":  40,      # 门的双向校准
}
REPORT = "val_mini"          # 369 格，与上面全部不交；铁律：门一步不能踩 val/val_mini
```

三条硬规矩，都是赔过机器时间换来的：

```python
assert compare["complete"], "比较器默认在交集上比，半格会给出像模像样的增益"
band = run_twice_same_library(REPORT)          # 同库同配置自比重复
assert band.moved_cells == 0                   # 实测 = 0，所以区间不需要噪声修正
# VIKI 一律 JSON-tolerant 口径（官方 scorer 会把 79.5% 的行误判 0 分）；
# 配对检验：VIKI 用 McNemar exact，PARTNR 的 percent_complete 是连续量，用 bootstrap。
```

**留出列的组装方式**（`scripts/viki_v2_baseline_comparison.py`）：族 F 的行，取自
"记忆里排除了 F"的那一次运行。所有记忆臂一起重跑——只给基线拿掉兄弟族就是把偏向反过来。

**重打分几乎不要 GPU**：提示词里没有库，记忆是在模型回答之后才被查询的，所以一批归档回复
可以对任意多个库重打分（`--replay`）。换库不换回复，隔离出来的正好是 Layer 1 的贡献。

---

## 5. 一句话总结每条不变量

| 不变量 | 违反它的那一次代价 |
|---|---|
| agent 只提议，模拟器说了算 | —— 这条没破过 |
| 验收要**边际**，不能只要"增益 > 噪声" | 第一版规则把 6 个候选全收了 |
| 变量的含义由规划器定，且必须写进任务文本 | 155 行 |
| 验收工具必须能验它要验的那类算子 | coordination 17 行，参考库自己的算子在旧工具上 0/6 |
| 某一格得零时，先把已知正确的答案送进同一个判据 | 09-05 有四次"模型不行"最后都是台子 |
| 报表脚本必须落盘（且落盘在打印之前） | 一份报表打完全部表格后崩在写 JSON 那一步 |

---

*出处：`docs/AGENTIC-OPERATOR-INDUCTION.md`（协议全文）、`HANDOVER-2026-09-07b.md`（PARTNR
的全部推导）、`RESULTS-2026-09-08.md`（全部表格，脚本直读，不要手改）。*
