import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 数据准备 ---
data_long = [
    ["Llama 3.1-8b", "Traj-based", 0.712],
    ["Llama 3.1-8b", "Skill-based", 0.794],
    ["Llama 3.3-70b", "Traj-based", 0.840],
    ["Llama 3.3-70b", "Skill-based", 0.865],
    ["Claude", "Traj-based", 0.840],
    ["Claude", "Skill-based", 0.890],
    ["Qwen 2.5-7b", "Traj-based", 0.424],
    ["Qwen 2.5-7b", "Skill-based", 0.580],
    ["Qwen 2.5-72b", "Traj-based", 0.760],
    ["Qwen 2.5-72b", "Skill-based", 0.820],
    ["ChatGPT-4o", "Traj-based", 0.800],
    ["ChatGPT-4o", "Skill-based", 0.893],
]

# 定义用户指定的显示顺序
desired_order = [
    "ChatGPT-4o",
    "Claude",
    "Llama 3.3-70b",
    "Qwen 2.5-72b",
    "Llama 3.1-8b",
    "Qwen 2.5-7b",
]

# 处理数据并按指定顺序排列
models = []
traj_scores = []
skill_scores = []

# 创建一个字典来临时存储数据，方便查找
data_map = {}
for model, method, score in data_long:
    if model not in data_map:
        data_map[model] = {}
    data_map[model][method] = score * 100  # 转换为百分比

# 按照指定顺序提取数据
for model in desired_order:
    if model in data_map:
        models.append(model)
        traj_scores.append(data_map[model].get("Traj-based", 0))
        skill_scores.append(data_map[model].get("Skill-based", 0))

traj_scores = np.array(traj_scores)
skill_scores = np.array(skill_scores)

# --- 2. 绘图设置 ---
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.grid(axis="y", linestyle="--", color="lightgrey", zorder=0)

# 柱状图参数
bar_width = 0.35
index = np.arange(len(models))
# 颜色和样式
color_fill = "#90CAF9"  # 浅蓝色
hatch_style = "///"  # 斜线纹理
edge_color = "black"  # 边框颜色

# --- 3. 绘制柱子 ---
# Traj-based (实心)
bars_traj = ax.bar(
    index - bar_width / 2,
    traj_scores,
    bar_width,
    label="Traj-based",
    color=color_fill,
    edgecolor=edge_color,
    zorder=3,
)

# Skill-based (带纹理)
bars_skill = ax.bar(
    index + bar_width / 2,
    skill_scores,
    bar_width,
    label="Skill-based",
    color=color_fill,
    hatch=hatch_style,
    edgecolor=edge_color,
    zorder=3,
)


# --- 4. 添加数值标签 ---
# 辅助函数：在柱子上方添加文本
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}",  # 保留一位小数
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="black",
            zorder=4,
        )


add_labels(bars_traj)
add_labels(bars_skill)

# --- 5. 美化和格式化 ---
ax.set_ylabel("Success Rate (%)", fontsize=18, fontweight="bold", labelpad=10)
# ax.set_xlabel('Model', fontsize=18, fontweight='bold', labelpad=10)

ax.set_xticks(index)
# 设置X轴标签为重新排列后的模型名称
ax.set_xticklabels(models, fontsize=16, fontweight="bold")
ax.tick_params(axis="y", labelsize=14)

# 设置Y轴范围，留出一点空间给顶部的数字标签
ax.set_ylim(0, 105)

# 移除顶部和右侧的边框脊
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- 6. 自定义顶部图例 ---
legend_traj = mpatches.Patch(
    facecolor=color_fill, edgecolor=edge_color, label="Traj-based"
)
legend_skill = mpatches.Patch(
    facecolor=color_fill, hatch=hatch_style, edgecolor=edge_color, label="Skill-based"
)

plt.legend(
    handles=[legend_traj, legend_skill],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False,
    fontsize=18,
    handlelength=1.5,
    handleheight=1.5,
)

plt.tight_layout()
plt.savefig("comparison_reordered.png", dpi=300, bbox_inches="tight")
plt.show()
