#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 heterogeneous_rerange_heuristic 数据集转换为 rerange_only_organized_by_skills_enhanced 格式
"""

import gzip
import json
import os
from collections import defaultdict


def extract_skill_categories_from_episode(episode):
    """从 episode 中提取技能类别（基于 instruction 和 task_type）"""
    categories = set()

    # 如果已经有 skill_categories 字段，直接使用
    if "skill_categories" in episode and episode["skill_categories"]:
        categories.update(episode["skill_categories"])
        return categories

    # 从 skills 字段中提取（如果有）
    if "skills" in episode and episode["skills"]:
        skill_text = " ".join(str(v) for v in episode["skills"].values() if v).lower()
    else:
        # 如果没有 skills 字段，从 instruction 和 task_type 中推断
        instruction = episode.get("instruction", "").lower()
        task_type = episode.get("task_type", "").lower()
        skill_text = f"{instruction} {task_type}"

    # Object manipulation keywords
    if any(
        word in skill_text
        for word in ["pick up", "grab", "hold", "lift", "take", "carry"]
    ):
        categories.add("Object Manipulation")

    # Object placement keywords
    if any(
        word in skill_text
        for word in ["place", "put", "drop", "set down", "position", "arrange"]
    ):
        categories.add("Object Placement")

    # Navigation keywords
    if any(
        word in skill_text
        for word in ["navigate", "move to", "go to", "walk to", "travel to", "move"]
    ):
        categories.add("Navigation")

    # Container management keywords
    if any(
        word in skill_text
        for word in [
            "container",
            "box",
            "drawer",
            "cabinet",
            "shelf",
            "fridge",
            "microwave",
        ]
    ):
        categories.add("Container Management")

    # Multi-agent coordination keywords
    if any(
        word in skill_text
        for word in [
            "coordinate",
            "cooperate",
            "wait for",
            "avoid collision",
            "coordination",
            "together",
            "both",
        ]
    ):
        categories.add("Multi-Agent Coordination")

    # Task planning keywords
    if any(
        word in skill_text
        for word in ["first", "then", "after", "before", "sequence", "plan", "step"]
    ):
        categories.add("Task Planning")

    # Room-specific organization - check for room names
    rooms = [
        "bedroom",
        "kitchen",
        "living room",
        "dining room",
        "bathroom",
        "hallway",
        "office",
    ]
    if any(room in skill_text for room in rooms):
        categories.add("Room-Specific Organization")

    # 如果没有找到任何类别，使用 General Task
    return categories if categories else {"General Task"}


def transform_heterogeneous_rerange_to_organized_skills():
    """将 heterogeneous_rerange_heuristic 转换为 rerange_only_organized_by_skills_enhanced 格式"""

    input_file = "/home/shuqing/partnr-planner/data/rag_datasets/heterogeneous_rerange_heuristic/react_trajectories.json.gz"
    output_dir = "/home/shuqing/partnr-planner/data/rag_datasets/heterogeneous_rerange_heuristic_skills"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "by_task_type"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "by_complexity"), exist_ok=True)

    print("=" * 80)
    print(
        "转换 heterogeneous_rerange_heuristic 到 heterogeneous_rerange_heuristic_skills 格式"
    )
    print("=" * 80)
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print()

    # 读取数据
    print("📖 读取数据...")
    with gzip.open(input_file, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data["episodes"]
    total_episodes = len(episodes)
    print(f"  ✓ 读取 {total_episodes} 个episodes")

    # 组织episodes
    skill_based_datasets = defaultdict(list)
    task_type_datasets = defaultdict(list)
    complexity_datasets = defaultdict(list)

    print("\n🔄 处理episodes...")

    for episode in episodes:
        # 提取技能类别
        skill_categories = extract_skill_categories_from_episode(episode)

        # 将episode添加到每个相关的技能类别数据集
        for skill_cat in skill_categories:
            skill_based_datasets[skill_cat].append(episode)

        # 同时按任务类型和复杂度组织
        task_type = episode.get("task_type", "General Task")
        complexity = episode.get("complexity", "Medium")

        task_type_datasets[task_type].append(episode)
        complexity_datasets[complexity].append(episode)

    # 创建技能类别数据集
    print("\n📁 创建技能类别数据集...")
    for skill_category, episodes in skill_based_datasets.items():
        output_file = os.path.join(
            output_dir,
            f'skill_{skill_category.lower().replace(" ", "_").replace("-", "_")}.json.gz',
        )

        skill_dataset = {
            "metadata": {
                "skill_category": skill_category,
                "total_episodes": len(episodes),
                "source": "heterogeneous_rerange_heuristic",
                "created_by": "transform_heterogeneous_rerange_to_organized_skills.py",
                "organization_type": "skill_based",
                "description": f"Episodes focused on {skill_category} skills",
            },
            "episodes": episodes,
        }

        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            json.dump(skill_dataset, f, indent=2, ensure_ascii=False)

        print(
            f"  ✓ {skill_category}: {len(episodes)} episodes -> {os.path.basename(output_file)}"
        )

    # 创建任务类型数据集
    print("\n📁 创建任务类型数据集...")
    task_type_dir = os.path.join(output_dir, "by_task_type")

    for task_type, episodes in task_type_datasets.items():
        # 清理任务类型名称用于文件名
        safe_task_type = (
            task_type.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        )
        output_file = os.path.join(task_type_dir, f"{safe_task_type}.json.gz")

        task_dataset = {
            "metadata": {
                "task_type": task_type,
                "total_episodes": len(episodes),
                "source": "heterogeneous_rerange_heuristic",
                "created_by": "transform_heterogeneous_rerange_to_organized_skills.py",
                "organization_type": "task_type_based",
                "description": f"Episodes for {task_type} tasks",
            },
            "episodes": episodes,
        }

        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            json.dump(task_dataset, f, indent=2, ensure_ascii=False)

        print(
            f"  ✓ {task_type}: {len(episodes)} episodes -> {os.path.basename(output_file)}"
        )

    # 创建复杂度数据集
    print("\n📁 创建复杂度数据集...")
    complexity_dir = os.path.join(output_dir, "by_complexity")

    for complexity, episodes in complexity_datasets.items():
        output_file = os.path.join(
            complexity_dir, f"{complexity.lower()}_complexity.json.gz"
        )

        complexity_dataset = {
            "metadata": {
                "complexity": complexity,
                "total_episodes": len(episodes),
                "source": "heterogeneous_rerange_heuristic",
                "created_by": "transform_heterogeneous_rerange_to_organized_skills.py",
                "organization_type": "complexity_based",
                "description": f"{complexity} complexity episodes",
            },
            "episodes": episodes,
        }

        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            json.dump(complexity_dataset, f, indent=2, ensure_ascii=False)

        print(
            f"  ✓ {complexity} Complexity: {len(episodes)} episodes -> {os.path.basename(output_file)}"
        )

    # 创建摘要报告
    summary_file = os.path.join(output_dir, "organization_summary.json")
    summary = {
        "original_dataset": {
            "total_episodes": total_episodes,
            "source": input_file,
            "dataset_name": "heterogeneous_rerange_heuristic",
        },
        "skill_based_organization": {
            category: len(episodes_list)
            for category, episodes_list in skill_based_datasets.items()
        },
        "task_type_organization": {
            task_type: len(episodes_list)
            for task_type, episodes_list in task_type_datasets.items()
        },
        "complexity_organization": {
            complexity: len(episodes_list)
            for complexity, episodes_list in complexity_datasets.items()
        },
        "total_organized_files": len(skill_based_datasets)
        + len(task_type_datasets)
        + len(complexity_datasets),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n✅ 转换完成!")
    print(f"  - 创建了 {len(skill_based_datasets)} 个技能类别数据集")
    print(f"  - 创建了 {len(task_type_datasets)} 个任务类型数据集")
    print(f"  - 创建了 {len(complexity_datasets)} 个复杂度数据集")
    print(f"  - 摘要报告保存到: {summary_file}")


if __name__ == "__main__":
    transform_heterogeneous_rerange_to_organized_skills()
